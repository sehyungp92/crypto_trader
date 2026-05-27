"""Parity event reporting and promotion gate helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crypto_trader.live.oms_store import OmsStore
from crypto_trader.parity.shadow import compare_event_streams


@dataclass(frozen=True, slots=True)
class ParityReport:
    stream_counts: dict[str, int]
    decision_drift_count: int = 0
    order_intent_drift_count: int = 0
    unresolved_oms_discrepancies: list[dict[str, Any]] = field(default_factory=list)
    fill_watermark_age_sec: float | None = None
    stale_fill_watermark: bool = False
    unprotected_entry_fills: list[dict[str, Any]] = field(default_factory=list)
    accounting_mismatch_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_counts": dict(self.stream_counts),
            "decision_drift_count": self.decision_drift_count,
            "order_intent_drift_count": self.order_intent_drift_count,
            "unresolved_oms_discrepancies": list(self.unresolved_oms_discrepancies),
            "fill_watermark_age_sec": self.fill_watermark_age_sec,
            "stale_fill_watermark": self.stale_fill_watermark,
            "unprotected_entry_fills": list(self.unprotected_entry_fills),
            "accounting_mismatch_count": self.accounting_mismatch_count,
        }


@dataclass(frozen=True, slots=True)
class PromotionGateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": list(self.failures)}


def load_parity_events(state_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(state_dir) / "parity_events.jsonl"
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def build_parity_report(
    state_dir: Path | str,
    *,
    expected_events: list[dict[str, Any]] | None = None,
    max_watermark_age_sec: float = 600.0,
) -> ParityReport:
    state = Path(state_dir)
    events = load_parity_events(state)
    stream_counts: dict[str, int] = {}
    for event in events:
        stream = str(event.get("stream", "unknown"))
        stream_counts[stream] = stream_counts.get(stream, 0) + 1

    decision_drift_count = 0
    order_intent_drift_count = 0
    if expected_events is not None:
        expected_decisions = _payloads(expected_events, "decision")
        actual_decisions = _payloads(events, "decision")
        expected_orders = _payloads(expected_events, "order_intent")
        actual_orders = _payloads(events, "order_intent")
        decision_drift_count = len(compare_event_streams(
            expected_decisions,
            actual_decisions,
            keys=("decision_id", "strategy_id", "symbol", "timeframe", "action"),
        ).drifts)
        order_intent_drift_count = len(compare_event_streams(
            expected_orders,
            actual_orders,
            keys=("intent_id", "decision_id", "strategy_id", "symbol", "side", "order_type"),
        ).drifts)

    discrepancies: list[dict[str, Any]] = []
    watermark_age: float | None = None
    if _oms_db_path(state).exists():
        oms = OmsStore(_oms_db_path(state))
        try:
            discrepancies = oms.list_unresolved_discrepancies()
            watermark_age = _watermark_age(oms.get_watermark("fills_since"))
        finally:
            oms.close()

    stale = watermark_age is not None and watermark_age > max_watermark_age_sec
    return ParityReport(
        stream_counts=stream_counts,
        decision_drift_count=decision_drift_count,
        order_intent_drift_count=order_intent_drift_count,
        unresolved_oms_discrepancies=discrepancies,
        fill_watermark_age_sec=watermark_age,
        stale_fill_watermark=stale,
        unprotected_entry_fills=_unprotected_entry_fills(events),
    )


def evaluate_promotion_gate(report: ParityReport) -> PromotionGateResult:
    failures = []
    if report.unresolved_oms_discrepancies:
        failures.append("unresolved_oms_discrepancies")
    if report.stale_fill_watermark:
        failures.append("stale_fill_watermark")
    if report.unprotected_entry_fills:
        failures.append("unprotected_entry_fills")
    if report.decision_drift_count > 0:
        failures.append("decision_drift")
    if report.order_intent_drift_count > 0:
        failures.append("order_intent_drift")
    if report.accounting_mismatch_count > 0:
        failures.append("accounting_mismatch")
    return PromotionGateResult(passed=not failures, failures=failures)


def _payloads(events: list[dict[str, Any]], stream: str) -> list[dict[str, Any]]:
    return [event.get("payload", {}) for event in events if event.get("stream") == stream]


def _oms_db_path(state_dir: Path) -> Path:
    return state_dir if state_dir.suffix else state_dir / "live_oms.sqlite3"


def _watermark_age(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _unprotected_entry_fills(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entry_fills = []
    stop_order_decisions: set[tuple[str, str]] = set()
    for event in events:
        payload = event.get("payload", {})
        if event.get("stream") == "order_intent" and payload.get("metadata", {}).get("tag") in {
            "protective_stop",
            "breakeven_stop",
            "proof_lock_stop",
            "trailing_stop",
        }:
            stop_order_decisions.add((payload.get("strategy_id", ""), payload.get("symbol", "")))
        if event.get("stream") == "execution" and payload.get("metadata", {}).get("tag") == "entry":
            entry_fills.append(payload)
    return [
        fill for fill in entry_fills
        if (fill.get("metadata", {}).get("strategy_id", ""), fill.get("symbol", "")) not in stop_order_decisions
    ]
