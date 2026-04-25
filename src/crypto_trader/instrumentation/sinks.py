"""Event sinks for instrumentation output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)

log = structlog.get_logger()


@runtime_checkable
class Sink(Protocol):
    """Protocol for event sinks."""

    def write_trade(self, event: InstrumentedTradeEvent) -> None: ...
    def write_missed(self, event: MissedOpportunityEvent) -> None: ...
    def write_daily(self, event: DailySnapshot) -> None: ...
    def write_error(self, event: ErrorEvent) -> None: ...
    def write_funnel(self, event: PipelineFunnelSnapshot) -> None: ...
    def write_health_report(self, event: HealthReportSnapshot) -> None: ...


class JsonlSink:
    """Appends serialized events to per-type JSONL files."""

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _append(self, filename: str, data: dict) -> None:
        path = self._dir / filename
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception:
            log.exception("jsonl_sink.append_failed", path=str(path))

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        self._append("instrumented_trades.jsonl", event.to_dict())

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        self._append("missed_opportunities.jsonl", event.to_dict())

    def write_daily(self, event: DailySnapshot) -> None:
        self._append("daily_snapshots.jsonl", event.to_dict())

    def write_error(self, event: ErrorEvent) -> None:
        self._append("errors.jsonl", event.to_dict())

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        self._append("pipeline_funnels.jsonl", event.to_dict())

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        self._append("health_reports.jsonl", event.to_dict())


class InMemorySink:
    """Collects events in lists. Used for backtest analysis and testing."""

    def __init__(self) -> None:
        self.trades: list[InstrumentedTradeEvent] = []
        self.missed: list[MissedOpportunityEvent] = []
        self.daily: list[DailySnapshot] = []
        self.errors: list[ErrorEvent] = []
        self.funnels: list[PipelineFunnelSnapshot] = []
        self.health_reports: list[HealthReportSnapshot] = []

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        self.trades.append(event)

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        self.missed.append(event)

    def write_daily(self, event: DailySnapshot) -> None:
        self.daily.append(event)

    def write_error(self, event: ErrorEvent) -> None:
        self.errors.append(event)

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        self.funnels.append(event)

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        self.health_reports.append(event)
