from __future__ import annotations

import json
import subprocess
import urllib.request
from datetime import datetime, timezone

from crypto_trader.instrumentation.sinks import JsonlSink
from crypto_trader.instrumentation.sidecar import SidecarForwarder
from crypto_trader.instrumentation.types import (
    EventMetadata,
    GenericInstrumentationEvent,
    InstrumentedTradeEvent,
    canonical_event_envelope,
)


def test_canonical_envelope_wraps_legacy_payload() -> None:
    payload = {
        "metadata": {
            "event_id": "e1",
            "bot_id": "bot1",
            "strategy_id": "momentum",
            "exchange_timestamp": "2026-05-31T00:00:00+00:00",
        },
        "pair": "BTC",
    }

    wrapped = canonical_event_envelope("trade", payload, bot_id="bot1")

    assert wrapped["schema_version"] == "assistant_event_v1"
    assert wrapped["event_type"] == "trade"
    assert wrapped["event_id"] == "e1"
    assert wrapped["symbol"] == "BTC"
    assert wrapped["payload"]["pair"] == "BTC"


def test_canonical_envelope_merges_source_for_existing_canonical_payload() -> None:
    payload = {
        "schema_version": "assistant_event_v1",
        "event_id": "e1",
        "logical_event_id": "e1",
        "event_type": "portfolio_snapshot",
        "bot_id": "bot1",
        "source": {"sink": "jsonl"},
        "payload": {"event_id": "e1"},
    }

    wrapped = canonical_event_envelope(
        "portfolio_snapshot",
        payload,
        source={"file_event_type": "portfolio_snapshot"},
    )

    assert wrapped["event_id"] == "e1"
    assert wrapped["source"] == {
        "sink": "jsonl",
        "file_event_type": "portfolio_snapshot",
    }
    assert payload["source"] == {"sink": "jsonl"}


def test_event_metadata_id_includes_strategy_id() -> None:
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)

    first = EventMetadata.create("bot1", "momentum", ts, "trade", "t1")
    second = EventMetadata.create("bot1", "trend", ts, "trade", "t1")

    assert first.event_id != second.event_id


def test_jsonl_sink_writes_legacy_and_date_partitioned_trade(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = InstrumentedTradeEvent(
        metadata=EventMetadata.create("bot1", "momentum", ts, "trade", "t1"),
        trade_id="t1",
        pair="BTC",
    )

    sink.write_trade(event)

    assert (tmp_path / "instrumented_trades.jsonl").exists()
    canonical_path = tmp_path / "instrumentation" / "events" / "trade" / "2026-05-31.jsonl"
    assert canonical_path.exists()
    row = json.loads(canonical_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["schema_version"] == "assistant_event_v1"
    assert row["event_type"] == "trade"
    assert row["payload"]["trade_id"] == "t1"


def test_generic_event_writes_to_canonical_type_file(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = GenericInstrumentationEvent(
        metadata=EventMetadata.create("bot1", "portfolio", ts, "portfolio_snapshot", "p1"),
        payload={"portfolio_id": "p1", "timestamp": ts.isoformat()},
    )

    sink.write_event("portfolio_snapshot", event)

    path = tmp_path / "instrumentation" / "events" / "portfolio_snapshot" / "2026-05-31.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["event_type"] == "portfolio_snapshot"
    assert row["payload"]["portfolio_id"] == "p1"
    assert row["source"]["sink"] == "jsonl"


def test_sidecar_keeps_unread_legacy_file_when_canonical_copy_exists(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = InstrumentedTradeEvent(
        metadata=EventMetadata.create("bot1", "momentum", ts, "trade", "t1"),
        trade_id="t1",
        pair="BTC",
    )
    sink.write_trade(event)

    forwarder = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")
    sources = forwarder._event_sources()

    canonical_source = (
        "trade",
        tmp_path / "instrumentation" / "events" / "trade" / "2026-05-31.jsonl",
        "instrumentation/events/trade/2026-05-31.jsonl",
    )
    legacy_source = (
        "instrumented_trades",
        tmp_path / "instrumented_trades.jsonl",
        "instrumented_trades",
    )
    assert canonical_source in sources
    assert legacy_source in sources

    forwarder._watermarks["instrumented_trades"] = (tmp_path / "instrumented_trades.jsonl").stat().st_size
    sources = forwarder._event_sources()
    assert canonical_source in sources
    assert legacy_source not in sources


def test_sidecar_emits_error_event_after_retry_exhaustion(tmp_path, monkeypatch) -> None:
    errors: list[dict] = []
    forwarder = SidecarForwarder(
        tmp_path,
        "http://localhost:8000",
        "bot1",
        "secret",
        error_callback=errors.append,
    )
    monkeypatch.setattr(forwarder._stop_event, "wait", lambda _wait: False)

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("relay unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    delivered = forwarder._send_batch([{"event_id": "e1"}], "instrumented_trades")

    assert delivered is False
    assert len(errors) == 1
    assert errors[0]["component"] == "sidecar"
    assert errors[0]["error_type"] == "OSError"
    assert errors[0]["event_type"] == "instrumented_trades"
    assert errors[0]["recovery_action"] == "retry_next_poll"
    assert "relay unavailable" in errors[0]["message"]


def test_sidecar_does_not_append_recursive_error_when_forwarding_errors(tmp_path, monkeypatch) -> None:
    errors: list[dict] = []
    forwarder = SidecarForwarder(
        tmp_path,
        "http://localhost:8000",
        "bot1",
        "secret",
        error_callback=errors.append,
    )
    monkeypatch.setattr(forwarder._stop_event, "wait", lambda _wait: False)

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("relay unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    delivered = forwarder._send_batch([{"event_id": "err_1"}], "errors")

    assert delivered is False
    assert errors == []
    assert forwarder.status()["consecutive_send_failures"] == 1
    assert forwarder.status()["last_error"] == "relay unavailable"


def test_acceptance_guide_is_review_visible_to_git() -> None:
    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "docs/2026-05-31-crypto-trader-instrumentation-implementation-guide.md",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
