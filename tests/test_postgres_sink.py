"""Unit tests for PostgresSink with mocked psycopg pool."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from types import ModuleType

import pytest

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    EventMetadata,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MarketContext,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)


# ---------------------------------------------------------------------------
# Mock psycopg_pool at module level (psycopg not installed in dev)
# ---------------------------------------------------------------------------

_mock_psycopg_pool = ModuleType("psycopg_pool")
_MockConnectionPool = MagicMock()
_mock_psycopg_pool.ConnectionPool = _MockConnectionPool
sys.modules.setdefault("psycopg_pool", _mock_psycopg_pool)

from crypto_trader.instrumentation.postgres_sink import PostgresSink  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(**overrides) -> EventMetadata:
    defaults = dict(
        event_id="evt_001",
        bot_id="test_bot",
        strategy_id="momentum",
        exchange_timestamp=datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return EventMetadata(**defaults)


def _make_trade_event(**overrides) -> InstrumentedTradeEvent:
    defaults = dict(
        metadata=_make_metadata(),
        trade_id="t_001",
        pair="BTC",
        side="long",
        entry_time=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc),
        entry_price=90000.0,
        exit_price=91000.0,
        position_size=0.1,
        pnl=100.0,
        commission=2.0,
        funding_paid=0.5,
        setup_grade="A",
        exit_reason="trailing_stop",
        entry_method="market",
        confluences=["ema_stack", "adx_trend"],
        r_multiple=1.5,
        mae_r=-0.3,
        mfe_r=2.0,
        exit_efficiency=0.75,
    )
    defaults.update(overrides)
    return InstrumentedTradeEvent(**defaults)


def _make_daily_event(**overrides) -> DailySnapshot:
    defaults = dict(
        metadata=_make_metadata(),
        date="2026-04-20",
        total_trades=5,
        win_count=3,
        loss_count=2,
        gross_pnl=500.0,
        net_pnl=480.0,
        max_drawdown_pct=2.5,
        sharpe_rolling_30d=1.8,
        sortino_rolling_30d=2.1,
        per_strategy_summary={"momentum": {"trades": 3}},
    )
    defaults.update(overrides)
    return DailySnapshot(**defaults)


def _make_sink():
    """Create PostgresSink with a fresh mock pool and connection."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    # connection() returns a context manager yielding mock_conn
    mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

    _MockConnectionPool.reset_mock()
    _MockConnectionPool.return_value = mock_pool

    sink = PostgresSink("postgresql://test:test@localhost/test")
    return sink, mock_conn, mock_pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWriteTrade:
    def test_maps_fields_correctly(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event()

        sink.write_trade(event)

        mock_conn.execute.assert_called_once()
        sql, params = mock_conn.execute.call_args.args
        assert "INSERT INTO trades" in sql
        assert "ON CONFLICT (trade_id) DO NOTHING" in sql
        # Verify key field positions
        assert params[0] == "t_001"  # trade_id
        assert params[1] == "momentum"  # strategy_id
        assert params[2] == "BTC"  # symbol
        assert params[3] == "long"  # direction
        assert params[9] == 100.0  # pnl
        assert params[10] == 100.0 - 0.5  # net_pnl = pnl(already net of commission) - funding
        assert params[11] == 1.5  # r_multiple

    def test_idempotent_no_exception(self):
        """Duplicate trade_id should not raise (ON CONFLICT DO NOTHING)."""
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event()

        # Call twice — should not raise
        sink.write_trade(event)
        sink.write_trade(event)
        assert mock_conn.execute.call_count == 2

    def test_with_market_context(self):
        sink, mock_conn, _ = _make_sink()
        ctx = MarketContext(atr=100.0, adx=25.0, rsi=55.0)
        event = _make_trade_event(market_context=ctx)

        sink.write_trade(event)

        _, params = mock_conn.execute.call_args.args
        # market_context should be JSON string
        mc_json = params[-1]  # last param
        parsed = json.loads(mc_json)
        assert parsed["atr"] == 100.0
        assert parsed["adx"] == 25.0


class TestWriteDaily:
    def test_upserts_daily_snapshot(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_daily_event()

        sink.write_daily(event)

        mock_conn.execute.assert_called_once()
        sql, params = mock_conn.execute.call_args.args
        assert "INSERT INTO daily_snapshots" in sql
        assert "ON CONFLICT (trade_date) DO UPDATE" in sql
        assert params[0] == "2026-04-20"
        assert params[1] == 5  # total_trades
        assert params[2] == 3  # win_count


class TestWriteHealthReport:
    def test_extracts_assessment(self):
        sink, mock_conn, _ = _make_sink()
        event = HealthReportSnapshot(
            timestamp="2026-04-20T12:00:00+00:00",
            report={
                "assessment": "healthy",
                "uptime_sec": 3600.0,
                "alerts": ["stale_feed_BTC_15m"],
                "positions": [],
            },
        )

        sink.write_health_report(event)

        mock_conn.execute.assert_called_once()
        sql, params = mock_conn.execute.call_args.args
        assert "INSERT INTO health_snapshots" in sql
        assert params[1] == "healthy"  # assessment extracted from report
        assert params[2] == 3600.0  # uptime_sec


class TestWriteEquity:
    def test_inserts_equity_snapshot(self):
        sink, mock_conn, _ = _make_sink()
        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)

        sink.write_equity(10500.0, ts)

        mock_conn.execute.assert_called_once()
        sql, params = mock_conn.execute.call_args.args
        assert "INSERT INTO equity_snapshots" in sql
        assert params[0] == ts
        assert params[1] == 10500.0


class TestUpsertPositions:
    def test_full_sync_delete_then_insert(self):
        sink, mock_conn, _ = _make_sink()
        positions = [
            {
                "strategy_id": "momentum",
                "symbol": "BTC",
                "direction": "long",
                "qty": 0.1,
                "avg_entry": 90000.0,
                "unrealized_pnl": 100.0,
                "risk_r": 0.5,
                "stop_price": 89000.0,
                "entry_time": datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
            },
            {
                "strategy_id": "trend",
                "symbol": "ETH",
                "direction": "short",
                "qty": 1.0,
                "avg_entry": 3000.0,
                "unrealized_pnl": -20.0,
                "risk_r": 0.3,
            },
        ]

        sink.upsert_positions(positions)

        calls = mock_conn.execute.call_args_list
        # First call: DELETE
        assert "DELETE FROM positions" in calls[0].args[0]
        # Then 2 INSERTs
        assert "INSERT INTO positions" in calls[1].args[0]
        assert "INSERT INTO positions" in calls[2].args[0]
        # Verify first position params
        assert calls[1].args[1][1] == "BTC"
        # Verify second position params
        assert calls[2].args[1][1] == "ETH"


class TestNoopMethods:
    def test_noop_methods_dont_fail(self):
        sink, mock_conn, _ = _make_sink()

        # These should be no-ops — no DB calls
        sink.write_missed(MagicMock(spec=MissedOpportunityEvent))
        sink.write_error(MagicMock(spec=ErrorEvent))
        sink.write_funnel(MagicMock(spec=PipelineFunnelSnapshot))

        mock_conn.execute.assert_not_called()


class TestConnectionErrorHandling:
    def test_write_trade_swallows_exception(self):
        sink, mock_conn, _ = _make_sink()
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        # Should not raise
        sink.write_trade(_make_trade_event())

    def test_write_equity_swallows_exception(self):
        sink, mock_conn, _ = _make_sink()
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        # Should not raise
        sink.write_equity(10000.0, datetime.now(timezone.utc))

    def test_upsert_positions_swallows_exception(self):
        sink, mock_conn, _ = _make_sink()
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        # Should not raise
        sink.upsert_positions([{"symbol": "BTC", "direction": "long", "qty": 0.1, "avg_entry": 90000.0}])

    def test_close_swallows_exception(self):
        sink, _, mock_pool = _make_sink()
        mock_pool.close.side_effect = RuntimeError("already closed")

        # Should not raise
        sink.close()
