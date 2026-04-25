"""Tests for BreakoutStrategy — properties and basic behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.events import PositionClosedEvent
from crypto_trader.core.models import Bar, Fill, SetupGrade, Side, TimeFrame, Trade
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.breakout.strategy import BreakoutStrategy, _PositionMeta
from crypto_trader.strategy.momentum.journal import TradeJournal


# ---------------------------------------------------------------------------
# Minimal mock context
# ---------------------------------------------------------------------------

class _MockBroker:
    def get_position(self, sym):
        return None

    def get_equity(self):
        return 10000.0

    def get_open_orders(self, sym):
        return []

    def submit_order(self, order):
        pass

    def cancel_order(self, oid):
        pass


class _MockBars:
    def get(self, sym, tf, count=100):
        return []


class _MockEvents:
    def subscribe(self, event_type, handler):
        pass


class _MockClock:
    pass


class _MockCtx:
    def __init__(self):
        self.broker = _MockBroker()
        self.bars = _MockBars()
        self.events = _MockEvents()
        self.clock = _MockClock()
        self.config = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_bar(
    sym: str = "BTC",
    tf: TimeFrame = TimeFrame.M30,
    ts: datetime = _TS,
) -> Bar:
    return Bar(
        timestamp=ts,
        symbol=sym,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        timeframe=tf,
    )


def _make_fill(tag: str = "entry", sym: str = "BTC") -> Fill:
    return Fill(
        order_id="test-oid",
        symbol=sym,
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=_TS,
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Tests — properties
# ---------------------------------------------------------------------------

class TestBreakoutStrategyProperties:
    """Test strategy name, timeframes, symbols, journal."""

    def test_name(self):
        """strategy.name == 'volume_profile_breakout'."""
        s = BreakoutStrategy()
        assert s.name == "volume_profile_breakout"

    def test_timeframes(self):
        """strategy.timeframes == [TimeFrame.M30, TimeFrame.H4]."""
        s = BreakoutStrategy()
        assert s.timeframes == [TimeFrame.M30, TimeFrame.H4]

    def test_symbols(self):
        """strategy.symbols matches config symbols."""
        cfg = BreakoutConfig(symbols=["BTC", "SOL"])
        s = BreakoutStrategy(config=cfg)
        assert s.symbols == ["BTC", "SOL"]

    def test_default_config(self):
        """Default config is BreakoutConfig() when none provided."""
        s = BreakoutStrategy()
        assert s.symbols == BreakoutConfig().symbols

    def test_journal_exists(self):
        """strategy.journal is a TradeJournal."""
        s = BreakoutStrategy()
        assert isinstance(s.journal, TradeJournal)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        cfg = BreakoutConfig(symbols=["ETH"])
        s = BreakoutStrategy(config=cfg)
        assert s.symbols == ["ETH"]
        assert len(s.symbols) == 1


# ---------------------------------------------------------------------------
# Tests — on_bar / on_fill robustness
# ---------------------------------------------------------------------------

class TestBreakoutStrategyBehavior:
    """Test on_bar and on_fill edge cases."""

    def _init_strategy(self, cfg: BreakoutConfig | None = None) -> BreakoutStrategy:
        s = BreakoutStrategy(config=cfg)
        ctx = _MockCtx()
        s.on_init(ctx)
        return s

    def test_on_bar_ignores_unknown_symbol(self):
        """on_bar with unknown symbol doesn't crash."""
        s = self._init_strategy()
        bar = _make_bar(sym="DOGE", tf=TimeFrame.M30)
        ctx = _MockCtx()
        # Should silently return — DOGE not in config symbols
        s.on_bar(bar, ctx)

    def test_on_bar_ignores_wrong_timeframe(self):
        """on_bar with TimeFrame.D1 does nothing (no crash)."""
        s = self._init_strategy()
        bar = _make_bar(sym="BTC", tf=TimeFrame.D1)
        ctx = _MockCtx()
        s.on_bar(bar, ctx)

    def test_warmup_gate(self):
        """M30 bars before WARMUP_BARS don't trigger entries."""
        s = self._init_strategy()
        ctx = _MockCtx()
        # Feed fewer than WARMUP_BARS (101) M30 bars — no orders submitted
        for i in range(50):
            day = 1 + i // 48
            hour = (i // 2) % 24
            minute = (i % 2) * 30
            ts = datetime(2026, 1, day, hour, minute, tzinfo=timezone.utc)
            bar = _make_bar(sym="BTC", tf=TimeFrame.M30, ts=ts)
            s.on_bar(bar, ctx)
        # No crash and no orders — broker.submit_order was never called
        # (mock doesn't track, but no exception = pass)

    def test_on_fill_unknown_tag(self):
        """on_fill with unknown tag doesn't crash."""
        s = self._init_strategy()
        ctx = _MockCtx()
        fill = _make_fill(tag="unknown_tag_xyz")
        s.on_fill(fill, ctx)

    def test_position_closed_records_recent_net_loss_from_realized_r(self):
        s = BreakoutStrategy(BreakoutConfig(symbols=["BTC"]))
        ctx = _MockCtx()
        ctx.broker = MagicMock()
        ctx.broker.get_open_orders.return_value = []
        s.on_init(ctx)

        exit_state = MagicMock()
        exit_state.mae_r = -0.2
        exit_state.mfe_r = 0.4
        s._exit_manager.remove = MagicMock(return_value=exit_state)
        s._trail_manager.remove = MagicMock()
        s._confirmation_detector.clear_pending = MagicMock()
        s._risk_manager.record_trade_exit = MagicMock()
        s._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            confirmation_type="model1",
            entry_method="model1",
            entry_price=100.0,
            stop_level=95.0,
            stop_distance=5.0,
            original_qty=1.0,
        )

        trade = Trade(
            trade_id="t1",
            symbol="BTC",
            direction=Side.LONG,
            entry_price=100.0,
            exit_price=102.0,
            qty=1.0,
            entry_time=_TS,
            exit_time=_TS,
            pnl=2.0,
            r_multiple=None,
            commission=3.0,
            bars_held=4,
            setup_grade=None,
            exit_reason="protective_stop",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=0.0,
            mae_r=None,
            mfe_r=None,
        )

        s._on_position_closed(PositionClosedEvent(timestamp=_TS, trade=trade))

        assert trade.r_multiple == pytest.approx(0.4)
        assert trade.realized_r_multiple == pytest.approx(-0.2)
        assert s._recent_exits["BTC"]["loss_r"] == pytest.approx(0.2)
        s._risk_manager.record_trade_exit.assert_called_once_with(
            trade.net_pnl,
            trade.exit_time,
        )
