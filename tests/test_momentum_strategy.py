"""Targeted tests for MomentumStrategy close-path accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.events import PositionClosedEvent
from crypto_trader.core.models import SetupGrade, Side, Trade
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.momentum.strategy import MomentumStrategy, _PositionMeta


def _make_ctx():
    broker = MagicMock()
    broker.get_open_orders.return_value = []
    events = MagicMock()
    clock = MagicMock()
    bars = MagicMock()
    return SimpleNamespace(broker=broker, events=events, clock=clock, bars=bars, config={})


class TestMomentumStrategyClosePath:
    def test_position_closed_computes_realized_r_and_records_net_pnl(self):
        strategy = MomentumStrategy(MomentumConfig(symbols=["BTC"]))
        ctx = _make_ctx()
        strategy.on_init(ctx)

        exit_state = SimpleNamespace(mae_r=-0.2, mfe_r=0.6, partial_exits=[])
        strategy._exit_manager.remove_position = MagicMock(return_value=exit_state)
        strategy._trail_manager.remove = MagicMock()
        strategy._risk_manager.record_trade = MagicMock()
        strategy._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            confluences=("m15_ema20",),
            confirmation_type="inside_bar_break",
            entry_method="close",
            entry_price=50_000.0,
            stop_level=49_500.0,
            stop_distance=500.0,
            original_qty=0.2,
        )

        trade = Trade(
            trade_id="t1",
            symbol="BTC",
            direction=Side.LONG,
            entry_price=50_000.0,
            exit_price=50_100.0,
            qty=0.1,
            entry_time=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 15, 18, 0, tzinfo=timezone.utc),
            pnl=10.0,
            r_multiple=None,
            commission=15.0,
            bars_held=8,
            setup_grade=None,
            exit_reason="tp1",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=0.0,
            mae_r=None,
            mfe_r=None,
        )

        strategy._on_position_closed(
            PositionClosedEvent(timestamp=trade.exit_time, trade=trade)
        )

        assert trade.r_multiple == pytest.approx(0.2)
        assert trade.realized_r_multiple == pytest.approx(-0.1)
        strategy._risk_manager.record_trade.assert_called_once_with(
            trade.net_pnl,
            trade.exit_time,
        )
