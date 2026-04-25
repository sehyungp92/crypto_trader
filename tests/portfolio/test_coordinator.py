"""Tests for portfolio coordinator (BrokerProxy + StrategyCoordinator)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto_trader.core.models import Fill, Order, OrderStatus, OrderType, Position, Side
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio.coordinator import BrokerProxy, StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager, PortfolioRuleResult
from crypto_trader.portfolio.state import PortfolioState


def _make_components(max_total_positions=9):
    cfg = PortfolioConfig(
        strategies=(
            StrategyAllocation(strategy_id="momentum"),
            StrategyAllocation(strategy_id="trend"),
        ),
        max_total_positions=max_total_positions,
    )
    state = PortfolioState(equity=10000.0, peak_equity=10000.0)
    manager = PortfolioManager(cfg, state)
    broker = MagicMock()
    broker._orders = {}
    broker._closed_trades = []
    return broker, manager, state


class TestBrokerProxy:
    def test_entry_order_approved(self):
        broker, manager, state = _make_components()
        broker.submit_order.return_value = "order_1"
        proxy = BrokerProxy(broker, manager, "momentum")

        order = Order(
            order_id="o1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            tag="entry", metadata={"risk_R": 1.0},
        )
        result = proxy.submit_order(order)
        assert result == "order_1"
        broker.submit_order.assert_called_once()
        # Strategy ID should be stamped in metadata
        assert order.metadata["strategy_id"] == "momentum"

    def test_entry_order_denied(self):
        broker, manager, state = _make_components(max_total_positions=3)
        proxy = BrokerProxy(broker, manager, "momentum")

        # Fill up positions to trigger denial (max_total_positions=3)
        from crypto_trader.portfolio.state import OpenRisk
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        state.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 1.0))
        state.add_risk(OpenRisk("trend", "SOL", Side.LONG, 1.0))

        order = Order(
            order_id="o2", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            tag="entry", metadata={"risk_R": 0.5},
        )
        result = proxy.submit_order(order)
        assert result == "o2"
        assert order.status == OrderStatus.REJECTED
        broker.submit_order.assert_not_called()

    def test_exit_order_passthrough(self):
        broker, manager, _ = _make_components()
        broker.submit_order.return_value = "order_3"
        proxy = BrokerProxy(broker, manager, "momentum")

        order = Order(
            order_id="o3", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1,
            stop_price=50000.0, tag="stop",
        )
        result = proxy.submit_order(order)
        assert result == "order_3"
        broker.submit_order.assert_called_once()

    def test_size_multiplier_applied(self):
        broker, manager, state = _make_components()
        broker.submit_order.return_value = "order_4"
        proxy = BrokerProxy(broker, manager, "momentum")

        # Trigger drawdown tier
        state.peak_equity = 10000.0
        state.equity = 8700.0  # 13% DD → 0.50 multiplier

        order = Order(
            order_id="o4", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            tag="entry", metadata={"risk_R": 0.5},
        )
        proxy.submit_order(order)
        assert order.qty == pytest.approx(0.05)

    def test_cancel_delegates(self):
        broker, manager, _ = _make_components()
        broker.cancel_order.return_value = True
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.cancel_order("x") is True
        broker.cancel_order.assert_called_once_with("x")

    def test_get_position_delegates(self):
        broker, manager, _ = _make_components()
        pos = Position(symbol="BTC", direction=Side.LONG, qty=0.1, avg_entry=50000.0)
        broker.get_position.return_value = pos
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.get_position("BTC") == pos

    def test_get_equity_delegates(self):
        broker, manager, _ = _make_components()
        broker.get_equity.return_value = 10500.0
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.get_equity() == 10500.0

    def test_getattr_fallback(self):
        broker, manager, _ = _make_components()
        broker.some_custom_method = MagicMock(return_value=42)
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.some_custom_method() == 42


class TestStrategyCoordinator:
    def test_get_proxy_creates_once(self):
        broker, manager, _ = _make_components()
        coord = StrategyCoordinator(broker, manager)
        p1 = coord.get_proxy("momentum")
        p2 = coord.get_proxy("momentum")
        assert p1 is p2
        assert isinstance(p1, BrokerProxy)

    def test_get_proxy_different_strategies(self):
        broker, manager, _ = _make_components()
        coord = StrategyCoordinator(broker, manager)
        p1 = coord.get_proxy("momentum")
        p2 = coord.get_proxy("trend")
        assert p1 is not p2
        assert p1.strategy_id == "momentum"
        assert p2.strategy_id == "trend"

    def test_on_fill_entry(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        coord.get_proxy("momentum")  # register strategy

        # Set up order with strategy metadata
        order = Order(
            order_id="o1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1, tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 1.0},
        )
        broker._orders = {"o1": order}
        broker.get_position.return_value = Position("BTC", Side.LONG, 0.1, 50000.0)

        fill = Fill(
            order_id="o1", symbol="BTC", side=Side.LONG,
            qty=0.1, fill_price=50000.0, commission=1.75,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc), tag="entry",
        )

        strategy_id = coord.on_fill(fill)
        assert strategy_id == "momentum"

    def test_on_trade_closed(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)

        # Register an entry first
        manager.register_entry("momentum", "BTC", Side.LONG, 1.0)
        assert state.total_heat_R() == 1.0

        coord.on_trade_closed("momentum", "BTC", 2.0)
        assert state.total_heat_R() == 0.0
        assert state.strategy_daily_pnl_R("momentum") == 2.0
