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
        assert result == "o1"
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
        assert result == "o3"
        broker.submit_order.assert_called_once()
        assert order.metadata["strategy_id"] == "momentum"

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
        assert order.metadata["risk_R"] == pytest.approx(0.25)

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

    def test_get_open_orders_filters_to_strategy_owner(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "momentum")
        own = Order(
            order_id="own", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "momentum"},
        )
        other = Order(
            order_id="other", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "trend"},
        )
        unknown = Order(
            order_id="unknown", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
        )
        broker.get_open_orders.return_value = [own, other, unknown]

        assert proxy.get_open_orders("BTC") == [own]

    def test_client_order_ids_cancel_broker_assigned_ids(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "trend")

        def assign_order_id(order):
            order.order_id = "broker_1"
            return "broker_1"

        broker.submit_order.side_effect = assign_order_id
        broker.cancel_order.return_value = True

        order = Order(
            order_id="trend_stop_BTC_abc",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=0.1,
            stop_price=49000.0,
            tag="protective_stop",
        )

        assert proxy.submit_order(order) == "trend_stop_BTC_abc"
        assert proxy.cancel_order("trend_stop_BTC_abc") is True
        broker.cancel_order.assert_called_once_with("broker_1")

    def test_open_orders_expose_client_order_ids_when_broker_assigns_ids(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "trend")

        def assign_order_id(order):
            order.order_id = "broker_1"
            return "broker_1"

        broker.submit_order.side_effect = assign_order_id
        order = Order(
            order_id="trend_stop_BTC_abc",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=0.1,
            stop_price=49000.0,
            tag="protective_stop",
        )
        proxy.submit_order(order)
        broker.get_open_orders.return_value = [order]

        visible = proxy.get_open_orders("BTC")

        assert len(visible) == 1
        assert visible[0].order_id == "trend_stop_BTC_abc"
        assert visible[0].metadata["broker_order_id"] == "broker_1"
        assert order.order_id == "broker_1"

    def test_cancel_all_cancels_only_strategy_owned_orders(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "momentum")
        own = Order(
            order_id="own", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "momentum"},
        )
        other = Order(
            order_id="other", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "trend"},
        )
        broker.get_open_orders.return_value = [own, other]
        broker.cancel_order.return_value = True

        assert proxy.cancel_all("BTC") == 1
        broker.cancel_order.assert_called_once_with("own")

    def test_manager_equity_source_for_portfolio_backtests(self):
        broker, manager, state = _make_components()
        state.equity = 12345.0
        broker.get_equity.return_value = 999.0
        proxy = BrokerProxy(broker, manager, "momentum", use_manager_equity=True)

        assert proxy.get_equity() == pytest.approx(12345.0)


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

    def test_on_fill_entry_uses_registered_order_risk_metadata(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        order = Order(
            order_id="",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 0.35},
        )
        coord.register_order("o_risk", "momentum", order)
        fill = Fill(
            order_id="o_risk", symbol="BTC", side=Side.LONG,
            qty=0.1, fill_price=50000.0, commission=1.75,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc), tag="entry",
        )

        strategy_id = coord.on_fill(fill)

        assert strategy_id == "momentum"
        assert state.total_heat_R() == pytest.approx(0.35)

    def test_on_trade_closed(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)

        # Register an entry first
        manager.register_entry("momentum", "BTC", Side.LONG, 1.0)
        assert state.total_heat_R() == 1.0

        coord.on_trade_closed("momentum", "BTC", 2.0)
        assert state.total_heat_R() == 0.0
        assert state.strategy_daily_pnl_R("momentum") == 2.0
