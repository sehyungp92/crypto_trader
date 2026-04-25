"""Portfolio coordinator — BrokerProxy and multi-strategy orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from crypto_trader.core.broker import BrokerAdapter
from crypto_trader.core.models import Fill, Order, OrderStatus, Position, Side
from crypto_trader.portfolio.manager import PortfolioManager

log = structlog.get_logger()


class BrokerProxy:
    """Wraps a real broker, intercepting entry orders for portfolio approval.

    - Entry orders (tag="entry"): check with PortfolioManager → approved? forward
      with size_multiplier applied : reject the order.
    - Exit/stop/TP orders: pass through unconditionally.
    - All other BrokerAdapter methods delegate directly.

    Implements the BrokerAdapter protocol so strategies see a uniform interface.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        manager: PortfolioManager,
        strategy_id: str,
        coordinator: "StrategyCoordinator | None" = None,
    ) -> None:
        self._broker = broker
        self._manager = manager
        self.strategy_id = strategy_id
        self._coordinator = coordinator

    def submit_order(self, order: Order) -> str:
        """Submit an order, intercepting entries for portfolio approval."""
        if order.tag == "entry":
            direction = order.side
            risk_R = order.metadata.get("risk_R", 1.0)

            result = self._manager.check_entry(
                strategy_id=self.strategy_id,
                symbol=order.symbol,
                direction=direction,
                new_risk_R=risk_R,
            )

            if not result.approved:
                log.info(
                    "portfolio.entry_blocked",
                    strategy=self.strategy_id,
                    symbol=order.symbol,
                    reason=result.denial_reason,
                )
                order.status = OrderStatus.REJECTED
                return order.order_id

            # Apply size multiplier from drawdown tiers
            if result.size_multiplier != 1.0:
                order.qty = order.qty * result.size_multiplier
                order.metadata["risk_R"] = risk_R * result.size_multiplier
                log.debug(
                    "portfolio.size_adjusted",
                    strategy=self.strategy_id,
                    multiplier=result.size_multiplier,
                    new_qty=order.qty,
                )

            # Store strategy_id in order metadata for fill routing
            order.metadata["strategy_id"] = self.strategy_id

        result_id = self._broker.submit_order(order)

        # Register order ownership for fill routing (works with any broker)
        if order.status != OrderStatus.REJECTED and self._coordinator is not None:
            self._coordinator.register_order(result_id, self.strategy_id)

        return result_id

    def cancel_order(self, order_id: str) -> bool:
        return self._broker.cancel_order(order_id)

    def cancel_all(self, symbol: str = "") -> int:
        return self._broker.cancel_all(symbol)

    def get_position(self, symbol: str) -> Position | None:
        return self._broker.get_position(symbol)

    def get_positions(self) -> list[Position]:
        return self._broker.get_positions()

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        return self._broker.get_open_orders(symbol)

    def get_equity(self) -> float:
        return self._broker.get_equity()

    def get_fills_since(self, since: datetime) -> list[Fill]:
        return self._broker.get_fills_since(since)

    def get_portfolio_snapshot(self, symbol: str, direction: Side) -> dict[str, float | int]:
        """Capture a compact pre-entry portfolio snapshot for instrumentation."""
        state = self._manager.state
        return {
            "heat_R": state.total_heat_R(),
            "heat_cap_R": self._manager.config.heat_cap_R,
            "open_risk_count": state.total_positions(),
            "directional_risk_R": state.directional_risk_R(direction),
            "symbol_risk_R": state.symbol_risk_R(symbol, direction),
            "portfolio_daily_pnl_R": state.portfolio_daily_pnl_R,
            "strategy_daily_pnl_R": state.strategy_daily_pnl_R(self.strategy_id),
        }

    # Delegate SimBroker-specific methods for backtest compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self._broker, name)


class StrategyCoordinator:
    """Orchestrates multiple strategies sharing one broker + one portfolio manager.

    Creates BrokerProxy per strategy, tracks position book, routes fills.

    Fill routing: uses _order_owners dict (populated by BrokerProxy on submit).
    Entry registration: done on entry fills (tag="entry").
    Exit registration: done via on_trade_closed() when a PositionClosedEvent fires.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        manager: PortfolioManager,
    ) -> None:
        self._broker = broker
        self._manager = manager
        self._proxies: dict[str, BrokerProxy] = {}
        self._order_owners: dict[str, str] = {}  # order_id → strategy_id

    def get_proxy(self, strategy_id: str) -> BrokerProxy:
        """Get or create a BrokerProxy for a strategy."""
        if strategy_id not in self._proxies:
            self._proxies[strategy_id] = BrokerProxy(
                broker=self._broker,
                manager=self._manager,
                strategy_id=strategy_id,
                coordinator=self,
            )
        return self._proxies[strategy_id]

    def register_order(self, order_id: str, strategy_id: str) -> None:
        """Track which strategy submitted an order."""
        self._order_owners[order_id] = strategy_id

    def get_strategy_for_order(self, order_id: str) -> str | None:
        """Look up which strategy submitted an order."""
        # Primary: our own tracking (works with any broker)
        if order_id in self._order_owners:
            return self._order_owners[order_id]
        # Fallback: broker._orders (for HyperliquidBroker)
        all_orders = getattr(self._broker, '_orders', {})
        order = all_orders.get(order_id)
        if order and "strategy_id" in order.metadata:
            return order.metadata["strategy_id"]
        return None

    def on_fill(self, fill: Fill) -> str | None:
        """Route a fill to update portfolio state. Returns strategy_id or None.

        Only handles entry registration. Exit registration is handled by
        on_trade_closed() to avoid double-counting.
        """
        strategy_id = self.get_strategy_for_order(fill.order_id)
        if strategy_id is None:
            return None

        if fill.tag == "entry":
            risk_R = self._get_fill_risk_R(fill)
            self._manager.register_entry(
                strategy_id=strategy_id,
                symbol=fill.symbol,
                direction=fill.side,
                risk_R=risk_R,
                entry_time=fill.timestamp,
            )

        return strategy_id

    def on_trade_closed(
        self,
        strategy_id: str,
        symbol: str,
        pnl_R: float,
    ) -> None:
        """Called when a complete trade (round-trip) closes."""
        self._manager.register_exit(
            strategy_id=strategy_id,
            symbol=symbol,
            pnl_R=pnl_R,
        )

    def _get_fill_risk_R(self, fill: Fill) -> float:
        """Extract risk_R from the order that generated a fill."""
        # Try HyperliquidBroker's _orders dict
        all_orders = getattr(self._broker, '_orders', {})
        order = all_orders.get(fill.order_id)
        if order:
            return order.metadata.get("risk_R", 1.0)
        # Try SimBroker's pending/deferred orders
        for lst_name in ('_pending_orders', '_deferred_orders'):
            for o in getattr(self._broker, lst_name, []):
                if o.order_id == fill.order_id:
                    return o.metadata.get("risk_R", 1.0)
        return 1.0
