"""Portfolio coordinator — BrokerProxy and multi-strategy orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

import structlog

from crypto_trader.core.broker import BrokerAdapter
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, Position, Side
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
        use_manager_equity: bool = False,
    ) -> None:
        self._broker = broker
        self._manager = manager
        self.strategy_id = strategy_id
        self._coordinator = coordinator
        self._use_manager_equity = use_manager_equity
        self._broker_id_by_client_id: dict[str, str] = {}
        self._client_id_by_broker_id: dict[str, str] = {}

    def submit_order(self, order: Order) -> str:
        """Submit an order, intercepting entries for portfolio approval."""
        # Stamp every order, including stops/targets, so live and backtest
        # adapters can keep order visibility and fill routing strategy-scoped.
        client_order_id = order.order_id
        order.metadata["strategy_id"] = self.strategy_id
        if client_order_id:
            order.metadata.setdefault("client_order_id", client_order_id)

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

        result_id = self._broker.submit_order(order)
        visible_order_id = client_order_id or result_id

        # Register order ownership for fill routing (works with any broker)
        if order.status != OrderStatus.REJECTED and self._coordinator is not None:
            for tracking_id in self._tracking_order_ids(result_id, visible_order_id, order):
                self._coordinator.register_order(tracking_id, self.strategy_id, order)

        if (
            order.status != OrderStatus.REJECTED
            and client_order_id
            and result_id
            and client_order_id != result_id
        ):
            self._broker_id_by_client_id[client_order_id] = result_id
            self._client_id_by_broker_id[result_id] = client_order_id

        return visible_order_id

    def _tracking_order_ids(self, result_id: str, visible_order_id: str, order: Order) -> list[str]:
        ids = [
            result_id,
            visible_order_id,
            order.metadata.get("exchange_order_id"),
            order.metadata.get("broker_order_id"),
            order.metadata.get("client_order_id"),
        ]
        local_to_oid = getattr(self._broker, "_local_to_oid", None)
        if isinstance(local_to_oid, dict) and (exchange_oid := local_to_oid.get(result_id)):
            ids.append(str(exchange_oid))
        return list(dict.fromkeys(str(oid) for oid in ids if oid))

    def cancel_order(self, order_id: str) -> bool:
        broker_order_id = self._broker_id_by_client_id.get(order_id, order_id)
        return self._broker.cancel_order(broker_order_id)

    def cancel_all(self, symbol: str = "") -> int:
        cancelled = 0
        for order in self.get_open_orders(symbol):
            if self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    def expire_ttl_orders_for_bar(self, bar: Bar) -> list:
        expire_fn = getattr(self._broker, "expire_ttl_orders_for_bar", None)
        if callable(expire_fn):
            return expire_fn(bar)
        return []

    def drain_immediate_fill_syncs(self) -> None:
        drain = getattr(self._broker, "drain_immediate_fill_syncs", None)
        if callable(drain):
            drain()

    def get_position(self, symbol: str) -> Position | None:
        return self._broker.get_position(symbol)

    def get_positions(self) -> list[Position]:
        return self._broker.get_positions()

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        orders: list[Order] = []
        for order in self._broker.get_open_orders(symbol):
            if not self._owns_order(order):
                continue
            orders.append(self._strategy_visible_order(order))
        return orders

    def get_equity(self) -> float:
        if self._use_manager_equity:
            return self._manager.state.equity
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

    def _owns_order(self, order: Order) -> bool:
        owner = order.metadata.get("strategy_id")
        if owner is None and self._coordinator is not None:
            owner = self._coordinator.get_strategy_for_order(order.order_id)
        return owner == self.strategy_id

    def _strategy_visible_order(self, order: Order) -> Order:
        client_order_id = (
            order.metadata.get("client_order_id")
            or self._client_id_by_broker_id.get(order.order_id)
        )
        if not client_order_id or client_order_id == order.order_id:
            return order
        metadata = dict(order.metadata)
        metadata.setdefault("broker_order_id", order.order_id)
        return replace(order, order_id=str(client_order_id), metadata=metadata)

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
        self._order_metadata: dict[str, dict] = {}
        self._order_owners: dict[str, str] = {}  # order_id → strategy_id

    def get_proxy(self, strategy_id: str, *, use_manager_equity: bool = False) -> BrokerProxy:
        """Get or create a BrokerProxy for a strategy."""
        if strategy_id not in self._proxies:
            self._proxies[strategy_id] = BrokerProxy(
                broker=self._broker,
                manager=self._manager,
                strategy_id=strategy_id,
                coordinator=self,
                use_manager_equity=use_manager_equity,
            )
        elif use_manager_equity:
            self._proxies[strategy_id]._use_manager_equity = True
        return self._proxies[strategy_id]

    def register_order(
        self,
        order_id: str,
        strategy_id: str,
        order: Order | None = None,
    ) -> None:
        """Track which strategy submitted an order."""
        self._order_owners[order_id] = strategy_id
        metadata = (
            dict(order.metadata)
            if order is not None
            else dict(self._order_metadata.get(order_id, {}))
        )
        metadata.setdefault("strategy_id", strategy_id)
        self._order_metadata[order_id] = metadata

    def get_strategy_for_order(self, order_id: str) -> str | None:
        """Look up which strategy submitted an order."""
        # Primary: our own tracking (works with any broker)
        if order_id in self._order_owners:
            return self._order_owners[order_id]
        if order_id in self._order_metadata:
            owner = self._order_metadata[order_id].get("strategy_id")
            if owner:
                return str(owner)
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
        if fill.order_id in self._order_metadata:
            return self._order_metadata[fill.order_id].get("risk_R", 1.0)
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
