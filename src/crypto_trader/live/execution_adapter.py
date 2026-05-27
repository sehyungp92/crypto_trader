"""Execution adapter wrapper for Hyperliquid live/paper trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from crypto_trader.core.execution_adapter import (
    ExecutionCapabilities,
    unsupported_order_intent_reasons,
)
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType, Side
from crypto_trader.core.runtime_types import (
    ExecutionReport,
    ExecutionReportKind,
    OrderIntent,
)
from crypto_trader.live.broker import HyperliquidBroker


class HyperliquidExecutionAdapter:
    """Translate canonical intents/reports to the existing live broker."""

    capabilities = ExecutionCapabilities(
        stop_limit=False,
        reduce_only=False,
        oca=False,
        bracket=False,
        ttl=True,
        partial_fills=True,
    )

    def __init__(self, broker: HyperliquidBroker, *, strategy_id: str = "") -> None:
        self._broker = broker
        self._strategy_id = strategy_id
        self._ttl_orders: dict[str, _TrackedTtlOrder] = {}

    def submit(self, intent: OrderIntent) -> list[ExecutionReport]:
        unsupported = self._unsupported_reason(intent)
        client_order_id = intent.client_order_id or intent.intent_id
        if unsupported:
            return [_reject_report(intent, client_order_id, unsupported)]

        metadata = dict(intent.metadata)
        metadata.update({
            "strategy_id": intent.strategy_id,
            "client_order_id": client_order_id,
            "decision_id": intent.decision_id,
            "time_in_force": intent.time_in_force,
            "ttl_bars": intent.ttl_bars,
            **intent.risk_metadata,
        })
        order = Order(
            order_id=client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            qty=intent.qty,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            tag=str(intent.metadata.get("tag") or ""),
            oca_group=intent.oca_group,
            time_in_force=intent.time_in_force,
            ttl_bars=intent.ttl_bars,
            metadata=metadata,
        )
        local_id = self._broker.submit_order(order)
        exchange_oid = _dict_attr(self._broker, "_local_to_oid").get(local_id, "")
        if (
            order.status in (OrderStatus.PENDING, OrderStatus.WORKING)
            and intent.ttl_bars is not None
        ):
            self._ttl_orders[local_id] = _TrackedTtlOrder(
                client_order_id=local_id,
                exchange_order_id=exchange_oid,
                strategy_id=intent.strategy_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                qty=order.qty,
                remaining_qty=order.qty,
                ttl_bars=intent.ttl_bars,
                metadata=_report_metadata(intent),
            )
        kind = _kind_from_status(order.status)
        return [ExecutionReport(
            report_id=f"hl_{kind.value}_{local_id}",
            kind=kind,
            timestamp=datetime.now(timezone.utc),
            symbol=order.symbol,
            side=order.side,
            client_order_id=local_id,
            exchange_order_id=exchange_oid,
            order_status=order.status,
            qty=order.qty,
            reject_reason="" if order.status != OrderStatus.REJECTED else "broker_rejected",
            metadata=_report_metadata(intent),
        )]

    def cancel(self, client_order_id: str) -> list[ExecutionReport]:
        exchange_oid = _dict_attr(self._broker, "_local_to_oid").get(client_order_id, "")
        ok = self._broker.cancel_order(client_order_id)
        if ok:
            self.clear_ttl_order(client_order_id, exchange_order_id=exchange_oid)
        return [ExecutionReport(
            report_id=f"hl_cancel_{client_order_id}",
            kind=ExecutionReportKind.CANCELLED if ok else ExecutionReportKind.REJECTED,
            timestamp=datetime.now(timezone.utc),
            symbol="",
            client_order_id=client_order_id,
            exchange_order_id=exchange_oid,
            order_status=OrderStatus.CANCELLED if ok else OrderStatus.REJECTED,
            reject_reason="" if ok else "cancel_rejected",
        )]

    def expire_ttl_orders_for_bar(self, bar: Bar) -> list[ExecutionReport]:
        """Cancel live-emulated TTL orders after their primary-timeframe bar budget."""
        reports: list[ExecutionReport] = []
        for tracked in list(self._ttl_orders.values()):
            if tracked.symbol != bar.symbol:
                continue

            tracked.bars_alive += 1
            self._sync_tracked_order_state(tracked)
            if tracked.bars_alive < tracked.ttl_bars:
                continue

            exchange_oid = _dict_attr(self._broker, "_local_to_oid").get(
                tracked.client_order_id,
                tracked.exchange_order_id,
            )
            ok = self._broker.cancel_order(tracked.client_order_id)
            if ok:
                self.clear_ttl_order(
                    tracked.client_order_id,
                    exchange_order_id=tracked.exchange_order_id,
                )
                reports.append(ExecutionReport(
                    report_id=f"hl_expired_{tracked.client_order_id}_{tracked.bars_alive}",
                    kind=ExecutionReportKind.EXPIRED,
                    timestamp=datetime.now(timezone.utc),
                    symbol=tracked.symbol,
                    side=tracked.side,
                    client_order_id=tracked.client_order_id,
                    exchange_order_id=exchange_oid,
                    order_status=OrderStatus.EXPIRED,
                    qty=tracked.qty,
                    metadata={
                        **tracked.metadata,
                        "ttl_bars_alive": tracked.bars_alive,
                    },
                ))
            else:
                reports.append(ExecutionReport(
                    report_id=(
                        f"hl_ttl_cancel_reject_"
                        f"{tracked.client_order_id}_{tracked.bars_alive}"
                    ),
                    kind=ExecutionReportKind.RESTING,
                    timestamp=datetime.now(timezone.utc),
                    symbol=tracked.symbol,
                    side=tracked.side,
                    client_order_id=tracked.client_order_id,
                    exchange_order_id=exchange_oid,
                    order_status=OrderStatus.WORKING,
                    qty=tracked.qty,
                    reject_reason="ttl_cancel_rejected",
                    metadata={
                        **tracked.metadata,
                        "ttl_bars_alive": tracked.bars_alive,
                        "ttl_cancel_failed": True,
                    },
                ))
        return reports

    def seed_ttl_orders_from_open_orders(self) -> int:
        """Seed local TTL emulation state from currently open exchange orders."""
        return self.seed_ttl_orders(self._broker.get_open_orders())

    def seed_ttl_orders(self, orders: list[Order]) -> int:
        """Seed local TTL emulation state from a caller-provided open-order snapshot."""
        seeded = 0
        for order in orders:
            if order.ttl_bars is None or not self._owns_order(order):
                continue
            exchange_oid = _dict_attr(self._broker, "_local_to_oid").get(order.order_id, "")
            if self._track_ttl_order(order, exchange_oid):
                seeded += 1
        return seeded

    def clear_ttl_for_fill(self, fill: Fill) -> bool:
        """Clear or reduce local TTL state when an order fill is observed."""
        return self.clear_ttl_order(
            fill.order_id,
            exchange_order_id=fill.exchange_order_id,
            filled_qty=fill.qty,
        )

    def clear_ttl_order(
        self,
        order_id: str,
        *,
        exchange_order_id: str = "",
        filled_qty: float | None = None,
    ) -> bool:
        key = self._ttl_key(order_id) or self._ttl_key(exchange_order_id)
        if key is None:
            return False

        tracked = self._ttl_orders[key]
        if tracked.order_type == OrderType.STOP or filled_qty is None:
            self._ttl_orders.pop(key, None)
            return True

        tracked.remaining_qty = max(0.0, tracked.remaining_qty - max(0.0, filled_qty))
        if tracked.remaining_qty <= 1e-12:
            self._ttl_orders.pop(key, None)
        else:
            self._sync_tracked_order_state(tracked)
        return True

    def active_ttl_orders(self) -> list[dict]:
        return [
            {
                "client_order_id": tracked.client_order_id,
                "exchange_order_id": tracked.exchange_order_id,
                "strategy_id": tracked.strategy_id,
                "symbol": tracked.symbol,
                "side": tracked.side.value,
                "order_type": tracked.order_type.value,
                "qty": tracked.qty,
                "remaining_qty": tracked.remaining_qty,
                "ttl_bars": tracked.ttl_bars,
                "ttl_bars_alive": tracked.bars_alive,
                "metadata": dict(tracked.metadata),
            }
            for tracked in self._ttl_orders.values()
        ]

    def sync_open_orders(self) -> list[ExecutionReport]:
        reports = []
        for order in self._broker.get_open_orders():
            if not self._owns_order(order):
                continue
            exchange_oid = _dict_attr(self._broker, "_local_to_oid").get(order.order_id, "")
            if order.ttl_bars is not None:
                self._track_ttl_order(order, exchange_oid)
            reports.append(ExecutionReport(
                report_id=f"hl_resting_{order.order_id}",
                kind=ExecutionReportKind.RESTING,
                timestamp=datetime.now(timezone.utc),
                symbol=order.symbol,
                side=order.side,
                client_order_id=order.order_id,
                exchange_order_id=exchange_oid,
                order_status=order.status,
                qty=order.qty,
                metadata=dict(order.metadata),
            ))
        return reports

    def sync_positions(self) -> list[dict]:
        return [position.__dict__.copy() for position in self._broker.get_positions()]

    def sync_fills(self, watermark: datetime) -> list[ExecutionReport]:
        fills = self._broker.get_fills_since(watermark)
        for fill in fills:
            self.clear_ttl_for_fill(fill)
        return [_fill_report(fill) for fill in fills]

    def _unsupported_reason(self, intent: OrderIntent) -> str:
        return next(iter(unsupported_order_intent_reasons(intent, self.capabilities)), "")

    def _owns_order(self, order: Order) -> bool:
        if not self._strategy_id:
            return True
        return str(order.metadata.get("strategy_id") or "") == self._strategy_id

    def _track_ttl_order(self, order: Order, exchange_oid: str = "") -> bool:
        if order.ttl_bars is None:
            return False
        tracked = self._ttl_orders.get(order.order_id)
        metadata = dict(order.metadata)
        bars_alive = _int_or_default(
            metadata.get("ttl_bars_alive"),
            getattr(order, "_bars_alive", 0),
        )
        if tracked is None:
            self._ttl_orders[order.order_id] = _TrackedTtlOrder(
                client_order_id=order.order_id,
                exchange_order_id=exchange_oid,
                strategy_id=str(metadata.get("strategy_id") or self._strategy_id),
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                qty=order.qty,
                remaining_qty=_float_or_default(metadata.get("ttl_remaining_qty"), order.qty),
                ttl_bars=int(order.ttl_bars),
                bars_alive=bars_alive,
                metadata=metadata,
            )
            return True

        tracked.exchange_order_id = exchange_oid or tracked.exchange_order_id
        tracked.bars_alive = max(tracked.bars_alive, bars_alive)
        tracked.remaining_qty = _float_or_default(
            metadata.get("ttl_remaining_qty"),
            tracked.remaining_qty,
        )
        tracked.metadata.update(metadata)
        return False

    def _ttl_key(self, order_id: str) -> str | None:
        if not order_id:
            return None
        if order_id in self._ttl_orders:
            return order_id
        local_id = _dict_attr(self._broker, "_oid_map").get(str(order_id))
        if local_id in self._ttl_orders:
            return local_id
        for key, tracked in self._ttl_orders.items():
            if tracked.exchange_order_id == order_id:
                return key
        return None

    def _sync_tracked_order_state(self, tracked: "_TrackedTtlOrder") -> None:
        tracked.metadata["ttl_bars_alive"] = tracked.bars_alive
        tracked.metadata["ttl_remaining_qty"] = tracked.remaining_qty
        broker_orders = getattr(self._broker, "_orders", None)
        if not isinstance(broker_orders, dict):
            return
        order = broker_orders.get(tracked.client_order_id)
        if order is None:
            return
        order._bars_alive = tracked.bars_alive
        order.metadata["ttl_bars_alive"] = tracked.bars_alive
        order.metadata["ttl_remaining_qty"] = tracked.remaining_qty


@dataclass(slots=True)
class _TrackedTtlOrder:
    client_order_id: str
    exchange_order_id: str
    strategy_id: str
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    remaining_qty: float
    ttl_bars: int
    bars_alive: int = 0
    metadata: dict = field(default_factory=dict)


def _int_or_default(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _dict_attr(source, name: str) -> dict:
    value = getattr(source, name, {})
    return value if isinstance(value, dict) else {}


def _kind_from_status(status: OrderStatus) -> ExecutionReportKind:
    if status == OrderStatus.FILLED:
        return ExecutionReportKind.FILL
    if status == OrderStatus.PARTIALLY_FILLED:
        return ExecutionReportKind.PARTIAL_FILL
    if status == OrderStatus.REJECTED:
        return ExecutionReportKind.REJECTED
    return ExecutionReportKind.ACCEPTED


def _reject_report(
    intent: OrderIntent,
    client_order_id: str,
    reason: str,
) -> ExecutionReport:
    return ExecutionReport(
        report_id=f"hl_reject_{client_order_id}",
        kind=ExecutionReportKind.REJECTED,
        timestamp=datetime.now(timezone.utc),
        symbol=intent.symbol,
        side=intent.side,
        client_order_id=client_order_id,
        order_status=OrderStatus.REJECTED,
        qty=intent.qty,
        reject_reason=reason,
        metadata=_report_metadata(intent),
    )


def _report_metadata(intent: OrderIntent) -> dict:
    metadata = dict(intent.metadata)
    metadata.update({
        "intent_id": intent.intent_id,
        "strategy_id": intent.strategy_id,
        "decision_id": intent.decision_id,
        "order_type": intent.order_type.value,
        "reduce_only": intent.reduce_only,
        "time_in_force": intent.time_in_force,
        "ttl_bars": intent.ttl_bars,
        "oca_group": intent.oca_group,
        "bracket_group": intent.bracket_group,
    })
    return metadata


def _fill_report(fill: Fill) -> ExecutionReport:
    return ExecutionReport(
        report_id=f"hl_fill_{fill.order_id}_{int(fill.timestamp.timestamp() * 1000)}",
        kind=ExecutionReportKind.FILL,
        timestamp=fill.timestamp,
        symbol=fill.symbol,
        side=fill.side,
        client_order_id=fill.order_id,
        exchange_order_id=fill.exchange_order_id or fill.order_id,
        fill_id=fill.exchange_fill_id or f"{fill.order_id}:{int(fill.timestamp.timestamp() * 1000)}",
        order_status=OrderStatus.FILLED,
        filled_qty=fill.qty,
        fill_price=fill.fill_price,
        commission=fill.commission,
        metadata={"tag": fill.tag, **dict(fill.raw)},
    )
