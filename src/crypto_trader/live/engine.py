"""Live trading engine — async polling loop for paper/live trading."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.runner import _create_strategy
from crypto_trader.core.clock import WallClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus, PositionClosedEvent
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType, Position, SetupGrade, Side, TimeFrame, Trade
from crypto_trader.core.runtime_types import MarketEvent
from crypto_trader.core.strategy_runtime import StrategySlotRuntime
from crypto_trader.exchange.meta import AssetMeta
from crypto_trader.live.broker import HyperliquidBroker
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter
from crypto_trader.live.feed import LiveFeed
from crypto_trader.live.health import HealthMonitor
from crypto_trader.live.lifecycle import PositionLifecycleLedger
from crypto_trader.live.oms_store import (
    FILL_COORDINATOR_APPLIED_STATUSES,
    FILL_FINALIZED_STATUSES,
    FILL_LIFECYCLE_APPLIED_STATUSES,
    FILL_STRATEGY_DISPATCHED_STATUSES,
    OmsStore,
    fill_identity,
)
from crypto_trader.live.reconciler import PositionReconciler
from crypto_trader.live.state import PersistentState
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState
from crypto_trader.instrumentation.backfill import MissedOpportunityBackfiller
from crypto_trader.instrumentation.emitter import EventEmitter
from crypto_trader.instrumentation.sinks import JsonlSink
from crypto_trader.instrumentation.sidecar import SidecarForwarder
from crypto_trader.instrumentation.daily_aggregator import DailyAggregator
from crypto_trader.instrumentation.types import HealthReportSnapshot, PipelineFunnelSnapshot
from crypto_trader.instrumentation.pipeline_tracker import PipelineTracker
from crypto_trader.live.health_report import HealthReportBuilder

log = structlog.get_logger()

# Expected bar intervals by timeframe (seconds)
_TF_INTERVALS: dict[str, float] = {
    "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}

# Warmup bar counts per timeframe
_WARMUP_COUNTS = {
    TimeFrame.M15: 200,
    TimeFrame.M30: 101,
    TimeFrame.H1: 50,
    TimeFrame.H4: 50,
    TimeFrame.D1: 60,
}


@dataclass(slots=True)
class _FillProcessingResult:
    processed: list[Fill]
    duplicates: list[Fill]
    unresolved: list[Fill]
    safe_watermark_fills: list[Fill]


def _enum_or_default(enum_cls, value, default):
    try:
        if isinstance(value, enum_cls):
            return value
        return enum_cls(value)
    except (TypeError, ValueError):
        return default


def _asset_meta_broker_kwargs(asset_meta_path: Path | None) -> dict[str, dict[str, float]]:
    if asset_meta_path is None:
        return {}
    asset_meta = AssetMeta.from_cache(asset_meta_path)
    log.info(
        "engine.asset_meta_loaded",
        path=str(asset_meta_path),
        symbols=len(asset_meta.asset_index),
    )
    return {
        "lot_sizes": dict(asset_meta.lot_sizes),
        "tick_sizes": dict(asset_meta.tick_sizes),
    }


class _WarmupBrokerProxy:
    """Null broker that silently rejects orders during warmup.

    Prevents strategies from placing real orders while processing
    historical warmup bars with stale data.
    """

    def submit_order(self, order):
        order.status = OrderStatus.REJECTED
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        return True

    def cancel_all(self, symbol: str = "") -> int:
        return 0

    def get_position(self, symbol: str):
        return None

    def get_positions(self) -> list:
        return []

    def get_open_orders(self, symbol: str = "") -> list:
        return []

    def get_equity(self) -> float:
        return 0.0

    def get_fills_since(self, since) -> list:
        return []

    def get_portfolio_snapshot(self, symbol: str, direction: Side) -> None:
        return None


class _StrategySlot:
    """Internal: holds one strategy's runtime state."""

    def __init__(
        self,
        strategy_id: str,
        strategy: Any,
        ctx: StrategyContext,
        bars: MultiTimeFrameBars,
        subscribed_tfs: set[TimeFrame],
        primary_tf: TimeFrame,
    ) -> None:
        self.strategy_id = strategy_id
        self.strategy = strategy
        self.ctx = ctx
        self.bars = bars
        self.subscribed_tfs = subscribed_tfs
        self.primary_tf = primary_tf
        self.runtime = StrategySlotRuntime(
            strategy=strategy,
            ctx=ctx,
            broker=ctx.broker,
            bars=bars,
            events=ctx.events,
            primary_timeframe=primary_tf,
            strategy_id=strategy_id,
        )


class LiveEngine:
    """Async polling engine for live/paper trading.

    Concurrent tasks:
    - _poll_candles_loop: poll for new bars, dispatch to strategies
    - _poll_fills_loop: poll for new fills, route to strategies
    - _equity_snapshot_loop: periodic equity recording
    - _daily_reset_loop: reset daily P&L at UTC midnight
    - _health_check_loop: heartbeat + stale data detection
    """

    def __init__(self, config: LiveConfig) -> None:
        self._config = config
        self._running = False
        self._slots: list[_StrategySlot] = []
        self._broker: HyperliquidBroker | None = None
        self._coordinator: StrategyCoordinator | None = None
        self._manager: PortfolioManager | None = None
        self._feed: LiveFeed | None = None
        self._health = HealthMonitor()
        self._persistent = PersistentState(config.state_dir)
        self._oms = OmsStore(config.state_dir)
        self._lifecycle = PositionLifecycleLedger()
        self._last_fill_check = self._load_fill_watermark() or datetime.now(timezone.utc)
        self._tracked_positions: dict[str, dict] = {}  # sym → tracked entry data
        self._strategy_dispatched_fill_ids: set[str] = set()
        self._coordinator_applied_fill_ids: set[str] = set()
        self._lifecycle_applied_fill_ids: set[str] = set()
        self._lifecycle_closed_trades_by_fill_id: dict[str, Trade | None] = {}
        self._tracked_fill_ids: set[str] = set()
        self._emitted_lifecycle_trade_ids: set[str] = set()
        self._finalized_fill_ids: set[str] = set()
        self._pending_missed: dict[str, Any] = {}
        self._last_funnels: dict[str, dict] = {}  # strategy_id → last funnel dict
        self._report_builder = HealthReportBuilder()

        # Instrumentation
        self._emitter = EventEmitter()
        self._emitter.add_sink(JsonlSink(config.state_dir))
        self._daily_aggregator = DailyAggregator(bot_id=getattr(config, "bot_id", ""))
        self._emitter.add_sink(self._daily_aggregator)  # aggregator receives all events
        self._sidecar: SidecarForwarder | None = None

        # PostgreSQL sink (optional — wired as additional Sink for trades/daily/health)
        self._pg_sink = None
        if config.postgres_dsn:
            try:
                from crypto_trader.instrumentation.postgres_sink import PostgresSink
                self._pg_sink = PostgresSink(config.postgres_dsn)
                self._emitter.add_sink(self._pg_sink)
                log.info("engine.postgres_sink_enabled")
            except Exception:
                log.exception("engine.postgres_sink_init_failed")

    async def start(self) -> None:
        """Initialize all components."""
        log.info("engine.starting", testnet=self._config.is_testnet)

        asset_meta_kwargs = _asset_meta_broker_kwargs(self._config.asset_meta_path)

        # Create broker
        self._broker = HyperliquidBroker(
            wallet_address=self._config.wallet_address,
            private_key=self._config.private_key,
            is_testnet=self._config.is_testnet,
            max_slippage_pct=self._config.max_slippage_pct,
            rate_limit_per_sec=self._config.rate_limit_per_sec,
            **asset_meta_kwargs,
        )

        # Load portfolio config
        portfolio_config = self._load_portfolio_config()

        # Create portfolio management
        state = PortfolioState(
            equity=self._broker.get_equity(),
            peak_equity=self._broker.get_equity(),
        )

        # Try to restore from persistent state
        saved_state = self._persistent.load_portfolio_state()
        if saved_state:
            state.peak_equity = max(state.equity, saved_state.get("peak_equity", state.equity))
            log.info("engine.state_restored", peak_equity=state.peak_equity)

        self._manager = PortfolioManager(config=portfolio_config, state=state)
        self._coordinator = StrategyCoordinator(broker=self._broker, manager=self._manager)

        # Create strategies
        strategy_tfs: dict[str, list[TimeFrame]] = {}

        for strategy_id, config_path in self._config.strategy_configs.items():
            alloc = portfolio_config.get_strategy(strategy_id)
            if alloc is None or not alloc.enabled:
                log.info("engine.strategy_skipped", strategy=strategy_id)
                continue

            strategy_config = self._load_strategy_config(strategy_id, config_path)
            strategy_config.symbols = self._config.symbols

            bot_id = getattr(self._config, "bot_id", "")
            strategy, feed_tfs, primary_tf = _create_strategy(strategy_id, strategy_config, bot_id=bot_id)
            strategy_tfs[strategy_id] = feed_tfs

            clock = WallClock()
            events = EventBus()
            events.subscribe(CanonicalRuntimeEvent, self._record_canonical_event)
            bars = MultiTimeFrameBars()
            execution_gateway = ExecutionGateway(
                adapter=HyperliquidExecutionAdapter(self._broker, strategy_id=strategy_id),
                broker=self._broker,
                events=events,
                oms_store=self._oms,
                immediate_fill_sync=self._sync_fills_after_submit,
            )
            proxy = self._coordinator.get_proxy(strategy_id)
            proxy._broker = execution_gateway

            ctx = StrategyContext(
                broker=proxy,
                clock=clock,
                bars=bars,
                events=events,
                config=strategy_config,
            )

            self._slots.append(_StrategySlot(
                strategy_id=strategy_id,
                strategy=strategy,
                ctx=ctx,
                bars=bars,
                subscribed_tfs=set(feed_tfs),
                primary_tf=primary_tf,
            ))

        self._rehydrate_oms_orders()

        # Create feed
        from hyperliquid.info import Info
        info = Info(self._config.base_url, skip_ws=True)
        self._feed = LiveFeed(info, self._config.symbols, strategy_tfs)

        # Load warmup bars
        warmup_bars = self._feed.load_warmup_bars(info, _WARMUP_COUNTS)

        # Init strategies with real broker (strategies may check initial state)
        for slot in self._slots:
            slot.strategy.on_init(slot.ctx)

        # Swap to warmup proxy — silently rejects all orders during warmup
        warmup_proxy = _WarmupBrokerProxy()
        real_brokers: list[Any] = []
        for slot in self._slots:
            real_brokers.append(slot.ctx.broker)
            slot.ctx.broker = warmup_proxy

        warmup_measurement_start = None
        if warmup_bars:
            warmup_measurement_start = max(
                bar.timestamp for bar in warmup_bars
            ) + timedelta(microseconds=1)

        original_start_dates: list[tuple[bool, Any]] = []
        for slot in self._slots:
            had_start_date = hasattr(slot.ctx.config, "start_date")
            original_start_dates.append((had_start_date, getattr(slot.ctx.config, "start_date", None)))
            if warmup_measurement_start is not None:
                setattr(slot.ctx.config, "start_date", warmup_measurement_start)

        # Feed warmup bars (orders silently rejected, no emitter wired)
        log.info("engine.warmup_start", bars=len(warmup_bars))
        for bar in warmup_bars:
            for slot in self._slots:
                if bar.timeframe in slot.subscribed_tfs and bar.symbol in slot.strategy.symbols:
                    slot.bars.append(bar)
                    slot.strategy.on_bar(bar, slot.ctx)
        log.info("engine.warmup_complete")

        # Restore real brokers after warmup
        for slot, real_broker, (had_start_date, original_start_date) in zip(
            self._slots,
            real_brokers,
            original_start_dates,
        ):
            slot.ctx.broker = real_broker
            if had_start_date:
                setattr(slot.ctx.config, "start_date", original_start_date)
            elif hasattr(slot.ctx.config, "start_date"):
                delattr(slot.ctx.config, "start_date")

        # Discard warmup-only instrumentation before wiring the live emitter.
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is None:
                continue
            collector.flush_missed()
            collector.pipeline.snapshot_and_reset()

        self._restore_strategy_snapshots()
        self._restore_lifecycle()

        # Wire instrumentation AFTER warmup (no stale telemetry)
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is not None:
                collector.emitter = self._emitter

        # Initial reconciliation — compare portfolio state expectations with exchange
        reconciler = PositionReconciler()
        actual = self._broker.get_positions()
        # On fresh start, no positions expected; on restart, portfolio state has open_risks
        expected: dict[str, Position | None] = {}
        for risk in self._manager.state.open_risks:
            expected[risk.symbol] = Position(
                symbol=risk.symbol,
                direction=risk.direction,
                qty=0.0,  # qty unknown from risk tracking; direction check is key
                avg_entry=0.0,
            )
        # Also mark symbols with no expected position
        for sym in self._config.symbols:
            if sym not in expected:
                expected[sym] = None
        discrepancies = reconciler.reconcile(expected, actual)
        if discrepancies:
            log.warning("engine.init_discrepancies", count=len(discrepancies))
            self._manager.entries_blocked_reason = "live OMS/exchange reconciliation unresolved"
            for discrepancy in discrepancies:
                self._oms.record_discrepancy(
                    kind=getattr(discrepancy, "kind", type(discrepancy).__name__),
                    description=str(discrepancy),
                    symbol=getattr(discrepancy, "symbol", ""),
                    metadata={
                        key: str(value)
                        for key, value in getattr(discrepancy, "__dict__", {}).items()
                    },
                )
        open_orders = self._sync_open_orders_to_oms()
        self._seed_ttl_trackers_from_open_orders(open_orders)

        # Start sidecar forwarder if relay is configured
        relay_url = getattr(self._config, "relay_url", "")
        relay_secret = getattr(self._config, "relay_secret", "")
        bot_id = getattr(self._config, "bot_id", "")
        if relay_url and relay_secret and bot_id:
            self._sidecar = SidecarForwarder(
                state_dir=self._config.state_dir,
                relay_url=relay_url,
                bot_id=bot_id,
                shared_secret=relay_secret,
            )
            self._sidecar.start()

        self._running = True
        log.info("engine.started", strategies=[s.strategy_id for s in self._slots])

    async def run(self) -> None:
        """Run the engine loop with concurrent tasks."""
        if not self._running:
            await self.start()

        tasks = [
            asyncio.create_task(self._poll_candles_loop()),
            asyncio.create_task(self._poll_fills_loop()),
            asyncio.create_task(self._equity_snapshot_loop()),
            asyncio.create_task(self._daily_reset_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._funnel_report_loop()),
            asyncio.create_task(self._health_report_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("engine.cancelled")
        except Exception:
            log.exception("engine.fatal_error")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info("engine.shutting_down")

        for slot in self._slots:
            try:
                slot.strategy.on_shutdown(slot.ctx)
            except Exception:
                log.exception("engine.shutdown_error", strategy=slot.strategy_id)

        # Stop sidecar forwarder
        if self._sidecar is not None:
            self._sidecar.stop()

        # Close PostgreSQL connection pool
        if self._pg_sink is not None:
            self._pg_sink.close()

        # Persist final state before closing the durable store.
        try:
            if self._manager:
                self._persistent.save_portfolio_state(self._manager.state.to_dict())
            self._persist_strategy_snapshots()
            self._persist_lifecycle()
        finally:
            self._oms.close()

        log.info("engine.shutdown_complete")

    # -------------------------------------------------------------------
    # Polling loops
    # -------------------------------------------------------------------

    async def _poll_candles_loop(self) -> None:
        """Poll for new bars at configured interval."""
        while self._running:
            try:
                self._health.on_poll()
                poll_events = getattr(self._feed, "poll_market_events", None)
                bars = poll_events() if callable(poll_events) else self._feed.poll()

                for bar in bars:
                    self._dispatch_bar(bar)

            except Exception:
                self._health.on_error("candle_poll")
                delay = self._health.get_backoff_delay()
                await asyncio.sleep(delay)
                continue

            await asyncio.sleep(self._config.poll_interval_sec)

    async def _poll_fills_loop(self) -> None:
        """Poll for new fills at configured interval."""
        while self._running:
            try:
                self._poll_and_process_fills()

            except Exception:
                self._health.on_error("fill_poll")

            await asyncio.sleep(self._config.fill_poll_interval_sec)

    def _sync_fills_after_submit(self, _order_id: str) -> None:
        """Immediately ingest fills that may have landed during submission."""
        try:
            self._poll_and_process_fills()
        except Exception:
            log.exception("engine.immediate_fill_sync_failed")
            self._health.on_error("immediate_fill_sync")

    def _poll_and_process_fills(self) -> list[Fill]:
        """Poll exchange fills with overlap and process them idempotently."""
        if self._broker is None:
            return []
        since = self._last_fill_check - timedelta(seconds=self._fill_query_overlap_sec())
        fills = self._broker.get_fills_since(since)
        result = self._process_fills(fills)

        latest_ts = max((fill.timestamp for fill in result.safe_watermark_fills), default=None)
        if latest_ts is not None:
            self._last_fill_check = max(self._last_fill_check, latest_ts)
        oms = getattr(self, "_oms", None)
        if oms is not None:
            oms.set_watermark("fills_since", self._last_fill_check.isoformat())
            oms.set_watermark("fills_last_poll_at", datetime.now(timezone.utc).isoformat())
        return result.processed

    def _process_fills(self, fills: list[Fill]) -> _FillProcessingResult:
        processed_fills: list[Fill] = []
        duplicate_fills: list[Fill] = []
        unresolved_fills: list[Fill] = []
        safe_watermark_fills: list[Fill] = []
        ledger_closed_symbols: set[str] = set()

        for fill in fills:
            fill_id = fill_identity(fill)
            if self._is_processed_oms_fill_id(fill_id):
                duplicate_fills.append(fill)
                safe_watermark_fills.append(fill)
                continue

            self._record_oms_fill_received(fill_id, fill, "")
            strategy_id, fill = self._resolve_fill_owner(fill)

            if self._is_processed_oms_fill_id(fill_id):
                duplicate_fills.append(fill)
                safe_watermark_fills.append(fill)
                continue

            self._record_oms_fill_received(fill_id, fill, strategy_id or "")
            if not strategy_id:
                log.warning("engine.unattributed_fill", order_id=fill.order_id)
                unresolved_fills.append(fill)
                self._mark_oms_fill_unresolved(
                    fill_id,
                    strategy_id="",
                    reason="unattributed_fill",
                )
                self._record_fill_discrepancy(
                    fill,
                    fill_id=fill_id,
                    kind="unattributed_fill",
                    description="Exchange fill could not be matched to a strategy owner.",
                )
                continue

            self._register_resolved_fill_order_ids(fill, strategy_id)
            slot = self._find_slot(strategy_id)
            if slot is None:
                unresolved_fills.append(fill)
                self._mark_oms_fill_unresolved(
                    fill_id,
                    strategy_id=strategy_id,
                    reason="missing_strategy_slot_fill",
                )
                self._record_fill_discrepancy(
                    fill,
                    fill_id=fill_id,
                    kind="missing_strategy_slot_fill",
                    description="Exchange fill owner was resolved but no live strategy slot exists.",
                    strategy_id=strategy_id,
                )
                continue

            self._clear_ttl_tracking_for_fill(slot, fill)
            try:
                status = self._oms_fill_status(fill_id)
                if status not in FILL_STRATEGY_DISPATCHED_STATUSES:
                    self._apply_strategy_fill_phase(fill_id, slot, fill, strategy_id)
                    status = self._oms_fill_status(fill_id)

                if status not in FILL_COORDINATOR_APPLIED_STATUSES:
                    strategy_id = self._apply_coordinator_fill_phase(
                        fill_id,
                        fill,
                        strategy_id,
                    )
                    status = self._oms_fill_status(fill_id)

                closed_trade = self._closed_trade_for_fill(fill_id)
                if status not in FILL_LIFECYCLE_APPLIED_STATUSES:
                    closed_trade = self._apply_lifecycle_fill_phase(
                        fill_id,
                        fill,
                        strategy_id,
                    )
                    status = self._oms_fill_status(fill_id)

                if status not in FILL_FINALIZED_STATUSES:
                    self._apply_finalization_fill_phase(
                        fill_id,
                        fill,
                        strategy_id,
                        closed_trade,
                        ledger_closed_symbols,
                    )

                self._mark_oms_fill_processed(fill_id, strategy_id=strategy_id)
                self._record_fill_telemetry(fill_id, slot, fill)
            except Exception as exc:
                unresolved_fills.append(fill)
                self._handle_fill_processing_exception(
                    fill_id,
                    fill,
                    strategy_id=strategy_id,
                    error=exc,
                )
                continue

            processed_fills.append(fill)
            safe_watermark_fills.append(fill)
            log.info(
                "engine.fill",
                strategy=strategy_id,
                symbol=fill.symbol,
                side=fill.side.value,
                qty=fill.qty,
                price=fill.fill_price,
            )

        fallback_fills = [
            fill for fill in processed_fills
            if fill.symbol not in ledger_closed_symbols
        ]
        if fallback_fills:
            self._detect_position_closures(fallback_fills)
        return _FillProcessingResult(
            processed=processed_fills,
            duplicates=duplicate_fills,
            unresolved=unresolved_fills,
            safe_watermark_fills=safe_watermark_fills,
        )

    def _apply_strategy_fill_phase(
        self,
        fill_id: str,
        slot: _StrategySlot,
        fill: Fill,
        strategy_id: str,
    ) -> None:
        dispatched = self._phase_fill_ids("_strategy_dispatched_fill_ids")
        if fill_id not in dispatched:
            slot.runtime.dispatch_fill(fill, notify_callback=False)
            dispatched.add(fill_id)
        self._mark_oms_fill_strategy_dispatched(fill_id, strategy_id=strategy_id)

    def _apply_coordinator_fill_phase(
        self,
        fill_id: str,
        fill: Fill,
        strategy_id: str,
    ) -> str:
        applied = self._phase_fill_ids("_coordinator_applied_fill_ids")
        applied_strategy_id = strategy_id
        if fill_id not in applied:
            if self._coordinator is not None:
                resolved_id = self._coordinator.on_fill(fill)
                if isinstance(resolved_id, str) and resolved_id:
                    applied_strategy_id = resolved_id
                elif isinstance(self._coordinator, StrategyCoordinator) and fill.tag == "entry":
                    self._record_fill_discrepancy(
                        fill,
                        fill_id=fill_id,
                        kind="coordinator_fill_unapplied",
                        description="Owned entry fill could not be applied by the strategy coordinator.",
                        strategy_id=strategy_id,
                    )
                    raise RuntimeError("coordinator could not apply owned entry fill")
            applied.add(fill_id)
        self._mark_oms_fill_coordinator_applied(fill_id, strategy_id=applied_strategy_id)
        return applied_strategy_id

    def _apply_lifecycle_fill_phase(
        self,
        fill_id: str,
        fill: Fill,
        strategy_id: str,
    ) -> Trade | None:
        applied = self._phase_fill_ids("_lifecycle_applied_fill_ids")
        closed_trades = self._lifecycle_closed_trades()
        if fill_id not in applied:
            lifecycle = getattr(self, "_lifecycle", None)
            closed_trades[fill_id] = (
                lifecycle.apply_fill(strategy_id, fill)
                if lifecycle is not None
                else None
            )
            applied.add(fill_id)
        closed_trade = closed_trades.get(fill_id)
        persist_phase = getattr(getattr(self, "_oms", None), "persist_lifecycle_phase", None)
        if callable(persist_phase):
            persist_phase(
                fill_id,
                self._lifecycle_snapshot(),
                strategy_id=strategy_id,
                closed_trade_event=self._closed_trade_event(fill_id, closed_trade),
            )
        else:
            self._persist_lifecycle()
            self._persist_lifecycle_closed_trade(fill_id, closed_trade)
            self._mark_oms_fill_lifecycle_applied(fill_id, strategy_id=strategy_id)
        return closed_trade

    def _apply_finalization_fill_phase(
        self,
        fill_id: str,
        fill: Fill,
        strategy_id: str,
        closed_trade: Trade | None,
        ledger_closed_symbols: set[str],
    ) -> None:
        finalized = self._phase_fill_ids("_finalized_fill_ids")
        if fill_id not in finalized:
            self._track_entry_fill_once(fill_id, strategy_id, fill)
            if closed_trade is not None:
                self._emit_lifecycle_trade_once(fill_id, strategy_id, closed_trade)
                ledger_closed_symbols.add(fill.symbol)
            finalized.add(fill_id)
        elif closed_trade is not None and fill_id in self._phase_fill_ids("_emitted_lifecycle_trade_ids"):
            ledger_closed_symbols.add(fill.symbol)
        self._mark_oms_fill_finalized(fill_id, strategy_id=strategy_id)

    def _track_entry_fill_once(self, fill_id: str, strategy_id: str, fill: Fill) -> None:
        if fill.tag != "entry":
            return
        tracked = self._phase_fill_ids("_tracked_fill_ids")
        if fill_id in tracked:
            return
        self._track_entry_fill(strategy_id, fill)
        tracked.add(fill_id)

    def _emit_lifecycle_trade_once(
        self,
        fill_id: str,
        strategy_id: str,
        trade: Trade,
    ) -> None:
        emitted = self._phase_fill_ids("_emitted_lifecycle_trade_ids")
        if fill_id not in emitted:
            self._emit_lifecycle_trade(strategy_id, trade)
            emitted.add(fill_id)
        self._record_lifecycle_trade_position(strategy_id, trade)

    def _record_fill_telemetry(self, fill_id: str, slot: _StrategySlot, fill: Fill) -> None:
        collector = getattr(slot.strategy, "_collector", None)
        if collector is None:
            return
        try:
            collector.pipeline.record_fill(fill.symbol)
        except Exception:
            log.exception("engine.fill_telemetry_failed", fill_id=fill_id)

    def _handle_fill_processing_exception(
        self,
        fill_id: str,
        fill: Fill,
        *,
        strategy_id: str,
        error: Exception,
    ) -> None:
        if self._fill_phase_started(fill_id):
            self._record_oms_fill_processing_error(
                fill_id,
                strategy_id=strategy_id,
                error=str(error),
            )
        else:
            self._mark_oms_fill_processing_failed(
                fill_id,
                strategy_id=strategy_id,
                error=str(error),
            )
        self._record_fill_discrepancy(
            fill,
            fill_id=fill_id,
            kind="fill_processing_failed",
            description="Owned exchange fill processing failed before it was safely consumed.",
            strategy_id=strategy_id,
            metadata={"error": str(error)},
        )
        health = getattr(self, "_health", None)
        if health is not None:
            health.on_error("fill_processing")
        log.exception(
            "engine.fill_processing_failed",
            strategy=strategy_id,
            order_id=fill.order_id,
            exchange_order_id=fill.exchange_order_id,
            fill_id=fill_id,
        )

    def _fill_phase_started(self, fill_id: str) -> bool:
        status = self._oms_fill_status(fill_id)
        if status in FILL_STRATEGY_DISPATCHED_STATUSES:
            return True
        phase_attrs = (
            "_strategy_dispatched_fill_ids",
            "_coordinator_applied_fill_ids",
            "_lifecycle_applied_fill_ids",
            "_tracked_fill_ids",
            "_emitted_lifecycle_trade_ids",
            "_finalized_fill_ids",
        )
        return any(fill_id in self._phase_fill_ids(attr) for attr in phase_attrs)

    def _phase_fill_ids(self, attr: str) -> set[str]:
        values = getattr(self, attr, None)
        if not isinstance(values, set):
            values = set()
            setattr(self, attr, values)
        return values

    def _lifecycle_closed_trades(self) -> dict[str, Trade | None]:
        values = getattr(self, "_lifecycle_closed_trades_by_fill_id", None)
        if not isinstance(values, dict):
            values = {}
            setattr(self, "_lifecycle_closed_trades_by_fill_id", values)
        return values

    def _closed_trade_for_fill(self, fill_id: str) -> Trade | None:
        closed_trades = self._lifecycle_closed_trades()
        if fill_id not in closed_trades:
            closed_trades[fill_id] = self._load_lifecycle_closed_trade(fill_id)
        return closed_trades.get(fill_id)

    def _persist_lifecycle_closed_trade(self, fill_id: str, trade: Trade | None) -> None:
        event = self._closed_trade_event(fill_id, trade)
        if event is None:
            return
        append_fn = getattr(getattr(self, "_oms", None), "append_event", None)
        if callable(append_fn):
            append_fn("fill_lifecycle_closed_trade", event[0], event[1])

    def _closed_trade_event(
        self,
        fill_id: str,
        trade: Trade | None,
    ) -> tuple[datetime, dict[str, Any]] | None:
        if trade is None:
            return None
        return trade.exit_time, {"fill_id": fill_id, "trade": self._trade_payload(trade)}

    def _load_lifecycle_closed_trade(self, fill_id: str) -> Trade | None:
        list_fn = getattr(getattr(self, "_oms", None), "list_events", None)
        if not callable(list_fn):
            return None
        for event in reversed(list_fn("fill_lifecycle_closed_trade")):
            payload = event.get("payload") or {}
            if payload.get("fill_id") == fill_id:
                return self._trade_from_payload(payload.get("trade") or {})
        return None

    def _trade_payload(self, trade: Trade) -> dict[str, Any]:
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": trade.direction.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "qty": trade.qty,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "pnl": trade.pnl,
            "r_multiple": trade.r_multiple,
            "commission": trade.commission,
            "bars_held": trade.bars_held,
            "setup_grade": trade.setup_grade.value if trade.setup_grade is not None else None,
            "exit_reason": trade.exit_reason,
            "confluences_used": trade.confluences_used,
            "confirmation_type": trade.confirmation_type,
            "entry_method": trade.entry_method,
            "funding_paid": trade.funding_paid,
            "mae_r": trade.mae_r,
            "mfe_r": trade.mfe_r,
            "realized_r_multiple": trade.realized_r_multiple,
            "signal_variant": trade.signal_variant,
        }

    def _trade_from_payload(self, data: dict[str, Any]) -> Trade:
        setup_grade = data.get("setup_grade")
        return Trade(
            trade_id=str(data["trade_id"]),
            symbol=str(data["symbol"]),
            direction=Side(data["direction"]),
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"]),
            qty=float(data["qty"]),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]),
            pnl=float(data["pnl"]),
            r_multiple=data.get("r_multiple"),
            commission=float(data["commission"]),
            bars_held=int(data.get("bars_held", 0)),
            setup_grade=SetupGrade(setup_grade) if setup_grade else None,
            exit_reason=str(data.get("exit_reason") or "exchange_fill"),
            confluences_used=data.get("confluences_used"),
            confirmation_type=data.get("confirmation_type"),
            entry_method=data.get("entry_method"),
            funding_paid=float(data.get("funding_paid", 0.0)),
            mae_r=data.get("mae_r"),
            mfe_r=data.get("mfe_r"),
            realized_r_multiple=data.get("realized_r_multiple"),
            signal_variant=data.get("signal_variant"),
        )

    def _register_resolved_fill_order_ids(self, fill: Fill, strategy_id: str) -> None:
        if self._coordinator is None:
            return
        register = getattr(self._coordinator, "register_order", None)
        if not callable(register):
            return
        order_ids = self._fill_order_ids(fill)
        broker = getattr(self, "_broker", None)
        local_to_oid = getattr(broker, "_local_to_oid", None)
        if isinstance(local_to_oid, dict):
            for order_id in list(order_ids):
                exchange_id = local_to_oid.get(order_id)
                if exchange_id:
                    order_ids.append(str(exchange_id))
        for order_id in dict.fromkeys(order_ids):
            register(order_id, strategy_id)

    def _resolve_fill_owner(self, fill: Fill) -> tuple[str | None, Fill]:
        """Resolve strategy ownership while keeping OMS client IDs canonical."""
        coordinator_owner = self._coordinator_fill_owner(fill)
        oms_owner, canonical_fill = self._oms_canonical_fill(fill, coordinator_owner)
        if oms_owner:
            return oms_owner, canonical_fill
        if coordinator_owner:
            return coordinator_owner, fill
        broker_owner = self._broker_fill_owner(fill)
        if broker_owner:
            return broker_owner, fill
        return None, fill

    def _fill_order_ids(self, fill: Fill) -> list[str]:
        """Return stable candidate order IDs from a fill without duplicates."""
        return list(dict.fromkeys(
            str(order_id)
            for order_id in (fill.order_id, fill.exchange_order_id)
            if order_id
        ))

    def _coordinator_fill_owner(self, fill: Fill) -> str | None:
        if self._coordinator is None:
            return None
        owner_fn = getattr(self._coordinator, "get_strategy_for_order", None)
        if owner_fn is None:
            return None
        for order_id in self._fill_order_ids(fill):
            owner = owner_fn(order_id)
            if isinstance(owner, str) and owner:
                return owner
        return None

    def _oms_canonical_fill(
        self,
        fill: Fill,
        coordinator_owner: str | None,
    ) -> tuple[str | None, Fill]:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return None, fill

        for order_id in self._fill_order_ids(fill):
            record = oms.get_order(order_id)
            if not record:
                continue
            strategy_id = str(record.get("strategy_id") or "")
            if not strategy_id:
                continue
            client_id = str(record.get("client_order_id") or fill.order_id or "")
            exchange_id = str(record.get("exchange_order_id") or fill.exchange_order_id or "")

            if coordinator_owner and coordinator_owner != strategy_id:
                log.warning(
                    "engine.fill_owner_mismatch",
                    order_id=fill.order_id,
                    exchange_order_id=fill.exchange_order_id,
                    coordinator_owner=coordinator_owner,
                    oms_owner=strategy_id,
                )
                self._record_fill_discrepancy(
                    fill,
                    kind="fill_owner_mismatch",
                    description="Coordinator and durable OMS disagree on fill ownership; OMS owner was used.",
                    strategy_id=strategy_id,
                    metadata={"coordinator_owner": coordinator_owner, "oms_owner": strategy_id},
                )

            if self._coordinator is not None:
                if client_id:
                    self._coordinator.register_order(client_id, strategy_id)
                if exchange_id:
                    self._coordinator.register_order(exchange_id, strategy_id)

            if client_id and (
                client_id != fill.order_id
                or (exchange_id and not fill.exchange_order_id)
            ):
                fill = replace(
                    fill,
                    order_id=client_id,
                    exchange_order_id=fill.exchange_order_id or exchange_id,
                )
            return strategy_id, fill

        return None, fill

    def _broker_fill_owner(self, fill: Fill) -> str | None:
        if self._broker is None:
            return None
        for order_id in self._fill_order_ids(fill):
            owner = self._broker.get_order_owner(order_id)
            if owner:
                return owner
        return None

    def _is_processed_oms_fill(self, fill: Fill) -> bool:
        return self._is_processed_oms_fill_id(fill_identity(fill))

    def _is_processed_oms_fill_id(self, fill_id: str) -> bool:
        oms = getattr(self, "_oms", None)
        is_processed = getattr(oms, "is_fill_processed", None)
        return bool(callable(is_processed) and is_processed(fill_id))

    def _oms_fill_status(self, fill_id: str) -> str | None:
        oms = getattr(self, "_oms", None)
        get_status = getattr(oms, "get_fill_status", None)
        if not callable(get_status):
            return None
        return get_status(fill_id)

    def _record_fill_discrepancy(
        self,
        fill: Fill,
        *,
        fill_id: str | None = None,
        kind: str,
        description: str,
        strategy_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return
        fill_id = fill_id or fill_identity(fill)
        list_fn = getattr(oms, "list_unresolved_discrepancies", None)
        if callable(list_fn):
            for discrepancy in list_fn():
                if (
                    discrepancy.get("kind") == kind
                    and (discrepancy.get("metadata") or {}).get("fill_id") == fill_id
                ):
                    return
        payload = {
            "fill_id": fill_id,
            "order_id": fill.order_id,
            "exchange_order_id": fill.exchange_order_id,
            "exchange_fill_id": fill.exchange_fill_id,
            "timestamp": fill.timestamp.isoformat(),
            "tag": fill.tag,
            **(metadata or {}),
        }
        record_fn = getattr(oms, "record_discrepancy", None)
        if callable(record_fn):
            record_fn(
                kind=kind,
                description=description,
                symbol=fill.symbol,
                strategy_id=strategy_id,
                metadata=payload,
            )

    def _track_entry_fill(self, strategy_id: str, fill: Fill) -> None:
        if fill.tag != "entry":
            return
        tracked = self._tracked_positions.get(fill.symbol)
        if (
            tracked is not None
            and tracked.get("strategy_id") == strategy_id
            and tracked.get("direction") == fill.side
        ):
            prev_qty = float(tracked.get("qty", 0.0))
            total_qty = prev_qty + fill.qty
            if total_qty > 0:
                tracked["entry_price"] = (
                    (tracked.get("entry_price", 0.0) * prev_qty)
                    + (fill.fill_price * fill.qty)
                ) / total_qty
            tracked["qty"] = total_qty
            tracked["entry_time"] = min(tracked["entry_time"], fill.timestamp)
            tracked["entry_commission"] = tracked.get("entry_commission", 0.0) + fill.commission
            return

        self._tracked_positions[fill.symbol] = {
            "strategy_id": strategy_id,
            "direction": fill.side,
            "entry_price": fill.fill_price,
            "entry_time": fill.timestamp,
            "qty": fill.qty,
            "entry_commission": fill.commission,
        }

    async def _equity_snapshot_loop(self) -> None:
        """Record equity snapshots periodically."""
        while self._running:
            await asyncio.sleep(self._config.equity_snapshot_interval_sec)
            try:
                equity = self._broker.get_equity()
                self._manager.update_equity(equity)
                self._persistent.append_equity_snapshot(equity)
                self._daily_aggregator.record_equity(datetime.now(timezone.utc), equity)
                self._persistent.save_portfolio_state(self._manager.state.to_dict())

                # Write equity + positions to PostgreSQL
                if self._pg_sink is not None:
                    self._pg_sink.write_equity(equity, datetime.now(timezone.utc))
                    self._pg_sink.upsert_positions(self._build_positions_snapshot())
            except Exception:
                self._health.on_error("equity_snapshot")

    async def _daily_reset_loop(self) -> None:
        """Reset daily P&L counters at UTC midnight."""
        from datetime import timedelta

        while self._running:
            now = datetime.now(timezone.utc)
            # Calculate seconds until next midnight
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if midnight <= now:
                midnight += timedelta(days=1)
            wait_secs = (midnight - now).total_seconds()
            await asyncio.sleep(min(wait_secs, 3600))  # check at least hourly

            today = datetime.now(timezone.utc).date()
            self._manager.maybe_reset_daily(today)

            # Compute and emit daily snapshot
            try:
                yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
                snapshot = self._daily_aggregator.compute_snapshot(yesterday)
                self._emitter.emit_daily(snapshot)
            except Exception:
                log.exception("engine.daily_snapshot_error")

    async def _health_check_loop(self) -> None:
        """Periodic health check, heartbeat, stale feed detection, reconnect check."""
        while self._running:
            await asyncio.sleep(self._config.health_check_interval_sec)
            self._health.heartbeat()

            if self._health.is_stale():
                log.warning("engine.stale_data")

            # Per-(sym, tf) stale feed detection
            stale_feeds = self._health.get_stale_feeds(_TF_INTERVALS)
            for sym, tf, elapsed in stale_feeds:
                log.error("engine.stale_feed", symbol=sym, tf=tf, elapsed_sec=round(elapsed))

            # Reconnect check
            if self._health.should_reconnect():
                status = self._health.get_status()
                log.error(
                    "engine.reconnect_needed",
                    consecutive_errors=status["consecutive_errors"],
                )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _dispatch_bar(self, bar: Bar | MarketEvent) -> None:
        """Route a bar to subscribing strategies."""
        visible_bar = bar.to_bar() if isinstance(bar, MarketEvent) else bar
        self._health.on_bar_received(visible_bar.symbol, visible_bar.timeframe.value)
        for slot in self._slots:
            if (
                visible_bar.timeframe in slot.subscribed_tfs
                and visible_bar.symbol in slot.strategy.symbols
            ):
                if visible_bar.timeframe == slot.primary_tf:
                    expire_fn = getattr(slot.ctx.broker, "expire_ttl_orders_for_bar", None)
                    if callable(expire_fn):
                        expire_fn(visible_bar)
                slot.runtime.process_bar(bar, process_broker=False, advance_clock=False)
        self._drain_and_backfill_missed()

    def _drain_and_backfill_missed(self) -> None:
        pending = getattr(self, "_pending_missed", {})
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is None:
                continue
            for event in collector.flush_missed():
                pending[event.metadata.event_id] = event

        if not pending:
            self._pending_missed = pending
            return

        bars_by_symbol = self._bars_by_symbol_for_backfill()
        if not bars_by_symbol:
            self._pending_missed = pending
            return

        for event_id, event in list(pending.items()):
            before = (
                event.outcome_1h,
                event.outcome_4h,
                event.outcome_24h,
                event.backfill_status,
            )
            MissedOpportunityBackfiller.backfill_from_bars([event], bars_by_symbol)
            after = (
                event.outcome_1h,
                event.outcome_4h,
                event.outcome_24h,
                event.backfill_status,
            )
            if after != before:
                self._emitter.emit_missed(event)
            if event.backfill_status == "complete":
                pending.pop(event_id, None)

        self._pending_missed = pending

    def _bars_by_symbol_for_backfill(self) -> dict[str, list[Bar]]:
        bars_by_symbol: dict[str, tuple[int, list[Bar]]] = {}
        for slot in self._slots:
            for tf in slot.subscribed_tfs:
                for sym in slot.strategy.symbols:
                    bars = slot.bars.get(sym, tf)
                    if not bars:
                        continue
                    current = bars_by_symbol.get(sym)
                    if current is None or tf.minutes < current[0]:
                        bars_by_symbol[sym] = (tf.minutes, bars)
        return {sym: bars for sym, (_, bars) in bars_by_symbol.items()}

    def _derive_bars_held(
        self,
        strategy_id: str,
        entry_time: datetime,
        exit_time: datetime,
    ) -> int:
        slot = self._find_slot(strategy_id)
        if slot is None:
            return 0
        primary_tf = slot.primary_tf if isinstance(slot.primary_tf, TimeFrame) else TimeFrame.M15
        interval_sec = _TF_INTERVALS.get(primary_tf.value)
        if not interval_sec or exit_time <= entry_time:
            return 0
        elapsed_sec = (exit_time - entry_time).total_seconds()
        return max(1, int((elapsed_sec + interval_sec - 1) // interval_sec))

    def _emit_lifecycle_trade(self, strategy_id: str, trade: Trade) -> None:
        """Emit a ledger-built live trade through the same close-event path."""
        slot = self._find_slot(strategy_id)
        if slot is None:
            return

        trade.bars_held = self._derive_bars_held(
            strategy_id,
            trade.entry_time,
            trade.exit_time,
        )
        slot.ctx.events.emit(PositionClosedEvent(
            timestamp=trade.exit_time,
            trade=trade,
        ))
        pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
        if self._coordinator is not None:
            self._coordinator.on_trade_closed(strategy_id, trade.symbol, pnl_R)
        self._tracked_positions.pop(trade.symbol, None)

    def _record_lifecycle_trade_position(self, strategy_id: str, trade: Trade) -> None:
        oms = getattr(self, "_oms", None)
        if oms is not None:
            oms.upsert_position(
                position_instance_id=f"{strategy_id}:{trade.symbol}:{trade.entry_time.isoformat()}",
                strategy_id=strategy_id,
                symbol=trade.symbol,
                direction=trade.direction.value,
                qty=0.0,
                avg_entry=trade.entry_price,
                status="CLOSED",
                metadata={"trade_id": trade.trade_id},
            )

    def _detect_position_closures(self, recent_fills: list[Fill]) -> None:
        """Check tracked positions against exchange; emit PositionClosedEvent for closures."""
        current_positions = {p.symbol: p for p in self._broker.get_positions()}

        for sym, tracked in list(self._tracked_positions.items()):
            if sym in current_positions and current_positions[sym].qty != 0:
                continue  # Still open

            # Position closed — find the exit fill
            exit_fill = None
            for fill in reversed(recent_fills):
                if fill.symbol == sym and fill.tag != "entry":
                    exit_fill = fill
                    break
            if exit_fill is None:
                continue

            # Build Trade from tracked data + strategy _position_meta
            slot = self._find_slot(tracked["strategy_id"])
            if slot is None:
                continue

            meta = getattr(slot.strategy, "_position_meta", {}).get(sym)
            entry_price = meta.entry_price if meta and hasattr(meta, "entry_price") else tracked["entry_price"]
            direction = tracked["direction"]
            qty = tracked["qty"]
            stop_distance = meta.stop_distance if meta and hasattr(meta, "stop_distance") else 0.0

            if direction == Side.LONG:
                pnl = (exit_fill.fill_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_fill.fill_price) * qty

            commission = tracked.get("entry_commission", 0.0) + exit_fill.commission
            bars_held = self._derive_bars_held(
                tracked["strategy_id"],
                tracked["entry_time"],
                exit_fill.timestamp,
            )

            trade = Trade(
                trade_id=f"live_{sym}_{exit_fill.timestamp.strftime('%Y%m%d_%H%M%S')}",
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_fill.fill_price,
                qty=qty,
                entry_time=tracked["entry_time"],
                exit_time=exit_fill.timestamp,
                pnl=pnl,
                r_multiple=None,
                commission=commission,
                bars_held=bars_held,
                setup_grade=None,
                exit_reason=exit_fill.tag or "exchange_fill",
                confluences_used=None,
                confirmation_type=None,
                entry_method=None,
                funding_paid=0.0,
                mae_r=None,
                mfe_r=None,
            )

            # Emit — synchronously fires strategy's _on_position_closed
            slot.ctx.events.emit(PositionClosedEvent(
                timestamp=exit_fill.timestamp, trade=trade,
            ))

            # Fire coordinator for portfolio heat release (AFTER strategy enrichment)
            pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
            self._coordinator.on_trade_closed(tracked["strategy_id"], sym, pnl_R)

            del self._tracked_positions[sym]
            log.info(
                "engine.position_closed",
                symbol=sym,
                strategy=tracked["strategy_id"],
                pnl=f"{trade.pnl:.2f}",
                r=f"{trade.r_multiple:.2f}" if trade.r_multiple else "N/A",
            )

    async def _funnel_report_loop(self) -> None:
        """Periodic pipeline funnel snapshots."""
        while self._running:
            await asyncio.sleep(self._funnel_report_interval())
            try:
                for slot in self._slots:
                    collector = getattr(slot.strategy, "_collector", None)
                    if collector is None:
                        continue
                    funnel = collector.pipeline.snapshot_and_reset()
                    assessment = PipelineTracker.assess(funnel)
                    funnel_dict = funnel.to_dict()

                    # Cache for health report (avoids double-reset)
                    self._last_funnels[slot.strategy_id] = funnel_dict

                    snapshot = PipelineFunnelSnapshot(
                        strategy_id=slot.strategy_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        period_start=funnel.period_start.isoformat(),
                        period_end=funnel.period_end.isoformat(),
                        funnel=funnel_dict,
                        assessment=assessment,
                    )
                    self._emitter.emit_funnel(snapshot)

                    if assessment in ("pipeline_broken", "stalled"):
                        log.error(
                            "engine.funnel_alert",
                            strategy=slot.strategy_id,
                            assessment=assessment,
                        )
            except Exception:
                log.exception("engine.funnel_report_error")
                self._health.on_error("funnel_report")

    async def _health_report_loop(self) -> None:
        """Periodic health report."""
        while self._running:
            await asyncio.sleep(self._health_report_interval())
            try:
                status = self._health.get_status()

                # Read last emitted funnel data (no reset — funnel_report_loop owns the reset)
                funnels: dict[str, dict] = {}
                for slot in self._slots:
                    collector = getattr(slot.strategy, "_collector", None)
                    if collector is not None:
                        funnels[slot.strategy_id] = self._last_funnels.get(
                            slot.strategy_id, {},
                        )

                # Collect positions
                positions = []
                if self._broker:
                    for p in self._broker.get_positions():
                        positions.append({
                            "symbol": p.symbol,
                            "direction": p.direction.value if p.direction else "unknown",
                            "qty": p.qty,
                        })

                # Portfolio state
                portfolio_state = {}
                if self._manager:
                    portfolio_state = {
                        "heat_R": sum(r.risk_R for r in self._manager.state.open_risks),
                        "heat_cap_R": self._manager.config.heat_cap_R,
                        "daily_pnl_R": self._manager.state.portfolio_daily_pnl_R,
                        "open_risk_count": len(self._manager.state.open_risks),
                    }

                stale_feeds = self._health.get_stale_feeds(_TF_INTERVALS)

                report = self._report_builder.build(
                    uptime_sec=status["uptime_sec"],
                    health_status=status,
                    stale_feeds=stale_feeds,
                    funnels=funnels,
                    positions=positions,
                    portfolio_state=portfolio_state,
                    tf_last_bar=self._health.get_tf_last_bar(),
                    now_mono=time.monotonic(),
                )

                report_payload = report.to_dict()
                report_payload["relay"] = self._relay_health_status()

                self._emitter.emit_health_report(HealthReportSnapshot(
                    timestamp=report.timestamp,
                    report=report_payload,
                ))

                if report.assessment == "critical":
                    log.error("engine.health_critical", alerts=len(report.alerts))

            except Exception:
                log.exception("engine.health_report_error")
                self._health.on_error("health_report")

    def _relay_health_status(self) -> dict:
        """Return relay/sidecar state for health report enrichment."""
        if self._sidecar is None:
            return {
                "enabled": False,
                "sidecar_running": False,
                "event_files": [],
            }

        try:
            status = self._sidecar.status()
        except Exception as exc:
            return {
                "enabled": True,
                "sidecar_running": False,
                "event_files": [],
                "status_error": str(exc),
            }

        return {
            "enabled": True,
            "sidecar_running": status.get("running", False),
            "event_files": status.get("event_files", []),
            "watermarks": status.get("watermarks", {}),
            "watermark_file": status.get("watermark_file"),
            "last_successful_send_at": status.get("last_successful_send_at"),
            "consecutive_send_failures": status.get("consecutive_send_failures", 0),
            "last_error": status.get("last_error"),
        }

    def _funnel_report_interval(self) -> float:
        return max(60.0, self._config.funnel_report_interval_sec)

    def _health_report_interval(self) -> float:
        return max(60.0, self._config.health_report_interval_sec)

    def _fill_query_overlap_sec(self) -> float:
        return max(0.0, float(getattr(self._config, "fill_query_overlap_sec", 300.0)))

    def _load_fill_watermark(self) -> datetime | None:
        raw = self._oms.get_watermark("fills_since")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            log.warning("engine.fill_watermark_invalid", value=raw)
            return None

    def _record_canonical_event(self, event: CanonicalRuntimeEvent) -> None:
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "stream": event.stream,
            "payload": event.payload,
        }
        path = self._config.state_dir / "parity_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        append_fn = getattr(self._oms, "append_event", None)
        if append_fn is not None:
            append_fn(event.stream, event.timestamp, event.payload)

    def _restore_strategy_snapshots(self) -> None:
        for slot in self._slots:
            snapshot = self._oms.get_strategy_snapshot(slot.strategy_id)
            restore_fn = getattr(slot.strategy, "restore_state", None)
            if snapshot is not None and restore_fn is not None:
                restore_fn(snapshot)

    def _persist_strategy_snapshots(self) -> None:
        for slot in self._slots:
            snapshot_fn = getattr(slot.strategy, "snapshot_state", None)
            if snapshot_fn is not None:
                self._oms.upsert_strategy_snapshot(slot.strategy_id, snapshot_fn())

    def _restore_lifecycle(self) -> None:
        restore_fn = getattr(self._lifecycle, "restore", None)
        if restore_fn is not None:
            restore_fn(self._oms.list_lifecycle_entries())

    def _persist_lifecycle(self) -> None:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return
        entries = self._lifecycle_snapshot()
        replace_fn = getattr(oms, "replace_lifecycle_entries", None)
        if replace_fn is not None:
            replace_fn(entries)
        else:
            for entry in entries:
                oms.upsert_lifecycle_entry(entry)

    def _lifecycle_snapshot(self) -> list[Any]:
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is None:
            return []
        snapshot_fn = getattr(lifecycle, "snapshot", None)
        if snapshot_fn is None:
            return []
        return list(snapshot_fn())

    def _rehydrate_oms_orders(self) -> None:
        """Restore in-memory order ownership from durable OMS records."""
        broker_orders = getattr(self._broker, "_orders", None)
        local_to_oid = getattr(self._broker, "_local_to_oid", None)
        oid_map = getattr(self._broker, "_oid_map", None)
        for record in self._oms.list_orders():
            strategy_id = record.get("strategy_id") or ""
            if not strategy_id:
                continue
            client_id = record.get("client_order_id") or ""
            exchange_id = record.get("exchange_order_id") or ""

            if client_id and exchange_id:
                if isinstance(local_to_oid, dict):
                    local_to_oid[str(client_id)] = str(exchange_id)
                if isinstance(oid_map, dict):
                    oid_map[str(exchange_id)] = str(client_id)

            if isinstance(broker_orders, dict) and client_id and client_id not in broker_orders:
                raw_metadata = dict(record.get("metadata") or {})
                nested_metadata = raw_metadata.get("metadata")
                metadata = {
                    **raw_metadata,
                    **(nested_metadata if isinstance(nested_metadata, dict) else {}),
                }
                metadata.setdefault("strategy_id", strategy_id)
                metadata.setdefault("client_order_id", client_id)
                ttl_bars = metadata.get("ttl_bars")
                if ttl_bars is not None:
                    try:
                        ttl_bars = int(ttl_bars)
                    except (TypeError, ValueError):
                        ttl_bars = None
                ttl_bars_alive = metadata.get("ttl_bars_alive", 0)
                try:
                    ttl_bars_alive = int(ttl_bars_alive)
                except (TypeError, ValueError):
                    ttl_bars_alive = 0
                broker_orders[str(client_id)] = Order(
                    order_id=str(client_id),
                    symbol=str(record.get("symbol") or ""),
                    side=_enum_or_default(Side, record.get("side"), Side.LONG),
                    order_type=_enum_or_default(
                        OrderType,
                        record.get("order_type") or metadata.get("order_type"),
                        OrderType.LIMIT,
                    ),
                    qty=0.0,
                    status=_enum_or_default(OrderStatus, record.get("status"), OrderStatus.WORKING),
                    tag=str(record.get("role") or metadata.get("tag") or ""),
                    ttl_bars=ttl_bars,
                    metadata=metadata,
                    _bars_alive=ttl_bars_alive,
                )

            if self._coordinator is not None and client_id:
                self._coordinator.register_order(client_id, strategy_id)
            if self._coordinator is not None and exchange_id:
                self._coordinator.register_order(exchange_id, strategy_id)

    def _sync_open_orders_to_oms(self) -> list[Order]:
        """Persist currently visible exchange orders into the OMS store."""
        if self._broker is None:
            return []
        local_to_oid = getattr(self._broker, "_local_to_oid", {})
        open_orders = self._broker.get_open_orders()
        for order in open_orders:
            strategy_id = ""
            if self._coordinator is not None:
                strategy_id = self._coordinator.get_strategy_for_order(order.order_id) or ""
            strategy_id = (
                strategy_id
                or str(order.metadata.get("strategy_id") or "")
                or (self._broker.get_order_owner(order.order_id) or "")
                or "unknown"
            )
            exchange_oid = str(local_to_oid.get(order.order_id, ""))
            self._oms.upsert_order(
                client_order_id=order.order_id,
                exchange_order_id=exchange_oid,
                strategy_id=strategy_id,
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                status=order.status.value,
                role=order.tag,
                reduce_only=bool(order.metadata.get("reduce_only", False)),
                oca_group=order.oca_group,
                metadata=dict(order.metadata),
            )
            if self._coordinator is not None and strategy_id != "unknown":
                self._coordinator.register_order(order.order_id, strategy_id, order)
                if exchange_oid:
                    self._coordinator.register_order(exchange_oid, strategy_id, order)
        return open_orders

    def _seed_ttl_trackers_from_open_orders(self, open_orders: list[Order] | None = None) -> None:
        """Seed per-strategy live TTL adapters after durable and exchange state sync."""
        for slot in self._slots:
            if open_orders is not None:
                seed_fn = getattr(slot.ctx.broker, "seed_ttl_orders", None)
                if callable(seed_fn):
                    seed_fn(open_orders)
                    continue
            seed_fn = getattr(slot.ctx.broker, "seed_ttl_orders_from_open_orders", None)
            if callable(seed_fn):
                seed_fn()

    def _clear_ttl_tracking_for_fill(self, slot: _StrategySlot, fill: Fill) -> None:
        clear_fn = getattr(slot.ctx.broker, "clear_ttl_for_fill", None)
        if callable(clear_fn):
            clear_fn(fill)

    def _record_oms_fill_received(self, fill_id: str, fill: Fill, strategy_id: str) -> None:
        """Persist a seen fill without treating it as safely consumed."""
        oms = getattr(self, "_oms", None)
        if oms is None:
            return
        broker = getattr(self, "_broker", None)
        exchange_oid = str(getattr(broker, "_local_to_oid", {}).get(fill.order_id, ""))
        exchange_oid = fill.exchange_order_id or exchange_oid
        record_fn = getattr(oms, "record_received_fill", None)
        if not callable(record_fn):
            return
        record_fn(
            fill_id=fill_id,
            client_order_id=fill.order_id,
            exchange_order_id=exchange_oid,
            strategy_id=strategy_id,
            symbol=fill.symbol,
            side=fill.side.value,
            qty=fill.qty,
            price=fill.fill_price,
            commission=fill.commission,
            timestamp=fill.timestamp,
            exchange_fill_id=fill.exchange_fill_id,
            raw={"tag": fill.tag, **dict(fill.raw)},
        )

    def _mark_oms_fill_strategy_dispatched(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_strategy_dispatched", strategy_id=strategy_id)

    def _mark_oms_fill_coordinator_applied(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_coordinator_applied", strategy_id=strategy_id)

    def _mark_oms_fill_lifecycle_applied(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_lifecycle_applied", strategy_id=strategy_id)

    def _mark_oms_fill_finalized(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_finalized", strategy_id=strategy_id)

    def _mark_oms_fill_processed(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_processed", strategy_id=strategy_id)

    def _mark_oms_fill_unresolved(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
        reason: str = "",
    ) -> None:
        self._mark_oms_fill(
            fill_id,
            "mark_fill_unresolved",
            strategy_id=strategy_id,
            reason=reason,
        )

    def _mark_oms_fill_processing_failed(
        self,
        fill_id: str,
        *,
        strategy_id: str,
        error: str,
    ) -> None:
        self._mark_oms_fill(
            fill_id,
            "mark_fill_processing_failed",
            strategy_id=strategy_id,
            error=error,
        )

    def _record_oms_fill_processing_error(
        self,
        fill_id: str,
        *,
        strategy_id: str,
        error: str,
    ) -> None:
        self._mark_oms_fill(
            fill_id,
            "record_fill_processing_error",
            strategy_id=strategy_id,
            error=error,
        )

    def _mark_oms_fill(self, fill_id: str, method_name: str, **kwargs: Any) -> None:
        oms = getattr(self, "_oms", None)
        mark_fn = getattr(oms, method_name, None)
        if callable(mark_fn):
            mark_fn(fill_id, **kwargs)

    def _find_slot(self, strategy_id: str) -> _StrategySlot | None:
        for slot in self._slots:
            if slot.strategy_id == strategy_id:
                return slot
        return None

    def _load_portfolio_config(self) -> PortfolioConfig:
        """Load portfolio config from file or create default."""
        if self._config.portfolio_config_path and self._config.portfolio_config_path.exists():
            with open(self._config.portfolio_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PortfolioConfig.from_dict(data)

        # Default: all strategies enabled
        from crypto_trader.portfolio.config import StrategyAllocation
        return PortfolioConfig(
            initial_equity=self._broker.get_equity() if self._broker else 10_000.0,
            strategies=tuple(
                StrategyAllocation(strategy_id=sid)
                for sid in self._config.strategy_configs
            ),
        )

    def _load_strategy_config(self, strategy_id: str, config_path: Path) -> Any:
        """Load strategy-specific config from JSON file."""
        if not config_path.exists():
            log.warning("engine.config_not_found", strategy=strategy_id, path=str(config_path))
            return self._default_strategy_config(strategy_id)

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unwrap "strategy" key if present (from optimization output)
        if "strategy" in data:
            data = data["strategy"]

        if strategy_id == "momentum":
            from crypto_trader.strategy.momentum.config import MomentumConfig
            return MomentumConfig.from_dict(data)
        elif strategy_id == "trend":
            from crypto_trader.strategy.trend.config import TrendConfig
            return TrendConfig.from_dict(data)
        elif strategy_id == "breakout":
            from crypto_trader.strategy.breakout.config import BreakoutConfig
            return BreakoutConfig.from_dict(data)
        else:
            raise ValueError(f"Unknown strategy: {strategy_id}")

    def _build_positions_snapshot(self) -> list[dict]:
        """Build position snapshot for PG upsert."""
        result = []
        if not self._broker:
            return result
        for pos in self._broker.get_positions():
            if pos.qty == 0:
                continue
            tracked = self._tracked_positions.get(pos.symbol, {})
            strategy_id = tracked.get("strategy_id", "unknown")
            risk_r = 0.0
            stop_price = None
            if self._manager:
                for risk in self._manager.state.open_risks:
                    if risk.symbol == pos.symbol:
                        risk_r = risk.risk_R
                        stop_price = getattr(risk, "stop_price", None)
                        break
            result.append({
                "strategy_id": strategy_id,
                "symbol": pos.symbol,
                "direction": pos.direction.value if pos.direction else "unknown",
                "qty": pos.qty,
                "avg_entry": pos.avg_entry,
                "unrealized_pnl": pos.unrealized_pnl,
                "risk_r": risk_r,
                "stop_price": stop_price,
                "entry_time": pos.open_time if hasattr(pos, "open_time") else None,
            })
        return result

    def _default_strategy_config(self, strategy_id: str) -> Any:
        """Create default config for a strategy."""
        if strategy_id == "momentum":
            from crypto_trader.strategy.momentum.config import MomentumConfig
            return MomentumConfig()
        elif strategy_id == "trend":
            from crypto_trader.strategy.trend.config import TrendConfig
            return TrendConfig()
        elif strategy_id == "breakout":
            from crypto_trader.strategy.breakout.config import BreakoutConfig
            return BreakoutConfig()
        else:
            raise ValueError(f"Unknown strategy: {strategy_id}")
