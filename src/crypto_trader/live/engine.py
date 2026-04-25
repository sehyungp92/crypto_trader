"""Live trading engine — async polling loop for paper/live trading."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.runner import _create_strategy
from crypto_trader.core.clock import WallClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import EventBus
from crypto_trader.core.events import PositionClosedEvent
from crypto_trader.core.models import Bar, Fill, Position, Side, TimeFrame, Trade
from crypto_trader.live.broker import HyperliquidBroker
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.feed import LiveFeed
from crypto_trader.live.health import HealthMonitor
from crypto_trader.live.reconciler import PositionReconciler
from crypto_trader.live.state import PersistentState
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState
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

# Strategy primary timeframes for fill detection
_STRATEGY_PRIMARY_TF = {
    "momentum": TimeFrame.M15,
    "trend": TimeFrame.H1,
    "breakout": TimeFrame.M30,
}


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
        self._last_fill_check = datetime.now(timezone.utc)
        self._tracked_positions: dict[str, dict] = {}  # sym → tracked entry data
        self._last_funnels: dict[str, dict] = {}  # strategy_id → last funnel dict
        self._report_builder = HealthReportBuilder()

        # Instrumentation
        self._emitter = EventEmitter()
        self._emitter.add_sink(JsonlSink(config.state_dir))
        self._daily_aggregator = DailyAggregator(bot_id=getattr(config, "bot_id", ""))
        self._emitter.add_sink(self._daily_aggregator)  # aggregator receives all events
        self._sidecar: SidecarForwarder | None = None

    async def start(self) -> None:
        """Initialize all components."""
        log.info("engine.starting", testnet=self._config.is_testnet)

        # Create broker
        self._broker = HyperliquidBroker(
            wallet_address=self._config.wallet_address,
            private_key=self._config.private_key,
            is_testnet=self._config.is_testnet,
            max_slippage_pct=self._config.max_slippage_pct,
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

            strategy, feed_tfs, primary_tf = _create_strategy(strategy_id, strategy_config)
            strategy_tfs[strategy_id] = feed_tfs

            proxy = self._coordinator.get_proxy(strategy_id)
            clock = WallClock()
            events = EventBus()
            bars = MultiTimeFrameBars()

            ctx = StrategyContext(
                broker=proxy,
                clock=clock,
                bars=bars,
                events=events,
            )

            self._slots.append(_StrategySlot(
                strategy_id=strategy_id,
                strategy=strategy,
                ctx=ctx,
                bars=bars,
                subscribed_tfs=set(feed_tfs),
                primary_tf=primary_tf,
            ))

        # Create feed
        from hyperliquid.info import Info
        info = Info(self._config.base_url, skip_ws=True)
        self._feed = LiveFeed(info, self._config.symbols, strategy_tfs)

        # Load warmup bars
        warmup_bars = self._feed.load_warmup_bars(info, _WARMUP_COUNTS)

        # Wire instrumentation into each strategy's collector
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is not None:
                collector.emitter = self._emitter

        # Init strategies
        for slot in self._slots:
            slot.strategy.on_init(slot.ctx)

        # Feed warmup bars (suppress orders by not processing fills)
        log.info("engine.warmup_start", bars=len(warmup_bars))
        for bar in warmup_bars:
            for slot in self._slots:
                if bar.timeframe in slot.subscribed_tfs and bar.symbol in slot.strategy.symbols:
                    slot.bars.append(bar)
                    slot.strategy.on_bar(bar, slot.ctx)
        log.info("engine.warmup_complete")

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

        # Start sidecar forwarder if relay is configured
        relay_url = getattr(self._config, "relay_url", "")
        relay_secret = getattr(self._config, "relay_secret", "")
        bot_id = getattr(self._config, "bot_id", "")
        if relay_url and relay_secret:
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

        # Persist final state
        if self._manager:
            self._persistent.save_portfolio_state(self._manager.state.to_dict())

        log.info("engine.shutdown_complete")

    # -------------------------------------------------------------------
    # Polling loops
    # -------------------------------------------------------------------

    async def _poll_candles_loop(self) -> None:
        """Poll for new bars at configured interval."""
        while self._running:
            try:
                self._health.on_poll()
                bars = self._feed.poll()

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
                fills = self._broker.get_fills_since(self._last_fill_check)

                for fill in fills:
                    strategy_id = self._broker.get_order_owner(fill.order_id)
                    if strategy_id:
                        slot = self._find_slot(strategy_id)
                        if slot:
                            # Track entry fills for position lifecycle
                            if fill.tag == "entry":
                                self._tracked_positions[fill.symbol] = {
                                    "strategy_id": strategy_id,
                                    "direction": fill.side,
                                    "entry_price": fill.fill_price,
                                    "entry_time": fill.timestamp,
                                    "qty": fill.qty,
                                }

                            slot.strategy.on_fill(fill, slot.ctx)
                            self._coordinator.on_fill(fill)

                            # Record fill in pipeline tracker
                            collector = getattr(slot.strategy, "_collector", None)
                            if collector is not None:
                                collector.pipeline.record_fill(fill.symbol)

                            log.info(
                                "engine.fill",
                                strategy=strategy_id,
                                symbol=fill.symbol,
                                side=fill.side.value,
                                qty=fill.qty,
                                price=fill.fill_price,
                            )
                    else:
                        log.warning("engine.unattributed_fill", order_id=fill.order_id)

                # After ALL fills processed, detect position closures
                if fills:
                    self._detect_position_closures(fills)

                # Update timestamp only after successful processing (fix race)
                self._last_fill_check = datetime.now(timezone.utc)

            except Exception:
                self._health.on_error("fill_poll")

            await asyncio.sleep(self._config.fill_poll_interval_sec)

    async def _equity_snapshot_loop(self) -> None:
        """Record equity snapshots periodically."""
        while self._running:
            await asyncio.sleep(self._config.equity_snapshot_interval_sec)
            try:
                equity = self._broker.get_equity()
                self._manager.update_equity(equity)
                self._persistent.append_equity_snapshot(equity)
                self._persistent.save_portfolio_state(self._manager.state.to_dict())
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

    def _dispatch_bar(self, bar: Bar) -> None:
        """Route a bar to subscribing strategies."""
        self._health.on_bar_received(bar.symbol, bar.timeframe.value)
        for slot in self._slots:
            if bar.timeframe in slot.subscribed_tfs and bar.symbol in slot.strategy.symbols:
                slot.bars.append(bar)
                slot.strategy.on_bar(bar, slot.ctx)

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
            qty = meta.original_qty if meta and hasattr(meta, "original_qty") else tracked["qty"]
            stop_distance = meta.stop_distance if meta and hasattr(meta, "stop_distance") else 0.0

            if direction == Side.LONG:
                pnl = (exit_fill.fill_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_fill.fill_price) * qty

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
                commission=0.0,
                bars_held=0,
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
        """Periodic pipeline funnel snapshots (every 60 min)."""
        while self._running:
            await asyncio.sleep(3600)
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
        """Periodic health report (every 60 min)."""
        while self._running:
            await asyncio.sleep(3600)
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
                        "heat_R": sum(r.r_risked for r in self._manager.state.open_risks),
                        "heat_cap_R": self._manager._config.max_portfolio_heat_R,
                        "daily_pnl_R": self._manager.state.daily_pnl_R,
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

                self._emitter.emit_health_report(HealthReportSnapshot(
                    timestamp=report.timestamp,
                    report=report.to_dict(),
                ))

                if report.assessment == "critical":
                    log.error("engine.health_critical", alerts=len(report.alerts))

            except Exception:
                log.exception("engine.health_report_error")
                self._health.on_error("health_report")

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
