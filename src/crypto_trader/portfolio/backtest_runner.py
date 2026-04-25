"""Portfolio backtest runner — multi-strategy backtest with per-strategy brokers and portfolio rules.

Each strategy gets its own SimBroker (isolated positions/margin), matching individual
backtest behavior. The portfolio manager provides cross-strategy coordination rules.
The only trade-count difference from individual runs should come from portfolio blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import heapq

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import PerformanceMetrics, compute_metrics
from crypto_trader.backtest.runner import _create_strategy
from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import EventBus, FillEvent, PositionClosedEvent
from crypto_trader.core.models import Bar, TimeFrame, Trade
from crypto_trader.data.historical_feed import HistoricalFeed, _TF_PRIORITY
from crypto_trader.data.store import ParquetStore
from crypto_trader.exchange.funding import FundingHelper
from crypto_trader.exchange.meta import AssetMeta
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState

log = structlog.get_logger()

# Strategy-specific warmup requirements (in days)
_STRATEGY_WARMUP = {
    "momentum": 0,    # 200 M15 bars ≈ 2 days, handled by bar count
    "trend": 60,      # D1 EMA50 needs 51 bars
    "breakout": 0,    # 101 M30 bars ≈ 2 days, handled by bar count
}


@dataclass
class RuleEvent:
    """A portfolio rule check event for audit logging."""

    timestamp: datetime
    strategy_id: str
    symbol: str
    direction: str
    risk_R: float
    approved: bool
    denial_reason: str | None = None
    size_multiplier: float = 1.0


@dataclass
class PortfolioBacktestResult:
    """Results from a portfolio backtest."""

    per_strategy_trades: dict[str, list[Trade]]
    all_trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    metrics: PerformanceMetrics
    rule_events: list[RuleEvent]
    per_strategy_metrics: dict[str, PerformanceMetrics | None] = field(default_factory=dict)
    config: BacktestConfig | None = None
    portfolio_config: PortfolioConfig | None = None


@dataclass
class _StrategySlot:
    """Internal: wraps one strategy within the portfolio backtest."""

    strategy_id: str
    strategy: Any  # Strategy protocol
    broker: SimBroker  # per-strategy broker (isolated positions)
    ctx: StrategyContext
    bars: MultiTimeFrameBars
    subscribed_tfs: set[TimeFrame]
    primary_tf: TimeFrame
    feed_tfs: list[TimeFrame]
    feed: HistoricalFeed | None = None


def run_portfolio_backtest(
    portfolio_config: PortfolioConfig,
    strategy_configs: dict[str, Any],
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
) -> PortfolioBacktestResult:
    """Run a multi-strategy portfolio backtest.

    Architecture: each strategy gets its own SimBroker for position isolation.
    Entry orders are intercepted by BrokerProxy → PortfolioManager for
    cross-strategy rule checks. The only trade-count difference from
    individual runs comes from portfolio blocks.

    Args:
        portfolio_config: Portfolio-level risk rules
        strategy_configs: {strategy_id: config_object} for each strategy
        backtest_config: Shared backtest parameters
        data_dir: Path to data directory
        meta_path: Path to asset metadata cache

    Returns:
        PortfolioBacktestResult with per-strategy and combined results
    """
    symbols = backtest_config.symbols or ["BTC", "ETH", "SOL"]

    store = ParquetStore(base_dir=data_dir)

    # Load asset meta
    asset_meta = None
    if meta_path and meta_path.exists():
        asset_meta = AssetMeta.from_cache(meta_path)

    # Load funding
    funding_helpers: dict[str, FundingHelper] = {}
    if backtest_config.apply_funding:
        for sym in symbols:
            df = store.load_funding(sym)
            if df is not None and not df.empty:
                funding_helpers[sym] = FundingHelper(df)

    # Compute warmup: max across all strategies
    max_warmup = max(
        _STRATEGY_WARMUP.get(sid, 0)
        for sid in strategy_configs
    )
    max_warmup = max(max_warmup, backtest_config.warmup_days)

    actual_start = backtest_config.start_date
    if max_warmup > 0 and actual_start is not None:
        warmup_start = actual_start - timedelta(days=max_warmup)
    else:
        warmup_start = actual_start

    clock = SimClock()

    # Create portfolio management (coordinator uses a dummy broker — not used for orders)
    state = PortfolioState(
        equity=portfolio_config.initial_equity,
        peak_equity=portfolio_config.initial_equity,
    )
    manager = PortfolioManager(config=portfolio_config, state=state)

    # Create a minimal broker reference for coordinator (only used for order owner lookup)
    # Each strategy has its own actual broker
    _coordinator_broker = SimBroker(initial_equity=0)
    coordinator = StrategyCoordinator(broker=_coordinator_broker, manager=manager)

    # Create strategy slots — each with its own SimBroker
    slots: list[_StrategySlot] = []
    rule_events: list[RuleEvent] = []

    for strategy_id, strategy_config in strategy_configs.items():
        alloc = portfolio_config.get_strategy(strategy_id)
        if alloc is None or not alloc.enabled:
            continue

        # Ensure symbols are set on strategy config
        strategy_config.symbols = symbols

        strategy, feed_tfs, primary_tf = _create_strategy(strategy_id, strategy_config)
        subscribed_tfs = set(feed_tfs)

        # Per-strategy broker (isolated positions, same initial equity)
        strategy_broker = SimBroker(
            initial_equity=portfolio_config.initial_equity,
            taker_fee_bps=backtest_config.taker_fee_bps,
            maker_fee_bps=backtest_config.maker_fee_bps,
            slippage_bps=backtest_config.slippage_bps,
            spread_bps=backtest_config.spread_bps,
            asset_meta=asset_meta,
            funding_helpers=funding_helpers if funding_helpers else None,
        )

        proxy = coordinator.get_proxy(strategy_id)
        # Point proxy at this strategy's broker (not the coordinator's dummy)
        proxy._broker = strategy_broker

        events = EventBus()
        bars = MultiTimeFrameBars()

        ctx = StrategyContext(
            broker=proxy,
            clock=clock,
            bars=bars,
            events=events,
            config=backtest_config,
        )

        slots.append(_StrategySlot(
            strategy_id=strategy_id,
            strategy=strategy,
            broker=strategy_broker,
            ctx=ctx,
            bars=bars,
            subscribed_tfs=subscribed_tfs,
            primary_tf=primary_tf,
            feed_tfs=feed_tfs,
        ))

    # Create per-strategy feeds (each with correct primary_timeframe)
    # Guarantees each strategy sees exactly the same bars as in individual mode
    for slot in slots:
        slot.feed = HistoricalFeed(
            store=store,
            symbols=symbols,
            timeframes=sorted(slot.feed_tfs, key=lambda tf: tf.minutes),
            start_date=warmup_start,
            end_date=backtest_config.end_date,
            primary_timeframe=slot.primary_tf,
        )

    # Wire up portfolio rule event logging
    _orig_check = manager.check_entry

    def _logging_check(strategy_id, symbol, direction, new_risk_R):
        result = _orig_check(strategy_id, symbol, direction, new_risk_R)
        rule_events.append(RuleEvent(
            timestamp=clock.now(),
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction.value,
            risk_R=new_risk_R,
            approved=result.approved,
            denial_reason=result.denial_reason,
            size_multiplier=result.size_multiplier,
        ))
        return result

    manager.check_entry = _logging_check  # type: ignore[method-assign]

    # Set module-level reference for equity calculation in _process_slot_primary
    global _all_slots_ref
    _all_slots_ref = slots

    # Init strategies
    for slot in slots:
        slot.strategy.on_init(slot.ctx)

    log.info(
        "portfolio_backtest.start",
        strategies=[s.strategy_id for s in slots],
        symbols=symbols,
        start=str(backtest_config.start_date),
        end=str(backtest_config.end_date),
    )

    # Determine measurement start for warmup filtering
    measurement_start = None
    if max_warmup > 0 and actual_start is not None:
        measurement_start = datetime.combine(
            actual_start, datetime.min.time(), tzinfo=timezone.utc
        )

    # Main loop — merged iteration from per-strategy feeds
    # Each strategy sees exactly the same bars as in individual mode
    slot_map = {s.strategy_id: s for s in slots}
    feed_iters: dict[str, object] = {}
    _bar_store: dict[int, Bar] = {}
    _heap: list[tuple] = []
    _seq = 0

    for slot in slots:
        it = iter(slot.feed)
        feed_iters[slot.strategy_id] = it
        try:
            bar = next(it)
            _bar_store[_seq] = bar
            heapq.heappush(_heap, (bar.timestamp, _TF_PRIORITY.get(bar.timeframe, 99), _seq, slot.strategy_id))
            _seq += 1
        except StopIteration:
            pass

    while _heap:
        _, _, seq_id, sid = heapq.heappop(_heap)
        bar = _bar_store.pop(seq_id)
        slot = slot_map[sid]

        if hasattr(clock, "advance"):
            clock.advance(bar.timestamp)

        today = bar.timestamp.date()
        manager.maybe_reset_daily(today)

        if bar.timeframe == slot.primary_tf:
            _process_slot_primary(bar, slot, coordinator, manager, clock)
        else:
            _process_slot_higher_tf(bar, slot)

        # Push next bar from this feed
        it = feed_iters[sid]
        try:
            next_bar = next(it)
            _bar_store[_seq] = next_bar
            heapq.heappush(_heap, (next_bar.timestamp, _TF_PRIORITY.get(next_bar.timeframe, 99), _seq, sid))
            _seq += 1
        except StopIteration:
            pass

    # Force-close open positions per strategy
    for slot in slots:
        _close_slot_positions(slot, coordinator, manager)

    # Trim warmup and collect results
    per_strategy_trades: dict[str, list[Trade]] = {}
    per_strategy_metrics: dict[str, PerformanceMetrics | None] = {}
    all_trades: list[Trade] = []

    # Build combined equity curve from all brokers
    combined_equity: list[tuple[datetime, float]] = []

    for slot in slots:
        broker = slot.broker

        # Trim warmup from this broker's equity
        if measurement_start is not None:
            broker._equity_history = [
                (ts, eq) for ts, eq in broker._equity_history
                if ts >= measurement_start
            ]
            liq_hist = getattr(broker, '_liquidation_equity_history', [])
            if liq_hist:
                broker._liquidation_equity_history = [
                    (ts, eq) for ts, eq in liq_hist
                    if ts >= measurement_start
                ]
            initial_curve = broker._liquidation_equity_history or broker._equity_history
            if initial_curve:
                broker._initial_equity = initial_curve[0][1]

        # Collect trades (filter warmup)
        strategy_trades = []
        for trade in broker._closed_trades:
            if measurement_start and trade.entry_time < measurement_start:
                continue
            strategy_trades.append(trade)

        per_strategy_trades[slot.strategy_id] = strategy_trades
        all_trades.extend(strategy_trades)

        # Per-strategy metrics
        sm = compute_metrics(broker)
        per_strategy_metrics[slot.strategy_id] = sm

        # Accumulate equity history for combined curve
        eq_curve = broker._liquidation_equity_history or broker._equity_history
        combined_equity.extend(eq_curve)

    all_trades.sort(key=lambda t: t.entry_time)

    # Sort combined equity by timestamp and compute portfolio-level equity
    # (sum of per-strategy equity deltas from initial)
    combined_equity.sort(key=lambda x: x[0])

    # Shutdown strategies
    for slot in slots:
        slot.strategy.on_shutdown(slot.ctx)

    # Build a synthetic broker for combined metrics
    combined_broker = SimBroker(initial_equity=portfolio_config.initial_equity)
    combined_broker._closed_trades = all_trades
    combined_broker._initial_equity = portfolio_config.initial_equity

    # Build combined equity: sum equity deltas across all strategies
    _build_combined_equity(slots, combined_broker, portfolio_config.initial_equity, measurement_start)

    metrics = compute_metrics(combined_broker)

    return PortfolioBacktestResult(
        per_strategy_trades=per_strategy_trades,
        all_trades=all_trades,
        equity_curve=combined_broker._liquidation_equity_history or combined_broker._equity_history,
        metrics=metrics,
        rule_events=rule_events,
        per_strategy_metrics=per_strategy_metrics,
        config=backtest_config,
        portfolio_config=portfolio_config,
    )


def _build_combined_equity(
    slots: list[_StrategySlot],
    combined_broker: SimBroker,
    initial_equity: float,
    measurement_start: datetime | None,
) -> None:
    """Build combined equity curve from per-strategy equity histories.

    Portfolio equity = initial + sum of per-strategy P&L at each timestamp.
    """
    # Collect all equity snapshots with strategy identity
    all_snapshots: list[tuple[datetime, str, float]] = []
    strategy_initial: dict[str, float] = {}

    for slot in slots:
        broker = slot.broker
        eq_curve = broker._liquidation_equity_history or broker._equity_history
        strategy_initial[slot.strategy_id] = broker._initial_equity

        for ts, eq in eq_curve:
            all_snapshots.append((ts, slot.strategy_id, eq))

    if not all_snapshots:
        return

    # Sort by timestamp
    all_snapshots.sort(key=lambda x: x[0])

    # Track latest equity per strategy, compute combined
    latest_equity: dict[str, float] = dict(strategy_initial)
    combined_history: list[tuple[datetime, float]] = []

    for ts, sid, eq in all_snapshots:
        latest_equity[sid] = eq
        # Combined = initial + sum(strategy_equity - strategy_initial)
        combined = initial_equity + sum(
            latest_equity[s] - strategy_initial[s]
            for s in latest_equity
        )
        combined_history.append((ts, combined))

    combined_broker._equity_history = combined_history
    if combined_history:
        combined_broker._initial_equity = combined_history[0][1]


def _process_slot_primary(
    bar: Bar,
    slot: _StrategySlot,
    coordinator: StrategyCoordinator,
    manager: PortfolioManager,
    clock: Any,
) -> None:
    """Process a primary-TF bar for one strategy: fills first, then dispatch."""
    broker = slot.broker
    closed_before = len(broker._closed_trades)

    # Process fills
    fills = broker.process_bar(bar)
    for fill in fills:
        coordinator.on_fill(fill)
        slot.strategy.on_fill(fill, slot.ctx)
        slot.ctx.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))

    # Recheck entry-bar stops
    if fills:
        check_fn = getattr(broker, 'check_entry_bar_stops', None)
        if check_fn is not None:
            recheck_fills = check_fn(bar)
            for fill in recheck_fills:
                coordinator.on_fill(fill)
                slot.strategy.on_fill(fill, slot.ctx)
                slot.ctx.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))
            if recheck_fills:
                refresh_fn = getattr(broker, "refresh_current_bar_equity", None)
                if refresh_fn is not None:
                    refresh_fn(bar.timestamp)

    # Emit PositionClosedEvents + update portfolio state
    for trade in broker._closed_trades[closed_before:]:
        pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
        coordinator.on_trade_closed(slot.strategy_id, trade.symbol, pnl_R)
        slot.ctx.events.emit(PositionClosedEvent(
            timestamp=trade.exit_time, trade=trade,
        ))

    # Activate deferred orders
    activate_fn = getattr(broker, 'activate_deferred', None)
    if activate_fn is not None:
        activate_fn()

    # Update portfolio equity (sum of all strategy equities)
    total_equity = sum(s.broker.get_equity() for s in _all_slots_ref)
    manager.update_equity(total_equity)

    # Dispatch bar to strategy
    slot.bars.append(bar)
    slot.strategy.on_bar(bar, slot.ctx)


def _process_slot_higher_tf(
    bar: Bar,
    slot: _StrategySlot,
) -> None:
    """Process a higher-TF bar for one strategy: defer orders, dispatch."""
    broker = slot.broker
    start_fn = getattr(broker, 'start_deferring', None)
    stop_fn = getattr(broker, 'stop_deferring', None)

    slot.bars.append(bar)

    if start_fn is not None:
        start_fn()

    slot.strategy.on_bar(bar, slot.ctx)

    if stop_fn is not None:
        stop_fn()


def _close_slot_positions(
    slot: _StrategySlot,
    coordinator: StrategyCoordinator,
    manager: PortfolioManager,
) -> None:
    """Force-close open positions for one strategy at backtest end."""
    broker = slot.broker
    close_fn = getattr(broker, "close_open_positions", None)
    if close_fn is None:
        return

    closed_before = len(broker._closed_trades)
    fills = close_fn()

    for fill in fills:
        coordinator.on_fill(fill)
        slot.strategy.on_fill(fill, slot.ctx)
        slot.ctx.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))

    for trade in broker._closed_trades[closed_before:]:
        pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
        coordinator.on_trade_closed(slot.strategy_id, trade.symbol, pnl_R)
        slot.ctx.events.emit(PositionClosedEvent(
            timestamp=trade.exit_time, trade=trade,
        ))

    # Final equity snapshot
    broker._equity = broker._cash


# Module-level reference to all slots (set during run_portfolio_backtest)
_all_slots_ref: list[_StrategySlot] = []
