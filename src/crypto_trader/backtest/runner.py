"""Backtest runner — wires components and executes strategy over historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import PerformanceMetrics, compute_metrics
from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import StrategyEngine
from crypto_trader.core.events import EventBus
from crypto_trader.core.models import TerminalMark, TimeFrame, Trade
from crypto_trader.data.historical_feed import HistoricalFeed
from crypto_trader.data.store import ParquetStore
from crypto_trader.exchange.funding import FundingHelper
from crypto_trader.exchange.meta import AssetMeta

log = structlog.get_logger()


def _create_strategy(strategy_type: str, strategy_config):
    """Factory: create strategy instance, timeframes list, and primary TF."""
    if strategy_type == "trend":
        from crypto_trader.strategy.trend.config import TrendConfig
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        cfg = strategy_config if isinstance(strategy_config, TrendConfig) else TrendConfig()
        return TrendStrategy(cfg), [TimeFrame.M15, TimeFrame.H1, TimeFrame.D1], TimeFrame.M15
    elif strategy_type == "breakout":
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        from crypto_trader.strategy.breakout.strategy import BreakoutStrategy
        cfg = strategy_config if isinstance(strategy_config, BreakoutConfig) else BreakoutConfig()
        return BreakoutStrategy(cfg), [TimeFrame.M30, TimeFrame.H4], TimeFrame.M30
    else:
        from crypto_trader.strategy.momentum.config import MomentumConfig
        from crypto_trader.strategy.momentum.strategy import MomentumStrategy
        cfg = strategy_config if isinstance(strategy_config, MomentumConfig) else MomentumConfig()
        return MomentumStrategy(cfg), [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4], TimeFrame.M15


@dataclass
class BacktestResult:
    trades: list[Trade]
    terminal_marks: list[TerminalMark]
    equity_curve: list[tuple[datetime, float]]
    metrics: PerformanceMetrics
    journal: object  # TradeJournal — strategy-agnostic
    config: BacktestConfig | None = None
    diagnostic_context: dict[str, object] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    train: BacktestResult
    test: BacktestResult


def run(
    strategy_config,
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    store=None,
    strategy_type: str = "momentum",
) -> BacktestResult:
    """Run a full backtest with the specified strategy."""
    symbols = backtest_config.symbols or strategy_config.symbols
    strategy_config.symbols = symbols

    if store is None:
        store = ParquetStore(base_dir=data_dir)

    # Load asset meta
    asset_meta = None
    if meta_path and meta_path.exists():
        asset_meta = AssetMeta.from_cache(meta_path)

    # Load funding if requested
    funding_helpers: dict[str, FundingHelper] = {}
    if backtest_config.apply_funding:
        for sym in symbols:
            df = store.load_funding(sym)
            if df is not None and not df.empty:
                funding_helpers[sym] = FundingHelper(df)

    # Create strategy, determine timeframes
    strategy, timeframes, primary_tf = _create_strategy(strategy_type, strategy_config)

    # Compute adjusted start for warmup data loading
    actual_start = backtest_config.start_date
    if backtest_config.warmup_days > 0 and actual_start is not None:
        warmup_start = actual_start - timedelta(days=backtest_config.warmup_days)
    else:
        warmup_start = actual_start

    # Create feed (uses warmup_start to load extra pre-measurement data)
    feed = HistoricalFeed(
        store=store,
        symbols=symbols,
        timeframes=timeframes,
        start_date=warmup_start,
        end_date=backtest_config.end_date,
        primary_timeframe=primary_tf,
    )

    # Create broker with per-symbol funding (Finding 6)
    broker = SimBroker(
        initial_equity=backtest_config.initial_equity,
        taker_fee_bps=backtest_config.taker_fee_bps,
        maker_fee_bps=backtest_config.maker_fee_bps,
        slippage_bps=backtest_config.slippage_bps,
        spread_bps=backtest_config.spread_bps,
        asset_meta=asset_meta,
        funding_helpers=funding_helpers if funding_helpers else None,
    )

    clock = SimClock()
    events = EventBus()

    engine = StrategyEngine(
        strategy=strategy,
        broker=broker,
        feed=feed,
        clock=clock,
        events=events,
        config=backtest_config,
        primary_timeframe=primary_tf,
    )

    log.info(
        "backtest.start",
        symbols=symbols,
        start=str(backtest_config.start_date),
        end=str(backtest_config.end_date),
        equity=backtest_config.initial_equity,
    )

    engine.run()
    terminal_marks = engine.mark_open_positions()

    # Re-save journal so persisted artifacts include the final realized trades.
    if hasattr(strategy, '_journal') and hasattr(strategy._journal, 'save'):
        strategy._journal.save()

    # Trim warmup-period equity only — entries are blocked before measurement start.
    if backtest_config.warmup_days > 0 and actual_start is not None:
        measurement_start = datetime.combine(
            actual_start, datetime.min.time(), tzinfo=timezone.utc
        )
        filtered_equity = [
            (ts, eq) for ts, eq in broker._equity_history
            if ts >= measurement_start
        ]
        filtered_liquidation_equity = [
            (ts, eq) for ts, eq in broker._liquidation_equity_history
            if ts >= measurement_start
        ]
        if filtered_equity:
            broker._equity_history = filtered_equity
        if filtered_liquidation_equity:
            broker._liquidation_equity_history = filtered_liquidation_equity
        initial_curve = filtered_liquidation_equity or filtered_equity
        if initial_curve:
            broker._initial_equity = initial_curve[0][1]

    metrics = compute_metrics(broker)
    diagnostic_context: dict[str, object] = {}
    export_diag_context = getattr(strategy, "export_diagnostic_context", None)
    if callable(export_diag_context):
        try:
            payload = export_diag_context()
            if isinstance(payload, dict):
                diagnostic_context = payload
        except Exception:
            log.exception("backtest.export_diagnostic_context.failed")

    return BacktestResult(
        trades=broker._closed_trades,
        terminal_marks=terminal_marks,
        equity_curve=broker._liquidation_equity_history or broker._equity_history,
        metrics=metrics,
        journal=strategy.journal,
        config=backtest_config,
        diagnostic_context=diagnostic_context,
    )


def run_walk_forward(
    strategy_config,
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    strategy_type: str = "momentum",
) -> WalkForwardResult:
    """Run walk-forward analysis: train on first portion, test on remainder."""
    if not backtest_config.start_date or not backtest_config.end_date:
        raise ValueError("start_date and end_date required for walk-forward")

    total_days = (backtest_config.end_date - backtest_config.start_date).days
    train_days = int(total_days * backtest_config.train_pct)

    split_date = backtest_config.start_date + timedelta(days=train_days)

    # Train
    train_cfg = BacktestConfig(
        symbols=backtest_config.symbols,
        start_date=backtest_config.start_date,
        end_date=split_date,
        initial_equity=backtest_config.initial_equity,
        taker_fee_bps=backtest_config.taker_fee_bps,
        maker_fee_bps=backtest_config.maker_fee_bps,
        slippage_bps=backtest_config.slippage_bps,
        spread_bps=backtest_config.spread_bps,
        apply_funding=backtest_config.apply_funding,
        warmup_days=backtest_config.warmup_days,
    )
    train_result = run(strategy_config, train_cfg, data_dir, meta_path, strategy_type=strategy_type)

    # Test
    test_cfg = BacktestConfig(
        symbols=backtest_config.symbols,
        start_date=split_date,
        end_date=backtest_config.end_date,
        initial_equity=backtest_config.initial_equity,
        taker_fee_bps=backtest_config.taker_fee_bps,
        maker_fee_bps=backtest_config.maker_fee_bps,
        slippage_bps=backtest_config.slippage_bps,
        spread_bps=backtest_config.spread_bps,
        apply_funding=backtest_config.apply_funding,
        warmup_days=backtest_config.warmup_days,
    )
    test_result = run(strategy_config, test_cfg, data_dir, meta_path, strategy_type=strategy_type)

    return WalkForwardResult(train=train_result, test=test_result)
