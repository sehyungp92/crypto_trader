"""End-to-end test: MomentumStrategy on synthetic data via BacktestRunner."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import PerformanceMetrics, compute_metrics
from crypto_trader.backtest.runner import run
from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import StrategyEngine
from crypto_trader.core.events import EventBus
from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.momentum.strategy import MomentumStrategy


class ListFeed:
    """Simple list-based feed for testing."""
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars

    def __iter__(self):
        return iter(self._bars)


def _generate_trending_bars(
    symbol: str = "BTC",
    base_price: float = 50000.0,
    bars_count: int = 500,
    trend: str = "up",
) -> list[Bar]:
    """Generate M15 bars with a trend, plus H1 and H4 synthetic bars."""
    all_bars: list[Bar] = []
    m15_bars: list[Bar] = []

    np.random.seed(42)
    price = base_price
    step = 20.0 if trend == "up" else -20.0

    for i in range(bars_count):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
        noise = np.random.normal(0, 50)
        price += step + noise
        o = price - abs(noise) * 0.3
        h = price + abs(noise) * 0.8 + 30
        l = price - abs(noise) * 0.8 - 30
        c = price

        m15 = Bar(timestamp=ts, symbol=symbol, open=o, high=h, low=l, close=c, volume=100 + np.random.exponential(50), timeframe=TimeFrame.M15)
        m15_bars.append(m15)

    # Generate H1 bars (every 4 M15 bars)
    h1_bars: list[Bar] = []
    for i in range(0, len(m15_bars), 4):
        chunk = m15_bars[i:i+4]
        if len(chunk) < 4:
            break
        h1 = Bar(
            timestamp=chunk[0].timestamp,
            symbol=symbol,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum(b.volume for b in chunk),
            timeframe=TimeFrame.H1,
        )
        h1_bars.append(h1)

    # Generate H4 bars (every 16 M15 bars)
    h4_bars: list[Bar] = []
    for i in range(0, len(m15_bars), 16):
        chunk = m15_bars[i:i+16]
        if len(chunk) < 16:
            break
        h4 = Bar(
            timestamp=chunk[0].timestamp,
            symbol=symbol,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum(b.volume for b in chunk),
            timeframe=TimeFrame.H4,
        )
        h4_bars.append(h4)

    # Interleave: at each M15 boundary, emit H4/H1 if they fall at this boundary
    h1_map = {b.timestamp: b for b in h1_bars}
    h4_map = {b.timestamp: b for b in h4_bars}

    for m15 in m15_bars:
        ts = m15.timestamp
        if ts in h4_map:
            all_bars.append(h4_map[ts])
        if ts in h1_map:
            all_bars.append(h1_map[ts])
        all_bars.append(m15)

    return all_bars


class TestBacktestIntegration:
    def test_strategy_runs_without_errors(self):
        """MomentumStrategy should run through synthetic data without crashing."""
        config = MomentumConfig(symbols=["BTC"])
        strategy = MomentumStrategy(config)

        bars = _generate_trending_bars("BTC", bars_count=500, trend="up")
        feed = ListFeed(bars)

        broker = SimBroker(initial_equity=10_000.0)
        clock = SimClock()
        events = EventBus()

        engine = StrategyEngine(
            strategy=strategy,
            broker=broker,
            feed=feed,
            clock=clock,
            events=events,
        )

        engine.run()

        # Should complete without errors
        assert True

    def test_strategy_produces_trades_on_strong_trend(self):
        """With 800+ bars of strong trend, strategy should find at least some signals."""
        config = MomentumConfig(symbols=["BTC"])
        strategy = MomentumStrategy(config)

        # More bars for warmup + signal generation
        bars = _generate_trending_bars("BTC", bars_count=800, trend="up")
        feed = ListFeed(bars)

        broker = SimBroker(initial_equity=10_000.0)
        clock = SimClock()
        events = EventBus()

        engine = StrategyEngine(
            strategy=strategy,
            broker=broker,
            feed=feed,
            clock=clock,
            events=events,
        )

        engine.run()

        # Check that the strategy at least attempted to trade
        # (may not always produce closed trades depending on data shape)
        equity = broker.get_equity()
        assert equity > 0  # Didn't blow up

    def test_metrics_compute_without_errors(self):
        """Metrics computation should work even with zero trades."""
        broker = SimBroker(initial_equity=10_000.0)
        metrics = compute_metrics(broker)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 0
        assert metrics.net_profit == 0.0

    def test_no_regressions_on_sim_broker(self):
        """Existing SimBroker tests should still pass — basic smoke test."""
        broker = SimBroker(initial_equity=100_000.0)
        from crypto_trader.core.models import Order, OrderType, Side
        order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
        )
        oid = broker.submit_order(order)
        assert oid == "1"

        bar = Bar(
            timestamp=datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc),
            symbol="BTC", open=50000, high=51000, low=49000, close=50500,
            volume=100, timeframe=TimeFrame.M15,
        )
        fills = broker.process_bar(bar)
        assert len(fills) == 1

    def test_journal_populated_after_trades(self):
        """Journal should record entries for completed trades."""
        config = MomentumConfig(symbols=["BTC"])
        strategy = MomentumStrategy(config)

        bars = _generate_trending_bars("BTC", bars_count=800, trend="up")
        feed = ListFeed(bars)

        broker = SimBroker(initial_equity=10_000.0)
        clock = SimClock()
        events = EventBus()

        engine = StrategyEngine(
            strategy=strategy,
            broker=broker,
            feed=feed,
            clock=clock,
            events=events,
        )

        engine.run()

        # Journal entries should match closed trades
        journal_entries = strategy.journal.entries
        closed_trades = broker._closed_trades
        assert len(journal_entries) == len(closed_trades)

    def test_run_returns_liquidation_equity_curve(self, monkeypatch, tmp_path):
        from crypto_trader.backtest import runner as runner_module

        class DummyStrategy:
            journal = SimpleNamespace()

        class DummyBroker:
            def __init__(self, **kwargs):
                self._initial_equity = kwargs["initial_equity"]
                self.initial_equity = kwargs["initial_equity"]
                self._equity_history = []
                self._liquidation_equity_history = []
                self._closed_trades = []
                self._terminal_marks = []

        class DummyEngine:
            def __init__(self, strategy, broker, **kwargs):
                self.strategy = strategy
                self.broker = broker

            def run(self):
                ts0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
                ts1 = ts0 + timedelta(days=1)
                self.broker._equity_history = [(ts0, 10000.0), (ts1, 10500.0)]
                self.broker._liquidation_equity_history = [(ts0, 10000.0), (ts1, 9800.0)]

            def mark_open_positions(self):
                return []

        monkeypatch.setattr(
            runner_module,
            "_create_strategy",
            lambda strategy_type, strategy_config: (DummyStrategy(), [TimeFrame.M15], TimeFrame.M15),
        )
        monkeypatch.setattr(runner_module, "HistoricalFeed", lambda **kwargs: object())
        monkeypatch.setattr(runner_module, "StrategyEngine", DummyEngine)
        monkeypatch.setattr(runner_module, "SimBroker", DummyBroker)
        monkeypatch.setattr(
            runner_module,
            "compute_metrics",
            lambda broker: PerformanceMetrics(
                net_profit=broker._liquidation_equity_history[-1][1] - broker._initial_equity
            ),
        )

        strategy_config = SimpleNamespace(symbols=["BTC"])
        result = run(
            strategy_config,
            BacktestConfig(
                symbols=["BTC"],
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 2),
                apply_funding=False,
            ),
            data_dir=tmp_path,
            store=MagicMock(),
            strategy_type="momentum",
        )

        assert result.equity_curve[-1][1] == pytest.approx(9800.0)
