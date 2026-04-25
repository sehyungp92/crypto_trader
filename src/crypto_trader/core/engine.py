"""Strategy engine: unified run loop for backtesting and live trading."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

import structlog

from crypto_trader.core.broker import BrokerAdapter
from crypto_trader.core.clock import Clock
from crypto_trader.core.events import BarEvent, EventBus, FillEvent, PositionClosedEvent
from crypto_trader.core.models import Bar, Fill, TerminalMark, TimeFrame, Trade

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Strategy(Protocol):
    """Protocol that all strategies must implement."""

    @property
    def name(self) -> str: ...

    @property
    def symbols(self) -> list[str]: ...

    @property
    def timeframes(self) -> list[TimeFrame]: ...

    def on_init(self, ctx: StrategyContext) -> None: ...
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None: ...
    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None: ...
    def on_shutdown(self, ctx: StrategyContext) -> None: ...


# ---------------------------------------------------------------------------
# Multi-timeframe bar storage
# ---------------------------------------------------------------------------

class MultiTimeFrameBars:
    """Rolling window of bars per (symbol, timeframe) pair.

    Provides O(1) latest-bar access and O(count) history retrieval.
    """

    def __init__(self, max_bars: int = 500) -> None:
        self._max_bars = max_bars
        self._bars: dict[tuple[str, TimeFrame], deque[Bar]] = defaultdict(
            lambda: deque(maxlen=max_bars)
        )

    def append(self, bar: Bar) -> None:
        """Add a bar to the rolling window."""
        self._bars[(bar.symbol, bar.timeframe)].append(bar)

    def get(
        self,
        symbol: str,
        tf: TimeFrame,
        count: int | None = None,
    ) -> list[Bar]:
        """Get bars for a symbol/timeframe. Returns up to `count` most recent."""
        buf = self._bars.get((symbol, tf))
        if buf is None:
            return []
        if count is None:
            return list(buf)
        return list(buf)[-count:]

    def latest(self, symbol: str, tf: TimeFrame) -> Bar | None:
        """Get the most recent bar, or None if no bars yet."""
        buf = self._bars.get((symbol, tf))
        if not buf:
            return None
        return buf[-1]


# ---------------------------------------------------------------------------
# Strategy context
# ---------------------------------------------------------------------------

@dataclass
class StrategyContext:
    """Injected into strategy callbacks — provides access to all engine services."""
    broker: BrokerAdapter
    clock: Clock
    bars: MultiTimeFrameBars
    events: EventBus
    config: Any = None


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """Unified engine that drives strategy execution over a data feed.

    The run loop:
      1. on_init(ctx)
      2. For each bar from feed:
         a. clock.advance(bar.timestamp)
         b. If primary TF: broker.process_bar(bar) -> dispatch fills
         c. bars.append(bar)
         d. strategy.on_bar(bar, ctx)
         e. Emit BarEvent
      3. on_shutdown(ctx)

    Critical invariants:
      - process_bar runs BEFORE on_bar: orders from bar N fill against bar N+1
      - process_bar only runs for primary TF bars (default M15). Higher-TF bars
        are synthetic aggregates with OHLC that don't represent tradeable prices.
    """

    def __init__(
        self,
        strategy: Strategy,
        broker: BrokerAdapter,
        feed: Any,  # DataFeed protocol — Any to avoid import cycle
        clock: Clock,
        events: EventBus | None = None,
        config: Any = None,
        primary_timeframe: TimeFrame = TimeFrame.M15,
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.feed = feed
        self.clock = clock
        self.events = events or EventBus()
        self.config = config
        self.primary_timeframe = primary_timeframe

        self._bars = MultiTimeFrameBars()
        self._ctx = StrategyContext(
            broker=self.broker,
            clock=self.clock,
            bars=self._bars,
            events=self.events,
            config=self.config,
        )
        self._bar_count = 0
        self._fill_count = 0

    def run(self) -> None:
        """Execute the full strategy run loop."""
        log.info(
            "engine.start",
            strategy=self.strategy.name,
            symbols=self.strategy.symbols,
            primary_tf=self.primary_timeframe.value,
        )

        self.strategy.on_init(self._ctx)

        for bar in self.feed:
            self._process_single_bar(bar)

        self.strategy.on_shutdown(self._ctx)

        log.info(
            "engine.complete",
            bars_processed=self._bar_count,
            fills=self._fill_count,
        )

    def _process_single_bar(self, bar: Bar) -> None:
        """Process a single bar through the engine pipeline.

        For primary TF bars:
          1. process_bar: fill pending orders (deferred NOT included yet)
          2. Dispatch fills to strategy (on_fill submits protective stops)
          3. Recheck entry-bar stops against the same bar (Finding 2)
          4. Emit PositionClosedEvents
          5. Activate deferred orders for NEXT primary bar (Finding 1)
          6. Append bar + notify strategy (new orders go to active, fill at NEXT process_bar)

        For higher-TF bars:
          - Defer all orders submitted during on_bar so they don't fill at the
            co-boundary primary bar (prevents timing leak — Finding 1).
        """
        # Step a: advance clock
        if hasattr(self.clock, "advance"):
            self.clock.advance(bar.timestamp)

        # Step b: process fills only on primary TF bars
        if bar.timeframe == self.primary_timeframe:
            # Snapshot closed trade count to detect new trades
            closed_before = len(getattr(self.broker, '_closed_trades', []))
            fills = self._try_process_bar(bar)
            for fill in fills:
                self._fill_count += 1
                self.strategy.on_fill(fill, self._ctx)
                self.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))

            # Step b2: recheck newly submitted protective stops against entry bar
            if fills:
                check_fn = getattr(self.broker, 'check_entry_bar_stops', None)
                if check_fn is not None:
                    recheck_fills = check_fn(bar)
                    for fill in recheck_fills:
                        self._fill_count += 1
                        self.strategy.on_fill(fill, self._ctx)
                        self.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))
                    if recheck_fills:
                        refresh_fn = getattr(self.broker, "refresh_current_bar_equity", None)
                        if refresh_fn is not None:
                            refresh_fn(bar.timestamp)

            # Step b3: emit PositionClosedEvents for ALL new trades since snapshot
            closed_trades = getattr(self.broker, '_closed_trades', [])
            for trade in closed_trades[closed_before:]:
                self.events.emit(PositionClosedEvent(timestamp=trade.exit_time, trade=trade))

            # Step b4: promote deferred orders for NEXT primary bar
            activate_fn = getattr(self.broker, 'activate_deferred', None)
            if activate_fn is not None:
                activate_fn()

            # Step c: append bar, then notify strategy
            self._bars.append(bar)
            self.strategy.on_bar(bar, self._ctx)
            self._bar_count += 1
            self.events.emit(BarEvent(timestamp=bar.timestamp, bar=bar))
        else:
            # Higher-TF bar: defer any orders submitted during on_bar
            self._bars.append(bar)

            start_fn = getattr(self.broker, 'start_deferring', None)
            stop_fn = getattr(self.broker, 'stop_deferring', None)

            if start_fn is not None:
                start_fn()

            self.strategy.on_bar(bar, self._ctx)
            self._bar_count += 1

            if stop_fn is not None:
                stop_fn()

            self.events.emit(BarEvent(timestamp=bar.timestamp, bar=bar))

    def close_open_positions(self) -> list[Fill]:
        """Force-close all open positions, dispatching fills through the strategy.

        Must be called instead of broker.close_open_positions() so that
        the strategy receives on_fill callbacks and PositionClosedEvents,
        enabling trade enrichment (r_multiple, mae_r, mfe_r, etc.).
        """
        close_fn = getattr(self.broker, "close_open_positions", None)
        if close_fn is None:
            return []

        closed_before = len(getattr(self.broker, '_closed_trades', []))
        fills = close_fn()

        for fill in fills:
            self._fill_count += 1
            self.strategy.on_fill(fill, self._ctx)
            self.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))

        closed_trades = getattr(self.broker, '_closed_trades', [])
        for trade in closed_trades[closed_before:]:
            self.events.emit(PositionClosedEvent(timestamp=trade.exit_time, trade=trade))

        return fills

    def mark_open_positions(self) -> list[TerminalMark]:
        """Create explicit terminal marks and let the strategy enrich them."""
        mark_fn = getattr(self.broker, "mark_open_positions", None)
        if mark_fn is None:
            return []

        terminal_marks = mark_fn()
        enrich_fn = getattr(self.strategy, "enrich_terminal_marks", None)
        if enrich_fn is not None and terminal_marks:
            enrich_fn(terminal_marks)
        return terminal_marks

    def _try_process_bar(self, bar: Bar) -> list[Fill]:
        """Call broker.process_bar if available (duck-type check)."""
        process_bar = getattr(self.broker, "process_bar", None)
        if process_bar is not None:
            return process_bar(bar)
        return []
