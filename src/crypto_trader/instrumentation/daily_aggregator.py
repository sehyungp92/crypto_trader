"""DailyAggregator — live mode only. Computes DailySnapshot at UTC midnight."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    EventMetadata,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)


class DailyAggregator:
    """Accumulates trade/missed events during the day, computes DailySnapshot.

    Implements the Sink protocol so it can be added directly to EventEmitter.
    """

    def __init__(self, bot_id: str = "") -> None:
        self._bot_id = bot_id
        self._today_trades: list[InstrumentedTradeEvent] = []
        self._today_missed: list[MissedOpportunityEvent] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._current_date: str = ""

    # --- Sink protocol methods ---

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        self._today_trades.append(event)

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        self._today_missed.append(event)

    def write_daily(self, event: DailySnapshot) -> None:
        pass  # We produce these, not consume them

    def write_error(self, event: ErrorEvent) -> None:
        pass  # Not tracked by daily aggregator

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        pass  # Not tracked by daily aggregator

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        pass  # Not tracked by daily aggregator

    # --- Legacy convenience aliases ---

    def record_trade(self, event: InstrumentedTradeEvent) -> None:
        self._today_trades.append(event)

    def record_missed(self, event: MissedOpportunityEvent) -> None:
        self._today_missed.append(event)

    def record_equity(self, timestamp: datetime, equity: float) -> None:
        self._equity_history.append((timestamp, equity))

    def compute_snapshot(self, date_str: str) -> DailySnapshot:
        """Build a DailySnapshot from today's accumulated events."""
        trades = self._today_trades
        missed = list({
            event.metadata.event_id: event
            for event in self._today_missed
        }.values())

        win_count = sum(1 for t in trades if t.pnl > 0)
        loss_count = sum(1 for t in trades if t.pnl <= 0)
        gross_pnl = sum(t.pnl + t.commission for t in trades)
        net_pnl = sum(t.pnl for t in trades)

        # Max drawdown from equity history
        max_dd = 0.0
        peak = 0.0
        for _, eq in self._equity_history:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100
                max_dd = max(max_dd, dd)

        # Process quality average
        scores = [t.process_quality_score for t in trades]
        avg_quality = sum(scores) / len(scores) if scores else 0.0

        # Root cause distribution
        rc_dist: dict[str, int] = defaultdict(int)
        for t in trades:
            for rc in t.root_causes:
                rc_dist[rc] += 1

        # Missed opportunities that would have won
        missed_won = sum(
            1 for m in missed
            if m.would_have_hit_tp is True
        )

        # Per-strategy summary
        per_strat: dict[str, dict] = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
        for t in trades:
            sid = t.metadata.strategy_id
            per_strat[sid]["trades"] += 1
            per_strat[sid]["pnl"] += t.pnl

        # Rolling metrics placeholder (need 30d equity history for proper calc)
        sharpe_30d = self._rolling_sharpe(30)
        sortino_30d = self._rolling_sortino(30)

        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id="portfolio",
            exchange_ts=datetime.now(timezone.utc),
            event_type="daily_snapshot",
            payload_key=date_str,
        )

        snapshot = DailySnapshot(
            metadata=metadata,
            date=date_str,
            total_trades=len(trades),
            win_count=win_count,
            loss_count=loss_count,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            max_drawdown_pct=max_dd,
            sharpe_rolling_30d=sharpe_30d,
            sortino_rolling_30d=sortino_30d,
            calmar_rolling_30d=0.0,
            exposure_pct=0.0,
            missed_count=len(missed),
            missed_would_have_won=missed_won,
            avg_process_quality=avg_quality,
            root_cause_distribution=dict(rc_dist),
            per_strategy_summary=dict(per_strat),
        )

        # Reset daily accumulators
        self._today_trades = []
        self._today_missed = []

        return snapshot

    def _rolling_sharpe(self, days: int) -> float:
        """Compute rolling Sharpe from equity history."""
        if len(self._equity_history) < 2:
            return 0.0

        # Get daily returns from equity snapshots
        daily: dict[str, float] = {}
        for ts, eq in self._equity_history:
            day = ts.strftime("%Y-%m-%d")
            daily[day] = eq

        sorted_days = sorted(daily.keys())[-days:]
        if len(sorted_days) < 2:
            return 0.0

        values = [daily[d] for d in sorted_days]
        returns = [(values[i] / values[i - 1]) - 1.0 for i in range(1, len(values))]

        if not returns:
            return 0.0

        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.0

        if std == 0:
            return 0.0

        return (mean_r / std) * math.sqrt(365)

    def _rolling_sortino(self, days: int) -> float:
        """Compute rolling Sortino from equity history."""
        if len(self._equity_history) < 2:
            return 0.0

        daily: dict[str, float] = {}
        for ts, eq in self._equity_history:
            day = ts.strftime("%Y-%m-%d")
            daily[day] = eq

        sorted_days = sorted(daily.keys())[-days:]
        if len(sorted_days) < 2:
            return 0.0

        values = [daily[d] for d in sorted_days]
        returns = [(values[i] / values[i - 1]) - 1.0 for i in range(1, len(values))]

        if not returns:
            return 0.0

        mean_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 99.0 if mean_r > 0 else 0.0

        down_var = sum(r ** 2 for r in downside) / len(downside)
        down_std = math.sqrt(down_var) if down_var > 0 else 0.0

        if down_std == 0:
            return 0.0

        return (mean_r / down_std) * math.sqrt(365)
