"""Instrumentation event types for trading assistant integration."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Root cause taxonomy (21 values matching reference)
# ---------------------------------------------------------------------------

ROOT_CAUSE_TAXONOMY = frozenset({
    "regime_mismatch",
    "weak_signal",
    "strong_signal",
    "late_entry",
    "early_exit",
    "premature_stop",
    "slippage_spike",
    "good_execution",
    "filter_blocked_good",
    "filter_saved_bad",
    "risk_cap_hit",
    "data_gap",
    "order_reject",
    "latency_spike",
    "correlation_crowding",
    "funding_adverse",
    "funding_favorable",
    "regime_aligned",
    "normal_loss",
    "normal_win",
    "exceptional_win",
})


# ---------------------------------------------------------------------------
# Filter / gate decision
# ---------------------------------------------------------------------------

@dataclass
class FilterDecision:
    """Per-gate evaluation detail."""
    filter_name: str
    passed: bool
    threshold: float | None = None
    actual_value: float | None = None
    margin_pct: float | None = None
    reason: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Market context snapshot
# ---------------------------------------------------------------------------

@dataclass
class MarketContext:
    """Indicator snapshot at signal time."""
    atr: float = 0.0
    adx: float = 0.0
    rsi: float | None = None
    ema_fast: float = 0.0
    ema_mid: float = 0.0
    ema_slow: float = 0.0
    volume_ma: float = 0.0
    funding_rate: float = 0.0
    # Strategy-specific (all optional)
    bias_direction: str | None = None
    bias_strength: float | None = None
    regime_tier: str | None = None
    regime_direction: str | None = None
    h4_context_direction: str | None = None
    h4_context_strength: str | None = None
    setup_grade: str | None = None
    setup_confluences: list[str] = field(default_factory=list)
    setup_room_r: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Signal factor
# ---------------------------------------------------------------------------

@dataclass
class SignalFactor:
    """What drove the entry signal."""
    factor: str
    value: float

    def to_dict(self) -> dict:
        return {"factor": self.factor, "value": self.value}


# ---------------------------------------------------------------------------
# Event metadata
# ---------------------------------------------------------------------------

@dataclass
class EventMetadata:
    """Deterministic event identity."""
    event_id: str
    bot_id: str
    strategy_id: str
    exchange_timestamp: datetime
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    @staticmethod
    def create(
        bot_id: str,
        strategy_id: str,
        exchange_ts: datetime,
        event_type: str,
        payload_key: str,
    ) -> EventMetadata:
        raw = f"{bot_id}|{exchange_ts.isoformat()}|{event_type}|{payload_key}"
        event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return EventMetadata(
            event_id=event_id,
            bot_id=bot_id,
            strategy_id=strategy_id,
            exchange_timestamp=exchange_ts,
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "bot_id": self.bot_id,
            "strategy_id": self.strategy_id,
            "exchange_timestamp": self.exchange_timestamp.isoformat(),
            "trace_id": self.trace_id,
        }


# ---------------------------------------------------------------------------
# Instrumented trade event
# ---------------------------------------------------------------------------

@dataclass
class InstrumentedTradeEvent:
    """Full trade record with context."""
    metadata: EventMetadata
    # Identity
    trade_id: str = ""
    pair: str = ""
    side: str = ""
    # Timing
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Prices & P&L
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float | None = None
    commission: float = 0.0
    funding_paid: float = 0.0
    # Signal
    entry_signal: str = ""
    entry_signal_strength: float = 0.0
    setup_grade: str = ""
    exit_reason: str = ""
    confluences: list[str] = field(default_factory=list)
    entry_method: str = ""
    signal_factors: list[SignalFactor] = field(default_factory=list)
    # Gate pipeline
    filter_decisions: list[FilterDecision] = field(default_factory=list)
    passed_filters: list[str] = field(default_factory=list)
    active_filters: list[str] = field(default_factory=list)
    # Market context at entry
    market_context: MarketContext | None = None
    # Excursion
    mfe_r: float | None = None
    mae_r: float | None = None
    exit_efficiency: float | None = None
    # Post-exit tracking (backfilled in live mode)
    post_exit_1h_move_pct: float | None = None
    post_exit_4h_move_pct: float | None = None
    # Quality
    process_quality_score: int = 100
    root_causes: list[str] = field(default_factory=list)
    # Snapshots at entry
    strategy_params_at_entry: dict = field(default_factory=dict)
    sizing_inputs: dict = field(default_factory=dict)
    portfolio_state_at_entry: dict | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "metadata": self.metadata.to_dict(),
            "trade_id": self.trade_id,
            "pair": self.pair,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "r_multiple": self.r_multiple,
            "commission": self.commission,
            "funding_paid": self.funding_paid,
            "entry_signal": self.entry_signal,
            "entry_signal_strength": self.entry_signal_strength,
            "setup_grade": self.setup_grade,
            "exit_reason": self.exit_reason,
            "confluences": self.confluences,
            "entry_method": self.entry_method,
            "signal_factors": [sf.to_dict() for sf in self.signal_factors],
            "filter_decisions": [fd.to_dict() for fd in self.filter_decisions],
            "passed_filters": self.passed_filters,
            "active_filters": self.active_filters,
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "exit_efficiency": self.exit_efficiency,
            "post_exit_1h_move_pct": self.post_exit_1h_move_pct,
            "post_exit_4h_move_pct": self.post_exit_4h_move_pct,
            "process_quality_score": self.process_quality_score,
            "root_causes": self.root_causes,
            "strategy_params_at_entry": self.strategy_params_at_entry,
            "sizing_inputs": self.sizing_inputs,
            "portfolio_state_at_entry": self.portfolio_state_at_entry,
        }
        return d


# ---------------------------------------------------------------------------
# Missed opportunity event
# ---------------------------------------------------------------------------

@dataclass
class MissedOpportunityEvent:
    """Blocked signals with backfill slots."""
    metadata: EventMetadata
    pair: str = ""
    signal: str = ""
    signal_strength: float = 0.0
    blocked_by: str = ""
    block_reason: str = ""
    margin_pct: float | None = None
    hypothetical_entry: float = 0.0
    market_context: MarketContext | None = None
    filter_decisions: list[FilterDecision] = field(default_factory=list)
    # Backfilled later
    outcome_1h: float | None = None
    outcome_4h: float | None = None
    outcome_24h: float | None = None
    would_have_hit_tp: bool | None = None
    would_have_hit_sl: bool | None = None
    backfill_status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "pair": self.pair,
            "signal": self.signal,
            "signal_strength": self.signal_strength,
            "blocked_by": self.blocked_by,
            "block_reason": self.block_reason,
            "margin_pct": self.margin_pct,
            "hypothetical_entry": self.hypothetical_entry,
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "filter_decisions": [fd.to_dict() for fd in self.filter_decisions],
            "outcome_1h": self.outcome_1h,
            "outcome_4h": self.outcome_4h,
            "outcome_24h": self.outcome_24h,
            "would_have_hit_tp": self.would_have_hit_tp,
            "would_have_hit_sl": self.would_have_hit_sl,
            "backfill_status": self.backfill_status,
        }


# ---------------------------------------------------------------------------
# Daily snapshot
# ---------------------------------------------------------------------------

@dataclass
class DailySnapshot:
    """End-of-day aggregate (live mode only)."""
    metadata: EventMetadata
    date: str = ""
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_rolling_30d: float = 0.0
    sortino_rolling_30d: float = 0.0
    calmar_rolling_30d: float = 0.0
    exposure_pct: float = 0.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    avg_process_quality: float = 0.0
    root_cause_distribution: dict[str, int] = field(default_factory=dict)
    per_strategy_summary: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "date": self.date,
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_rolling_30d": self.sharpe_rolling_30d,
            "sortino_rolling_30d": self.sortino_rolling_30d,
            "calmar_rolling_30d": self.calmar_rolling_30d,
            "exposure_pct": self.exposure_pct,
            "missed_count": self.missed_count,
            "missed_would_have_won": self.missed_would_have_won,
            "avg_process_quality": self.avg_process_quality,
            "root_cause_distribution": self.root_cause_distribution,
            "per_strategy_summary": self.per_strategy_summary,
        }


# ---------------------------------------------------------------------------
# Error event
# ---------------------------------------------------------------------------

@dataclass
class ErrorEvent:
    """Error telemetry."""
    metadata: EventMetadata
    error_type: str = ""
    message: str = ""
    stack_trace: str = ""
    severity: str = "low"

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "error_type": self.error_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# Pipeline funnel snapshot
# ---------------------------------------------------------------------------

@dataclass
class PipelineFunnelSnapshot:
    """Periodic pipeline funnel snapshot for a strategy."""
    strategy_id: str
    timestamp: str
    period_start: str
    period_end: str
    funnel: dict = field(default_factory=dict)
    assessment: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Health report snapshot
# ---------------------------------------------------------------------------

@dataclass
class HealthReportSnapshot:
    """Periodic system health report."""
    timestamp: str
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
