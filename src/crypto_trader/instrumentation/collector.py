"""InstrumentationCollector — accumulates gate decisions and market context per bar cycle."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from crypto_trader.instrumentation.pipeline_tracker import PipelineTracker
from crypto_trader.instrumentation.types import (
    EventMetadata,
    FilterDecision,
    InstrumentedTradeEvent,
    MarketContext,
    MissedOpportunityEvent,
    SignalFactor,
)

if TYPE_CHECKING:
    from crypto_trader.core.models import Trade
    from crypto_trader.instrumentation.emitter import EventEmitter


class InstrumentationCollector:
    """Accumulates gate decisions and market context per bar cycle.

    Zero I/O in hot path — everything stored in-memory lists.
    """

    def __init__(self, strategy_id: str, bot_id: str = "") -> None:
        self._strategy_id = strategy_id
        self._bot_id = bot_id

        # Pipeline funnel tracker
        self._pipeline = PipelineTracker(strategy_id)

        # Per-bar-cycle accumulator (reset each primary TF bar)
        self._current_decisions: dict[str, list[FilterDecision]] = {}
        self._current_context: dict[str, MarketContext] = {}
        self._current_signal_factors: dict[str, list[SignalFactor]] = {}
        self._current_bar_close: dict[str, float] = {}

        # Per-position accumulator (persists across bars until trade closes)
        self._entry_decisions: dict[str, list[FilterDecision]] = {}
        self._entry_context: dict[str, MarketContext] = {}
        self._entry_signal_factors: dict[str, list[SignalFactor]] = {}
        self._entry_config: dict[str, dict] = {}
        self._entry_sizing_inputs: dict[str, dict] = {}
        self._entry_portfolio_state: dict[str, dict | None] = {}
        self._entry_signal_strength: dict[str, float] = {}

        # Missed opportunities buffer (flushed by emitter)
        self._missed_buffer: list[MissedOpportunityEvent] = []

        # Emitter reference (set by engine/runner)
        self._emitter: EventEmitter | None = None

    @property
    def emitter(self) -> EventEmitter | None:
        return self._emitter

    @emitter.setter
    def emitter(self, value: EventEmitter) -> None:
        self._emitter = value

    @property
    def pipeline(self) -> PipelineTracker:
        """Read-only access to the pipeline tracker."""
        return self._pipeline

    def begin_bar(self, sym: str, bar_close: float = 0.0) -> None:
        """Reset per-bar accumulator for this symbol."""
        self._current_decisions[sym] = []
        self._current_signal_factors[sym] = []
        self._current_bar_close[sym] = bar_close
        self._pipeline.record_bar(sym)

    def record_gate(
        self,
        sym: str,
        gate_name: str,
        passed: bool,
        reason: str = "",
        threshold: float | None = None,
        actual_value: float | None = None,
        context: dict | None = None,
    ) -> None:
        """Record a gate evaluation with filter-level detail."""
        margin = None
        if threshold is not None and actual_value is not None and threshold != 0:
            margin = (actual_value - threshold) / abs(threshold) * 100

        self._current_decisions.setdefault(sym, []).append(
            FilterDecision(
                filter_name=gate_name,
                passed=passed,
                threshold=threshold,
                actual_value=actual_value,
                margin_pct=margin,
                reason=reason,
                context=context or {},
            )
        )
        self._pipeline.record_gate(sym, gate_name, passed)

    def snapshot_context(self, sym: str, indicators: object, **kwargs) -> None:
        """Capture market state from indicator snapshot + strategy-specific data."""
        self._current_context[sym] = MarketContext(
            atr=getattr(indicators, "atr", 0.0),
            adx=getattr(indicators, "adx", 0.0),
            rsi=getattr(indicators, "rsi", None),
            ema_fast=getattr(indicators, "ema_fast", 0.0),
            ema_mid=getattr(indicators, "ema_mid", 0.0),
            ema_slow=getattr(indicators, "ema_slow", 0.0),
            volume_ma=getattr(indicators, "volume_ma", 0.0),
            funding_rate=kwargs.get("funding_rate", 0.0),
            bias_direction=kwargs.get("bias_direction"),
            bias_strength=kwargs.get("bias_strength"),
            regime_tier=kwargs.get("regime_tier"),
            regime_direction=kwargs.get("regime_direction"),
            h4_context_direction=kwargs.get("h4_context_direction"),
            h4_context_strength=kwargs.get("h4_context_strength"),
            setup_grade=kwargs.get("setup_grade"),
            setup_confluences=kwargs.get("setup_confluences", []),
            setup_room_r=kwargs.get("setup_room_r"),
        )

    def record_signal_factor(self, sym: str, factor: str, value: float) -> None:
        """Record a signal factor that drove the entry decision."""
        self._current_signal_factors.setdefault(sym, []).append(
            SignalFactor(factor=factor, value=value)
        )

    def record_entry(
        self,
        sym: str,
        config_dict: dict,
        sizing_inputs: dict,
        portfolio_state: dict | None = None,
        signal_strength: float = 0.0,
    ) -> None:
        """Freeze current gate decisions + context at entry submission time."""
        self._entry_decisions[sym] = list(self._current_decisions.get(sym, []))
        self._entry_context[sym] = self._current_context.get(sym)  # type: ignore[assignment]
        self._entry_signal_factors[sym] = list(
            self._current_signal_factors.get(sym, [])
        )
        self._entry_config[sym] = config_dict
        self._entry_sizing_inputs[sym] = sizing_inputs
        self._entry_portfolio_state[sym] = portfolio_state
        self._entry_signal_strength[sym] = signal_strength

    def on_trade_closed(
        self,
        sym: str,
        trade: Trade,
        process_score: int = 100,
        root_causes: list[str] | None = None,
    ) -> InstrumentedTradeEvent:
        """Build full instrumented event from accumulated data + trade outcome."""
        self._pipeline.record_trade_closed(sym)
        decisions = self._entry_decisions.pop(sym, [])
        ctx = self._entry_context.pop(sym, None)
        factors = self._entry_signal_factors.pop(sym, [])
        config = self._entry_config.pop(sym, {})
        sizing = self._entry_sizing_inputs.pop(sym, {})
        portfolio = self._entry_portfolio_state.pop(sym, None)
        strength = self._entry_signal_strength.pop(sym, 0.0)

        passed = [d.filter_name for d in decisions if d.passed]
        active = [d.filter_name for d in decisions]

        exit_eff = None
        if (
            trade.r_multiple is not None
            and trade.mfe_r is not None
            and trade.r_multiple > 0
            and trade.mfe_r > 0
        ):
            exit_eff = trade.r_multiple / trade.mfe_r

        pnl_pct = 0.0
        if trade.entry_price > 0 and trade.qty > 0:
            notional = trade.entry_price * trade.qty
            if notional > 0:
                pnl_pct = trade.net_pnl / notional * 100

        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id=self._strategy_id,
            exchange_ts=trade.exit_time,
            event_type="trade",
            payload_key=trade.trade_id,
        )

        return InstrumentedTradeEvent(
            metadata=metadata,
            trade_id=trade.trade_id,
            pair=trade.symbol,
            side=trade.direction.value,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            position_size=trade.qty,
            pnl=trade.net_pnl,
            pnl_pct=pnl_pct,
            r_multiple=trade.r_multiple,
            commission=trade.commission,
            funding_paid=trade.funding_paid,
            entry_signal=trade.confirmation_type or "",
            entry_signal_strength=strength,
            setup_grade=trade.setup_grade.value if trade.setup_grade else "",
            exit_reason=trade.exit_reason,
            confluences=list(trade.confluences_used or []),
            entry_method=trade.entry_method or "",
            signal_factors=factors,
            filter_decisions=decisions,
            passed_filters=passed,
            active_filters=active,
            market_context=ctx,
            mfe_r=trade.mfe_r,
            mae_r=trade.mae_r,
            exit_efficiency=exit_eff,
            process_quality_score=process_score,
            root_causes=root_causes or [],
            strategy_params_at_entry=config,
            sizing_inputs=sizing,
            portfolio_state_at_entry=portfolio,
        )

    def end_bar(self, sym: str) -> None:
        """Finalize bar cycle. Detect missed opportunities from gate failures.

        A signal is "missed" when the pipeline progressed past the setup gate
        but was blocked by a downstream gate. If setup itself failed, there
        was no actionable signal to miss.
        """
        decisions = self._current_decisions.get(sym, [])
        if not decisions:
            return

        # Find setup gate
        setup_idx = None
        for i, d in enumerate(decisions):
            if d.filter_name == "setup":
                setup_idx = i
                break

        if setup_idx is None:
            return  # Pipeline didn't reach setup evaluation

        if not decisions[setup_idx].passed:
            return  # No setup found — not a missed signal

        # Find first failing gate AFTER setup
        blocker = None
        for d in decisions[setup_idx + 1 :]:
            if not d.passed:
                blocker = d
                break

        if blocker is None:
            return  # All gates passed — entry was submitted, not a miss

        # Build MissedOpportunityEvent
        ctx = self._current_context.get(sym)
        grade_str = ctx.setup_grade if ctx else ""
        room_r = ctx.setup_room_r if ctx else 0.0

        blocker_idx = decisions.index(blocker)
        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id=self._strategy_id,
            exchange_ts=datetime.now(timezone.utc),
            event_type="missed_opportunity",
            payload_key=f"{sym}_{blocker.filter_name}",
        )

        missed = MissedOpportunityEvent(
            metadata=metadata,
            pair=sym,
            signal=f"{self._strategy_id}_{grade_str}" if grade_str else self._strategy_id,
            signal_strength=room_r or 0.0,
            blocked_by=blocker.filter_name,
            block_reason=blocker.reason,
            margin_pct=blocker.margin_pct,
            hypothetical_entry=self._current_bar_close.get(sym, 0.0),
            market_context=ctx,
            filter_decisions=list(decisions[: blocker_idx + 1]),
            backfill_status="pending",
        )
        self._missed_buffer.append(missed)

        # Auto-emit if emitter is wired
        if self._emitter is not None:
            self._emitter.emit_missed(missed)

    def flush_missed(self) -> list[MissedOpportunityEvent]:
        """Return and clear the missed opportunity buffer."""
        buf = self._missed_buffer
        self._missed_buffer = []
        return buf
