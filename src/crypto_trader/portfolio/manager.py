"""Portfolio manager — synchronous rule checker for multi-strategy coordination."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import structlog

from crypto_trader.core.models import Side
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.state import OpenRisk, PortfolioState

log = structlog.get_logger()


@dataclass(frozen=True)
class PortfolioRuleResult:
    """Result of a portfolio entry check."""

    approved: bool
    denial_reason: str | None = None
    size_multiplier: float = 1.0


class PortfolioManager:
    """Evaluates portfolio-level risk rules before allowing new entries.

    Rule sequence (9 checks, evaluated in order):
      1. Strategy enabled
      2. Max total positions
      3. Per-strategy max concurrent
      4. Heat cap (total open risk)
      5. Directional cap (with priority reservation)
      6. Symbol exposure cap / collision
      7. Portfolio daily stop
      8. Per-strategy daily stop
      9. Drawdown tiers (cascading size multiplier)

    Any rule can deny entry. Drawdown tiers may reduce size via multiplier.
    """

    def __init__(self, config: PortfolioConfig, state: PortfolioState) -> None:
        self.config = config
        self.state = state

    def check_entry(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side,
        new_risk_R: float,
    ) -> PortfolioRuleResult:
        """Check whether a new entry is allowed by portfolio rules.

        Returns PortfolioRuleResult with approved=True and any size multiplier,
        or approved=False with denial_reason.
        """
        cfg = self.config
        state = self.state

        # Rule 1: Strategy enabled
        alloc = cfg.get_strategy(strategy_id)
        if alloc is None:
            return PortfolioRuleResult(False, f"strategy '{strategy_id}' not in portfolio config")
        if not alloc.enabled:
            return PortfolioRuleResult(False, f"strategy '{strategy_id}' disabled")

        # Rule 2: Max total positions
        if state.total_positions() >= cfg.max_total_positions:
            return PortfolioRuleResult(False, "max_total_positions reached")

        # Rule 3: Per-strategy max concurrent
        if state.strategy_position_count(strategy_id) >= alloc.max_concurrent:
            return PortfolioRuleResult(False, f"strategy '{strategy_id}' max_concurrent reached")

        # Rule 4: Heat cap (total open risk)
        if state.total_heat_R() + new_risk_R > cfg.heat_cap_R:
            return PortfolioRuleResult(False, "heat_cap_R exceeded")

        # Rule 5: Directional cap (with priority reservation)
        dir_risk = state.directional_risk_R(direction) + new_risk_R
        effective_cap = cfg.directional_cap_R
        if cfg.priority_headroom_R > 0 and alloc.priority >= cfg.priority_reserve_threshold:
            remaining = cfg.directional_cap_R - state.directional_risk_R(direction)
            if remaining <= cfg.priority_headroom_R:
                return PortfolioRuleResult(
                    False,
                    f"directional_cap_R headroom reserved (remaining={remaining:.2f}R, "
                    f"headroom={cfg.priority_headroom_R:.2f}R)",
                )
        if dir_risk > effective_cap:
            return PortfolioRuleResult(False, "directional_cap_R exceeded")

        # Rule 6: Symbol exposure cap / collision
        deny = self._check_symbol_collision(strategy_id, symbol, direction, new_risk_R)
        if deny is not None:
            return deny

        # Rule 7: Portfolio daily stop
        if state.portfolio_daily_pnl_R <= -cfg.portfolio_daily_stop_R:
            return PortfolioRuleResult(False, "portfolio_daily_stop_R hit")

        # Rule 8: Per-strategy daily stop
        if state.strategy_daily_pnl_R(strategy_id) <= -alloc.daily_stop_R:
            return PortfolioRuleResult(False, f"strategy '{strategy_id}' daily_stop_R hit")

        # Rule 9: Drawdown tiers (cascading multiplier)
        multiplier = self._dd_multiplier()
        if multiplier <= 0.0:
            return PortfolioRuleResult(False, "drawdown tier blocks all entries")

        log.debug(
            "portfolio.entry_approved",
            strategy=strategy_id,
            symbol=symbol,
            direction=direction.value,
            risk_R=new_risk_R,
            multiplier=multiplier,
        )
        return PortfolioRuleResult(True, size_multiplier=multiplier)

    def register_entry(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side,
        risk_R: float,
        entry_time=None,
    ) -> None:
        """Record a new open risk after entry fill."""
        self.state.add_risk(OpenRisk(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            risk_R=risk_R,
            entry_time=entry_time,
        ))
        log.debug(
            "portfolio.entry_registered",
            strategy=strategy_id,
            symbol=symbol,
            direction=direction.value,
            risk_R=risk_R,
            total_heat=self.state.total_heat_R(),
        )

    def register_exit(
        self,
        strategy_id: str,
        symbol: str,
        pnl_R: float,
    ) -> None:
        """Remove an open risk and record daily P&L."""
        removed = self.state.remove_risk(strategy_id, symbol)
        if removed is None:
            log.warning(
                "portfolio.exit_no_matching_risk",
                strategy=strategy_id,
                symbol=symbol,
            )

        # Update daily P&L
        current = self.state.daily_pnl_R.get(strategy_id, 0.0)
        self.state.daily_pnl_R[strategy_id] = current + pnl_R
        self.state.portfolio_daily_pnl_R += pnl_R

        log.debug(
            "portfolio.exit_registered",
            strategy=strategy_id,
            symbol=symbol,
            pnl_R=pnl_R,
            total_heat=self.state.total_heat_R(),
        )

    def update_equity(self, equity: float) -> None:
        """Update portfolio equity (call periodically or after fills)."""
        self.state.update_equity(equity)

    def maybe_reset_daily(self, today: date) -> None:
        """Reset daily counters if the day has changed."""
        if self.state.current_day != today:
            self.state.reset_daily(today)

    def _check_symbol_collision(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side,
        new_risk_R: float,
    ) -> PortfolioRuleResult | None:
        """Check symbol collision rules. Returns denial result or None if OK."""
        mode = self.config.symbol_collision

        if mode == "allow":
            return None

        # Check if another strategy has an open risk on this symbol
        other_risks = [
            r for r in self.state.open_risks
            if r.symbol == symbol and r.strategy_id != strategy_id
        ]

        if mode == "block" and other_risks:
            blockers = ", ".join(r.strategy_id for r in other_risks)
            return PortfolioRuleResult(
                False,
                f"symbol_collision=block: {symbol} already held by {blockers}",
            )

        if mode == "cap":
            current_sym_risk = self.state.symbol_risk_R(symbol, direction)
            if current_sym_risk + new_risk_R > self.config.symbol_exposure_cap_R:
                return PortfolioRuleResult(
                    False,
                    f"symbol_exposure_cap_R exceeded for {symbol} {direction.value} "
                    f"(current={current_sym_risk:.2f}R + new={new_risk_R:.2f}R "
                    f"> cap={self.config.symbol_exposure_cap_R:.2f}R)",
                )

        return None

    def _dd_multiplier(self) -> float:
        """Compute sizing multiplier from drawdown tiers."""
        dd = self.state.dd_pct()
        multiplier = 1.0
        for threshold, mult in self.config.dd_tiers:
            if dd >= threshold:
                multiplier = mult
            else:
                break
        return multiplier
