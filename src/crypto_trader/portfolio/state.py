"""Portfolio state — mutable shared state for multi-strategy position tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from crypto_trader.core.models import Side


@dataclass
class OpenRisk:
    """A single open risk entry tracked by the portfolio."""

    strategy_id: str
    symbol: str
    direction: Side
    risk_R: float
    entry_time: object = None  # datetime | None


@dataclass
class PortfolioState:
    """Shared mutable state across all strategies in a portfolio.

    Tracks open risks, daily P&L, and equity for portfolio-level rule evaluation.
    """

    equity: float = 0.0
    peak_equity: float = 0.0
    open_risks: list[OpenRisk] = field(default_factory=list)
    daily_pnl_R: dict[str, float] = field(default_factory=dict)  # per-strategy
    portfolio_daily_pnl_R: float = 0.0
    current_day: date | None = None

    def total_heat_R(self) -> float:
        """Total open risk across all strategies."""
        return sum(r.risk_R for r in self.open_risks)

    def directional_risk_R(self, direction: Side) -> float:
        """Total open risk in one direction."""
        return sum(r.risk_R for r in self.open_risks if r.direction == direction)

    def symbol_risk_R(self, symbol: str, direction: Side | None = None) -> float:
        """Total open risk for a symbol, optionally filtered by direction."""
        total = 0.0
        for r in self.open_risks:
            if r.symbol == symbol:
                if direction is None or r.direction == direction:
                    total += r.risk_R
        return total

    def strategy_position_count(self, strategy_id: str) -> int:
        """Count open positions for a specific strategy."""
        return sum(1 for r in self.open_risks if r.strategy_id == strategy_id)

    def total_positions(self) -> int:
        """Count all open positions across all strategies."""
        return len(self.open_risks)

    def dd_pct(self) -> float:
        """Current drawdown as fraction of peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def strategy_daily_pnl_R(self, strategy_id: str) -> float:
        """Get daily P&L in R-units for a strategy."""
        return self.daily_pnl_R.get(strategy_id, 0.0)

    def reset_daily(self, new_day: date) -> None:
        """Reset daily P&L counters for a new trading day."""
        self.daily_pnl_R.clear()
        self.portfolio_daily_pnl_R = 0.0
        self.current_day = new_day

    def add_risk(self, risk: OpenRisk) -> None:
        """Register a new open risk."""
        self.open_risks.append(risk)

    def remove_risk(self, strategy_id: str, symbol: str) -> OpenRisk | None:
        """Remove and return the first matching open risk."""
        for i, r in enumerate(self.open_risks):
            if r.strategy_id == strategy_id and r.symbol == symbol:
                return self.open_risks.pop(i)
        return None

    def update_equity(self, equity: float) -> None:
        """Update equity and peak equity."""
        self.equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "open_risks": [
                {
                    "strategy_id": r.strategy_id,
                    "symbol": r.symbol,
                    "direction": r.direction.value,
                    "risk_R": r.risk_R,
                    "entry_time": str(r.entry_time) if r.entry_time else None,
                }
                for r in self.open_risks
            ],
            "daily_pnl_R": dict(self.daily_pnl_R),
            "portfolio_daily_pnl_R": self.portfolio_daily_pnl_R,
            "current_day": str(self.current_day) if self.current_day else None,
        }
