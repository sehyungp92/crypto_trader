"""Portfolio phase-auto round 2.

This round optimizes the three latest strategy configs as a portfolio.  It is
intentionally conservative about inference: broad replay sweeps are scored on a
development window plus a forward holdout, and the selected policy is validated
with the real portfolio backtester before artifacts are written.  Independent
work is capped at two workers to keep the run reproducible and resource-light.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.runner import run as run_individual
from crypto_trader.core.models import Side, Trade
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
from crypto_trader.portfolio.backtest_runner import run_portfolio_backtest
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.trend.config import TrendConfig


logging.basicConfig(level=logging.ERROR)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))

SYMBOLS = ["BTC", "ETH", "SOL"]
STRATEGIES = ["momentum", "trend", "breakout"]
INITIAL_EQUITY = 25_000.0
DATA_DIR = ROOT / "data"
ROUND_NAME = "round_2"
MAX_WORKERS = 2
MAX_SCORE_COMPONENTS = 7
SCORE_COMPONENT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("dev_return", 0.31),
    ("dev_frequency", 0.19),
    ("dev_profit_factor", 0.17),
    ("dev_drawdown_resilience", 0.11),
    ("holdout_return", 0.10),
    ("holdout_profit_factor", 0.05),
    ("strategy_balance", 0.07),
)
ACTUAL_VALIDATION_COMPONENT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("full_return", 0.32),
    ("full_frequency", 0.18),
    ("full_profit_factor", 0.16),
    ("full_drawdown_resilience", 0.12),
    ("holdout_return", 0.10),
    ("holdout_profit_factor", 0.05),
    ("holdout_drawdown_resilience", 0.07),
)
ROUND_DIR = ROOT / "output" / "portfolio" / ROUND_NAME
RECOMMENDED_CONFIG_DIR = ROUND_DIR / "recommended_strategy_configs"

# The saved diagnostics end on 2026-04-18.  2026-04-19..2026-04-30 is a
# forward holdout from the perspective of those latest round artifacts.
DEV_START = date(2025, 12, 1)
DEV_END = date(2026, 4, 18)
HOLDOUT_START = date(2026, 4, 19)
HOLDOUT_END = date(2026, 4, 30)
FULL_START = DEV_START
FULL_END = HOLDOUT_END


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: date
    end: date


DEV_WINDOW = WindowSpec("development", DEV_START, DEV_END)
HOLDOUT_WINDOW = WindowSpec("forward_holdout", HOLDOUT_START, HOLDOUT_END)
FULL_WINDOW = WindowSpec("full_refreshed", FULL_START, FULL_END)


@dataclass(frozen=True)
class PolicyDelta:
    name: str
    phase: int
    thesis: str
    strategy_mutations: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_scales: dict[str, float] = field(default_factory=dict)
    portfolio_overrides: dict[str, Any] = field(default_factory=dict)
    filter_rules: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PortfolioPolicy:
    name: str
    strategy_mutations: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_scales: dict[str, float] = field(
        default_factory=lambda: {sid: 1.0 for sid in STRATEGIES}
    )
    portfolio_overrides: dict[str, Any] = field(default_factory=dict)
    filter_rules: tuple[dict[str, Any], ...] = ()
    accepted_deltas: tuple[str, ...] = ()


@dataclass
class ReplayMetrics:
    trades: int = 0
    filtered: int = 0
    blocked: int = 0
    net_pnl: float = 0.0
    net_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    total_r: float = 0.0
    per_strategy_trades: dict[str, int] = field(default_factory=dict)
    block_reasons: dict[str, int] = field(default_factory=dict)
    filter_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "filtered": self.filtered,
            "blocked": self.blocked,
            "net_pnl": self.net_pnl,
            "net_return_pct": self.net_return_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_r": self.total_r,
            "per_strategy_trades": dict(self.per_strategy_trades),
            "block_reasons": dict(self.block_reasons),
            "filter_reasons": dict(self.filter_reasons),
        }


@dataclass
class ReplayEvaluation:
    policy_name: str
    score: float
    rejected: bool
    reject_reason: str
    development: ReplayMetrics
    holdout: ReplayMetrics
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "score": self.score,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "development": self.development.to_dict(),
            "holdout": self.holdout.to_dict(),
            "score_components": dict(self.score_components),
        }


@dataclass
class ActualValidation:
    policy_name: str
    score: float
    rejected: bool
    reject_reason: str
    full: dict[str, float]
    holdout: dict[str, float]
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "score": self.score,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "full": dict(self.full),
            "holdout": dict(self.holdout),
            "score_components": dict(self.score_components),
        }


@dataclass
class TimelineEntry:
    strategy_id: str
    trade: Trade
    risk_R: float
    pnl_scale: float


@dataclass
class OpenReplayRisk:
    entry: TimelineEntry
    multiplier: float

    @property
    def risk_R(self) -> float:
        return self.entry.risk_R * self.multiplier


def _bt_config(window: WindowSpec) -> BacktestConfig:
    return BacktestConfig(
        symbols=list(SYMBOLS),
        start_date=window.start,
        end_date=window.end,
        initial_equity=INITIAL_EQUITY,
        taker_fee_bps=4.5,
        maker_fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        apply_funding=True,
        warmup_days=60,
    )


def _load_strategy_config(strategy_id: str) -> Any:
    path = ROOT / "output" / strategy_id / "round_3" / "optimized_config.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)["strategy"]
    if strategy_id == "momentum":
        return MomentumConfig.from_dict(payload)
    if strategy_id == "trend":
        return TrendConfig.from_dict(payload)
    if strategy_id == "breakout":
        return BreakoutConfig.from_dict(payload)
    raise ValueError(f"Unknown strategy: {strategy_id}")


def _load_base_configs() -> dict[str, Any]:
    return {sid: _load_strategy_config(sid) for sid in STRATEGIES}


def _scale(value: float, multiplier: float, cap: float) -> float:
    return round(min(value * multiplier, cap), 8)


def _risk_scale_mutations(strategy_id: str, cfg: Any, multiplier: float) -> dict[str, Any]:
    if abs(multiplier - 1.0) < 1e-9:
        return {}

    if strategy_id == "momentum":
        risk = cfg.risk
        return {
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.030),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.025),
            "risk.max_gross_risk": _scale(risk.max_gross_risk, max(1.0, multiplier), 0.060),
            "risk.max_correlated_risk": _scale(risk.max_correlated_risk, max(1.0, multiplier), 0.040),
        }

    if strategy_id == "trend":
        risk = cfg.risk
        limits = cfg.limits
        return {
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.028),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.028),
            "risk.max_risk_pct": _scale(risk.max_risk_pct, max(1.0, multiplier), 0.030),
            "limits.max_correlated_risk_pct": _scale(
                limits.max_correlated_risk_pct, max(1.0, multiplier), 0.070
            ),
        }

    if strategy_id == "breakout":
        risk = cfg.risk
        limits = cfg.limits
        return {
            "risk.risk_pct_a_plus": _scale(risk.risk_pct_a_plus, multiplier, 0.018),
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.018),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.032),
            "risk.max_risk_pct": _scale(risk.max_risk_pct, max(1.0, multiplier), 0.033),
            "limits.max_correlated_risk_pct": _scale(
                limits.max_correlated_risk_pct, max(1.0, multiplier), 0.030
            ),
        }

    raise ValueError(f"Unknown strategy: {strategy_id}")


def _apply_policy_configs(policy: PortfolioPolicy, base_configs: dict[str, Any]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for sid, cfg in base_configs.items():
        risk_mut = _risk_scale_mutations(sid, cfg, policy.risk_scales.get(sid, 1.0))
        explicit = policy.strategy_mutations.get(sid, {})
        mutations = merge_mutations(risk_mut, explicit)
        new_cfg = apply_mutations(cfg, mutations) if mutations else apply_mutations(cfg, {})
        new_cfg.symbols = list(SYMBOLS)
        configs[sid] = new_cfg
    return configs


def _base_allocations(policy: PortfolioPolicy) -> tuple[StrategyAllocation, ...]:
    # Higher priority strategies keep access to directional headroom when a
    # headroom candidate is active: trend has the broadest sample, breakout the
    # highest expectancy but smaller sample, momentum the newest M15 sample.
    priority = {"trend": 0, "breakout": 1, "momentum": 2}
    return tuple(
        StrategyAllocation(
            strategy_id=sid,
            base_risk_pct=0.01 * policy.risk_scales.get(sid, 1.0),
            max_concurrent=5 if sid == "trend" else 3,
            daily_stop_R=4.0 if sid == "trend" else 3.0,
            priority=priority[sid],
        )
        for sid in STRATEGIES
    )


def _portfolio_config(policy: PortfolioPolicy) -> PortfolioConfig:
    kwargs = {
        "initial_equity": INITIAL_EQUITY,
        "strategies": _base_allocations(policy),
        "heat_cap_R": 6.0,
        "directional_cap_R": 4.0,
        "portfolio_daily_stop_R": 5.0,
        "max_total_positions": 9,
        "dd_tiers": (
            (0.08, 1.00),
            (0.12, 0.50),
            (0.15, 0.25),
            (1.00, 0.00),
        ),
        "symbol_collision": "cap",
        "symbol_exposure_cap_R": 3.0,
        "priority_headroom_R": 0.0,
        "priority_reserve_threshold": 0,
    }
    kwargs.update(policy.portfolio_overrides)
    if "dd_tiers" in kwargs:
        kwargs["dd_tiers"] = tuple(tuple(x) for x in kwargs["dd_tiers"])
    return PortfolioConfig(**kwargs)


def _merge_policy(base: PortfolioPolicy, delta: PolicyDelta) -> PortfolioPolicy:
    strategy_mutations = {
        sid: dict(muts) for sid, muts in base.strategy_mutations.items()
    }
    for sid, muts in delta.strategy_mutations.items():
        strategy_mutations[sid] = merge_mutations(strategy_mutations.get(sid, {}), muts)

    risk_scales = dict(base.risk_scales)
    for sid, mult in delta.risk_scales.items():
        risk_scales[sid] = round(risk_scales.get(sid, 1.0) * mult, 8)

    overrides = dict(base.portfolio_overrides)
    overrides.update(delta.portfolio_overrides)

    return PortfolioPolicy(
        name=f"{base.name}+{delta.name}" if base.name != "baseline" else delta.name,
        strategy_mutations=strategy_mutations,
        risk_scales=risk_scales,
        portfolio_overrides=overrides,
        filter_rules=(*base.filter_rules, *delta.filter_rules),
        accepted_deltas=(*base.accepted_deltas, delta.name),
    )


def _rule_direction(value: str | Side | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Side):
        return value.value
    return str(value).upper()


def _filter_reason(entry: TimelineEntry, rules: tuple[dict[str, Any], ...]) -> str | None:
    trade = entry.trade
    for rule in rules:
        sid = rule.get("strategy_id")
        if sid not in (None, entry.strategy_id):
            continue

        kind = rule["kind"]
        if kind == "symbol_direction":
            if trade.symbol == rule["symbol"] and trade.direction.value == _rule_direction(rule["direction"]):
                return rule.get("reason", f"{entry.strategy_id}_{trade.symbol}_{trade.direction.value}")

        elif kind == "entry_hour":
            hours = set(rule.get("hours", []))
            if trade.entry_time.hour in hours:
                return rule.get("reason", f"{entry.strategy_id}_hour_{trade.entry_time.hour}")

        elif kind == "confluence_lte":
            confluence_count = len(trade.confluences_used or [])
            if confluence_count <= int(rule["threshold"]):
                return rule.get("reason", f"{entry.strategy_id}_confluence_lte_{rule['threshold']}")

        elif kind == "confirmation_symbol":
            if trade.symbol == rule["symbol"] and trade.confirmation_type == rule["confirmation"]:
                return rule.get(
                    "reason",
                    f"{entry.strategy_id}_{trade.symbol}_{trade.confirmation_type}",
                )

    return None


def _build_timeline(
    trade_lists: dict[str, list[Trade]],
    policy: PortfolioPolicy,
) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    for sid, trades in trade_lists.items():
        scale = policy.risk_scales.get(sid, 1.0)
        for trade in trades:
            entries.append(TimelineEntry(
                strategy_id=sid,
                trade=trade,
                risk_R=scale,
                pnl_scale=scale,
            ))
    entries.sort(key=lambda item: item.trade.entry_time)
    return entries


def _dd_multiplier(dd: float, tiers: tuple[tuple[float, float], ...]) -> float:
    multiplier = 1.0
    for threshold, mult in tiers:
        if dd >= threshold:
            multiplier = mult
        else:
            break
    return multiplier


def _close_before(
    open_risks: list[OpenReplayRisk],
    before: datetime,
    equity_state: dict[str, float],
    daily_pnl: dict[str, float],
    closed_pnls: list[float],
    closed_rs: list[float],
) -> None:
    remaining: list[OpenReplayRisk] = []
    for risk in open_risks:
        trade = risk.entry.trade
        if trade.exit_time is None or trade.exit_time < before:
            r_mult = trade.r_multiple
            if r_mult is None:
                r_mult = trade.realized_r_multiple
            # A few historical Trade objects carry pathological realized-R
            # values from partial-fill accounting.  Portfolio stops should be
            # driven by bounded geometric R, not those artifacts.
            if r_mult is None or abs(r_mult) > 20.0:
                r_mult = 0.0
            pnl = trade.net_pnl * risk.entry.pnl_scale * risk.multiplier
            pnl_r = r_mult * risk.entry.risk_R * risk.multiplier
            equity_state["equity"] += pnl
            equity_state["peak"] = max(equity_state["peak"], equity_state["equity"])
            equity_state["max_dd"] = max(
                equity_state["max_dd"],
                (equity_state["peak"] - equity_state["equity"]) / equity_state["peak"],
            )
            daily_pnl["portfolio"] += pnl_r
            daily_pnl[risk.entry.strategy_id] += pnl_r
            closed_pnls.append(pnl)
            closed_rs.append(pnl_r)
        else:
            remaining.append(risk)
    open_risks[:] = remaining


def _replay_portfolio(
    policy: PortfolioPolicy,
    trade_lists: dict[str, list[Trade]],
) -> ReplayMetrics:
    config = _portfolio_config(policy)
    timeline = _build_timeline(trade_lists, policy)
    open_risks: list[OpenReplayRisk] = []
    equity_state = {
        "equity": INITIAL_EQUITY,
        "peak": INITIAL_EQUITY,
        "max_dd": 0.0,
    }
    current_day: date | None = None
    daily_pnl: defaultdict[str, float] = defaultdict(float)
    accepted_by_strategy: Counter[str] = Counter()
    block_reasons: Counter[str] = Counter()
    filter_reasons: Counter[str] = Counter()
    closed_pnls: list[float] = []
    closed_rs: list[float] = []
    filtered = 0
    blocked = 0

    for entry in timeline:
        trade = entry.trade
        if current_day != trade.entry_time.date():
            current_day = trade.entry_time.date()
            daily_pnl.clear()

        _close_before(open_risks, trade.entry_time, equity_state, daily_pnl, closed_pnls, closed_rs)

        reason = _filter_reason(entry, policy.filter_rules)
        if reason:
            filtered += 1
            filter_reasons[reason] += 1
            continue

        alloc = config.get_strategy(entry.strategy_id)
        if alloc is None or not alloc.enabled:
            blocked += 1
            block_reasons["strategy_disabled"] += 1
            continue

        total_positions = len(open_risks)
        strat_positions = sum(1 for r in open_risks if r.entry.strategy_id == entry.strategy_id)
        heat = sum(r.risk_R for r in open_risks)
        dir_heat = sum(r.risk_R for r in open_risks if r.entry.trade.direction == trade.direction)
        symbol_heat = sum(
            r.risk_R for r in open_risks
            if r.entry.trade.symbol == trade.symbol and r.entry.trade.direction == trade.direction
        )

        reason = ""
        if total_positions >= config.max_total_positions:
            reason = "max_total_positions"
        elif strat_positions >= alloc.max_concurrent:
            reason = f"{entry.strategy_id}_max_concurrent"
        elif heat + entry.risk_R > config.heat_cap_R:
            reason = "heat_cap_R"
        elif dir_heat + entry.risk_R > config.directional_cap_R:
            reason = "directional_cap_R"
        elif config.symbol_collision == "cap" and symbol_heat + entry.risk_R > config.symbol_exposure_cap_R:
            reason = "symbol_exposure_cap_R"
        elif daily_pnl["portfolio"] <= -config.portfolio_daily_stop_R:
            reason = "portfolio_daily_stop_R"
        elif daily_pnl[entry.strategy_id] <= -alloc.daily_stop_R:
            reason = f"{entry.strategy_id}_daily_stop_R"

        dd = (equity_state["peak"] - equity_state["equity"]) / equity_state["peak"]
        multiplier = _dd_multiplier(dd, config.dd_tiers)
        if not reason and multiplier <= 0.0:
            reason = "drawdown_tier_block"

        if reason:
            blocked += 1
            block_reasons[reason] += 1
            continue

        open_risks.append(OpenReplayRisk(entry=entry, multiplier=multiplier))
        accepted_by_strategy[entry.strategy_id] += 1

    _close_before(
        open_risks,
        datetime.max.replace(tzinfo=timezone.utc),
        equity_state,
        daily_pnl,
        closed_pnls,
        closed_rs,
    )

    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    trades = len(closed_pnls)
    return ReplayMetrics(
        trades=trades,
        filtered=filtered,
        blocked=blocked,
        net_pnl=sum(closed_pnls),
        net_return_pct=(sum(closed_pnls) / INITIAL_EQUITY) * 100.0,
        win_rate=(len(wins) / trades * 100.0) if trades else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss else (math.inf if gross_win else 0.0),
        max_drawdown_pct=equity_state["max_dd"] * 100.0,
        total_r=sum(closed_rs),
        per_strategy_trades=dict(accepted_by_strategy),
        block_reasons=dict(block_reasons),
        filter_reasons=dict(filter_reasons),
    )


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def _strategy_balance_score(metrics: ReplayMetrics) -> float:
    if metrics.trades <= 0:
        return 0.0

    shares = [
        metrics.per_strategy_trades.get(strategy_id, 0) / metrics.trades
        for strategy_id in STRATEGIES
    ]
    active_shares = [share for share in shares if share > 0.0]
    if not active_shares:
        return 0.0

    entropy = -sum(share * math.log(share) for share in active_shares) / math.log(len(STRATEGIES))
    min_share_score = _clip(min(shares) / 0.10)
    return _clip(0.70 * entropy + 0.30 * min_share_score)


def _score_component_values(dev: ReplayMetrics, holdout: ReplayMetrics) -> dict[str, float]:
    components = {
        "dev_return": _clip(dev.net_return_pct / 100.0),
        "dev_frequency": _clip(dev.trades / 95.0),
        "dev_profit_factor": _clip((dev.profit_factor - 1.0) / 5.0),
        "dev_drawdown_resilience": _clip(1.0 - dev.max_drawdown_pct / 8.5),
        "holdout_return": _clip((holdout.net_return_pct + 2.0) / 6.0),
        "holdout_profit_factor": _clip((holdout.profit_factor - 0.6) / 2.5),
        "strategy_balance": _strategy_balance_score(dev),
    }
    if len(components) > MAX_SCORE_COMPONENTS:
        raise ValueError(
            f"Score has {len(components)} components, exceeding the limit of {MAX_SCORE_COMPONENTS}"
        )
    return components


def _score_metrics(
    dev: ReplayMetrics,
    holdout: ReplayMetrics,
) -> tuple[float, bool, str, dict[str, float]]:
    hard_failures = []
    if dev.trades < 76:
        hard_failures.append(f"development trades too low ({dev.trades} < 76)")
    if holdout.trades < 8:
        hard_failures.append(f"holdout trades too low ({holdout.trades} < 8)")
    if dev.profit_factor < 1.75:
        hard_failures.append(f"development PF too low ({dev.profit_factor:.2f} < 1.75)")
    if dev.max_drawdown_pct > 8.5:
        hard_failures.append(f"development DD too high ({dev.max_drawdown_pct:.2f}% > 8.5%)")
    if holdout.max_drawdown_pct > 7.5:
        hard_failures.append(f"holdout DD too high ({holdout.max_drawdown_pct:.2f}% > 7.5%)")
    if holdout.net_return_pct < -3.0:
        hard_failures.append(f"holdout return too weak ({holdout.net_return_pct:.2f}% < -3.0%)")

    components = _score_component_values(dev, holdout)
    score = sum(weight * components[name] for name, weight in SCORE_COMPONENT_WEIGHTS)
    return score, bool(hard_failures), "; ".join(hard_failures), components


def _actual_validation_score(
    full: dict[str, float],
    holdout: dict[str, float],
) -> tuple[float, bool, str, dict[str, float]]:
    full_trades = float(full.get("total_trades", 0.0))
    holdout_trades = float(holdout.get("total_trades", 0.0))
    full_pf = float(full.get("profit_factor", 0.0))
    holdout_pf = float(holdout.get("profit_factor", 0.0))
    full_dd = float(full.get("max_drawdown_pct", 0.0))
    holdout_dd = float(holdout.get("max_drawdown_pct", 0.0))
    holdout_return = float(holdout.get("net_return_pct", 0.0))

    hard_failures = []
    if full_trades < 90:
        hard_failures.append(f"full trades too low ({full_trades:.0f} < 90)")
    if holdout_trades < 8:
        hard_failures.append(f"holdout trades too low ({holdout_trades:.0f} < 8)")
    if full_pf < 1.75:
        hard_failures.append(f"full PF too low ({full_pf:.2f} < 1.75)")
    if full_dd > 8.5:
        hard_failures.append(f"full DD too high ({full_dd:.2f}% > 8.5%)")
    if holdout_dd > 7.5:
        hard_failures.append(f"holdout DD too high ({holdout_dd:.2f}% > 7.5%)")
    if holdout_return < -3.0:
        hard_failures.append(f"holdout return too weak ({holdout_return:.2f}% < -3.0%)")

    components = {
        "full_return": _clip(float(full.get("net_return_pct", 0.0)) / 100.0),
        "full_frequency": _clip(full_trades / 96.0),
        "full_profit_factor": _clip((full_pf - 1.0) / 5.0),
        "full_drawdown_resilience": _clip(1.0 - full_dd / 8.5),
        "holdout_return": _clip((holdout_return + 2.0) / 6.0),
        "holdout_profit_factor": _clip((holdout_pf - 0.6) / 2.5),
        "holdout_drawdown_resilience": _clip(1.0 - holdout_dd / 7.5),
    }
    if len(components) > MAX_SCORE_COMPONENTS:
        raise ValueError(
            f"Actual validation score has {len(components)} components, "
            f"exceeding the limit of {MAX_SCORE_COMPONENTS}"
        )

    score = sum(
        weight * components[name]
        for name, weight in ACTUAL_VALIDATION_COMPONENT_WEIGHTS
    )
    return score, bool(hard_failures), "; ".join(hard_failures), components


def _evaluate_policy(
    policy: PortfolioPolicy,
    trade_windows: dict[str, dict[str, list[Trade]]],
) -> ReplayEvaluation:
    dev = _replay_portfolio(policy, trade_windows[DEV_WINDOW.name])
    holdout = _replay_portfolio(policy, trade_windows[HOLDOUT_WINDOW.name])
    score, rejected, reason, components = _score_metrics(dev, holdout)
    return ReplayEvaluation(
        policy_name=policy.name,
        score=score,
        rejected=rejected,
        reject_reason=reason,
        development=dev,
        holdout=holdout,
        score_components=components,
    )


def _run_individual_strategy_window(
    window: WindowSpec,
    strategy_id: str,
    configs: dict[str, Any],
) -> tuple[str, str, list[Trade]]:
    cfg = apply_mutations(configs[strategy_id], {})
    cfg.symbols = list(SYMBOLS)
    bt_result = run_individual(
        strategy_config=cfg,
        backtest_config=_bt_config(window),
        data_dir=DATA_DIR,
        strategy_type=strategy_id,
    )
    return window.name, strategy_id, bt_result.trades


def _run_trade_harvest_windows(
    windows: tuple[WindowSpec, ...],
    configs: dict[str, Any],
) -> dict[str, dict[str, list[Trade]]]:
    results: dict[str, dict[str, list[Trade]]] = {
        window.name: {} for window in windows
    }
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_run_individual_strategy_window, window, strategy_id, configs)
            for window in windows
            for strategy_id in STRATEGIES
        ]
        for future in as_completed(futures):
            window_name, strategy_id, trades = future.result()
            results[window_name][strategy_id] = trades

    for window in windows:
        for strategy_id in STRATEGIES:
            trades = results[window.name][strategy_id]
            print(f"  {window.name}: {strategy_id} individual trades={len(trades)}")
    return results


def _actual_portfolio_run(
    policy: PortfolioPolicy,
    window: WindowSpec,
    base_configs: dict[str, Any],
) -> tuple[dict[str, float], Any]:
    configs = _apply_policy_configs(policy, base_configs)
    result = run_portfolio_backtest(
        portfolio_config=_portfolio_config(policy),
        strategy_configs=configs,
        backtest_config=_bt_config(window),
        data_dir=DATA_DIR,
    )
    return metrics_to_dict(result.metrics), result


def _actual_portfolio_worker(
    policy: PortfolioPolicy,
    window: WindowSpec,
) -> tuple[dict[str, float], list[Trade]]:
    base_configs = _load_base_configs()
    metrics, result = _actual_portfolio_run(policy, window, base_configs)
    return metrics, result.all_trades


def _actual_portfolio_runs(
    jobs: tuple[tuple[str, PortfolioPolicy, WindowSpec], ...],
    base_configs: dict[str, Any],
) -> dict[str, tuple[dict[str, float], list[Trade]]]:
    del base_configs
    results: dict[str, tuple[dict[str, float], list[Trade]]] = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_actual_portfolio_worker, policy, window): label
            for label, policy, window in jobs
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _actual_validation_grid(
    policies: list[PortfolioPolicy],
    base_configs: dict[str, Any],
) -> tuple[list[ActualValidation], dict[str, Any]]:
    deduped: list[PortfolioPolicy] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.name in seen:
            continue
        seen.add(policy.name)
        deduped.append(policy)

    jobs: list[tuple[str, PortfolioPolicy, WindowSpec]] = []
    policy_labels: dict[str, PortfolioPolicy] = {}
    for idx, policy in enumerate(deduped):
        label = f"policy_{idx}"
        policy_labels[label] = policy
        jobs.append((f"{label}_full", policy, FULL_WINDOW))
        jobs.append((f"{label}_holdout", policy, HOLDOUT_WINDOW))

    raw_results = _actual_portfolio_runs(tuple(jobs), base_configs)
    validations: list[ActualValidation] = []
    full_results: dict[str, Any] = {}
    for label, policy in policy_labels.items():
        full_metrics, full_result = raw_results[f"{label}_full"]
        holdout_metrics, _ = raw_results[f"{label}_holdout"]
        full_summary = _actual_summary(full_metrics)
        holdout_summary = _actual_summary(holdout_metrics)
        score, rejected, reason, components = _actual_validation_score(
            full_summary,
            holdout_summary,
        )
        validations.append(
            ActualValidation(
                policy_name=policy.name,
                score=score,
                rejected=rejected,
                reject_reason=reason,
                full=full_summary,
                holdout=holdout_summary,
                score_components=components,
            )
        )
        full_results[policy.name] = full_result
    return validations, full_results


def _choose_actual_validated_policy(
    policies: list[PortfolioPolicy],
    validations: list[ActualValidation],
) -> tuple[PortfolioPolicy, ActualValidation]:
    policy_by_name = {policy.name: policy for policy in policies}
    viable = [item for item in validations if not item.rejected]
    if not viable:
        viable = validations
    selected_validation = max(viable, key=lambda item: item.score)
    return policy_by_name[selected_validation.policy_name], selected_validation


def _candidate_phases() -> dict[int, list[PolicyDelta]]:
    return {
        1: [
            PolicyDelta(
                name="breakout_block_sol_longs",
                phase=1,
                thesis="Breakout diagnostics show SOL longs 0/2 while SOL short is positive; block only that weak side.",
                strategy_mutations={"breakout": {"symbol_filter.sol_direction": "short_only"}},
                filter_rules=(
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "SOL",
                        "direction": "LONG",
                        "reason": "breakout_SOL_LONG_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="breakout_block_eth_shorts",
                phase=1,
                thesis="Breakout ETH edge is concentrated in longs; ETH shorts are the recurring weak side.",
                strategy_mutations={
                    "breakout": {
                        "symbol_filter.eth_direction": "long_only",
                        "symbol_filter.eth_relaxed_body_direction": "long_only",
                    }
                },
                filter_rules=(
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "ETH",
                        "direction": "SHORT",
                        "reason": "breakout_ETH_SHORT_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="breakout_block_sol_longs_eth_shorts",
                phase=1,
                thesis="Combine only the two breakout side filters with diagnostic support.",
                strategy_mutations={
                    "breakout": {
                        "symbol_filter.sol_direction": "short_only",
                        "symbol_filter.eth_direction": "long_only",
                        "symbol_filter.eth_relaxed_body_direction": "long_only",
                    }
                },
                filter_rules=(
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "SOL",
                        "direction": "LONG",
                        "reason": "breakout_SOL_LONG_filter",
                    },
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "ETH",
                        "direction": "SHORT",
                        "reason": "breakout_ETH_SHORT_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="momentum_require_one_confluence",
                phase=1,
                thesis="Momentum zero-confluence trades are slightly negative; require one confluence for B setups.",
                strategy_mutations={"momentum": {"setup.min_confluences_b": 1}},
                filter_rules=(
                    {
                        "kind": "confluence_lte",
                        "strategy_id": "momentum",
                        "threshold": 0,
                        "reason": "momentum_zero_confluence_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="trend_raise_weighted_b_score",
                phase=1,
                thesis="Nudge trend B setup quality higher without disabling a whole symbol or direction.",
                strategy_mutations={"trend": {"setup.min_setup_score_b": 1.45}},
            ),
        ],
        2: [
            PolicyDelta(
                name="trend_reentry_more_patient",
                phase=2,
                thesis="Trend has the broadest sample; allow one more controlled reentry window to lift frequency.",
                strategy_mutations={
                    "trend": {
                        "reentry.max_reentries": 2,
                        "reentry.max_wait_bars": 8,
                        "reentry.risk_scale": 0.65,
                    }
                },
            ),
            PolicyDelta(
                name="momentum_reentry_faster",
                phase=2,
                thesis="Momentum holds short but recovers quickly; reduce cooldown without increasing max reentries.",
                strategy_mutations={"momentum": {"reentry.cooldown_bars": 2}},
            ),
            PolicyDelta(
                name="breakout_relaxed_body_cautious_expand",
                phase=2,
                thesis="Probe more breakout frequency with lower relaxed-body confluence but smaller relaxed risk.",
                strategy_mutations={
                    "breakout": {
                        "setup.relaxed_body_min_confluences": 4,
                        "setup.relaxed_body_min_room_r": 1.6,
                        "setup.relaxed_body_risk_scale": 0.4,
                    }
                },
            ),
        ],
        3: [
            PolicyDelta(
                name="risk_all_115",
                phase=3,
                thesis="Uniform modest risk lift; tests whether the low DD headroom is real.",
                risk_scales={"momentum": 1.15, "trend": 1.15, "breakout": 1.15},
            ),
            PolicyDelta(
                name="risk_trend_core_130",
                phase=3,
                thesis="Overweight the most statistically supported strategy while keeping smaller-sample strategies near baseline.",
                risk_scales={"momentum": 0.95, "trend": 1.30, "breakout": 1.05},
            ),
            PolicyDelta(
                name="risk_trend_breakout_lean",
                phase=3,
                thesis="Lean into trend sample depth and breakout expectancy, with momentum slightly reduced.",
                risk_scales={"momentum": 0.90, "trend": 1.25, "breakout": 1.15},
            ),
            PolicyDelta(
                name="risk_frequency_lean",
                phase=3,
                thesis="Small momentum and trend lift for trade count, while leaving breakout concentration unchanged.",
                risk_scales={"momentum": 1.10, "trend": 1.20, "breakout": 1.00},
            ),
            PolicyDelta(
                name="risk_breakout_alpha_probe",
                phase=3,
                thesis="Probe breakout overweight, but only the holdout/scoring can approve it due low sample concentration.",
                risk_scales={"momentum": 0.85, "trend": 1.15, "breakout": 1.25},
            ),
        ],
        4: [
            PolicyDelta(
                name="caps_unlock_one_directional_unit",
                phase=4,
                thesis="The only saved portfolio block was directional_cap_R; loosen one unit to recover frequency.",
                portfolio_overrides={
                    "directional_cap_R": 5.0,
                    "heat_cap_R": 7.5,
                    "symbol_exposure_cap_R": 3.5,
                    "max_total_positions": 10,
                },
            ),
            PolicyDelta(
                name="caps_aggressive_with_dd_guard",
                phase=4,
                thesis="Allow more concurrent alpha, but cut size dynamically if drawdown starts compounding.",
                portfolio_overrides={
                    "directional_cap_R": 5.5,
                    "heat_cap_R": 8.0,
                    "symbol_exposure_cap_R": 4.0,
                    "max_total_positions": 10,
                    "dd_tiers": (
                        (0.05, 0.75),
                        (0.08, 0.50),
                        (0.11, 0.25),
                        (0.14, 0.00),
                    ),
                },
            ),
            PolicyDelta(
                name="trend_priority_headroom",
                phase=4,
                thesis="Reserve directional capacity for trend when lower-priority strategies crowd one side.",
                portfolio_overrides={
                    "priority_headroom_R": 1.0,
                    "priority_reserve_threshold": 1,
                    "directional_cap_R": 5.0,
                    "heat_cap_R": 7.5,
                },
            ),
        ],
        5: [
            PolicyDelta(
                name="postcap_risk_all_110",
                phase=5,
                thesis="After cap unlock, test a smaller uniform risk lift than the pre-cap 1.15 probe.",
                risk_scales={"momentum": 1.10, "trend": 1.10, "breakout": 1.10},
            ),
            PolicyDelta(
                name="postcap_risk_trend_core_120",
                phase=5,
                thesis="After cap unlock, overweight trend modestly while keeping lower-sample sleeves restrained.",
                risk_scales={"momentum": 0.95, "trend": 1.20, "breakout": 1.05},
            ),
            PolicyDelta(
                name="postcap_risk_trend_breakout_lean",
                phase=5,
                thesis="After cap unlock, lean into trend sample depth and breakout expectancy with moderate sizing.",
                risk_scales={"momentum": 0.90, "trend": 1.18, "breakout": 1.12},
            ),
            PolicyDelta(
                name="postcap_risk_frequency_lean",
                phase=5,
                thesis="After cap unlock, lift the higher-frequency sleeves without increasing breakout concentration.",
                risk_scales={"momentum": 1.08, "trend": 1.15, "breakout": 1.00},
            ),
        ],
    }


def _evaluate_delta_candidate(
    current: PortfolioPolicy,
    delta: PolicyDelta,
    trade_windows: dict[str, dict[str, list[Trade]]],
) -> tuple[str, PortfolioPolicy, ReplayEvaluation]:
    policy = _merge_policy(current, delta)
    return delta.name, policy, _evaluate_policy(policy, trade_windows)


def _run_phase_auto(
    trade_windows: dict[str, dict[str, list[Trade]]],
) -> tuple[PortfolioPolicy, list[dict[str, Any]], list[tuple[PortfolioPolicy, ReplayEvaluation]]]:
    current = PortfolioPolicy(name="baseline")
    phase_log: list[dict[str, Any]] = []
    current_eval = _evaluate_policy(current, trade_windows)
    min_delta = 0.005
    final_phase_candidates: list[tuple[PortfolioPolicy, ReplayEvaluation]] = []

    print(
        f"  baseline replay score={current_eval.score:.4f} "
        f"dev_ret={current_eval.development.net_return_pct:.2f}% "
        f"holdout_ret={current_eval.holdout.net_return_pct:.2f}%"
    )

    for phase, candidates in _candidate_phases().items():
        phase_results = []
        best_candidate_policy: PortfolioPolicy | None = None
        best_candidate_eval: ReplayEvaluation | None = None

        completed: dict[str, tuple[PortfolioPolicy, ReplayEvaluation]] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(_evaluate_delta_candidate, current, delta, trade_windows)
                for delta in candidates
            ]
            for future in as_completed(futures):
                delta_name, policy, evaluation = future.result()
                completed[delta_name] = (policy, evaluation)

        for delta in candidates:
            policy, evaluation = completed[delta.name]
            phase_results.append({
                "delta": delta.name,
                "thesis": delta.thesis,
                "policy": policy.name,
                "evaluation": evaluation.to_dict(),
            })
            status = "REJECT" if evaluation.rejected else "ok"
            print(
                f"  phase {phase} {delta.name}: {status} score={evaluation.score:.4f} "
                f"dev_ret={evaluation.development.net_return_pct:.2f}% "
                f"holdout_ret={evaluation.holdout.net_return_pct:.2f}%"
            )
            if (
                not evaluation.rejected
                and (
                    best_candidate_eval is None
                    or evaluation.score > best_candidate_eval.score
                )
            ):
                best_candidate_policy = policy
                best_candidate_eval = evaluation

        final_phase_candidates = [
            completed[delta.name]
            for delta in candidates
            if not completed[delta.name][1].rejected
        ]

        accepted = (
            best_candidate_policy is not None
            and best_candidate_eval is not None
            and best_candidate_eval.score > current_eval.score + min_delta
        )
        if accepted:
            current = best_candidate_policy
            current_eval = best_candidate_eval

        phase_log.append({
            "phase": phase,
            "accepted": accepted,
            "accepted_policy": current.name,
            "accepted_deltas": list(current.accepted_deltas),
            "current_evaluation": current_eval.to_dict(),
            "candidates": phase_results,
        })
        print(
            f"  phase {phase} {'accepted' if accepted else 'kept baseline/current'}: "
            f"{current.name} score={current_eval.score:.4f}"
        )

    finalists: dict[str, tuple[PortfolioPolicy, ReplayEvaluation]] = {
        current.name: (current, current_eval),
    }
    for policy, evaluation in final_phase_candidates:
        finalists.setdefault(policy.name, (policy, evaluation))
    return current, phase_log, list(finalists.values())


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _actual_summary(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "total_trades",
        "win_rate",
        "profit_factor",
        "net_return_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "calmar_ratio",
    ]
    return {k: float(metrics.get(k, 0.0)) for k in keys}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _save_recommended_configs(policy: PortfolioPolicy, base_configs: dict[str, Any]) -> None:
    RECOMMENDED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = _apply_policy_configs(policy, base_configs)
    for sid, cfg in configs.items():
        _write_json(RECOMMENDED_CONFIG_DIR / f"{sid}.json", {"strategy": cfg.to_dict()})
    _write_json(ROUND_DIR / "recommended_portfolio_config.json", _portfolio_config(policy).to_dict())


def _diagnostic_trades(trades: list[Trade]) -> list[Trade]:
    """Return copies that force diagnostics to use geometric R.

    Some partial-fill paths store extreme realized_r_multiple values even when
    dollar P&L is normal.  The diagnostics renderer prefers realized/economic R,
    so for this portfolio artifact we clear realized_r_multiple and leave net P&L
    untouched.
    """
    return [replace(trade, realized_r_multiple=None) for trade in trades]


def _build_report(
    selected: PortfolioPolicy,
    replay_selected: PortfolioPolicy,
    actual_validations: list[ActualValidation],
    phase_log: list[dict[str, Any]],
    actual_baseline_full: dict[str, float],
    actual_selected_full: dict[str, float],
    actual_baseline_holdout: dict[str, float],
    actual_selected_holdout: dict[str, float],
) -> str:
    lines = []
    lines.append("PORTFOLIO PHASE-AUTO ROUND 2")
    lines.append("=" * 80)
    lines.append(f"Initial equity assumption: ${INITIAL_EQUITY:,.0f}")
    lines.append(f"Development: {DEV_START} to {DEV_END}")
    lines.append(f"Forward holdout: {HOLDOUT_START} to {HOLDOUT_END}")
    lines.append(f"Max workers: {MAX_WORKERS}")
    lines.append(
        f"Replay score components: {len(SCORE_COMPONENT_WEIGHTS)} "
        f"(limit {MAX_SCORE_COMPONENTS})"
    )
    for name, weight in SCORE_COMPONENT_WEIGHTS:
        lines.append(f"  {name}: weight={weight:.2f}")
    lines.append(
        f"Actual validation score components: {len(ACTUAL_VALIDATION_COMPONENT_WEIGHTS)} "
        f"(limit {MAX_SCORE_COMPONENTS})"
    )
    for name, weight in ACTUAL_VALIDATION_COMPONENT_WEIGHTS:
        lines.append(f"  {name}: weight={weight:.2f}")
    lines.append("Risk stance: aggressive-leaning, hard-gated at roughly 8.5% replay DD.")
    lines.append("")

    lines.append("Selected policy:")
    lines.append(f"  name: {selected.name}")
    if selected.name != replay_selected.name:
        lines.append(f"  replay-selected candidate: {replay_selected.name}")
        lines.append("  final selection basis: highest non-rejected actual-validation score")
    lines.append(f"  accepted deltas: {', '.join(selected.accepted_deltas) or 'none'}")
    lines.append(f"  risk scales: {selected.risk_scales}")
    lines.append(f"  portfolio overrides: {selected.portfolio_overrides or '{}'}")
    if selected.strategy_mutations:
        lines.append("  strategy mutations:")
        for sid, muts in selected.strategy_mutations.items():
            if muts:
                lines.append(f"    {sid}: {muts}")
    lines.append("")

    lines.append("Actual portfolio validation:")
    lines.append("  Full refreshed window")
    lines.append(
        "    baseline: "
        f"trades={actual_baseline_full['total_trades']:.0f}, "
        f"return={_format_pct(actual_baseline_full['net_return_pct'])}, "
        f"PF={actual_baseline_full['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_baseline_full['max_drawdown_pct'])}"
    )
    lines.append(
        "    selected: "
        f"trades={actual_selected_full['total_trades']:.0f}, "
        f"return={_format_pct(actual_selected_full['net_return_pct'])}, "
        f"PF={actual_selected_full['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_selected_full['max_drawdown_pct'])}"
    )
    lines.append("  Forward holdout")
    lines.append(
        "    baseline: "
        f"trades={actual_baseline_holdout['total_trades']:.0f}, "
        f"return={_format_pct(actual_baseline_holdout['net_return_pct'])}, "
        f"PF={actual_baseline_holdout['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_baseline_holdout['max_drawdown_pct'])}"
    )
    lines.append(
        "    selected: "
        f"trades={actual_selected_holdout['total_trades']:.0f}, "
        f"return={_format_pct(actual_selected_holdout['net_return_pct'])}, "
        f"PF={actual_selected_holdout['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_selected_holdout['max_drawdown_pct'])}"
    )
    lines.append("")

    lines.append("Actual-validation candidates:")
    for item in sorted(actual_validations, key=lambda x: x.score, reverse=True):
        status = "REJECT" if item.rejected else "ok"
        lines.append(
            f"  {status} {item.policy_name}: "
            f"score={item.score:.4f}, "
            f"full_ret={item.full['net_return_pct']:.2f}%, "
            f"trades={item.full['total_trades']:.0f}, "
            f"PF={item.full['profit_factor']:.2f}, "
            f"DD={item.full['max_drawdown_pct']:.2f}%, "
            f"holdout_ret={item.holdout['net_return_pct']:.2f}%"
        )
    lines.append("")

    lines.append("Phase decisions:")
    for phase in phase_log:
        evaluation = phase["current_evaluation"]
        dev = evaluation["development"]
        holdout = evaluation["holdout"]
        lines.append(
            f"  Phase {phase['phase']}: "
            f"{'accepted' if phase['accepted'] else 'no change'} -> "
            f"{phase['accepted_policy']} "
            f"(score={evaluation['score']:.4f}, "
            f"dev_ret={dev['net_return_pct']:.2f}%, "
            f"holdout_ret={holdout['net_return_pct']:.2f}%)"
        )
    lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "  The saved diagnostics show real portfolio-level edge, but the forward "
        "holdout after 2026-04-18 is weak. The round therefore rewards return and "
        "frequency only when the holdout degradation does not worsen and drawdown "
        "stays controlled."
    )
    lines.append(
        "  Treat signal-discrimination deltas as candidates, not final truth, when "
        "they rely on fewer than roughly 20 supporting trades. The risk/cap policy "
        "is more robust because it preserves the signal set and changes allocation."
    )
    return "\n".join(lines)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    print("Portfolio phase-auto round 2")
    print(f"Output: {ROUND_DIR}")
    print(f"Max workers: {MAX_WORKERS}")
    print("Loading latest round_3 strategy configs...")
    base_configs = _load_base_configs()

    print("Running individual strategy trade harvest for replay windows...")
    trade_windows = _run_trade_harvest_windows((DEV_WINDOW, HOLDOUT_WINDOW), base_configs)

    print("Running phased replay optimizer...")
    replay_selected_policy, phase_log, replay_finalists = _run_phase_auto(trade_windows)

    print("Running actual portfolio validation for baseline and replay finalists...")
    baseline_policy = PortfolioPolicy(name="baseline")
    candidate_policies = [baseline_policy] + [
        policy for policy, _evaluation in replay_finalists
    ]
    actual_validations, full_results = _actual_validation_grid(
        candidate_policies,
        base_configs,
    )
    selected_policy, selected_actual_validation = _choose_actual_validated_policy(
        candidate_policies,
        actual_validations,
    )
    baseline_actual_validation = next(
        item for item in actual_validations if item.policy_name == baseline_policy.name
    )
    selected_full_trades = full_results[selected_policy.name]
    baseline_full_metrics = baseline_actual_validation.full
    selected_full_metrics = selected_actual_validation.full
    baseline_holdout_metrics = baseline_actual_validation.holdout
    selected_holdout_metrics = selected_actual_validation.holdout

    print("Saving artifacts...")
    _save_recommended_configs(selected_policy, base_configs)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round": ROUND_NAME,
        "initial_equity": INITIAL_EQUITY,
        "max_workers": MAX_WORKERS,
        "score_component_limit": MAX_SCORE_COMPONENTS,
        "score_component_weights": dict(SCORE_COMPONENT_WEIGHTS),
        "actual_validation_component_weights": dict(ACTUAL_VALIDATION_COMPONENT_WEIGHTS),
        "windows": {
            "development": {"start": str(DEV_START), "end": str(DEV_END)},
            "forward_holdout": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END)},
            "full_refreshed": {"start": str(FULL_START), "end": str(FULL_END)},
        },
        "replay_selected_policy": {
            "name": replay_selected_policy.name,
            "accepted_deltas": list(replay_selected_policy.accepted_deltas),
            "strategy_mutations": replay_selected_policy.strategy_mutations,
            "risk_scales": replay_selected_policy.risk_scales,
            "portfolio_overrides": replay_selected_policy.portfolio_overrides,
            "filter_rules": list(replay_selected_policy.filter_rules),
        },
        "selected_policy": {
            "name": selected_policy.name,
            "accepted_deltas": list(selected_policy.accepted_deltas),
            "strategy_mutations": selected_policy.strategy_mutations,
            "risk_scales": selected_policy.risk_scales,
            "portfolio_overrides": selected_policy.portfolio_overrides,
            "filter_rules": list(selected_policy.filter_rules),
        },
        "phase_log": phase_log,
        "actual_validation": {
            "baseline_full": baseline_full_metrics,
            "selected_full": selected_full_metrics,
            "baseline_holdout": baseline_holdout_metrics,
            "selected_holdout": selected_holdout_metrics,
            "selected_policy_name": selected_policy.name,
            "replay_selected_policy_name": replay_selected_policy.name,
            "candidates": [item.to_dict() for item in actual_validations],
        },
    }
    _write_json(ROUND_DIR / "phase_auto_results.json", payload)

    report = _build_report(
        selected_policy,
        replay_selected_policy,
        actual_validations,
        phase_log,
        baseline_full_metrics,
        selected_full_metrics,
        baseline_holdout_metrics,
        selected_holdout_metrics,
    )
    (ROUND_DIR / "phase_auto_report.txt").write_text(report, encoding="utf-8")

    diagnostics = (
        "# R-multiple sections use geometric R; dollar P&L is from the actual "
        "selected portfolio backtest.\n\n"
    )
    diagnostics += generate_diagnostics(
        _diagnostic_trades(selected_full_trades),
        initial_equity=INITIAL_EQUITY,
    )
    (ROUND_DIR / "recommended_portfolio_diagnostics.txt").write_text(
        diagnostics,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
