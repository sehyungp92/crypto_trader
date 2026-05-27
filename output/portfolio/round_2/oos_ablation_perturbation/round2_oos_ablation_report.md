# Portfolio Round 2 OOS Ablation And Perturbation Report

Generated: 2026-05-27T14:42:22.547854+00:00
IS window: 2026-02-25 to 2026-04-20
OOS window: 2026-04-21 to 2026-05-23
Candidates evaluated: 433

## Current Round 2 Baseline
- IS: 66.65% return, 45 trades, PF 6.321492107306187, expR 0.526, exitEff 0.502, DD 3.85%
- OOS: 33.74% return, 39 trades, PF 2.5531918767231168, expR 0.340, exitEff 0.429, DD 5.42%

## OOS Weakness Autopsy
Worst current OOS trades:
- 2026-05-17T05:00:00+00:00 momentum SOL SHORT inside_bar_break protective_stop R=-1.211 PnL=-713.84 MFE=0.0
- 2026-05-23T14:15:00+00:00 momentum SOL SHORT inside_bar_break protective_stop R=-1.186 PnL=-702.99 MFE=0.0
- 2026-05-01T01:15:00+00:00 momentum SOL SHORT micro_structure_shift protective_stop R=-1.243 PnL=-607.11 MFE=0.024492975793628698
- 2026-05-05T17:00:00+00:00 breakout ETH SHORT model1_close protective_stop R=-0.618 PnL=-571.37 MFE=0.27674933748294994
- 2026-04-22T23:45:00+00:00 momentum BTC LONG inside_bar_break protective_stop R=-1.204 PnL=-567.51 MFE=0.2826956085589761
- 2026-04-29T12:45:00+00:00 momentum ETH LONG inside_bar_break protective_stop R=-1.170 PnL=-546.68 MFE=0.20034486932531215

Current OOS group attribution, weakest first:
- breakout|ETH|SHORT|model1_close: n=1, pnl=-571.37, avgR=-0.618, WR=0.0%
- momentum|SOL|SHORT|micro_structure_shift: n=2, pnl=-561.03, avgR=-0.572, WR=50.0%
- momentum|SOL|SHORT|bearish_engulfing: n=1, pnl=-395.10, avgR=-0.650, WR=0.0%
- momentum|SOL|SHORT|inside_bar_break: n=5, pnl=-269.62, avgR=-0.040, WR=40.0%
- trend|SOL|LONG|none: n=2, pnl=-264.87, avgR=-0.159, WR=0.0%
- trend|SOL|SHORT|none: n=1, pnl=-0.43, avgR=-0.000, WR=0.0%
- trend|ETH|LONG|none: n=3, pnl=39.72, avgR=0.003, WR=33.3%
- momentum|BTC|LONG|inside_bar_break: n=4, pnl=250.95, avgR=0.155, WR=50.0%
- trend|BTC|LONG|none: n=3, pnl=329.89, avgR=0.264, WR=66.7%
- momentum|ETH|LONG|hammer: n=1, pnl=339.54, avgR=0.650, WR=100.0%

Interpretation: OOS underperformance is broad edge decay plus several repeatable weak sleeves. It is not caused by one or two outsized loss events; the worst losses are ordinary stop/quick-exit events clustered in momentum SOL shorts and BTC/ETH inside-bar longs. Risk scaling amplifies both sides but does not by itself create the losing trades.

## Best Preserving OOS Improvements
- perturb_strategy_active_cumulative__trend__setup.orderly_max_body_frac__1.08 (perturbation): OOS 40.10% return, 43 trades, PF 2.6457917764117185, expR 0.336, exitEff 0.432, DD 5.42% [delta return +6.36pp, trades +4; IS return -1.17pp, trades +1]
- perturb_strategy__momentum__risk.risk_pct_b__0.018 (perturbation): OOS 36.56% return, 39 trades, PF 2.4305216726489047, expR 0.340, exitEff 0.429, DD 5.69% [delta return +2.82pp, trades +0; IS return +5.70pp, trades +0]
- perturb_risk_all_1.25 (perturbation): OOS 36.04% return, 39 trades, PF 2.523012686240192, expR 0.340, exitEff 0.429, DD 5.68% [delta return +2.30pp, trades +0; IS return +3.61pp, trades +0]
- perturb_strategy__momentum__trail.trail_activation_bars__8 (perturbation): OOS 34.90% return, 39 trades, PF 2.6688370439530997, expR 0.352, exitEff 0.429, DD 5.42% [delta return +1.16pp, trades +0; IS return +0.03pp, trades +0]
- ablate_strategy_active_cumulative_to_base__trend__setup.min_setup_score_a (ablation): OOS 35.27% return, 39 trades, PF 2.5945578685818362, expR 0.340, exitEff 0.429, DD 5.42% [delta return +1.53pp, trades +0; IS return +0.00pp, trades +0]
- perturb_strategy_active_cumulative__trend__setup.min_setup_score_a__2.255 (perturbation): OOS 35.27% return, 39 trades, PF 2.5945578685818362, expR 0.340, exitEff 0.429, DD 5.42% [delta return +1.53pp, trades +0; IS return +0.00pp, trades +0]
- perturb_strategy_active_cumulative__trend__setup.min_setup_score_a__2.46 (perturbation): OOS 35.27% return, 39 trades, PF 2.5945578685818362, expR 0.340, exitEff 0.429, DD 5.42% [delta return +1.53pp, trades +0; IS return +0.00pp, trades +0]
- perturb_strategy_active_cumulative__breakout__setup.body_ratio_min__0.715 (perturbation): OOS 35.87% return, 38 trades, PF 2.8273770296528027, expR 0.365, exitEff 0.429, DD 4.65% [delta return +2.13pp, trades -1; IS return +0.89pp, trades +0]
- perturb_risk_all_1.20 (perturbation): OOS 35.11% return, 39 trades, PF 2.5436272725346467, expR 0.340, exitEff 0.429, DD 5.59% [delta return +1.37pp, trades +0; IS return +2.20pp, trades +0]
- perturb_risk_momentum_1.25 (perturbation): OOS 35.11% return, 39 trades, PF 2.5005377390693937, expR 0.340, exitEff 0.429, DD 5.54% [delta return +1.37pp, trades +0; IS return +2.44pp, trades +0]
- perturb_strategy__momentum__risk.risk_pct_b__0.0165 (perturbation): OOS 35.15% return, 39 trades, PF 2.48798594583094, expR 0.340, exitEff 0.429, DD 5.56% [delta return +1.41pp, trades +0; IS return +2.83pp, trades +0]
- perturb_risk_breakout_1.25 (perturbation): OOS 34.45% return, 39 trades, PF 2.5738999952517414, expR 0.340, exitEff 0.429, DD 5.53% [delta return +0.71pp, trades +0; IS return +0.66pp, trades +0]

## Ablation Winners
- ablate_strategy_active_cumulative_to_base__trend__setup.pullback_max_bars: OOS return 9.81% (-23.93pp), OOS trades 65 (+26), IS return 120.05% (+53.39pp)
- ablate_strategy_to_round1__trend__setup.pullback_max_bars: OOS return 5.77% (-27.97pp), OOS trades 66 (+27), IS return 128.54% (+61.88pp)
- ablate_strategy_shadowed_to_round1__momentum__symbol_filter.sol_direction: OOS return 29.92% (-3.82pp), OOS trades 50 (+11), IS return 80.35% (+13.69pp)
- ablate_strategy_round_3__trend: OOS return 4.23% (-29.51pp), OOS trades 64 (+25), IS return 129.79% (+63.13pp)
- ablate_strategy_all_to_round1__trend: OOS return 4.99% (-28.75pp), OOS trades 64 (+25), IS return 114.95% (+48.29pp)
- ablate_portfolio_field__strategy.momentum.symbol_filter.sol_direction: OOS return 31.97% (-1.77pp), OOS trades 44 (+5), IS return 73.98% (+7.33pp)
- ablate_strategy_active_cumulative_to_base__trend__setup.min_setup_score_a: OOS return 35.27% (+1.53pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- ablate_strategy_active_cumulative_to_base__trend__regime.b_adx_rising_required: OOS return 33.26% (-0.48pp), OOS trades 40 (+1), IS return 66.65% (+0.00pp)
- ablate_strategy_active_cumulative_to_base__trend__exits.tp2_frac: OOS return 34.07% (+0.33pp), OOS trades 39 (+0), IS return 66.41% (-0.24pp)
- ablate_strategy_active_cumulative_to_base__trend__risk.risk_pct_a: OOS return 34.00% (+0.26pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)

## Perturbation Winners
- perturb_strategy_active_cumulative__trend__setup.orderly_max_body_frac__1.08: OOS return 40.10% (+6.36pp), OOS trades 43 (+4), IS return 65.48% (-1.17pp)
- perturb_strategy_active_cumulative__trend__setup.min_setup_score_b__1.08: OOS return 33.30% (-0.44pp), OOS trades 41 (+2), IS return 66.49% (-0.17pp)
- perturb_strategy_active_cumulative__trend__setup.min_setup_score_b__1.215: OOS return 33.30% (-0.44pp), OOS trades 41 (+2), IS return 66.49% (-0.17pp)
- perturb_strategy__momentum__risk.risk_pct_b__0.018: OOS return 36.56% (+2.82pp), OOS trades 39 (+0), IS return 72.35% (+5.70pp)
- perturb_risk_all_1.25: OOS return 36.04% (+2.30pp), OOS trades 39 (+0), IS return 70.26% (+3.61pp)
- perturb_strategy__momentum__trail.trail_activation_bars__8: OOS return 34.90% (+1.16pp), OOS trades 39 (+0), IS return 66.69% (+0.03pp)
- perturb_strategy_active_cumulative__trend__setup.min_setup_score_a__2.255: OOS return 35.27% (+1.53pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- perturb_strategy_active_cumulative__trend__setup.min_setup_score_a__2.46: OOS return 35.27% (+1.53pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- perturb_strategy_active_cumulative__breakout__setup.body_ratio_min__0.715: OOS return 35.87% (+2.13pp), OOS trades 38 (-1), IS return 67.54% (+0.89pp)
- perturb_risk_all_1.20: OOS return 35.11% (+1.37pp), OOS trades 39 (+0), IS return 68.85% (+2.20pp)

## Targeted Repair Winners
- targeted_momentum_sol_both: OOS return 29.92% (-3.82pp), OOS trades 50 (+11), IS return 80.35% (+13.69pp)
- targeted_momentum_sol_long_only: OOS return 31.97% (-1.77pp), OOS trades 44 (+5), IS return 73.98% (+7.33pp)
- targeted_breakout_faster_trail: OOS return 34.00% (+0.27pp), OOS trades 39 (+0), IS return 67.10% (+0.45pp)
- targeted_momentum_quick_exit_less_loss: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- targeted_momentum_quick_exit_flat: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- targeted_momentum_proof_lock_045_b2: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 65.63% (-1.02pp)
- targeted_momentum_proof_lock_065_b4: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 74.21% (+7.56pp)
- targeted_trend_mfe_lock_075_floor010: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- targeted_trend_funding_filter_on: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)
- targeted_breakout_quick_exit_on: OOS return 33.74% (+0.00pp), OOS trades 39 (+0), IS return 66.65% (+0.00pp)

## Checkpoint Lessons
- checkpoint_trend_round_2_others_current: OOS return 4.23% (-29.51pp), OOS trades 64 (+25), IS return 129.79% (+63.13pp)
- checkpoint_trend_round_1_others_current: OOS return 4.99% (-28.75pp), OOS trades 64 (+25), IS return 114.95% (+48.29pp)
- checkpoint_all_strategies_round_2_plus_portfolio_round2: OOS return -12.79% (-46.53pp), OOS trades 62 (+23), IS return 126.24% (+59.58pp)
- checkpoint_all_strategies_round_1_plus_portfolio_round2: OOS return -10.32% (-44.06pp), OOS trades 61 (+22), IS return 82.60% (+15.94pp)
- checkpoint_breakout_round_2_others_current: OOS return 20.30% (-13.44pp), OOS trades 37 (-2), IS return 67.39% (+0.74pp)

## Recommended Next Action
Promote only after a fresh full backtest/replay: perturb_strategy_active_cumulative__trend__setup.orderly_max_body_frac__1.08 currently has the best OOS/IS trade-off under this diagnostic objective.
Policy / overrides:
```json
{
  "policy": {
    "strategy.momentum.symbol_filter.sol_direction": "short_only",
    "strategy.momentum.exits.proof_lock_trigger_r": 0.55,
    "strategy.momentum.exits.proof_lock_min_bars": 3,
    "portfolio.symbol_collision": "cap",
    "portfolio.symbol_exposure_cap_R": 2.5,
    "risk_scale.momentum": 1.15,
    "risk_scale.trend": 1.15,
    "risk_scale.breakout": 1.15,
    "portfolio.dd_tiers": [
      [
        0.06,
        0.75
      ],
      [
        0.09,
        0.5
      ],
      [
        0.12,
        0.25
      ],
      [
        0.15,
        0.0
      ]
    ]
  },
  "base_overrides": {
    "trend": {
      "setup.orderly_max_body_frac": 1.08
    }
  },
  "strategy_sources": {}
}
```
