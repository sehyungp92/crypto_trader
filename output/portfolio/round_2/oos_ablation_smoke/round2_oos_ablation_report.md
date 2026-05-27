# Portfolio Round 2 OOS Ablation And Perturbation Report

Generated: 2026-05-27T11:54:17.484478+00:00
IS window: 2026-02-25 to 2026-04-20
OOS window: 2026-04-21 to 2026-05-23
Candidates evaluated: 1

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
- No candidate improved OOS while preserving IS within the configured tolerance.

## Ablation Winners
- None passed the IS preservation screen with positive OOS uplift.

## Perturbation Winners
- None passed the IS preservation screen with positive OOS uplift.

## Targeted Repair Winners
- None passed the IS preservation screen with positive OOS uplift.

## Checkpoint Lessons
- None passed the IS preservation screen with positive OOS uplift.

## Recommended Next Action
Do not promote a repair from this run; use the ablation evidence to design a narrower next search.
