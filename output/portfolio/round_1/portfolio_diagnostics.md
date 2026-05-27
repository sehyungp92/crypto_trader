# Live-Config Portfolio Parity Refresh

Date range: 2025-12-01 to 2026-04-30
Initial equity: $25,000.00

## Individual
- momentum: trades=31, net=$2,759.70, return=11.04%, expectancy_r=0.3704, exit_eff=0.4201, terminal_marks=0
- trend: trades=51, net=$12,803.34, return=51.21%, expectancy_r=0.3932, exit_eff=0.5817, terminal_marks=0
- breakout: trades=16, net=$6,429.25, return=25.72%, expectancy_r=0.7042, exit_eff=0.6368, terminal_marks=0

## Portfolio
- trades=90, net=$23,214.58, return=92.86%, expectancy_r=0.4057, exit_eff=0.5409, terminal_marks=0
- rule_checks=112, approved=91, blocked=21
- blocked_by_reason={'symbol_collision=block: SOL already held by trend': 7, 'symbol_collision=block: BTC already held by trend': 6, 'symbol_collision=block: ETH already held by trend': 5, 'symbol_collision=block: BTC already held by momentum': 2, 'symbol_collision=block: SOL already held by momentum': 1}

## Reconciliation
- individual_total_trades=98
- portfolio_total_trades=90
- trade_count_delta=-8
- portfolio_vs_individual_net_profit_delta=$1,222.29

## Diagnostics
======================================================================
  STRATEGY DIAGNOSTICS
  90 realized trades | 0 terminal marks | 2026-05-25 03:23 UTC
======================================================================

══════════════════════════════════════════════════════════════════════
  1. Overview
══════════════════════════════════════════════════════════════════════
  Closed Trades:  90  (W:49 / L:41)
  Win Rate:       54.4%
  Profit Factor:  3.41
  Mean R:         +0.406
  Median R:       +0.120
  Total R:        +36.51
  Avg Hold:       3.6h  (13.4 bars)
  Realized P&L:   $23,214.58  (+92.86%)
  Net Liq P&L:    $23,214.58  (+92.86%)

  ✓ No critical flags

══════════════════════════════════════════════════════════════════════
  R-Multiple Distribution
══════════════════════════════════════════════════════════════════════
          < -1.0 |   6 ███████
    -1.0 to -0.5 |   3 ███
       -0.5 to 0 |  32 ████████████████████████████████████████
        0 to 0.5 |  14 █████████████████
      0.5 to 1.0 |  10 ████████████
      1.0 to 2.0 |  20 █████████████████████████
           > 2.0 |   5 ██████

  Mean: +0.406  Median: +0.120  Std: 0.929  Skew: +0.70

══════════════════════════════════════════════════════════════════════
  2. Winner vs Loser Profiles
══════════════════════════════════════════════════════════════════════
  Metric                          Winners     Losers      Delta
  --------------------------------------------------------------
  Avg R                             1.049     -0.363     +1.412
  Avg P&L ($)                     670.676   -235.330   +906.007
  Avg Hold (h)                      3.954      3.165     +0.789
  Avg Bars Held                    14.408     12.268     +2.140
  Avg MFE R                         1.800      0.427     +1.372
  Avg MAE R                        -0.240     -0.423     +0.183

  Setup Grade Distribution:
    A-grade:  W=2  L=4
    B-grade:  W=47  L=37

  Confirmation Distribution:
                       hammer:  W=2  L=1
             inside_bar_break:  W=7  L=6
        micro_structure_shift:  W=4  L=5
                 model1_close:  W=8  L=3
                model2_retest:  W=3  L=2
                         none:  W=25  L=24

══════════════════════════════════════════════════════════════════════
  3. MFE/MAE & Capture Efficiency
══════════════════════════════════════════════════════════════════════
  Overall (87 trades with MFE data):
    Avg MFE:               1.215R
    Avg MAE:               -0.323R
    Winner capture efficiency: 54.1%
    All-trades capture:       -79.8%
    Winner avg giveback:      0.751R

  Capture by Exit Reason:
    Exit Reason               n  Avg MFE    Avg R  Capture  Giveback
    ----------------------------------------------------------------
    protective_stop          53    1.269   +0.539   -120%    +0.730
    trailing_stop            18    1.352   +0.451     -6%    +0.901
    scratch_exit             13    0.437   -0.132    -33%    +0.569
    reversal_candle           1    5.317   +3.481     65%    +1.836
    structure_break           1    2.764   +1.387     50%    +1.377
    followthrough_exit        1    0.374   -0.592   -158%    +0.965

  Winner Giveback: avg +0.751R (range 0.150 to 3.328)
  Losers with positive MFE: 38/41 (avg peak 0.461R before reversal)

══════════════════════════════════════════════════════════════════════
  4. Stop Calibration
══════════════════════════════════════════════════════════════════════
  Stop exits: 74/90 (82%)
      n=74, WR=62.2%, Mean R=+0.459, Median R=+0.344, PF=3.99, Total R=+33.95
    Avg MAE at stop: -0.326R
    Median MAE:      -0.212R

  Non-stop exits: 16/90
      n=16, WR=18.8%, Mean R=+0.160, Median R=-0.071, PF=1.19, Total R=+2.56

  MAE Distribution (stop tightness check):
    MAE <= -1.0R:  0% of trades
    MAE <= -0.5R:  29% of trades
    → Stops rarely hit -1R but many stop exits — consider if trailing is too tight

══════════════════════════════════════════════════════════════════════
  5. Exit Reason Attribution
══════════════════════════════════════════════════════════════════════
  Exit Reason               n     WR    Avg R    Med R     PF      $ P&L  Share
  --------------------------------------------------------------------------
  protective_stop          56   62%   +0.461   +0.466   3.81 $20,024.57   86%
  trailing_stop            18   61%   +0.451   +0.192   6.36 $ 2,817.32   12%
  scratch_exit             13    8%   -0.132   -0.089   0.02 $-1,735.49   -7%
  reversal_candle           1  100%   +3.481   +3.481    inf $ 1,172.45    5%
  structure_break           1  100%   +1.387   +1.387    inf $ 1,163.07    5%
  followthrough_exit        1    0%   -0.592   -0.592   0.00 $  -227.34   -1%

══════════════════════════════════════════════════════════════════════
  6. Streak Analysis
══════════════════════════════════════════════════════════════════════
  Max win streak:   9  (best run: +10.05R)
  Max loss streak:  5  (worst run: -2.46R)
  Trend: DEGRADING (1st half: +0.569R, 2nd half: +0.242R)

  Sequence: WWWLWLLWWWWWWWWWLWWLLWWLLLWWWLLWWWWLWLWWLWWLWWLWLLLLWWLWWLLLLLWWWWWWLLLWLLWLLLLWLLWWLLWWLL

══════════════════════════════════════════════════════════════════════
  7. R-Curve Drawdown
══════════════════════════════════════════════════════════════════════
  Max R-drawdown:  -3.14R  (trade #86)
  Peak before DD:  +37.99R
  Final cum R:     +36.51R
  Recovery:        NOT YET — still -1.48R below peak

  Drawdown Episodes (> 0.3R):
    Trades #4-#5: -0.39R over 1 trades
    Trades #20-#22: -0.49R over 2 trades
    Trades #24-#27: -0.34R over 3 trades
    Trades #30-#32: -1.15R over 2 trades
    Trades #38-#40: -1.12R over 2 trades
    Trades #41-#45: -1.11R over 4 trades
    Trades #49-#53: -1.24R over 4 trades
    Trades #58-#64: -1.30R over 6 trades
    Trades #69-#90: -3.14R over 21 trades

══════════════════════════════════════════════════════════════════════
  8. Rolling Expectancy
══════════════════════════════════════════════════════════════════════
  Window: 10-trade rolling average
  Current:  -0.001R
  Best:     +1.004R  (at trade #17)
  Worst:    -0.220R  (at trade #86)

  Rolling R sparkline:
     +1.00 |   █                   
           |   ██        █         
           |  ████  █    █   █     
           |  ████████  ██  ███    
           |  █████████████████    
     -0.22 |  █████████████████████

══════════════════════════════════════════════════════════════════════
  9. Per-Asset Breakdown
══════════════════════════════════════════════════════════════════════

  BTC:
      n=27, WR=55.6%, Mean R=+0.371, Median R=+0.164, PF=2.77, Total R=+10.02
    Long:  n=25, WR=52%, avg R=+0.333
    Short: n=2, WR=100%, avg R=+0.853
    Core:         n=26 WR=54% R=+0.32
    Relaxed body: n= 1 WR=100% R=+1.68
    Avg MFE: 1.232R, Avg MAE: -0.319R

  ETH:
      n=34, WR=61.8%, Mean R=+0.562, Median R=+0.328, PF=7.13, Total R=+19.09
    Long:  n=22, WR=64%, avg R=+0.594
    Short: n=12, WR=58%, avg R=+0.503
    Core:         n=29 WR=62% R=+0.57
    Relaxed body: n= 5 WR=60% R=+0.51
    Avg MFE: 1.271R, Avg MAE: -0.291R

  SOL:
      n=29, WR=44.8%, Mean R=+0.255, Median R=-0.026, PF=2.09, Total R=+7.39
    Long:  n=16, WR=38%, avg R=+0.100
    Short: n=13, WR=54%, avg R=+0.446
    Core:         n=28 WR=43% R=+0.22
    Relaxed body: n= 1 WR=100% R=+1.14
    Avg MFE: 1.008R, Avg MAE: -0.365R

══════════════════════════════════════════════════════════════════════
  10. Direction Analysis
══════════════════════════════════════════════════════════════════════

  LONG:
      n=63, WR=52.4%, Mean R=+0.365, Median R=+0.040, PF=2.95, Total R=+22.98
    Exits: protective_stop=36, trailing_stop=15, scratch_exit=9, reversal_candle=1, structure_break=1, followthrough_exit=1

  SHORT:
      n=27, WR=59.3%, Mean R=+0.501, Median R=+0.616, PF=4.70, Total R=+13.54
    Exits: protective_stop=20, scratch_exit=4, trailing_stop=3

══════════════════════════════════════════════════════════════════════
  11. Confirmation Type Analysis
══════════════════════════════════════════════════════════════════════
  Type                         n     WR    Avg R    Med R  Total R      $ P&L
  --------------------------------------------------------------------
  none                        49   51%   +0.360   +0.040   +17.64 $12,949.61
  inside_bar_break            13   54%   +0.238   +0.037    +3.09 $ 1,561.53
  model1_close                11   73%   +0.629   +0.616    +6.92 $ 3,978.55
  micro_structure_shift        9   44%   +0.206   -0.012    +1.85 $   608.69
  model2_retest                5   60%   +0.870   +1.565    +4.35 $ 3,176.04
  hammer                       3   67%   +0.886   +1.095    +2.66 $   940.15

══════════════════════════════════════════════════════════════════════
  12. Confluence Count → Outcome
══════════════════════════════════════════════════════════════════════
   Confluences    n     WR    Avg R  Total R
  --------------------------------------------
             0    3   67%   +0.287    +0.86
             1    8   75%   +0.784    +6.27
             2   28   54%   +0.387   +10.84
             3   34   44%   +0.198    +6.74
             4    3   33%   +0.101    +0.30
             5    2  100%   +1.988    +3.98
             6    6   50%   +0.415    +2.49
             7    4   75%   +0.511    +2.04
             8    2  100%   +1.491    +2.98

  ✗ Non-monotonic: confluence count is NOT reliably predictive

══════════════════════════════════════════════════════════════════════
  13. Session & Timing Patterns
══════════════════════════════════════════════════════════════════════
  Entry Hour (UTC):
      Hour    n     WR    Avg R      $ P&L
    --------------------------------------
       0h    3   67%   +0.688 $ 1,254.26
       1h    3   67%   +0.402 $   713.98
       2h    2    0%   -0.151 $  -106.65
       3h    8   50%   +0.421 $ 2,797.89
       4h   10   30%   +0.130 $   789.39
       5h    6   67%   +0.723 $ 3,633.21
       6h    5  100%   +0.657 $ 2,310.47
       7h    4   50%   +0.874 $ 1,160.46
       8h    3   67%   +0.532 $ 1,360.75
       9h    3   67%   +1.159 $ 2,681.08
      10h    4  100%   +1.101 $ 2,003.97
      11h    3   67%   +0.871 $ 1,675.09
      12h    2    0%   -0.649 $  -610.26
      13h    3    0%   -0.284 $  -698.16
      14h    9   78%   +0.735 $ 3,496.01
      15h    5   60%   +0.201 $   277.50
      16h    2   50%   -0.500 $  -439.30
      17h    2   50%   +0.446 $   785.27
      18h    1  100%   +1.565 $ 1,084.48
      19h    2   50%   -0.175 $  -144.03
      20h    3   33%   -0.096 $  -114.05
      21h    2   50%   +0.411 $   884.79
      22h    2    0%   -0.586 $-1,077.71
      23h    3   33%   -0.090 $  -503.88

  Day of Week:
       Day    n     WR    Avg R      $ P&L
    --------------------------------------
       Mon    6   33%   -0.433 $-2,204.51
       Tue   19   68%   +0.567 $ 7,443.86
       Wed   13   46%   +0.054 $ 2,025.67
       Thu   16   56%   +0.511 $ 2,937.47
       Fri   20   60%   +0.777 $ 9,546.40
       Sat   11   18%   -0.126 $  -401.92
       Sun    5  100%   +1.064 $ 3,867.61

  Session Performance:
    Asia:
        n=41, WR=53.7%, Mean R=+0.457, Median R=+0.051, PF=4.99, Total R=+18.76
    London:
        n=15, WR=66.7%, Mean R=+0.719, Median R=+1.081, PF=4.65, Total R=+10.79
    Overlap:
        n=17, WR=58.8%, Mean R=+0.398, Median R=+0.260, PF=3.28, Total R=+6.77
    New York:
        n=10, WR=50.0%, Mean R=+0.082, Median R=+0.013, PF=1.86, Total R=+0.82
    Off-hours:
        n=7, WR=28.6%, Mean R=-0.089, Median R=-0.253, PF=0.62, Total R=-0.62

══════════════════════════════════════════════════════════════════════
  14. Trade Duration Analysis
══════════════════════════════════════════════════════════════════════
  Bars held:  avg=13.4, median=10, range=[1, 56]
  Hours:      avg=3.6h, median=3.0h

  Duration Buckets:
    ≤2 bars   : n=10, WR=40%, avg R=-0.197
    3-5 bars  : n=19, WR=32%, avg R=+0.156
    >5 bars   : n=61, WR=64%, avg R=+0.582

══════════════════════════════════════════════════════════════════════
  15. Concentration & Dependency Risk
══════════════════════════════════════════════════════════════════════
  Largest single winner: $1,864.64 (8% of total profit)
    Symbol: ETH, R=+2.62, Direction: LONG
  Top 9 trade(s) (9/49 = 18%): $12,059.34 (52% of total profit)
  Symbol concentration: ETH contributes $12,521.92 (54% of total)

  Concentration Risk: LOW

══════════════════════════════════════════════════════════════════════
  16. Worst Trades Autopsy
══════════════════════════════════════════════════════════════════════

  #1 worst: SOL LONG
    R-multiple:    -1.294
    P&L:           $-533.70
    Entry/Exit:    $86.25 → $85.91
    Bars held:     1
    Hold time:     0.2h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  inside_bar_break
    Confluences:   h1_ema50, prior_hl_flip
    MFE/MAE:       +0.000R / -0.746R
    Entry time:    2026-04-25 16:15 UTC

  #2 worst: BTC LONG
    R-multiple:    -1.209
    P&L:           $-473.79
    Entry/Exit:    $75,832.95 → $75,423.94
    Bars held:     4
    Hold time:     1.0h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  inside_bar_break
    Confluences:   prior_hl_flip
    MFE/MAE:       +0.056R / -0.749R
    Entry time:    2026-04-18 17:30 UTC

  #3 worst: ETH LONG
    R-multiple:    -1.170
    P&L:           $-444.62
    Entry/Exit:    $2,313.88 → $2,298.34
    Bars held:     3
    Hold time:     0.8h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  inside_bar_break
    Confluences:   h1_ema20, fib_zone
    MFE/MAE:       +0.200R / -0.628R
    Entry time:    2026-04-29 12:45 UTC

  #4 worst: ETH SHORT
    R-multiple:    -1.121
    P&L:           $-622.78
    Entry/Exit:    $1,995.80 → $2,014.41
    Bars held:     1
    Hold time:     0.5h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  model2_retest
    Confluences:   h4_alignment, volume_surge, lvn_runway, volume_contraction, ema_support, poc_alignment, multi_hvn
    MFE/MAE:       +0.000R / -0.813R
    Entry time:    2026-03-30 01:00 UTC

  #5 worst: BTC LONG
    R-multiple:    -1.118
    P&L:           $-1,036.18
    Entry/Exit:    $69,486.37 → $68,832.84
    Bars held:     3
    Hold time:     0.8h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  none
    Confluences:   h1_ema_zone, rsi_pullback
    MFE/MAE:       +0.085R / -0.568R
    Entry time:    2026-04-06 22:00 UTC

  Common Patterns in Worst Trades:
    Exit reasons: protective_stop appears 5x
    Confirmations: inside_bar_break appears 3x
    Symbols: BTC appears 2x

══════════════════════════════════════════════════════════════════════
  17. Interaction Analysis
══════════════════════════════════════════════════════════════════════
  Grade x Direction:
                             Long              Short
    ------------------------------------------------
       A-grade  n= 6 WR=33% R=+0.33                 --
       B-grade  n=57 WR=54% R=+0.37  n=27 WR=59% R=+0.50

  Confirmation x Symbol:
                                       BTC          ETH          SOL
    ----------------------------------------------------------------
                       hammer   2/100%/+1.89           --   1/ 0%/-1.11
             inside_bar_break   4/50%/-0.22   4/75%/+0.65   5/40%/+0.28
        micro_structure_shift   2/ 0%/-0.32   2/ 0%/-0.04   5/80%/+0.51
                 model1_close   3/100%/+0.62   6/67%/+0.69   2/50%/+0.46
                model2_retest   1/100%/+1.68   3/67%/+1.02   1/ 0%/-0.39
                         none  15/47%/+0.28  19/63%/+0.49  15/40%/+0.27

  Symbol x Direction x Signal Variant (Core):
    Symbol                   Long                Short
    ----------------------------------------------------
    BTC        n=24 WR=50% R=+0.28  n= 2 WR=100% R=+0.85
    ETH        n=20 WR=60% R=+0.51   n= 9 WR=67% R=+0.70
    SOL        n=16 WR=38% R=+0.10   n=12 WR=50% R=+0.39

  Symbol x Direction x Signal Variant (Relaxed Body):
    Symbol                   Long                Short
    ----------------------------------------------------
    BTC       n= 1 WR=100% R=+1.68    n= 0 WR= -- R=  --
    ETH       n= 2 WR=100% R=+1.42   n= 3 WR=33% R=-0.10
    SOL         n= 0 WR= -- R=  --  n= 1 WR=100% R=+1.14

══════════════════════════════════════════════════════════════════════
  18. Friction Analysis (Commission + Funding)
══════════════════════════════════════════════════════════════════════
  Total commissions:   $  5,049.15  (avg $56.10/trade)
  Total funding:       $     51.99  (avg $0.58/trade)
  Total friction:      $  5,101.14
  Net P&L:             $ 23,214.58
  Gross P&L (pre-fric):$ 28,315.72
  Friction as % of gross: 18.0%

  Per-Symbol Friction:
    Symbol    n       Comm    Funding      Total
    --------------------------------------------
    BTC      27 $ 1,789.22 $     9.16 $ 1,798.38
    ETH      34 $ 1,732.58 $    13.71 $ 1,746.29
    SOL      29 $ 1,527.35 $    29.11 $ 1,556.46

  Funding direction: 48 adverse, 41 favorable, 1 neutral

══════════════════════════════════════════════════════════════════════
  19. Weekly P&L Calendar
══════════════════════════════════════════════════════════════════════
  Week         n    WR        R        P&L    Cum P&L                  Bar
  ----------------------------------------------------------------------
  2026-W03     1 100%   +0.74 $   262.11 $   262.11  
  2026-W04     2 100%   +2.02 $   723.48 $   985.59  ++
  2026-W05     3  33%   +1.12 $   793.28 $ 1,778.86  ++
  2026-W07     2  50%   +2.39 $ 1,707.60 $ 3,486.46  +++++
  2026-W09     1 100%   +0.16 $   124.83 $ 3,611.29  
  2026-W10     3 100%   +3.10 $ 2,260.84 $ 5,872.13  ++++++
  2026-W11     3 100%   +4.12 $ 2,858.23 $ 8,730.36  ++++++++
  2026-W12     8  62%   +4.05 $ 3,126.25 $11,856.62  +++++++++
  2026-W13     7  43%   +2.82 $ 2,044.26 $13,900.88  +++++
  2026-W14     7  71%   +3.10 $ 2,830.75 $16,731.63  ++++++++
  2026-W15    16  50%   +6.25 $ 1,772.47 $18,504.10  +++++
  2026-W16    20  50%   +6.86 $ 5,122.03 $23,626.13  +++++++++++++++
  2026-W17    12  33%   -0.71 $  -549.37 $23,076.77  -
  2026-W18     5  40%   +0.49 $   137.81 $23,214.58  

  Positive weeks: 13/14  Negative weeks: 1/14
  Best week: $5,122.03  Worst week: $-549.37

══════════════════════════════════════════════════════════════════════
  20. Best Trades Autopsy
══════════════════════════════════════════════════════════════════════

  #1 best: ETH LONG
    R-multiple:    +3.481
    P&L:           $1,172.45
    Entry/Exit:    $2,183.85 → $2,233.36
    Bars held:     56
    Hold time:     14.0h
    Exit reason:   reversal_candle
    Grade:         B
    Confirmation:  inside_bar_break
    Confluences:   prior_hl_flip
    MFE/MAE:       +5.317R / -0.622R  (captured 65%)
    Entry time:    2026-04-10 07:15 UTC

  #2 best: BTC LONG
    R-multiple:    +2.677
    P&L:           $881.50
    Entry/Exit:    $70,781.69 → $71,905.74
    Bars held:     35
    Hold time:     8.8h
    Exit reason:   trailing_stop
    Grade:         B
    Confirmation:  hammer
    Confluences:   h1_ema50
    MFE/MAE:       +6.005R / -0.724R  (captured 45%)
    Entry time:    2026-04-09 14:00 UTC

  #3 best: ETH LONG
    R-multiple:    +2.616
    P&L:           $1,864.64
    Entry/Exit:    $1,960.19 → $2,027.98
    Bars held:     12
    Hold time:     6.0h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  model2_retest
    Confluences:   lvn_runway, volume_contraction, ema_support, poc_alignment, multi_hvn
    MFE/MAE:       +3.166R / -0.588R  (captured 83%)
    Entry time:    2026-02-13 09:30 UTC

  #4 best: ETH LONG
    R-multiple:    +2.101
    P&L:           $1,259.07
    Entry/Exit:    $2,258.46 → $2,316.26
    Bars held:     5
    Hold time:     2.5h
    Exit reason:   protective_stop
    Grade:         B
    Confirmation:  model1_close
    Confluences:   h4_alignment, volume_surge, lvn_runway, balance_duration, volume_contraction, ema_support, poc_alignment, multi_hvn
    MFE/MAE:       +2.666R / -0.221R  (captured 79%)
    Entry time:    2026-04-11 17:00 UTC

  #5 best: SOL LONG
    R-multiple:    +2.085
    P&L:           $754.13
    Entry/Exit:    $87.45 → $89.35
    Bars held:     26
    Hold time:     6.5h
    Exit reason:   trailing_stop
    Grade:         B
    Confirmation:  micro_structure_shift
    Confluences:   prior_hl_flip, fib_zone
    MFE/MAE:       +3.819R / -0.102R  (captured 55%)
    Entry time:    2026-04-17 10:45 UTC

  Common Patterns in Best Trades:
    Exit reasons: trailing_stop appears 2x
    Symbols: ETH appears 3x

══════════════════════════════════════════════════════════════════════
  21. Risk & Sizing Analysis
══════════════════════════════════════════════════════════════════════
  Position value at entry:
    Avg:    $ 62,298.02
    Median: $ 58,083.38
    Range:  $ 12,331.20 — $147,909.24

  Geometric R-Multiple vs Dollar P&L Alignment:
    8 trade(s) where R and $ P&L disagree:
      ETH LONG: R=+0.042, P&L=$-8.15
      SOL SHORT: R=+0.003, P&L=$-41.53
      SOL SHORT: R=+0.057, P&L=$-22.82
      ETH LONG: R=+0.066, P&L=$-4.24
      SOL SHORT: R=+0.116, P&L=$-8.61
      BTC LONG: R=+0.086, P&L=$-4.06
      ETH LONG: R=+0.069, P&L=$-60.32
      ETH LONG: R=+0.062, P&L=$-8.62
    R vs $ rank correlation: 0.977 (strong)
    Worst-3 overlap ($ vs R): 0/3 (sizing may distort)

══════════════════════════════════════════════════════════════════════
  22. Entry Method Analysis
══════════════════════════════════════════════════════════════════════
  Method                  n     WR    Avg R    Med R     PF      $ P&L
  ------------------------------------------------------------------
  aggressive             49   51%   +0.360   +0.040   3.22 $12,949.61
  close                  25   52%   +0.304   +0.037   2.21 $ 3,110.38
  model1_close           11   73%   +0.629   +0.616  12.27 $ 3,978.55
  model2_retest           5   60%   +0.870   +1.565   4.56 $ 3,176.04

  Winner Capture Efficiency by Method:
    model2_retest        80.1%
    model1_close         57.5%
    aggressive           55.3%
    close                43.7%

══════════════════════════════════════════════════════════════════════
  23. Blocked Relaxed-Body Audit
══════════════════════════════════════════════════════════════════════
  No blocked relaxed-body signals recorded.

══════════════════════════════════════════════════════════════════════
  VERDICT & RECOMMENDATIONS
══════════════════════════════════════════════════════════════════════
  Net liquidation P&L: $23,214.58 (+92.86%)
  Realized closed-trade P&L: $23,214.58

  STRENGTHS:
    + Strong profit factor (3.41)
    + Good expectancy (+0.406R)

  WEAKNESSES:
    - High giveback (+0.751R avg) — exits too late or trail too slow
    - micro_structure_shift confirmation underperforms (44% WR, +0.206R)

  RECOMMENDATIONS:
    → No critical changes recommended — continue monitoring
