# crypto_trader

Systematic crypto trading system with three independent alpha strategies, a shared backtest engine, and a phased auto-optimisation framework. Trades BTC, ETH, and SOL perpetual futures.

## Strategies

### 1. Momentum Pullback

**Timeframes:** H4 + H1 (directional bias) → M15 (execution)

**Edge:** Captures trend continuation after Fibonacci retracement pullbacks (0.382–0.618 zone) within confirmed momentum regimes. The alpha comes from buying exhausted pullbacks—where retail participants panic-sell into predictable support—when a 12-step sequential gate pipeline confirms that the impulse structure is intact.

**Entry conditions:**
- H4/H1 bias requires ≥2 of: EMA alignment, ADX >15, slope confirmation, structural HH/HL
- Price must sit inside the 0.382–0.618 Fib retracement of a prior 20-bar impulse
- RSI pullback filter confirms exhausted selling
- Candlestick confirmation at the zone (engulfing, hammer, EMA reclaim, or structure break) with volume gate
- Graded A/B by confluence count and R-room to target

**Trade management:** TP1 at 1.2R (16% off), TP2 at 2.5R (20% off), smart breakeven (entry + 0.2R), R-adaptive trailing stop that tightens as profit grows, quick exit on stagnation, thesis-based exits (structure break, reversal candle, adverse funding).

---

### 2. Institutional Anchor Pro (Trend Continuation)

**Timeframes:** D1 (regime classification) → H1 (setup + entry)

**Edge:** Exploits persistent higher-timeframe trends by layering a D1 structural regime classifier (ADX + EMA alignment + swing HH/HL detection) over H1 impulse-pullback entries. The alpha is that crypto trends persist longer than most participants expect due to reflexive momentum and thin order books; the D1 structure tracker filters out mean-reverting ranges where trend-following bleeds.

**Entry conditions:**
- D1 regime must show ADX ≥12, EMA alignment, and structural HH/HL progression (A or B tier)
- H1 must show a completed impulse (≥0.8 ATR move) followed by a measured pullback (≤75% retracement, RSI 30–65)
- Six confluence sources scored: EMA zone, structure support, ADX confirmation, pullback quality, weekly H/L room, volume pattern
- Trigger confirmation (engulfing, EMA reclaim, or structure break) on H1
- Graded A/B by confluence score and R-room

**Trade management:** TP1 at 0.8R (25% off—early lock to reduce DD), TP2 at 2.0R (50% off, 25% runner), R-adaptive trail (ceiling 1.5R), time stop at 16 bars, EMA failsafe exit.

---

### 3. Volume Profile Breakout

**Timeframes:** M30 (volume profile + balance zone + breakout detection) → H4 (directional context)

**Edge:** Exploits market microstructure via volume profile analysis. Price consolidates around High Volume Nodes (HVN) forming balance zones, then breaks out through Low Volume Nodes (LVN) where resting liquidity is thin. The alpha is that breakouts through LVN "runways" face structurally less resistance and travel farther before finding the next HVN equilibrium—an auction-theory edge invisible to price-action-only systems.

**Entry conditions:**
- Volume profile computed from M30 bars (50 bins, 36-bar lookback): identifies POC, value area (VAH/VAL), HVN (≥1.2x avg volume), and LVN (≤0.5x avg volume)
- Balance zone must form around HVN (≥4 bars, ≥2 touches, within 1.2 ATR width)
- Breakout scored on 8 confluences: EMA alignment, ADX strength, LVN runway ahead, volume surge (≥1.3x), H4 context alignment, body ratio, breakout distance, R-room
- Graded A+/A/B by confluence count; three-tier position sizing
- Invalidation exit if price re-enters balance zone by ≥1.2 ATR (thesis failure)

**Trade management:** TP1 at 0.8R (30% off), TP2 at 2.0R (40% off, 30% runner), smart BE (entry + 0.525R), R-adaptive trail (activates at 0.3R or 4 bars), aggressive quick exit (4 bars) for failed breakouts, invalidation exit unique to this strategy.

---

## Shared Infrastructure

| Component | Description |
|---|---|
| **SimBroker** | Simulated exchange with margin, leverage, partial fills, funding rates, liquidation |
| **HistoricalFeed** | Multi-timeframe bar emission (D1 → H4 → H1 → M30 → M15 per boundary) from Parquet data |
| **R-adaptive trail** | `buffer = wide*(1-r) + tight*r` where `r = clamp(current_r/ceiling, 0, 1)` — tightens as profit grows |
| **Greedy optimiser** | 6-phase forward selection with diagnostic-driven experiment suggestion, phase-specific scoring, gate criteria, and round continuity |
| **Diagnostics** | 22-section deep analysis: MFE capture, stop calibration, exit attribution, R-curve drawdown, rolling expectancy, concentration, and more |
| **Portfolio layer** | Per-strategy broker isolation, 9-rule cross-strategy entry gating, merged equity tracking |

## Quick Start

```bash
# Install
pip install -e .

# Backtest a single strategy
python -m crypto_trader backtest --strategy momentum --symbols BTC,ETH,SOL
python -m crypto_trader backtest --strategy trend --symbols BTC,ETH,SOL
python -m crypto_trader backtest --strategy breakout --symbols BTC,ETH,SOL

# Run optimisation
python -m crypto_trader optimize --strategy momentum --workers 2
python -m crypto_trader optimize --strategy trend --workers 2
python -m crypto_trader optimize --strategy breakout --workers 2

# Portfolio backtest (all strategies)
python -m crypto_trader backtest --portfolio --symbols BTC,ETH,SOL
```

## Project Structure

```
src/crypto_trader/
  core/           # Engine, models, broker protocol, data feed
  strategy/
    momentum/     # 14 modules - Fibonacci pullback strategy
    trend/        # 13 modules - D1 regime + H1 continuation
    breakout/     # 15 modules - Volume profile breakout
  backtest/       # Runner, metrics, diagnostics, analysis
  optimize/       # Greedy optimiser, plugins, parallel eval, scoring
  portfolio/      # Cross-strategy management and risk gating
  live/           # Live/paper trading engine
  instrumentation/ # Logging, metrics sinks, health reporting
tests/            # 1328 tests
```
