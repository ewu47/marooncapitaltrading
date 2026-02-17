# Strategy Report — Maroon Capital Trading System

## Table of Contents

1. [How the ATR Breakout Strategy Works](#1-how-the-atr-breakout-strategy-works)
2. [How ATR Kelly + Adaptive Works](#2-how-atr-kelly--adaptive-works)
3. [Fractional Quantity Mechanism for Backtests](#3-fractional-quantity-mechanism-for-backtests)
4. [Strategy Deployment Recommendations](#4-strategy-deployment-recommendations)

---

## 1. How the ATR Breakout Strategy Works

The ATR (Average True Range) Breakout strategy is the foundation of our trading system. It identifies momentum entries by detecting when price breaks out of a volatility-adjusted range.

### Core Concept

The strategy computes a rolling **ATR** value — a measure of how much an asset's price typically moves per bar — and uses it to set dynamic breakout thresholds above recent highs and below recent lows.

### Step-by-Step

1. **True Range (TR):** For each bar, compute the maximum of:
   - `|High - Low|`
   - `|High - Previous Close|`
   - `|Low - Previous Close|`

2. **ATR:** Take the rolling mean of TR over `atr_period` bars (default: 14).

3. **Breakout Levels:**
   - `prior_high` = highest High over the last `breakout_lookback` bars (default: 20)
   - `prior_low` = lowest Low over the last `breakout_lookback` bars
   - `upper_band = prior_high + atr_multiplier * ATR`
   - `lower_band = prior_low  - atr_multiplier * ATR`

4. **Signals:**
   - **BUY** when `Close > upper_band` (price breaks above the volatility-adjusted resistance)
   - **SELL** when `Close < lower_band` (price breaks below the volatility-adjusted support)
   - **HOLD** otherwise

5. **Position:** The signal is forward-filled, so once a BUY fires the system stays long until a SELL fires, and vice versa.

### Why It Works

ATR breakout captures momentum moves. By scaling the bands with volatility, the strategy:
- Avoids false breakouts during choppy, high-volatility periods (bands widen)
- Catches genuine trend starts during calm periods (bands tighten)

The base ATR strategy uses a **fixed multiplier** (e.g., 1.5) and a **fixed position size** (e.g., 10 shares per trade).

---

## 2. How ATR Kelly + Adaptive Works

The `ATRBreakoutAdaptiveKellyStrategy` combines two enhancements on top of the base ATR breakout:

1. **Adaptive (dynamic) ATR multiplier** — adjusts breakout band width based on the current volatility regime
2. **Kelly-fraction position sizing** — determines position size as a fraction of equity based on the volatility-implied stop distance

### 2.1 The Adaptive Multiplier Mechanism

Instead of a fixed multiplier, the adaptive version dynamically widens or tightens breakout bands based on where current ATR sits relative to its own recent history.

**Calculation:**

```
atr_mean = rolling mean of ATR over z_lookback bars (default: 100)
atr_std  = rolling std  of ATR over z_lookback bars
atr_z    = (ATR - atr_mean) / atr_std        # z-score of current ATR
dyn_mult = base_multiplier + z_scale * clip(atr_z, 0, max_z)
```

**Defaults:** `base_multiplier=1.2`, `z_scale=0.5`, `max_z=2.0`

**How it behaves:**

| Regime | ATR z-score | Dynamic Multiplier | Effect |
|--------|------------|-------------------|--------|
| Low volatility | ~0 (ATR near mean) | ~1.2 (base only) | Tighter bands, more trades captured |
| Normal volatility | ~0.5 | ~1.45 | Moderate band width |
| High volatility | ~2.0 (capped at max_z) | ~2.2 | Wider bands, fewer false breakouts |

The adaptive multiplier prevents the strategy from entering noisy, whipsaw markets while still capturing clean breakouts in trending conditions.

### 2.2 The Kelly Criterion Position Sizing Mechanism

The Kelly criterion determines the mathematically optimal fraction of bankroll to wager on a bet with a known edge. In our context, it sizes positions based on the ratio of tolerable risk to the implied stop distance.

**Calculation:**

```
stop_distance = dyn_mult * ATR                         # dollar stop distance
stop_frac     = stop_distance / price                  # stop as fraction of price
kelly_frac    = clip(risk_per_trade / stop_frac, min_frac, max_frac)
target_qty    = kelly_frac * equity                    # USD notional
```

**Defaults:** `risk_per_trade=0.01` (1%), `min_notional_frac=0.01` (1%), `max_notional_frac=0.20` (20%)

**How it behaves:**

| Market Condition | stop_frac | kelly_frac | Position (on $100k equity) |
|-----------------|-----------|------------|---------------------------|
| Calm trend (small ATR) | 0.5% | 2.0 → clipped to 0.20 (max) | $20,000 |
| Normal volatility | 1.0% | 1.0 → clipped to 0.20 (max) | $20,000 |
| High volatility | 3.0% | 0.33 (33%) → clipped to 0.20 | $20,000 |
| Very high volatility | 5.0% | 0.20 | $20,000 |
| Extreme volatility | 10.0% | 0.10 | $10,000 |

**Key insight:** When the stop distance is large (volatile market), the Kelly fraction shrinks and the position gets smaller. When the stop is tight (calm market), the fraction grows (up to the cap) and the position gets larger. This means:
- **Calm, trending markets** → larger positions to capture the trend
- **Volatile, choppy markets** → smaller positions to limit downside

The `target_qty` output is in **USD notional** (not shares or coins). The execution layer then converts to actual units depending on asset class.

### 2.3 Why the Combination Works

The adaptive multiplier and Kelly sizing reinforce each other:

- **Adaptive multiplier** reduces *entry risk* by filtering out false breakouts in high-vol regimes
- **Kelly sizing** reduces *position risk* by scaling down allocation in high-vol regimes

Together, this creates a strategy that is aggressive in favorable conditions and conservative in unfavorable ones — precisely what a robust trading system needs.

---

## 3. Fractional Quantity Mechanism for Backtests

### The Problem

The Kelly strategies output `target_qty` as **USD notional** (e.g., $20,000). The backtester must convert this to actual tradeable units:

- **Stocks (AAPL, SPY):** Must be **whole integer shares** — you cannot buy 73.5 shares of AAPL
- **Crypto (BTC, ETH):** Can be **fractional units** — buying 0.21 BTC is perfectly valid

### The Solution: `--fractional` Flag

The backtester accepts a `fractional_qty` parameter controlled by the `--fractional` CLI flag:

| Flag | Conversion | Use Case |
|------|-----------|----------|
| Default (no flag) | `qty = int(notional / price)` | Stocks — floors to whole shares |
| `--fractional` | `qty = round(notional / price, 6)` | Crypto — keeps fractional units |

### Examples

**Stock backtest (AAPL at $271, $20k notional):**
```
int(20000 / 271) = 73 shares
```

**Crypto backtest (BTC at $97,000, $20k notional):**
```
round(20000 / 97000, 6) = 0.206186 BTC
```

### Usage

```bash
# Stock backtest — integer shares (default)
python3 run_backtest.py --csv data/AAPL_1Min.csv --strategy atr_kelly --equity 100000 --capital 100000

# Crypto backtest — fractional units
python3 run_backtest.py --csv data/BTC-USD_5m.csv --strategy atr_adaptive_kelly --equity 100000 --capital 100000 --fractional
```

The `--fractional` flag is also available in the Monte Carlo runner (`run_monte_carlo copy.py`).

---

## 4. Strategy Deployment Recommendations

### BTC-USD (5-Minute Timeframe) → ATR Adaptive Kelly

The `atr_adaptive_kelly` strategy is best suited for BTC-USD on 5-minute bars because:

1. **High volatility:** Bitcoin frequently exhibits large ATR swings. The adaptive multiplier dynamically adjusts bands to avoid entering during short-lived spikes while still capturing sustained breakouts.

2. **Regime changes:** Crypto markets alternate between tight consolidation and explosive moves. The adaptive z-score mechanism detects these regime shifts and adjusts the breakout threshold accordingly.

3. **Kelly sizing scales with risk:** When BTC's 5-minute ATR is high (e.g., during a liquidation cascade), Kelly automatically reduces position size. When ATR is calm during a trending phase, Kelly allocates more capital to ride the trend.

4. **Fractional quantities:** BTC trades at ~$97,000+ per coin. With $100k equity and a 20% max Kelly fraction, the maximum allocation is $20,000 — approximately 0.21 BTC. The fractional execution system handles this correctly.

**Limitation — No Shorting on Alpaca:** Alpaca's paper and live trading API does not support short-selling crypto. This means the strategy can only act on **BUY signals** for BTC-USD. SELL signals are generated by the strategy but blocked at the execution layer with a `crypto shorting disabled` message. This effectively makes it a **long-only** crypto strategy — it enters on upward breakouts and exits when the position is closed, but cannot profit from downward moves.

### SP500 / Stocks (1-Minute Timeframe) → Standard ATR Kelly or ATR Breakout

For equities like SPY or individual stocks on 1-minute timeframes, the standard `atr_kelly` strategy (without the adaptive multiplier) is recommended because:

1. **Shorter timeframe = more noise:** On 1-minute bars, the adaptive z-score calculation over 100 bars represents less than 2 hours of data. The z-score oscillates rapidly and can destabilize the multiplier. The fixed multiplier is more reliable at this granularity.

2. **Stocks support both long and short:** Unlike crypto on Alpaca, equities can be shorted. The full signal set (BUY and SELL) is actionable, making it more effective than the long-only crypto constraint.

3. **Lower volatility:** Stocks like SPY and AAPL have significantly lower per-bar volatility than BTC. The Kelly fraction calculations still provide appropriate position sizing, but the adaptive band adjustment adds less value when volatility is already relatively stable.

4. **Integer share execution:** Stock orders must be in whole shares. The default (non-fractional) backtester mode handles this automatically by flooring `notional / price` to an integer.

### Summary Table

| Asset | Timeframe | Strategy | `--fractional` | Shorting |
|-------|-----------|----------|---------------|----------|
| BTC-USD | 5-min | `atr_adaptive_kelly` | Yes | Long-only (Alpaca restriction) |
| SPY / AAPL | 1-min | `atr_kelly` | No (default) | Long and short |

### Example Commands

```bash
# BTC-USD 5-min adaptive Kelly backtest
python3 run_backtest.py \
  --csv data/legacy/clean_data_crypto/BTC-USD_5m_clean.csv \
  --strategy atr_adaptive_kelly \
  --equity 100000 --capital 100000 \
  --fractional --plot

# AAPL 1-min Kelly backtest
python3 run_backtest.py \
  --csv data/legacy/clean_data/AAPL_1Min_alpaca_clean.csv \
  --strategy atr_kelly \
  --equity 100000 --capital 100000 \
  --plot

# BTC-USD live trading (dry run)
python3 run_live.py \
  --symbol BTCUSD --asset-class crypto --timeframe 5Min \
  --strategy atr_adaptive_kelly \
  --equity 100000 --dry-run

# SPY live trading
python3 run_live.py \
  --symbol SPY --strategy atr_kelly \
  --equity 100000 --live
```
