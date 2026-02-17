# Trading System

## What this includes
- Alpaca paper-data download + cleaning helpers
- A paper-trading runner that submits orders through Alpaca
- A lightweight backtester with market data gateway, order book, order manager, and matching engine
- Monte Carlo backtesting: execution randomness (fill/partial/cancel) and/or synthetic GBM price paths
- Automatic trade logging to `logs/trades.csv` and `logs/system.log`
- Built-in strategies:
  - **MovingAverageStrategy** (`ma`) - Simple moving average crossover
  - **TemplateStrategy** (`template`) - Momentum-based starter template
  - **CryptoTrendStrategy** (`crypto_trend`) - EMA trend follower for crypto (long-only)
  - **DemoStrategy** (`demo`) - Minimal demo strategy for testing
  - **ATRBreakoutStrategy** (`atr_breakout`) - ATR-adjusted high/low breakout
  - **ATRBreakoutOptimizedStrategy** (`atr_optimized`) - ATR breakout with trend/volume filters and trailing stop
  - **ATRBreakoutKellyStrategy** (`atr_kelly`) - ATR breakout with Kelly-lite volatility-adjusted sizing
  - **ATRBreakoutAdaptiveStrategy** (`atr_adaptive`) - ATR breakout with dynamic multiplier based on ATR z-score
  - **ATRBreakoutAdaptiveKellyStrategy** (`atr_adaptive_kelly`) - Adaptive multiplier + Kelly sizing (best overall)

## Strategy Performance Notes

| Strategy | Best Timeframe | Best Market | Notes |
|----------|---------------|-------------|-------|
| `atr_breakout` | 1-min | Stocks | Solid baseline; roughly equal returns to ATR Kelly on 1-min data |
| `atr_kelly` | 1-min | Stocks | Kelly sizing scales well on 1-min; approximately equal returns to base ATR breakout |
| `atr_adaptive_kelly` | 5-min+ | High-volatility (crypto) | Best performer on longer timeframes and volatile markets like BTC, ETH, SOL |
| `atr_optimized` | — | — | Underperforms in practice; filters over-suppress signals |
| `atr_adaptive` | — | — | Adaptive bands help, but without Kelly sizing the edge is limited |
| `ma` | — | — | Too slow for intraday; frequent whipsaws on 1-min/5-min data |
| `template` | — | — | Starter template; not tuned for live performance |
| `crypto_trend` | — | — | Basic EMA crossover; outperformed by ATR variants on the same crypto data |
| `demo` | — | — | For testing only |

**Key takeaway:** For 1-minute stock data, `atr_breakout` and `atr_kelly` deliver approximately equal returns. For longer timeframes (5-min+) and high-volatility markets like crypto, `atr_adaptive_kelly` is the best-performing strategy — the adaptive multiplier widens bands in volatile regimes to avoid false breakouts, while Kelly sizing scales positions appropriately.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` file (only 2 fields required!)
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

### 3. Run live trading
```bash
python run_live.py --symbol AAPL --strategy ma --live
```

Press `Ctrl+C` to stop and see performance summary (P&L, Sharpe ratio, win rate, etc.)

---

## Available Timeframes

| Timeframe | Description | Best For |
|-----------|-------------|----------|
| `1Min` | 1-minute bars | High-frequency, intraday trading |
| `5Min` | 5-minute bars | Short-term intraday |
| `15Min` | 15-minute bars | Intraday swing |
| `30Min` | 30-minute bars | Intraday swing |
| `1Hour` | 1-hour bars | Intraday/multi-day |
| `1Day` | Daily bars | Swing/position trading |

**Example usage:**
```bash
# Fast trading with 1-minute bars, checking every 5 seconds
python run_live.py --symbol AAPL --strategy ma --timeframe 1Min --sleep 5 --live

# Daily strategy
python run_live.py --symbol AAPL --strategy ma --timeframe 1Day --live
```

**Note:** Alpaca's minimum bar resolution is 1 minute. The `--sleep` parameter controls how often we check for new signals (can be as low as 1 second).

---

## CLI Reference

### List available strategies
```bash
python run_live.py --list-strategies
```

### Live trading
```bash
python run_live.py --symbol AAPL --strategy ma --live
```

### Dry run (no real orders)
```bash
python run_live.py --symbol AAPL --strategy ma --dry-run --iterations 10
```

### Crypto trading
```bash
python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live
```

### Fast demo mode (check every 5 seconds)
```bash
python run_live.py --symbol AAPL --strategy template --timeframe 1Min --sleep 5 --live
```

### All options
| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | AAPL | Stock ticker or crypto symbol |
| `--asset-class` | stock | `stock` or `crypto` |
| `--strategy` | ma | Strategy name |
| `--timeframe` | 1Min | Bar timeframe |
| `--sleep` | 60 | Seconds between iterations |
| `--position-size` | 10.0 | Shares (stocks) or USD (crypto) per trade |
| `--live` | false | Run continuously until Ctrl+C |
| `--dry-run` | false | Print decisions without placing orders |
| `--save-data` | false | Save market data to CSV |

---

## Backtest Mode

```bash
python run_backtest.py --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy ma --plot
```

---

## Monte Carlo Backtesting

Monte Carlo backtesting runs many simulations to see how strategy performance varies under randomness:

- **Execution MC** – Same CSV, N runs. The matching engine’s fill/partial/cancel randomness changes each run → distribution of PnL, Sharpe, max drawdown, win rate.
- **Price-path MC** – N synthetic GBM price paths. Each path is one backtest; execution randomness also varies per run.

**Module:** `monte_carlo.py` – Core library with `run_execution_mc()`, `run_price_path_mc()`, `run_custom_mc()`, and `print_mc_summary()`.

**CLI runner:** `run_monte_carlo copy.py`

### Execution mode (same CSV, multiple runs)

```bash
python "run_monte_carlo copy.py" --mode execution --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy ma --runs 100
```

### Price-path mode (synthetic GBM paths)

```bash
python "run_monte_carlo copy.py" --mode price_path --strategy ma --paths 50 --n-bars 500 --seed-base 42
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | execution | `execution` or `price_path` |
| `--csv` | — | CSV path (required for execution mode) |
| `--strategy` | ma | Strategy name |
| `--runs` | 100 | Execution MC: number of runs |
| `--paths` | 50 | Price-path MC: number of synthetic paths |
| `--n-bars` | 500 | Price-path: bars per path |
| `--seed-base` | — | Price-path: RNG seed (for reproducibility) |
| `--capital` | 50000 | Initial capital |

---

## Build Your Own Strategy

Edit `strategies/strategy_base.py` and add your class:

```python
class MyStrategy(Strategy):
    def __init__(self, lookback=20, position_size=10.0):
        self.lookback = lookback
        self.position_size = position_size

    def add_indicators(self, df):
        df['sma'] = df['Close'].rolling(self.lookback).mean()
        return df

    def generate_signals(self, df):
        df['signal'] = 0
        df.loc[df['Close'] > df['sma'], 'signal'] = 1
        df.loc[df['Close'] < df['sma'], 'signal'] = -1
        df['position'] = df['signal']
        df['target_qty'] = self.position_size
        return df
```

Then run:
```bash
python run_live.py --symbol AAPL --strategy mystrategy --live
```

---

## Logs and Output

All trades are logged to:
- `logs/trades.csv` - Trade records (timestamp, symbol, side, qty, price, P&L)
- `logs/system.log` - Full system log with debug info

When you stop with `Ctrl+C`, you'll see a performance summary:
```
SESSION SUMMARY
  Trades: 15 | Wins: 9 (60.0%)
  Net P&L: $+127.50
  Sharpe Ratio: 1.23
  Volatility: 0.82%
```

---

## Project Structure
```
trading-system/
  core/           # Trading engine, logger
  pipeline/       # Alpaca data helpers
  strategies/     # Strategy implementations
  logs/           # Trade logs (auto-created)
  data/           # Market data CSVs
  run_live.py     # Live trading CLI
  run_backtest.py # Backtesting CLI
  monte_carlo.py  # Monte Carlo backtesting library
  run_monte_carlo copy.py  # Monte Carlo CLI runner
```

## Fetch data with Alpaca
Use the notebooks in `notebooks/`:
- `notebooks/fetch_data_stock.ipynb`
- `notebooks/fetch_data_crypto.ipynb`

They download bars from Alpaca and save raw/clean CSVs to `data/`.

## Build your own strategy
Open `strategies/strategy_base.py` and edit `TemplateStrategy` (recommended), or add your own class.
The backtester expects:
- `signal`: 1 for buy, -1 for sell, 0 for no action.
- `target_qty`: the quantity to trade when a signal triggers.
- Optionally, `limit_price` if you want a limit price different from `Close`.

The Alpaca runner uses these same fields to submit paper orders.
To run your custom class from the CLI, give it a no-arg constructor and call (case-insensitive):
```
python run_backtest.py --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy MyStrategy
```
or
```
python run_live.py --symbol AAPL --asset-class stock --strategy MyStrategy --timeframe 1Min --live
```

Example signal logic inside `generate_signals`:
```python
df["signal"] = 0
buy = df["momentum"] > 0.01
sell = df["momentum"] < -0.01
df.loc[buy, "signal"] = 1
df.loc[sell, "signal"] = -1
df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
df["target_qty"] = df["position"].abs() * self.position_size
```

## Trade output format (live)
Each fill prints a line like:
```
2024-01-01 09:31:00 | BUY 10 AAPL @ 101.23 | order_id=1234 | net_pnl=+12.50
```

## Project structure
```
core/
pipeline/
strategies/
data/
notebooks/
.env
monte_carlo.py
run_backtest.py
run_live.py
run_monte_carlo copy.py
test_system.py
```
