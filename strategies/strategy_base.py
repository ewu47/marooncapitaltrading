"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
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
"""

import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)

    Class attributes:
        - target_qty_is_notional: If True, target_qty is in USD notional.
          The execution layer converts to shares/units. If False (default),
          target_qty is in asset units (shares for stocks, USD for crypto).
    """

    target_qty_is_notional: bool = False

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df

class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1   # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
##


class ATRBreakoutStrategy(Strategy):
    """
    ATR-adjusted breakout strategy using prior N-bar highs/lows.

    Performance: Good on 1-minute stock data (e.g. AAPL, TSLA). Returns are
    approximately equal to ATR Kelly on 1-min timeframes. Simple and reliable
    baseline for breakout trading.
    """

    def __init__(
        self,
        atr_period: int = 14,
        breakout_lookback: int = 20,
        atr_multiplier: float = 1.5,
        position_size: float = 10.0,
    ):
        if atr_period < 2:
            raise ValueError("atr_period must be at least 2.")
        if breakout_lookback < 2:
            raise ValueError("breakout_lookback must be at least 2.")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.atr_multiplier = atr_multiplier
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["Close"].shift(1)
        tr_components = pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        df["ATR"] = tr_components.max(axis=1).rolling(self.atr_period, min_periods=self.atr_period).mean()

        df["prior_high"] = df["High"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).max()
        df["prior_low"] = df["Low"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).min()
        df["ATR_upper_breakout"] = df["prior_high"] + self.atr_multiplier * df["ATR"]
        df["ATR_lower_breakout"] = df["prior_low"] - self.atr_multiplier * df["ATR"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["Close"] > df["ATR_upper_breakout"], "signal"] = 1
        df.loc[df["Close"] < df["ATR_lower_breakout"], "signal"] = -1
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class ATRBreakoutOptimizedStrategy(Strategy):
    """
    Optimized ATR breakout strategy with:
      - SMA trend filter (only long in uptrend, short in downtrend)
      - Volume confirmation (require above-average volume on breakout)
      - ATR trailing stop for exits (lock in gains)
      - Cooldown period to avoid whipsaw re-entries
      - Inverse-ATR dynamic position sizing

    Performance: Underperforms in practice. The combination of filters
    (trend + volume + cooldown) over-suppresses signals, leaving too few
    trades to generate meaningful returns. Not recommended for live use.
    """

    def __init__(
        self,
        atr_period: int = 14,
        breakout_lookback: int = 20,
        atr_multiplier: float = 1.0,
        position_size: float = 10.0,
        trend_sma_period: int = 50,
        use_trend_filter: bool = True,
        volume_multiplier: float = 1.5,
        use_volume_filter: bool = True,
        trailing_atr_multiplier: float = 2.0,
        use_trailing_stop: bool = True,
        cooldown_bars: int = 10,
        use_cooldown: bool = True,
        use_dynamic_sizing: bool = True,
        max_size_multiplier: float = 3.0,
    ):
        if atr_period < 2:
            raise ValueError("atr_period must be at least 2.")
        if breakout_lookback < 2:
            raise ValueError("breakout_lookback must be at least 2.")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if trend_sma_period < 2:
            raise ValueError("trend_sma_period must be at least 2.")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive.")
        if trailing_atr_multiplier <= 0:
            raise ValueError("trailing_atr_multiplier must be positive.")
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative.")

        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.atr_multiplier = atr_multiplier
        self.position_size = position_size
        self.trend_sma_period = trend_sma_period
        self.use_trend_filter = use_trend_filter
        self.volume_multiplier = volume_multiplier
        self.use_volume_filter = use_volume_filter
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.cooldown_bars = cooldown_bars
        self.use_cooldown = use_cooldown
        self.use_dynamic_sizing = use_dynamic_sizing
        self.max_size_multiplier = max_size_multiplier

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # ATR calculation
        prev_close = df["Close"].shift(1)
        tr_components = pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        df["ATR"] = tr_components.max(axis=1).rolling(
            self.atr_period, min_periods=self.atr_period
        ).mean()

        # Breakout levels
        df["prior_high"] = df["High"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).max()
        df["prior_low"] = df["Low"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).min()
        df["ATR_upper_breakout"] = df["prior_high"] + self.atr_multiplier * df["ATR"]
        df["ATR_lower_breakout"] = df["prior_low"] - self.atr_multiplier * df["ATR"]

        # Trend filter
        if self.use_trend_filter:
            df["trend_sma"] = df["Close"].rolling(
                self.trend_sma_period, min_periods=1
            ).mean()

        # Volume filter
        if self.use_volume_filter:
            df["vol_avg"] = df["Volume"].rolling(
                self.breakout_lookback, min_periods=1
            ).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        # Raw breakout conditions
        raw_buy = df["Close"] > df["ATR_upper_breakout"]
        raw_sell = df["Close"] < df["ATR_lower_breakout"]

        # Apply trend filter
        if self.use_trend_filter and "trend_sma" in df.columns:
            raw_buy = raw_buy & (df["Close"] > df["trend_sma"])
            raw_sell = raw_sell & (df["Close"] < df["trend_sma"])

        # Apply volume filter
        if self.use_volume_filter and "vol_avg" in df.columns:
            high_volume = df["Volume"] > self.volume_multiplier * df["vol_avg"]
            raw_buy = raw_buy & high_volume
            raw_sell = raw_sell & high_volume

        df.loc[raw_buy, "signal"] = 1
        df.loc[raw_sell, "signal"] = -1

        # Apply cooldown: suppress signals within N bars of the last signal
        if self.use_cooldown and self.cooldown_bars > 0:
            signal_col = df["signal"].values.copy()
            last_signal_bar = -self.cooldown_bars - 1  # allow first signal
            for i in range(len(signal_col)):
                if signal_col[i] != 0:
                    if (i - last_signal_bar) < self.cooldown_bars:
                        signal_col[i] = 0  # suppress
                    else:
                        last_signal_bar = i
            df["signal"] = signal_col

        # Build position with forward-fill
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

        # Apply trailing stop to exit positions
        if self.use_trailing_stop and "ATR" in df.columns:
            position_vals = df["position"].values.copy()
            close_vals = df["Close"].values
            high_vals = df["High"].values
            low_vals = df["Low"].values
            atr_vals = df["ATR"].values
            trail_mult = self.trailing_atr_multiplier

            tracking_high = np.nan
            tracking_low = np.nan

            for i in range(len(position_vals)):
                atr_i = atr_vals[i]
                if np.isnan(atr_i):
                    continue

                if position_vals[i] == 1:
                    # Long position: track highest high
                    if np.isnan(tracking_high):
                        tracking_high = high_vals[i]
                    else:
                        tracking_high = max(tracking_high, high_vals[i])
                    trail_stop = tracking_high - trail_mult * atr_i
                    if close_vals[i] < trail_stop:
                        position_vals[i] = 0
                        tracking_high = np.nan
                        tracking_low = np.nan
                elif position_vals[i] == -1:
                    # Short position: track lowest low
                    if np.isnan(tracking_low):
                        tracking_low = low_vals[i]
                    else:
                        tracking_low = min(tracking_low, low_vals[i])
                    trail_stop = tracking_low + trail_mult * atr_i
                    if close_vals[i] > trail_stop:
                        position_vals[i] = 0
                        tracking_high = np.nan
                        tracking_low = np.nan
                else:
                    tracking_high = np.nan
                    tracking_low = np.nan

            df["position"] = position_vals

        # Dynamic position sizing (inverse ATR)
        if self.use_dynamic_sizing and "ATR" in df.columns:
            inv_atr = 1.0 / df["ATR"].clip(lower=0.01)
            inv_atr_mean = inv_atr.mean()
            if inv_atr_mean > 0:
                size_scale = inv_atr / inv_atr_mean
            else:
                size_scale = 1.0
            df["target_qty"] = (
                df["position"].abs()
                * self.position_size
                * size_scale
            ).clip(lower=0, upper=self.max_size_multiplier * self.position_size)
        else:
            df["target_qty"] = df["position"].abs() * self.position_size

        return df


class ATRBreakoutKellyStrategy(Strategy):
    """
    ATR breakout strategy with Kelly-fraction volatility-adjusted position sizing.

    The Kelly fraction determines what proportion of equity to allocate per
    trade based on the stop distance relative to price:

        stop_frac   = (atr_multiplier × ATR) / price
        kelly_frac  = clip(risk_per_trade / stop_frac, min_frac, max_frac)
        target_qty  = kelly_frac × equity          (USD notional)

    The execution layer converts USD notional to shares (stocks) or base
    units (crypto) automatically.

    Benefits:
      - Scales up in calm, trending markets (low ATR → larger fraction)
      - Scales down in violent, choppy markets (high ATR → smaller fraction)
      - Works correctly for both stocks and crypto without unit mismatches
      - Clamping operates on the fraction, not on raw quantities

    Performance: Good on 1-minute stock data. Returns are approximately equal
    to the base ATR breakout strategy on 1-min timeframes, but with better
    risk-adjusted sizing. Kelly sizing shines when volatility varies across
    the session.
    """

    target_qty_is_notional = True

    def __init__(
        self,
        atr_period: int = 14,
        breakout_lookback: int = 20,
        atr_multiplier: float = 1.2,
        equity: float = 100_000.0,
        risk_per_trade: float = 0.01,
        max_notional_frac: float = 0.20,
        min_notional_frac: float = 0.01,
        # Kept for CLI backward compatibility; not used in sizing
        max_position_size: float = 500.0,
        min_position_size: float = 1.0,
    ):
        if atr_period < 2:
            raise ValueError("atr_period must be at least 2.")
        if breakout_lookback < 2:
            raise ValueError("breakout_lookback must be at least 2.")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive.")
        if equity <= 0:
            raise ValueError("equity must be positive.")
        if not 0 < risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1).")
        if not 0 < max_notional_frac <= 1:
            raise ValueError("max_notional_frac must be in (0, 1].")
        if not 0 < min_notional_frac <= 1:
            raise ValueError("min_notional_frac must be in (0, 1].")

        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.atr_multiplier = atr_multiplier
        self.equity = equity
        self.risk_per_trade = risk_per_trade
        self.max_notional_frac = max_notional_frac
        self.min_notional_frac = min_notional_frac

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["Close"].shift(1)
        tr_components = pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        df["ATR"] = tr_components.max(axis=1).rolling(
            self.atr_period, min_periods=self.atr_period
        ).mean()

        df["prior_high"] = df["High"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).max()
        df["prior_low"] = df["Low"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).min()
        df["ATR_upper_breakout"] = df["prior_high"] + self.atr_multiplier * df["ATR"]
        df["ATR_lower_breakout"] = df["prior_low"] - self.atr_multiplier * df["ATR"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["Close"] > df["ATR_upper_breakout"], "signal"] = 1
        df.loc[df["Close"] < df["ATR_lower_breakout"], "signal"] = -1
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

        # --- Kelly fraction sizing ---
        price = df["Close"].clip(lower=1e-6)
        stop_distance = (self.atr_multiplier * df["ATR"]).clip(lower=1e-6)

        # Stop as a fraction of price
        stop_frac = stop_distance / price

        # Kelly fraction: proportion of equity to allocate
        kelly_frac = (self.risk_per_trade / stop_frac).clip(
            lower=self.min_notional_frac, upper=self.max_notional_frac
        )

        # Convert fraction to USD notional
        target_notional = kelly_frac * self.equity

        df["target_qty"] = df["position"].abs() * target_notional
        return df


class ATRBreakoutAdaptiveStrategy(Strategy):
    """
    ATR breakout strategy with an adaptive (dynamic) ATR multiplier.

    Instead of a fixed multiplier, the breakout thresholds widen or tighten
    based on where current ATR sits relative to its own recent distribution:

        atr_z     = (ATR - ATR_rolling_mean) / ATR_rolling_std
        dyn_mult  = base_mult + scale * clip(atr_z, 0, max_z)

    Benefits:
      - Low-volatility regimes → tighter breakout bands → more trades captured
      - High-volatility regimes → wider bands → fewer false breakouts
      - Increases overall trade efficiency across market regimes

    Performance: The adaptive bands help reduce false breakouts, but without
    Kelly sizing the edge is limited. Outperformed by ATRBreakoutAdaptiveKellyStrategy
    which combines adaptive bands with Kelly position sizing.
    """

    def __init__(
        self,
        atr_period: int = 14,
        breakout_lookback: int = 20,
        base_multiplier: float = 1.2,
        z_scale: float = 0.5,
        z_lookback: int = 100,
        max_z: float = 2.0,
        position_size: float = 10.0,
    ):
        if atr_period < 2:
            raise ValueError("atr_period must be at least 2.")
        if breakout_lookback < 2:
            raise ValueError("breakout_lookback must be at least 2.")
        if base_multiplier <= 0:
            raise ValueError("base_multiplier must be positive.")
        if z_scale < 0:
            raise ValueError("z_scale must be non-negative.")
        if z_lookback < 2:
            raise ValueError("z_lookback must be at least 2.")
        if max_z <= 0:
            raise ValueError("max_z must be positive.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")

        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.base_multiplier = base_multiplier
        self.z_scale = z_scale
        self.z_lookback = z_lookback
        self.max_z = max_z
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["Close"].shift(1)
        tr_components = pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        df["ATR"] = tr_components.max(axis=1).rolling(
            self.atr_period, min_periods=self.atr_period
        ).mean()

        # Adaptive multiplier: z-score of ATR relative to its own rolling window
        atr_mean = df["ATR"].rolling(self.z_lookback, min_periods=self.z_lookback).mean()
        atr_std = df["ATR"].rolling(self.z_lookback, min_periods=self.z_lookback).std()
        df["atr_z"] = ((df["ATR"] - atr_mean) / atr_std.clip(lower=1e-8)).fillna(0.0)
        df["dynamic_mult"] = self.base_multiplier + self.z_scale * df["atr_z"].clip(0, self.max_z)

        df["prior_high"] = df["High"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).max()
        df["prior_low"] = df["Low"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).min()

        # Use dynamic multiplier instead of fixed
        df["ATR_upper_breakout"] = df["prior_high"] + df["dynamic_mult"] * df["ATR"]
        df["ATR_lower_breakout"] = df["prior_low"] - df["dynamic_mult"] * df["ATR"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["Close"] > df["ATR_upper_breakout"], "signal"] = 1
        df.loc[df["Close"] < df["ATR_lower_breakout"], "signal"] = -1
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class ATRBreakoutAdaptiveKellyStrategy(Strategy):
    """
    ATR breakout strategy combining:
      1. Adaptive (dynamic) ATR multiplier — widens bands in high-vol regimes,
         tightens them in low-vol regimes (from ATRBreakoutAdaptiveStrategy).
      2. Kelly-fraction volatility-adjusted position sizing
         (from ATRBreakoutKellyStrategy).

    Breakout bands:
        atr_z        = (ATR − ATR_mean) / ATR_std
        dyn_mult     = base_mult + z_scale × clip(atr_z, 0, max_z)
        upper_band   = prior_high + dyn_mult × ATR
        lower_band   = prior_low  − dyn_mult × ATR

    Position sizing (Kelly fraction):
        stop_frac   = (dyn_mult × ATR) / price
        kelly_frac  = clip(risk_per_trade / stop_frac, min_frac, max_frac)
        target_qty  = kelly_frac × equity          (USD notional)

    The execution layer converts USD notional to shares or crypto base units.

    Performance: Best overall strategy. Excels on longer timeframes (5-min+)
    and high-volatility markets like crypto (BTC, ETH, SOL). The adaptive
    multiplier avoids false breakouts in choppy regimes while Kelly sizing
    scales positions appropriately to capture trending moves. Recommended as
    the primary strategy for crypto and longer timeframe trading.
    """

    target_qty_is_notional = True

    def __init__(
        self,
        atr_period: int = 14,
        breakout_lookback: int = 20,
        # Adaptive multiplier params
        base_multiplier: float = 1.2,
        z_scale: float = 0.5,
        z_lookback: int = 100,
        max_z: float = 2.0,
        # Kelly sizing params
        equity: float = 100_000.0,
        risk_per_trade: float = 0.01,
        max_notional_frac: float = 0.20,
        min_notional_frac: float = 0.01,
        # Kept for CLI backward compatibility; not used in sizing
        max_position_size: float = 500.0,
        min_position_size: float = 1.0,
    ):
        if atr_period < 2:
            raise ValueError("atr_period must be at least 2.")
        if breakout_lookback < 2:
            raise ValueError("breakout_lookback must be at least 2.")
        if base_multiplier <= 0:
            raise ValueError("base_multiplier must be positive.")
        if z_scale < 0:
            raise ValueError("z_scale must be non-negative.")
        if z_lookback < 2:
            raise ValueError("z_lookback must be at least 2.")
        if max_z <= 0:
            raise ValueError("max_z must be positive.")
        if equity <= 0:
            raise ValueError("equity must be positive.")
        if not 0 < risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1).")
        if not 0 < max_notional_frac <= 1:
            raise ValueError("max_notional_frac must be in (0, 1].")
        if not 0 < min_notional_frac <= 1:
            raise ValueError("min_notional_frac must be in (0, 1].")

        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.base_multiplier = base_multiplier
        self.z_scale = z_scale
        self.z_lookback = z_lookback
        self.max_z = max_z
        self.equity = equity
        self.risk_per_trade = risk_per_trade
        self.max_notional_frac = max_notional_frac
        self.min_notional_frac = min_notional_frac

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["Close"].shift(1)
        tr_components = pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        df["ATR"] = tr_components.max(axis=1).rolling(
            self.atr_period, min_periods=self.atr_period
        ).mean()

        # Adaptive multiplier: z-score of ATR relative to its own rolling window
        atr_mean = df["ATR"].rolling(self.z_lookback, min_periods=self.z_lookback).mean()
        atr_std = df["ATR"].rolling(self.z_lookback, min_periods=self.z_lookback).std()
        df["atr_z"] = ((df["ATR"] - atr_mean) / atr_std.clip(lower=1e-8)).fillna(0.0)
        df["dynamic_mult"] = self.base_multiplier + self.z_scale * df["atr_z"].clip(0, self.max_z)

        df["prior_high"] = df["High"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).max()
        df["prior_low"] = df["Low"].shift(1).rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).min()

        # Breakout bands use the dynamic multiplier
        df["ATR_upper_breakout"] = df["prior_high"] + df["dynamic_mult"] * df["ATR"]
        df["ATR_lower_breakout"] = df["prior_low"] - df["dynamic_mult"] * df["ATR"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["Close"] > df["ATR_upper_breakout"], "signal"] = 1
        df.loc[df["Close"] < df["ATR_lower_breakout"], "signal"] = -1
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

        # --- Kelly fraction sizing with adaptive stop distance ---
        price = df["Close"].clip(lower=1e-6)
        stop_distance = (df["dynamic_mult"] * df["ATR"]).clip(lower=1e-6)

        # Stop as a fraction of price
        stop_frac = stop_distance / price

        # Kelly fraction: proportion of equity to allocate
        kelly_frac = (self.risk_per_trade / stop_frac).clip(
            lower=self.min_notional_frac, upper=self.max_notional_frac
        )

        # Convert fraction to USD notional
        target_notional = kelly_frac * self.equity

        df["target_qty"] = df["position"].abs() * target_notional
        return df


