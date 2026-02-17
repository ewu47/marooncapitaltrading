"""
Offline backtest runner for a CSV file.

Usage:
    python run_backtest.py --csv data/AAPL_1Min_alpaca_clean.csv --strategy atr_breakout
    python run_backtest.py --csv data/SPY_1m_clean.csv --strategy atr_kelly --atr-multiplier 1.5
    python run_backtest.py --csv data/AAPL_1Min_alpaca_clean.csv --strategy atr_adaptive_kelly
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtester import Backtester, PerformanceAnalyzer, plot_equity
from core.gateway import MarketDataGateway
from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.order_manager import OrderLoggingGateway, OrderManager
from strategies import (
    ATRBreakoutAdaptiveKellyStrategy,
    ATRBreakoutAdaptiveStrategy,
    ATRBreakoutKellyStrategy,
    ATRBreakoutOptimizedStrategy,
    ATRBreakoutStrategy,
    CryptoTrendStrategy,
    DemoStrategy,
    MovingAverageStrategy,
    TemplateStrategy,
    get_strategy_class,
)


DATA_DIR = Path("data")


def create_sample_data(path: Path, periods: int = 200) -> None:
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range(start="2024-01-01 09:30", periods=periods, freq="T"),
            "Open": np.random.uniform(100, 105, periods),
            "High": np.random.uniform(105, 110, periods),
            "Low": np.random.uniform(95, 100, periods),
            "Close": np.random.uniform(100, 110, periods),
            "Volume": np.random.randint(1_000, 5_000, periods),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline CSV backtest.")
    parser.add_argument("--csv", type=str, default="", help="Path to a CSV with OHLCV data.")
    parser.add_argument("--strategy", default="atr_breakout", help="Strategy name.")
    # OG strategy args
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (MA / crypto).")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (MA / crypto).")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-trade position size.")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback (template).")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold (template).")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold (template).")
    # ATR common args
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period.")
    parser.add_argument("--breakout-lookback", type=int, default=20, help="Breakout lookback window.")
    parser.add_argument("--atr-multiplier", type=float, default=1.5, help="ATR breakout multiplier.")
    # ATR Adaptive args
    parser.add_argument("--base-multiplier", type=float, default=1.2, help="Adaptive: base multiplier.")
    parser.add_argument("--z-scale", type=float, default=0.5, help="Adaptive: z-score scale factor.")
    parser.add_argument("--z-lookback", type=int, default=100, help="Adaptive: ATR z-score lookback.")
    parser.add_argument("--max-z", type=float, default=2.0, help="Adaptive: max z-score clamp.")
    # ATR Kelly args
    parser.add_argument("--equity", type=float, default=50_000, help="Kelly: equity for sizing.")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Kelly: fraction of equity risked.")
    parser.add_argument("--max-position-size", type=float, default=500.0, help="Kelly: max shares per trade.")
    parser.add_argument("--min-position-size", type=float, default=1.0, help="Kelly: min shares per trade.")
    parser.add_argument("--max-notional-frac", type=float, default=0.20, help="Kelly: max notional as fraction of equity.")
    parser.add_argument("--min-notional-frac", type=float, default=0.01, help="Kelly: min notional as fraction of equity.")
    # General
    parser.add_argument("--capital", type=float, default=50_000, help="Initial capital.")
    parser.add_argument("--fractional", action="store_true", help="Allow fractional quantities (use for crypto).")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve at the end.")
    return parser.parse_args()


def build_strategy(args: argparse.Namespace):
    """Construct the strategy instance from CLI args."""
    cls = get_strategy_class(args.strategy)

    if cls is MovingAverageStrategy:
        return MovingAverageStrategy(
            short_window=args.short_window, long_window=args.long_window,
            position_size=args.position_size,
        )
    if cls is TemplateStrategy:
        return TemplateStrategy(
            lookback=args.momentum_lookback, position_size=args.position_size,
            buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold,
        )
    if cls is CryptoTrendStrategy:
        return CryptoTrendStrategy(
            short_window=args.short_window, long_window=args.long_window,
            position_size=args.position_size,
        )
    if cls is DemoStrategy:
        return DemoStrategy(position_size=args.position_size)
    if cls is ATRBreakoutStrategy:
        return ATRBreakoutStrategy(
            atr_period=args.atr_period, breakout_lookback=args.breakout_lookback,
            atr_multiplier=args.atr_multiplier, position_size=args.position_size,
        )
    if cls is ATRBreakoutOptimizedStrategy:
        return ATRBreakoutOptimizedStrategy(
            atr_period=args.atr_period, breakout_lookback=args.breakout_lookback,
            atr_multiplier=args.atr_multiplier, position_size=args.position_size,
        )
    if cls is ATRBreakoutKellyStrategy:
        return ATRBreakoutKellyStrategy(
            atr_period=args.atr_period, breakout_lookback=args.breakout_lookback,
            atr_multiplier=args.atr_multiplier, equity=args.equity,
            risk_per_trade=args.risk_per_trade, max_notional_frac=args.max_notional_frac,
            min_notional_frac=args.min_notional_frac,
        )
    if cls is ATRBreakoutAdaptiveStrategy:
        return ATRBreakoutAdaptiveStrategy(
            atr_period=args.atr_period, breakout_lookback=args.breakout_lookback,
            base_multiplier=args.base_multiplier, z_scale=args.z_scale,
            z_lookback=args.z_lookback, max_z=args.max_z,
            position_size=args.position_size,
        )
    if cls is ATRBreakoutAdaptiveKellyStrategy:
        return ATRBreakoutAdaptiveKellyStrategy(
            atr_period=args.atr_period, breakout_lookback=args.breakout_lookback,
            base_multiplier=args.base_multiplier, z_scale=args.z_scale,
            z_lookback=args.z_lookback, max_z=args.max_z,
            equity=args.equity, risk_per_trade=args.risk_per_trade,
            max_notional_frac=args.max_notional_frac, min_notional_frac=args.min_notional_frac,
        )
    # Fallback: try no-arg constructor
    try:
        return cls()
    except TypeError as exc:
        raise SystemExit(
            f"{cls.__name__} requires explicit parameters — pick a known strategy name."
        ) from exc


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv) if args.csv else DATA_DIR / "sample_system_test_data.csv"
    if not csv_path.exists():
        if args.csv:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        create_sample_data(csv_path)
        print(f"Sample data generated at {csv_path}.")

    strategy = build_strategy(args)

    gateway = MarketDataGateway(csv_path)
    order_book = OrderBook()
    order_manager = OrderManager(capital=args.capital, max_long_position=1_000, max_short_position=1_000)
    matching_engine = MatchingEngine()
    logger = OrderLoggingGateway()

    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
        default_position_size=int(max(1, args.position_size)),
        fractional_qty=args.fractional,
    )

    equity_df = backtester.run()
    analyzer = PerformanceAnalyzer(equity_df["equity"].dropna().tolist(), backtester.trades)

    filled = [t.qty for t in backtester.trades if t.qty > 0]
    avg_pos = sum(filled) / len(filled) if filled else 0.0

    print("\n=== Backtest Summary ===")
    print(f"Strategy:             {args.strategy}")
    print(f"Equity data points:   {len(equity_df)}")
    print(f"Trades executed:      {len(filled)}")
    print(f"Avg position size:    {avg_pos:.2f}")
    print(f"Final portfolio value: {equity_df['equity'].dropna().iloc[-1]:.2f}")
    print(f"PnL:                  {analyzer.pnl():.2f}")
    print(f"Sharpe:               {analyzer.sharpe():.2f}")
    print(f"Max Drawdown:         {analyzer.max_drawdown():.4f}")
    print(f"Win Rate:             {analyzer.win_rate():.2%}")

    if args.plot:
        plot_equity(equity_df)


if __name__ == "__main__":
    main()
