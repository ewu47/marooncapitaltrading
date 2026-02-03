"""
Monte Carlo backtest runner.

Execution MC: same CSV, N runs — execution randomness (fill/partial/cancel) varies.
Price-path MC: N synthetic GBM paths — both path and execution vary.

Usage:
  python run_monte_carlo.py --mode execution --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy ma --runs 100
  python run_monte_carlo.py --mode price_path --strategy ma --paths 50 --n-bars 500 --seed-base 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.monte_carlo import (
    run_execution_mc,
    run_price_path_mc,
    print_mc_summary,
)
from strategies import (
    MovingAverageStrategy,
    TemplateStrategy,
    CryptoTrendStrategy,
    DemoStrategy,
    get_strategy_class,
    list_strategies,
)


DATA_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo backtests: execution and/or price-path randomness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Strategies: {', '.join(list_strategies())}",
    )
    p.add_argument("--mode", choices=["execution", "price_path"], default="execution",
                   help="execution = same CSV, N runs; price_path = N synthetic GBM paths")
    p.add_argument("--csv", type=str, default="",
                   help="CSV path (required for execution mode)")
    p.add_argument("--strategy", default="ma", help="Strategy name")
    p.add_argument("--runs", type=int, default=100, help="Execution MC: number of runs")
    p.add_argument("--paths", type=int, default=50, help="Price-path MC: number of paths")
    p.add_argument("--n-bars", type=int, default=500, help="Price-path: bars per path")
    p.add_argument("--bar-minutes", type=int, default=1, help="Price-path: minutes per bar")
    p.add_argument("--start-price", type=float, default=100.0, help="Price-path: S0")
    p.add_argument("--mu", type=float, default=0.0001, help="Price-path: drift")
    p.add_argument("--sigma", type=float, default=0.01, help="Price-path: volatility")
    p.add_argument("--volume", type=int, default=5000, help="Price-path: base volume")
    p.add_argument("--seed-base", type=int, default=None, help="Price-path: RNG seed (reproducible)")
    p.add_argument("--capital", type=float, default=50_000, help="Initial capital")
    p.add_argument("--position-size", type=float, default=10.0, help="Per-trade position size")
    p.add_argument("--short-window", type=int, default=20, help="MA / crypto short window")
    p.add_argument("--long-window", type=int, default=60, help="MA / crypto long window")
    p.add_argument("--momentum-lookback", type=int, default=14, help="Template lookback")
    p.add_argument("--buy-threshold", type=float, default=0.01, help="Template buy threshold")
    p.add_argument("--sell-threshold", type=float, default=-0.01, help="Template sell threshold")
    return p.parse_args()


def build_strategy(args: argparse.Namespace):
    cls = get_strategy_class(args.strategy)
    if cls is MovingAverageStrategy:
        return MovingAverageStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    if cls is TemplateStrategy:
        return TemplateStrategy(
            lookback=args.momentum_lookback,
            position_size=args.position_size,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    if cls is CryptoTrendStrategy:
        return CryptoTrendStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    if cls is DemoStrategy:
        return DemoStrategy(position_size=args.position_size)
    return cls()


def main() -> None:
    args = parse_args()
    strategy = build_strategy(args)
    default_position_size = int(max(1, args.position_size))

    if args.mode == "execution":
        csv_path = Path(args.csv) if args.csv else DATA_DIR / "sample_system_test_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}. Use --csv or add sample data.")
        result = run_execution_mc(
            csv_path=csv_path,
            strategy=strategy,
            n_runs=args.runs,
            capital=args.capital,
            default_position_size=default_position_size,
            verbose=False,
        )
        print_mc_summary(result, title=f"Execution MC — {csv_path.name} | {args.strategy} | {args.runs} runs")
    else:
        result = run_price_path_mc(
            strategy=strategy,
            n_paths=args.paths,
            n_bars=args.n_bars,
            bar_minutes=args.bar_minutes,
            start_price=args.start_price,
            mu=args.mu,
            sigma=args.sigma,
            volume=args.volume,
            capital=args.capital,
            default_position_size=default_position_size,
            seed_base=args.seed_base,
            verbose=False,
        )
        print_mc_summary(result, title=f"Price-path MC — {args.strategy} | {args.paths} paths")


if __name__ == "__main__":
    main()
