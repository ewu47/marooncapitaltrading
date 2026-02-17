"""
Alpaca paper-trading runner.

Requires .env file with:
    ALPACA_API_KEY      (required)
    ALPACA_API_SECRET   (required)

Usage:
    # Single iteration
    python run_live.py --symbol AAPL --strategy atr_breakout

    # Continuous live trading
    python run_live.py --symbol AAPL --strategy atr_breakout --live

    # Dry run (no real orders)
    python run_live.py --symbol AAPL --strategy atr_kelly --dry-run

    # Crypto trading
    python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live

Logs are saved to: logs/trades.csv, logs/signals.csv, logs/system.log
"""

from __future__ import annotations

import argparse
import sys
import time

from core.alpaca_trader import AlpacaTrader
from core.logger import get_logger, get_trade_logger
from pipeline.alpaca import clean_market_data, save_bars
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
    list_strategies,
)

logger = get_logger("run_live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-trading loop with Alpaca.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies: {', '.join(list_strategies())}

Examples:
  python run_live.py --symbol AAPL --strategy atr_breakout --live
  python run_live.py --symbol AAPL --strategy atr_kelly --atr-multiplier 1.5 --live
  python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live
  python run_live.py --symbol AAPL --strategy atr_adaptive_kelly --dry-run --iterations 5
        """,
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker or crypto symbol (default: AAPL)")
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock", help="Asset class (default: stock)")
    parser.add_argument("--timeframe", default="1Min", help="Alpaca timeframe: 1Min, 5Min, 15Min, 1H, 1D (default: 1Min)")
    parser.add_argument("--lookback", type=int, default=200, help="Bars to fetch each iteration (default: 200)")
    parser.add_argument("--strategy", default="atr_breakout", help="Strategy name (default: atr_breakout)")
    # OG strategy args
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (MA / crypto)")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (MA / crypto)")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-trade position size")
    parser.add_argument("--max-order-notional", type=float, default=None, help="Max notional per order (crypto only)")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback for template strategy")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold for template strategy")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold for template strategy")
    # ATR common args
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period")
    parser.add_argument("--breakout-lookback", type=int, default=20, help="Breakout lookback window")
    parser.add_argument("--atr-multiplier", type=float, default=1.5, help="ATR breakout multiplier")
    # ATR Adaptive args
    parser.add_argument("--base-multiplier", type=float, default=1.2, help="Adaptive: base multiplier")
    parser.add_argument("--z-scale", type=float, default=0.5, help="Adaptive: z-score scale factor")
    parser.add_argument("--z-lookback", type=int, default=100, help="Adaptive: ATR z-score lookback")
    parser.add_argument("--max-z", type=float, default=2.0, help="Adaptive: max z-score clamp")
    # ATR Kelly args
    parser.add_argument("--equity", type=float, default=50_000, help="Kelly: equity for sizing")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Kelly: fraction of equity risked")
    parser.add_argument("--max-position-size", type=float, default=500.0, help="Kelly: max shares per trade")
    parser.add_argument("--min-position-size", type=float, default=1.0, help="Kelly: min shares per trade")
    parser.add_argument("--max-notional-frac", type=float, default=0.20, help="Kelly: max notional as fraction of equity")
    parser.add_argument("--min-notional-frac", type=float, default=0.01, help="Kelly: min notional as fraction of equity")
    # Loop controls
    parser.add_argument("--iterations", type=int, default=1, help="Number of loops to run (default: 1)")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between loops (default: 60)")
    parser.add_argument("--live", action="store_true", help="Run continuously until Ctrl+C")
    parser.add_argument("--save-data", action="store_true", help="Save raw+clean CSVs to data/")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without placing orders")
    parser.add_argument("--feed", default=None, help="Data feed (iex or sip for stocks)")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies and exit")
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
    try:
        return cls()
    except TypeError as exc:
        raise SystemExit(
            f"{cls.__name__} requires explicit parameters — pick a known strategy name."
        ) from exc


def main() -> None:
    args = parse_args()

    # Handle --list-strategies
    if args.list_strategies:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        sys.exit(0)

    # Build strategy
    strategy = build_strategy(args)

    # Log startup
    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info(f"Starting {mode} trading: {args.symbol} | strategy={args.strategy} | timeframe={args.timeframe}")

    trader = AlpacaTrader(
        symbol=args.symbol,
        asset_class=args.asset_class,
        timeframe=args.timeframe,
        lookback=args.lookback,
        strategy=strategy,
        feed=args.feed,
        dry_run=args.dry_run,
        max_order_notional=args.max_order_notional,
    )

    trade_logger = get_trade_logger()
    start_equity = trader.starting_equity
    iteration_count = 0

    def handle_iteration() -> None:
        nonlocal iteration_count
        iteration_count += 1
        logger.debug(f"Iteration {iteration_count}: fetching data for {args.symbol}")
        df = trader.run_once()
        if args.save_data and df is not None:
            raw_path = save_bars(df, args.symbol, args.timeframe, args.asset_class)
            clean_market_data(raw_path)

    def print_summary() -> None:
        summary = trade_logger.get_session_summary(start_equity)
        logger.info("")
        logger.info("=" * 60)
        logger.info("                    SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Iterations:      {iteration_count}")
        logger.info(f"  Total Trades:    {summary['total_trades']}")
        logger.info(f"  Buys / Sells:    {summary['buys']} / {summary['sells']}")
        logger.info("-" * 60)
        logger.info(f"  Wins / Losses:   {summary['wins']} / {summary['losses']}")
        logger.info(f"  Win Rate:        {summary['win_rate']:.1f}%")
        logger.info(f"  Avg Trade P&L:   ${summary['avg_trade_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Start Equity:    ${summary['start_equity']:,.2f}")
        logger.info(f"  End Equity:      ${summary['end_equity']:,.2f}")
        logger.info(f"  Net P&L:         ${summary['net_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Sharpe Ratio:    {summary['sharpe_ratio']:.2f}")
        logger.info(f"  Volatility:      {summary['volatility']:.2f}%")
        logger.info(f"  Max Drawdown:    {summary['max_drawdown']:.2f}%")
        logger.info("=" * 60)
        logger.info("Logs: logs/trades.csv, logs/system.log")

    if args.live:
        logger.info(f"Running continuously (Ctrl+C to stop). Sleep: {args.sleep}s between iterations.")
        try:
            while True:
                handle_iteration()
                time.sleep(args.sleep)
        except KeyboardInterrupt:
            logger.info("Received stop signal.")
            print_summary()
    else:
        logger.info(f"Running {args.iterations} iteration(s)...")
        for i in range(args.iterations):
            handle_iteration()
            if i < args.iterations - 1:
                time.sleep(args.sleep)
        print_summary()


if __name__ == "__main__":
    main()
