"""
Monte Carlo backtesting: multiple runs with random execution and/or synthetic price paths.

- Execution MC: same CSV, N runs. MatchingEngine fill/partial/cancel randomness
  varies each run → distribution of PnL, Sharpe, etc.
- Price-path MC: N synthetic (e.g. GBM) paths. Each path is one backtest;
  optionally combined with execution randomness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

from core.backtester import Backtester, PerformanceAnalyzer
from core.gateway import MarketDataGateway, SyntheticDataGateway
from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.order_manager import OrderManager
from strategies import Strategy


def _run_single(
    gateway: Any,
    strategy: Strategy,
    capital: float,
    max_long: int,
    max_short: int,
    default_position_size: int,
    verbose: bool,
    fractional_qty: bool = False,
) -> tuple[List[float], List[Any], PerformanceAnalyzer]:
    order_book = OrderBook()
    order_manager = OrderManager(
        capital=capital,
        max_long_position=max_long,
        max_short_position=max_short,
    )
    matching_engine = MatchingEngine()
    bt = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=None,
        default_position_size=default_position_size,
        verbose=verbose,
        fractional_qty=fractional_qty,
    )
    equity_df = bt.run()
    equity_clean = equity_df["equity"].dropna().tolist()
    analyzer = PerformanceAnalyzer(
        equity_clean,
        bt.trades,
    )
    return equity_clean, bt.trades, analyzer


def _collect_metrics(analyzer: PerformanceAnalyzer, trades: List[Any]) -> Dict[str, float]:
    filled = [t.qty for t in trades if t.qty > 0]
    avg_pos = float(np.mean(filled)) if filled else 0.0
    return {
        "pnl": analyzer.pnl(),
        "sharpe": analyzer.sharpe(),
        "max_drawdown": analyzer.max_drawdown(),
        "win_rate": analyzer.win_rate(),
        "trade_count": sum(1 for t in trades if t.qty > 0),
        "avg_position_size": avg_pos,
    }


def run_execution_mc(
    csv_path: str | Path,
    strategy: Strategy,
    n_runs: int = 100,
    capital: float = 50_000.0,
    max_long_position: int = 1_000,
    max_short_position: int = 1_000,
    default_position_size: int = 10,
    verbose: bool = False,
    fractional_qty: bool = False,
) -> Dict[str, Any]:
    """
    Monte Carlo over execution only: same price path (CSV), N runs.
    MatchingEngine randomness (fill/partial/cancel) differs each run.

    Returns dict with per-run metrics lists and summary (mean, std, percentiles).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    keys = ["pnl", "sharpe", "max_drawdown", "win_rate", "trade_count", "avg_position_size"]
    runs: Dict[str, List[float]] = {k: [] for k in keys}
    equity_curves: List[List[float]] = []

    for i in range(n_runs):
        print(f"\r  Execution MC: run {i + 1}/{n_runs}", end="", flush=True)
        gateway = MarketDataGateway(csv_path)
        equity, trades, analyzer = _run_single(
            gateway=gateway,
            strategy=strategy,
            capital=capital,
            max_long=max_long_position,
            max_short=max_short_position,
            default_position_size=default_position_size,
            verbose=verbose,
            fractional_qty=fractional_qty,
        )
        equity_curves.append(equity)
        m = _collect_metrics(analyzer, trades)
        for k in keys:
            runs[k].append(m[k])
    print()

    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array(runs[k])
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "runs": runs,
        "summary": summary,
        "equity_curves": equity_curves,
        "n_runs": n_runs,
        "mode": "execution",
    }


def run_price_path_mc(
    strategy: Strategy,
    n_paths: int = 50,
    n_bars: int = 500,
    bar_minutes: int = 1,
    start_price: float = 100.0,
    mu: float = 0.0001,
    sigma: float = 0.01,
    volume: int = 5_000,
    capital: float = 50_000.0,
    max_long_position: int = 1_000,
    max_short_position: int = 1_000,
    default_position_size: int = 10,
    seed_base: Optional[int] = None,
    verbose: bool = False,
    fractional_qty: bool = False,
) -> Dict[str, Any]:
    """
    Monte Carlo over synthetic price paths (GBM). Each path uses a different
    seed. Execution randomness (MatchingEngine) also varies per run.

    Returns dict with per-path metrics lists and summary statistics.
    """
    keys = ["pnl", "sharpe", "max_drawdown", "win_rate", "trade_count"]
    runs: Dict[str, List[float]] = {k: [] for k in keys}
    rng = np.random.default_rng(seed_base)

    for i in range(n_paths):
        print(f"\r  Price-path MC: path {i + 1}/{n_paths}", end="", flush=True)
        seed = int(rng.integers(0, 2**31)) if seed_base is not None else None
        gateway = SyntheticDataGateway(
            n_bars=n_bars,
            bar_minutes=bar_minutes,
            start_price=start_price,
            mu=mu,
            sigma=sigma,
            volume=volume,
            seed=seed,
            symbol="SYNTH",
        )
        _, trades, analyzer = _run_single(
            gateway=gateway,
            strategy=strategy,
            capital=capital,
            max_long=max_long_position,
            max_short=max_short_position,
            default_position_size=default_position_size,
            verbose=verbose,
            fractional_qty=fractional_qty,
        )
        m = _collect_metrics(analyzer, trades)
        for k in keys:
            runs[k].append(m[k])
    print()

    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array(runs[k])
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "runs": runs,
        "summary": summary,
        "n_paths": n_paths,
        "mode": "price_path",
    }


def run_custom_mc(
    gateway_factory: Callable[[], Any],
    strategy: Strategy,
    n_runs: int,
    capital: float = 50_000.0,
    max_long_position: int = 1_000,
    max_short_position: int = 1_000,
    default_position_size: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Monte Carlo with a custom gateway factory. Each run creates a new gateway
    via `gateway_factory()`, runs one backtest, and collects metrics.

    Use this for bootstrap resampling, custom synthetic models, or multi-CSV
    setups. The factory can use `range(n_runs)` or external state to vary paths.
    """
    keys = ["pnl", "sharpe", "max_drawdown", "win_rate", "trade_count"]
    runs: Dict[str, List[float]] = {k: [] for k in keys}

    for i in range(n_runs):
        print(f"\r  Custom MC: run {i + 1}/{n_runs}", end="", flush=True)
        gateway = gateway_factory()
        _, trades, analyzer = _run_single(
            gateway=gateway,
            strategy=strategy,
            capital=capital,
            max_long=max_long_position,
            max_short=max_short_position,
            default_position_size=default_position_size,
            verbose=verbose,
        )
        m = _collect_metrics(analyzer, trades)
        for k in keys:
            runs[k].append(m[k])
    print()

    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array(runs[k])
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "runs": runs,
        "summary": summary,
        "n_runs": n_runs,
        "mode": "custom",
    }


def print_mc_summary(result: Dict[str, Any], title: Optional[str] = None) -> None:
    """Pretty-print Monte Carlo result summary."""
    mode = result.get("mode", "?")
    n = result["n_paths"] if "n_paths" in result else result.get("n_runs", 0)
    label = "paths" if mode == "price_path" else "runs"

    if title:
        print(f"\n=== {title} ===")
    print(f"Monte Carlo: {n} {label} (mode={mode})")
    print("-" * 50)
    for k, s in result["summary"].items():
        print(f"  {k}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"min={s['min']:.4f}  max={s['max']:.4f}  "
              f"p5={s['p5']:.4f}  p50={s['p50']:.4f}  p95={s['p95']:.4f}")
    print()


def plot_mc_equity(result: Dict[str, Any], title: Optional[str] = None, save_path: Optional[Path] = None) -> None:
    """Plot every equity curve from a Monte Carlo result."""
    curves = result.get("equity_curves", [])
    if not curves:
        print("No equity curves to plot.")
        return

    n = len(curves)
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, curve in enumerate(curves):
        ax.plot(curve, alpha=0.25, linewidth=0.8, color="steelblue")

    # Compute and plot the mean equity curve (use shortest length for alignment)
    min_len = min(len(c) for c in curves)
    aligned = np.array([c[:min_len] for c in curves])
    mean_curve = aligned.mean(axis=0)
    p5_curve = np.percentile(aligned, 5, axis=0)
    p95_curve = np.percentile(aligned, 95, axis=0)

    ax.plot(mean_curve, color="navy", linewidth=2, label="Mean")
    ax.fill_between(range(min_len), p5_curve, p95_curve, alpha=0.15, color="navy", label="5th–95th pctl")

    ax.set_title(title or f"Monte Carlo Equity Curves ({n} runs)")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
