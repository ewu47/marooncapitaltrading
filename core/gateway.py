"""
Market data gateway used by the backtester to stream cleaned historical data
row-by-row, simulating a live feed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional
import time

import numpy as np
import pandas as pd


class MarketDataGateway:
    """
    Streams historical market data to consumers. Supports iterator interface and
    an explicit generator via the `stream` method.
    """

    def __init__(self, csv_path: str | Path, symbol: Optional[str] = None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.symbol = symbol or self._infer_symbol()
        self.data = pd.read_csv(self.csv_path, parse_dates=["Datetime"])
        if "Datetime" not in self.data.columns:
            raise ValueError("CSV must contain a Datetime column.")

        self.data.sort_values("Datetime", inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        self.length = len(self.data)
        self.pointer = 0

    def _infer_symbol(self) -> str:
        stem = self.csv_path.stem
        token = stem.split("_")[0] if stem else "ASSET"
        return token.upper()

    # Iterator protocol -----------------------------------------------------

    def __iter__(self) -> Iterator[Dict]:
        self.reset()
        return self

    def __next__(self) -> Dict:
        if self.pointer >= self.length:
            raise StopIteration

        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        self.pointer += 1
        return row

    # Helpers ----------------------------------------------------------------

    def reset(self) -> None:
        self.pointer = 0

    def has_next(self) -> bool:
        return self.pointer < self.length

    def get_next(self) -> Optional[Dict]:
        try:
            return next(self)
        except StopIteration:
            return None

    def peek(self) -> Optional[Dict]:
        if not self.has_next():
            return None
        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        return row

    # Generator --------------------------------------------------------------

    def stream(self, delay: Optional[float] = None, reset: bool = False):
        """
        Yields rows sequentially. Optional delay (seconds) mimics websocket feed.
        """
        if reset:
            self.reset()

        while self.has_next():
            row = next(self)
            yield row

            if delay:
                time.sleep(delay)


# Backwards compatible alias for historical imports.
Gateway = MarketDataGateway


class SyntheticDataGateway:
    """
    Streams synthetic OHLCV bars generated from a geometric Brownian motion path.
    """

    def __init__(
        self,
        n_bars: int = 500,
        bar_minutes: int = 1,
        start_price: float = 100.0,
        mu: float = 0.0001,
        sigma: float = 0.01,
        volume: int = 5_000,
        seed: Optional[int] = None,
        symbol: str = "SYNTH",
    ):
        if n_bars < 2:
            raise ValueError("n_bars must be at least 2.")
        if bar_minutes < 1:
            raise ValueError("bar_minutes must be at least 1.")
        if start_price <= 0:
            raise ValueError("start_price must be positive.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")

        self.n_bars = n_bars
        self.bar_minutes = bar_minutes
        self.start_price = start_price
        self.mu = mu
        self.sigma = sigma
        self.volume = volume
        self.seed = seed
        self.symbol = symbol

        self.data = self._build_data()
        self.length = len(self.data)
        self.pointer = 0

    def _build_data(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / (252.0 * 6.5 * 60.0)  # approximately one minute of trading time

        prices = np.empty(self.n_bars, dtype=float)
        prices[0] = self.start_price
        if self.n_bars > 1:
            shocks = rng.standard_normal(self.n_bars - 1)
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * shocks
            for i in range(1, self.n_bars):
                prices[i] = prices[i - 1] * np.exp(drift + diffusion[i - 1])

        open_px = np.empty_like(prices)
        open_px[0] = prices[0]
        open_px[1:] = prices[:-1]
        close_px = prices

        spread = np.maximum(0.0005 * close_px, np.abs(rng.normal(0.03, 0.02, self.n_bars)))
        high_px = np.maximum(open_px, close_px) + spread
        low_px = np.minimum(open_px, close_px) - spread
        volume = rng.integers(max(1, int(self.volume * 0.5)), max(2, int(self.volume * 1.5)), self.n_bars)

        start = pd.Timestamp("2024-01-01 09:30")
        dt_index = pd.date_range(start=start, periods=self.n_bars, freq=f"{self.bar_minutes}min")
        return pd.DataFrame(
            {
                "Datetime": dt_index,
                "Open": open_px,
                "High": high_px,
                "Low": low_px,
                "Close": close_px,
                "Volume": volume,
            }
        )

    def __iter__(self) -> Iterator[Dict]:
        self.reset()
        return self

    def __next__(self) -> Dict:
        if self.pointer >= self.length:
            raise StopIteration
        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        self.pointer += 1
        return row

    def reset(self) -> None:
        self.pointer = 0

    def has_next(self) -> bool:
        return self.pointer < self.length

    def stream(self, delay: Optional[float] = None, reset: bool = False):
        if reset:
            self.reset()
        while self.has_next():
            row = next(self)
            yield row
            if delay:
                time.sleep(delay)
