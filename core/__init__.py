"""Core backtesting components."""

from .logger import get_logger, get_trade_logger, TradeLogger

# Lazy import: AlpacaTrader requires alpaca_trade_api which may not be installed
# in environments that only need the backtester / offline tools.
try:
    from .alpaca_trader import AlpacaTrader
except ImportError:
    AlpacaTrader = None  # type: ignore[assignment,misc]

__all__ = ["AlpacaTrader", "get_logger", "get_trade_logger", "TradeLogger"]
