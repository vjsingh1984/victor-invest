"""
CLI command groups for InvestiGator
"""

from .analyze import analyze
from .backtest import backtest
from .cache import cache
from .data import data
from .macro import macro
from .system import system

__all__ = ["analyze", "backtest", "cache", "data", "macro", "system"]
