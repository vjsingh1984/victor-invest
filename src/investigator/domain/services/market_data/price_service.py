# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Price Service - Handles historical stock price lookup.

This service provides:
- Historical price lookup from stock database
- Current price fetching
- Price range queries for technical analysis

Note: Prices in the database are typically split-adjusted, meaning historical
prices are retroactively adjusted for splits. When using historical prices
with shares data, use SharesService.get_shares_history() which normalizes
shares to match split-adjusted prices.

Example:
    service = PriceService()

    # Get price on a specific date
    price = service.get_price("AAPL", date(2024, 6, 15))

    # Get current price
    current = service.get_current_price("AAPL")

    # Get price history for a range
    history = service.get_price_history("AAPL", start_date, end_date)
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Container for price data."""

    symbol: str
    date: date
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: float
    volume: Optional[int]


class PriceService:
    """
    Service for stock price data.

    Connects to stock database for historical and current prices.
    All prices are split-adjusted.
    """

    def __init__(
        self,
        stock_db_url: str = None,
    ):
        """
        Initialize PriceService with database connection.

        Args:
            stock_db_url: Connection string for stock database.
                         If None, builds from environment variables.
        """
        from investigator.domain.services.market_data import get_stock_db_url

        if stock_db_url is None:
            stock_db_url = get_stock_db_url()

        self.stock_engine = create_engine(
            stock_db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.StockSession = sessionmaker(bind=self.stock_engine)

    def get_price(
        self,
        symbol: str,
        target_date: date,
        search_days: int = 5,
    ) -> Optional[float]:
        """
        Get stock closing price on or near target date.

        Searches backwards from target_date up to search_days to handle
        weekends and holidays.

        Args:
            symbol: Stock ticker
            target_date: Date to get price for
            search_days: Max days to search backward (default: 5)

        Returns:
            Closing price or None if not found
        """
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT close
                    FROM tickerdata
                    WHERE ticker = :symbol
                      AND date <= :target_date
                    ORDER BY date DESC
                    LIMIT 1
                """
                ),
                {"symbol": symbol, "target_date": target_date},
            ).fetchone()

            if result:
                return float(result[0])
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get most recent closing price.

        Args:
            symbol: Stock ticker

        Returns:
            Most recent closing price or None
        """
        return self.get_price(symbol, date.today())

    def get_price_data(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[PriceData]:
        """
        Get full OHLCV data for a date.

        Args:
            symbol: Stock ticker
            target_date: Date to get data for

        Returns:
            PriceData with OHLCV or None
        """
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM tickerdata
                    WHERE ticker = :symbol
                      AND date <= :target_date
                    ORDER BY date DESC
                    LIMIT 1
                """
                ),
                {"symbol": symbol, "target_date": target_date},
            ).fetchone()

            if result:
                return PriceData(
                    symbol=symbol,
                    date=result[0],
                    open=float(result[1]) if result[1] else None,
                    high=float(result[2]) if result[2] else None,
                    low=float(result[3]) if result[3] else None,
                    close=float(result[4]),
                    volume=int(result[5]) if result[5] else None,
                )
            return None

    def get_price_history(
        self,
        symbol: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get price history for a date range.

        Args:
            symbol: Stock ticker
            start_date: Start date (inclusive)
            end_date: End date (inclusive, default: today)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if end_date is None:
            end_date = date.today()

        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM tickerdata
                    WHERE ticker = :symbol
                      AND date >= :start_date
                      AND date <= :end_date
                    ORDER BY date ASC
                """
                ),
                {"symbol": symbol, "start_date": start_date, "end_date": end_date},
            ).fetchall()

            if result:
                df = pd.DataFrame(result, columns=["date", "open", "high", "low", "close", "volume"])
                return df
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    def get_price_at_lookback(
        self,
        symbol: str,
        months_back: int,
        reference_date: Optional[date] = None,
    ) -> Optional[float]:
        """
        Get price at a specific lookback period.

        Convenience method for backtesting scenarios.

        Args:
            symbol: Stock ticker
            months_back: Number of months back from reference_date
            reference_date: Reference date (default: today)

        Returns:
            Closing price or None
        """
        from dateutil.relativedelta import relativedelta

        if reference_date is None:
            reference_date = date.today()

        target_date = reference_date - relativedelta(months=months_back)
        return self.get_price(symbol, target_date)

    def get_prices_for_lookbacks(
        self,
        symbol: str,
        lookback_months: List[int],
        reference_date: Optional[date] = None,
    ) -> Dict[int, Optional[float]]:
        """
        Get prices for multiple lookback periods.

        Args:
            symbol: Stock ticker
            lookback_months: List of months back (e.g., [36, 24, 12, 6, 3])
            reference_date: Reference date (default: today)

        Returns:
            Dict mapping months_back -> price
        """
        from dateutil.relativedelta import relativedelta

        if reference_date is None:
            reference_date = date.today()

        result = {}
        for months in lookback_months:
            target_date = reference_date - relativedelta(months=months)
            result[months] = self.get_price(symbol, target_date)

        return result

    def calculate_return(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[float]:
        """
        Calculate price return between two dates.

        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            Return as decimal (e.g., 0.15 for 15% return) or None
        """
        start_price = self.get_price(symbol, start_date)
        end_price = self.get_price(symbol, end_date)

        if start_price and end_price and start_price > 0:
            return (end_price - start_price) / start_price

        return None

    def get_volatility(
        self,
        symbol: str,
        days: int = 30,
        end_date: Optional[date] = None,
    ) -> Optional[float]:
        """
        Calculate historical volatility (annualized standard deviation of returns).

        Args:
            symbol: Stock ticker
            days: Number of trading days to use
            end_date: End date (default: today)

        Returns:
            Annualized volatility or None
        """
        import numpy as np

        if end_date is None:
            end_date = date.today()

        start_date = end_date - timedelta(days=days * 2)  # Get extra days for weekends
        df = self.get_price_history(symbol, start_date, end_date)

        if len(df) < 10:
            return None

        # Take last N days
        df = df.tail(days + 1)

        # Calculate daily returns
        df["return"] = df["close"].pct_change()

        # Annualize (252 trading days)
        daily_vol = df["return"].std()
        return daily_vol * np.sqrt(252) if daily_vol else None
