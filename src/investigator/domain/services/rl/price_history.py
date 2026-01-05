"""
Price History Service

Fetches historical prices for outcome tracking in the RL system.
Provides methods to get prices on specific dates for reward calculation.

Usage:
    from investigator.domain.services.rl import PriceHistoryService

    service = PriceHistoryService()

    # Get price on specific date
    price = await service.get_price_on_date("AAPL", date(2024, 6, 15))

    # Batch fetch for multiple symbols
    prices = await service.batch_get_prices(["AAPL", "MSFT"], date(2024, 6, 15))
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from investigator.config import get_config

logger = logging.getLogger(__name__)


class PriceHistoryService:
    """
    Service for fetching historical stock prices.

    Used by OutcomeTracker to get actual prices for reward calculation.
    Connects to the market data database containing OHLCV data.
    """

    def __init__(self, engine: Optional[Engine] = None):
        """
        Initialize PriceHistoryService.

        Args:
            engine: SQLAlchemy engine for database connection.
                   If not provided, creates one from config.
        """
        self.engine = engine or self._create_engine()
        self._cache: Dict[str, float] = {}  # Simple cache: "SYMBOL_YYYY-MM-DD" -> price

    def _create_engine(self) -> Engine:
        """Create database engine for market data."""
        try:
            config = get_config()
            # Market data is in the 'stock' database
            db_url = config.database.url.replace("/sec_database", "/stock")
            return create_engine(
                db_url,
                pool_size=3,
                max_overflow=5,
                pool_pre_ping=True,
                echo=False,
            )
        except Exception as e:
            logger.warning(f"Failed to create engine from config: {e}")
            # Fallback to default connection
            return create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/stock",
                pool_size=3,
                max_overflow=5,
                pool_pre_ping=True,
            )

    async def get_price_on_date(
        self,
        symbol: str,
        target_date: date,
        use_adj_close: bool = True,
        search_days: int = 5,
    ) -> Optional[float]:
        """
        Get closing price on a specific date.

        If the target date is not a trading day (weekend/holiday),
        searches nearby dates to find the nearest trading day.

        Args:
            symbol: Stock ticker symbol.
            target_date: Target date for price lookup.
            use_adj_close: If True, use adjusted close; else use close.
            search_days: Number of days to search if target not available.

        Returns:
            Closing price on the date, or None if not found.
        """
        # Check cache first
        cache_key = f"{symbol.upper()}_{target_date.isoformat()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Run synchronous DB query in thread pool
        loop = asyncio.get_event_loop()
        price = await loop.run_in_executor(
            None,
            self._get_price_sync,
            symbol,
            target_date,
            use_adj_close,
            search_days,
        )

        # Cache result
        if price is not None:
            self._cache[cache_key] = price

        return price

    def _get_price_sync(
        self,
        symbol: str,
        target_date: date,
        use_adj_close: bool,
        search_days: int,
    ) -> Optional[float]:
        """Synchronous implementation of price fetch."""
        try:
            # Convert date to datetime for query
            if isinstance(target_date, datetime):
                target_date = target_date.date()

            price_column = "adjclose" if use_adj_close else "close"

            # Query for exact date or nearest trading day
            # Note: target_date is already a Python date, no PostgreSQL cast needed
            query = text(
                f"""
                SELECT {price_column}, date
                FROM tickerdata
                WHERE ticker = :symbol
                  AND date BETWEEN :start_date AND :end_date
                ORDER BY ABS(date - :target_date) ASC
                LIMIT 1
            """
            )

            start_date = target_date - timedelta(days=search_days)
            end_date = target_date + timedelta(days=search_days)

            with self.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "symbol": symbol.upper(),
                        "target_date": target_date,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                ).fetchone()

            if result:
                price = float(result[0]) if result[0] else None
                actual_date = result[1]
                if actual_date != target_date:
                    logger.debug(
                        f"No data for {symbol} on {target_date}, "
                        f"using {actual_date} (±{abs((actual_date - target_date).days)}d)"
                    )
                return price

            logger.warning(f"No price data found for {symbol} near {target_date}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol} on {target_date}: {e}")
            return None

    async def batch_get_prices(
        self,
        symbols: List[str],
        target_date: date,
        use_adj_close: bool = True,
    ) -> Dict[str, Optional[float]]:
        """
        Batch fetch prices for multiple symbols on the same date.

        More efficient than individual calls when fetching many symbols.

        Args:
            symbols: List of stock ticker symbols.
            target_date: Target date for price lookup.
            use_adj_close: If True, use adjusted close; else use close.

        Returns:
            Dict mapping symbols to their prices (None if not found).
        """
        if not symbols:
            return {}

        # Run synchronous batch query in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._batch_get_prices_sync,
            symbols,
            target_date,
            use_adj_close,
        )

    def _batch_get_prices_sync(
        self,
        symbols: List[str],
        target_date: date,
        use_adj_close: bool,
    ) -> Dict[str, Optional[float]]:
        """Synchronous batch price fetch."""
        try:
            price_column = "adjclose" if use_adj_close else "close"
            symbols_upper = [s.upper() for s in symbols]

            # Query for all symbols at once
            # Note: target_date is already a Python date, no PostgreSQL cast needed
            query = text(
                f"""
                WITH target_prices AS (
                    SELECT
                        ticker,
                        {price_column} as price,
                        date,
                        ROW_NUMBER() OVER (
                            PARTITION BY ticker
                            ORDER BY ABS(date - :target_date) ASC
                        ) as rn
                    FROM tickerdata
                    WHERE ticker = ANY(:symbols)
                      AND date BETWEEN :start_date AND :end_date
                )
                SELECT ticker, price, date
                FROM target_prices
                WHERE rn = 1
            """
            )

            start_date = target_date - timedelta(days=5)
            end_date = target_date + timedelta(days=5)

            with self.engine.connect() as conn:
                results = conn.execute(
                    query,
                    {
                        "symbols": symbols_upper,
                        "target_date": target_date,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                ).fetchall()

            # Build result dict
            prices = {s: None for s in symbols_upper}
            for row in results:
                ticker = row[0]
                price = float(row[1]) if row[1] else None
                prices[ticker] = price

                # Cache individual results
                cache_key = f"{ticker}_{target_date.isoformat()}"
                if price is not None:
                    self._cache[cache_key] = price

            # Log missing symbols
            missing = [s for s in symbols_upper if prices[s] is None]
            if missing:
                logger.warning(f"No price data found for {missing} near {target_date}")

            return prices

        except Exception as e:
            logger.error(f"Error batch fetching prices: {e}")
            return {s.upper(): None for s in symbols}

    async def get_price_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        use_adj_close: bool = True,
    ) -> Dict[date, float]:
        """
        Get prices for a date range.

        Args:
            symbol: Stock ticker symbol.
            start_date: Start of date range.
            end_date: End of date range.
            use_adj_close: If True, use adjusted close.

        Returns:
            Dict mapping dates to prices.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._get_price_range_sync,
            symbol,
            start_date,
            end_date,
            use_adj_close,
        )

    def _get_price_range_sync(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        use_adj_close: bool,
    ) -> Dict[date, float]:
        """Synchronous price range fetch."""
        try:
            price_column = "adjclose" if use_adj_close else "close"

            query = text(
                f"""
                SELECT date, {price_column}
                FROM tickerdata
                WHERE ticker = :symbol
                  AND date BETWEEN :start_date AND :end_date
                ORDER BY date ASC
            """
            )

            with self.engine.connect() as conn:
                results = conn.execute(
                    query,
                    {
                        "symbol": symbol.upper(),
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                ).fetchall()

            prices = {}
            for row in results:
                row_date = row[0]
                if isinstance(row_date, datetime):
                    row_date = row_date.date()
                price = float(row[1]) if row[1] else None
                if price is not None:
                    prices[row_date] = price

            return prices

        except Exception as e:
            logger.error(f"Error fetching price range for {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the most recent price for a symbol.

        Synchronous method for immediate use.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Most recent closing price, or None if not found.
        """
        try:
            query = text(
                """
                SELECT adjclose
                FROM tickerdata
                WHERE ticker = :symbol
                ORDER BY date DESC
                LIMIT 1
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol.upper()}).fetchone()

            if result:
                return float(result[0]) if result[0] else None
            return None

        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the price cache."""
        self._cache.clear()
        logger.info("Price cache cleared")


# Singleton instance
_price_history_service: Optional[PriceHistoryService] = None


def get_price_history_service() -> PriceHistoryService:
    """Get singleton PriceHistoryService instance."""
    global _price_history_service
    if _price_history_service is None:
        _price_history_service = PriceHistoryService()
    return _price_history_service
