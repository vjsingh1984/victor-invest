"""
Symbol Repository - Shared stock symbol fetching across batch runner and RL backtest.

Provides consistent access to:
- Russell 1000, S&P 500, and all stock symbols
- Domestic filer filtering (excludes foreign issuers filing 20-F/6-K)
- Top N symbols by market cap

Usage:
    from investigator.infrastructure.database.symbol_repository import SymbolRepository

    repo = SymbolRepository()
    symbols = repo.get_russell1000_symbols()
    domestic = repo.get_domestic_filers()

Author: Victor-Invest Team
Date: 2026-01-05
"""

import logging
import os
from typing import List, Optional, Set

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class SymbolRepository:
    """
    Repository for fetching stock symbols from databases.

    Connects to both stock database (symbol metadata) and SEC database
    (filing data) to provide filtered symbol lists.
    """

    def __init__(
        self,
        stock_engine: Optional[Engine] = None,
        sec_engine: Optional[Engine] = None,
    ):
        """
        Initialize with database connections.

        Args:
            stock_engine: SQLAlchemy engine for stock database (optional, will create from env)
            sec_engine: SQLAlchemy engine for SEC database (optional, will create from env)
        """
        self.stock_engine = stock_engine or self._create_stock_engine()
        self.sec_engine = sec_engine or self._create_sec_engine()

    def _create_stock_engine(self) -> Engine:
        """Create stock database engine from environment variables."""
        host = os.environ.get("STOCK_DB_HOST", "localhost")
        password = os.environ.get("STOCK_DB_PASSWORD", "")
        user = os.environ.get("STOCK_DB_USER", "stockuser")
        db = os.environ.get("STOCK_DB_NAME", "stock")

        return create_engine(
            f"postgresql://{user}:{password}@{host}:5432/{db}",
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def _create_sec_engine(self) -> Engine:
        """Create SEC database engine from environment variables."""
        host = os.environ.get("SEC_DB_HOST", "localhost")
        password = os.environ.get("SEC_DB_PASSWORD", "")
        user = os.environ.get("SEC_DB_USER", "investigator")
        db = os.environ.get("SEC_DB_NAME", "sec_database")

        return create_engine(
            f"postgresql://{user}:{password}@{host}:5432/{db}",
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def get_russell1000_symbols(self) -> List[str]:
        """
        Get Russell 1000 symbols from stock database.

        Returns:
            List of ticker symbols sorted by market cap (descending)
        """
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT ticker
                    FROM symbol
                    WHERE russell1000 = TRUE
                      AND islisted = TRUE
                      AND isstock = TRUE
                      AND (isetf IS NULL OR isetf = FALSE)
                    ORDER BY mktcap DESC NULLS LAST
                """
                )
            )
            symbols = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(symbols)} Russell 1000 symbols")
            return symbols

    def get_sp500_symbols(self) -> List[str]:
        """
        Get S&P 500 symbols from stock database.

        Returns:
            List of ticker symbols sorted by market cap (descending)
        """
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT ticker
                    FROM symbol
                    WHERE sp500 = TRUE
                      AND islisted = TRUE
                      AND isstock = TRUE
                      AND (isetf IS NULL OR isetf = FALSE)
                    ORDER BY mktcap DESC NULLS LAST
                """
                )
            )
            symbols = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(symbols)} S&P 500 symbols")
            return symbols

    def get_all_symbols(self, us_only: bool = True, order_by: str = "mktcap") -> List[str]:
        """
        Get ALL stocks from symbol table (excludes ETFs/ETNs).

        Args:
            us_only: If True, only return stocks with SEC CIK (domestic filers)
            order_by: Sort order - "mktcap" (descending), "stockid" (ascending), or "ticker" (alphabetical)

        Returns:
            List of ticker symbols sorted by specified order
        """
        # Build ORDER BY clause based on parameter
        order_clause = {
            "mktcap": "ORDER BY mktcap DESC",
            "stockid": "ORDER BY stockid ASC",
            "ticker": "ORDER BY ticker ASC",
        }.get(order_by, "ORDER BY mktcap DESC")

        with self.stock_engine.connect() as conn:
            if us_only:
                result = conn.execute(
                    text(
                        f"""
                        SELECT ticker
                        FROM symbol
                        WHERE islisted = TRUE
                          AND isstock = TRUE
                          AND (isetf IS NULL OR isetf = FALSE)
                          AND mktcap IS NOT NULL
                          AND mktcap > 0
                          AND cik IS NOT NULL
                        {order_clause}
                    """
                    )
                )
            else:
                result = conn.execute(
                    text(
                        f"""
                        SELECT ticker
                        FROM symbol
                        WHERE islisted = TRUE
                          AND isstock = TRUE
                          AND (isetf IS NULL OR isetf = FALSE)
                          AND mktcap IS NOT NULL
                          AND mktcap > 0
                        {order_clause}
                    """
                    )
                )
            symbols = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(symbols)} total stocks (us_only={us_only}, order_by={order_by})")
            return symbols

    def get_top_n_symbols(self, n: int, us_only: bool = True) -> List[str]:
        """
        Get top N stocks by market cap (excludes ETFs/ETNs).

        Args:
            n: Number of symbols to return
            us_only: If True, only return stocks with SEC CIK

        Returns:
            List of ticker symbols sorted by market cap (descending)
        """
        with self.stock_engine.connect() as conn:
            if us_only:
                result = conn.execute(
                    text(
                        """
                        SELECT ticker
                        FROM symbol
                        WHERE islisted = TRUE
                          AND isstock = TRUE
                          AND (isetf IS NULL OR isetf = FALSE)
                          AND mktcap IS NOT NULL
                          AND mktcap > 0
                          AND cik IS NOT NULL
                        ORDER BY mktcap DESC
                        LIMIT :n
                    """
                    ),
                    {"n": n},
                )
            else:
                result = conn.execute(
                    text(
                        """
                        SELECT ticker
                        FROM symbol
                        WHERE islisted = TRUE
                          AND isstock = TRUE
                          AND (isetf IS NULL OR isetf = FALSE)
                          AND mktcap IS NOT NULL
                          AND mktcap > 0
                        ORDER BY mktcap DESC
                        LIMIT :n
                    """
                    ),
                    {"n": n},
                )
            symbols = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(symbols)} top symbols by market cap")
            return symbols

    def get_domestic_filers(self) -> Set[str]:
        """
        Get symbols that file 10-K/10-Q (domestic filers).

        Excludes foreign private issuers who file 20-F/6-K as they
        lack quarterly data needed for proper valuation.

        Returns:
            Set of ticker symbols with quarterly SEC data
        """
        with self.sec_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT DISTINCT symbol
                    FROM sec_companyfacts_processed
                    WHERE fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
                """
                )
            )
            domestic = {row[0] for row in result.fetchall()}
            logger.info(f"Found {len(domestic)} domestic filers with quarterly data")
            return domestic

    def get_symbols_with_sec_data(self, min_market_cap: float = 1_000_000_000) -> List[str]:
        """
        Get symbols that exist in BOTH stock and SEC databases.

        Useful for RL backtesting where both price data and fundamentals are needed.

        Args:
            min_market_cap: Minimum market cap filter (default: $1B)

        Returns:
            List of ticker symbols sorted by market cap
        """
        # Get symbols from SEC database with financial data
        with self.sec_engine.connect() as conn:
            sec_result = conn.execute(
                text(
                    """
                    SELECT DISTINCT symbol
                    FROM sec_companyfacts_processed
                    WHERE total_revenue IS NOT NULL
                      AND net_income IS NOT NULL
                """
                )
            )
            sec_symbols = {row[0] for row in sec_result.fetchall()}

        # Get symbols from stock database with market cap
        with self.stock_engine.connect() as conn:
            stock_result = conn.execute(
                text(
                    """
                    SELECT ticker
                    FROM symbol
                    WHERE islisted = TRUE
                      AND isstock = TRUE
                      AND (isetf IS NULL OR isetf = FALSE)
                      AND mktcap > :min_cap
                    ORDER BY mktcap DESC
                """
                ),
                {"min_cap": min_market_cap},
            )
            stock_symbols = [row[0] for row in stock_result.fetchall()]

        # Return intersection, maintaining market cap order
        valid_symbols = [s for s in stock_symbols if s in sec_symbols]
        logger.info(
            f"Found {len(valid_symbols)} symbols with both stock and SEC data "
            f"(min_cap=${min_market_cap:,.0f})"
        )
        return valid_symbols

    def filter_domestic_filers(
        self,
        symbols: List[str],
        skip_filter: bool = False,
    ) -> List[str]:
        """
        Filter symbols to only include domestic filers.

        Args:
            symbols: List of symbols to filter
            skip_filter: If True, return original list without filtering

        Returns:
            Filtered list of symbols (or original if skip_filter=True)
        """
        if skip_filter:
            return symbols

        domestic = self.get_domestic_filers()
        filtered = [s for s in symbols if s in domestic]
        removed = len(symbols) - len(filtered)

        if removed > 0:
            logger.info(
                f"Filtered out {removed} foreign filers (20-F/6-K) - "
                f"{len(filtered)} domestic filers remaining"
            )

        return filtered


# Singleton instance for convenience
_default_repository: Optional[SymbolRepository] = None


def get_symbol_repository() -> SymbolRepository:
    """Get or create the default SymbolRepository instance."""
    global _default_repository
    if _default_repository is None:
        _default_repository = SymbolRepository()
    return _default_repository
