# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Symbol Metadata Service - Centralized symbol metadata with caching.

This service provides:
- Symbol metadata lookup (sector, industry, beta, market cap)
- Caching for performance
- Consistent interface across all consumers

Example:
    service = SymbolMetadataService()

    # Get metadata
    metadata = service.get_metadata("AAPL")
    print(f"Sector: {metadata.sector}, Industry: {metadata.industry}")

    # Batch lookup
    all_meta = service.get_metadata_batch(["AAPL", "GOOGL", "MSFT"])
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


@dataclass
class SymbolMetadata:
    """Container for symbol metadata."""

    symbol: str
    sector: str
    industry: str
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    beta: Optional[float]
    is_sp500: bool = False
    is_russell1000: bool = False
    cik: Optional[str] = None

    @property
    def size_category(self) -> str:
        """Categorize by market cap."""
        if not self.market_cap:
            return "unknown"
        if self.market_cap > 200e9:
            return "mega_cap"
        elif self.market_cap > 10e9:
            return "large_cap"
        elif self.market_cap > 2e9:
            return "mid_cap"
        elif self.market_cap > 300e6:
            return "small_cap"
        else:
            return "micro_cap"


class SymbolMetadataService:
    """
    Service for symbol metadata lookup with caching.

    Provides consistent metadata access across all consumers
    (rl_backtest, batch_analysis_runner, victor_invest).
    """

    def __init__(
        self,
        stock_db_url: str = None,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize SymbolMetadataService.

        Args:
            stock_db_url: Connection string for stock database.
                         If None, builds from environment variables.
            cache_ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
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
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}  # symbol -> (metadata, timestamp)

    def get_metadata(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> Optional[SymbolMetadata]:
        """
        Get metadata for a symbol.

        Args:
            symbol: Stock ticker
            use_cache: Whether to use cache (default: True)

        Returns:
            SymbolMetadata or None if not found
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache and symbol in self._cache:
            metadata, timestamp = self._cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return metadata

        # Fetch from database
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT
                        ticker,
                        "Sector",
                        "Industry",
                        mktcap,
                        outstandingshares,
                        b_12_month,
                        sp500,
                        russell1000,
                        cik
                    FROM symbol
                    WHERE ticker = :symbol
                """
                ),
                {"symbol": symbol},
            ).fetchone()

            if not result:
                return None

            metadata = SymbolMetadata(
                symbol=result[0],
                sector=result[1] or "Unknown",
                industry=result[2] or "Unknown",
                market_cap=float(result[3]) if result[3] else None,
                shares_outstanding=float(result[4]) if result[4] else None,
                beta=float(result[5]) if result[5] else None,
                is_sp500=bool(result[6]) if result[6] else False,
                is_russell1000=bool(result[7]) if result[7] else False,
                cik=result[8],
            )

            # Cache it
            self._cache[symbol] = (metadata, datetime.now())

            return metadata

    def get_metadata_batch(
        self,
        symbols: List[str],
        use_cache: bool = True,
    ) -> Dict[str, SymbolMetadata]:
        """
        Get metadata for multiple symbols.

        Args:
            symbols: List of stock tickers
            use_cache: Whether to use cache

        Returns:
            Dict mapping symbol -> SymbolMetadata
        """
        result = {}
        uncached = []

        # Check cache first
        if use_cache:
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol in self._cache:
                    metadata, timestamp = self._cache[symbol]
                    if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                        result[symbol] = metadata
                    else:
                        uncached.append(symbol)
                else:
                    uncached.append(symbol)
        else:
            uncached = [s.upper() for s in symbols]

        # Fetch uncached symbols
        if uncached:
            with self.stock_engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT
                            ticker,
                            "Sector",
                            "Industry",
                            mktcap,
                            outstandingshares,
                            b_12_month,
                            sp500,
                            russell1000,
                            cik
                        FROM symbol
                        WHERE ticker = ANY(:symbols)
                    """
                    ),
                    {"symbols": uncached},
                ).fetchall()

                for row in rows:
                    metadata = SymbolMetadata(
                        symbol=row[0],
                        sector=row[1] or "Unknown",
                        industry=row[2] or "Unknown",
                        market_cap=float(row[3]) if row[3] else None,
                        shares_outstanding=float(row[4]) if row[4] else None,
                        beta=float(row[5]) if row[5] else None,
                        is_sp500=bool(row[6]) if row[6] else False,
                        is_russell1000=bool(row[7]) if row[7] else False,
                        cik=row[8],
                    )
                    result[metadata.symbol] = metadata
                    self._cache[metadata.symbol] = (metadata, datetime.now())

        return result

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        metadata = self.get_metadata(symbol)
        return metadata.sector if metadata else "Unknown"

    def get_industry(self, symbol: str) -> str:
        """Get industry for a symbol."""
        metadata = self.get_metadata(symbol)
        return metadata.industry if metadata else "Unknown"

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap for a symbol."""
        metadata = self.get_metadata(symbol)
        return metadata.market_cap if metadata else None

    def get_beta(self, symbol: str, default: float = 1.0) -> float:
        """Get beta for a symbol."""
        metadata = self.get_metadata(symbol)
        return metadata.beta if metadata and metadata.beta else default

    def clear_cache(self):
        """Clear the metadata cache."""
        self._cache.clear()

    def get_symbols_by_sector(self, sector: str, min_market_cap: float = 0) -> List[str]:
        """
        Get all symbols in a sector.

        Args:
            sector: Sector name
            min_market_cap: Minimum market cap filter

        Returns:
            List of symbols
        """
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT ticker
                    FROM symbol
                    WHERE "Sector" = :sector
                      AND islisted = TRUE
                      AND isstock = TRUE
                      AND (mktcap IS NULL OR mktcap >= :min_mktcap)
                    ORDER BY mktcap DESC NULLS LAST
                """
                ),
                {"sector": sector, "min_mktcap": min_market_cap},
            ).fetchall()
            return [row[0] for row in result]

    def get_russell1000_symbols(self) -> List[str]:
        """Get Russell 1000 symbols."""
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT ticker
                    FROM symbol
                    WHERE russell1000 = TRUE
                      AND islisted = TRUE
                      AND isstock = TRUE
                    ORDER BY mktcap DESC NULLS LAST
                """
                )
            ).fetchall()
            return [row[0] for row in result]

    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols."""
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT ticker
                    FROM symbol
                    WHERE sp500 = TRUE
                      AND islisted = TRUE
                      AND isstock = TRUE
                    ORDER BY mktcap DESC NULLS LAST
                """
                )
            ).fetchall()
            return [row[0] for row in result]
