#!/usr/bin/env python3
"""
InvestiGator - Market Data Database Fetcher
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Fetches market data from PostgreSQL database instead of external APIs
Connects to read-only market data database on ${DB_HOST:-localhost}
"""

import logging
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# Singleton instance and lock for thread-safe access
_market_data_fetcher_instance: Optional["DatabaseMarketDataFetcher"] = None
_market_data_fetcher_lock = threading.Lock()


def get_market_data_fetcher(config) -> "DatabaseMarketDataFetcher":
    """
    Get singleton instance of DatabaseMarketDataFetcher.

    Uses thread-safe lazy initialization to avoid creating multiple
    database connections and engine instances.

    Args:
        config: Configuration object with analysis and database settings

    Returns:
        Shared DatabaseMarketDataFetcher instance
    """
    global _market_data_fetcher_instance

    if _market_data_fetcher_instance is not None:
        return _market_data_fetcher_instance

    with _market_data_fetcher_lock:
        # Double-check locking pattern
        if _market_data_fetcher_instance is None:
            _market_data_fetcher_instance = DatabaseMarketDataFetcher(config)
        return _market_data_fetcher_instance


class DatabaseMarketDataFetcher:
    """Fetches market data from PostgreSQL database"""

    def __init__(self, config):
        self.config = config
        self.min_volume = getattr(config.analysis, "min_volume", 10000)
        self.default_days = getattr(config.analysis, "technical_lookback_days", 365)
        self._symbol_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Warning tuning – prevent false alarms on intentionally short lookbacks.
        self.history_warning_days = getattr(config.analysis, "history_warning_days", 50)
        self.history_warning_tolerance = getattr(config.analysis, "history_warning_tolerance", 0.8)
        # Clamp tolerance to sensible bounds (10%-100%)
        self.history_warning_tolerance = min(1.0, max(0.1, self.history_warning_tolerance))
        self.low_volume_notice_min_days = getattr(
            config.analysis, "low_volume_notice_min_days", 30
        )  # only escalate low volume when we have a reasonable sample size

        # Database connection parameters for read-only market data database
        self.db_config = {
            "host": "${DB_HOST:-localhost}",
            "port": 5432,
            "database": "stock",  # Based on the psql command showing stock-# prompt
            "username": "investigator",
            "password": "investigator",
        }

        # Create database engine with connection pooling
        self.engine = self._create_engine()
        logger.info("Initialized DatabaseMarketDataFetcher with market data database")

    def _create_engine(self):
        """Create SQLAlchemy engine for database connection"""
        connection_string = (
            f"postgresql://{self.db_config['username']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )

        return create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,
        )

    def get_stock_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """
        Fetch comprehensive stock data from database

        Args:
            symbol: Stock ticker symbol
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        try:
            if days is None:
                days = self.default_days

            logger.info(f"Fetching {days} days of market data for {symbol} from database")

            # SQL query to fetch OHLCV data - get exact N trading days using LIMIT
            # Database contains only trading days (no weekends/holidays)
            query = text(
                """
                SELECT
                    date,
                    open,
                    high,
                    low,
                    close,
                    adjclose,
                    volume
                FROM tickerdata
                WHERE ticker = :symbol
                ORDER BY date DESC
                LIMIT :limit_days
            """
            )

            # Execute query and fetch data
            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params={"symbol": symbol.upper(), "limit_days": days},
                    index_col="date",
                    parse_dates=["date"],
                )

            if df.empty:
                logger.warning(f"No data returned for {symbol} from database")
                return pd.DataFrame()

            # Reverse order since we fetched DESC (most recent first)
            df = df.sort_index(ascending=True)

            # Rename columns to match yfinance format for compatibility
            df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            # Convert numeric columns to float
            numeric_cols = ["Open", "High", "Low", "Close", "Adj Close"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert volume to int
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(np.int64)

            # Remove any rows with all NaN values
            df = df.dropna(how="all")

            # Determine dynamic thresholds to avoid noisy warnings on short lookbacks
            expected_rows = self._expected_history_rows(days)
            if len(df) < expected_rows:
                logger.warning(
                    "Limited history for %s: received %d rows (< %d expected for %d-day request)",
                    symbol,
                    len(df),
                    expected_rows,
                    days,
                )

            # Check volume requirement – only escalate when we have a reasonable sample
            avg_volume = df["Volume"].mean()
            if avg_volume < self.min_volume:
                log_fn = logger.info if len(df) < self.low_volume_notice_min_days else logger.warning
                log_fn(
                    "Low volume for %s: %s < %s (lookback %d days)",
                    symbol,
                    f"{avg_volume:,.0f}",
                    f"{self.min_volume:,.0f}",
                    len(df),
                )

            logger.info("Successfully fetched %d days of data for %s from database", len(df), symbol)
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from database: {e}")
            return pd.DataFrame()

    def _expected_history_rows(self, requested_days: int) -> int:
        """
        Determine how many rows we reasonably expect from the database for a given request.

        Short windows (10d/21d) should not trigger 50-row warnings; we scale expectation by the
        configured tolerance and cap it at `history_warning_days`.
        """
        days = requested_days or self.default_days
        cap = max(1, min(self.history_warning_days, days))
        expected = int(cap * self.history_warning_tolerance)
        return max(5, expected)

    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Async alias for get_stock_data to match technical agent expectations

        Args:
            symbol: Stock ticker symbol
            period: Period string (e.g., '1y', '3mo', '1mo') - converted to days

        Returns:
            DataFrame with OHLCV data
        """
        import asyncio

        # Convert period string to days
        period_map = {"1d": 1, "1mo": 30, "3mo": 90, "1y": 365, "5y": 1825}

        days = period_map.get(period, 365)

        # Run synchronous method in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stock_data, symbol, days)

    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get additional stock information from database including symbol table data
        """
        try:
            company_info = self._get_symbol_metadata(symbol)

            # Get latest price data for basic calculations
            query_price = text(
                """
                SELECT 
                    close as current_price,
                    volume as current_volume
                FROM tickerdata
                WHERE ticker = :symbol
                ORDER BY date DESC
                LIMIT 1
            """
            )

            with self.engine.connect() as conn:
                price_info = conn.execute(query_price, {"symbol": symbol.upper()}).fetchone()

            if not price_info:
                logger.warning(f"No recent price data found for {symbol}")
                current_price = None
                current_volume = None
            else:
                current_price = float(price_info.current_price) if price_info.current_price else None
                current_volume = int(price_info.current_volume) if price_info.current_volume else None

            # Calculate 52-week high/low
            query_52w = text(
                """
                SELECT 
                    MAX(high) as week_52_high,
                    MIN(low) as week_52_low,
                    AVG(volume) as avg_volume
                FROM tickerdata
                WHERE ticker = :symbol
                    AND date >= CURRENT_DATE - INTERVAL '52 weeks'
            """
            )

            with self.engine.connect() as conn:
                result_52w = conn.execute(query_52w, {"symbol": symbol.upper()}).fetchone()

            # Calculate market cap if we have shares outstanding and current price
            market_cap = None
            shares_outstanding = self._extract_int(
                company_info.get("outstandingshares") or company_info.get("shares_outstanding")
            )
            if shares_outstanding and current_price:
                market_cap = shares_outstanding * current_price

            # Use 12-month beta as default, fallback to longer periods if not available
            beta = None
            beta_candidates = [
                company_info.get("b_12_month"),
                company_info.get("beta_12m"),
                company_info.get("b_24_month"),
                company_info.get("beta_24m"),
                company_info.get("b_36_month"),
                company_info.get("beta_36m"),
                company_info.get("b_60_month"),
                company_info.get("beta_60m"),
            ]
            for candidate in beta_candidates:
                if candidate is not None:
                    try:
                        beta = float(candidate)
                        break
                    except (TypeError, ValueError):
                        continue

            is_etf = self._infer_is_etf(symbol, company_info)

            info = {
                "current_price": current_price,
                "current_volume": current_volume,
                "52_week_high": float(result_52w.week_52_high) if result_52w and result_52w.week_52_high else None,
                "52_week_low": float(result_52w.week_52_low) if result_52w and result_52w.week_52_low else None,
                "avg_volume": int(result_52w.avg_volume) if result_52w and result_52w.avg_volume else None,
                "market_cap": market_cap,
                "sector": self._extract_first_nonempty(company_info, ["sec_sector", "sector", "gics_sector"]),
                "industry": self._extract_first_nonempty(company_info, ["sec_industry", "industry", "gics_industry"]),
                "beta": beta,
                "cik": self._format_cik(company_info.get("cik")),
                "sic_code": company_info.get("sic_code"),
                "shares_outstanding": shares_outstanding,
                "float_shares": None,  # Not available in current schema
                "pe_ratio": None,  # Would need earnings data
                "forward_pe": None,  # Would need forward earnings data
                "dividend_yield": None,  # Would need dividend data
                "is_etf": is_etf,
                "asset_type": self._extract_first_nonempty(
                    company_info,
                    ["asset_type", "assettype", "security_type", "instrument_type", "asset_class"],
                ),
            }

            return info

        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol} from database: {e}")
            return {}

    async def get_quote(self, symbol: str) -> Dict:
        """
        Async method to get current quote data for fundamental analysis

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with current price, market cap, and other quote data
        """
        import asyncio

        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stock_info, symbol)

    def check_database_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_available_symbols(self) -> list:
        """Get list of available symbols in the database"""
        try:
            query = text(
                """
                SELECT DISTINCT ticker 
                FROM tickerdata 
                ORDER BY ticker
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query)
                symbols = [row[0] for row in result]

            logger.info(f"Found {len(symbols)} symbols in database")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return []

    def _get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw symbol table metadata with caching."""
        symbol_key = symbol.upper()
        if symbol_key in self._symbol_metadata_cache:
            return self._symbol_metadata_cache[symbol_key]

        query = text("SELECT * FROM symbol WHERE ticker = :symbol")
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query, {"symbol": symbol_key}).mappings().fetchone()
        except Exception as e:
            logger.error(f"Error fetching symbol metadata for {symbol_key}: {e}")
            row = None

        metadata = dict(row) if row else {}
        self._symbol_metadata_cache[symbol_key] = metadata
        return metadata

    @staticmethod
    def _extract_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            if isinstance(value, Decimal):
                value = float(value)
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_cik(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            return f"{int(value):010d}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _extract_first_nonempty(metadata: Dict[str, Any], keys: list) -> Optional[Any]:
        for key in keys:
            if key in metadata and metadata[key]:
                return metadata[key]
        return None

    @classmethod
    def _infer_is_etf(cls, symbol: str, metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return False

        def coerce_bool(value: Any) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float, Decimal)):
                return bool(value)
            if isinstance(value, str):
                text_value = value.strip().upper()
                if text_value in {"Y", "YES", "TRUE", "T", "1"}:
                    return True
                if text_value in {"N", "NO", "FALSE", "F", "0"}:
                    return False
            return None

        bool_fields = ["is_etf", "is_etf_flag", "etf_flag", "fund_is_etf", "is_exchange_traded_fund"]
        for field in bool_fields:
            if field in metadata:
                coerced = coerce_bool(metadata[field])
                if coerced is not None:
                    return coerced

        text_fields = [
            "asset_type",
            "assettype",
            "security_type",
            "instrument_type",
            "asset_class",
            "fund_type",
            "category",
            "type",
        ]
        for field in text_fields:
            value = metadata.get(field)
            if isinstance(value, str):
                upper = value.upper()
                if "ETF" in upper or "EXCHANGE-TRADED FUND" in upper or "EXCHANGE TRADED FUND" in upper or "ETP" in upper:
                    return True

        name_fields = ["companyname", "security_name", "name", "description"]
        for field in name_fields:
            value = metadata.get(field)
            if isinstance(value, str) and "ETF" in value.upper():
                return True

        return False
