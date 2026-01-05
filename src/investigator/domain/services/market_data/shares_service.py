# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Shares Service - Handles shares outstanding with split detection and normalization.

This service provides:
- Historical shares lookup from SEC filings
- Automatic stock split detection (forward and reverse)
- Split-adjusted shares normalization for historical analysis

Key insight: Stock prices in most databases are split-adjusted, but SEC filing
shares are NOT split-adjusted. This creates a mismatch when calculating historical
market cap or ratios. This service detects splits and normalizes historical
shares to match split-adjusted prices.

Example:
    service = SharesService()

    # Get split-normalized shares for backtesting
    shares_df = service.get_shares_history("NVDA", [36, 24, 12, 6, 3])
    # Returns DataFrame with columns: months_back, as_of_date, raw_shares, split_factor, adjusted_shares

    # NVDA had a 10:1 split in June 2024
    # Pre-split periods will have split_factor=10, adjusted_shares = raw_shares * 10
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Split detection thresholds
FORWARD_SPLIT_THRESHOLD = 1.8  # Shares increased by 1.8x+ indicates forward split
REVERSE_SPLIT_THRESHOLD = 0.55  # Shares decreased by 55%+ indicates reverse split


@dataclass
class SharesHistory:
    """Container for shares history with split information."""

    symbol: str
    months_back: int
    as_of_date: date
    raw_shares: float
    split_factor: float
    adjusted_shares: float

    @property
    def has_split_adjustment(self) -> bool:
        """True if this period required split normalization."""
        return self.split_factor != 1.0


class SharesService:
    """
    Service for shares outstanding data with split detection and normalization.

    Connects to both SEC database (for SEC filing shares) and stock database
    (for symbol table shares as fallback).
    """

    def __init__(
        self,
        sec_db_url: str = None,
        stock_db_url: str = None,
    ):
        """
        Initialize SharesService with database connections.

        Args:
            sec_db_url: Connection string for SEC database.
                       If None, builds from environment variables.
            stock_db_url: Connection string for stock database.
                         If None, builds from environment variables.
        """
        from investigator.domain.services.market_data import get_sec_db_url, get_stock_db_url

        if sec_db_url is None:
            sec_db_url = get_sec_db_url()
        if stock_db_url is None:
            stock_db_url = get_stock_db_url()

        self.sec_engine = create_engine(
            sec_db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.stock_engine = create_engine(
            stock_db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.SecSession = sessionmaker(bind=self.sec_engine)
        self.StockSession = sessionmaker(bind=self.stock_engine)

    def get_sec_shares(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[float]:
        """
        Get shares outstanding from SEC filing.

        Args:
            symbol: Stock ticker
            as_of_date: Get shares as of this date (default: most recent)

        Returns:
            Shares outstanding or None if not found
        """
        with self.sec_engine.connect() as conn:
            if as_of_date:
                query = text(
                    """
                    SELECT shares_outstanding
                    FROM sec_companyfacts_processed
                    WHERE symbol = :symbol
                      AND filed_date <= :as_of_date
                      AND shares_outstanding IS NOT NULL
                    ORDER BY filed_date DESC
                    LIMIT 1
                """
                )
                result = conn.execute(query, {"symbol": symbol, "as_of_date": as_of_date}).fetchone()
            else:
                query = text(
                    """
                    SELECT shares_outstanding
                    FROM sec_companyfacts_processed
                    WHERE symbol = :symbol
                      AND shares_outstanding IS NOT NULL
                    ORDER BY filed_date DESC
                    LIMIT 1
                """
                )
                result = conn.execute(query, {"symbol": symbol}).fetchone()

            if result and result[0]:
                return float(result[0])
            return None

    def get_symbol_table_shares(self, symbol: str) -> Optional[float]:
        """
        Get shares outstanding from symbol table (fallback).

        Args:
            symbol: Stock ticker

        Returns:
            Shares outstanding or None if not found
        """
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text("SELECT outstandingshares FROM symbol WHERE ticker = :symbol"), {"symbol": symbol}
            ).fetchone()

            if result and result[0]:
                return float(result[0])
            return None

    def get_shares(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
        prefer_sec: bool = True,
    ) -> Optional[float]:
        """
        Get shares outstanding with source priority.

        Priority:
        1. SEC filing shares (most accurate, from actual filings)
        2. Symbol table shares (fallback)

        Args:
            symbol: Stock ticker
            as_of_date: Get shares as of this date
            prefer_sec: If True, prefer SEC filing data (default: True)

        Returns:
            Shares outstanding or None if not found
        """
        if prefer_sec:
            sec_shares = self.get_sec_shares(symbol, as_of_date)
            if sec_shares:
                return sec_shares

        symbol_shares = self.get_symbol_table_shares(symbol)
        if symbol_shares:
            return symbol_shares

        return None

    def get_shares_history(
        self,
        symbol: str,
        lookback_months: List[int],
        reference_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get shares outstanding history with split detection and normalization.

        Since stock prices in most databases are split-adjusted, but SEC filing
        shares are NOT split-adjusted, this method detects splits and normalizes
        historical shares to match split-adjusted prices.

        Algorithm:
        1. Fetch shares for each lookback period from SEC filings
        2. Compare consecutive periods to detect splits (>1.8x or <0.55x change)
        3. Apply cumulative split factor to pre-split periods

        Args:
            symbol: Stock ticker
            lookback_months: List of months back to fetch (e.g., [36, 24, 12, 6, 3])
            reference_date: Reference date for lookback (default: today)

        Returns:
            DataFrame with columns:
            - months_back: Months before reference_date
            - as_of_date: The date for this period
            - raw_shares: Shares from SEC filing (not split-adjusted)
            - split_factor: Cumulative split multiplier to apply
            - adjusted_shares: raw_shares * split_factor (matches split-adjusted prices)
        """
        if reference_date is None:
            reference_date = date.today()

        records = []

        # Get shares for each lookback period
        for months_back in sorted(lookback_months, reverse=True):  # Start from oldest
            as_of_date = reference_date - relativedelta(months=months_back)
            shares = self.get_sec_shares(symbol, as_of_date)
            if shares:
                records.append(
                    {
                        "months_back": months_back,
                        "as_of_date": as_of_date,
                        "raw_shares": shares,
                    }
                )

        if not records:
            return pd.DataFrame(columns=["months_back", "as_of_date", "raw_shares", "split_factor", "adjusted_shares"])

        # Sort by months_back descending (oldest first, e.g., 36, 33, 30, ...)
        df = pd.DataFrame(records).sort_values("months_back", ascending=False).reset_index(drop=True)

        # Detect splits by comparing each period to the next (more recent) period
        df["split_factor"] = 1.0

        # Work backwards from most recent (smallest months_back) to oldest
        # The most recent period is our "reference" with factor 1.0
        # Earlier periods need to be multiplied by the split factor
        cumulative_factor = 1.0

        # Reverse iterate: from most recent to oldest
        for i in range(len(df) - 1, 0, -1):
            current_shares = df.loc[i, "raw_shares"]  # More recent
            prev_shares = df.loc[i - 1, "raw_shares"]  # Older

            if current_shares > 0 and prev_shares > 0:
                ratio = current_shares / prev_shares  # How much did shares increase?

                if ratio > FORWARD_SPLIT_THRESHOLD:  # Forward split detected (10:1, 4:1, etc.)
                    split_mult = round(ratio)
                    cumulative_factor *= split_mult
                    logger.info(
                        f"{symbol}: Detected {split_mult}:1 split between "
                        f"{df.loc[i-1, 'months_back']}m and {df.loc[i, 'months_back']}m"
                    )
                elif ratio < REVERSE_SPLIT_THRESHOLD:  # Reverse split detected
                    split_mult = round(1 / ratio)
                    cumulative_factor /= split_mult
                    logger.info(
                        f"{symbol}: Detected 1:{split_mult} reverse split between "
                        f"{df.loc[i-1, 'months_back']}m and {df.loc[i, 'months_back']}m"
                    )

            # Apply cumulative factor to the older period
            df.loc[i - 1, "split_factor"] = cumulative_factor

        # Adjusted shares = raw_shares * split_factor
        # Pre-split periods get multiplied to match post-split share count
        df["adjusted_shares"] = df["raw_shares"] * df["split_factor"]

        return df[["months_back", "as_of_date", "raw_shares", "split_factor", "adjusted_shares"]]

    def get_adjusted_shares(
        self,
        symbol: str,
        as_of_date: date,
        lookback_months: List[int],
        reference_date: Optional[date] = None,
    ) -> Optional[float]:
        """
        Get split-adjusted shares for a specific lookback period.

        Convenience method that extracts a single value from get_shares_history.

        Args:
            symbol: Stock ticker
            as_of_date: The date to get shares for
            lookback_months: All lookback periods (needed for split detection)
            reference_date: Reference date for lookback (default: today)

        Returns:
            Split-adjusted shares or None if not found
        """
        df = self.get_shares_history(symbol, lookback_months, reference_date)

        if df.empty:
            return None

        # Find the row closest to as_of_date
        df["date_diff"] = abs((df["as_of_date"] - as_of_date).dt.days)
        closest_row = df.loc[df["date_diff"].idxmin()]

        return closest_row["adjusted_shares"]

    def detect_splits(
        self,
        symbol: str,
        lookback_months: List[int],
        reference_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect stock splits in historical data.

        Args:
            symbol: Stock ticker
            lookback_months: List of months back to check
            reference_date: Reference date for lookback (default: today)

        Returns:
            List of detected splits with details:
            - split_ratio: e.g., "10:1" for forward, "1:10" for reverse
            - between_periods: (older_months, newer_months)
            - approximate_date: Estimated split date
        """
        df = self.get_shares_history(symbol, lookback_months, reference_date)

        if df.empty:
            return []

        splits = []

        for i in range(len(df) - 1, 0, -1):
            current_shares = df.loc[i, "raw_shares"]
            prev_shares = df.loc[i - 1, "raw_shares"]

            if current_shares > 0 and prev_shares > 0:
                ratio = current_shares / prev_shares

                if ratio > FORWARD_SPLIT_THRESHOLD:
                    split_mult = round(ratio)
                    splits.append(
                        {
                            "type": "forward",
                            "split_ratio": f"{split_mult}:1",
                            "between_periods": (df.loc[i - 1, "months_back"], df.loc[i, "months_back"]),
                            "approximate_date_range": (df.loc[i - 1, "as_of_date"], df.loc[i, "as_of_date"]),
                            "shares_before": prev_shares,
                            "shares_after": current_shares,
                        }
                    )
                elif ratio < REVERSE_SPLIT_THRESHOLD:
                    split_mult = round(1 / ratio)
                    splits.append(
                        {
                            "type": "reverse",
                            "split_ratio": f"1:{split_mult}",
                            "between_periods": (df.loc[i - 1, "months_back"], df.loc[i, "months_back"]),
                            "approximate_date_range": (df.loc[i - 1, "as_of_date"], df.loc[i, "as_of_date"]),
                            "shares_before": prev_shares,
                            "shares_after": current_shares,
                        }
                    )

        return splits
