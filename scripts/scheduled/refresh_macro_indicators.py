#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Refresh FRED Macroeconomic Indicators.

Schedule: Daily at 9AM ET (before market open)
Source: FRED API (Federal Reserve Economic Data)

Collects:
- VIX (volatility index)
- Credit spreads (BAA, AAA)
- Interest rates (Fed Funds, Treasury yields)
- Economic indicators (GDP, unemployment, inflation)
- Market indicators (S&P 500, Dow Jones)

Usage:
    python scripts/scheduled/refresh_macro_indicators.py
    python scripts/scheduled/refresh_macro_indicators.py --category rates
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
    retry_with_backoff,
)

# Key FRED indicators to refresh
INDICATOR_CATEGORIES = {
    "rates": [
        "DGS1MO",   # 1-Month Treasury
        "DGS3MO",   # 3-Month Treasury
        "DGS2",     # 2-Year Treasury
        "DGS10",    # 10-Year Treasury
        "DGS30",    # 30-Year Treasury
        "FEDFUNDS", # Fed Funds Rate
        "DPRIME",   # Prime Rate
    ],
    "credit": [
        "BAMLC0A0CM",     # ICE BofA Corporate Index
        "BAMLH0A0HYM2",   # ICE BofA High Yield
        "BAA10Y",         # BAA Corporate Bond - 10Y Treasury Spread
        "AAA10Y",         # AAA Corporate Bond - 10Y Treasury Spread
    ],
    "volatility": [
        "VIXCLS",   # VIX (CBOE Volatility Index)
    ],
    "economic": [
        "GDP",       # Gross Domestic Product
        "GDPC1",     # Real GDP
        "UNRATE",    # Unemployment Rate
        "CPIAUCSL",  # CPI (Consumer Price Index)
        "PCEPI",     # PCE Price Index
        "UMCSENT",   # Consumer Sentiment
    ],
    "markets": [
        "SP500",     # S&P 500
        "DJIA",      # Dow Jones Industrial Average
        "NASDAQCOM", # NASDAQ Composite
        "WILL5000PR", # Wilshire 5000
    ],
    "money": [
        "M2SL",      # M2 Money Supply
        "WALCL",     # Fed Balance Sheet
    ],
    "housing": [
        "MORTGAGE30US", # 30-Year Mortgage Rate
        "CSUSHPINSA",   # Case-Shiller Home Price Index
    ],
    "labor": [
        "PAYEMS",    # Nonfarm Payrolls
        "ICSA",      # Initial Jobless Claims
        "CCSA",      # Continued Jobless Claims
    ],
}


class MacroIndicatorCollector(BaseCollector):
    """Collector for FRED macroeconomic indicators."""

    def __init__(
        self,
        categories: List[str] = None,
        lookback_days: int = 30,
    ):
        super().__init__("refresh_macro_indicators")
        self.categories = categories or list(INDICATOR_CATEGORIES.keys())
        self.lookback_days = lookback_days

    def collect(self) -> CollectionMetrics:
        """Collect FRED macro indicators with incremental fetching."""
        try:
            # Import the FRED fetcher
            from investigator.infrastructure.external.fred.macro_indicators import (
                get_macro_indicator_fetcher,
            )

            fetcher = get_macro_indicator_fetcher()

            # Get list of indicators to fetch
            indicators = []
            for category in self.categories:
                if category in INDICATOR_CATEGORIES:
                    indicators.extend(INDICATOR_CATEGORIES[category])

            indicators = list(set(indicators))  # Remove duplicates
            self.logger.info(
                f"Refreshing {len(indicators)} FRED indicators from "
                f"{len(self.categories)} categories"
            )

            # Fetch each indicator
            conn = get_database_connection()
            cursor = conn.cursor()

            end_date = datetime.now().date()

            for indicator in indicators:
                try:
                    self.metrics.records_processed += 1

                    # Get indicator ID and last observation date for incremental fetch
                    cursor.execute("""
                        SELECT mi.id, MAX(mv.date) as last_date
                        FROM macro_indicators mi
                        LEFT JOIN macro_indicator_values mv ON mi.id = mv.indicator_id
                        WHERE mi.series_id = %s
                        GROUP BY mi.id
                    """, (indicator,))
                    row = cursor.fetchone()

                    indicator_id = row[0] if row else None
                    last_date = row[1] if row and row[1] else None

                    # Determine start date based on watermark
                    if last_date:
                        # Incremental: fetch from day after last observation
                        start_date = last_date + timedelta(days=1)
                        self.logger.debug(
                            f"{indicator}: Incremental fetch from {start_date} "
                            f"(last: {last_date})"
                        )
                    else:
                        # Full fetch: use configured lookback
                        start_date = end_date - timedelta(days=self.lookback_days)
                        self.logger.debug(
                            f"{indicator}: Full fetch from {start_date}"
                        )

                    # Skip if no new dates
                    if start_date > end_date:
                        self.metrics.records_skipped += 1
                        continue

                    # Fetch indicator data
                    data = asyncio.run(
                        fetcher.get_indicator_data(
                            series_id=indicator,
                            start_date=start_date.isoformat(),
                            end_date=end_date.isoformat(),
                        )
                    )

                    if not data or not data.get("values"):
                        self.logger.debug(f"No new data for {indicator}")
                        continue

                    # Get or create indicator record with hash
                    indicator_hash = compute_record_hash({
                        "series_id": indicator,
                        "name": data.get("name", indicator),
                        "category": data.get("category", "unknown"),
                        "frequency": data.get("frequency", "daily"),
                        "units": data.get("units", ""),
                    })

                    cursor.execute("""
                        INSERT INTO macro_indicators
                            (series_id, name, category, frequency, units, source_hash, source_fetch_timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (series_id) DO UPDATE SET
                            name = COALESCE(EXCLUDED.name, macro_indicators.name),
                            source_hash = EXCLUDED.source_hash,
                            source_fetch_timestamp = NOW(),
                            updated_at = NOW()
                        RETURNING id
                    """, (
                        indicator,
                        data.get("name", indicator),
                        data.get("category", "unknown"),
                        data.get("frequency", "daily"),
                        data.get("units", ""),
                        indicator_hash,
                    ))
                    indicator_id = cursor.fetchone()[0]

                    # Insert values with hash-based change detection
                    values = data.get("values", [])
                    for value in values:
                        value_date = value.get("date")
                        value_data = value.get("value")

                        # Compute hash for this value
                        value_hash = compute_record_hash({
                            "indicator_id": indicator_id,
                            "date": str(value_date),
                            "value": str(value_data),
                        })

                        # Check for existing with hash
                        cursor.execute("""
                            SELECT source_hash FROM macro_indicator_values
                            WHERE indicator_id = %s AND date = %s
                        """, (indicator_id, value_date))
                        existing = cursor.fetchone()

                        if existing:
                            if existing[0] == value_hash:
                                self.metrics.records_skipped += 1
                                continue
                            # Update
                            cursor.execute("""
                                UPDATE macro_indicator_values SET
                                    value = %s, source_hash = %s,
                                    source_fetch_timestamp = NOW(), updated_at = NOW()
                                WHERE indicator_id = %s AND date = %s
                            """, (value_data, value_hash, indicator_id, value_date))
                            self.metrics.records_updated += 1
                        else:
                            # Insert
                            cursor.execute("""
                                INSERT INTO macro_indicator_values
                                    (indicator_id, date, value, source_hash, source_fetch_timestamp)
                                VALUES (%s, %s, %s, %s, NOW())
                            """, (indicator_id, value_date, value_data, value_hash))
                            self.metrics.records_inserted += 1

                    # Update watermark
                    if values:
                        max_date = max(v.get("date") for v in values)
                        cursor.execute("""
                            INSERT INTO macro_indicator_watermarks
                                (indicator_id, last_observation_date)
                            VALUES (%s, %s)
                            ON CONFLICT (indicator_id) DO UPDATE SET
                                last_observation_date = GREATEST(
                                    macro_indicator_watermarks.last_observation_date,
                                    EXCLUDED.last_observation_date
                                ),
                                last_fetch_timestamp = NOW(),
                                fetch_count = macro_indicator_watermarks.fetch_count + 1
                        """, (indicator_id, max_date))

                    self.logger.debug(f"Updated {indicator}: {len(values)} values")

                except Exception as e:
                    self.logger.warning(f"Failed to fetch {indicator}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{indicator}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Macro indicators: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except ImportError as e:
            self.logger.error(f"FRED fetcher not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"Macro indicator refresh failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description="Refresh FRED macroeconomic indicators"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=list(INDICATOR_CATEGORIES.keys()) + ["all"],
        default="all",
        help="Category to refresh (default: all)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)"
    )
    args = parser.parse_args()

    categories = None
    if args.category != "all":
        categories = [args.category]

    collector = MacroIndicatorCollector(
        categories=categories,
        lookback_days=args.days,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
