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

"""Collect FINRA Short Interest Data.

Schedule: 1st and 15th of each month at 10AM ET
Source: FINRA Short Interest Reports

Collects:
- Short interest positions by symbol
- Days to cover calculations
- Short interest ratio changes
- Squeeze potential indicators

Note: FINRA releases short interest data twice monthly,
around the 9th and 24th for settlement dates of the 15th
and end of month respectively.

Usage:
    python scripts/scheduled/collect_short_interest.py
    python scripts/scheduled/collect_short_interest.py --symbols AAPL,GME,AMC
    python scripts/scheduled/collect_short_interest.py --threshold 10
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    get_database_connection,
    get_sp500_symbols,
    retry_with_backoff,
)


class ShortInterestCollector(BaseCollector):
    """Collector for FINRA short interest data."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        days_to_cover_threshold: float = 5.0,
    ):
        super().__init__("collect_short_interest")
        self.symbols = symbols
        self.days_to_cover_threshold = days_to_cover_threshold

    def collect(self) -> CollectionMetrics:
        """Collect short interest data from FINRA."""
        try:
            # Import the short interest fetcher
            from investigator.infrastructure.external.finra.short_interest import (
                get_short_interest_fetcher,
            )

            fetcher = get_short_interest_fetcher()

            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_sp500_symbols()

            self.logger.info(
                f"Fetching short interest data for {len(symbols)} symbols"
            )

            conn = get_database_connection()
            cursor = conn.cursor()

            for symbol in symbols:
                try:
                    # Fetch latest short interest
                    data = asyncio.run(
                        fetcher.get_short_interest(symbol=symbol)
                    )

                    if not data:
                        continue

                    self.metrics.records_processed += 1

                    # Calculate derived metrics
                    short_interest = data.get("short_interest", 0)
                    avg_volume = data.get("avg_daily_volume", 1)
                    shares_outstanding = data.get("shares_outstanding", 0)

                    days_to_cover = (
                        short_interest / avg_volume if avg_volume > 0 else None
                    )
                    short_interest_ratio = (
                        (short_interest / shares_outstanding * 100)
                        if shares_outstanding > 0
                        else None
                    )

                    # Detect squeeze potential
                    squeeze_potential = (
                        days_to_cover is not None
                        and days_to_cover >= self.days_to_cover_threshold
                        and short_interest_ratio is not None
                        and short_interest_ratio >= 15
                    )

                    cursor.execute("""
                        INSERT INTO short_interest
                            (symbol, settlement_date, short_interest,
                             avg_daily_volume, days_to_cover, short_interest_ratio,
                             shares_outstanding, squeeze_potential)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, settlement_date) DO UPDATE SET
                            short_interest = EXCLUDED.short_interest,
                            avg_daily_volume = EXCLUDED.avg_daily_volume,
                            days_to_cover = EXCLUDED.days_to_cover,
                            short_interest_ratio = EXCLUDED.short_interest_ratio,
                            squeeze_potential = EXCLUDED.squeeze_potential,
                            updated_at = NOW()
                    """, (
                        symbol,
                        data.get("settlement_date"),
                        short_interest,
                        avg_volume,
                        days_to_cover,
                        short_interest_ratio,
                        shares_outstanding,
                        squeeze_potential,
                    ))

                    self.metrics.records_inserted += 1

                    # Update short interest history for change tracking
                    self._update_change_metrics(cursor, symbol, data)

                except Exception as e:
                    self.logger.warning(f"Failed to process {symbol}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{symbol}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Collected short interest for {self.metrics.records_inserted} symbols"
            )

        except ImportError as e:
            self.logger.error(f"Short interest fetcher not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"Short interest collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _update_change_metrics(self, cursor, symbol: str, data: dict) -> None:
        """Calculate and store period-over-period changes."""
        try:
            settlement_date = data.get("settlement_date")
            if not settlement_date:
                return

            # Get previous period data
            cursor.execute("""
                SELECT short_interest, settlement_date
                FROM short_interest
                WHERE symbol = %s
                  AND settlement_date < %s
                ORDER BY settlement_date DESC
                LIMIT 1
            """, (symbol, settlement_date))

            prev = cursor.fetchone()
            if prev:
                prev_short_interest, prev_date = prev
                current_short_interest = data.get("short_interest", 0)

                if prev_short_interest and prev_short_interest > 0:
                    pct_change = (
                        (current_short_interest - prev_short_interest)
                        / prev_short_interest * 100
                    )

                    cursor.execute("""
                        UPDATE short_interest
                        SET short_interest_change_pct = %s,
                            prev_settlement_date = %s
                        WHERE symbol = %s AND settlement_date = %s
                    """, (pct_change, prev_date, symbol, settlement_date))

        except Exception as e:
            self.logger.debug(f"Could not update change metrics for {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect FINRA short interest data"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: S&P 500)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Days to cover threshold for squeeze detection (default: 5.0)"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    collector = ShortInterestCollector(
        symbols=symbols,
        days_to_cover_threshold=args.threshold,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
