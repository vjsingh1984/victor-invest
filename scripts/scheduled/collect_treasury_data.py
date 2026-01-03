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

"""Collect Treasury Yield Curve Data.

Schedule: Daily at 6PM ET (after market close)
Source: Treasury Fiscal Data API

Collects:
- Daily yield curve rates (1M to 30Y)
- Calculates 10Y-2Y and 10Y-3M spreads
- Detects yield curve inversion
- Updates treasury_yields table

Usage:
    python scripts/scheduled/collect_treasury_data.py
    python scripts/scheduled/collect_treasury_data.py --days 30
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
    get_last_date,
    retry_with_backoff,
)


class TreasuryDataCollector(BaseCollector):
    """Collector for Treasury yield curve data."""

    def __init__(self, lookback_days: int = 7):
        super().__init__("collect_treasury_data")
        self.lookback_days = lookback_days

    def collect(self) -> CollectionMetrics:
        """Collect Treasury yield curve data with incremental fetching."""
        try:
            # Import the treasury fetcher
            from investigator.infrastructure.external.treasury.treasury_api import (
                get_treasury_fetcher,
            )

            fetcher = get_treasury_fetcher()
            conn = get_database_connection()
            cursor = conn.cursor()

            # Get last date in database for incremental fetching
            last_date = get_last_date(cursor, "treasury_yields", "date")

            end_date = datetime.now().date()

            if last_date:
                # Incremental: fetch from day after last date
                start_date = last_date + timedelta(days=1)
                self.logger.info(
                    f"Incremental fetch: {start_date} to {end_date} "
                    f"(last record: {last_date})"
                )
            else:
                # Full fetch: use configured lookback
                start_date = end_date - timedelta(days=self.lookback_days)
                self.logger.info(
                    f"Full fetch: {start_date} to {end_date} (no existing data)"
                )

            # Skip if no new dates to fetch
            if start_date > end_date:
                self.logger.info("No new dates to fetch, data is current")
                return self.metrics

            # Use async fetch
            yields = asyncio.run(
                fetcher.get_yield_curve_history(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
            )

            if not yields:
                self.logger.warning("No Treasury yield data returned")
                cursor.close()
                conn.close()
                return self.metrics

            self.metrics.records_processed = len(yields)
            self.logger.info(f"Fetched {len(yields)} yield curve records")

            for yield_data in yields:
                try:
                    record_date = yield_data.get("date")

                    # Calculate spreads
                    yield_10y = yield_data.get("yield_10y")
                    yield_2y = yield_data.get("yield_2y")
                    yield_3m = yield_data.get("yield_3m")

                    spread_10y_2y = None
                    spread_10y_3m = None
                    is_inverted = False

                    if yield_10y and yield_2y:
                        spread_10y_2y = yield_10y - yield_2y
                        is_inverted = spread_10y_2y < 0

                    if yield_10y and yield_3m:
                        spread_10y_3m = yield_10y - yield_3m

                    # Compute hash for change detection
                    record_hash = compute_record_hash(yield_data)

                    # Check if record exists and has changed
                    cursor.execute(
                        "SELECT source_hash FROM treasury_yields WHERE date = %s",
                        (record_date,),
                    )
                    existing = cursor.fetchone()

                    if existing:
                        if existing[0] == record_hash:
                            # No change, skip
                            self.metrics.records_skipped += 1
                            continue
                        # Update existing record
                        cursor.execute("""
                            UPDATE treasury_yields SET
                                yield_1m = %s, yield_3m = %s, yield_6m = %s, yield_1y = %s,
                                yield_2y = %s, yield_5y = %s, yield_10y = %s, yield_20y = %s,
                                yield_30y = %s, spread_10y_2y = %s, spread_10y_3m = %s,
                                is_inverted = %s, source_hash = %s,
                                source_fetch_timestamp = NOW(), updated_at = NOW()
                            WHERE date = %s
                        """, (
                            yield_data.get("yield_1m"),
                            yield_data.get("yield_3m"),
                            yield_data.get("yield_6m"),
                            yield_data.get("yield_1y"),
                            yield_data.get("yield_2y"),
                            yield_data.get("yield_5y"),
                            yield_data.get("yield_10y"),
                            yield_data.get("yield_20y"),
                            yield_data.get("yield_30y"),
                            spread_10y_2y,
                            spread_10y_3m,
                            is_inverted,
                            record_hash,
                            record_date,
                        ))
                        self.metrics.records_updated += 1
                    else:
                        # Insert new record
                        cursor.execute("""
                            INSERT INTO treasury_yields
                                (date, yield_1m, yield_3m, yield_6m, yield_1y,
                                 yield_2y, yield_5y, yield_10y, yield_20y, yield_30y,
                                 spread_10y_2y, spread_10y_3m, is_inverted,
                                 source_hash, source_fetch_timestamp, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        """, (
                            record_date,
                            yield_data.get("yield_1m"),
                            yield_data.get("yield_3m"),
                            yield_data.get("yield_6m"),
                            yield_data.get("yield_1y"),
                            yield_data.get("yield_2y"),
                            yield_data.get("yield_5y"),
                            yield_data.get("yield_10y"),
                            yield_data.get("yield_20y"),
                            yield_data.get("yield_30y"),
                            spread_10y_2y,
                            spread_10y_3m,
                            is_inverted,
                            record_hash,
                        ))
                        self.metrics.records_inserted += 1

                    # Track high watermark
                    if self.metrics.high_watermark_date is None or record_date > self.metrics.high_watermark_date:
                        self.metrics.high_watermark_date = record_date

                except Exception as e:
                    self.logger.warning(f"Failed to process yield for {yield_data.get('date')}: {e}")
                    self.metrics.records_failed += 1

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Treasury yields: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except ImportError as e:
            self.logger.error(f"Treasury fetcher not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"Treasury data collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description="Collect Treasury yield curve data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    args = parser.parse_args()

    collector = TreasuryDataCollector(lookback_days=args.days)
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
