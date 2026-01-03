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

"""Collect SEC Form 4 Insider Transactions.

Schedule: Every 4 hours during market hours (8AM, 12PM, 4PM, 8PM ET)
Source: SEC EDGAR Form 3/4/5

Collects:
- Insider buy/sell transactions
- Transaction values and share counts
- Insider roles and relationships
- Updates sentiment scores

Usage:
    python scripts/scheduled/collect_insider_transactions.py
    python scripts/scheduled/collect_insider_transactions.py --symbols AAPL,MSFT
    python scripts/scheduled/collect_insider_transactions.py --hours 6
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

from psycopg2.extras import Json

from investigator.config.lookback_periods import INSIDER_PERIODS
from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
    get_sp500_symbols,
    get_watermark,
    retry_with_backoff,
    update_watermark,
)


class InsiderTransactionCollector(BaseCollector):
    """Collector for SEC Form 4 insider transactions."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookback_hours: int = 6,
    ):
        super().__init__("collect_insider_transactions")
        self.symbols = symbols
        self.lookback_hours = lookback_hours

    def collect(self) -> CollectionMetrics:
        """Collect insider transactions from SEC EDGAR with incremental fetching."""
        try:
            # Import the insider trading fetcher
            from investigator.infrastructure.external.sec.insider_transactions import (
                get_insider_fetcher,
            )

            fetcher = get_insider_fetcher()

            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_sp500_symbols()

            self.logger.info(
                f"Fetching insider transactions for {len(symbols)} symbols "
                f"(lookback: {self.lookback_hours} hours)"
            )

            conn = get_database_connection()
            cursor = conn.cursor()

            lookback_days = max(1, self.lookback_hours // 24)

            for symbol in symbols:
                try:
                    # Get watermark for this symbol to enable incremental fetching
                    watermark = get_watermark(
                        cursor, "form4_fetch_watermarks", "symbol", symbol
                    )
                    last_accession = watermark.get("last_accession_number") if watermark else None
                    last_filing_date = watermark.get("last_filing_date") if watermark else None

                    if last_accession:
                        self.logger.debug(
                            f"{symbol}: Incremental fetch after accession {last_accession}"
                        )

                    # Fetch recent Form 4 filings for symbol
                    filings = asyncio.run(
                        fetcher.fetch_recent_filings(
                            symbol=symbol,
                            days=lookback_days,
                        )
                    )

                    if not filings:
                        continue

                    self.metrics.records_processed += len(filings)

                    # Track newest accession for watermark update
                    newest_accession = last_accession
                    newest_filing_date = last_filing_date

                    for filing in filings:
                        accession = filing.accession_number

                        # Skip if we've already processed this accession
                        if last_accession and accession <= last_accession:
                            self.metrics.records_skipped += 1
                            continue

                        # Track newest for watermark
                        if newest_accession is None or accession > newest_accession:
                            newest_accession = accession
                            newest_filing_date = filing.filing_date

                        # Each filing may have multiple transactions
                        for txn in filing.transactions:
                            # Get insider info from reporting owner
                            owner = filing.reporting_owner
                            owner_name = owner.name if owner else None
                            owner_title = owner.title if owner else None
                            is_director = owner.is_director if owner else False
                            is_officer = owner.is_officer if owner else False

                            # Get transaction type and code
                            txn_code = txn.transaction_code
                            txn_type = txn.transaction_type.value if txn.transaction_type else txn_code

                            # Determine significance
                            is_significant = filing.is_significant
                            significance_reasons = []
                            if owner and owner.is_key_insider:
                                significance_reasons.append("key_insider")
                            if txn.total_value and txn.total_value > 500000:
                                significance_reasons.append("large_transaction")

                            # Store full filing data as JSON
                            filing_data = filing.to_dict()

                            # Compute hash for change detection
                            record_hash = compute_record_hash({
                                "accession": accession,
                                "owner": owner_name,
                                "txn_code": txn_code,
                                "shares": str(txn.shares),
                                "price": str(txn.price_per_share),
                                "value": str(txn.total_value),
                            })

                            # Check if record already exists with hash comparison
                            cursor.execute("""
                                SELECT id, source_hash FROM form4_filings
                                WHERE accession_number = %s AND owner_name = %s
                            """, (accession, owner_name))

                            existing = cursor.fetchone()
                            if existing:
                                existing_id, existing_hash = existing
                                if existing_hash == record_hash:
                                    # No change, skip
                                    self.metrics.records_skipped += 1
                                    continue
                                # Update existing record
                                cursor.execute("""
                                    UPDATE form4_filings SET
                                        transaction_type = %s, transaction_code = %s,
                                        shares = %s, price_per_share = %s, total_value = %s,
                                        is_significant = %s, significance_reasons = %s,
                                        filing_data = %s, source_hash = %s,
                                        source_fetch_timestamp = NOW(), updated_at = NOW()
                                    WHERE id = %s
                                """, (
                                    txn_type, txn_code, txn.shares, txn.price_per_share,
                                    txn.total_value, is_significant, significance_reasons,
                                    Json(filing_data), record_hash, existing_id,
                                ))
                                self.metrics.records_updated += 1
                            else:
                                # Insert new record
                                cursor.execute("""
                                    INSERT INTO form4_filings
                                        (symbol, cik, accession_number, filing_date,
                                         owner_name, owner_title, is_director, is_officer,
                                         transaction_type, transaction_code,
                                         shares, price_per_share, total_value,
                                         is_significant, significance_reasons, filing_data,
                                         source_hash, source_fetch_timestamp)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                                """, (
                                    symbol,
                                    filing.issuer_cik,
                                    accession,
                                    filing.filing_date,
                                    owner_name,
                                    owner_title,
                                    is_director,
                                    is_officer,
                                    txn_type,
                                    txn_code,
                                    txn.shares,
                                    txn.price_per_share,
                                    txn.total_value,
                                    is_significant,
                                    significance_reasons,
                                    Json(filing_data),
                                    record_hash,
                                ))
                                self.metrics.records_inserted += 1

                    # Update watermark for this symbol
                    if newest_accession and (last_accession is None or newest_accession > last_accession):
                        update_watermark(
                            cursor,
                            "form4_fetch_watermarks",
                            "symbol",
                            symbol,
                            {
                                "last_accession_number": newest_accession,
                                "last_filing_date": newest_filing_date,
                            },
                        )
                        self.metrics.high_watermark_value = newest_accession

                    # Update sentiment score for symbol
                    self._update_sentiment_score(cursor, symbol)

                except Exception as e:
                    self.logger.warning(f"Failed to process {symbol}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{symbol}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Insider transactions: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except ImportError as e:
            self.logger.error(f"Insider fetcher not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"Insider transaction collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _update_sentiment_score(self, cursor, symbol: str) -> None:
        """Calculate and update insider sentiment score for a symbol."""
        try:
            # Calculate sentiment using standard period from shared config
            sentiment_days = INSIDER_PERIODS.standard_days
            cluster_days = INSIDER_PERIODS.recent_days
            cursor.execute(f"""
                WITH recent_txns AS (
                    SELECT
                        transaction_type,
                        total_value,
                        shares
                    FROM form4_filings
                    WHERE symbol = %s
                      AND filing_date >= CURRENT_DATE - INTERVAL '{sentiment_days} days'
                )
                SELECT
                    COUNT(*) FILTER (WHERE transaction_type IN ('P', 'A')) as buy_count,
                    COUNT(*) FILTER (WHERE transaction_type IN ('S', 'D')) as sell_count,
                    COALESCE(SUM(total_value) FILTER (WHERE transaction_type IN ('P', 'A')), 0) as buy_value,
                    COALESCE(SUM(total_value) FILTER (WHERE transaction_type IN ('S', 'D')), 0) as sell_value
                FROM recent_txns
            """, (symbol,))

            row = cursor.fetchone()
            if row:
                buy_count, sell_count, buy_value, sell_value = row

                # Calculate sentiment score (-1 to +1)
                total_count = buy_count + sell_count
                total_value = buy_value + sell_value

                if total_count > 0:
                    count_ratio = (buy_count - sell_count) / total_count
                else:
                    count_ratio = 0

                if total_value > 0:
                    value_ratio = (buy_value - sell_value) / total_value
                else:
                    value_ratio = 0

                # Weighted average (value matters more)
                sentiment_score = 0.3 * count_ratio + 0.7 * value_ratio

                # Detect cluster (3+ insiders buying/selling in cluster period)
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT owner_name)
                    FROM form4_filings
                    WHERE symbol = %s
                      AND filing_date >= CURRENT_DATE - INTERVAL '{cluster_days} days'
                      AND transaction_type IN ('P', 'A')
                """, (symbol,))
                buy_insiders = cursor.fetchone()[0] or 0

                cursor.execute(f"""
                    SELECT COUNT(DISTINCT owner_name)
                    FROM form4_filings
                    WHERE symbol = %s
                      AND filing_date >= CURRENT_DATE - INTERVAL '{cluster_days} days'
                      AND transaction_type IN ('S', 'D')
                """, (symbol,))
                sell_insiders = cursor.fetchone()[0] or 0

                cluster_detected = buy_insiders >= 3 or sell_insiders >= 3

                # Store sentiment
                cursor.execute("""
                    INSERT INTO insider_sentiment
                        (symbol, calculation_date, period_days,
                         buy_count, sell_count, buy_value, sell_value,
                         sentiment_score, cluster_detected)
                    VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, calculation_date, period_days)
                    DO UPDATE SET
                        buy_count = EXCLUDED.buy_count,
                        sell_count = EXCLUDED.sell_count,
                        buy_value = EXCLUDED.buy_value,
                        sell_value = EXCLUDED.sell_value,
                        sentiment_score = EXCLUDED.sentiment_score,
                        cluster_detected = EXCLUDED.cluster_detected,
                        updated_at = NOW()
                """, (
                    symbol, sentiment_days, buy_count, sell_count, buy_value, sell_value,
                    sentiment_score, cluster_detected,
                ))

        except Exception as e:
            self.logger.debug(f"Could not update sentiment for {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect SEC Form 4 insider transactions"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: S&P 500)"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=INSIDER_PERIODS.collector_hours,
        help=f"Hours to look back (default: {INSIDER_PERIODS.collector_hours})"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    collector = InsiderTransactionCollector(
        symbols=symbols,
        lookback_hours=args.hours,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
