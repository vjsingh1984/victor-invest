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

"""Collect SEC Form 13F Institutional Holdings.

Schedule: Daily at 7AM ET
Source: SEC EDGAR Form 13F

Collects:
- New 13F filings from major institutions
- Holdings positions and values
- Quarter-over-quarter changes
- Updates institutional ownership metrics

Note: 13F filings are quarterly, but we check daily for new filings
during the 45-day filing window after quarter end.

Usage:
    python scripts/scheduled/collect_13f_filings.py
    python scripts/scheduled/collect_13f_filings.py --days 3
    python scripts/scheduled/collect_13f_filings.py --top-institutions 50
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
    retry_with_backoff,
)

# Top institutional investors to track (by 13F CIK)
TOP_INSTITUTIONS = {
    "0001067983": "BERKSHIRE HATHAWAY INC",
    "0001350694": "BLACKROCK INC.",
    "0001166559": "VANGUARD GROUP INC",
    "0001364742": "STATE STREET CORP",
    "0001037389": "FMR LLC",  # Fidelity
    "0001536411": "CITADEL ADVISORS LLC",
    "0001649339": "BRIDGEWATER ASSOCIATES, LP",
    "0001336528": "TWO SIGMA INVESTMENTS, LP",
    "0001061768": "RENAISSANCE TECHNOLOGIES LLC",
    "0001159159": "T. ROWE PRICE ASSOCIATES INC",
    "0000810265": "CAPITAL RESEARCH GLOBAL INVESTORS",
    "0001569205": "AQR CAPITAL MANAGEMENT LLC",
    "0000913760": "WELLINGTON MANAGEMENT GROUP LLP",
    "0000070858": "BANK OF AMERICA CORP",
    "0001078163": "GOLDMAN SACHS GROUP INC",
    "0001167483": "JPMORGAN CHASE & CO",
    "0000093751": "MORGAN STANLEY",
    "0001009207": "GEODE CAPITAL MANAGEMENT, LLC",
    "0001454502": "INVESCO LTD.",
    "0000856927": "CAPITAL WORLD INVESTORS",
}


class Form13FCollector(BaseCollector):
    """Collector for SEC Form 13F institutional holdings."""

    def __init__(
        self,
        lookback_days: int = 1,
        top_institutions: int = 100,
    ):
        super().__init__("collect_13f_filings")
        self.lookback_days = lookback_days
        self.top_institutions = top_institutions

    def collect(self) -> CollectionMetrics:
        """Collect 13F filings from SEC EDGAR."""
        try:
            # Import the 13F fetcher
            from investigator.infrastructure.external.sec.institutional_holdings import (
                get_institutional_fetcher,
            )

            fetcher = get_institutional_fetcher()

            self.logger.info(
                f"Checking for new 13F filings (lookback: {self.lookback_days} days, "
                f"top institutions: {self.top_institutions})"
            )

            conn = get_database_connection()
            cursor = conn.cursor()

            since_date = datetime.now() - timedelta(days=self.lookback_days)

            # Check each top institution for new filings
            institutions = list(TOP_INSTITUTIONS.items())[:self.top_institutions]

            for cik, name in institutions:
                try:
                    # Check for recent filings
                    filings = asyncio.run(
                        fetcher.get_recent_filings(
                            cik=cik,
                            since_date=since_date,
                        )
                    )

                    if not filings:
                        continue

                    self.logger.info(f"Found {len(filings)} new filings for {name}")

                    for filing in filings:
                        self.metrics.records_processed += 1

                        # Get filing details and holdings
                        holdings = asyncio.run(
                            fetcher.get_filing_holdings(
                                accession_number=filing.get("accession_number"),
                            )
                        )

                        if not holdings:
                            continue

                        # Store institution if not exists
                        cursor.execute("""
                            INSERT INTO institutions (cik, name)
                            VALUES (%s, %s)
                            ON CONFLICT (cik) DO UPDATE SET
                                name = EXCLUDED.name,
                                updated_at = NOW()
                            RETURNING id
                        """, (cik, name))
                        institution_id = cursor.fetchone()[0]

                        # Store filing
                        cursor.execute("""
                            INSERT INTO form13f_filings
                                (institution_id, accession_number, report_quarter,
                                 filing_date, total_value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (accession_number) DO UPDATE SET
                                updated_at = NOW()
                            RETURNING id
                        """, (
                            institution_id,
                            filing.get("accession_number"),
                            filing.get("report_quarter"),
                            filing.get("filing_date"),
                            filing.get("total_value"),
                        ))
                        filing_id = cursor.fetchone()[0]

                        # Store holdings
                        for holding in holdings:
                            cursor.execute("""
                                INSERT INTO form13f_holdings
                                    (filing_id, symbol, cusip, shares,
                                     value_thousands, put_call, investment_discretion)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (filing_id, cusip) DO UPDATE SET
                                    shares = EXCLUDED.shares,
                                    value_thousands = EXCLUDED.value_thousands,
                                    updated_at = NOW()
                            """, (
                                filing_id,
                                holding.get("symbol"),
                                holding.get("cusip"),
                                holding.get("shares"),
                                holding.get("value_thousands"),
                                holding.get("put_call"),
                                holding.get("investment_discretion"),
                            ))

                            self.metrics.records_inserted += 1

                        # Update institutional ownership aggregates
                        self._update_ownership_aggregates(cursor, filing)

                except Exception as e:
                    self.logger.warning(f"Failed to process {name}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{name}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Collected {self.metrics.records_inserted} holdings from 13F filings"
            )

        except ImportError as e:
            self.logger.error(f"13F fetcher not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"13F collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _update_ownership_aggregates(self, cursor, filing) -> None:
        """Update aggregate institutional ownership for affected symbols."""
        try:
            report_quarter = filing.get("report_quarter")
            if not report_quarter:
                return

            # Aggregate ownership by symbol for this quarter
            cursor.execute("""
                WITH holdings_agg AS (
                    SELECT
                        h.symbol,
                        COUNT(DISTINCT f.institution_id) as num_institutions,
                        SUM(h.shares) as total_shares,
                        SUM(h.value_thousands * 1000) as total_value
                    FROM form13f_holdings h
                    JOIN form13f_filings f ON h.filing_id = f.id
                    WHERE f.report_quarter = %s
                      AND h.symbol IS NOT NULL
                    GROUP BY h.symbol
                )
                INSERT INTO institutional_ownership
                    (symbol, report_quarter, num_institutions,
                     total_institutional_shares, total_institutional_value)
                SELECT symbol, %s, num_institutions, total_shares, total_value
                FROM holdings_agg
                ON CONFLICT (symbol, report_quarter) DO UPDATE SET
                    num_institutions = EXCLUDED.num_institutions,
                    total_institutional_shares = EXCLUDED.total_institutional_shares,
                    total_institutional_value = EXCLUDED.total_institutional_value,
                    updated_at = NOW()
            """, (report_quarter, report_quarter))

        except Exception as e:
            self.logger.debug(f"Could not update ownership aggregates: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect SEC Form 13F institutional holdings"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Days to look back for new filings (default: 1)"
    )
    parser.add_argument(
        "--top-institutions",
        type=int,
        default=100,
        help="Number of top institutions to track (default: 100)"
    )
    args = parser.parse_args()

    collector = Form13FCollector(
        lookback_days=args.days,
        top_institutions=args.top_institutions,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
