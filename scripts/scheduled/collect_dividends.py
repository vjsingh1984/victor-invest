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

"""Collect Dividend History Data.

Schedule: Daily at 6AM ET
Source: Finnhub API (free tier)

Collects:
- Dividend history (ex-date, payment date, amount)
- Dividend yield calculations
- Dividend growth metrics
- Consecutive dividend years (streak)

Usage:
    python scripts/scheduled/collect_dividends.py
    python scripts/scheduled/collect_dividends.py --symbols AAPL,MSFT
    python scripts/scheduled/collect_dividends.py --years 5
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
    get_finnhub_rate_limiter,
    get_last_date,
    get_sp500_symbols,
    retry_with_backoff,
)

# Finnhub API configuration
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def _get_finnhub_api_key() -> str:
    """Get Finnhub API key from victor keyring or environment.

    Resolution order:
    1. Environment variable FINNHUB_API_KEY (for CI/automation)
    2. Victor keyring (secure storage)
    3. Empty string (will fail gracefully)
    """
    # Priority 1: Environment variable
    env_key = os.environ.get("FINNHUB_API_KEY", "")
    if env_key:
        return env_key

    # Priority 2: Try victor keyring
    try:
        from victor.config.api_keys import get_service_key
        key = get_service_key("finnhub")
        if key:
            return key
    except ImportError:
        pass  # victor framework not available

    return ""


FINNHUB_API_KEY = _get_finnhub_api_key()


class DividendCollector(BaseCollector):
    """Collector for dividend history from Finnhub."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookback_years: int = 5,
    ):
        super().__init__("collect_dividends")
        self.symbols = symbols
        self.lookback_years = lookback_years
        self.api_key = FINNHUB_API_KEY
        self.rate_limiter = get_finnhub_rate_limiter()

        if not self.api_key:
            self.logger.warning(
                "FINNHUB_API_KEY not configured. "
                "Set via: victor keys --set-service finnhub --keyring"
            )

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Make a rate-limited request to Finnhub API with Fibonacci backoff."""
        if not self.api_key:
            return None

        params["token"] = self.api_key
        url = f"{FINNHUB_BASE_URL}/{endpoint}"

        try:
            # Use rate limiter with automatic retry on 429
            with self.rate_limiter:
                response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                self.rate_limiter.record_success()
                return response.json()
            elif response.status_code == 429:
                delay = self.rate_limiter.record_rate_limit()
                time.sleep(delay)
                return self._make_request(endpoint, params)
            else:
                self.logger.debug(f"API error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            self.logger.warning(f"Request failed: {e}")
            return None

    def collect(self) -> CollectionMetrics:
        """Collect dividend data with incremental fetching."""
        if not self.api_key:
            self.metrics.errors.append(
                "FINNHUB_API_KEY not configured. "
                "Run: victor keys --set-service finnhub --keyring"
            )
            return self.metrics

        try:
            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_sp500_symbols()[:100]  # Limit for free tier

            self.logger.info(f"Collecting dividend data for {len(symbols)} symbols")

            conn = get_database_connection()
            cursor = conn.cursor()

            end_date = datetime.now().date()
            default_start = end_date - timedelta(days=365 * self.lookback_years)

            for symbol in symbols:
                try:
                    self.metrics.records_processed += 1

                    # Get last dividend date for incremental fetch
                    last_date = get_last_date(
                        cursor, "dividend_history", "ex_dividend_date",
                        "symbol", symbol
                    )

                    if last_date:
                        start_date = last_date + timedelta(days=1)
                    else:
                        start_date = default_start

                    # Fetch dividend history
                    data = self._make_request("stock/dividend", {
                        "symbol": symbol,
                        "from": start_date.isoformat(),
                        "to": end_date.isoformat(),
                    })

                    if not data:
                        continue

                    for div in data:
                        ex_date = div.get("exDate")
                        if not ex_date:
                            continue

                        record_hash = compute_record_hash(div)

                        # Check existing
                        cursor.execute("""
                            SELECT source_hash FROM dividend_history
                            WHERE symbol = %s AND ex_dividend_date = %s
                        """, (symbol, ex_date))
                        existing = cursor.fetchone()

                        if existing:
                            if existing[0] == record_hash:
                                self.metrics.records_skipped += 1
                                continue
                            # Update
                            cursor.execute("""
                                UPDATE dividend_history SET
                                    payment_date = %s, record_date = %s,
                                    declaration_date = %s, dividend_amount = %s,
                                    adjusted_amount = %s, source_hash = %s,
                                    source_fetch_timestamp = NOW(), updated_at = NOW()
                                WHERE symbol = %s AND ex_dividend_date = %s
                            """, (
                                div.get("payDate"),
                                div.get("recordDate"),
                                div.get("declarationDate"),
                                div.get("amount"),
                                div.get("adjustedAmount"),
                                record_hash,
                                symbol,
                                ex_date,
                            ))
                            self.metrics.records_updated += 1
                        else:
                            # Insert
                            cursor.execute("""
                                INSERT INTO dividend_history
                                    (symbol, ex_dividend_date, payment_date, record_date,
                                     declaration_date, dividend_amount, adjusted_amount,
                                     source_hash, source_fetch_timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            """, (
                                symbol,
                                ex_date,
                                div.get("payDate"),
                                div.get("recordDate"),
                                div.get("declarationDate"),
                                div.get("amount"),
                                div.get("adjustedAmount"),
                                record_hash,
                            ))
                            self.metrics.records_inserted += 1

                    # Update shareholder yield summary
                    self._update_shareholder_yield(cursor, symbol)

                    # Update watermark
                    cursor.execute("""
                        INSERT INTO dividend_watermarks (symbol, last_ex_date)
                        VALUES (%s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                            last_ex_date = GREATEST(
                                dividend_watermarks.last_ex_date,
                                EXCLUDED.last_ex_date
                            ),
                            last_fetch_timestamp = NOW()
                    """, (symbol, end_date))

                    # Commit periodically
                    if self.metrics.records_processed % 10 == 0:
                        conn.commit()

                except Exception as e:
                    self.logger.warning(f"Failed to process {symbol}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{symbol}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Dividends: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"Dividend collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _update_shareholder_yield(self, cursor, symbol: str) -> None:
        """Calculate and update shareholder yield summary."""
        try:
            # Calculate TTM dividend
            cursor.execute("""
                SELECT SUM(dividend_amount) as ttm_div
                FROM dividend_history
                WHERE symbol = %s
                  AND ex_dividend_date >= CURRENT_DATE - INTERVAL '1 year'
            """, (symbol,))
            row = cursor.fetchone()
            ttm_div = row[0] if row and row[0] else 0

            # Count consecutive years with dividends
            cursor.execute("""
                WITH yearly_divs AS (
                    SELECT EXTRACT(YEAR FROM ex_dividend_date) as year,
                           SUM(dividend_amount) as total
                    FROM dividend_history
                    WHERE symbol = %s
                    GROUP BY EXTRACT(YEAR FROM ex_dividend_date)
                    HAVING SUM(dividend_amount) > 0
                    ORDER BY year DESC
                )
                SELECT COUNT(*) as consecutive
                FROM (
                    SELECT year, year - ROW_NUMBER() OVER (ORDER BY year DESC) as grp
                    FROM yearly_divs
                ) sub
                WHERE grp = (SELECT MIN(year - ROW_NUMBER() OVER (ORDER BY year DESC))
                             FROM yearly_divs LIMIT 1)
            """, (symbol,))
            streak_row = cursor.fetchone()
            streak = streak_row[0] if streak_row else 0

            # Calculate dividend growth (1Y)
            cursor.execute("""
                WITH current_year AS (
                    SELECT SUM(dividend_amount) as total
                    FROM dividend_history
                    WHERE symbol = %s AND ex_dividend_date >= CURRENT_DATE - INTERVAL '1 year'
                ),
                prior_year AS (
                    SELECT SUM(dividend_amount) as total
                    FROM dividend_history
                    WHERE symbol = %s
                      AND ex_dividend_date >= CURRENT_DATE - INTERVAL '2 years'
                      AND ex_dividend_date < CURRENT_DATE - INTERVAL '1 year'
                )
                SELECT
                    CASE WHEN p.total > 0 THEN (c.total - p.total) / p.total ELSE NULL END
                FROM current_year c, prior_year p
            """, (symbol, symbol))
            growth_row = cursor.fetchone()
            growth_1y = growth_row[0] if growth_row else None

            record_hash = compute_record_hash({
                "symbol": symbol,
                "ttm_div": str(ttm_div),
                "streak": streak,
                "growth_1y": str(growth_1y),
            })

            # Upsert
            cursor.execute("""
                INSERT INTO shareholder_yield
                    (symbol, calculation_date, dividend_yield_ttm,
                     dividend_growth_1y, consecutive_dividend_years,
                     dividend_aristocrat, dividend_king, source_hash, source_fetch_timestamp)
                VALUES (%s, CURRENT_DATE, NULL, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    calculation_date = CURRENT_DATE,
                    dividend_growth_1y = EXCLUDED.dividend_growth_1y,
                    consecutive_dividend_years = EXCLUDED.consecutive_dividend_years,
                    dividend_aristocrat = EXCLUDED.dividend_aristocrat,
                    dividend_king = EXCLUDED.dividend_king,
                    source_hash = EXCLUDED.source_hash,
                    source_fetch_timestamp = NOW(),
                    updated_at = NOW()
            """, (
                symbol,
                growth_1y,
                streak,
                streak >= 25,  # Dividend Aristocrat
                streak >= 50,  # Dividend King
                record_hash,
            ))

        except Exception as e:
            self.logger.debug(f"Could not update shareholder yield for {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect dividend history data"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: S&P 500)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years to fetch (default: 5)"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    collector = DividendCollector(
        symbols=symbols,
        lookback_years=args.years,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
