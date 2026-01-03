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

"""Collect Analyst Estimates and Price Targets.

Schedule: Daily at 7AM ET
Source: Finnhub API (free tier: 60 calls/minute)

Collects:
- EPS and Revenue estimates (quarterly/annual)
- Price targets (high, low, mean, median)
- Analyst recommendations (buy/hold/sell)
- Estimate revisions (up/down momentum)

Usage:
    python scripts/scheduled/collect_analyst_estimates.py
    python scripts/scheduled/collect_analyst_estimates.py --symbols AAPL,MSFT
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
    get_sp500_symbols,
    get_watermark,
    retry_with_backoff,
    update_watermark,
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


class AnalystEstimatesCollector(BaseCollector):
    """Collector for analyst estimates and price targets from Finnhub."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        include_recommendations: bool = True,
        include_price_targets: bool = True,
    ):
        super().__init__("collect_analyst_estimates")
        self.symbols = symbols
        self.include_recommendations = include_recommendations
        self.include_price_targets = include_price_targets
        self.api_key = FINNHUB_API_KEY
        self.rate_limiter = get_finnhub_rate_limiter()

        if not self.api_key:
            self.logger.warning(
                "FINNHUB_API_KEY not set. Set environment variable to enable collection."
            )

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
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
        """Collect analyst estimates with incremental fetching."""
        if not self.api_key:
            self.metrics.errors.append("FINNHUB_API_KEY not set")
            return self.metrics

        try:
            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_sp500_symbols()[:100]  # Limit for free tier

            self.logger.info(f"Collecting analyst estimates for {len(symbols)} symbols")

            conn = get_database_connection()
            cursor = conn.cursor()

            for symbol in symbols:
                try:
                    self.metrics.records_processed += 1

                    # 1. Collect EPS estimates
                    self._collect_eps_estimates(cursor, symbol)

                    # 2. Collect price targets
                    if self.include_price_targets:
                        self._collect_price_targets(cursor, symbol)

                    # 3. Collect recommendations
                    if self.include_recommendations:
                        self._collect_recommendations(cursor, symbol)

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
                f"Analyst estimates: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"Analyst estimates collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _collect_eps_estimates(self, cursor, symbol: str) -> None:
        """Collect EPS and revenue estimates for a symbol."""
        data = self._make_request("stock/eps-estimate", {"symbol": symbol})
        if not data or "data" not in data:
            return

        for estimate in data.get("data", []):
            period = estimate.get("period")
            if not period:
                continue

            record_hash = compute_record_hash(estimate)

            # Check existing
            cursor.execute("""
                SELECT source_hash FROM analyst_estimates
                WHERE symbol = %s AND period_type = 'quarterly' AND period_end_date = %s
            """, (symbol, period))
            existing = cursor.fetchone()

            if existing:
                if existing[0] == record_hash:
                    self.metrics.records_skipped += 1
                    continue
                # Update
                cursor.execute("""
                    UPDATE analyst_estimates SET
                        eps_estimate_avg = %s, eps_estimate_high = %s, eps_estimate_low = %s,
                        eps_estimate_count = %s, source_hash = %s,
                        source_fetch_timestamp = NOW(), updated_at = NOW()
                    WHERE symbol = %s AND period_type = 'quarterly' AND period_end_date = %s
                """, (
                    estimate.get("epsAvg"),
                    estimate.get("epsHigh"),
                    estimate.get("epsLow"),
                    estimate.get("numberAnalysts"),
                    record_hash,
                    symbol,
                    period,
                ))
                self.metrics.records_updated += 1
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO analyst_estimates
                        (symbol, period_type, period_end_date, eps_estimate_avg,
                         eps_estimate_high, eps_estimate_low, eps_estimate_count,
                         source_hash, source_fetch_timestamp)
                    VALUES (%s, 'quarterly', %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    symbol,
                    period,
                    estimate.get("epsAvg"),
                    estimate.get("epsHigh"),
                    estimate.get("epsLow"),
                    estimate.get("numberAnalysts"),
                    record_hash,
                ))
                self.metrics.records_inserted += 1

    def _collect_price_targets(self, cursor, symbol: str) -> None:
        """Collect price targets for a symbol."""
        data = self._make_request("stock/price-target", {"symbol": symbol})
        if not data:
            return

        record_hash = compute_record_hash(data)

        # Check existing
        cursor.execute(
            "SELECT source_hash FROM analyst_price_targets WHERE symbol = %s",
            (symbol,),
        )
        existing = cursor.fetchone()

        if existing:
            if existing[0] == record_hash:
                self.metrics.records_skipped += 1
                return
            # Update
            cursor.execute("""
                UPDATE analyst_price_targets SET
                    target_high = %s, target_low = %s, target_mean = %s,
                    target_median = %s, analyst_count = %s, last_updated = %s,
                    source_hash = %s, source_fetch_timestamp = NOW(), updated_at = NOW()
                WHERE symbol = %s
            """, (
                data.get("targetHigh"),
                data.get("targetLow"),
                data.get("targetMean"),
                data.get("targetMedian"),
                data.get("numberOfAnalysts"),
                data.get("lastUpdated"),
                record_hash,
                symbol,
            ))
            self.metrics.records_updated += 1
        else:
            # Insert
            cursor.execute("""
                INSERT INTO analyst_price_targets
                    (symbol, target_high, target_low, target_mean, target_median,
                     analyst_count, last_updated, source_hash, source_fetch_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                symbol,
                data.get("targetHigh"),
                data.get("targetLow"),
                data.get("targetMean"),
                data.get("targetMedian"),
                data.get("numberOfAnalysts"),
                data.get("lastUpdated"),
                record_hash,
            ))
            self.metrics.records_inserted += 1

    def _collect_recommendations(self, cursor, symbol: str) -> None:
        """Collect analyst recommendations for a symbol."""
        data = self._make_request("stock/recommendation", {"symbol": symbol})
        if not data:
            return

        for rec in data:
            period = rec.get("period")
            if not period:
                continue

            record_hash = compute_record_hash(rec)

            # Calculate consensus score (1=strong sell, 5=strong buy)
            total = (
                rec.get("strongBuy", 0) + rec.get("buy", 0) + rec.get("hold", 0) +
                rec.get("sell", 0) + rec.get("strongSell", 0)
            )
            if total > 0:
                consensus = (
                    5 * rec.get("strongBuy", 0) + 4 * rec.get("buy", 0) +
                    3 * rec.get("hold", 0) + 2 * rec.get("sell", 0) +
                    1 * rec.get("strongSell", 0)
                ) / total
            else:
                consensus = None

            # Check existing
            cursor.execute("""
                SELECT source_hash FROM analyst_recommendations
                WHERE symbol = %s AND period_date = %s
            """, (symbol, period))
            existing = cursor.fetchone()

            if existing:
                if existing[0] == record_hash:
                    self.metrics.records_skipped += 1
                    continue
                # Update
                cursor.execute("""
                    UPDATE analyst_recommendations SET
                        strong_buy = %s, buy = %s, hold = %s, sell = %s, strong_sell = %s,
                        consensus_score = %s, total_analysts = %s,
                        source_hash = %s, source_fetch_timestamp = NOW(), updated_at = NOW()
                    WHERE symbol = %s AND period_date = %s
                """, (
                    rec.get("strongBuy", 0),
                    rec.get("buy", 0),
                    rec.get("hold", 0),
                    rec.get("sell", 0),
                    rec.get("strongSell", 0),
                    consensus,
                    total,
                    record_hash,
                    symbol,
                    period,
                ))
                self.metrics.records_updated += 1
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO analyst_recommendations
                        (symbol, period_date, strong_buy, buy, hold, sell, strong_sell,
                         consensus_score, total_analysts, source_hash, source_fetch_timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    symbol,
                    period,
                    rec.get("strongBuy", 0),
                    rec.get("buy", 0),
                    rec.get("hold", 0),
                    rec.get("sell", 0),
                    rec.get("strongSell", 0),
                    consensus,
                    total,
                    record_hash,
                ))
                self.metrics.records_inserted += 1


def main():
    parser = argparse.ArgumentParser(
        description="Collect analyst estimates and price targets"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: S&P 500)"
    )
    parser.add_argument(
        "--no-recommendations",
        action="store_true",
        help="Skip analyst recommendations"
    )
    parser.add_argument(
        "--no-price-targets",
        action="store_true",
        help="Skip price targets"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    collector = AnalystEstimatesCollector(
        symbols=symbols,
        include_recommendations=not args.no_recommendations,
        include_price_targets=not args.no_price_targets,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
