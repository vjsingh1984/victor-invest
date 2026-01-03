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

"""Collect News Sentiment Data.

Schedule: Daily at 8AM ET
Source: Finnhub News Sentiment API (free tier: 60 calls/minute)

Collects:
- Daily sentiment scores by symbol
- Article counts (positive/negative/neutral)
- Buzz scores (relative news volume)
- Sector average sentiment for comparison

Usage:
    python scripts/scheduled/collect_news_sentiment.py
    python scripts/scheduled/collect_news_sentiment.py --symbols AAPL,MSFT
    python scripts/scheduled/collect_news_sentiment.py --days 7
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


class NewsSentimentCollector(BaseCollector):
    """Collector for news sentiment from Finnhub."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 7,
    ):
        super().__init__("collect_news_sentiment")
        self.symbols = symbols
        self.lookback_days = lookback_days
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
        """Collect news sentiment with incremental fetching."""
        if not self.api_key:
            self.metrics.errors.append("FINNHUB_API_KEY not set")
            return self.metrics

        try:
            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_sp500_symbols()[:100]  # Limit for free tier

            self.logger.info(f"Collecting news sentiment for {len(symbols)} symbols")

            conn = get_database_connection()
            cursor = conn.cursor()

            end_date = datetime.now().date()

            for symbol in symbols:
                try:
                    self.metrics.records_processed += 1

                    # Get last sentiment date for incremental fetch
                    last_date = get_last_date(
                        cursor, "news_sentiment", "sentiment_date",
                        "symbol", symbol
                    )

                    if last_date:
                        start_date = last_date + timedelta(days=1)
                    else:
                        start_date = end_date - timedelta(days=self.lookback_days)

                    if start_date > end_date:
                        self.metrics.records_skipped += 1
                        continue

                    # Fetch news sentiment
                    data = self._make_request("news-sentiment", {"symbol": symbol})
                    if not data:
                        continue

                    # Process buzz and sentiment
                    buzz = data.get("buzz", {})
                    sentiment = data.get("sentiment", {})
                    company_news_score = data.get("companyNewsScore", 0)

                    # Calculate aggregate metrics
                    articles = buzz.get("articlesInLastWeek", 0)
                    buzz_score = buzz.get("buzz", 0)

                    # Sentiment breakdown isn't in this endpoint, use score
                    sentiment_score = sentiment.get("bearishPercent", 0) * -1 + sentiment.get("bullishPercent", 0)

                    record_hash = compute_record_hash({
                        "symbol": symbol,
                        "date": str(end_date),
                        "buzz": buzz,
                        "sentiment": sentiment,
                    })

                    # Check existing
                    cursor.execute("""
                        SELECT source_hash FROM news_sentiment
                        WHERE symbol = %s AND sentiment_date = %s
                    """, (symbol, end_date))
                    existing = cursor.fetchone()

                    if existing:
                        if existing[0] == record_hash:
                            self.metrics.records_skipped += 1
                            continue
                        # Update
                        cursor.execute("""
                            UPDATE news_sentiment SET
                                articles_in_period = %s, sentiment_score = %s,
                                buzz_score = %s, company_news_score = %s,
                                sector_avg_sentiment = %s, sector_avg_news_volume = %s,
                                source_hash = %s, source_fetch_timestamp = NOW(),
                                updated_at = NOW()
                            WHERE symbol = %s AND sentiment_date = %s
                        """, (
                            articles,
                            sentiment_score,
                            buzz_score,
                            company_news_score,
                            data.get("sectorAverageBullishPercent", 0) - data.get("sectorAverageBearishPercent", 0),
                            data.get("sectorAverageNewsScore", 0),
                            record_hash,
                            symbol,
                            end_date,
                        ))
                        self.metrics.records_updated += 1
                    else:
                        # Insert
                        cursor.execute("""
                            INSERT INTO news_sentiment
                                (symbol, sentiment_date, articles_in_period, sentiment_score,
                                 buzz_score, company_news_score, sector_avg_sentiment,
                                 sector_avg_news_volume, source_hash, source_fetch_timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            symbol,
                            end_date,
                            articles,
                            sentiment_score,
                            buzz_score,
                            company_news_score,
                            data.get("sectorAverageBullishPercent", 0) - data.get("sectorAverageBearishPercent", 0),
                            data.get("sectorAverageNewsScore", 0),
                            record_hash,
                        ))
                        self.metrics.records_inserted += 1

                    # Update watermark
                    cursor.execute("""
                        INSERT INTO news_sentiment_watermarks (symbol, last_sentiment_date)
                        VALUES (%s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                            last_sentiment_date = GREATEST(
                                news_sentiment_watermarks.last_sentiment_date,
                                EXCLUDED.last_sentiment_date
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
                f"News sentiment: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"News sentiment collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description="Collect news sentiment data"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: S&P 500)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    collector = NewsSentimentCollector(
        symbols=symbols,
        lookback_days=args.days,
    )
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
