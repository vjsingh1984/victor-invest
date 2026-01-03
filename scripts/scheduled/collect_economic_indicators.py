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

"""Collect Economic Indicators from All Sources.

This is the main entry point for collecting economic data from:
- All 12 Federal Reserve District Banks
- CBOE volatility data
- Treasury yield curves
- ISM PMI data (via FRED)

Uses the robust DataFetcher with automatic fallbacks:
1. Primary URLs from Fed websites
2. Backup URLs
3. FRED API as final fallback

The data source registry (config/data_sources_registry.yaml) can be updated
without code changes when URLs change.

Schedule: Daily at 6AM ET
Sources: See data_sources_registry.yaml

Usage:
    python scripts/scheduled/collect_economic_indicators.py
    python scripts/scheduled/collect_economic_indicators.py --source atlanta_fed
    python scripts/scheduled/collect_economic_indicators.py --indicator gdpnow
    python scripts/scheduled/collect_economic_indicators.py --health-check
    python scripts/scheduled/collect_economic_indicators.py --dry-run
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
)


class EconomicIndicatorsCollector(BaseCollector):
    """Collector for all economic indicators using robust data fetcher."""

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
        dry_run: bool = False,
    ):
        super().__init__("collect_economic_indicators")
        self.source_filter = sources
        self.indicator_filter = indicators
        self.dry_run = dry_run
        self._results: Dict[str, Any] = {}
        self._fetcher = None

    async def _get_fetcher(self):
        """Get the data fetcher (lazy init)."""
        if self._fetcher is None:
            from investigator.infrastructure.external.data_fetcher import get_data_fetcher
            self._fetcher = get_data_fetcher()
        return self._fetcher

    def _should_fetch(self, source_id: str, indicator_id: str) -> bool:
        """Check if we should fetch this source/indicator."""
        if self.source_filter and source_id not in self.source_filter:
            return False
        if self.indicator_filter and indicator_id not in self.indicator_filter:
            return False
        return True

    async def _collect_all_async(self) -> Dict[str, Any]:
        """Collect all configured indicators."""
        fetcher = await self._get_fetcher()
        registry = fetcher.registry

        all_sources = registry.list_all_sources()
        results = {}

        # Group by source for logging
        sources_to_fetch = {}
        for source_id, indicator_id in all_sources:
            if self._should_fetch(source_id, indicator_id):
                if source_id not in sources_to_fetch:
                    sources_to_fetch[source_id] = []
                sources_to_fetch[source_id].append(indicator_id)

        # Fetch all indicators
        for source_id, indicator_ids in sources_to_fetch.items():
            results[source_id] = {}

            for indicator_id in indicator_ids:
                self.metrics.records_processed += 1

                try:
                    result = await fetcher.fetch(source_id, indicator_id)

                    if result.success:
                        results[source_id][indicator_id] = {
                            "data": result.data,
                            "source": result.source,
                            "url": result.url_used,
                            "fetch_time": result.fetch_time.isoformat(),
                        }
                        self.logger.debug(
                            f"Fetched {source_id}/{indicator_id} from {result.source}"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to fetch {source_id}/{indicator_id}: {result.error}"
                        )
                        self.metrics.warnings.append(f"{source_id}/{indicator_id}: {result.error}")

                except Exception as e:
                    self.logger.warning(f"Error fetching {source_id}/{indicator_id}: {e}")
                    self.metrics.warnings.append(f"{source_id}/{indicator_id}: {e}")

            # Log summary for source
            success_count = len(results[source_id])
            total_count = len(indicator_ids)
            self.logger.info(f"{source_id}: collected {success_count}/{total_count} indicators")

        return results

    def _store_to_database(self, data: Dict[str, Any]) -> None:
        """Store collected data to database."""
        if self.dry_run:
            self.logger.info("Dry run - not storing to database")
            return

        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            for source_id, indicators in data.items():
                for indicator_id, indicator_data in indicators.items():
                    if not indicator_data or not indicator_data.get("data"):
                        continue

                    raw_data = indicator_data["data"]

                    # Extract observation date
                    obs_date = raw_data.get("date", datetime.now().date())
                    if isinstance(obs_date, str):
                        from datetime import datetime as dt
                        obs_date = dt.fromisoformat(obs_date).date()

                    # Compute hash for change detection
                    record_hash = compute_record_hash(raw_data)

                    # Check existing
                    cursor.execute("""
                        SELECT source_hash FROM regional_fed_indicators
                        WHERE district = %s AND indicator_name = %s AND observation_date = %s
                    """, (source_id, indicator_id, obs_date))
                    existing = cursor.fetchone()

                    if existing:
                        if existing[0] == record_hash:
                            self.metrics.records_skipped += 1
                            continue
                        # Update
                        cursor.execute("""
                            UPDATE regional_fed_indicators SET
                                indicator_data = %s,
                                source_hash = %s,
                                source_fetch_timestamp = NOW(),
                                updated_at = NOW()
                            WHERE district = %s AND indicator_name = %s AND observation_date = %s
                        """, (
                            json.dumps(raw_data, default=str),
                            record_hash,
                            source_id,
                            indicator_id,
                            obs_date,
                        ))
                        self.metrics.records_updated += 1
                    else:
                        # Insert
                        cursor.execute("""
                            INSERT INTO regional_fed_indicators
                                (district, indicator_name, observation_date, indicator_data,
                                 source_hash, source_fetch_timestamp)
                            VALUES (%s, %s, %s, %s, %s, NOW())
                        """, (
                            source_id,
                            indicator_id,
                            obs_date,
                            json.dumps(raw_data, default=str),
                            record_hash,
                        ))
                        self.metrics.records_inserted += 1

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            self.metrics.errors.append(f"Database: {e}")

    def collect(self) -> CollectionMetrics:
        """Collect all economic indicators."""
        try:
            self.logger.info("Starting economic indicators collection")

            # Run async collection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self._results = loop.run_until_complete(self._collect_all_async())
            finally:
                # Clean up fetcher
                if self._fetcher:
                    loop.run_until_complete(self._fetcher.close())
                loop.close()

            # Store to database
            self._store_to_database(self._results)

            self.logger.info(
                f"Economic indicators: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"Collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    @property
    def results(self) -> Dict[str, Any]:
        """Get the collected results."""
        return self._results


async def run_health_check() -> Dict[str, Any]:
    """Run health check on all data sources."""
    from investigator.infrastructure.external.data_fetcher import get_data_fetcher

    fetcher = get_data_fetcher()

    print("Running health check on all data sources...")
    print("This may take a few minutes.\n")

    await fetcher.health_check()
    report = fetcher.get_health_report()

    await fetcher.close()

    return report


def print_health_report(report: Dict[str, Any]):
    """Print health report in a readable format."""
    print("\n" + "=" * 60)
    print("DATA SOURCES HEALTH REPORT")
    print("=" * 60)
    print(f"\nTotal sources: {report['total_sources']}")
    print(f"Healthy: {report['healthy']}")
    print(f"Unhealthy: {report['unhealthy']}")
    print(f"\nUsing primary URL: {report['using_primary_url']}")
    print(f"Using fallback URL: {report['using_fallback_url']}")
    print(f"Using FRED fallback: {report['using_fred']}")

    print("\n" + "-" * 60)
    print("SOURCE DETAILS:")
    print("-" * 60)

    for source_key, status in sorted(report["sources"].items()):
        status_icon = "✓" if status["healthy"] else "✗"
        fallback_info = ""
        if status["using_fred"]:
            fallback_info = " [FRED]"
        elif status["using_fallback"]:
            fallback_info = " [fallback]"

        print(f"  {status_icon} {source_key}{fallback_info}")
        if status["failure_count"] > 0:
            print(f"      Failures: {status['failure_count']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect economic indicators from all sources"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Filter by source (e.g., atlanta_fed, chicago_fed)"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        help="Filter by indicator (e.g., gdpnow, cfnai)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect but don't store to database"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check on all data sources"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    # Health check mode
    if args.health_check:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            report = loop.run_until_complete(run_health_check())
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print_health_report(report)
        finally:
            loop.close()
        sys.exit(0)

    # Parse filters
    sources = None
    if args.source:
        sources = [s.strip().lower() for s in args.source.split(",")]

    indicators = None
    if args.indicator:
        indicators = [i.strip().lower() for i in args.indicator.split(",")]

    # Run collector
    collector = EconomicIndicatorsCollector(
        sources=sources,
        indicators=indicators,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        # Run and print results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(collector._collect_all_async())
            if collector._fetcher:
                loop.run_until_complete(collector._fetcher.close())
        finally:
            loop.close()

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print("\n=== Collected Economic Indicators ===\n")
            for source_id, indicators in results.items():
                print(f"\n{source_id.upper()}:")
                for ind_id, data in indicators.items():
                    if data and data.get("data"):
                        value = data["data"].get("value", "N/A")
                        source = data.get("source", "unknown")
                        print(f"  {ind_id}: {value} (from {source})")
    else:
        exit_code = collector.run()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
