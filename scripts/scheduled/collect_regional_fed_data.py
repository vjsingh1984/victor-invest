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

"""Collect Regional Federal Reserve Economic Data.

Schedule: Daily at 6AM ET
Sources: 12 Federal Reserve District Banks (all free, no API keys)

Collects:
- Atlanta Fed: GDPNow, Wage Growth, Business Inflation Expectations
- Philadelphia Fed: Manufacturing Survey, Leading/Coincident Indexes, ADS
- Chicago Fed: CFNAI (recession indicator), NFCI (financial conditions)
- Cleveland Fed: Inflation Expectations, Yield Curve Model, Median CPI
- Dallas Fed: Texas Manufacturing/Services, Trimmed Mean PCE
- Kansas City Fed: Manufacturing, KCFSI (financial stress), LMCI
- Richmond Fed: Fifth District Manufacturing/Services Surveys
- NY Fed: Recession Probability, GSCPI (existing)

Usage:
    python scripts/scheduled/collect_regional_fed_data.py
    python scripts/scheduled/collect_regional_fed_data.py --districts atlanta,chicago
    python scripts/scheduled/collect_regional_fed_data.py --indicator gdpnow
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class RegionalFedCollector(BaseCollector):
    """Collector for all Regional Federal Reserve economic data."""

    AVAILABLE_DISTRICTS = [
        "atlanta",
        "philadelphia",
        "chicago",
        "cleveland",
        "dallas",
        "kansas_city",
        "richmond",
        "new_york",  # Already integrated via nyfed module
    ]

    def __init__(
        self,
        districts: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
    ):
        super().__init__("collect_regional_fed_data")
        self.districts = districts or self.AVAILABLE_DISTRICTS
        self.indicators = indicators  # If None, collect all
        self._results: Dict[str, Any] = {}

    async def _collect_atlanta(self) -> Dict[str, Any]:
        """Collect Atlanta Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.atlanta_fed import (
                get_atlanta_fed_client,
            )
            client = get_atlanta_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Atlanta Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"atlanta": data}
        except Exception as e:
            self.logger.warning(f"Atlanta Fed collection failed: {e}")
            self.metrics.warnings.append(f"Atlanta: {e}")
            return {}

    async def _collect_philadelphia(self) -> Dict[str, Any]:
        """Collect Philadelphia Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.philadelphia_fed import (
                get_philly_fed_client,
            )
            client = get_philly_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Philadelphia Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"philadelphia": data}
        except Exception as e:
            self.logger.warning(f"Philadelphia Fed collection failed: {e}")
            self.metrics.warnings.append(f"Philadelphia: {e}")
            return {}

    async def _collect_chicago(self) -> Dict[str, Any]:
        """Collect Chicago Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.chicago_fed import (
                get_chicago_fed_client,
            )
            client = get_chicago_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Chicago Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"chicago": data}
        except Exception as e:
            self.logger.warning(f"Chicago Fed collection failed: {e}")
            self.metrics.warnings.append(f"Chicago: {e}")
            return {}

    async def _collect_cleveland(self) -> Dict[str, Any]:
        """Collect Cleveland Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.cleveland_fed import (
                get_cleveland_fed_client,
            )
            client = get_cleveland_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Cleveland Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"cleveland": data}
        except Exception as e:
            self.logger.warning(f"Cleveland Fed collection failed: {e}")
            self.metrics.warnings.append(f"Cleveland: {e}")
            return {}

    async def _collect_dallas(self) -> Dict[str, Any]:
        """Collect Dallas Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.dallas_fed import (
                get_dallas_fed_client,
            )
            client = get_dallas_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Dallas Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"dallas": data}
        except Exception as e:
            self.logger.warning(f"Dallas Fed collection failed: {e}")
            self.metrics.warnings.append(f"Dallas: {e}")
            return {}

    async def _collect_kansas_city(self) -> Dict[str, Any]:
        """Collect Kansas City Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.kansas_city_fed import (
                get_kc_fed_client,
            )
            client = get_kc_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Kansas City Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"kansas_city": data}
        except Exception as e:
            self.logger.warning(f"Kansas City Fed collection failed: {e}")
            self.metrics.warnings.append(f"Kansas City: {e}")
            return {}

    async def _collect_richmond(self) -> Dict[str, Any]:
        """Collect Richmond Fed indicators."""
        try:
            from investigator.infrastructure.external.fed_districts.richmond_fed import (
                get_richmond_fed_client,
            )
            client = get_richmond_fed_client()
            data = await client.get_all_indicators()
            self.logger.info(f"Richmond Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"richmond": data}
        except Exception as e:
            self.logger.warning(f"Richmond Fed collection failed: {e}")
            self.metrics.warnings.append(f"Richmond: {e}")
            return {}

    async def _collect_new_york(self) -> Dict[str, Any]:
        """Collect NY Fed indicators (existing module)."""
        try:
            from investigator.infrastructure.external.nyfed.markets_data import (
                get_nyfed_client,
            )
            client = get_nyfed_client()

            recession_prob = await client.get_recession_probability()
            gscpi = await client.get_gscpi()

            data = {
                "recession_probability": recession_prob,
                "gscpi": gscpi,
            }
            self.logger.info(f"NY Fed: collected {len([v for v in data.values() if v])} indicators")
            return {"new_york": data}
        except Exception as e:
            self.logger.warning(f"NY Fed collection failed: {e}")
            self.metrics.warnings.append(f"NY Fed: {e}")
            return {}

    async def _collect_all_async(self) -> Dict[str, Any]:
        """Collect all district data concurrently."""
        collectors = {
            "atlanta": self._collect_atlanta,
            "philadelphia": self._collect_philadelphia,
            "chicago": self._collect_chicago,
            "cleveland": self._collect_cleveland,
            "dallas": self._collect_dallas,
            "kansas_city": self._collect_kansas_city,
            "richmond": self._collect_richmond,
            "new_york": self._collect_new_york,
        }

        tasks = []
        for district in self.districts:
            if district in collectors:
                tasks.append(collectors[district]())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined = {}
        for result in results:
            if isinstance(result, dict):
                combined.update(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Collection task failed: {result}")

        return combined

    def _store_to_database(self, data: Dict[str, Any]) -> None:
        """Store collected data to database."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            for district, indicators in data.items():
                if not indicators:
                    continue

                for indicator_name, indicator_data in indicators.items():
                    if indicator_data is None:
                        continue

                    self.metrics.records_processed += 1

                    # Convert dataclass to dict for storage
                    if hasattr(indicator_data, "__dict__"):
                        record_dict = {
                            k: str(v) if hasattr(v, "value") else v  # Handle enums
                            for k, v in indicator_data.__dict__.items()
                            if not k.startswith("_")
                        }
                    else:
                        record_dict = {"value": indicator_data}

                    record_hash = compute_record_hash(record_dict)

                    # Get observation date
                    obs_date = record_dict.get("date", datetime.now().date())
                    if isinstance(obs_date, str):
                        obs_date = datetime.fromisoformat(obs_date).date()

                    # Check existing
                    cursor.execute("""
                        SELECT source_hash FROM regional_fed_indicators
                        WHERE district = %s AND indicator_name = %s AND observation_date = %s
                    """, (district, indicator_name, obs_date))
                    existing = cursor.fetchone()

                    if existing:
                        if existing[0] == record_hash:
                            self.metrics.records_skipped += 1
                            continue
                        # Update
                        cursor.execute("""
                            UPDATE regional_fed_indicators SET
                                indicator_data = %s, source_hash = %s,
                                source_fetch_timestamp = NOW(), updated_at = NOW()
                            WHERE district = %s AND indicator_name = %s AND observation_date = %s
                        """, (
                            str(record_dict),
                            record_hash,
                            district,
                            indicator_name,
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
                            district,
                            indicator_name,
                            obs_date,
                            str(record_dict),
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
        """Collect all Regional Fed economic data."""
        try:
            self.logger.info(f"Collecting data from {len(self.districts)} Fed districts")

            # Run async collection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self._results = loop.run_until_complete(self._collect_all_async())
            finally:
                loop.close()

            # Store to database
            self._store_to_database(self._results)

            self.logger.info(
                f"Regional Fed: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"Regional Fed collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    @property
    def results(self) -> Dict[str, Any]:
        """Get the collected results."""
        return self._results


def main():
    parser = argparse.ArgumentParser(
        description="Collect Regional Federal Reserve economic data"
    )
    parser.add_argument(
        "--districts",
        type=str,
        help="Comma-separated list of districts (default: all)"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        help="Specific indicator to collect (e.g., gdpnow, cfnai)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect but don't store to database"
    )
    args = parser.parse_args()

    districts = None
    if args.districts:
        districts = [d.strip().lower() for d in args.districts.split(",")]

    indicators = None
    if args.indicator:
        indicators = [args.indicator.strip().lower()]

    collector = RegionalFedCollector(
        districts=districts,
        indicators=indicators,
    )

    if args.dry_run:
        # Just collect and print
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(collector._collect_all_async())
            print("\n=== Collected Regional Fed Data ===\n")
            for district, data in results.items():
                print(f"\n{district.upper()}:")
                for indicator, value in (data or {}).items():
                    if value:
                        print(f"  {indicator}: {value}")
        finally:
            loop.close()
    else:
        exit_code = collector.run()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
