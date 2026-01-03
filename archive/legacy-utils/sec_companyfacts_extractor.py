#!/usr/bin/env python3
"""
InvestiGator - SEC Company Facts Extractor
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Extracts financial data from SEC Company Facts API (with database fallback)
"""

import logging
from typing import Dict, Optional, List, Tuple, Any
from sqlalchemy import create_engine, text
from datetime import datetime

logger = logging.getLogger(__name__)


class SECCompanyFactsExtractor:
    """
    Extracts structured financial data from SEC Company Facts API.
    Uses database-first strategy with quarterly staleness check.

    STRATEGY (Database-First for Cost Optimization):
    1. PRIMARY: Check database first (fast, local, no network cost)
    2. STALENESS CHECK: If data < 90 days old (1 quarter), use cached data
    3. REFRESH: If stale (>= 90 days) or missing, fetch from SEC API
    4. FALLBACK: If API fails, use stale database data (better than nothing)

    Rationale: SEC filings (10-K/10-Q) are released quarterly, so data
    older than 90 days is likely stale. This reduces network cost by 90%+
    while maintaining data freshness aligned with filing frequency.
    """

    def __init__(self, db_config: Dict = None):
        """
        Initialize with database configuration

        NOTE: This class NO LONGER fetches from SEC API directly.
        SEC API fetching is handled by SEC Agent (agents/sec_agent.py).
        This class only reads raw data from cache and processes it.

        Args:
            db_config: Database configuration dict
        """
        self.db_config = db_config or {
            "host": "${DB_HOST:-localhost}",
            "port": 5432,
            "database": "sec_database",
            "username": "investigator",
            "password": "investigator",
        }
        self.engine = self._create_engine()
        self.default_max_age_days = 90  # 1 quarter staleness threshold

    def _create_engine(self):
        """Create SQLAlchemy engine for database connection"""
        connection_string = (
            f"postgresql://{self.db_config['username']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )

        return create_engine(connection_string, pool_size=5, max_overflow=10, pool_pre_ping=True, echo=False)

    def get_company_facts(
        self, symbol: str, cik: str = None, max_age_days: int = None, force_refresh: bool = False
    ) -> Optional[Dict]:
        """
        Get company facts with database-first strategy and staleness check

        STRATEGY:
        1. Try database first (unless force_refresh=True)
        2. If cached data < max_age_days old, return cached data
        3. If stale or missing, fetch from SEC API and update cache
        4. If API fails, fallback to stale database data

        Args:
            symbol: Stock ticker symbol
            cik: Optional CIK (if not provided, will look up from database)
            max_age_days: Maximum age in days before data considered stale (default: 90 days = 1 quarter)
            force_refresh: Force API fetch even if cached data is fresh (default: False)

        Returns:
            Dictionary with company facts data or None if not found
        """
        if max_age_days is None:
            max_age_days = self.default_max_age_days

        # Step 1: Try database first (unless force_refresh)
        db_data = None
        age_days = None

        if not force_refresh:
            db_data = self._fetch_from_database(symbol)

            if db_data:
                age_days = self._calculate_age_days(db_data["fetched_at"])

                # Return cached data if fresh enough
                if age_days < max_age_days:
                    logger.info(
                        f"✓ Using cached data for {symbol} " f"({age_days:.1f} days old < {max_age_days} day threshold)"
                    )
                    db_data["source"] = "database_cache"
                    db_data["cache_age_days"] = age_days
                    return db_data
                else:
                    logger.info(
                        f"Data stale for {symbol} "
                        f"({age_days:.1f} days >= {max_age_days} day threshold), "
                        f"fetching fresh from SEC API"
                    )

        # Step 2: No fresh data in cache - return None to signal caller to run SEC Agent
        logger.info(f"No fresh data for {symbol} in cache. " f"Caller should trigger SEC Agent to fetch from API.")
        return None

    def _calculate_age_days(self, fetched_at_str: str) -> float:
        """
        Calculate age in days from fetched_at timestamp

        Args:
            fetched_at_str: ISO format timestamp string

        Returns:
            Age in days as float
        """
        try:
            if not fetched_at_str:
                return float("inf")  # Very old if no timestamp

            # Parse ISO format timestamp
            if isinstance(fetched_at_str, str):
                fetched_at = datetime.fromisoformat(fetched_at_str.replace("Z", "+00:00"))
            else:
                fetched_at = fetched_at_str

            age_delta = datetime.utcnow() - fetched_at.replace(tzinfo=None)
            return age_delta.total_seconds() / 86400  # Convert to days

        except Exception as e:
            logger.warning(f"Error calculating age from {fetched_at_str}: {e}")
            return float("inf")  # Treat as very old if can't calculate

    def _fetch_from_database(self, symbol: str) -> Optional[Dict]:
        """
        Fetch RAW company facts from sec_companyfacts_raw table (3-table architecture)

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with RAW SEC API response (with us-gaap structure) or None if not found
        """
        try:
            query = text(
                """
                SELECT
                    id,
                    symbol,
                    cik,
                    entity_name,
                    companyfacts,
                    fetched_at,
                    api_checksum
                FROM sec_companyfacts_raw
                WHERE symbol = :symbol
                LIMIT 1
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol.upper()}).fetchone()

            if not result:
                logger.debug(f"No cached data found in sec_companyfacts_raw for {symbol}")
                return None

            # Validate that cached data has us-gaap structure
            companyfacts = result.companyfacts
            if "facts" not in companyfacts or "us-gaap" not in companyfacts.get("facts", {}):
                logger.warning(
                    f"❌ Cached data for {symbol} in database has INVALID structure (missing us-gaap). "
                    f"Treating as stale and will re-fetch from API."
                )
                # Delete invalid cache entry
                try:
                    delete_query = text("DELETE FROM sec_companyfacts_raw WHERE symbol = :symbol")
                    with self.engine.connect() as conn:
                        conn.execute(delete_query, {"symbol": symbol.upper()})
                        conn.commit()
                    logger.info(f"🗑️  Deleted invalid cache for {symbol}")
                except Exception as del_e:
                    logger.warning(f"Failed to delete invalid cache: {del_e}")
                return None

            logger.debug(f"✅ Found valid cached data with us-gaap structure for {symbol}")

            return {
                "symbol": result.symbol,
                "cik": result.cik,
                "entityName": result.entity_name,
                "facts": companyfacts.get("facts", {}),
                "fetched_at": result.fetched_at.isoformat() if result.fetched_at else None,
                "source": "database_cache",
                "raw_data_id": result.id,
            }

        except Exception as e:
            logger.error(f"Error fetching {symbol} from sec_companyfacts_raw: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _fetch_from_api(self, symbol: str, cik: str) -> Optional[Dict]:
        """
        Fetch company facts from SEC API

        Args:
            symbol: Stock ticker symbol
            cik: CIK number for API call

        Returns:
            Dictionary with company facts or None if failed
        """
        try:
            # Fetch from SEC API (run in new thread to avoid event loop conflicts)
            import asyncio
            import concurrent.futures

            async def fetch_from_api():
                """Async wrapper for API call"""
                return await self.sec_api_client.get_company_facts(cik)

            # Run in separate thread with its own event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, fetch_from_api())
                api_data = future.result(timeout=30)

            if api_data and "facts" in api_data:
                # Validate raw SEC structure
                raw_facts = api_data.get("facts", {})
                has_us_gaap = "us-gaap" in raw_facts
                has_dei = "dei" in raw_facts

                logger.info(f"📦 SEC API response for {symbol}: " f"has us-gaap={has_us_gaap}, has dei={has_dei}")

                if has_us_gaap:
                    us_gaap_tags = len(raw_facts["us-gaap"])
                    logger.info(f"✅ Raw us-gaap structure preserved: {us_gaap_tags} XBRL tags")
                else:
                    logger.warning(
                        f"⚠️  SEC API response missing us-gaap structure for {symbol}! "
                        f"Keys found: {list(raw_facts.keys())}"
                    )

                # Store FULL raw SEC API response (preserve us-gaap structure for hybrid strategy)
                result = {
                    "symbol": symbol,
                    "cik": api_data.get("cik", cik),
                    "company_name": api_data.get("entityName", ""),
                    "facts": api_data.get("facts", {}),  # Raw SEC structure: {'us-gaap': {...}, 'dei': {...}}
                    "entityName": api_data.get("entityName", ""),  # Preserve entity name
                    "fetched_at": datetime.utcnow().isoformat(),
                    "source": "sec_api",
                }

                logger.info(f"🔄 Returning raw SEC data for {symbol} to be cached")
                return result

            return None

        except Exception as e:
            logger.error(f"Error fetching {symbol} from SEC API: {e}")
            return None

    def _save_to_database(self, data: Dict) -> int:
        """
        Save RAW SEC API response to sec_companyfacts_raw table (3-table architecture)

        This method saves the EXACT SEC API response with us-gaap structure.

        Args:
            data: Dictionary with RAW SEC API response (must include 'facts' with 'us-gaap')

        Returns:
            raw_data_id (int) from sec_companyfacts_raw table, or None if failed
        """
        try:
            import json
            import hashlib

            symbol = data.get("symbol", "UNKNOWN")
            cik = data.get("cik", "UNKNOWN")

            # SAFEGUARD: Validate that we're saving RAW data, not processed data
            if "facts" not in data:
                logger.error(f"❌ Data for {symbol} missing 'facts' key, cannot save")
                return None

            facts = data["facts"]
            has_us_gaap = "us-gaap" in facts
            has_processed_fields = any(
                field in facts for field in ["revenues", "net_income", "assets", "liabilities", "equity"]
            )

            if not has_us_gaap and has_processed_fields:
                logger.error(
                    f"🚫 BLOCKED: Attempted to save PROCESSED data (no us-gaap) for {symbol}! "
                    f"Skipping database save to preserve data integrity."
                )
                return None

            if not has_us_gaap:
                logger.error(f"❌ Data for {symbol} missing us-gaap structure, cannot save to raw table")
                return None

            us_gaap_tag_count = len(facts["us-gaap"])
            logger.info(f"✅ Saving RAW SEC data for {symbol}: {us_gaap_tag_count} us-gaap XBRL tags")

            # Calculate checksum for integrity verification
            raw_json = json.dumps(data, sort_keys=True)
            checksum = hashlib.sha256(raw_json.encode()).hexdigest()

            # Save to sec_companyfacts_raw table (3-table architecture)
            query = text(
                """
                INSERT INTO sec_companyfacts_raw
                (symbol, cik, entity_name, companyfacts, api_version, api_response_size, api_checksum)
                VALUES (:symbol, :cik, :entity_name, CAST(:companyfacts AS jsonb), 'v1.0', :size, :checksum)
                ON CONFLICT (symbol) DO UPDATE SET
                    cik = EXCLUDED.cik,
                    entity_name = EXCLUDED.entity_name,
                    companyfacts = EXCLUDED.companyfacts,
                    fetched_at = NOW(),
                    api_response_size = EXCLUDED.api_response_size,
                    api_checksum = EXCLUDED.api_checksum
                RETURNING id
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "symbol": symbol.upper(),
                        "cik": cik,
                        "entity_name": data.get("entityName", ""),
                        "companyfacts": raw_json,
                        "size": len(raw_json),
                        "checksum": checksum,
                    },
                )
                raw_id = result.fetchone()[0]
                conn.commit()

            logger.info(
                f"💾 Saved raw SEC data to sec_companyfacts_raw: "
                f"{symbol} (id={raw_id}, size={len(raw_json)} bytes, checksum={checksum[:8]}...)"
            )

            # Trigger processing of raw data → processed table
            self._trigger_processing(symbol, data, raw_id)

            return raw_id

        except Exception as e:
            logger.error(f"❌ Error saving {data.get('symbol', 'UNKNOWN')} to database: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _trigger_processing(self, symbol: str, raw_data: Dict, raw_data_id: int):
        """
        Trigger processing of raw SEC data to populate sec_companyfacts_processed table

        This is called immediately after saving raw data to ensure processed table
        is always up-to-date with raw table.

        Args:
            symbol: Stock ticker
            raw_data: Raw SEC API response
            raw_data_id: ID from sec_companyfacts_raw table
        """
        try:
            from investigator.infrastructure.sec.data_processor import get_sec_data_processor

            logger.info(f"⚙️  Triggering processing for {symbol} (raw_data_id={raw_data_id})")

            processor = get_sec_data_processor(self.engine)
            processed_filings = processor.process_raw_data(symbol, raw_data, raw_data_id)

            if processed_filings:
                saved_count = processor.save_processed_data(processed_filings)
                logger.info(
                    f"✅ Processing complete for {symbol}: "
                    f"{len(processed_filings)} filings extracted, {saved_count} saved to sec_companyfacts_processed"
                )
            else:
                logger.warning(f"⚠️  No filings extracted during processing for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error processing raw data for {symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _get_cik_from_db(self, symbol: str) -> Optional[str]:
        """
        Get CIK for symbol using TickerCIKMapper (reads from data/ticker_cik_map.txt).

        This method uses the ticker-to-CIK mapping file instead of querying the database,
        which solves the chicken-and-egg problem where the database would be empty for
        first-time symbol lookups.
        """
        try:
            from utils.ticker_cik_mapper import TickerCIKMapper

            mapper = TickerCIKMapper()
            cik = mapper.get_cik(symbol)

            if cik:
                # CIK from map file is NOT zero-padded, need to pad to 10 digits
                cik_padded = cik.zfill(10)
                logger.debug(f"Found CIK {cik_padded} for symbol {symbol} via TickerCIKMapper")
                return cik_padded
            else:
                logger.warning(f"No CIK found for {symbol} in ticker map")
                return None

        except Exception as e:
            logger.error(f"Error fetching CIK for {symbol} via TickerCIKMapper: {e}")
            return None

    def _get_latest_value(self, units_data: List[Dict], prefer_annual: bool = True) -> Optional[float]:
        """
        Get the most recent value from units array.

        Args:
            units_data: List of data points with 'val', 'end', 'fy', 'fp' fields
            prefer_annual: Prefer annual (FY) data over quarterly

        Returns:
            Latest value as float or None
        """
        if not units_data:
            return None

        try:
            # Sort by FISCAL YEAR + PERIOD (not just end date) to get truly latest data
            # This ensures we get 2024-FY instead of 2018-FY with a recent 'end' date
            def get_sort_key(entry):
                """Sort by fiscal year (descending), then period priority (FY > Q4 > Q3 > Q2 > Q1)"""
                fy = entry.get("fy", 0)  # Fiscal year (e.g., 2024)
                fp = entry.get("fp", "")  # Fiscal period (e.g., 'FY', 'Q4', 'Q3')

                # Period priority: FY=5, Q4=4, Q3=3, Q2=2, Q1=1, unknown=0
                period_priority = {"FY": 5, "Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}.get(fp, 0)

                return (fy, period_priority)

            sorted_data = sorted(units_data, key=get_sort_key, reverse=True)  # Most recent fiscal year + period first

            # If prefer_annual, try to find most recent annual data first
            if prefer_annual:
                annual_data = [d for d in sorted_data if d.get("fp") == "FY" or d.get("form") in ["10-K", "20-F"]]
                if annual_data:
                    return float(annual_data[0]["val"])

            # Return most recent value (quarterly or annual)
            if sorted_data:
                return float(sorted_data[0]["val"])

            return None

        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Error extracting latest value: {e}")
            return None

    def _get_latest_value_with_period(
        self, units_data: List[Dict], prefer_annual: bool = True
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        """
        Get the most recent value from units array along with fiscal period info.

        DEPRECATION NOTE: This method is now wrapped by get_metric_with_hybrid_strategy()
        which adds DERA bulk table support. Direct calls to this method will only use
        JSON API data and miss potential bulk table data for recent quarters.

        Consider using get_metric_with_hybrid_strategy() instead for dual-source extraction.

        Args:
            units_data: List of data points with 'val', 'end', 'fy', 'fp' fields
            prefer_annual: Prefer annual (FY) data over quarterly

        Returns:
            Tuple of (value, fiscal_year, fiscal_period) e.g. (100.0, 2024, 'Q3')
        """
        if not units_data:
            return (None, None, None)

        try:
            # CRITICAL FIX: Filter to only standard 10-Q/10-K filings
            # Excludes amendments (10-Q/A, 10-K/A), restatements, and other non-standard forms
            # This prevents old data (e.g., 2012-FY) from being selected over recent quarters
            filtered_data = [
                entry
                for entry in units_data
                if entry.get("form") in ["10-Q", "10-K", "20-F"]  # Standard quarterly/annual filings only
            ]

            # Fallback: If no standard forms found, use all data (better than nothing)
            if not filtered_data:
                logger.warning(f"No standard 10-Q/10-K filings found in data, using all {len(units_data)} entries")
                filtered_data = units_data
            else:
                logger.debug(f"Filtered to {len(filtered_data)} standard filings from {len(units_data)} total entries")

            # Sort by FISCAL YEAR + PERIOD (not just end date) to get truly latest data
            def get_sort_key(entry):
                """Sort by fiscal year (descending), then period priority (FY > Q4 > Q3 > Q2 > Q1)"""
                fy = entry.get("fy", 0)
                fp = entry.get("fp", "")
                period_priority = {"FY": 5, "Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}.get(fp, 0)
                return (fy, period_priority)

            sorted_data = sorted(
                filtered_data,  # Use filtered data instead of raw units_data
                key=get_sort_key,
                reverse=True,  # Most recent fiscal year + period first
            )

            # If prefer_annual, try to find most recent annual data first
            if prefer_annual:
                annual_data = [d for d in sorted_data if d.get("fp") == "FY" or d.get("form") in ["10-K", "20-F"]]
                if annual_data:
                    latest = annual_data[0]
                    return (float(latest["val"]), latest.get("fy"), latest.get("fp", "FY"))

            # Return most recent value (quarterly or annual) with period info
            if sorted_data:
                latest = sorted_data[0]
                return (float(latest["val"]), latest.get("fy"), latest.get("fp", "FY"))

            return (None, None, None)

        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Error extracting latest value with period: {e}")
            return (None, None, None)

    def _derive_fiscal_period_from_date(self, end_date: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Derive fiscal period from end date (fallback when raw SEC data not available).

        Args:
            end_date: Date string in format YYYY-MM-DD

        Returns:
            Tuple of (fiscal_year, fiscal_period) e.g. (2024, 'Q2')
        """
        try:
            from datetime import datetime

            date_obj = datetime.fromisoformat(end_date)
            year = date_obj.year
            month = date_obj.month

            # Determine quarter from month
            quarter = ((month - 1) // 3) + 1
            fiscal_period = f"Q{quarter}"

            return (year, fiscal_period)

        except (ValueError, TypeError) as e:
            logger.debug(f"Error deriving fiscal period from date {end_date}: {e}")
            return (None, None)

    def _determine_latest_fiscal_period(
        self, symbol: str, us_gaap: Dict[str, Any], cik: str = None
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        CRITICAL: Determine the latest fiscal period (FY, FP) FIRST by collecting ALL
        distinct fiscal periods across ALL tags in the JSON API.

        SIMPLE STRATEGY (no tag-specific logic):
        1. Check bulk tables first (if available and fresh)
        2. Collect ALL distinct (fy, fp, filed) tuples from ALL tags in JSON API
        3. Sort by filed date (descending) and pick the FIRST one
        4. This ensures we get the truly latest period that applies to ALL metrics

        Args:
            symbol: Stock ticker
            us_gaap: us-gaap taxonomy dict from CompanyFacts API
            cik: CIK number (optional, enables bulk table check)

        Returns:
            Tuple of (fiscal_year, fiscal_period, filed_date)
            Example: (2025, 'Q3', '2025-10-28')
        """
        try:
            # TIER 1: Try bulk tables FIRST (if CIK provided and data fresh)
            if cik:
                from utils.sec_data_strategy import get_fiscal_period_strategy

                strategy = get_fiscal_period_strategy()
                bulk_age = strategy._check_bulk_data_age(cik)

                if bulk_age is not None and bulk_age <= 180:  # Fresh bulk data
                    latest_fy, latest_fp, _ = strategy._get_from_bulk_tables(cik)
                    if latest_fy and latest_fp:
                        logger.info(
                            f"✓ Determined latest period from bulk tables for {symbol}: " f"{latest_fy}-{latest_fp}"
                        )
                        return (latest_fy, latest_fp, None)  # No filed date from bulk

            # TIER 2: Collect ALL distinct fiscal periods from JSON API (across ALL tags)
            # SIMPLE: No tag-specific logic - just get all (fy, fp, filed) tuples
            all_periods = set()

            for tag_name, tag_data in us_gaap.items():
                units = tag_data.get("units", {})
                usd_data = units.get("USD", [])
                if not usd_data:
                    continue

                # Filter to standard forms only (10-Q, 10-K, 20-F)
                for entry in usd_data:
                    form = entry.get("form")
                    if form not in ["10-Q", "10-K", "20-F"]:
                        continue

                    fy = entry.get("fy")
                    fp = entry.get("fp")
                    filed = entry.get("filed")

                    if fy and fp and filed:
                        # Add to set of all periods found
                        all_periods.add((filed, fy, fp))

            if not all_periods:
                logger.warning(f"Could not find any fiscal periods in JSON API for {symbol}")
                return (None, None, None)

            # Sort by filed date (descending) and pick the FIRST one
            sorted_periods = sorted(all_periods, reverse=True)
            filed_date, fiscal_year, fiscal_period = sorted_periods[0]

            logger.info(
                f"✓ Determined latest period from JSON API for {symbol}: "
                f"{fiscal_year}-{fiscal_period} (filed: {filed_date}) "
                f"[found {len(all_periods)} total periods]"
            )
            return (fiscal_year, fiscal_period, filed_date)

        except Exception as e:
            logger.warning(f"Error determining latest fiscal period for {symbol}: {e}")
            return (None, None, None)

    def _extract_period_from_cache(self, cached_data: Dict) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract fiscal period from cached/flattened data.

        CRITICAL: Uses ACTUAL fiscal period from SEC data if available,
        NOT calendar-derived period.

        Args:
            cached_data: Flattened cache data with 'facts' dict

        Returns:
            Tuple of (fiscal_year, fiscal_period)
        """
        try:
            facts = cached_data.get("facts", {})

            # PRIORITY 1: Check if fiscal period is already in cached facts
            # This would be from actual SEC data (fy/fp fields)
            cached_fy = facts.get("fiscal_year")
            cached_fp = facts.get("fiscal_period")

            if cached_fy and cached_fp:
                # CRITICAL: Validate this isn't a future period
                # SEC filings have lag time (45 days for 10-Q, 90 days for 10-K)
                # So current quarter can't be filed yet
                from datetime import datetime

                current_year = datetime.now().year
                current_month = datetime.now().month
                current_quarter = ((current_month - 1) // 3) + 1

                # Extract quarter number from fiscal_period (Q1=1, Q2=2, etc.)
                if isinstance(cached_fp, str) and cached_fp.startswith("Q"):
                    cached_quarter = int(cached_fp[1])
                elif cached_fp == "FY":
                    cached_quarter = 4  # Full year treated as Q4
                else:
                    # Unknown format, accept it
                    return (int(cached_fy), cached_fp)

                # Check if this is a future/current period (impossible to have filed)
                # SEC has filing lag: 45 days for 10-Q, 90 days for 10-K
                # Current quarter can't be filed until NEXT quarter
                is_future = (cached_fy > current_year) or (
                    cached_fy == current_year and cached_quarter >= current_quarter
                )

                if is_future:
                    logger.warning(
                        f"Cached period {cached_fy}-{cached_fp} is current/future quarter. "
                        f"SEC filings have 45-90 day lag, so this can't be filed yet. "
                        f"This indicates calendar-based period, not actual SEC filing. "
                        f"Returning None to force re-extraction from SEC data."
                    )
                    return (None, None)

                # Return validated historical period
                return (int(cached_fy), cached_fp)

            # PRIORITY 2: Do NOT derive from data_date (calendar date)
            # This creates future-dated periods. Return None instead.
            # The caller should extract from actual SEC CompanyFacts fy/fp fields.

            logger.debug(
                "No valid fiscal period in cached data. " "Caller should extract from SEC CompanyFacts fy/fp fields."
            )
            return (None, None)

        except Exception as e:
            logger.debug(f"Error extracting period from cache: {e}")
            return (None, None)

    def get_metric_with_hybrid_strategy(
        self,
        symbol: str,
        cik: str,
        metric_tag: str,
        fiscal_year: int = None,
        fiscal_period: str = None,
        max_bulk_age_days: int = 180,
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        """
        STANDARDIZED HYBRID EXTRACTION: Get a single metric value using dual strategy.

        This is the REUSABLE function that both multi-quarter and single-quarter
        extraction should use for consistent data retrieval.

        Strategy:
        1. Try DERA bulk tables FIRST (authoritative, fast)
        2. If bulk data is stale (>max_bulk_age_days), fallback to JSON API

        Args:
            symbol: Stock ticker
            cik: CIK number
            metric_tag: XBRL tag name (e.g., 'Revenues', 'Assets')
            fiscal_year: Specific fiscal year to extract (None = latest)
            fiscal_period: Specific fiscal period to extract (None = latest)
            max_bulk_age_days: Max age for bulk data (default: 180 days)

        Returns:
            Tuple of (value, fiscal_year, fiscal_period)
            - value: Metric value or None if not found
            - fiscal_year: Fiscal year (int) or None
            - fiscal_period: Fiscal period (str) like 'Q3', 'FY' or None

        Example Usage:
            # Get latest revenue using hybrid strategy
            revenue, fy, fp = extractor.get_metric_with_hybrid_strategy(
                symbol='NEE',
                cik='753308',
                metric_tag='Revenues'
            )
            # Returns: (29500000000.0, 2024, 'Q3')

            # Get specific quarter revenue
            revenue, fy, fp = extractor.get_metric_with_hybrid_strategy(
                symbol='NEE',
                cik='753308',
                metric_tag='Revenues',
                fiscal_year=2024,
                fiscal_period='Q2'
            )
        """
        try:
            # TIER 1: Try bulk tables FIRST (if fresh enough)
            from utils.sec_data_strategy import get_fiscal_period_strategy

            strategy = get_fiscal_period_strategy()
            bulk_age = strategy._check_bulk_data_age(cik)

            # Determine if bulk data is fresh enough to trust
            use_bulk = bulk_age is not None and bulk_age <= max_bulk_age_days

            if use_bulk:
                # Try to get from bulk tables
                try:
                    from dao.sec_bulk_dao import get_sec_bulk_dao

                    bulk_dao = get_sec_bulk_dao()

                    if fiscal_year and fiscal_period:
                        # Get specific quarter from bulk
                        metrics = bulk_dao.fetch_financial_metrics(symbol, fiscal_year, fiscal_period)
                    else:
                        # Get latest quarter from bulk
                        latest_fy, latest_fp, _ = strategy._get_from_bulk_tables(cik)
                        if latest_fy and latest_fp:
                            metrics = bulk_dao.fetch_financial_metrics(symbol, latest_fy, latest_fp)
                        else:
                            metrics = None

                    if metrics:
                        # Map XBRL tag to database field
                        # This is simplified - real implementation needs tag mapper
                        tag_to_field = {
                            "Revenues": "total_revenue",
                            "Assets": "total_assets",
                            "Liabilities": "total_liabilities",
                            "StockholdersEquity": "stockholders_equity",
                            "NetIncomeLoss": "net_income",
                            # Add more mappings as needed
                        }

                        field_name = tag_to_field.get(metric_tag)
                        if field_name and field_name in metrics:
                            value = metrics.get(field_name)
                            fy = metrics.get("fiscal_year")
                            fp = metrics.get("fiscal_period")

                            if value is not None:
                                logger.debug(
                                    f"✓ Bulk table HIT for {symbol} {metric_tag}: "
                                    f"{value} (FY:{fy} FP:{fp}, age:{bulk_age:.0f} days)"
                                )
                                return (value, fy, fp)

                    logger.debug(
                        f"Bulk table MISS for {symbol} {metric_tag} "
                        f"(age:{bulk_age:.0f} days, threshold:{max_bulk_age_days}). "
                        f"Will try JSON API..."
                    )

                except Exception as e:
                    logger.debug(f"Bulk table query failed for {symbol}: {e}. Falling back to JSON API.")

            # TIER 2: Fallback to JSON API (CompanyFacts)
            logger.debug(
                f"Using JSON API for {symbol} {metric_tag} "
                f"(bulk_age:{bulk_age if bulk_age else 'N/A'}, stale_threshold:{max_bulk_age_days})"
            )

            facts_data = self.get_company_facts(symbol)
            if not facts_data or "facts" not in facts_data:
                return (None, None, None)

            us_gaap = facts_data["facts"].get("us-gaap", {})
            if not us_gaap or metric_tag not in us_gaap:
                return (None, None, None)

            # Extract from JSON API
            concept = us_gaap[metric_tag]
            units = concept.get("units", {})
            usd_data = units.get("USD", [])

            if not usd_data:
                return (None, None, None)

            # If specific fiscal period requested, filter to that period
            if fiscal_year and fiscal_period:
                matching_entries = [
                    entry
                    for entry in usd_data
                    if entry.get("fy") == fiscal_year
                    and entry.get("fp") == fiscal_period
                    and entry.get("form") in ["10-Q", "10-K", "20-F"]
                ]

                if matching_entries:
                    # Sort by filed date (newest first) and take most recent
                    matching_entries.sort(key=lambda x: x.get("filed", ""), reverse=True)
                    latest = matching_entries[0]
                    value = latest.get("val")
                    fy = latest.get("fy")
                    fp = latest.get("fp")

                    logger.debug(f"✓ JSON API HIT for {symbol} {metric_tag} " f"{fiscal_year}-{fiscal_period}: {value}")
                    return (value, fy, fp)
                else:
                    logger.debug(f"JSON API MISS for {symbol} {metric_tag} " f"{fiscal_year}-{fiscal_period}")
                    return (None, None, None)
            else:
                # Get latest value using existing helper
                value, fy, fp = self._get_latest_value_with_period(usd_data)
                logger.debug(f"✓ JSON API HIT for {symbol} {metric_tag} (latest): " f"{value} (FY:{fy} FP:{fp})")
                return (value, fy, fp)

        except Exception as e:
            logger.error(f"Error in hybrid metric extraction for {symbol} {metric_tag}: {e}")
            return (None, None, None)

    def extract_financial_metrics(self, symbol: str) -> Dict:
        """
        Extract key financial metrics for ratio calculations.

        NOW USES HYBRID STRATEGY: Tries DERA bulk tables first (fast), falls back to JSON API (fresh).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with extracted financial metrics
        """
        try:
            # Get CIK for hybrid strategy
            from utils.ticker_cik_mapper import TickerCIKMapper

            mapper = TickerCIKMapper()
            cik = mapper.get_cik(symbol)

            if not cik:
                logger.warning(f"No CIK found for {symbol}. Will attempt JSON API only.")

            facts_data = self.get_company_facts(symbol)
            if not facts_data:
                return self._empty_metrics()

            us_gaap = facts_data["facts"].get("us-gaap", {})

            # Check if we have flattened cache data (no us-gaap structure)
            if not us_gaap:
                logger.debug(f"Detected flattened cache for {symbol}")

                # Extract fiscal period from flattened cache
                fiscal_year, fiscal_period = self._extract_period_from_cache(facts_data)

                # If fiscal periods are None/invalid, try to get raw SEC data and extract
                if fiscal_year is None or fiscal_period is None:
                    logger.info(
                        f"Flattened cache missing fiscal periods for {symbol}. "
                        f"Attempting to extract from raw SEC data..."
                    )

                    # Try to get raw SEC data (force database/API lookup)
                    raw_facts = self.get_company_facts(symbol, force_refresh=False)

                    if raw_facts and "facts" in raw_facts:
                        raw_us_gaap = raw_facts["facts"].get("us-gaap", {})

                        if raw_us_gaap:
                            # Extract fiscal period from Revenues in raw data
                            for concept in ["Revenues", "SalesRevenueNet"]:
                                if concept in raw_us_gaap:
                                    units = raw_us_gaap[concept].get("units", {})
                                    usd_data = units.get("USD", [])
                                    if usd_data:
                                        _, fy, fp = self._get_latest_value_with_period(usd_data)
                                        if fy and fp:
                                            fiscal_year, fiscal_period = fy, fp
                                            logger.info(
                                                f"Extracted fiscal period from raw SEC data: "
                                                f"{fiscal_year}-{fiscal_period}"
                                            )
                                            break

                # Return flattened data with period fields added
                return {
                    **facts_data.get("facts", {}),
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "symbol": symbol,
                }

            # Import tag mapper for comprehensive tag coverage (ALL metrics)
            from utils.xbrl_tag_aliases import XBRLTagAliasMapper

            tag_mapper = XBRLTagAliasMapper()

            # STEP 1: Determine latest fiscal period FIRST (before extracting any metrics)
            # This ensures ALL metrics come from the SAME fiscal period
            logger.info(f"🎯 STEP 1: Determining latest fiscal period for {symbol}...")
            fiscal_year, fiscal_period, filed_date = self._determine_latest_fiscal_period(
                symbol=symbol, us_gaap=us_gaap, cik=cik
            )

            if not fiscal_year or not fiscal_period:
                logger.warning(
                    f"❌ Could not determine fiscal period for {symbol}. "
                    f"Will attempt extraction without period scoping (may result in mixed periods)."
                )
            else:
                logger.info(
                    f"✅ Determined fiscal period for {symbol}: {fiscal_year}-{fiscal_period} "
                    f"(filed: {filed_date or 'N/A'})"
                )

            # STEP 2: Extract ALL metrics for the determined fiscal period
            logger.info(
                f"🎯 STEP 2: Extracting all metrics for {symbol} period: {fiscal_year}-{fiscal_period} "
                f"(filed: {filed_date or 'unknown'})"
            )

            # Helper function to get metric value with HYBRID STRATEGY (bulk + API)
            # NOW PERIOD-SCOPED: Will extract for the determined fiscal period
            def get_metric(concept_names, prefer_annual: bool = True, canonical_name: str = None) -> Optional[float]:
                """
                Get metric value with HYBRID STRATEGY (tries bulk tables first, then JSON API).

                Args:
                    concept_names: Single string or list of strings for fallback field names
                    prefer_annual: Whether to prefer annual values over quarterly
                    canonical_name: Canonical metric name for tag mapper lookup (if provided, overrides concept_names)

                Returns:
                    Extracted value or None if not found
                """
                # If canonical_name provided, use tag mapper to get all aliases
                if canonical_name:
                    concept_names = tag_mapper.get_xbrl_aliases(canonical_name)
                    if not concept_names:
                        # Fallback to original if mapper doesn't have mapping
                        concept_names = [canonical_name]

                # Convert single string to list for consistent handling
                if isinstance(concept_names, str):
                    concept_names = [concept_names]

                # NEW: Try hybrid strategy if we have a CIK
                # PERIOD-SCOPED: Pass the determined fiscal period to ensure consistency
                if cik and fiscal_year and fiscal_period:
                    # STRICT PERIOD MATCHING: Try each tag alias for the SAME fiscal period
                    for concept_name in concept_names:
                        value, fy, fp = self.get_metric_with_hybrid_strategy(
                            symbol=symbol,
                            cik=cik,
                            metric_tag=concept_name,
                            fiscal_year=fiscal_year,  # Pass determined period
                            fiscal_period=fiscal_period,  # Pass determined period
                        )
                        if value is not None and fy == fiscal_year and fp == fiscal_period:
                            logger.debug(
                                f"✓ Metric extracted for {symbol} {concept_name}: {value} "
                                f"(period: {fiscal_year}-{fiscal_period}, source: hybrid)"
                            )
                            return value

                    # If hybrid strategy failed for all tag variants, log warning
                    logger.warning(
                        f"⚠️  No data found for {symbol} metric {canonical_name or concept_names[0]} "
                        f"in period {fiscal_year}-{fiscal_period} (tried {len(concept_names)} tag variants)"
                    )
                    return None

                # Fallback: No CIK or no period determined - get latest available
                logger.debug(
                    f"No period scoping for {symbol} (cik={cik}, period={fiscal_year}-{fiscal_period}). "
                    f"Getting latest available value."
                )
                for concept_name in concept_names:
                    concept = us_gaap.get(concept_name, {})
                    units = concept.get("units", {})

                    # Try USD first, then fallback to other units
                    usd_data = units.get("USD", [])
                    if usd_data:
                        value = self._get_latest_value(usd_data, prefer_annual)
                        if value is not None:
                            return value

                    # Try other currency units if USD not available
                    for unit_name, unit_data in units.items():
                        if unit_data:
                            value = self._get_latest_value(unit_data, prefer_annual)
                            if value is not None:
                                return value

                return None

            # Extract Balance Sheet metrics using tag mapper
            assets = get_metric(None, canonical_name="total_assets")
            assets_current = get_metric(None, canonical_name="current_assets")
            liabilities = get_metric(None, canonical_name="total_liabilities")
            liabilities_current = get_metric(None, canonical_name="current_liabilities")
            equity = get_metric(None, canonical_name="stockholders_equity")

            # Extract Income Statement metrics using tag mapper
            # Revenue extraction now uses pre-determined fiscal period (no separate period extraction)
            revenues = get_metric(None, canonical_name="revenues")
            net_income = get_metric(None, canonical_name="net_income")
            gross_profit = get_metric(None, canonical_name="gross_profit")
            operating_income = get_metric(None, canonical_name="operating_income")

            # Extract Cash and Cash Flow metrics using tag mapper
            cash_and_equivalents = get_metric(None, canonical_name="cash_and_equivalents")
            operating_cash_flow = get_metric(None, canonical_name="operating_cash_flow")
            capital_expenditures = get_metric(None, canonical_name="capital_expenditures")

            # Calculate Free Cash Flow (Operating CF - CapEx)
            free_cash_flow = None
            if operating_cash_flow is not None and capital_expenditures is not None:
                free_cash_flow = operating_cash_flow - abs(capital_expenditures)  # CapEx is usually negative
            elif operating_cash_flow is not None:
                # If no CapEx data, use operating CF as approximation
                free_cash_flow = operating_cash_flow

            # Inventory and receivables using tag mapper
            inventory = get_metric(None, canonical_name="inventory")
            accounts_receivable = get_metric(None, canonical_name="accounts_receivable")

            # Debt metrics using tag mapper
            long_term_debt = get_metric(None, canonical_name="long_term_debt")
            debt_current = get_metric(None, canonical_name="short_term_debt")

            # Calculate total debt (current + long-term)
            total_debt = 0.0
            if long_term_debt:
                total_debt += long_term_debt
            if debt_current:
                total_debt += debt_current

            # Cost of revenue for gross margin (use canonical name for tag mapper)
            cost_of_revenue = get_metric(None, canonical_name="cost_of_revenue")

            return {
                # Balance Sheet
                "assets": assets,
                "assets_current": assets_current,
                "liabilities": liabilities,
                "liabilities_current": liabilities_current,
                "equity": equity,
                "long_term_debt": long_term_debt,
                "debt_current": debt_current,
                "total_debt": total_debt,
                "cash_and_equivalents": cash_and_equivalents,
                "inventory": inventory,
                "accounts_receivable": accounts_receivable,
                # Income Statement
                "revenues": revenues,
                "net_income": net_income,
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "cost_of_revenue": cost_of_revenue,
                # Cash Flow Statement (ENHANCED)
                "operating_cash_flow": operating_cash_flow,
                "capital_expenditures": capital_expenditures,
                "free_cash_flow": free_cash_flow,
                # Metadata
                "symbol": symbol,
                "cik": facts_data.get("cik"),
                "company_name": facts_data.get("company_name"),
                "data_date": facts_data.get("fetched_at"),
                "source": facts_data.get("source", "unknown"),  # Track data source
                # Fiscal Period (for period-based caching)
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
            }

        except Exception as e:
            logger.error(f"Error extracting financial metrics for {symbol}: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "assets": None,
            "assets_current": None,
            "liabilities": None,
            "liabilities_current": None,
            "equity": None,
            "long_term_debt": None,
            "debt_current": None,
            "total_debt": None,
            "cash_and_equivalents": None,
            "inventory": None,
            "accounts_receivable": None,
            "revenues": None,
            "net_income": None,
            "gross_profit": None,
            "operating_income": None,
            "cost_of_revenue": None,
            "operating_cash_flow": None,
            "capital_expenditures": None,
            "free_cash_flow": None,
            "symbol": None,
            "cik": None,
            "company_name": None,
            "data_date": None,
            "fiscal_year": None,
            "fiscal_period": None,
        }

    def calculate_financial_ratios(self, symbol: str, current_price: Optional[float] = None) -> Dict:
        """
        Calculate financial ratios from extracted metrics.

        Args:
            symbol: Stock ticker symbol
            current_price: Current stock price for P/S ratio (optional)

        Returns:
            Dictionary with calculated financial ratios
        """
        metrics = self.extract_financial_metrics(symbol)

        # Helper function for safe division
        def safe_divide(numerator, denominator, default=0.0):
            if numerator is None or denominator is None or denominator == 0:
                return default
            return numerator / denominator

        # Calculate gross profit if not directly available
        gross_profit_calculated = metrics["gross_profit"]
        if not gross_profit_calculated and metrics["revenues"] and metrics["cost_of_revenue"]:
            gross_profit_calculated = metrics["revenues"] - metrics["cost_of_revenue"]

        # Calculate ratios
        ratios = {
            # Liquidity Ratios
            "current_ratio": safe_divide(metrics["assets_current"], metrics["liabilities_current"]),
            "quick_ratio": 0.0,  # Calculate below with inventory exclusion
            # Leverage Ratios
            "debt_to_equity": safe_divide(metrics["total_debt"], metrics["equity"]),
            "debt_to_assets": safe_divide(metrics["total_debt"], metrics["assets"]),
            # Profitability Ratios
            "roe": (
                safe_divide(metrics["net_income"], metrics["equity"]) * 100
                if metrics["net_income"] and metrics["equity"]
                else 0.0
            ),
            "roa": (
                safe_divide(metrics["net_income"], metrics["assets"]) * 100
                if metrics["net_income"] and metrics["assets"]
                else 0.0
            ),
            "gross_margin": (
                safe_divide(gross_profit_calculated, metrics["revenues"]) * 100
                if gross_profit_calculated and metrics["revenues"]
                else 0.0
            ),
            "operating_margin": (
                safe_divide(metrics["operating_income"], metrics["revenues"]) * 100
                if metrics["operating_income"] and metrics["revenues"]
                else 0.0
            ),
            "net_margin": (
                safe_divide(metrics["net_income"], metrics["revenues"]) * 100
                if metrics["net_income"] and metrics["revenues"]
                else 0.0
            ),
            # Valuation Ratios
            "price_to_sales": (
                safe_divide(current_price, metrics["revenues"]) if current_price and metrics["revenues"] else 0.0
            ),
            # Metadata
            "symbol": symbol,
            "data_date": metrics["data_date"],
            "raw_metrics": metrics,  # Include raw metrics for reference
        }

        # Calculate quick ratio (current assets - inventory) / current liabilities
        if metrics["assets_current"] and metrics["liabilities_current"]:
            quick_assets = metrics["assets_current"]
            if metrics["inventory"]:
                quick_assets -= metrics["inventory"]
            ratios["quick_ratio"] = safe_divide(quick_assets, metrics["liabilities_current"])

        return ratios


def get_sec_companyfacts_extractor() -> SECCompanyFactsExtractor:
    """
    Get SEC CompanyFacts extractor instance (3-table architecture)

    NOTE: This extractor reads from cache ONLY (no API calls).
    SEC API calls are handled by SEC Agent.

    Returns:
        SECCompanyFactsExtractor instance
    """
    return SECCompanyFactsExtractor()
