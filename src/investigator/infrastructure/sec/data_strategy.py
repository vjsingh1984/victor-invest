"""
SEC Data Strategy - 2-Tier Fiscal Period Detection

Priority 1: Bulk-loaded tables (sec_sub_data) - Authoritative source
Priority 2: CompanyFacts API - Fallback when bulk data unavailable

Author: InvestiGator Team
Date: 2025-11-02
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _fiscal_period_to_int(fp: str) -> int:
    """
    Convert fiscal period string to integer for chronological sorting.

    Ensures proper descending order: FY → Q4 → Q3 → Q2 → Q1 (most recent to oldest)

    Args:
        fp: Fiscal period string ('FY', 'Q4', 'Q3', 'Q2', 'Q1')

    Returns:
        Integer: FY=5, Q4=4, Q3=3, Q2=2, Q1=1, unknown=0

    Examples:
        >>> _fiscal_period_to_int('FY')  # 5
        >>> _fiscal_period_to_int('Q4')  # 4
        >>> _fiscal_period_to_int('Q1')  # 1
    """
    if not fp:
        return 0

    fp_upper = str(fp).upper().strip()

    if fp_upper == "FY":
        return 5  # Annual filing (most recent within a fiscal year)
    elif fp_upper.startswith("Q"):
        try:
            return int(fp_upper[1])  # Extract quarter number: Q4=4, Q3=3, Q2=2, Q1=1
        except (ValueError, IndexError):
            return 0

    return 0


class SECDataStrategy:
    """
    Implements 2-tier strategy for fiscal period detection:

    TIER 1 (Preferred): Bulk-loaded SEC DERA tables
    - sec_sub_data: Submission metadata (fy, fp, filed, adsh)
    - sec_num_data: Tag values per ADSH
    - sec_pre_data: Presentation linkbase
    - sec_tag_data: Tag definitions

    TIER 2 (Fallback): SEC CompanyFacts API
    - When bulk data >90 days old or unavailable
    - Direct API call for latest filings
    """

    def __init__(self, engine):
        self.engine = engine

    def get_latest_fiscal_period(self, symbol: str, cik: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Get latest fiscal period using 2-tier strategy

        Args:
            symbol: Stock ticker
            cik: CIK number (as string, will convert to int)

        Returns:
            Tuple of (fiscal_year, fiscal_period, adsh)
            e.g., (2025, 'Q2', '0000320193-25-000057')
        """
        # TIER 1: Try bulk-loaded tables first (authoritative)
        fy, fp, adsh = self._get_from_bulk_tables(cik)

        if fy and fp:
            # Check if data is fresh enough (< 90 days old)
            age_days = self._check_bulk_data_age(cik)

            if age_days is not None and age_days < 90:
                logger.info(
                    f"Using bulk-loaded data for {symbol}: {fy}-{fp} " f"({age_days:.0f} days old, ADSH: {adsh})"
                )
                return (fy, fp, adsh)
            else:
                logger.warning(
                    f"Bulk data for {symbol} is stale ({age_days:.0f} days old). "
                    f"Will attempt CompanyFacts API as fallback."
                )

        # TIER 2: Fallback to CompanyFacts API
        logger.info(f"Bulk data unavailable or stale for {symbol}. " f"Using CompanyFacts API fallback.")

        # Note: CompanyFacts API extraction handled by caller
        # Return None to signal caller to use API
        return (None, None, None)

    def _get_from_bulk_tables(self, cik: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Query sec_sub_data for latest filed quarter

        Args:
            cik: CIK number (string or int)

        Returns:
            Tuple of (fiscal_year, fiscal_period, adsh)
        """
        try:
            # Convert CIK to integer for bulk tables
            cik_int = int(cik) if isinstance(cik, str) else cik

            query = text(
                """
                SELECT fy, fp, filed, period, adsh, form
                FROM sec_sub_data
                WHERE cik = :cik
                AND form IN ('10-Q', '10-K', '20-F')
                ORDER BY filed DESC
                LIMIT 1
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, {"cik": cik_int}).fetchone()

            if result:
                logger.debug(
                    f"Found filing in bulk tables: FY:{result.fy} FP:{result.fp} "
                    f"Filed:{result.filed} Form:{result.form} ADSH:{result.adsh}"
                )
                return (result.fy, result.fp, result.adsh)

            logger.debug(f"No filings found in bulk tables for CIK {cik}")
            return (None, None, None)

        except Exception as e:
            logger.warning(f"Error querying bulk tables for CIK {cik}: {e}")
            return (None, None, None)

    def _check_bulk_data_age(self, cik: str) -> Optional[float]:
        """
        Check age of most recent filing in bulk tables

        Args:
            cik: CIK number

        Returns:
            Age in days, or None if not found
        """
        try:
            cik_int = int(cik) if isinstance(cik, str) else cik

            query = text(
                """
                SELECT filed
                FROM sec_sub_data
                WHERE cik = :cik
                AND form IN ('10-Q', '10-K', '20-F', '6-K')
                ORDER BY filed DESC
                LIMIT 1
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, {"cik": cik_int}).fetchone()

            if result and result.filed:
                filed_date = result.filed
                age = datetime.now().date() - filed_date
                return age.days

            return None

        except Exception as e:
            logger.debug(f"Error checking bulk data age for CIK {cik}: {e}")
            return None

    def get_8_quarters_hybrid(self, symbol: str, cik: str, max_bulk_age_days: int = 180) -> List[Dict]:
        """
        ALWAYS return 8 quarters using hybrid strategy

        Strategy: Mix bulk table data (fast) with API data (fresh)
        - 6-8 quarters from bulk tables (if fresh enough)
        - 0-2 quarters from API (fill gaps for freshness)

        This ensures consistent 8-quarter data for trend analysis while
        minimizing API calls and maximizing speed.

        Args:
            symbol: Stock ticker
            cik: CIK number
            max_bulk_age_days: Max age for bulk data (default: 180 days)

        Returns:
            List of exactly 8 quarter dicts, sorted by filed date (newest first)

        Example Results:
            Scenario 1 (bulk fresh <90 days):
              - Quarters 1-8: All from bulk tables
              - API calls: 0

            Scenario 2 (bulk moderately stale 90-180 days):
              - Quarters 1-6: From bulk tables
              - Quarters 7-8: From API
              - API calls: 2

            Scenario 3 (bulk very stale >180 days):
              - Quarters 1-4: From bulk tables (if available)
              - Quarters 5-8: From API
              - API calls: 4
        """
        try:
            # Step 1: Get as many quarters as possible from bulk tables
            bulk_quarters = self.get_multiple_quarters(symbol, cik, num_quarters=8)

            # Step 2: Check freshness of bulk data
            bulk_age = self._check_bulk_data_age(cik)

            # Step 3: Determine if we need API supplementation
            use_bulk_quarters = []

            if bulk_age is not None and bulk_age <= max_bulk_age_days:
                # Bulk data fresh enough - use what we have
                use_bulk_quarters = bulk_quarters[:8]  # Take up to 8

                logger.info(
                    f"Using {len(use_bulk_quarters)} quarters from bulk tables for {symbol} "
                    f"(age: {bulk_age:.0f} days)"
                )
            else:
                # Bulk data moderately stale - use tiered strategy
                # Key insight: Historical quarters (6+ months old) are STABLE and unlikely to change
                # Only recent quarters may have restatements - fetch those from API for freshness

                if bulk_age <= 270:  # ~9 months - use bulk for historical data
                    # Sort bulk quarters chronologically (most recent first: 2025-FY, 2025-Q4, 2025-Q3, etc.)
                    # Uses fiscal_period_to_int() for proper FY/Q4/Q3/Q2/Q1 ordering
                    try:
                        sorted_bulk = sorted(
                            bulk_quarters,
                            key=lambda q: (q["fiscal_year"], _fiscal_period_to_int(q.get("fiscal_period", ""))),
                            reverse=True,
                        )
                    except Exception:
                        sorted_bulk = bulk_quarters  # Fallback to unsorted if sorting fails

                    if bulk_age <= 180:
                        # Scenario: Bulk is reasonably fresh (90-180 days)
                        # Strategy: Use most bulk (7 quarters), fetch latest 1 from API
                        # Rationale: Latest quarter may have restatements, older quarters are stable
                        use_bulk_quarters = sorted_bulk[1:8]  # Skip most recent, use next 7
                        logger.info(
                            f"Bulk data moderately stale ({bulk_age:.0f} days, threshold: 90 days). "
                            f"Using {len(use_bulk_quarters)} historical quarters from bulk, "
                            f"will fetch latest quarter from API for freshness."
                        )
                    else:
                        # Scenario: Bulk is stale (180-270 days / ~6-9 months)
                        # Strategy: Use bulk for older quarters (6), fetch recent 2 from API
                        # Rationale: Recent 2 quarters may have updates, older 6 are stable
                        use_bulk_quarters = sorted_bulk[2:8]  # Skip 2 most recent, use next 6
                        logger.info(
                            f"Bulk data stale ({bulk_age:.0f} days, threshold: 180 days). "
                            f"Using {len(use_bulk_quarters)} historical quarters from bulk, "
                            f"will fetch latest 2 quarters from API."
                        )
                else:
                    # Scenario: Bulk is very stale (>270 days / ~9 months)
                    # Strategy: Use bulk only for oldest quarters (4), fetch recent 4 from API
                    # Rationale: >9 months old = potentially outdated for recent periods
                    try:
                        sorted_bulk = sorted(
                            bulk_quarters,
                            key=lambda q: (q["fiscal_year"], _fiscal_period_to_int(q.get("fiscal_period", ""))),
                            reverse=True,
                        )
                        use_bulk_quarters = sorted_bulk[4:8]  # Use oldest 4 from bulk
                    except Exception:
                        use_bulk_quarters = bulk_quarters[4:8] if len(bulk_quarters) >= 8 else bulk_quarters[:]

                    logger.info(
                        f"Bulk data very stale ({bulk_age:.0f} days, threshold: 270 days). "
                        f"Using {len(use_bulk_quarters)} oldest historical quarters from bulk, "
                        f"will fetch {8 - len(use_bulk_quarters)} recent quarters from API."
                    )

            # Step 4: Fill gap with API quarters if needed
            needed_quarters = 8 - len(use_bulk_quarters)

            if needed_quarters > 0:
                logger.info(f"Need {needed_quarters} additional quarters from API for {symbol}")

                # Fetch from CompanyFacts API using clean architecture
                api_quarters = []
                try:
                    from investigator.infrastructure.database.db import get_db_manager
                    from investigator.infrastructure.sec.companyfacts_extractor import SECCompanyFactsExtractor

                    db_manager = get_db_manager()
                    extractor = SECCompanyFactsExtractor(db_engine=db_manager.engine)
                    api_data = extractor.get_company_facts(symbol)

                    if api_data and "facts" in api_data:
                        us_gaap = api_data["facts"].get("us-gaap", {})

                        if us_gaap and "Revenues" in us_gaap:
                            # Extract fiscal periods from Revenues data
                            revenues_units = us_gaap["Revenues"].get("units", {}).get("USD", [])

                            # Filter to SEC filings (including foreign 20-F/6-K)
                            quarterly_filings = [
                                entry
                                for entry in revenues_units
                                if entry.get("form") in ["10-Q", "10-K", "20-F", "6-K"]
                            ]

                            # Sort by filed date (newest first) and get latest N
                            quarterly_filings.sort(key=lambda x: x.get("filed", ""), reverse=True)

                            # Get the needed quarters
                            for entry in quarterly_filings[: needed_quarters * 2]:  # Get extra to filter
                                # Skip if we already have this quarter from bulk
                                fy = entry.get("fy")
                                fp = entry.get("fp")

                                # Check if already in bulk quarters
                                already_have = any(
                                    q["fiscal_year"] == fy and q["fiscal_period"] == fp for q in use_bulk_quarters
                                )

                                if not already_have and len(api_quarters) < needed_quarters:
                                    api_quarters.append(
                                        {
                                            "fiscal_year": fy,
                                            "fiscal_period": fp,
                                            "adsh": entry.get("accn", "api_unknown"),
                                            "filed": entry.get("filed"),
                                            "period_end": entry.get("end"),
                                            "form": entry.get("form"),
                                        }
                                    )

                            logger.info(
                                f"Retrieved {len(api_quarters)} unique quarters from CompanyFacts API for {symbol}"
                            )

                except Exception as e:
                    logger.warning(f"CompanyFacts API fetch failed for {symbol}: {e}")

                # Combine: bulk (historical) + API (recent)
                combined_quarters = use_bulk_quarters + api_quarters

                # Sort chronologically by fiscal year and period (newest first)
                # Uses the same sorting logic as bulk quarters for consistency
                combined_quarters.sort(
                    key=lambda q: (q.get("fiscal_year", 0), _fiscal_period_to_int(q.get("fiscal_period", ""))),
                    reverse=True,
                )

                logger.info(
                    f"Hybrid strategy: {len(use_bulk_quarters)} bulk + {len(api_quarters)} API "
                    f"= {len(combined_quarters[:8])} total quarters for {symbol}"
                )

                return combined_quarters[:8]  # Return List[Dict] - fundamental_agent will convert to QuarterlyData

            # Step 5: Return exactly 8 quarters (or less if not enough data exists)
            return use_bulk_quarters[:8]  # Return List[Dict] - fundamental_agent will convert to QuarterlyData

        except Exception as e:
            logger.error(f"Error in hybrid 8-quarter retrieval for {symbol}: {e}")
            return []

    def get_multiple_quarters(
        self, symbol: str, cik: str, num_quarters: int = 8, include_fy: bool = True
    ) -> List[Dict]:
        """
        Get data for last N quarters from bulk tables

        CRITICAL: Handles different fiscal year calendars
        - Some companies file FY (10-K) in Q1 (Dec fiscal year end)
        - Others file FY in Q2 (Mar fiscal year end)
        - Others file FY in Q4 (Sep fiscal year end like AAPL)

        This function retrieves by FILED date, not fiscal period,
        so it automatically handles any fiscal calendar.

        Args:
            symbol: Stock ticker
            cik: CIK number
            num_quarters: Number of filings to retrieve (default: 8)
            include_fy: Include annual filings (10-K) (default: True)

        Returns:
            List of dicts, each containing:
            {
                'fiscal_year': int,
                'fiscal_period': str (Q1/Q2/Q3/Q4/FY),
                'adsh': str,
                'filed': date,
                'period_end': date,
                'form': str (10-Q or 10-K)
            }

        Example for AAPL (Sep fiscal year end):
            Latest filings by FILED date (not fiscal period):
            1. 2025-Q2 filed 2025-05-02 (10-Q)
            2. 2025-Q1 filed 2025-01-31 (10-Q)
            3. 2024-FY filed 2024-11-01 (10-K) ← FY filed in Q4
            4. 2024-Q3 filed 2024-08-02 (10-Q)

        Example for company with Dec fiscal year end:
            1. 2025-Q2 filed 2025-08-01 (10-Q)
            2. 2025-Q1 filed 2025-05-01 (10-Q)
            3. 2024-FY filed 2025-02-15 (10-K) ← FY filed in Q1
            4. 2024-Q3 filed 2024-11-01 (10-Q)
        """
        try:
            cik_int = int(cik) if isinstance(cik, str) else cik

            # Build form filter based on include_fy
            if include_fy:
                form_filter = "AND form IN ('10-Q', '10-K', '20-F')"
            else:
                form_filter = "AND form = '10-Q'"

            query = text(
                f"""
                SELECT fy, fp, filed, period, adsh, form
                FROM sec_sub_data
                WHERE cik = :cik
                {form_filter}
                ORDER BY filed DESC
                LIMIT :limit
            """
            )

            with self.engine.connect() as conn:
                results = conn.execute(query, {"cik": cik_int, "limit": num_quarters}).fetchall()

            # Detect fiscal_year_end from FY periods in results
            fiscal_year_end = None
            for row in results:
                if row.fp == "FY" and row.period:
                    try:
                        fy_end_date = datetime.strptime(str(row.period), "%Y-%m-%d")
                        fiscal_year_end = f"-{fy_end_date.month:02d}-{fy_end_date.day:02d}"
                        logger.debug(f"[Q1 Fix] {symbol}: Detected fiscal_year_end = {fiscal_year_end} from FY period")
                        break  # Use first FY found
                    except Exception as e:
                        logger.warning(f"[Q1 Fix] {symbol}: Failed to parse FY period date: {e}")

            quarters = []
            for row in results:
                fiscal_year = row.fy
                fiscal_period = row.fp

                # CRITICAL FIX: Adjust fiscal_year for ALL quarterly periods in non-calendar fiscal years
                # Q1/Q2/Q3/Q4 can cross calendar year boundary. If period_end is after fiscal_year_end,
                # the quarter belongs to the NEXT fiscal year.
                # Examples: ORCL (FY ends May 31): Q2 (Nov), Q3 (Feb) → fiscal_year = period_year + 1
                if fiscal_period in ["Q1", "Q2", "Q3", "Q4"] and fiscal_year_end and row.period:
                    try:
                        period_end_date = datetime.strptime(str(row.period), "%Y-%m-%d")
                        # Extract month/day from fiscal_year_end (format: '-MM-DD')
                        fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split("-"))

                        # Check if period_end is after fiscal_year_end
                        if (period_end_date.month > fy_end_month) or (
                            period_end_date.month == fy_end_month and period_end_date.day > fy_end_day
                        ):
                            original_fy = fiscal_year
                            fiscal_year += 1
                            logger.debug(
                                f"[Fiscal Year Adjustment] {symbol} {fiscal_period} ending {row.period}: "
                                f"Adjusted fiscal_year from {original_fy} to {fiscal_year} "
                                f"(fiscal year ends {fiscal_year_end})"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[Fiscal Year Adjustment] {symbol}: Failed to adjust {fiscal_period} fiscal year: {e}"
                        )

                quarters.append(
                    {
                        "fiscal_year": fiscal_year,
                        "fiscal_period": fiscal_period,
                        "adsh": row.adsh,
                        "filed": row.filed,
                        "period_end": row.period,
                        "form": row.form,
                    }
                )

            logger.info(
                f"Retrieved {len(quarters)} filings for {symbol} from bulk tables "
                f"(sorted by filed date, handles any fiscal calendar)"
            )

            return quarters

        except Exception as e:
            logger.error(f"Error retrieving multi-quarter data for {symbol}: {e}")
            return []

    def get_complete_fiscal_year(self, symbol: str, cik: str, fiscal_year: int) -> List[Dict]:
        """
        Get all filings for a complete fiscal year (Q1, Q2, Q3, Q4, FY)

        Handles companies with different fiscal year end dates:
        - Retrieves all filings where fy = fiscal_year
        - Automatically includes FY filing regardless of WHEN it was filed

        Args:
            symbol: Stock ticker
            cik: CIK number
            fiscal_year: The fiscal year to retrieve (e.g., 2024)

        Returns:
            List of dicts for ALL filings in that fiscal year
            Typically: [Q1, Q2, Q3, Q4, FY] but FY timing varies by company
        """
        try:
            cik_int = int(cik) if isinstance(cik, str) else cik

            query = text(
                """
                SELECT fy, fp, filed, period, adsh, form
                FROM sec_sub_data
                WHERE cik = :cik
                AND fy = :fiscal_year
                AND form IN ('10-Q', '10-K', '20-F')
                ORDER BY
                    CASE
                        WHEN fp = 'Q1' THEN 1
                        WHEN fp = 'Q2' THEN 2
                        WHEN fp = 'Q3' THEN 3
                        WHEN fp = 'Q4' THEN 4
                        WHEN fp = 'FY' THEN 5
                        ELSE 6
                    END
            """
            )

            with self.engine.connect() as conn:
                results = conn.execute(query, {"cik": cik_int, "fiscal_year": fiscal_year}).fetchall()

            # Detect fiscal_year_end from FY period in results
            fiscal_year_end = None
            for row in results:
                if row.fp == "FY" and row.period:
                    try:
                        fy_end_date = datetime.strptime(str(row.period), "%Y-%m-%d")
                        fiscal_year_end = f"-{fy_end_date.month:02d}-{fy_end_date.day:02d}"
                        logger.debug(f"[Q1 Fix] {symbol}: Detected fiscal_year_end = {fiscal_year_end} from FY period")
                        break  # Use first FY found
                    except Exception as e:
                        logger.warning(f"[Q1 Fix] {symbol}: Failed to parse FY period date: {e}")

            quarters = []
            for row in results:
                fy = row.fy
                fp = row.fp

                # CRITICAL FIX: Adjust fiscal_year for ALL quarterly periods in non-calendar fiscal years
                # Q1/Q2/Q3/Q4 can cross calendar year boundary. If period_end is after fiscal_year_end,
                # the quarter belongs to the NEXT fiscal year.
                if fp in ["Q1", "Q2", "Q3", "Q4"] and fiscal_year_end and row.period:
                    try:
                        period_end_date = datetime.strptime(str(row.period), "%Y-%m-%d")
                        fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split("-"))

                        if (period_end_date.month > fy_end_month) or (
                            period_end_date.month == fy_end_month and period_end_date.day > fy_end_day
                        ):
                            original_fy = fy
                            fy += 1
                            logger.debug(
                                f"[Fiscal Year Adjustment] {symbol} {fp} ending {row.period}: "
                                f"Adjusted fiscal_year from {original_fy} to {fy} "
                                f"(fiscal year ends {fiscal_year_end})"
                            )
                    except Exception as e:
                        logger.warning(f"[Fiscal Year Adjustment] {symbol}: Failed to adjust {fp} fiscal year: {e}")

                quarters.append(
                    {
                        "fiscal_year": fy,
                        "fiscal_period": fp,
                        "adsh": row.adsh,
                        "filed": row.filed,
                        "period_end": row.period,
                        "form": row.form,
                    }
                )

            logger.info(
                f"Retrieved complete FY{fiscal_year} for {symbol}: "
                f"{len(quarters)} filings ({[q['fiscal_period'] for q in quarters]})"
            )

            return quarters

        except Exception as e:
            logger.error(f"Error retrieving FY{fiscal_year} data for {symbol}: {e}")
            return []

    def get_num_data_for_adsh(self, adsh: str, tags: List[str]) -> Dict[str, float]:
        """
        Get specific tag values for a filing (ADSH)

        Args:
            adsh: Accession number
            tags: List of tags to retrieve (e.g., ['Revenues', 'Assets'])

        Returns:
            Dict mapping tag -> value (total company values only, not segment breakdowns)
        """
        try:
            # Build query for multiple tags
            # CRITICAL FIX (2025-12-28): Filter for total company values only
            # SEC filings contain both total company (segments=NULL) and segment breakdowns
            # Without this filter, arbitrary segment values could be picked (e.g., JNJ Q2 2024
            # was returning $0.17B product-level revenue instead of $22.45B total revenue)
            query = text(
                """
                SELECT tag, value, qtrs
                FROM sec_num_data
                WHERE adsh = :adsh
                AND tag = ANY(:tags)
                AND qtrs IN (0, 1)  -- 0=point-in-time, 1=quarterly
                AND (segments IS NULL OR segments = '')  -- Total company values only
                ORDER BY ddate DESC, value DESC  -- Prefer most recent, then largest value
            """
            )

            with self.engine.connect() as conn:
                results = conn.execute(query, {"adsh": adsh, "tags": tags}).fetchall()

            tag_values = {}
            for row in results:
                if row.tag not in tag_values:  # Use first (most recent, largest) value
                    tag_values[row.tag] = float(row.value)

            return tag_values

        except Exception as e:
            logger.error(f"Error retrieving num data for ADSH {adsh}: {e}")
            return {}


# Convenience function
def get_fiscal_period_strategy(engine=None):
    """Get SECDataStrategy instance"""
    if engine is None:
        from investigator.infrastructure.database.db import get_db_manager

        engine = get_db_manager().engine
    return SECDataStrategy(engine)
