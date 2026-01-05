#!/usr/bin/env python3
"""
SEC Data Processor - Extract quarterly data from raw SEC API responses

This module processes RAW SEC CompanyFacts API data (us-gaap structure) and extracts
flattened quarterly/annual financial data for fast querying and analysis.

Part of 3-table architecture:
  sec_companyfacts_raw (input) → SECDataProcessor → sec_companyfacts_processed (output)

Uses CanonicalKeyMapper for sector-aware XBRL tag resolution with automatic
fallback chains and derived metric calculation.

Author: InvestiGator Team
Date: 2025-11-03
"""

import copy
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text

# Import FiscalPeriodService for centralized fiscal period handling
from investigator.domain.services.fiscal_period_service import get_fiscal_period_service

# Keep canonical_key_mapper and industry_classifier in utils as they're shared across the system
from investigator.infrastructure.sec.canonical_mapper import get_canonical_mapper

# Import MetricExtractionOrchestrator for SOLID-based metric extraction
from investigator.infrastructure.sec.metric_extraction import MetricExtractionOrchestrator
from utils.industry_classifier import classify_company

logger = logging.getLogger(__name__)


class SECDataProcessor:
    """
    Extract and process quarterly/annual financial data from raw SEC API responses

    Uses CanonicalKeyMapper for sector-aware XBRL tag resolution with automatic
    fallback chains and derived metric calculation.
    """

    # Keep at least 12 quarters + 4 annual filings when available
    MAX_FILINGS_TO_PROCESS = 16

    def _build_adsh_fiscal_lookup(self, cik: str, engine) -> Dict[str, Dict]:
        """
        Build authoritative ADSH→fiscal period lookup from sec_sub_data bulk table.

        This uses the PROVEN bulk table structure as the source of truth for
        fiscal year/period labeling, avoiding CompanyFacts JSON ambiguities.

        Args:
            cik: Company CIK (numeric string, e.g., '0001792789')
            engine: SQLAlchemy engine for database access

        Returns:
            Dict mapping ADSH to {fy, fp, period, filed}

        Example:
            {
                '0001628280-23-035973': {'fy': 2023, 'fp': 'Q3', 'period': '2023-09-30', 'filed': '2023-11-01'},
                ...
            }
        """
        from sqlalchemy import text

        # Clean CIK (remove leading zeros for numeric comparison)
        cik_int = int(cik.lstrip("0")) if cik else None
        if not cik_int:
            return {}

        try:
            with engine.begin() as conn:
                query = text(
                    """
                    SELECT adsh, fy, fp, period, filed
                    FROM sec_sub_data
                    WHERE cik = :cik
                      AND form IN ('10-K', '10-Q')
                    ORDER BY period DESC
                    LIMIT 50
                """
                )

                result = conn.execute(query, {"cik": cik_int})

                lookup = {}
                for row in result:
                    lookup[row.adsh] = {"fy": row.fy, "fp": row.fp, "period": str(row.period), "filed": str(row.filed)}

                logger.info(f"Built ADSH lookup from bulk table: {len(lookup)} entries for CIK {cik}")
                return lookup

        except Exception as e:
            logger.warning(f"Failed to build ADSH lookup from bulk table: {e}")
            return {}

    def _correct_period_end_dates(self, filings: Dict, us_gaap: Dict, cik: str):
        """
        Correct period_end dates for each filing using multiple strategies:
        1. Scan actual extracted data to find most common period_end for this ADSH+fy+fp
        2. Parse frame field (e.g., "CY2023Q3") to derive period_end
        3. Fallback to bulk table lookup

        This fixes the issue where filing discovery captures wrong period_end
        (often FY-end instead of actual quarter-end).

        Args:
            filings: Dict of filings to correct
            us_gaap: SEC us-gaap JSON structure
            cik: Company CIK
        """
        from collections import Counter

        # Build bulk table lookup as fallback
        bulk_lookup = self._build_adsh_fiscal_lookup(cik, self.engine) if self.engine else {}

        for filing_key, filing in filings.items():
            adsh = filing["adsh"]
            fy = filing["fiscal_year"]
            fp = filing["fiscal_period"]
            current_period_end = filing.get("period_end_date")

            # Strategy 1: Collect all period_end dates for this ADSH+fy+fp from actual data
            period_ends = []
            for tag_name, tag_data in us_gaap.items():
                units = tag_data.get("units", {})
                for unit_type, unit_data in units.items():
                    for entry in unit_data:
                        if entry.get("accn") == adsh and entry.get("fy") == fy and entry.get("fp") == fp:
                            end_date = entry.get("end")
                            if end_date and end_date not in period_ends:
                                period_ends.append(end_date)

            # Find most common period_end (mode)
            if period_ends:
                period_end_counts = Counter(period_ends)
                most_common_period_end = period_end_counts.most_common(1)[0][0]

                if most_common_period_end != current_period_end:
                    logger.info(
                        f"📅 Corrected period_end for {adsh[:15]}... FY{fy} {fp}: "
                        f"{current_period_end} → {most_common_period_end} "
                        f"(found in {period_end_counts[most_common_period_end]}/{len(period_ends)} entries)"
                    )
                    filing["period_end_date"] = most_common_period_end
                    continue

            # Strategy 2: Fallback to bulk table
            if adsh in bulk_lookup:
                bulk_period = bulk_lookup[adsh]["period"]
                if bulk_period != current_period_end:
                    logger.info(
                        f"📅 Corrected period_end for {adsh[:15]}... FY{fy} {fp} from bulk table: "
                        f"{current_period_end} → {bulk_period}"
                    )
                    filing["period_end_date"] = bulk_period
                    continue

            # Strategy 3: Derive from fiscal period using detected fiscal year-end
            # Detect company's fiscal year-end from FY filings, then calculate quarter-ends
            if fp in ["Q1", "Q2", "Q3"] and fy:
                # Find the fiscal year-end month by looking at FY filings
                fy_period_end = None
                for other_filing in filings.values():
                    if other_filing.get("fiscal_period") == "FY" and other_filing.get("fiscal_year") == fy:
                        fy_period_end = other_filing.get("period_end_date")
                        break
                    # Also check prior year FY if current year not found
                    if other_filing.get("fiscal_period") == "FY" and other_filing.get("fiscal_year") == fy - 1:
                        fy_period_end = other_filing.get("period_end_date")

                if fy_period_end:
                    from datetime import datetime

                    from dateutil.relativedelta import relativedelta

                    # Parse the FY end date to get the fiscal year-end month/day
                    fy_date = datetime.strptime(fy_period_end, "%Y-%m-%d")

                    # Calculate quarter-ends by going back from FY-end
                    # Q4 ends on FY-end, Q3 ends 3 months before, Q2 ends 6 months before, Q1 ends 9 months before
                    quarters_back = {"Q3": 3, "Q2": 6, "Q1": 9}
                    months_back = quarters_back.get(fp, 0)

                    if months_back:
                        # Start from the FY date with the correct fiscal year
                        base_date = fy_date.replace(year=fy)
                        quarter_end = base_date - relativedelta(months=months_back)
                        derived_period_end = quarter_end.strftime("%Y-%m-%d")

                        if derived_period_end != current_period_end:
                            logger.info(
                                f"📅 Corrected period_end for {adsh[:15]}... FY{fy} {fp} from fiscal pattern: "
                                f"{current_period_end} → {derived_period_end} "
                                f"(derived from FY-end {fy_period_end})"
                            )
                            filing["period_end_date"] = derived_period_end

    def _compute_quarter_end_dates(self, filings: Dict, symbol: str):
        """
        Compute correct period_end_date for quarterly periods based on fiscal year-end pattern.

        SEC companyfacts data often has incorrect period_end_date values for quarterly periods
        (showing FY-end date instead of actual quarter-end). This function derives the correct
        quarter-end dates from the detected fiscal year-end pattern.

        For example, if FY ends September 28:
        - Q4 ends September 28 (same as FY)
        - Q3 ends June 28 (3 months before)
        - Q2 ends March 28 (6 months before)
        - Q1 ends December 28 of prior calendar year (9 months before)

        Args:
            filings: Dict of filings to correct
            symbol: Stock symbol for logging
        """
        from datetime import datetime

        from dateutil.relativedelta import relativedelta

        # Find the fiscal year-end pattern from FY periods
        fy_periods = [
            (f["fiscal_year"], f.get("period_end_date"))
            for f in filings.values()
            if f.get("fiscal_period") == "FY" and f.get("period_end_date")
        ]

        if not fy_periods:
            logger.warning(f"{symbol}: No FY periods found, cannot compute quarter-end dates")
            return

        # Use the most recent FY to detect the fiscal year-end month/day
        fy_periods.sort(key=lambda x: x[0], reverse=True)
        latest_fy_year, latest_fy_end = fy_periods[0]

        try:
            fy_end_date = datetime.strptime(latest_fy_end, "%Y-%m-%d")
            fy_end_month = fy_end_date.month
            fy_end_day = fy_end_date.day
        except (ValueError, TypeError):
            logger.warning(f"{symbol}: Invalid FY period_end_date format: {latest_fy_end}")
            return

        logger.info(f"{symbol}: Detected fiscal year-end pattern: month={fy_end_month}, day={fy_end_day}")

        # Quarters are offset from FY-end by: Q3=-3mo, Q2=-6mo, Q1=-9mo, Q4=0mo
        # FY itself should have period_end = fiscal_year + (month, day)
        quarter_offsets = {"Q1": 9, "Q2": 6, "Q3": 3, "Q4": 0, "FY": 0}
        corrections_made = 0

        for filing in filings.values():
            fp = filing.get("fiscal_period")
            fy = filing.get("fiscal_year")
            current_period_end = filing.get("period_end_date")

            if fp not in quarter_offsets or not fy:
                continue

            months_back = quarter_offsets[fp]

            # Compute the expected period-end date
            # For FY and Q4: use fiscal_year directly with month/day pattern
            # For Q1/Q2/Q3: offset from the fiscal year-end
            try:
                fy_end_for_year = datetime(fy, fy_end_month, fy_end_day)
            except ValueError:
                # Handle edge case like Feb 29
                fy_end_for_year = datetime(fy, fy_end_month, min(fy_end_day, 28))

            if months_back > 0:
                expected_end = fy_end_for_year - relativedelta(months=months_back)
            else:
                expected_end = fy_end_for_year

            expected_period_end = expected_end.strftime("%Y-%m-%d")

            # Only correct if different
            if expected_period_end != current_period_end:
                logger.debug(
                    f"{symbol} FY{fy} {fp}: Correcting period_end_date " f"{current_period_end} → {expected_period_end}"
                )
                filing["period_end_date"] = expected_period_end
                corrections_made += 1

        if corrections_made > 0:
            logger.info(f"{symbol}: Corrected period_end_date for {corrections_made} periods (including FY)")

    def _detect_fiscal_year_end(self, company_facts_data: Dict, symbol: str) -> Optional[str]:
        """
        Detect company's fiscal year end from FY periods only.

        UPDATED: Now delegates to FiscalPeriodService for centralized fiscal period handling.

        Args:
            company_facts_data: Raw CompanyFacts JSON
            symbol: Company symbol (for logging)

        Returns:
            Fiscal year end in '-MM-DD' format (e.g., '-12-31')
            None if cannot determine
        """
        try:
            # Use centralized FiscalPeriodService for fiscal year end detection
            fiscal_period_service = get_fiscal_period_service()
            return fiscal_period_service.detect_fiscal_year_end(company_facts_data)
        except Exception as e:
            logger.error(f"[Fiscal Year End] {symbol}: Error detecting fiscal year end: {e}")
            return None

    def _compute_fiscal_year_start(self, fiscal_year: int, fiscal_year_end: str) -> str:
        """
        Compute fiscal year start date from fiscal year end.

        Args:
            fiscal_year: e.g., 2024
            fiscal_year_end: e.g., '-12-31' or '-02-29'

        Returns:
            Fiscal year start date, e.g., '2024-01-01'
        """
        from calendar import isleap
        from datetime import date, timedelta

        # Extract month and day
        month_str, day_str = fiscal_year_end[1:].split("-")
        month = int(month_str)
        day = int(day_str)

        # LEAP YEAR HANDLING: Adjust Feb 29 to Feb 28 for non-leap years
        if month == 2 and day == 29:
            if not isleap(fiscal_year):
                logger.warning(f"[Fiscal Year Start] Adjusted Feb 29 to Feb 28 for " f"non-leap year {fiscal_year}")
                day = 28

        try:
            # Construct fiscal year end date
            fy_end = date(fiscal_year, month, day)
        except ValueError as e:
            logger.error(
                f"[Fiscal Year Start] Invalid date for FY {fiscal_year} " f"with fiscal_year_end={fiscal_year_end}: {e}"
            )
            # Fallback: Use Jan 1 of fiscal year
            return date(fiscal_year, 1, 1).strftime("%Y-%m-%d")

        # Fiscal year start = 1 day after previous fiscal year end
        # (which is ~365 days before this fiscal year end)
        fy_start = fy_end - timedelta(days=364)  # Use 364 to land on next day

        return fy_start.strftime("%Y-%m-%d")

    def _score_period_for_selection(self, entry: Dict, fiscal_year_start: Optional[str], symbol: str) -> int:
        """
        Score a period entry to prefer quarterly over YTD versions.

        Higher score = better candidate for selection.

        Args:
            entry: Period entry from CompanyFacts
            fiscal_year_start: Detected fiscal year start date (e.g., '2024-01-01')
            symbol: Company symbol (for logging)

        Returns:
            Score (higher is better)
        """
        score = 0
        start_date = entry.get("start")
        end_date = entry.get("end")
        fp = entry.get("fp")
        form = entry.get("form", "")
        duration_days = entry.get("duration_days", 0)

        # FY periods: Always highest priority
        if fp == "FY":
            score += 200
            if form in ["10-K", "10-K/A"]:
                score += 100
            logger.debug(f"[SCORE] {symbol} {fp} {end_date}: FY period, score={score}")
            return score

        # Q1 periods: No YTD ambiguity (always starts at fiscal year start)
        if fp == "Q1":
            score += 150
            logger.debug(f"[SCORE] {symbol} {fp} {end_date}: Q1 period, score={score}")
            return score

        # Q2, Q3 periods: Check for YTD vs quarterly
        # YTD detection: start_date matches fiscal_year_start
        is_ytd = False
        if fiscal_year_start and start_date == fiscal_year_start and fp in ["Q2", "Q3"]:
            is_ytd = True
            logger.debug(
                f"[SCORE] {symbol} {fp} {end_date}: YTD detected "
                f"(start={start_date} matches fiscal_year_start={fiscal_year_start})"
            )
        elif duration_days >= 120 and fp in ["Q2", "Q3"]:
            # Fallback: Use duration if fiscal_year_start not available
            is_ytd = True
            logger.debug(
                f"[SCORE] {symbol} {fp} {end_date}: YTD detected by duration " f"({duration_days} days >= 120)"
            )

        # Scoring
        if not is_ytd:
            score += 200  # STRONGLY prefer quarterly
            logger.debug(
                f"[SCORE] {symbol} {fp} {end_date}: Quarterly version "
                f"(start={start_date}, duration={duration_days}), score=+200"
            )
        else:
            score += 0  # YTD gets no bonus
            logger.debug(
                f"[SCORE] {symbol} {fp} {end_date}: YTD version "
                f"(start={start_date}, duration={duration_days}), score=+0"
            )

        # Additional criteria
        if form in ["10-Q", "10-K"]:
            score += 50

        if entry.get("filed"):
            score += 25

        logger.debug(f"[SCORE] {symbol} {fp} {end_date}: Final score={score}")
        return score

    @staticmethod
    def _determine_fiscal_year_from_end_date(period_end_date: str, fiscal_period: str) -> Optional[int]:
        """
        DEPRECATED: Use bulk table lookup instead (_build_adsh_fiscal_lookup).

        Determine actual fiscal year from period end date.

        This is a fallback heuristic when bulk table lookup is unavailable.

        Args:
            period_end_date: Period end date in YYYY-MM-DD format
            fiscal_period: Fiscal period (FY, Q1, Q2, Q3, Q4)

        Returns:
            Fiscal year derived from period_end_date year
        """
        if not period_end_date:
            return None

        try:
            end_date = datetime.strptime(period_end_date, "%Y-%m-%d")
            return end_date.year

        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse period_end_date '{period_end_date}': {e}")
            return None

    # Canonical keys to extract (replaces hardcoded FIELD_MAPPINGS)
    CANONICAL_KEYS_TO_EXTRACT = [
        # Income Statement
        "total_revenue",
        "net_income",
        "gross_profit",
        "operating_income",
        "cost_of_revenue",
        "research_and_development_expense",
        "selling_general_administrative_expense",
        "operating_expenses",
        "interest_expense",
        "income_tax_expense",
        "earnings_per_share",
        "earnings_per_share_diluted",
        # Balance Sheet - Assets
        "total_assets",
        "current_assets",
        "cash_and_equivalents",
        "accounts_receivable",
        "inventory",
        "property_plant_equipment",
        "accumulated_depreciation",
        "property_plant_equipment_net",
        "goodwill",
        "intangible_assets",
        "retained_earnings",
        "deferred_revenue",
        "accounts_payable",
        "accrued_liabilities",
        "treasury_stock",
        "other_comprehensive_income",
        "book_value",
        "book_value_per_share",
        "working_capital",
        # Balance Sheet - Liabilities
        "total_liabilities",
        "current_liabilities",
        "stockholders_equity",
        "preferred_stock_dividends",
        "common_stock_dividends",
        # Debt
        "long_term_debt",
        "short_term_debt",
        "total_debt",
        "net_debt",
        "financial_total_deposits",
        "financial_repo_borrowings",
        "financial_fhlb_borrowings",
        "financial_other_short_term_borrowings",
        # Cash Flow
        "operating_cash_flow",
        "capital_expenditures",
        "dividends_paid",
        "investing_cash_flow",
        "financing_cash_flow",
        "depreciation_amortization",
        "stock_based_compensation",
        # Shares Outstanding (for per-share valuations)
        "weighted_average_diluted_shares_outstanding",
        "shares_outstanding",
        # Market data
        "market_cap",
        "enterprise_value",
        # Derived metrics (automatically calculated by CanonicalKeyMapper)
        "free_cash_flow",
        "dividend_payout_ratio",
        "dividend_yield",
        "effective_tax_rate",
        "interest_coverage",
        "asset_turnover",
    ]

    def __init__(self, db_engine=None, sector: Optional[str] = None, industry: Optional[str] = None):
        """
        Initialize processor with database connection and canonical key mapper

        Args:
            db_engine: SQLAlchemy engine (optional, will create from config if not provided)
            sector: Company sector for sector-aware tag resolution (optional)
            industry: Company industry for industry-specific tag resolution (optional)
        """
        self.engine = db_engine
        if not self.engine:
            from investigator.infrastructure.database.db import get_db_manager

            self.engine = get_db_manager().engine

        # Initialize canonical key mapper for sector-aware XBRL tag resolution
        self.canonical_mapper = get_canonical_mapper()
        self.sector = sector
        self.industry = industry

        # Initialize MetricExtractionOrchestrator for SOLID-based extraction
        # Uses period_end date matching (reliable) instead of fy field (unreliable)
        self.metric_orchestrator = MetricExtractionOrchestrator(
            sector=sector,
            industry=industry,
            canonical_mapper=self.canonical_mapper,
            enable_audit=False,  # Disable audit trail in production for performance
        )

        logger.info(
            f"Initialized SECDataProcessor with CanonicalKeyMapper + MetricExtractionOrchestrator "
            f"(sector={sector or 'auto-detect'}, industry={industry or 'auto-detect'})"
        )

        # Synonyms used when persisting ratio-style metrics (now removed - using canonical names only)
        self._ratio_synonyms = {}

    @staticmethod
    def _mapping_has_direct_tags(mapping: Optional[Dict[str, Any]]) -> bool:
        """Return True when a canonical mapping exposes real XBRL tags (not purely derived)."""
        if not mapping:
            return False

        global_fallback = mapping.get("global_fallback") or []
        if global_fallback:
            return True

        sector_specific = mapping.get("sector_specific") or {}
        return any(tags for tags in sector_specific.values())

    def _detect_statement_qtrs(self, us_gaap: Dict, adsh: str, fiscal_year: int, fiscal_period: str) -> Tuple[int, int]:
        """
        Detect optimal qtrs values for income statement and cash flow statement.

        Strategy:
        1. Try to find individual quarter entries (qtrs=1, duration < 120 days)
        2. If not available, use YTD entries (qtrs=2/3, duration 120-270 days)
        3. Fallback to safe defaults if no entries found

        Args:
            us_gaap: SEC us-gaap JSON structure
            adsh: Accession number for filtering
            fiscal_year: Fiscal year for filtering
            fiscal_period: Fiscal period (Q1, Q2, Q3, FY)

        Returns:
            Tuple of (income_statement_qtrs, cash_flow_statement_qtrs)

        Examples:
            AAPL Q2 2024: (1, 2) - Income has individual, Cash Flow only YTD
            MSFT Q2 2024: (1, 1) - Both have individual quarters
        """
        # Q1 and FY always use individual/full year
        if fiscal_period == "Q1":
            return (1, 1)
        elif fiscal_period == "FY":
            return (4, 4)

        # For Q2/Q3, try to find individual quarter entries

        # Income Statement Tags (in priority order)
        income_tags = [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ]

        # Cash Flow Tags
        cashflow_tags = ["NetCashProvidedByUsedInOperatingActivities"]

        income_statement_qtrs = self._find_optimal_qtrs(us_gaap, income_tags, adsh, fiscal_year, fiscal_period)
        cash_flow_statement_qtrs = self._find_optimal_qtrs(us_gaap, cashflow_tags, adsh, fiscal_year, fiscal_period)

        logger.debug(
            f"Detected qtrs for {adsh[:10]}... FY:{fiscal_year} {fiscal_period}: "
            f"income_statement={income_statement_qtrs}, cash_flow_statement={cash_flow_statement_qtrs}"
        )

        return (income_statement_qtrs, cash_flow_statement_qtrs)

    def _enrich_share_counts(self, filing: Dict) -> None:
        """
        Ensure shares_outstanding is populated using diluted share counts when necessary.
        """
        data = filing.setdefault("data", {})
        if data.get("shares_outstanding") is None:
            diluted = data.get("weighted_average_diluted_shares_outstanding")
            if diluted is not None:
                data["shares_outstanding"] = diluted

    def _enrich_book_value_per_share(self, filing: Dict) -> None:
        """
        Derive book_value_per_share when equity and share counts exist but the metric is absent.
        """
        data = filing.setdefault("data", {})
        if data.get("book_value_per_share") is not None:
            return

        equity = data.get("stockholders_equity") or data.get("book_value")
        shares = data.get("shares_outstanding")
        if equity is None or shares in (None, 0):
            return

        data["book_value_per_share"] = equity / shares

    def _enrich_gross_and_cost_fields(self, filing: Dict) -> None:
        """
        Fill missing gross_profit or cost_of_revenue using available totals/operating metrics.
        """
        data = filing.setdefault("data", {})
        total_revenue = data.get("total_revenue")
        gross_profit = data.get("gross_profit")
        cost_of_revenue = data.get("cost_of_revenue")
        operating_income = data.get("operating_income")
        operating_expenses = data.get("operating_expenses")

        # If gross profit missing but we know total revenue and cost, compute it.
        if gross_profit is None and total_revenue is not None and cost_of_revenue is not None:
            gross_profit = total_revenue - cost_of_revenue
            data["gross_profit"] = gross_profit

        # If cost missing but we have revenue and gross profit, backfill.
        if cost_of_revenue is None and total_revenue is not None and gross_profit is not None:
            cost_of_revenue = total_revenue - gross_profit
            data["cost_of_revenue"] = cost_of_revenue

        # Use operating metrics when gross profit still unavailable.
        if gross_profit is None and operating_income is not None and operating_expenses is not None:
            gross_profit = operating_income + operating_expenses
            data["gross_profit"] = gross_profit

        # Recompute cost when we derived gross profit above.
        if cost_of_revenue is None and total_revenue is not None and gross_profit is not None:
            data["cost_of_revenue"] = total_revenue - gross_profit

    def _find_optimal_qtrs(
        self, us_gaap: Dict, tags: List[str], adsh: str, fiscal_year: int, fiscal_period: str
    ) -> int:
        """
        Find optimal qtrs value by checking for individual quarter availability.

        Returns:
            qtrs value: 1 (individual), 2 (Q2 YTD), 3 (Q3 YTD), or safe default
        """
        has_individual = False
        has_ytd = False

        for tag in tags:
            if tag not in us_gaap:
                continue

            usd_data = us_gaap[tag].get("units", {}).get("USD", [])

            for entry in usd_data:
                # Filter by ADSH and fiscal period
                # NOTE: We match by ADSH and fiscal_period only, NOT fiscal_year
                # This is because the processor's fiscal_year calculation may differ
                # from the JSON's fy field by +/- 1 year due to fiscal year end timing.
                # The ADSH uniquely identifies the filing, so fy matching is redundant.
                if entry.get("accn") != adsh:
                    continue
                if entry.get("fp") != fiscal_period:
                    continue

                # Check duration
                start = entry.get("start")
                end = entry.get("end")
                if not start or not end:
                    continue

                try:
                    start_date = datetime.strptime(start, "%Y-%m-%d")
                    end_date = datetime.strptime(end, "%Y-%m-%d")
                    days = (end_date - start_date).days

                    if days < 120:
                        has_individual = True
                    elif 120 <= days < 270:
                        has_ytd = True
                except ValueError:
                    continue

        # Prefer individual quarter
        if has_individual:
            return 1
        elif has_ytd:
            return 2 if fiscal_period == "Q2" else 3
        else:
            # Safe fallback
            return {"Q2": 2, "Q3": 3}.get(fiscal_period, 1)

    def _discover_all_period_entries(self, us_gaap: Dict, symbol: str) -> List[Dict]:
        """
        Discover ALL period entries from representative XBRL tags.

        Scans NetIncomeLoss and Revenues tags to find all unique (start, end, frame, fy, fp, accn, filed) combinations.
        This captures all filings including comparative data and amendments.

        Args:
            us_gaap: SEC us-gaap JSON structure
            symbol: Stock ticker for logging

        Returns:
            List of period entry dictionaries with start, end, frame, fy, fp, accn, filed, form
        """
        all_entries = []
        seen_entries = set()  # Track unique (accn, start, end) to avoid duplicates

        # Use representative tags that appear in most filings
        representative_tags = ["NetIncomeLoss", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]

        for tag_name in representative_tags:
            if tag_name not in us_gaap:
                continue

            usd_data = us_gaap[tag_name].get("units", {}).get("USD", [])

            for entry in usd_data:
                form = entry.get("form", "")
                if form not in ["10-K", "10-Q"]:
                    continue

                accn = entry.get("accn", "")
                start = entry.get("start", "")
                end = entry.get("end", "")

                # Skip if missing critical fields
                if not accn or not end:
                    continue

                # Create unique key to avoid duplicate entries from different tags
                entry_key = (accn, start, end)
                if entry_key in seen_entries:
                    continue

                seen_entries.add(entry_key)

                all_entries.append(
                    {
                        "start": start,
                        "end": end,
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                        "frame": entry.get("frame", ""),
                        "filed": entry.get("filed", ""),
                        "accn": accn,
                        "form": form,
                    }
                )

        logger.info(
            f"[ADSH Discovery] {symbol}: Found {len(all_entries)} unique period entries from {len(seen_entries)} (accn, start, end) combinations"
        )
        return all_entries

    def _select_best_entries_per_period(
        self, all_entries: List[Dict], symbol: str, company_facts_data: Dict
    ) -> List[Dict]:
        """
        Select best entry for each unique period using comprehensive scoring strategy.

        Strategy:
        1. Detect fiscal year end from CompanyFacts (only fp='FY' periods)
        2. Group by (end, fp) - same period_end can have YTD and quarterly versions
        3. For each period group, score entries:
           - FY periods: 200-300 points (highest)
           - Q1 periods: 150 points (no YTD ambiguity)
           - Quarterly (Q2/Q3): 200 points
           - YTD (Q2/Q3): 0 points
        4. Select highest scoring entry per group

        Args:
            all_entries: List of all discovered period entries
            symbol: Stock ticker for logging
            company_facts_data: Raw CompanyFacts JSON for fiscal year detection

        Returns:
            List of best entries (one per unique period)
        """
        from collections import defaultdict

        # Detect fiscal year end for scoring
        fiscal_year_end = self._detect_fiscal_year_end(company_facts_data, symbol)
        fiscal_year_starts = {}  # Cache fiscal year starts per fiscal year

        # Group by (end, fp) - same period_end can have YTD and quarterly versions
        # We'll prefer quarterly (shorter duration) over YTD within each group
        period_groups = defaultdict(list)
        for entry in all_entries:
            # Group by end date and fiscal period to catch YTD vs quarterly versions
            # that share the same end date but have different start dates
            period_key = (entry["end"], entry.get("fp"))
            period_groups[period_key].append(entry)

        best_entries = []
        comparative_data_filtered = 0

        for period_key, group in period_groups.items():
            end_str, fp = period_key

            # CRITICAL: Calculate duration for ALL entries (needed for fiscal_period determination later)
            # Previously only calculated for multi-candidate groups, causing single entries to have duration_days=None
            for entry in group:
                if entry["start"] and entry["end"]:
                    try:
                        start_date = datetime.strptime(entry["start"], "%Y-%m-%d")
                        end_date = datetime.strptime(entry["end"], "%Y-%m-%d")
                        entry["duration_days"] = (end_date - start_date).days
                    except ValueError:
                        entry["duration_days"] = 999  # Unknown
                else:
                    entry["duration_days"] = 999

            # Single entry: Use it directly (duration already calculated above)
            if len(group) == 1:
                best_entries.append(group[0])
                continue

            # Multiple candidates: Use scoring
            # Determine fiscal year for this group
            fy = group[0].get("fy")  # All entries in group should have same fy

            # Compute fiscal year start if not cached
            if fy and fy not in fiscal_year_starts and fiscal_year_end:
                fiscal_year_starts[fy] = self._compute_fiscal_year_start(fy, fiscal_year_end)

            fiscal_year_start = fiscal_year_starts.get(fy)

            # Score all candidates
            scored = [(self._score_period_for_selection(entry, fiscal_year_start, symbol), entry) for entry in group]
            scored.sort(key=lambda x: x[0], reverse=True)  # Highest score first

            # Filter 3: Validate fiscal year (reject if > 2 years from period_end)
            if end_str:
                try:
                    period_end_year = datetime.strptime(end_str, "%Y-%m-%d").year

                    # Find first entry with valid fiscal year
                    best_entry = None
                    best_score = None
                    for score, candidate in scored:
                        candidate_fy = candidate.get("fy")
                        if candidate_fy is None:
                            # No fy field, accept it (will derive from end date)
                            best_entry = candidate
                            best_score = score
                            break

                        # CRITICAL FIX: Q1/Q2/Q3 in non-calendar fiscal years can legitimately have fy = period_end_year + 1
                        # For non-calendar fiscal years, calculate expected fy based on fiscal year end month
                        # Example: ORCL Q2 ending Nov 2024 belongs to FY 2025 (fiscal year ends May 31)
                        # This is NOT comparative data - it's the correct fiscal year assignment

                        # Calculate expected fiscal_year for non-calendar fiscal years
                        expected_fy = None
                        if fiscal_year_end:
                            # Extract fiscal_year_end_month from fiscal_year_end (format: '-MM-DD')
                            try:
                                fiscal_year_end_month = int(fiscal_year_end.split("-")[1])
                                period_end_date = datetime.strptime(end_str, "%Y-%m-%d")
                                period_month = period_end_date.month

                                # Calculate expected fiscal_year:
                                # If period_month > fiscal_year_end_month, period is in NEXT fiscal year
                                if period_month > fiscal_year_end_month:
                                    expected_fy = period_end_year + 1
                                else:
                                    expected_fy = period_end_year
                            except (ValueError, IndexError, AttributeError):
                                # Parsing failed, skip expected_fy calculation
                                pass

                        # Accept entry if fiscal_year matches expected value for non-calendar fiscal years
                        if expected_fy is not None and candidate_fy == expected_fy:
                            best_entry = candidate
                            best_score = score
                            logger.info(
                                f"[ADSH Filter] {symbol}: ✅ Accepted {fp} with fy={candidate_fy} for period ending {end_str} "
                                f"(matches expected fy={expected_fy} for non-calendar fiscal year ending {fiscal_year_end})"
                            )
                            break

                        # Validate fy is reasonable (reject if 1+ years from period_end)
                        if abs(candidate_fy - period_end_year) >= 1:
                            comparative_data_filtered += 1
                            logger.info(
                                f"[ADSH Filter] {symbol}: ❌ Rejected {fp} entry with fy={candidate_fy} for period ending {end_str} "
                                f"(diff={abs(candidate_fy - period_end_year)} years, expected_fy={expected_fy}, likely comparative data)"
                            )
                        else:
                            best_entry = candidate
                            best_score = score
                            break

                    if not best_entry:
                        # All entries had invalid fy, use highest scored anyway
                        best_score, best_entry = scored[0]
                        logger.warning(
                            f"[ADSH Filter] {symbol}: All entries for period ending {end_str} had invalid fy, "
                            f"using highest scored entry: {best_entry.get('filed')}"
                        )

                except ValueError:
                    # Invalid date format, use highest scored
                    best_score, best_entry = scored[0]

                if best_entry:
                    best_entries.append(best_entry)

                    logger.info(
                        f"[ADSH Selection] {symbol} {fp} {end_str}: Selected entry with score {best_score} "
                        f"(start={best_entry.get('start')}, duration={best_entry.get('duration_days', 'N/A')} days)"
                    )

                    # Log rejected alternatives for debugging
                    if len(scored) > 1:
                        for score, entry in scored[1:]:
                            logger.debug(
                                f"[ADSH Selection] {symbol} {fp} {end_str}: Rejected entry with score {score} "
                                f"(start={entry.get('start')}, duration={entry.get('duration_days', 'N/A')} days)"
                            )
            else:
                # No end date, use highest scored
                best_score, best_entry = scored[0]
                best_entries.append(best_entry)

        logger.info(
            f"[ADSH Selection] {symbol}: Selected {len(best_entries)} best entries from {len(all_entries)} total entries "
            f"(filtered {comparative_data_filtered} comparative data)"
        )

        return best_entries

    def _enrich_debt_fields(self, filing: Dict) -> None:
        """
        Derive missing short-term debt, total debt, and net debt fields when underlying components exist.

        For financial institutions (banks), derives debt from deposits, repo borrowings, FHLB borrowings.
        For non-financial companies, derives from long-term and short-term debt components.
        """
        data = filing.setdefault("data", {})

        long_term = data.get("long_term_debt")
        short_term = data.get("short_term_debt")
        total_debt = data.get("total_debt")

        # Financial institution specific debt components
        deposits_total = data.get("financial_total_deposits")
        repo_borrowings = data.get("financial_repo_borrowings")
        fhlb_borrowings = data.get("financial_fhlb_borrowings")
        other_short_borrowings = data.get("financial_other_short_term_borrowings")

        # Derive short-term debt from financial components if available
        short_term_components = [value for value in (repo_borrowings, other_short_borrowings) if value is not None]
        if short_term is None and short_term_components:
            short_term = sum(short_term_components)
            data["short_term_debt"] = short_term

        # Derive total debt if both components exist
        if total_debt is None and long_term is not None and short_term is not None:
            total_debt = long_term + short_term
            data["total_debt"] = total_debt

        # Derive short-term debt when total and long-term are available
        if short_term is None and total_debt is not None and long_term is not None:
            derived_short = total_debt - long_term
            if abs(derived_short) > 1e-6:
                short_term = derived_short
                data["short_term_debt"] = short_term

        # If total debt missing but components available after derivation, fill it
        if total_debt is None and long_term is not None and short_term is not None:
            total_debt = long_term + short_term
            data["total_debt"] = total_debt

        # Financial institution heuristics for deposits/FHLB/other borrowings
        if total_debt is None:
            total_components = []
            if long_term is not None:
                total_components.append(long_term)
            if short_term is not None:
                total_components.append(short_term)
            if deposits_total is not None:
                total_components.append(deposits_total)
            if fhlb_borrowings is not None:
                total_components.append(fhlb_borrowings)

            if total_components:
                total_debt = sum(total_components)
                data["total_debt"] = total_debt

        # Derive net debt if possible
        if data.get("net_debt") is None and total_debt is not None:
            cash_equivalents = data.get("cash_and_equivalents")
            if cash_equivalents is None:
                cash_equivalents = data.get("cash")
            if cash_equivalents is not None:
                data["net_debt"] = total_debt - cash_equivalents

    def _extract_from_json_for_filing(
        self,
        canonical_key: str,
        us_gaap: Dict,
        adsh: str,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract a canonical key for a specific filing using SOLID-based MetricExtractionOrchestrator.

        STRATEGY (SOLID Multi-Level Fallback):
        1. PRIMARY: Use MetricExtractionOrchestrator with period_end date matching (MOST RELIABLE)
           - ByPeriodEndMatcher: Exact end date match (fixes STX fy=2027 bug)
           - ByDateRangeMatcher: Fuzzy date matching (±7 days tolerance)
           - ByFrameFieldMatcher: Uses CY2024Q3 frame field
           - ByAdshOnlyMatcher: Uses ADSH with duration filtering
           - ByAdshFyFpMatcher: Legacy fallback (unreliable fy field)

        2. FALLBACK: Legacy approach if orchestrator fails (backward compatibility)
           - Direct ADSH + fy/fp matching with duration-based entry selection

        Args:
            canonical_key: Financial concept to extract (e.g., 'total_revenue')
            us_gaap: SEC us-gaap JSON structure
            adsh: Accession number (filing identifier)
            fiscal_year: Fiscal year for filtering (optional, may be wrong in SEC data)
            fiscal_period: Fiscal period for filtering (optional)
            period_end: Period end date (MOST RELIABLE - use this for matching)

        Returns:
            Tuple of (value, source_tag) or (None, None) if not found
        """
        # ===================================================================
        # PRIMARY STRATEGY: Use MetricExtractionOrchestrator (period_end matching)
        # This fixes the STX bug where SEC's fy field is wrong (fy=2027 vs actual fy=2025)
        # ===================================================================
        if period_end:
            result = self.metric_orchestrator.extract(
                canonical_key=canonical_key,
                us_gaap=us_gaap,
                target_period_end=period_end,
                target_fiscal_year=fiscal_year,  # Passed but not primary filter
                target_fiscal_period=fiscal_period,
                target_adsh=adsh,
            )

            if result.success and result.value is not None:
                logger.debug(
                    f"✓ [Orchestrator] Extracted {canonical_key} = {result.value:,.0f} "
                    f"via {result.match_method.value} using tag '{result.source_tag}' "
                    f"(period_end={period_end}, confidence={result.confidence.value})"
                )
                return (result.value, result.source_tag)

            # Log orchestrator failure for debugging
            logger.debug(
                f"[Orchestrator] Failed to extract {canonical_key} for period_end={period_end}: " f"{result.error}"
            )

        # ===================================================================
        # FALLBACK STRATEGY: Legacy approach (ADSH + fy/fp matching)
        # Kept for backward compatibility when period_end is not available
        # ===================================================================
        mapping = self.canonical_mapper.mappings.get(canonical_key)
        if not mapping:
            return (None, None)

        fallback_tags = self.canonical_mapper.get_tags(canonical_key, sector=self.sector, industry=self.industry)
        expected_unit = mapping.get("unit", "USD")

        for tag_name in fallback_tags:
            if tag_name not in us_gaap:
                continue

            tag_data = us_gaap[tag_name]
            units = tag_data.get("units", {})
            unit_data = units.get(expected_unit, [])

            matching_entries = []
            for entry in unit_data:
                form = entry.get("form", "")
                if form not in ["10-K", "10-Q", "10-K/A", "10-Q/A"]:
                    continue

                entry_adsh = entry.get("accn", "")
                if entry_adsh != adsh:
                    continue

                # Filter by fiscal year/period (legacy - may be unreliable)
                if fiscal_year and entry.get("fy") != fiscal_year:
                    continue
                if fiscal_period and entry.get("fp") != fiscal_period:
                    continue

                matching_entries.append(entry)

            # Fallback: Try without ADSH filter if no matches
            if not matching_entries and (fiscal_year or fiscal_period):
                logger.debug(
                    f"[Legacy Fallback] No matches for {canonical_key} with ADSH {adsh[:10]}..., "
                    f"trying by fiscal_year={fiscal_year}, fiscal_period={fiscal_period}"
                )
                for entry in unit_data:
                    form = entry.get("form", "")
                    if form not in ["10-K", "10-Q", "10-K/A", "10-Q/A"]:
                        continue

                    if fiscal_year and entry.get("fy") != fiscal_year:
                        continue
                    if fiscal_period and entry.get("fp") != fiscal_period:
                        continue

                    matching_entries.append(entry)

            if not matching_entries:
                continue

            best_entry = self._select_best_entry(
                matching_entries,
                fiscal_period=fiscal_period,
                period_end=period_end,
                fiscal_year=fiscal_year,
            )

            if best_entry:
                value = best_entry.get("val")
                if value is not None:
                    duration_str = ""
                    start = best_entry.get("start")
                    end = best_entry.get("end")
                    if start and end:
                        try:
                            start_date = datetime.strptime(start, "%Y-%m-%d")
                            end_date = datetime.strptime(end, "%Y-%m-%d")
                            days = (end_date - start_date).days
                            duration_str = f", {days} days"
                        except ValueError:
                            pass

                    logger.debug(
                        f"✓ [Legacy] Extracted {canonical_key} = {value:,.0f} using tag '{tag_name}' "
                        f"(ADSH: {adsh[:10]}...{duration_str})"
                    )
                    return (value, tag_name)

        logger.debug(f"Could not extract {canonical_key} for ADSH {adsh[:10]}... (tried {len(fallback_tags)} tags)")
        return (None, None)

    def _select_best_entry(
        self,
        entries: List[Dict],
        fiscal_period: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Select the best entry from multiple matches based on duration.

        Preference order:
        1. Individual quarter (days < 120)
        2. YTD entry (120 <= days < 270)
        3. Annual entry (days >= 270)
        4. Entry without start/end dates (fallback)

        Args:
            entries: List of matching SEC data entries
            fiscal_period: Fiscal period (Q1/Q2/Q3/FY) for context

        Returns:
            Best entry to use, or None if no valid entry
        """

        def choose(candidates: List[Dict]) -> Optional[Dict]:
            if not candidates:
                return None

            if len(candidates) == 1:
                return candidates[0]

            individual_entries = []
            ytd_entries = []
            annual_entries = []
            unknown_entries = []

            for entry in candidates:
                start = entry.get("start")
                end = entry.get("end")

                if not start or not end:
                    unknown_entries.append(entry)
                    continue

                try:
                    start_date = datetime.strptime(start, "%Y-%m-%d")
                    end_date = datetime.strptime(end, "%Y-%m-%d")
                    days = (end_date - start_date).days

                    if days < 120:
                        individual_entries.append((entry, days))
                    elif 120 <= days < 270:
                        ytd_entries.append((entry, days))
                    else:
                        annual_entries.append((entry, days))
                except ValueError:
                    unknown_entries.append(entry)

            if individual_entries:
                individual_entries.sort(key=lambda x: x[1])
                selected = individual_entries[0][0]
                logger.debug(
                    "  → Selected individual quarter entry (%s days) over %s alternatives",
                    individual_entries[0][1],
                    max(len(candidates) - 1, 0),
                )
                return selected

            if ytd_entries:
                ytd_entries.sort(key=lambda x: x[1])
                selected = ytd_entries[0][0]
                logger.debug(
                    "  → Selected YTD entry (%s days) over %s alternatives",
                    ytd_entries[0][1],
                    max(len(candidates) - 1, 0),
                )
                return selected

            if annual_entries:
                annual_entries.sort(key=lambda x: x[1])
                selected = annual_entries[0][0]
                logger.debug(
                    "  → Selected annual entry (%s days) over %s alternatives",
                    annual_entries[0][1],
                    max(len(candidates) - 1, 0),
                )
                return selected

            if unknown_entries:
                # Prefer entries whose end matches period_end
                if period_end:
                    for entry in unknown_entries:
                        if entry.get("end") == period_end:
                            logger.debug("  → Selected unknown entry matching period_end %s", period_end)
                            return entry
                logger.debug("  → Selected entry without duration info (no start/end dates)")
                return unknown_entries[0]

            return None

        if not entries:
            return None

        period_end = period_end or None

        if period_end:
            end_matches = [entry for entry in entries if entry.get("end") == period_end]
            candidate = choose(end_matches)
            if candidate:
                return candidate

        return choose(entries)

    def process_raw_data(
        self,
        symbol: str,
        raw_data: Dict,
        raw_data_id: int,
        extraction_version: str = "1.0.0",
        persist: bool = True,
        current_price: Optional[float] = None,
    ) -> List[Dict]:
        """
        Extract all quarterly/annual filings from raw us-gaap structure

        Args:
            symbol: Stock ticker
            raw_data: Raw SEC API response with us-gaap structure
            raw_data_id: ID from sec_companyfacts_raw table (for lineage)
            extraction_version: Version of extraction logic (for re-processing tracking)
            persist: When False, skip writing results to sec_companyfacts_processed. Useful for dry runs
                     and verification workflows (default: True).
            current_price: Current stock price (optional). When provided, enables market cap calculation
                          (market_cap = shares_outstanding × current_price) for each filing.

        Returns:
            List of processed filing dicts (one per quarter/year)
        """
        try:
            if "facts" not in raw_data or "us-gaap" not in raw_data["facts"]:
                logger.error(f"Raw data for {symbol} missing us-gaap structure, cannot process")
                return []

            us_gaap = raw_data["facts"]["us-gaap"]
            cik = str(raw_data.get("cik", ""))
            entity_name = raw_data.get("entityName", "")

            # Auto-detect sector/industry if not provided in __init__
            if not self.sector or not self.industry:
                detected_sector, detected_industry = classify_company(symbol)
                self.sector = self.sector or detected_sector
                self.industry = self.industry or detected_industry

                if self.sector and self.industry:
                    logger.info(f"[SEC Processor] Auto-detected {symbol} industry: {self.sector}/{self.industry}")
                elif self.sector:
                    logger.info(f"[SEC Processor] Auto-detected {symbol} sector: {self.sector} (industry unknown)")
                else:
                    logger.warning(
                        f"[SEC Processor] Could not detect sector/industry for {symbol}, using generic XBRL tags"
                    )

            # Group all metrics by ADSH (accession number = unique filing identifier)
            filings = {}  # {adsh: {fiscal_year, fiscal_period, data}}

            logger.info(f"Processing raw SEC data for {symbol}: {len(us_gaap)} us-gaap tags")

            # PHASE 1: Discover ALL period entries using comprehensive strategy
            # Strategy: Scan representative tags to find all unique (start, end, frame, accn) combinations
            # This captures comparative filings and amendments
            all_period_entries = self._discover_all_period_entries(us_gaap, symbol)

            # PHASE 1.5: Select best entry for each period
            # Applies filtering: individual quarter preference, latest filed date, fy validation
            best_entries = self._select_best_entries_per_period(all_period_entries, symbol, raw_data)

            # Detect fiscal year end for Q1 fiscal year adjustment
            fiscal_year_end = self._detect_fiscal_year_end(raw_data, symbol)
            if fiscal_year_end:
                logger.info(f"[Fiscal Year End] {symbol}: Detected fiscal year end: {fiscal_year_end}")
            else:
                logger.warning(
                    f"[Fiscal Year End] {symbol}: Could not detect fiscal year end, Q1 fiscal year may be incorrect"
                )

            # PHASE 2: Create filings dict with DERIVED fiscal periods
            # IMPORTANT: Fiscal year/period are DERIVED from period_end_date and duration
            # NOT from unreliable fy/fp fields (which indicate filing document, not reporting period)
            for entry in best_entries:
                adsh = entry["accn"]

                # Derive actual fiscal year from period_end (not from fy field!)
                period_end_str = entry["end"]
                if not period_end_str:
                    continue

                try:
                    period_end_date = datetime.strptime(period_end_str, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid period_end_date format: {period_end_str}")
                    continue

                actual_fiscal_year = period_end_date.year

                # Derive fiscal period using fp field (authoritative after filtering comparative data)
                # After filtering out entries where abs(fy - period_end_year) >= 1, the fp field is trustworthy
                duration = entry.get("duration_days", 999)
                raw_fp = entry.get("fp", "")

                # Use fp from entry if available and valid (after comparative data filtering)
                # Full year filings or entries with duration >= 330 days are FY
                if raw_fp == "FY" or duration >= 330:
                    actual_fp = "FY"
                elif raw_fp in ["Q1", "Q2", "Q3", "Q4"]:
                    # Use the fp field from the entry (authoritative for non-comparative data)
                    # This handles edge cases like Oct 1-3 (Q3 ending on weekend) correctly
                    actual_fp = raw_fp
                else:
                    # Fallback: derive quarter from end month
                    # Only used if fp field is missing or invalid
                    month = period_end_date.month
                    if month <= 3:
                        actual_fp = "Q1"
                    elif month <= 6:
                        actual_fp = "Q2"
                    elif month <= 9:
                        actual_fp = "Q3"
                    else:
                        actual_fp = "Q4"

                # CRITICAL FIX: Adjust fiscal_year for ALL quarterly periods in non-calendar fiscal years
                # For non-calendar fiscal years, quarters can cross calendar year boundary.
                # If period_end is after fiscal_year_end, the quarter belongs to the NEXT fiscal year.
                # Examples:
                #   - ZS (fiscal year ends Jul 31): Q1 ending Oct 31, 2023 is FY2024
                #   - ORCL (fiscal year ends May 31): Q2 ending Nov 30, 2024 is FY2025
                #   - period_end_date.year gives calendar year, but fiscal_year calculation needed
                if fiscal_year_end and actual_fp in ["Q1", "Q2", "Q3", "Q4"]:
                    try:
                        # Extract month and day from fiscal_year_end (format: '-MM-DD')
                        fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split("-"))

                        # Check if period_end is after fiscal_year_end
                        # If so, the quarter belongs to the next fiscal year
                        if (period_end_date.month > fy_end_month) or (
                            period_end_date.month == fy_end_month and period_end_date.day > fy_end_day
                        ):
                            original_fy = actual_fiscal_year
                            actual_fiscal_year += 1
                            logger.debug(
                                f"[Fiscal Year Adjustment] {symbol} {actual_fp} ending {period_end_str}: "
                                f"Adjusted fiscal_year from {original_fy} to {actual_fiscal_year} "
                                f"(fiscal year ends {fiscal_year_end})"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[Fiscal Year Adjustment] {symbol}: Failed to adjust {actual_fp} fiscal year: {e}"
                        )

                filings[adsh] = {
                    "symbol": symbol.upper(),
                    "cik": cik,
                    "fiscal_year": actual_fiscal_year,  # ✅ DERIVED from end date
                    "fiscal_period": actual_fp,  # ✅ DERIVED from duration + month
                    "adsh": adsh,
                    "form_type": entry["form"],
                    "filed_date": entry["filed"],
                    "period_end_date": period_end_str,
                    "period_start_date": entry["start"],  # ✅ ADD THIS
                    "frame": entry["frame"],
                    "duration_days": duration,  # ✅ ADD THIS
                    "data": {},
                    "raw_data_id": raw_data_id,
                    "extraction_version": extraction_version,
                }

            logger.info(f"✅ Selected {len(filings)} best filings for {symbol} (frame/start/end-based, PIT preferred)")

            # PHASE 1b: Deduplicate by (period_start_date, period_end_date)
            # Prevents selecting multiple fiscal_period labels (Q2, Q3) for the same actual period
            logger.info(
                f"[ADSH Dedup] {symbol}: Deduplicating {len(filings)} filings by (period_start_date, period_end_date)"
            )
            period_map = {}
            duplicates_removed = 0

            for adsh, filing in filings.items():
                key = (filing["period_start_date"], filing["period_end_date"])
                if key not in period_map:
                    period_map[key] = filing
                else:
                    # Keep latest filed
                    if filing["filed_date"] > period_map[key]["filed_date"]:
                        logger.debug(
                            f"[ADSH Dedup] {symbol}: Replacing {period_map[key]['adsh']} ({period_map[key]['filed_date']}) "
                            f"with {filing['adsh']} ({filing['filed_date']}) for period {key}"
                        )
                        period_map[key] = filing
                        duplicates_removed += 1
                    else:
                        logger.debug(
                            f"[ADSH Dedup] {symbol}: Keeping {period_map[key]['adsh']} ({period_map[key]['filed_date']}) "
                            f"over {filing['adsh']} ({filing['filed_date']}) for period {key}"
                        )
                        duplicates_removed += 1

            # Replace filings with deduplicated entries
            filings = {f["adsh"]: f for f in period_map.values()}

            if duplicates_removed > 0:
                logger.info(
                    f"[ADSH Dedup] {symbol}: Removed {duplicates_removed} duplicate periods, "
                    f"final count: {len(filings)} unique filings"
                )
            else:
                logger.info(f"[ADSH Dedup] {symbol}: No duplicates found, {len(filings)} unique filings")

            # Populate current_price if provided (enables market cap calculation)
            if current_price is not None:
                for filing in filings.values():
                    filing.setdefault("data", {})["current_price"] = current_price

            # PHASE 2: Extract all canonical keys using CanonicalKeyMapper
            # Use sector-aware extraction with automatic fallback chains
            extracted_fields = set()

            for period_key, filing in filings.items():
                adsh = filing["adsh"]  # Extract adsh from filing dict
                for canonical_key in self.CANONICAL_KEYS_TO_EXTRACT:
                    mapping = self.canonical_mapper.mappings.get(canonical_key)
                    derived_enabled = mapping.get("derived", {}).get("enabled", False) if mapping else False
                    has_direct_tags = self._mapping_has_direct_tags(mapping)

                    # Skip purely-derived metrics until the derivation pass
                    if derived_enabled and not has_direct_tags:
                        continue

                    # Extract using CanonicalKeyMapper
                    value, source_tag = self._extract_from_json_for_filing(
                        canonical_key,
                        us_gaap,
                        adsh,
                        fiscal_year=filing["fiscal_year"],
                        fiscal_period=filing["fiscal_period"],
                        period_end=filing.get("period_end_date") or filing.get("period_end"),
                    )

                    if value is not None:
                        filing["data"][canonical_key] = value
                        extracted_fields.add(canonical_key)

                        # Log key metric extractions
                        if canonical_key in [
                            "operating_cash_flow",
                            "capital_expenditures",
                            "dividends_paid",
                            "total_revenue",
                            "net_income",
                        ]:
                            logger.debug(
                                f"✓ Extracted {canonical_key} = {value:,.0f} using tag '{source_tag}' "
                                f"(FY:{filing['fiscal_year']}, FP:{filing['fiscal_period']})"
                            )

            logger.info(
                f"✅ Extracted {len(extracted_fields)} unique canonical keys using CanonicalKeyMapper (sector={self.sector or 'global'})"
            )

            # PHASE 2.5: Compute correct quarter-end dates from fiscal year-end pattern
            # SEC data often has incorrect period_end_date for quarterly periods (showing FY-end instead)
            # This derives correct dates: Q1=-9mo, Q2=-6mo, Q3=-3mo, Q4=0mo from FY-end
            self._compute_quarter_end_dates(filings, symbol)

            # Ensure share counts are available before derived metrics so ratios can compute
            for filing in filings.values():
                self._enrich_share_counts(filing)

            # PHASE 3: Calculate derived metrics using CanonicalKeyMapper
            # Derived metrics include free_cash_flow, total_debt, and other calculated values
            processed_filings = []
            for period_key, filing in filings.items():
                adsh = filing["adsh"]  # Extract adsh from filing dict
                # Use CanonicalKeyMapper to calculate derived values
                for canonical_key in self.CANONICAL_KEYS_TO_EXTRACT:
                    mapping = self.canonical_mapper.mappings.get(canonical_key)
                    if not mapping or not mapping.get("derived", {}).get("enabled", False):
                        continue

                    # Skip if already extracted directly
                    if canonical_key in filing["data"] and filing["data"][canonical_key] is not None:
                        continue

                    # Calculate derived value
                    derived_value = self.canonical_mapper.calculate_derived_value(canonical_key, filing["data"])

                    if derived_value is not None:
                        filing["data"][canonical_key] = derived_value
                        logger.debug(
                            f"✓ Calculated derived metric {canonical_key} = {derived_value:,.2f} "
                            f"(FY:{filing['fiscal_year']}, FP:{filing['fiscal_period']})"
                        )

                # PHASE 3.5: Detect statement-specific qtrs values
                income_statement_qtrs, cash_flow_statement_qtrs = self._detect_statement_qtrs(
                    us_gaap, adsh, filing["fiscal_year"], filing["fiscal_period"]
                )
                filing["income_statement_qtrs"] = income_statement_qtrs
                filing["cash_flow_statement_qtrs"] = cash_flow_statement_qtrs

                logger.debug(
                    f"📊 Filing {adsh[:10]}... FY:{filing['fiscal_year']} {filing['fiscal_period']}: "
                    f"income_statement_qtrs={income_statement_qtrs}, cash_flow_statement_qtrs={cash_flow_statement_qtrs}"
                )

                # Enrich debt fields (derive missing debt metrics from components)
                self._enrich_debt_fields(filing)
                self._enrich_share_counts(filing)
                self._enrich_book_value_per_share(filing)
                self._enrich_gross_and_cost_fields(filing)

                processed_filings.append(filing)

            # PHASE 4: Normalize YTD to point-in-time and calculate ratios
            # Must be done after all filings are extracted so we can access previous periods

            # Make a deep copy of raw filings for normalization lookups
            # This prevents using already-normalized values when normalizing later periods
            raw_filings = copy.deepcopy(processed_filings)

            for filing in processed_filings:
                # Normalize YTD values to point-in-time BEFORE calculating ratios
                # Use raw_filings to look up previous periods (not yet normalized)
                filing["data"], income_normalized, cashflow_normalized = self._normalize_ytd_to_pit(
                    data=filing["data"],
                    income_qtrs=filing["income_statement_qtrs"],
                    cashflow_qtrs=filing["cash_flow_statement_qtrs"],
                    fiscal_period=filing["fiscal_period"],
                    fiscal_year=filing["fiscal_year"],
                    all_filings=raw_filings,  # Use raw (pre-normalization) filings
                    symbol=symbol,
                )

                # Once normalized, mark statements as point-in-time for downstream use
                if income_normalized and filing["fiscal_period"] in ["Q2", "Q3"]:
                    filing["income_statement_qtrs"] = 1
                if cashflow_normalized and filing["fiscal_period"] in ["Q2", "Q3"]:
                    filing["cash_flow_statement_qtrs"] = 1

                # Calculate ratios using NORMALIZED data
                filing["ratios"] = self._calculate_ratios(filing["data"])

                # Assess data quality
                filing["quality"] = self._assess_quality(filing["data"], filing["ratios"])

            # Sort by fiscal year and period (most recent first)
            processed_filings.sort(
                key=lambda f: (f["fiscal_year"] or 0, self._fiscal_period_to_int(f["fiscal_period"])), reverse=True
            )

            # Save processed filings to database (skip when explicitly requested)
            if persist:
                self.save_processed_data(processed_filings)

            return processed_filings

        except Exception as e:
            logger.error(f"Error processing raw data for {symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def _fiscal_period_to_int(self, fp: str) -> int:
        """Convert fiscal period to int for sorting (FY=5, Q4=4, Q3=3, Q2=2, Q1=1)"""
        if not fp:
            return 0
        fp_upper = str(fp).upper().strip()
        if fp_upper == "FY":
            return 5
        elif fp_upper.startswith("Q"):
            try:
                return int(fp_upper[1])
            except (ValueError, IndexError):
                return 0
        return 0

    def _calculate_ratios(self, data: Dict) -> Dict[str, Optional[float]]:
        """
        Calculate financial ratios from extracted data

        Args:
            data: Dictionary with flattened financial metrics

        Returns:
            Dictionary with calculated ratios
        """
        ratios = {}

        # Current Ratio = Current Assets / Current Liabilities
        if data.get("current_assets") and data.get("current_liabilities"):
            ratios["current_ratio"] = round(data["current_assets"] / data["current_liabilities"], 4)
        else:
            ratios["current_ratio"] = None

        # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        if data.get("current_assets") and data.get("current_liabilities"):
            inventory = data.get("inventory") or 0
            ratios["quick_ratio"] = round((data["current_assets"] - inventory) / data["current_liabilities"], 4)
        else:
            ratios["quick_ratio"] = None

        # Debt to Equity = Total Debt / Stockholders Equity
        if data.get("total_debt") and data.get("stockholders_equity"):
            ratios["debt_to_equity"] = round(data["total_debt"] / data["stockholders_equity"], 4)
        else:
            ratios["debt_to_equity"] = None

        # ROA = Net Income / Total Assets
        if data.get("net_income") and data.get("total_assets"):
            ratios["roa"] = round(data["net_income"] / data["total_assets"], 4)
        else:
            ratios["roa"] = None

        # ROE = Net Income / Stockholders Equity
        if data.get("net_income") and data.get("stockholders_equity"):
            ratios["roe"] = round(data["net_income"] / data["stockholders_equity"], 4)
        else:
            ratios["roe"] = None

        # Gross Margin = Gross Profit / Revenue
        if data.get("gross_profit") and data.get("total_revenue"):
            ratios["gross_margin"] = round(data["gross_profit"] / data["total_revenue"], 4)
        else:
            ratios["gross_margin"] = None

        # Operating Margin = Operating Income / Revenue
        if data.get("operating_income") and data.get("total_revenue"):
            ratios["operating_margin"] = round(data["operating_income"] / data["total_revenue"], 4)
        else:
            ratios["operating_margin"] = None

        # Net Margin = Net Income / Revenue
        if data.get("net_income") and data.get("total_revenue"):
            ratios["net_margin"] = round(data["net_income"] / data["total_revenue"], 4)
        else:
            ratios["net_margin"] = None

        return ratios

    def _normalize_ytd_to_pit(
        self,
        data: Dict,
        income_qtrs: int,
        cashflow_qtrs: int,
        fiscal_period: str,
        fiscal_year: int,
        all_filings: List[Dict],
        symbol: str,
    ) -> Tuple[Dict, bool, bool]:
        """
        Convert YTD values to point-in-time for income/cash flow statements

        Strategy:
        1. Income statement metrics: Use income_qtrs to determine if YTD
        2. Cash flow metrics: Use cashflow_qtrs to determine if YTD
        3. Balance sheet metrics: Always PIT (no conversion needed)
        4. For YTD: Find previous period in all_filings and subtract (Q2_PIT = Q2_YTD - Q1_PIT)

        Args:
            data: Raw extracted data dictionary
            income_qtrs: Number of quarters in income statement data (1=PIT, 2/3/4=YTD)
            cashflow_qtrs: Number of quarters in cash flow data (1=PIT, 2/3/4=YTD)
            fiscal_period: e.g., "Q2", "Q3", "FY"
            fiscal_year: e.g., 2024
            all_filings: List of all processed filings to find previous periods
            symbol: Stock ticker

        Returns:
            Tuple of:
                - Normalized data dictionary (all values point-in-time when possible)
                - Boolean flag indicating income statement was normalized
                - Boolean flag indicating cash flow statement was normalized
        """
        # DEBUG: Log entry point
        logger.info(
            f"[YTD_NORM_DEBUG] {symbol} {fiscal_year}-{fiscal_period}: "
            f"income_qtrs={income_qtrs}, cashflow_qtrs={cashflow_qtrs}, "
            f"all_filings_count={len(all_filings)}"
        )

        normalized = copy.deepcopy(data)

        # Income statement metrics needing normalization
        income_metrics = ["total_revenue", "net_income", "gross_profit", "operating_income", "cost_of_revenue"]

        # Cash flow metrics needing normalization
        cashflow_metrics = ["operating_cash_flow", "capital_expenditures", "free_cash_flow", "dividends_paid"]

        # Check if normalization is needed
        needs_income_normalization = income_qtrs > 1 and fiscal_period in ["Q2", "Q3"]
        needs_cashflow_normalization = cashflow_qtrs > 1 and fiscal_period in ["Q2", "Q3"]

        # DEBUG: Log normalization decision
        logger.info(
            f"[YTD_NORM_DEBUG] {symbol} {fiscal_year}-{fiscal_period}: "
            f"needs_income_norm={needs_income_normalization}, "
            f"needs_cashflow_norm={needs_cashflow_normalization}"
        )

        income_normalized = False
        cashflow_normalized = False

        if not (needs_income_normalization or needs_cashflow_normalization):
            # No normalization needed - data is already point-in-time
            logger.info(
                f"[YTD_NORM_DEBUG] {symbol} {fiscal_year}-{fiscal_period}: " f"NO normalization needed (already PIT)"
            )
            return normalized, income_normalized, cashflow_normalized

        # Find previous period data
        prev_period = "Q1" if fiscal_period == "Q2" else "Q2"  # Q2 needs Q1, Q3 needs Q2

        # DEBUG: Log all available filings for troubleshooting
        logger.info(
            f"[YTD_NORM_DEBUG] {symbol} {fiscal_year}-{fiscal_period}: "
            f"Looking for prev_period={prev_period} in {len(all_filings)} filings"
        )
        for idx, f in enumerate(all_filings[:10]):  # Show first 10 for debugging
            logger.debug(
                f"[YTD_NORM_DEBUG]   Filing[{idx}]: " f"{f.get('fiscal_year', 'N/A')}-{f.get('fiscal_period', 'N/A')}"
            )

        prev_filing = next(
            (f for f in all_filings if f["fiscal_year"] == fiscal_year and f["fiscal_period"] == prev_period), None
        )

        if not prev_filing:
            # CRITICAL: This is likely where Q2-2025 gets lost!
            logger.warning(
                f"[YTD_NORM_CRITICAL] {symbol} {fiscal_year}-{fiscal_period}: "
                f"Cannot normalize - Previous period {prev_period} not found! "
                f"Available periods: {[(f['fiscal_year'], f['fiscal_period']) for f in all_filings[:10]]}"
            )
            return normalized, income_normalized, cashflow_normalized

        prev_data = prev_filing["data"]

        # Normalize income statement if YTD (qtrs > 1)
        if needs_income_normalization:
            for metric in income_metrics:
                if metric in data and metric in prev_data:
                    ytd_value = data[metric]
                    prev_value = prev_data[metric]

                    if ytd_value is not None and prev_value is not None:
                        normalized[metric] = ytd_value - prev_value

                        logger.debug(
                            f"[YTD_NORM] {symbol} {fiscal_year}-{fiscal_period} {metric}: "
                            f"{ytd_value/1e6:.2f}M (YTD) - {prev_value/1e6:.2f}M (prev) = "
                            f"{normalized[metric]/1e6:.2f}M (PIT)"
                        )
                        income_normalized = True

        # Normalize cash flow if YTD (qtrs > 1)
        if needs_cashflow_normalization:
            for metric in cashflow_metrics:
                if metric in data and metric in prev_data:
                    ytd_value = data[metric]
                    prev_value = prev_data[metric]

                    if ytd_value is not None and prev_value is not None:
                        normalized[metric] = ytd_value - prev_value

                        logger.debug(
                            f"[YTD_NORM] {symbol} {fiscal_year}-{fiscal_period} {metric}: "
                            f"{ytd_value/1e6:.2f}M (YTD) - {prev_value/1e6:.2f}M (prev) = "
                            f"{normalized[metric]/1e6:.2f}M (PIT)"
                        )
                        cashflow_normalized = True

        # Balance sheet items are always PIT - no normalization needed

        return normalized, income_normalized, cashflow_normalized

    def _assess_quality(self, data: Dict, ratios: Dict) -> Dict[str, Any]:
        """
        Assess data completeness and quality

        Args:
            data: Dictionary with financial metrics
            ratios: Dictionary with calculated ratios

        Returns:
            Quality assessment dictionary
        """
        # Core metrics required for basic analysis
        core_metrics = [
            "total_revenue",
            "net_income",
            "total_assets",
            "total_liabilities",
            "current_assets",
            "current_liabilities",
            "stockholders_equity",
        ]

        # Count how many core metrics have non-null values
        core_present = sum(1 for metric in core_metrics if data.get(metric) is not None)
        core_total = len(core_metrics)

        # Count how many ratios are calculable
        ratios_present = sum(1 for ratio_val in ratios.values() if ratio_val is not None)
        ratios_total = len(ratios)

        # Calculate completeness score
        completeness_score = round((core_present / core_total) * 0.7 + (ratios_present / ratios_total) * 0.3, 2) * 100

        # Assign grade
        if completeness_score >= 90:
            grade = "A"
        elif completeness_score >= 75:
            grade = "B"
        elif completeness_score >= 60:
            grade = "C"
        elif completeness_score >= 40:
            grade = "D"
        else:
            grade = "F"

        return {
            "core_metrics_count": core_present,
            "core_metrics_total": core_total,
            "ratio_metrics_count": ratios_present,
            "ratio_metrics_total": ratios_total,
            "completeness_score": completeness_score,
            "grade": grade,
        }

    def save_processed_data(self, processed_filings: List[Dict]) -> int:
        """Save processed filings to sec_companyfacts_processed table."""
        if not processed_filings:
            logger.warning("No processed filings to save")
            return 0

        symbol = processed_filings[0]["symbol"] if processed_filings else "UNKNOWN"
        cik = processed_filings[0]["cik"] if processed_filings else "UNKNOWN"
        entity_name = ""  # Will be populated from metadata if needed
        symbol_upper = symbol.upper() if symbol else "UNKNOWN"

        def to_float_safe(value):
            if value is None:
                return None
            if isinstance(value, Decimal):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        saved_count = 0

        try:
            delete_query = text("DELETE FROM sec_companyfacts_processed WHERE symbol = :symbol")
            insert_query = text(
                """
                INSERT INTO sec_companyfacts_processed
                (symbol, cik, fiscal_year, fiscal_period,
                 total_revenue, net_income, gross_profit, operating_income, cost_of_revenue,
                 research_and_development_expense, selling_general_administrative_expense, operating_expenses,
                 interest_expense, income_tax_expense, depreciation_amortization, stock_based_compensation,
                 total_assets, total_liabilities, current_assets, current_liabilities,
                 stockholders_equity, retained_earnings, accounts_receivable, accounts_payable, accrued_liabilities,
                 inventory, cash_and_equivalents,
                 property_plant_equipment, accumulated_depreciation, property_plant_equipment_net,
                 goodwill, intangible_assets, deferred_revenue, treasury_stock, other_comprehensive_income,
                 book_value, book_value_per_share, working_capital,
                 long_term_debt, short_term_debt, total_debt, net_debt,
                 operating_cash_flow, capital_expenditures, free_cash_flow, dividends_paid,
                 preferred_stock_dividends, common_stock_dividends,
                 investing_cash_flow, financing_cash_flow,
                 weighted_average_diluted_shares_outstanding, shares_outstanding,
                 earnings_per_share, earnings_per_share_diluted,
                 market_cap, enterprise_value,
                 current_ratio, quick_ratio, debt_to_equity, roa, roe,
                 gross_margin, operating_margin, net_margin,
                 dividend_payout_ratio, dividend_yield, effective_tax_rate, interest_coverage,
                 asset_turnover,
                 income_statement_qtrs, cash_flow_statement_qtrs,
                 adsh, form_type, filed_date, period_end_date, frame,
                 extraction_version, data_quality_score, raw_data_id)
                VALUES
                (:symbol, :cik, :fiscal_year, :fiscal_period,
                 :total_revenue, :net_income, :gross_profit, :operating_income, :cost_of_revenue,
                 :research_and_development_expense, :selling_general_administrative_expense, :operating_expenses,
                 :interest_expense, :income_tax_expense, :depreciation_amortization, :stock_based_compensation,
                 :total_assets, :total_liabilities, :current_assets, :current_liabilities,
                 :stockholders_equity, :retained_earnings, :accounts_receivable, :accounts_payable, :accrued_liabilities,
                 :inventory, :cash_and_equivalents,
                 :property_plant_equipment, :accumulated_depreciation, :property_plant_equipment_net,
                 :goodwill, :intangible_assets, :deferred_revenue, :treasury_stock, :other_comprehensive_income,
                 :book_value, :book_value_per_share, :working_capital,
                 :long_term_debt, :short_term_debt, :total_debt, :net_debt,
                 :operating_cash_flow, :capital_expenditures, :free_cash_flow, :dividends_paid,
                 :preferred_stock_dividends, :common_stock_dividends,
                 :investing_cash_flow, :financing_cash_flow,
                 :weighted_average_diluted_shares_outstanding, :shares_outstanding,
                 :earnings_per_share, :earnings_per_share_diluted,
                 :market_cap, :enterprise_value,
                 :current_ratio, :quick_ratio, :debt_to_equity, :roa, :roe,
                 :gross_margin, :operating_margin, :net_margin,
                 :dividend_payout_ratio, :dividend_yield, :effective_tax_rate, :interest_coverage,
                 :asset_turnover,
                 :income_statement_qtrs, :cash_flow_statement_qtrs,
                 :adsh, :form_type, :filed_date, :period_end_date, :frame,
                 :extraction_version, :data_quality_score, :raw_data_id)
                """
            )

            with self.engine.begin() as conn:
                conn.execute(delete_query, {"symbol": symbol_upper})

                for filing in processed_filings:
                    data = filing["data"]
                    ratios = filing["ratios"]
                    quality = filing["quality"]

                    def prefer_value(key: str):
                        value = data.get(key)
                        if value is not None:
                            return value
                        if ratios:
                            ratio_val = ratios.get(key)
                            if ratio_val is not None:
                                return ratio_val

                            synonym = self._ratio_synonyms.get(key)
                            if synonym:
                                # Prefer data values first so we preserve PIT adjustments
                                syn_data_val = data.get(synonym)
                                if syn_data_val is not None:
                                    return syn_data_val
                                syn_ratio_val = ratios.get(synonym)
                                if syn_ratio_val is not None:
                                    return syn_ratio_val
                        return None

                    ocf_val = to_float_safe(data.get("operating_cash_flow"))
                    capex_val = to_float_safe(data.get("capital_expenditures"))
                    fcf_val = data.get("free_cash_flow")
                    fcf_needs_derivation = fcf_val is None
                    if isinstance(fcf_val, (int, float)):
                        fcf_needs_derivation = abs(fcf_val) < 1e-6

                    if fcf_needs_derivation and ocf_val is not None and capex_val is not None:
                        derived_fcf = ocf_val - abs(capex_val)
                        data["free_cash_flow"] = derived_fcf
                        logger.debug(
                            "🔄 [SAVE] Derived free_cash_flow for %s %s-%s via OCF %.2f - |CapEx| %.2f = %.2f",
                            symbol,
                            filing["fiscal_year"],
                            filing["fiscal_period"],
                            ocf_val,
                            capex_val,
                            derived_fcf,
                        )

                    conn.execute(
                        insert_query,
                        {
                            "symbol": symbol_upper,
                            "cik": filing["cik"],
                            "fiscal_year": filing["fiscal_year"],
                            "fiscal_period": filing["fiscal_period"],
                            "total_revenue": data.get("total_revenue"),
                            "net_income": data.get("net_income"),
                            "gross_profit": data.get("gross_profit"),
                            "operating_income": data.get("operating_income"),
                            "cost_of_revenue": data.get("cost_of_revenue"),
                            "research_and_development_expense": data.get("research_and_development_expense"),
                            "selling_general_administrative_expense": data.get(
                                "selling_general_administrative_expense"
                            ),
                            "operating_expenses": data.get("operating_expenses"),
                            "interest_expense": data.get("interest_expense"),
                            "income_tax_expense": data.get("income_tax_expense"),
                            "depreciation_amortization": data.get("depreciation_amortization"),
                            "stock_based_compensation": data.get("stock_based_compensation"),
                            "total_assets": data.get("total_assets"),
                            "total_liabilities": data.get("total_liabilities"),
                            "current_assets": data.get("current_assets"),
                            "current_liabilities": data.get("current_liabilities"),
                            "stockholders_equity": data.get("stockholders_equity"),
                            "retained_earnings": data.get("retained_earnings"),
                            "accounts_receivable": data.get("accounts_receivable"),
                            "accounts_payable": data.get("accounts_payable"),
                            "accrued_liabilities": data.get("accrued_liabilities"),
                            "inventory": data.get("inventory"),
                            "cash_and_equivalents": data.get("cash_and_equivalents"),
                            "property_plant_equipment": data.get("property_plant_equipment"),
                            "accumulated_depreciation": data.get("accumulated_depreciation"),
                            "property_plant_equipment_net": data.get("property_plant_equipment_net"),
                            "goodwill": data.get("goodwill"),
                            "intangible_assets": data.get("intangible_assets"),
                            "deferred_revenue": data.get("deferred_revenue"),
                            "treasury_stock": data.get("treasury_stock"),
                            "other_comprehensive_income": data.get("other_comprehensive_income"),
                            "book_value": data.get("book_value"),
                            "book_value_per_share": data.get("book_value_per_share"),
                            "working_capital": data.get("working_capital"),
                            "long_term_debt": data.get("long_term_debt"),
                            "short_term_debt": data.get("short_term_debt"),
                            "total_debt": data.get("total_debt"),
                            "net_debt": data.get("net_debt"),
                            "operating_cash_flow": data.get("operating_cash_flow"),
                            "capital_expenditures": data.get("capital_expenditures"),
                            "free_cash_flow": data.get("free_cash_flow"),
                            "dividends_paid": data.get("dividends_paid"),
                            "preferred_stock_dividends": data.get("preferred_stock_dividends"),
                            "common_stock_dividends": data.get("common_stock_dividends"),
                            "investing_cash_flow": data.get("investing_cash_flow"),
                            "financing_cash_flow": data.get("financing_cash_flow"),
                            "weighted_average_diluted_shares_outstanding": data.get(
                                "weighted_average_diluted_shares_outstanding"
                            ),
                            "shares_outstanding": data.get("shares_outstanding"),
                            "earnings_per_share": data.get("earnings_per_share"),
                            "earnings_per_share_diluted": data.get("earnings_per_share_diluted"),
                            "market_cap": data.get("market_cap"),
                            "enterprise_value": data.get("enterprise_value"),
                            "current_ratio": prefer_value("current_ratio"),
                            "quick_ratio": prefer_value("quick_ratio"),
                            "debt_to_equity": prefer_value("debt_to_equity"),
                            "roa": prefer_value("roa"),
                            "roe": prefer_value("roe"),
                            "gross_margin": prefer_value("gross_margin"),
                            "operating_margin": prefer_value("operating_margin"),
                            "net_margin": prefer_value("net_margin"),
                            "dividend_payout_ratio": prefer_value("dividend_payout_ratio"),
                            "dividend_yield": prefer_value("dividend_yield"),
                            "effective_tax_rate": prefer_value("effective_tax_rate"),
                            "interest_coverage": prefer_value("interest_coverage"),
                            "asset_turnover": prefer_value("asset_turnover"),
                            "income_statement_qtrs": filing.get("income_statement_qtrs"),
                            "cash_flow_statement_qtrs": filing.get("cash_flow_statement_qtrs"),
                            "adsh": filing["adsh"],
                            "form_type": filing["form_type"],
                            "filed_date": filing["filed_date"],
                            "period_end_date": filing["period_end_date"],
                            "frame": filing["frame"],
                            "extraction_version": filing["extraction_version"],
                            "data_quality_score": quality["completeness_score"],
                            "raw_data_id": filing["raw_data_id"],
                        },
                    )
                    saved_count += 1

            logger.info(f"✅ Saved {saved_count} processed filings for {symbol} to sec_companyfacts_processed")

            # Update metadata table
            self._update_metadata(symbol, cik, entity_name, processed_filings)

            return saved_count

        except Exception as e:
            logger.error(f"Error saving processed data for {symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return 0

    def _update_metadata(self, symbol: str, cik: str, entity_name: str, processed_filings: List[Dict]):
        """
        Update sec_companyfacts_metadata table with cache control and quality stats

        Args:
            symbol: Stock ticker
            cik: CIK number
            entity_name: Company name
            processed_filings: List of processed filings for quality assessment
        """
        try:
            if not processed_filings:
                return

            # Calculate aggregate statistics
            total_filings = len(processed_filings)
            quarters_available = sum(1 for f in processed_filings if f["fiscal_period"] in ["Q1", "Q2", "Q3", "Q4"])

            # Get earliest and latest filings
            filing_dates = [f["filed_date"] for f in processed_filings if f["filed_date"]]
            earliest_filing = min(filing_dates) if filing_dates else None
            latest_filing = max(filing_dates) if filing_dates else None

            # Calculate average data quality
            quality_scores = [f["quality"]["completeness_score"] for f in processed_filings]
            avg_quality_score = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0

            # Assign aggregate grade
            if avg_quality_score >= 90:
                grade = "A"
            elif avg_quality_score >= 75:
                grade = "B"
            elif avg_quality_score >= 60:
                grade = "C"
            elif avg_quality_score >= 40:
                grade = "D"
            else:
                grade = "F"

            # Calculate average core and ratio metrics
            avg_core = round(sum(f["quality"]["core_metrics_count"] for f in processed_filings) / total_filings, 0)
            avg_ratios = round(sum(f["quality"]["ratio_metrics_count"] for f in processed_filings) / total_filings, 0)

            # Calculate next refresh date (90 days from now)
            from datetime import timedelta

            next_refresh = datetime.now() + timedelta(days=90)

            query = text(
                """
                INSERT INTO sec_companyfacts_metadata
                (symbol, cik, entity_name, last_fetched, last_processed, fetch_count,
                 cache_ttl_days, next_refresh_due, raw_data_complete, processing_status,
                 data_quality_grade, core_metrics_count, ratio_metrics_count,
                 earliest_filing, latest_filing, total_filings, quarters_available)
                VALUES
                (:symbol, :cik, :entity_name, NOW(), NOW(), 1,
                 90, :next_refresh, TRUE, 'completed',
                 :grade, :core_metrics, :ratio_metrics,
                 :earliest, :latest, :total, :quarters)
                ON CONFLICT (symbol) DO UPDATE SET
                    last_processed = NOW(),
                    fetch_count = sec_companyfacts_metadata.fetch_count + 1,
                    next_refresh_due = EXCLUDED.next_refresh_due,
                    raw_data_complete = TRUE,
                    processing_status = 'completed',
                    data_quality_grade = EXCLUDED.data_quality_grade,
                    core_metrics_count = EXCLUDED.core_metrics_count,
                    ratio_metrics_count = EXCLUDED.ratio_metrics_count,
                    earliest_filing = EXCLUDED.earliest_filing,
                    latest_filing = EXCLUDED.latest_filing,
                    total_filings = EXCLUDED.total_filings,
                    quarters_available = EXCLUDED.quarters_available,
                    updated_at = NOW()
            """
            )

            with self.engine.connect() as conn:
                conn.execute(
                    query,
                    {
                        "symbol": symbol.upper(),
                        "cik": cik,
                        "entity_name": entity_name,
                        "next_refresh": next_refresh,
                        "grade": grade,
                        "core_metrics": int(avg_core),
                        "ratio_metrics": int(avg_ratios),
                        "earliest": earliest_filing,
                        "latest": latest_filing,
                        "total": total_filings,
                        "quarters": quarters_available,
                    },
                )
                conn.commit()

            logger.info(
                f"📊 Updated metadata for {symbol}: "
                f"grade={grade}, filings={total_filings}, quality={avg_quality_score}%"
            )

        except Exception as e:
            logger.warning(f"Failed to update metadata for {symbol}: {e}")


# Convenience function
def get_sec_data_processor(db_engine=None):
    """Get SECDataProcessor instance"""
    return SECDataProcessor(db_engine)
