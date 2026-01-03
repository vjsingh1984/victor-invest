"""
Fiscal Period Service - Centralized fiscal period handling

This service provides deterministic fiscal period operations including:
- Period normalization (Q1, FIRST QUARTER, 1Q → "Q1")
- Period parsing (2024-Q1 → (2024, "Q1"))
- YTD detection based on authoritative qtrs field
- Fiscal year end detection

Author: Claude Code (Architecture Redesign Phase 1)
Date: 2025-11-12
"""

import re
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class FiscalPeriod:
    """Represents a parsed fiscal period"""

    fiscal_year: int
    period: str  # Q1, Q2, Q3, Q4, FY
    period_str: str  # Original format like "2024-Q1"

    def __str__(self) -> str:
        return f"{self.fiscal_year}-{self.period}"


class FiscalPeriodService:
    """
    Centralized service for fiscal period operations.

    Design Principles:
    1. Deterministic - No AI, fixed mappings only
    2. Single Source of Truth - All period logic here
    3. Testable - Pure functions, no side effects
    4. Documented - Clear examples for each method
    """

    # Period normalization mapping table (deterministic)
    PERIOD_MAPPINGS = {
        # Standard formats
        "Q1": "Q1",
        "Q2": "Q2",
        "Q3": "Q3",
        "Q4": "Q4",
        "FY": "FY",
        # Variations
        "FIRST QUARTER": "Q1",
        "SECOND QUARTER": "Q2",
        "THIRD QUARTER": "Q3",
        "FOURTH QUARTER": "Q4",
        "FULL YEAR": "FY",
        "ANNUAL": "FY",
        # Numeric formats
        "1Q": "Q1",
        "2Q": "Q2",
        "3Q": "Q3",
        "4Q": "Q4",
        # With YTD suffix (strip suffix, return quarter)
        "Q1-YTD": "Q1",
        "Q2-YTD": "Q2",
        "Q3-YTD": "Q3",
        # SEC XBRL standard forms
        "QTR1": "Q1",
        "QTR2": "Q2",
        "QTR3": "Q3",
        "QTR4": "Q4",
        # Alternative formats
        "1": "Q1",
        "2": "Q2",
        "3": "Q3",
        "4": "Q4",
        "Y": "FY",
    }

    # Fiscal period sorting order (for chronological ordering)
    PERIOD_SORT_ORDER = {
        "FY": 5,  # Full year comes first conceptually
        "Q4": 4,
        "Q3": 3,
        "Q2": 2,
        "Q1": 1,
    }

    def __init__(self):
        """Initialize the fiscal period service"""
        self.logger = logger

    def normalize_period(self, fiscal_period: str) -> str:
        """
        Normalize period variations to standard format.

        Args:
            fiscal_period: Period string (e.g., "Q1", "FIRST QUARTER", "1Q", "Q1-YTD")

        Returns:
            Normalized period ("Q1", "Q2", "Q3", "Q4", or "FY")

        Examples:
            >>> normalize_period("FIRST QUARTER")
            "Q1"
            >>> normalize_period("Q2-YTD")
            "Q2"
            >>> normalize_period("1Q")
            "Q1"

        Raises:
            ValueError: If period format is not recognized
        """
        if not fiscal_period:
            raise ValueError("Fiscal period cannot be empty")

        # Normalize to uppercase and strip whitespace
        period_upper = fiscal_period.strip().upper()

        # Direct lookup in mapping table
        if period_upper in self.PERIOD_MAPPINGS:
            normalized = self.PERIOD_MAPPINGS[period_upper]
            if period_upper != normalized:
                self.logger.debug(f"Normalized period '{fiscal_period}' → '{normalized}'")
            return normalized

        # If not in mapping, raise error
        raise ValueError(
            f"Unknown fiscal period format: '{fiscal_period}'. "
            f"Supported formats: {list(self.PERIOD_MAPPINGS.keys())}"
        )

    def parse_period(self, period_str: str) -> FiscalPeriod:
        """
        Parse period string to fiscal year and period.

        Args:
            period_str: Period string (e.g., "2024-Q1", "2024-FY", "2024")

        Returns:
            FiscalPeriod object with parsed components

        Examples:
            >>> parse_period("2024-Q1")
            FiscalPeriod(fiscal_year=2024, period="Q1", period_str="2024-Q1")
            >>> parse_period("2024")
            FiscalPeriod(fiscal_year=2024, period="FY", period_str="2024")

        Raises:
            ValueError: If period string format is invalid
        """
        if not period_str:
            raise ValueError("Period string cannot be empty")

        period_str = period_str.strip()

        # Pattern: YYYY-QN or YYYY-FY
        match = re.match(r"^(\d{4})(?:-(.+))?$", period_str)

        if not match:
            raise ValueError(
                f"Invalid period string format: '{period_str}'. " f"Expected format: 'YYYY-QN' or 'YYYY-FY' or 'YYYY'"
            )

        fiscal_year_str, period_part = match.groups()
        fiscal_year = int(fiscal_year_str)

        # If no period part, assume FY
        if not period_part:
            period = "FY"
        else:
            # Normalize the period part
            period = self.normalize_period(period_part)

        return FiscalPeriod(fiscal_year=fiscal_year, period=period, period_str=period_str)

    def is_ytd(self, qtrs: int) -> bool:
        """
        Deterministic YTD detection based on qtrs field.

        Uses authoritative qtrs field from SEC bulk tables:
        - qtrs=1: Individual quarter (NOT YTD)
        - qtrs=2: YTD through Q2
        - qtrs=3: YTD through Q3
        - qtrs=4: Full year

        Args:
            qtrs: Number of quarters (from SEC bulk tables or companyfacts data)

        Returns:
            True if data is YTD/cumulative, False if individual quarter

        Examples:
            >>> is_ytd(1)
            False  # Individual quarter
            >>> is_ytd(2)
            True   # YTD through Q2
            >>> is_ytd(3)
            True   # YTD through Q3
            >>> is_ytd(4)
            True   # Full year

        Note:
            This is the AUTHORITATIVE method for YTD detection.
            Do NOT infer YTD from start dates or duration!
        """
        if not isinstance(qtrs, int):
            raise TypeError(f"qtrs must be int, got {type(qtrs)}")

        if qtrs < 1 or qtrs > 4:
            raise ValueError(f"qtrs must be 1-4, got {qtrs}")

        return qtrs >= 2

    def detect_fiscal_year_end(self, company_facts: Dict[str, Any]) -> str:
        """
        Detect fiscal year end from CompanyFacts API data.

        Args:
            company_facts: CompanyFacts JSON response from SEC API

        Returns:
            Fiscal year end suffix (e.g., "-12-31", "-06-30", "-09-30")

        Examples:
            >>> detect_fiscal_year_end(company_facts_json)
            "-12-31"  # Calendar year
            >>> detect_fiscal_year_end(company_facts_json)
            "-06-30"  # June fiscal year end

        Algorithm:
            1. Find FY (10-K) filings in company facts
            2. Extract period_end dates
            3. Determine most common month-day suffix

        Raises:
            ValueError: If no fiscal year data found
        """
        if not company_facts or "facts" not in company_facts:
            raise ValueError("Invalid company facts data: missing 'facts' key")

        # Collect all FY period end dates
        fy_period_ends = []

        # Iterate through all taxonomies and concepts
        facts = company_facts.get("facts", {})
        for taxonomy in ["us-gaap", "dei", "ifrs-full"]:
            if taxonomy not in facts:
                continue

            for concept, concept_data in facts[taxonomy].items():
                if "units" not in concept_data:
                    continue

                for unit_type, unit_data in concept_data["units"].items():
                    for entry in unit_data:
                        # Look for fiscal year entries (form 10-K)
                        if entry.get("form") == "10-K" and entry.get("fy"):
                            period_end = entry.get("end")
                            if period_end:
                                fy_period_ends.append(period_end)

        if not fy_period_ends:
            raise ValueError("No fiscal year (10-K) data found in company facts")

        # Extract month-day suffix from period end dates
        # Format: YYYY-MM-DD → extract -MM-DD
        suffixes = {}
        for period_end in fy_period_ends:
            # Extract last 6 characters: -MM-DD
            if len(period_end) >= 10:
                suffix = period_end[-6:]  # "-MM-DD"
                suffixes[suffix] = suffixes.get(suffix, 0) + 1

        if not suffixes:
            raise ValueError("Could not extract fiscal year end from period dates")

        # Return most common suffix
        fiscal_year_end = max(suffixes, key=suffixes.get)

        self.logger.info(f"Detected fiscal year end: {fiscal_year_end} " f"(from {len(fy_period_ends)} FY filings)")

        return fiscal_year_end

    def get_period_sort_key(self, period: str) -> int:
        """
        Get sort key for chronological ordering of fiscal periods.

        Args:
            period: Normalized period (Q1, Q2, Q3, Q4, FY)

        Returns:
            Sort key (higher = later in fiscal year)

        Examples:
            >>> get_period_sort_key("Q1")
            1
            >>> get_period_sort_key("FY")
            5

        Usage:
            periods = ["Q3", "Q1", "FY", "Q2"]
            sorted_periods = sorted(periods, key=service.get_period_sort_key)
            # Result: ["Q1", "Q2", "Q3", "FY"]
        """
        normalized = self.normalize_period(period)
        return self.PERIOD_SORT_ORDER.get(normalized, 0)

    def format_period(self, fiscal_year: int, period: str) -> str:
        """
        Format fiscal year and period to standard string format.

        Args:
            fiscal_year: Fiscal year (e.g., 2024)
            period: Period (Q1, Q2, Q3, Q4, FY)

        Returns:
            Formatted period string (e.g., "2024-Q1")

        Examples:
            >>> format_period(2024, "Q1")
            "2024-Q1"
            >>> format_period(2024, "FY")
            "2024-FY"
        """
        normalized_period = self.normalize_period(period)
        return f"{fiscal_year}-{normalized_period}"

    def calculate_fiscal_year(
        self,
        period_end_date: str,
        fiscal_year_end_month: int,
        fiscal_year_end_day: int = 31,
        fiscal_period: Optional[str] = None,
    ) -> int:
        """
        Calculate fiscal year from period end date and fiscal year end.

        This is the AUTHORITATIVE method for fiscal year calculation.
        Use this instead of ad-hoc calculations scattered throughout the codebase.

        Fiscal Year Labeling Convention:
        - The fiscal year is labeled by the calendar year in which it ENDS
        - Oracle FY2025 ends May 31, 2025
        - Walmart FY2025 ends January 31, 2025

        Args:
            period_end_date: Period end date (YYYY-MM-DD)
            fiscal_year_end_month: Fiscal year end month (1=Jan, 5=May, 12=Dec)
            fiscal_year_end_day: Fiscal year end day (default 31)
            fiscal_period: Optional period identifier (Q1, Q2, Q3, Q4, FY)
                          Helps disambiguate edge cases

        Returns:
            Fiscal year (e.g., 2025)

        Examples:
            # Oracle (FY ends May 31)
            >>> service.calculate_fiscal_year("2024-11-30", 5, 31)
            2025  # Nov > May, so next fiscal year

            >>> service.calculate_fiscal_year("2025-02-28", 5, 31)
            2025  # Feb < May, still in FY2025

            >>> service.calculate_fiscal_year("2025-05-31", 5, 31, "FY")
            2025  # May == May, FY filing → current year

            # Walmart (FY ends January 31)
            >>> service.calculate_fiscal_year("2024-10-31", 1, 31)
            2025  # Oct > Jan, so next fiscal year

            >>> service.calculate_fiscal_year("2025-01-31", 1, 31, "Q4")
            2025  # Jan == Jan, Q4 filing → current year
        """
        try:
            period_end = datetime.strptime(period_end_date, "%Y-%m-%d")
            period_month = period_end.month
            period_day = period_end.day
            period_year = period_end.year

            # Case 1: Period month AFTER fiscal year end month
            # This period is in the NEXT fiscal year
            if period_month > fiscal_year_end_month:
                return period_year + 1

            # Case 2: Period month BEFORE fiscal year end month
            # This period is in the CURRENT fiscal year
            if period_month < fiscal_year_end_month:
                return period_year

            # Case 3: Period month EQUALS fiscal year end month
            # Need to compare days and/or use fiscal_period context
            if period_day > fiscal_year_end_day:
                # Period ends AFTER fiscal year end → next fiscal year
                return period_year + 1

            # Period is at or before fiscal year end day
            # If this is a FY or Q4 filing ending on/near FY end, it's current year
            if fiscal_period in ["FY", "Q4"]:
                return period_year

            # For quarterly filings (Q1-Q3) ending in the FY end month,
            # this is unusual and likely the current fiscal year
            return period_year

        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error calculating fiscal year from date {period_end_date}: {e}")
            # Fallback: extract year from date
            try:
                return int(period_end_date[:4])
            except (ValueError, IndexError):
                return datetime.now().year

    def validate_fiscal_year_assignment(
        self,
        period_end_date: str,
        assigned_fiscal_year: int,
        fiscal_year_end_month: int,
        fiscal_year_end_day: int = 31,
        fiscal_period: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate that a fiscal year assignment is correct.

        Args:
            period_end_date: Period end date (YYYY-MM-DD)
            assigned_fiscal_year: The fiscal year that was assigned
            fiscal_year_end_month: Fiscal year end month
            fiscal_year_end_day: Fiscal year end day
            fiscal_period: Optional period identifier

        Returns:
            Tuple of (is_valid, error_message, suggested_fiscal_year)
            - is_valid: True if assignment appears correct
            - error_message: Description of the issue if invalid
            - suggested_fiscal_year: Corrected fiscal year if invalid

        Examples:
            >>> service.validate_fiscal_year_assignment(
            ...     "2025-01-31", 2026, 1, 31, "Q4"
            ... )
            (False, "FY mismatch: assigned 2026, expected 2025", 2025)
        """
        expected_fy = self.calculate_fiscal_year(
            period_end_date, fiscal_year_end_month, fiscal_year_end_day, fiscal_period
        )

        if assigned_fiscal_year != expected_fy:
            return (False, f"FY mismatch: assigned {assigned_fiscal_year}, expected {expected_fy}", expected_fy)

        return (True, None, None)

    def get_fiscal_year_end_from_month(self, month: int) -> Tuple[int, int]:
        """
        Get typical fiscal year end day for a given month.

        Args:
            month: Fiscal year end month (1-12)

        Returns:
            Tuple of (month, day) for fiscal year end

        Note:
            Most fiscal years end on the last day of the month.
            Some companies use specific dates (e.g., last Saturday).
            This returns standard month-end dates.
        """
        # Standard month-end days
        month_end_days = {
            1: 31,  # January
            2: 28,  # February (ignore leap years for FY end)
            3: 31,  # March
            4: 30,  # April
            5: 31,  # May
            6: 30,  # June
            7: 31,  # July
            8: 31,  # August
            9: 30,  # September
            10: 31,  # October
            11: 30,  # November
            12: 31,  # December
        }
        return (month, month_end_days.get(month, 31))

    def validate_q4_computation_inputs(self, fy_qtrs: int, q1_qtrs: int, q2_qtrs: int, q3_qtrs: int) -> bool:
        """
        Validate that Q4 can be computed as FY - (Q1+Q2+Q3).

        CRITICAL: Q4 computation only valid if ALL quarters are individual (qtrs=1).
        If any quarter is YTD (qtrs>=2), computation will produce incorrect results.

        Args:
            fy_qtrs: qtrs value for full year
            q1_qtrs: qtrs value for Q1
            q2_qtrs: qtrs value for Q2
            q3_qtrs: qtrs value for Q3

        Returns:
            True if Q4 can be safely computed, False otherwise

        Examples:
            >>> validate_q4_computation_inputs(4, 1, 1, 1)
            True  # All individual quarters, safe to compute Q4
            >>> validate_q4_computation_inputs(4, 1, 2, 3)
            False  # Q2 and Q3 are YTD, CANNOT compute Q4

        Raises:
            ValueError: If qtrs values are invalid
        """
        # Validate inputs
        for qtrs_val, name in [(fy_qtrs, "fy_qtrs"), (q1_qtrs, "q1_qtrs"), (q2_qtrs, "q2_qtrs"), (q3_qtrs, "q3_qtrs")]:
            if not isinstance(qtrs_val, int):
                raise TypeError(f"{name} must be int, got {type(qtrs_val)}")
            if qtrs_val < 1 or qtrs_val > 4:
                raise ValueError(f"{name} must be 1-4, got {qtrs_val}")

        # FY must be 4 quarters
        if fy_qtrs != 4:
            self.logger.warning(f"FY has qtrs={fy_qtrs}, expected 4. Cannot compute Q4.")
            return False

        # All quarters must be individual (qtrs=1)
        if q1_qtrs != 1 or q2_qtrs != 1 or q3_qtrs != 1:
            self.logger.warning(
                f"Q4 computation requires individual quarters (qtrs=1). "
                f"Got Q1={q1_qtrs}, Q2={q2_qtrs}, Q3={q3_qtrs}. "
                f"Cannot compute Q4 from YTD data."
            )
            return False

        return True


# Singleton instance
_fiscal_period_service = None


def get_fiscal_period_service() -> FiscalPeriodService:
    """
    Get the singleton FiscalPeriodService instance.

    Returns:
        FiscalPeriodService instance

    Usage:
        from investigator.domain.services.fiscal_period_service import get_fiscal_period_service

        service = get_fiscal_period_service()
        period = service.parse_period("2024-Q1")
    """
    global _fiscal_period_service
    if _fiscal_period_service is None:
        _fiscal_period_service = FiscalPeriodService()
    return _fiscal_period_service
