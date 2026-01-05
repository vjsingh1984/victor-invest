"""
Period Matching Strategies

Implements the Strategy pattern for matching SEC entries to target periods.
Each strategy uses a different approach, ordered by reliability.

SOLID Principles:
- Open/Closed: New strategies can be added without modifying existing code
- Liskov Substitution: All strategies implement PeriodMatchStrategy
- Single Responsibility: Each strategy handles one matching approach

Strategy Priority (most to least reliable):
1. ByPeriodEndMatcher - Matches by exact end date (most reliable)
2. ByDateRangeMatcher - Matches by start/end date range (handles edge cases)
3. ByFrameFieldMatcher - Matches by CY2024Q3 frame field (calendar-based)
4. ByAdshFyFpMatcher - Legacy ADSH + fy + fp match (least reliable)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .result import MatchMethod

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """Context for period matching operations."""

    target_period_end: Optional[str] = None
    target_period_start: Optional[str] = None
    target_fiscal_year: Optional[int] = None
    target_fiscal_period: Optional[str] = None
    target_adsh: Optional[str] = None
    fiscal_year_end: Optional[str] = None  # e.g., '-06-27' for June FYE
    tolerance_days: int = 7  # For fuzzy date matching


@dataclass
class MatchResult:
    """Result of a period matching attempt."""

    matched: bool
    entries: List[Dict]
    method: MatchMethod
    reason: Optional[str] = None


class PeriodMatchStrategy(ABC):
    """
    Abstract base class for period matching strategies.

    Each strategy implements a different approach to matching
    SEC entries to a target fiscal period.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        pass

    @property
    @abstractmethod
    def match_method(self) -> MatchMethod:
        """MatchMethod enum value for this strategy."""
        pass

    @abstractmethod
    def match(self, entries: List[Dict], context: MatchContext) -> MatchResult:
        """
        Find entries matching the target period.

        Args:
            entries: List of SEC entry dicts (from us_gaap[tag]['units']['USD'])
            context: Match context with target period info

        Returns:
            MatchResult with matched entries
        """
        pass

    def _filter_valid_forms(self, entries: List[Dict]) -> List[Dict]:
        """Filter to only 10-K and 10-Q forms."""
        return [e for e in entries if e.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A")]


class ByPeriodEndMatcher(PeriodMatchStrategy):
    """
    Matches entries by exact period end date.

    This is the MOST RELIABLE strategy because:
    - period_end (the 'end' field) is authoritative in SEC data
    - Unlike 'fy' field which can be wrong, 'end' date is accurate
    - Matches the actual reporting period, not filing metadata

    Example:
        Target: period_end='2025-06-27', fiscal_period='FY'
        Matches: entry with end='2025-06-27' (regardless of fy field value)
    """

    @property
    def name(self) -> str:
        return "ByPeriodEndMatcher"

    @property
    def match_method(self) -> MatchMethod:
        return MatchMethod.BY_PERIOD_END

    def match(self, entries: List[Dict], context: MatchContext) -> MatchResult:
        if not context.target_period_end:
            return MatchResult(
                matched=False, entries=[], method=self.match_method, reason="No target_period_end provided"
            )

        valid_entries = self._filter_valid_forms(entries)

        # Parse target date for comparison
        try:
            target_date = datetime.strptime(context.target_period_end, "%Y-%m-%d")
        except ValueError:
            return MatchResult(
                matched=False,
                entries=[],
                method=self.match_method,
                reason=f"Invalid target_period_end format: {context.target_period_end}",
            )

        # Find exact matches
        exact_matches = []
        for entry in valid_entries:
            end_date = entry.get("end")
            if end_date == context.target_period_end:
                exact_matches.append(entry)

        if exact_matches:
            # Also filter by fiscal_period if provided (FY vs Q1/Q2/Q3/Q4)
            if context.target_fiscal_period:
                fp_matches = [
                    e for e in exact_matches if self._matches_fiscal_period_by_duration(e, context.target_fiscal_period)
                ]
                if fp_matches:
                    logger.debug(
                        f"[{self.name}] Found {len(fp_matches)} entries with "
                        f"end={context.target_period_end}, fp={context.target_fiscal_period}"
                    )
                    return MatchResult(matched=True, entries=fp_matches, method=self.match_method)

            logger.debug(f"[{self.name}] Found {len(exact_matches)} entries with end={context.target_period_end}")
            return MatchResult(matched=True, entries=exact_matches, method=self.match_method)

        return MatchResult(
            matched=False,
            entries=[],
            method=self.match_method,
            reason=f"No entries with end={context.target_period_end}",
        )

    def _matches_fiscal_period_by_duration(self, entry: Dict, target_fp: str) -> bool:
        """Check if entry duration matches target fiscal period."""
        start = entry.get("start")
        end = entry.get("end")
        if not start or not end:
            # No duration info, accept based on fp field if present
            return entry.get("fp") == target_fp

        try:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            days = (end_date - start_date).days
        except ValueError:
            return entry.get("fp") == target_fp

        # FY: ~365 days, Q1-Q4: ~90 days
        if target_fp == "FY":
            return days >= 330  # Allow some tolerance
        else:
            return days < 120  # Individual quarter


class ByDateRangeMatcher(PeriodMatchStrategy):
    """
    Matches entries by date range with tolerance.

    Useful for handling:
    - Off-by-one date differences (weekends, holidays)
    - Slightly different period end dates across tags
    - Edge cases where exact match fails

    Example:
        Target: period_end='2025-06-27' with tolerance=7
        Matches: entries with end between 2025-06-20 and 2025-07-04
    """

    @property
    def name(self) -> str:
        return "ByDateRangeMatcher"

    @property
    def match_method(self) -> MatchMethod:
        return MatchMethod.BY_DATE_RANGE

    def match(self, entries: List[Dict], context: MatchContext) -> MatchResult:
        if not context.target_period_end:
            return MatchResult(
                matched=False, entries=[], method=self.match_method, reason="No target_period_end provided"
            )

        valid_entries = self._filter_valid_forms(entries)

        try:
            target_date = datetime.strptime(context.target_period_end, "%Y-%m-%d")
        except ValueError:
            return MatchResult(
                matched=False,
                entries=[],
                method=self.match_method,
                reason=f"Invalid target_period_end format: {context.target_period_end}",
            )

        tolerance = timedelta(days=context.tolerance_days)
        range_start = target_date - tolerance
        range_end = target_date + tolerance

        # Find entries within date range
        range_matches = []
        for entry in valid_entries:
            end_str = entry.get("end")
            if not end_str:
                continue

            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                if range_start <= end_date <= range_end:
                    range_matches.append(entry)
            except ValueError:
                continue

        if range_matches:
            # Filter by fiscal period if provided
            if context.target_fiscal_period:
                fp_matches = [
                    e for e in range_matches if self._matches_fiscal_period_by_duration(e, context.target_fiscal_period)
                ]
                if fp_matches:
                    logger.debug(
                        f"[{self.name}] Found {len(fp_matches)} entries within "
                        f"±{context.tolerance_days} days of {context.target_period_end}"
                    )
                    return MatchResult(matched=True, entries=fp_matches, method=self.match_method)

            logger.debug(
                f"[{self.name}] Found {len(range_matches)} entries within "
                f"±{context.tolerance_days} days of {context.target_period_end}"
            )
            return MatchResult(matched=True, entries=range_matches, method=self.match_method)

        return MatchResult(
            matched=False,
            entries=[],
            method=self.match_method,
            reason=f"No entries within ±{context.tolerance_days} days of {context.target_period_end}",
        )

    def _matches_fiscal_period_by_duration(self, entry: Dict, target_fp: str) -> bool:
        """Check if entry duration matches target fiscal period."""
        start = entry.get("start")
        end = entry.get("end")
        if not start or not end:
            return entry.get("fp") == target_fp

        try:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            days = (end_date - start_date).days
        except ValueError:
            return entry.get("fp") == target_fp

        if target_fp == "FY":
            return days >= 330
        else:
            return days < 120


class ByFrameFieldMatcher(PeriodMatchStrategy):
    """
    Matches entries by the 'frame' field (e.g., CY2024Q3).

    The frame field uses CALENDAR year, not fiscal year, making it
    useful as a cross-reference when fy field is unreliable.

    Example:
        Target: fiscal_year=2025, fiscal_period='FY' for June FYE
        Fiscal period ends June 2025, which is calendar CY2025
        Matches: entries with frame containing 'CY2025'
    """

    @property
    def name(self) -> str:
        return "ByFrameFieldMatcher"

    @property
    def match_method(self) -> MatchMethod:
        return MatchMethod.BY_FRAME_FIELD

    def match(self, entries: List[Dict], context: MatchContext) -> MatchResult:
        if not context.target_period_end:
            return MatchResult(
                matched=False, entries=[], method=self.match_method, reason="No target_period_end for frame derivation"
            )

        valid_entries = self._filter_valid_forms(entries)

        # Derive expected calendar year from period_end
        try:
            target_date = datetime.strptime(context.target_period_end, "%Y-%m-%d")
            calendar_year = target_date.year
        except ValueError:
            return MatchResult(
                matched=False,
                entries=[],
                method=self.match_method,
                reason=f"Invalid target_period_end format: {context.target_period_end}",
            )

        # Build expected frame patterns
        if context.target_fiscal_period == "FY":
            # For FY, look for annual frame
            expected_patterns = [f"CY{calendar_year}"]
        else:
            # For quarters, derive calendar quarter from end date
            month = target_date.month
            if month <= 3:
                cal_quarter = "Q1"
            elif month <= 6:
                cal_quarter = "Q2"
            elif month <= 9:
                cal_quarter = "Q3"
            else:
                cal_quarter = "Q4"
            expected_patterns = [f"CY{calendar_year}{cal_quarter}"]

        # Find matches
        frame_matches = []
        for entry in valid_entries:
            frame = entry.get("frame", "")
            if any(pattern in frame for pattern in expected_patterns):
                frame_matches.append(entry)

        if frame_matches:
            logger.debug(
                f"[{self.name}] Found {len(frame_matches)} entries with " f"frame matching {expected_patterns}"
            )
            return MatchResult(matched=True, entries=frame_matches, method=self.match_method)

        return MatchResult(
            matched=False,
            entries=[],
            method=self.match_method,
            reason=f"No entries with frame matching {expected_patterns}",
        )


class ByAdshFyFpMatcher(PeriodMatchStrategy):
    """
    Matches entries by ADSH, fy, and fp fields.

    WARNING: This is the LEAST RELIABLE strategy because the SEC's
    'fy' field can be incorrect (sometimes 2+ years off).

    Only use as a last resort fallback.

    Example:
        Target: adsh='0001137789-25-000157', fy=2025, fp='FY'
        Problem: SEC may tag this as fy=2027 incorrectly
    """

    @property
    def name(self) -> str:
        return "ByAdshFyFpMatcher"

    @property
    def match_method(self) -> MatchMethod:
        return MatchMethod.BY_ADSH_FY_FP

    def match(self, entries: List[Dict], context: MatchContext) -> MatchResult:
        valid_entries = self._filter_valid_forms(entries)
        matches = []

        for entry in valid_entries:
            # Check ADSH if provided
            if context.target_adsh:
                if entry.get("accn") != context.target_adsh:
                    continue

            # Check fiscal year if provided
            if context.target_fiscal_year:
                if entry.get("fy") != context.target_fiscal_year:
                    continue

            # Check fiscal period if provided
            if context.target_fiscal_period:
                if entry.get("fp") != context.target_fiscal_period:
                    continue

            matches.append(entry)

        if matches:
            logger.debug(
                f"[{self.name}] Found {len(matches)} entries with "
                f"adsh={context.target_adsh}, fy={context.target_fiscal_year}, fp={context.target_fiscal_period}"
            )
            return MatchResult(matched=True, entries=matches, method=self.match_method)

        return MatchResult(
            matched=False,
            entries=[],
            method=self.match_method,
            reason=f"No entries with fy={context.target_fiscal_year}, fp={context.target_fiscal_period}",
        )


class ByAdshOnlyMatcher(PeriodMatchStrategy):
    """
    Matches entries by ADSH only, ignoring fy/fp fields.

    Useful when we know the filing but fy/fp are unreliable.
    Selects best entry based on duration matching.
    """

    @property
    def name(self) -> str:
        return "ByAdshOnlyMatcher"

    @property
    def match_method(self) -> MatchMethod:
        return MatchMethod.BY_ADSH_ONLY

    def match(self, entries: List[Dict], context: MatchContext) -> MatchResult:
        if not context.target_adsh:
            return MatchResult(matched=False, entries=[], method=self.match_method, reason="No target_adsh provided")

        valid_entries = self._filter_valid_forms(entries)
        adsh_matches = [e for e in valid_entries if e.get("accn") == context.target_adsh]

        if not adsh_matches:
            return MatchResult(
                matched=False,
                entries=[],
                method=self.match_method,
                reason=f"No entries with accn={context.target_adsh}",
            )

        # Filter by fiscal period type if provided
        if context.target_fiscal_period:
            duration_matches = []
            for entry in adsh_matches:
                start = entry.get("start")
                end = entry.get("end")
                if start and end:
                    try:
                        start_date = datetime.strptime(start, "%Y-%m-%d")
                        end_date = datetime.strptime(end, "%Y-%m-%d")
                        days = (end_date - start_date).days

                        if context.target_fiscal_period == "FY" and days >= 330:
                            duration_matches.append(entry)
                        elif context.target_fiscal_period in ("Q1", "Q2", "Q3", "Q4") and days < 120:
                            duration_matches.append(entry)
                    except ValueError:
                        continue

            if duration_matches:
                logger.debug(
                    f"[{self.name}] Found {len(duration_matches)} entries with "
                    f"adsh={context.target_adsh[:15]}... matching {context.target_fiscal_period} duration"
                )
                return MatchResult(matched=True, entries=duration_matches, method=self.match_method)

        logger.debug(f"[{self.name}] Found {len(adsh_matches)} entries with adsh={context.target_adsh[:15]}...")
        return MatchResult(matched=True, entries=adsh_matches, method=self.match_method)
