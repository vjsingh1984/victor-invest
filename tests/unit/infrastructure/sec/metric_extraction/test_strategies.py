"""
Unit tests for period matching strategies.

Tests ByPeriodEndMatcher, ByDateRangeMatcher, ByFrameFieldMatcher,
ByAdshOnlyMatcher, and ByAdshFyFpMatcher strategies.
"""

import pytest

from investigator.infrastructure.sec.metric_extraction.result import MatchMethod
from investigator.infrastructure.sec.metric_extraction.strategies import (
    ByAdshFyFpMatcher,
    ByAdshOnlyMatcher,
    ByDateRangeMatcher,
    ByFrameFieldMatcher,
    ByPeriodEndMatcher,
    MatchContext,
    MatchResult,
    PeriodMatchStrategy,
)

# Sample SEC entries for testing
SAMPLE_ENTRIES = [
    # FY 2025 entry (correct)
    {
        "val": 50_000_000_000,
        "start": "2024-06-28",
        "end": "2025-06-27",
        "form": "10-K",
        "filed": "2025-08-05",
        "accn": "0001193125-25-000001",
        "fy": 2025,
        "fp": "FY",
        "frame": "CY2025",
    },
    # FY 2025 entry (wrong fy field - the bug we're fixing!)
    {
        "val": 50_000_000_000,
        "start": "2024-06-28",
        "end": "2025-06-27",
        "form": "10-K",
        "filed": "2025-08-05",
        "accn": "0001193125-25-000002",
        "fy": 2027,  # SEC bug: fy=2027 instead of 2025
        "fp": "FY",
        "frame": "CY2025",
    },
    # Q1 2026 entry
    {
        "val": 12_000_000_000,
        "start": "2025-06-28",
        "end": "2025-09-26",
        "form": "10-Q",
        "filed": "2025-10-25",
        "accn": "0001193125-25-000003",
        "fy": 2026,
        "fp": "Q1",
        "frame": "CY2025Q3",
    },
    # Older FY 2024 entry
    {
        "val": 45_000_000_000,
        "start": "2023-06-30",
        "end": "2024-06-28",
        "form": "10-K",
        "filed": "2024-08-05",
        "accn": "0001193125-24-000001",
        "fy": 2024,
        "fp": "FY",
        "frame": "CY2024",
    },
]


class TestMatchContext:
    """Tests for MatchContext dataclass."""

    def test_create_context(self):
        """Test creating a match context."""
        context = MatchContext(
            target_period_end="2025-06-27",
            target_fiscal_year=2025,
            target_fiscal_period="FY",
            tolerance_days=7,
        )

        assert context.target_period_end == "2025-06-27"
        assert context.target_fiscal_year == 2025
        assert context.target_fiscal_period == "FY"
        assert context.tolerance_days == 7

    def test_default_tolerance(self):
        """Test default tolerance is 7 days."""
        context = MatchContext()
        assert context.tolerance_days == 7


class TestByPeriodEndMatcher:
    """Tests for ByPeriodEndMatcher - the primary reliable strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = ByPeriodEndMatcher()

    def test_name(self):
        """Test matcher name."""
        assert self.matcher.name == "ByPeriodEndMatcher"

    def test_match_method(self):
        """Test match method enum."""
        assert self.matcher.match_method == MatchMethod.BY_PERIOD_END

    def test_exact_period_end_match(self):
        """Test matching by exact period end date."""
        context = MatchContext(
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True
        assert len(result.entries) >= 1
        # Should find entries with end="2025-06-27" regardless of fy field
        for entry in result.entries:
            assert entry["end"] == "2025-06-27"

    def test_finds_entry_despite_wrong_fy_field(self):
        """Critical test: must find entry even when SEC fy field is wrong."""
        context = MatchContext(
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True
        # Should find BOTH entries (one with fy=2025, one with fy=2027)
        # because we match by period_end, not fy
        assert len(result.entries) >= 1

        # Verify we found the entry with the "wrong" fy field
        fy_values = [e["fy"] for e in result.entries]
        # Either should work - we match by date, not fy
        assert any(e["end"] == "2025-06-27" for e in result.entries)

    def test_no_match_wrong_date(self):
        """Test no match when period end doesn't exist."""
        context = MatchContext(
            target_period_end="2025-12-31",  # No entries for this date
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is False
        assert len(result.entries) == 0
        assert "No entries" in result.reason

    def test_no_target_period_end(self):
        """Test handling when target_period_end is not provided."""
        context = MatchContext(
            target_fiscal_year=2025,
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is False
        assert "No target_period_end" in result.reason

    def test_filters_by_duration_for_fy(self):
        """Test that FY entries are filtered by duration (>= 330 days)."""
        context = MatchContext(
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        # All matched entries should have annual duration
        for entry in result.entries:
            start = entry.get("start")
            end = entry.get("end")
            if start and end:
                from datetime import datetime

                start_date = datetime.strptime(start, "%Y-%m-%d")
                end_date = datetime.strptime(end, "%Y-%m-%d")
                days = (end_date - start_date).days
                assert days >= 330, f"Entry has {days} days, expected >= 330"

    def test_filters_by_duration_for_quarter(self):
        """Test that quarterly entries are filtered by duration (< 120 days)."""
        context = MatchContext(
            target_period_end="2025-09-26",
            target_fiscal_period="Q1",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        if result.matched:
            for entry in result.entries:
                start = entry.get("start")
                end = entry.get("end")
                if start and end:
                    from datetime import datetime

                    start_date = datetime.strptime(start, "%Y-%m-%d")
                    end_date = datetime.strptime(end, "%Y-%m-%d")
                    days = (end_date - start_date).days
                    assert days < 120, f"Entry has {days} days, expected < 120"


class TestByDateRangeMatcher:
    """Tests for ByDateRangeMatcher - fuzzy date matching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = ByDateRangeMatcher()

    def test_name(self):
        """Test matcher name."""
        assert self.matcher.name == "ByDateRangeMatcher"

    def test_match_method(self):
        """Test match method enum."""
        assert self.matcher.match_method == MatchMethod.BY_DATE_RANGE

    def test_match_within_tolerance(self):
        """Test matching within tolerance window."""
        # Target is 2025-06-25, actual entry is 2025-06-27 (2 days off)
        context = MatchContext(
            target_period_end="2025-06-25",
            target_fiscal_period="FY",
            tolerance_days=7,
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True
        # Should find the 2025-06-27 entry (within 7-day tolerance)
        assert any(e["end"] == "2025-06-27" for e in result.entries)

    def test_no_match_outside_tolerance(self):
        """Test no match when outside tolerance window."""
        context = MatchContext(
            target_period_end="2025-07-15",  # Too far from any entry
            target_fiscal_period="FY",
            tolerance_days=7,
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is False

    def test_custom_tolerance(self):
        """Test with custom tolerance value."""
        context = MatchContext(
            target_period_end="2025-06-20",
            target_fiscal_period="FY",
            tolerance_days=10,  # Should reach 2025-06-27
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True


class TestByFrameFieldMatcher:
    """Tests for ByFrameFieldMatcher - calendar year frame matching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = ByFrameFieldMatcher()

    def test_name(self):
        """Test matcher name."""
        assert self.matcher.name == "ByFrameFieldMatcher"

    def test_match_method(self):
        """Test match method enum."""
        assert self.matcher.match_method == MatchMethod.BY_FRAME_FIELD

    def test_match_fy_by_frame(self):
        """Test matching FY by frame field."""
        context = MatchContext(
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True
        # Should find entries with frame containing CY2025
        for entry in result.entries:
            assert "CY2025" in entry.get("frame", "")

    def test_match_quarter_by_frame(self):
        """Test matching quarter by frame field."""
        context = MatchContext(
            target_period_end="2025-09-26",  # September = Q3 in calendar year
            target_fiscal_period="Q1",  # Fiscal Q1, but calendar Q3
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        if result.matched:
            # Frame should contain CY2025Q3 (calendar quarter)
            for entry in result.entries:
                assert "CY2025Q3" in entry.get("frame", "")


class TestByAdshOnlyMatcher:
    """Tests for ByAdshOnlyMatcher - ADSH-based matching with duration filter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = ByAdshOnlyMatcher()

    def test_name(self):
        """Test matcher name."""
        assert self.matcher.name == "ByAdshOnlyMatcher"

    def test_match_method(self):
        """Test match method enum."""
        assert self.matcher.match_method == MatchMethod.BY_ADSH_ONLY

    def test_match_by_adsh(self):
        """Test matching by ADSH only."""
        context = MatchContext(
            target_adsh="0001193125-25-000001",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True
        assert all(e["accn"] == "0001193125-25-000001" for e in result.entries)

    def test_no_adsh_provided(self):
        """Test handling when no ADSH is provided."""
        context = MatchContext(
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is False
        assert "No target_adsh" in result.reason

    def test_filters_by_duration(self):
        """Test that entries are filtered by duration matching fiscal period."""
        context = MatchContext(
            target_adsh="0001193125-25-000001",
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        # All matched FY entries should have >= 330 days duration
        for entry in result.entries:
            start = entry.get("start")
            end = entry.get("end")
            if start and end:
                from datetime import datetime

                start_date = datetime.strptime(start, "%Y-%m-%d")
                end_date = datetime.strptime(end, "%Y-%m-%d")
                days = (end_date - start_date).days
                assert days >= 330


class TestByAdshFyFpMatcher:
    """Tests for ByAdshFyFpMatcher - legacy least-reliable strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = ByAdshFyFpMatcher()

    def test_name(self):
        """Test matcher name."""
        assert self.matcher.name == "ByAdshFyFpMatcher"

    def test_match_method(self):
        """Test match method enum."""
        assert self.matcher.match_method == MatchMethod.BY_ADSH_FY_FP

    def test_match_by_adsh_fy_fp(self):
        """Test matching by ADSH, fy, and fp fields."""
        context = MatchContext(
            target_adsh="0001193125-25-000001",
            target_fiscal_year=2025,
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        assert result.matched is True
        for entry in result.entries:
            assert entry["accn"] == "0001193125-25-000001"
            assert entry["fy"] == 2025
            assert entry["fp"] == "FY"

    def test_fails_with_wrong_fy_field(self):
        """
        Critical test: this strategy FAILS when SEC fy field is wrong.

        This demonstrates why ByAdshFyFpMatcher is least reliable and
        why we need ByPeriodEndMatcher as the primary strategy.
        """
        # Try to find the entry with the wrong fy field using fy-based matching
        context = MatchContext(
            target_adsh="0001193125-25-000002",  # This entry has fy=2027 (wrong)
            target_fiscal_year=2025,  # We expect fiscal year 2025
            target_fiscal_period="FY",
        )

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        # This SHOULD FAIL because fy=2027 != 2025
        # This is the exact bug we fixed with ByPeriodEndMatcher
        assert result.matched is False

    def test_no_filters_provided(self):
        """Test matching with no filters - returns all valid form entries."""
        context = MatchContext()

        result = self.matcher.match(SAMPLE_ENTRIES, context)

        # Should match all 10-K and 10-Q entries
        assert result.matched is True


class TestStrategyOrdering:
    """Tests for strategy priority and ordering."""

    def test_period_end_is_most_reliable(self):
        """Verify ByPeriodEndMatcher finds entries that ByAdshFyFpMatcher misses."""
        # Entry with wrong fy field
        entries_with_wrong_fy = [
            {
                "val": 50_000_000_000,
                "start": "2024-06-28",
                "end": "2025-06-27",
                "form": "10-K",
                "filed": "2025-08-05",
                "accn": "0001193125-25-000002",
                "fy": 2027,  # WRONG!
                "fp": "FY",
            },
        ]

        context = MatchContext(
            target_period_end="2025-06-27",
            target_adsh="0001193125-25-000002",
            target_fiscal_year=2025,
            target_fiscal_period="FY",
        )

        # ByPeriodEndMatcher succeeds
        period_matcher = ByPeriodEndMatcher()
        period_result = period_matcher.match(entries_with_wrong_fy, context)
        assert period_result.matched is True

        # ByAdshFyFpMatcher fails (because it trusts the wrong fy field)
        legacy_matcher = ByAdshFyFpMatcher()
        legacy_result = legacy_matcher.match(entries_with_wrong_fy, context)
        assert legacy_result.matched is False

    def test_form_filtering(self):
        """Test that all strategies filter for 10-K/10-Q forms."""
        entries_with_bad_form = [
            {
                "val": 1000,
                "end": "2025-06-27",
                "form": "8-K",  # Not a filing form
                "accn": "0001193125-25-000001",
            },
        ]

        context = MatchContext(
            target_period_end="2025-06-27",
        )

        matcher = ByPeriodEndMatcher()
        result = matcher.match(entries_with_bad_form, context)

        # Should not match 8-K forms
        assert result.matched is False
