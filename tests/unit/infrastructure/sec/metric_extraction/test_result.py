"""
Unit tests for metric extraction result types.

Tests ExtractionResult, ExtractionAudit, and related dataclasses.
"""

import pytest

from investigator.infrastructure.sec.metric_extraction.result import (
    ExtractionResult,
    ExtractionAudit,
    ExtractionAttempt,
    MatchMethod,
    ExtractionConfidence,
)


class TestMatchMethod:
    """Tests for MatchMethod enum."""

    def test_all_methods_have_values(self):
        """Verify all match methods have string values."""
        assert MatchMethod.BY_PERIOD_END.value == "by_period_end"
        assert MatchMethod.BY_DATE_RANGE.value == "by_date_range"
        assert MatchMethod.BY_FRAME_FIELD.value == "by_frame_field"
        assert MatchMethod.BY_ADSH_FY_FP.value == "by_adsh_fy_fp"
        assert MatchMethod.BY_ADSH_ONLY.value == "by_adsh_only"
        assert MatchMethod.DERIVED.value == "derived"
        assert MatchMethod.NOT_FOUND.value == "not_found"


class TestExtractionConfidence:
    """Tests for ExtractionConfidence enum."""

    def test_all_confidence_levels_have_values(self):
        """Verify all confidence levels have string values."""
        assert ExtractionConfidence.HIGH.value == "high"
        assert ExtractionConfidence.MEDIUM.value == "medium"
        assert ExtractionConfidence.LOW.value == "low"
        assert ExtractionConfidence.DERIVED.value == "derived"
        assert ExtractionConfidence.NONE.value == "none"


class TestExtractionAttempt:
    """Tests for ExtractionAttempt dataclass."""

    def test_create_attempt(self):
        """Test creating an extraction attempt."""
        attempt = ExtractionAttempt(
            strategy_name="ByPeriodEndMatcher",
            tag_name="Revenues",
            matched=True,
            entries_found=3,
            selected_entry={"val": 1000000, "end": "2025-06-27"},
            reason=None,
            duration_ms=1.5,
        )

        assert attempt.strategy_name == "ByPeriodEndMatcher"
        assert attempt.tag_name == "Revenues"
        assert attempt.matched is True
        assert attempt.entries_found == 3
        assert attempt.selected_entry["val"] == 1000000
        assert attempt.duration_ms == 1.5

    def test_failed_attempt(self):
        """Test creating a failed extraction attempt."""
        attempt = ExtractionAttempt(
            strategy_name="ByAdshFyFpMatcher",
            tag_name="Revenues",
            matched=False,
            entries_found=0,
            reason="No entries with fy=2025, fp=FY",
        )

        assert attempt.matched is False
        assert attempt.entries_found == 0
        assert "No entries" in attempt.reason


class TestExtractionAudit:
    """Tests for ExtractionAudit dataclass."""

    def test_create_audit(self):
        """Test creating an extraction audit."""
        audit = ExtractionAudit(
            canonical_key="total_revenue",
            target_period_end="2025-06-27",
            target_fiscal_year=2025,
            target_fiscal_period="FY",
            started_at="2025-01-01T12:00:00",
        )

        assert audit.canonical_key == "total_revenue"
        assert audit.target_period_end == "2025-06-27"
        assert audit.target_fiscal_year == 2025
        assert audit.target_fiscal_period == "FY"
        assert audit.attempts == []
        assert audit.total_duration_ms == 0.0

    def test_add_attempt(self):
        """Test adding attempts to audit trail."""
        audit = ExtractionAudit(
            canonical_key="total_revenue",
            target_period_end="2025-06-27",
        )

        attempt1 = ExtractionAttempt(
            strategy_name="ByPeriodEndMatcher",
            tag_name="Revenues",
            matched=False,
            entries_found=0,
            reason="Tag not found",
            duration_ms=0.5,
        )

        attempt2 = ExtractionAttempt(
            strategy_name="ByPeriodEndMatcher",
            tag_name="RevenueFromContractWithCustomerExcludingAssessedTax",
            matched=True,
            entries_found=1,
            duration_ms=1.2,
        )

        audit.add_attempt(attempt1)
        audit.add_attempt(attempt2)

        assert len(audit.attempts) == 2
        assert audit.total_duration_ms == pytest.approx(1.7, rel=0.01)

    def test_summary_with_success(self):
        """Test audit summary when extraction succeeded."""
        audit = ExtractionAudit(
            canonical_key="total_revenue",
            target_period_end="2025-06-27",
            target_fiscal_year=2025,
            target_fiscal_period="FY",
        )

        audit.add_attempt(ExtractionAttempt(
            strategy_name="ByPeriodEndMatcher",
            tag_name="Revenues",
            matched=True,
            entries_found=1,
        ))

        summary = audit.summary()
        assert "total_revenue" in summary
        assert "ByPeriodEndMatcher" in summary
        assert "Revenues" in summary

    def test_summary_with_failure(self):
        """Test audit summary when extraction failed."""
        audit = ExtractionAudit(
            canonical_key="total_revenue",
            target_period_end="2025-06-27",
        )

        audit.add_attempt(ExtractionAttempt(
            strategy_name="ByPeriodEndMatcher",
            tag_name="Revenues",
            matched=False,
            entries_found=0,
        ))

        audit.add_attempt(ExtractionAttempt(
            strategy_name="ByDateRangeMatcher",
            tag_name="Revenues",
            matched=False,
            entries_found=0,
        ))

        summary = audit.summary()
        assert "Failed strategies" in summary or "0 succeeded" in summary


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_not_found_factory(self):
        """Test creating a not-found result."""
        result = ExtractionResult.not_found(
            canonical_key="total_revenue",
            reason="No matching entry found",
        )

        assert result.success is False
        assert result.value is None
        assert result.match_method == MatchMethod.NOT_FOUND
        assert result.confidence == ExtractionConfidence.NONE
        assert "No matching entry" in result.error

    def test_not_found_with_audit(self):
        """Test not-found result with audit trail."""
        audit = ExtractionAudit(
            canonical_key="total_revenue",
            target_period_end="2025-06-27",
        )

        result = ExtractionResult.not_found(
            canonical_key="total_revenue",
            audit=audit,
            reason="Exhausted all strategies",
        )

        assert result.audit is audit
        assert "Exhausted" in result.error

    def test_from_entry_factory(self):
        """Test creating result from SEC entry."""
        entry = {
            "val": 50_000_000_000,
            "start": "2024-06-28",
            "end": "2025-06-27",
            "form": "10-K",
            "filed": "2025-08-05",
            "accn": "0001193125-25-123456",
            "fy": 2025,
            "fp": "FY",
        }

        result = ExtractionResult.from_entry(
            value=50_000_000_000,
            source_tag="Revenues",
            entry=entry,
            match_method=MatchMethod.BY_PERIOD_END,
            confidence=ExtractionConfidence.HIGH,
        )

        assert result.success is True
        assert result.value == 50_000_000_000
        assert result.source_tag == "Revenues"
        assert result.match_method == MatchMethod.BY_PERIOD_END
        assert result.confidence == ExtractionConfidence.HIGH
        assert result.period_end == "2025-06-27"
        assert result.period_start == "2024-06-28"
        assert result.duration_days == 364  # Leap year consideration
        assert result.form == "10-K"
        assert result.sec_fy == 2025
        assert result.sec_fp == "FY"

    def test_from_entry_with_medium_confidence(self):
        """Test creating result with medium confidence (fallback tag)."""
        entry = {
            "val": 25_000_000_000,
            "end": "2025-06-27",
            "form": "10-K",
        }

        result = ExtractionResult.from_entry(
            value=25_000_000_000,
            source_tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            entry=entry,
            match_method=MatchMethod.BY_DATE_RANGE,
            confidence=ExtractionConfidence.MEDIUM,
        )

        assert result.success is True
        assert result.confidence == ExtractionConfidence.MEDIUM

    def test_derived_factory(self):
        """Test creating derived/calculated result."""
        result = ExtractionResult.derived(
            value=5_000_000_000,
            formula="operating_cash_flow - capital_expenditures",
            components={
                "operating_cash_flow": 8_000_000_000,
                "capital_expenditures": 3_000_000_000,
            },
        )

        assert result.success is True
        assert result.value == 5_000_000_000
        assert "derived:" in result.source_tag
        assert result.match_method == MatchMethod.DERIVED
        assert result.confidence == ExtractionConfidence.DERIVED

    def test_repr_success(self):
        """Test string representation of successful result."""
        entry = {"val": 1_000_000, "end": "2025-06-27"}
        result = ExtractionResult.from_entry(
            value=1_000_000,
            source_tag="Revenues",
            entry=entry,
            match_method=MatchMethod.BY_PERIOD_END,
        )

        repr_str = repr(result)
        assert "1,000,000" in repr_str
        assert "Revenues" in repr_str
        assert "by_period_end" in repr_str

    def test_repr_failure(self):
        """Test string representation of failed result."""
        result = ExtractionResult.not_found(
            canonical_key="total_revenue",
            reason="No data",
        )

        repr_str = repr(result)
        assert "success=False" in repr_str
        assert "No data" in repr_str

    def test_duration_days_calculation(self):
        """Test that duration days are calculated correctly."""
        entry = {
            "val": 10_000_000,
            "start": "2025-01-01",
            "end": "2025-03-31",
            "form": "10-Q",
        }

        result = ExtractionResult.from_entry(
            value=10_000_000,
            source_tag="Revenues",
            entry=entry,
            match_method=MatchMethod.BY_PERIOD_END,
        )

        # Jan 1 to Mar 31 = 89 days (non-leap year)
        assert result.duration_days == 89

    def test_duration_days_missing_dates(self):
        """Test handling when start/end dates are missing."""
        entry = {
            "val": 10_000_000,
            "form": "10-Q",
        }

        result = ExtractionResult.from_entry(
            value=10_000_000,
            source_tag="Revenues",
            entry=entry,
            match_method=MatchMethod.BY_PERIOD_END,
        )

        assert result.duration_days is None
