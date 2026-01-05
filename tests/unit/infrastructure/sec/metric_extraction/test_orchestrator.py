"""
Unit tests for MetricExtractionOrchestrator.

Tests the multi-level fallback chain orchestration for robust metric extraction.

Note: conftest.py handles mocking of parent package imports to avoid circular
import issues during test collection.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from investigator.infrastructure.sec.metric_extraction.orchestrator import (
    MetricExtractionOrchestrator,
)

# Import the modules directly (conftest.py handles parent package mocking)
from investigator.infrastructure.sec.metric_extraction.result import (
    ExtractionConfidence,
    ExtractionResult,
    MatchMethod,
)
from investigator.infrastructure.sec.metric_extraction.strategies import (
    ByAdshFyFpMatcher,
    ByAdshOnlyMatcher,
    ByDateRangeMatcher,
    ByFrameFieldMatcher,
    ByPeriodEndMatcher,
    MatchContext,
)

# Mock us-gaap data structure simulating SEC CompanyFacts
MOCK_US_GAAP = {
    "Revenues": {
        "label": "Revenues",
        "units": {
            "USD": [
                # FY 2025 - correct fy field
                {
                    "val": 50_000_000_000,
                    "start": "2024-06-28",
                    "end": "2025-06-27",
                    "form": "10-K",
                    "filed": "2025-08-05",
                    "accn": "0001193125-25-000001",
                    "fy": 2025,
                    "fp": "FY",
                },
                # FY 2024
                {
                    "val": 45_000_000_000,
                    "start": "2023-06-30",
                    "end": "2024-06-28",
                    "form": "10-K",
                    "filed": "2024-08-05",
                    "accn": "0001193125-24-000001",
                    "fy": 2024,
                    "fp": "FY",
                },
            ]
        },
    },
    "OperatingIncomeLoss": {
        "label": "Operating Income",
        "units": {
            "USD": [
                {
                    "val": 15_000_000_000,
                    "start": "2024-06-28",
                    "end": "2025-06-27",
                    "form": "10-K",
                    "filed": "2025-08-05",
                    "accn": "0001193125-25-000001",
                    "fy": 2025,
                    "fp": "FY",
                },
            ]
        },
    },
}

# Mock us-gaap with WRONG fy field (simulating STX bug)
MOCK_US_GAAP_WRONG_FY = {
    "Revenues": {
        "label": "Revenues",
        "units": {
            "USD": [
                # FY 2025 - WRONG fy field (fy=2027 instead of 2025)
                {
                    "val": 50_000_000_000,
                    "start": "2024-06-28",
                    "end": "2025-06-27",
                    "form": "10-K",
                    "filed": "2025-08-05",
                    "accn": "0001193125-25-000001",
                    "fy": 2027,  # SEC bug!
                    "fp": "FY",
                },
            ]
        },
    },
}


class MockCanonicalMapper:
    """Mock canonical mapper for testing."""

    def __init__(self):
        self.mappings = {
            "total_revenue": {
                "tags": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                "unit": "USD",
            },
            "operating_income": {
                "tags": ["OperatingIncomeLoss"],
                "unit": "USD",
            },
            "free_cash_flow": {
                "tags": [],  # No direct tags
                "unit": "USD",
                "derived": {
                    "enabled": True,
                    "formula": "operating_cash_flow - capital_expenditures",
                    "required_fields": ["operating_cash_flow", "capital_expenditures"],
                },
            },
        }

    def get_tags(self, canonical_key, sector=None, industry=None):
        """Return fallback tag chain for canonical key."""
        mapping = self.mappings.get(canonical_key, {})
        return mapping.get("tags", [])


class TestMetricExtractionOrchestrator:
    """Tests for MetricExtractionOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mapper = MockCanonicalMapper()
        self.orchestrator = MetricExtractionOrchestrator(
            sector="Technology",
            industry="Electronic Components",
            canonical_mapper=self.mock_mapper,
            enable_audit=True,
        )

    def test_init(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.sector == "Technology"
        assert self.orchestrator.industry == "Electronic Components"
        assert len(self.orchestrator.matchers) == 5  # Default matchers

    def test_extract_by_period_end(self):
        """Test extraction using period end date matching."""
        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
            target_fiscal_year=2025,
            target_fiscal_period="FY",
        )

        assert result.success is True
        assert result.value == 50_000_000_000
        assert result.source_tag == "Revenues"
        assert result.match_method == MatchMethod.BY_PERIOD_END
        assert result.confidence in [ExtractionConfidence.HIGH, ExtractionConfidence.MEDIUM]

    def test_extract_with_wrong_fy_field(self):
        """
        Critical test: extraction succeeds even when SEC fy field is wrong.

        This is the STX bug fix - we match by period_end, not fy.
        """
        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP_WRONG_FY,
            target_period_end="2025-06-27",
            target_fiscal_year=2025,  # We expect FY 2025
            target_fiscal_period="FY",
        )

        # Should succeed because ByPeriodEndMatcher uses end date, not fy
        assert result.success is True
        assert result.value == 50_000_000_000
        # The entry has fy=2027 (wrong), but we found it anyway
        assert result.sec_fy == 2027  # Reflects what SEC says (wrong)

    def test_extract_not_found(self):
        """Test extraction when no matching entry exists."""
        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2030-06-27",  # Future date, no entries
            target_fiscal_year=2030,
            target_fiscal_period="FY",
        )

        assert result.success is False
        assert result.value is None
        assert result.match_method == MatchMethod.NOT_FOUND
        assert result.error is not None

    def test_extract_no_tags_configured(self):
        """Test extraction when no XBRL tags are configured for canonical key."""
        result = self.orchestrator.extract(
            canonical_key="unknown_metric",  # Not in mapper
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
        )

        assert result.success is False
        assert "No XBRL tags" in result.error

    def test_extract_tag_not_in_us_gaap(self):
        """Test extraction when configured tag doesn't exist in us-gaap."""
        # Empty us-gaap
        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap={},
            target_period_end="2025-06-27",
        )

        assert result.success is False

    def test_audit_trail_enabled(self):
        """Test that audit trail captures extraction attempts."""
        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        assert result.audit is not None
        assert result.audit.canonical_key == "total_revenue"
        assert result.audit.target_period_end == "2025-06-27"
        assert len(result.audit.attempts) > 0

    def test_audit_trail_disabled(self):
        """Test orchestrator with audit trail disabled."""
        orchestrator = MetricExtractionOrchestrator(
            canonical_mapper=self.mock_mapper,
            enable_audit=False,
        )

        result = orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
        )

        assert result.success is True
        assert result.audit is None

    def test_statistics_tracking(self):
        """Test that extraction statistics are tracked."""
        # Initial stats
        assert self.orchestrator.stats["extractions"] == 0
        assert self.orchestrator.stats["successes"] == 0

        # Perform extraction
        self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
        )

        # Stats should be updated
        assert self.orchestrator.stats["extractions"] == 1
        assert self.orchestrator.stats["successes"] == 1

    def test_get_stats(self):
        """Test get_stats returns formatted statistics."""
        # Perform some extractions
        self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
        )

        stats = self.orchestrator.get_stats()

        assert "extractions" in stats
        assert "successes" in stats
        assert "success_rate" in stats
        assert "100.0%" in stats["success_rate"]

    def test_fallback_to_secondary_tag(self):
        """Test fallback to secondary XBRL tag when primary not found."""
        # us-gaap without primary tag, but with fallback
        us_gaap_with_fallback = {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue from Contract",
                "units": {
                    "USD": [
                        {
                            "val": 48_000_000_000,
                            "start": "2024-06-28",
                            "end": "2025-06-27",
                            "form": "10-K",
                            "fy": 2025,
                            "fp": "FY",
                        },
                    ]
                },
            },
        }

        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=us_gaap_with_fallback,
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        assert result.success is True
        assert result.value == 48_000_000_000
        # Used fallback tag
        assert result.source_tag == "RevenueFromContractWithCustomerExcludingAssessedTax"

    def test_confidence_levels(self):
        """Test confidence level assignment based on strategy and tag position."""
        # First tag, first strategy = HIGH
        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=MOCK_US_GAAP,
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        assert result.confidence == ExtractionConfidence.HIGH

    def test_select_best_entry_prefers_annual_for_fy(self):
        """Test that annual entries are preferred for FY fiscal period."""
        us_gaap_mixed = {
            "Revenues": {
                "units": {
                    "USD": [
                        # Quarterly entry
                        {
                            "val": 12_000_000_000,
                            "start": "2025-03-28",
                            "end": "2025-06-27",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q4",
                        },
                        # Annual entry
                        {
                            "val": 50_000_000_000,
                            "start": "2024-06-28",
                            "end": "2025-06-27",
                            "form": "10-K",
                            "fy": 2025,
                            "fp": "FY",
                        },
                    ]
                },
            },
        }

        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=us_gaap_mixed,
            target_period_end="2025-06-27",
            target_fiscal_period="FY",
        )

        # Should select annual (50B) not quarterly (12B)
        assert result.value == 50_000_000_000

    def test_select_best_entry_prefers_quarterly_for_q(self):
        """Test that quarterly entries are preferred for Q1/Q2/Q3/Q4."""
        us_gaap_mixed = {
            "Revenues": {
                "units": {
                    "USD": [
                        # Quarterly entry (preferred for Q4)
                        {
                            "val": 12_000_000_000,
                            "start": "2025-03-28",
                            "end": "2025-06-27",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q4",
                        },
                        # YTD entry (less preferred)
                        {
                            "val": 50_000_000_000,
                            "start": "2024-06-28",
                            "end": "2025-06-27",
                            "form": "10-K",
                            "fy": 2025,
                            "fp": "FY",
                        },
                    ]
                },
            },
        }

        result = self.orchestrator.extract(
            canonical_key="total_revenue",
            us_gaap=us_gaap_mixed,
            target_period_end="2025-06-27",
            target_fiscal_period="Q4",
        )

        # Should select quarterly (12B) not annual (50B)
        assert result.value == 12_000_000_000


class TestOrchestratorWithRealMapper:
    """Integration-style tests with real canonical mapper (if available)."""

    def test_orchestrator_creates_default_mapper(self):
        """Test that orchestrator creates mapper if not provided."""
        # This will try to import the real canonical mapper
        try:
            orchestrator = MetricExtractionOrchestrator(
                sector="Technology",
                enable_audit=False,
            )
            # If it works, canonical_mapper should be set
            assert orchestrator.canonical_mapper is not None
        except ImportError:
            # Skip if canonical mapper not available
            pytest.skip("Canonical mapper not available")


class TestDerivedValueCalculation:
    """Tests for derived value calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Custom mapper with derived field config
        self.mock_mapper = Mock()
        self.mock_mapper.get_tags = Mock(return_value=[])
        self.mock_mapper.mappings = {
            "free_cash_flow": {
                "tags": [],
                "unit": "USD",
                "derived": {
                    "enabled": True,
                    "formula": "operating_cash_flow - capital_expenditures",
                    "required_fields": ["operating_cash_flow", "capital_expenditures"],
                },
            },
            "operating_cash_flow": {
                "tags": ["NetCashProvidedByUsedInOperatingActivities"],
                "unit": "USD",
            },
            "capital_expenditures": {
                "tags": ["PaymentsToAcquirePropertyPlantAndEquipment"],
                "unit": "USD",
            },
        }

        def get_tags_side_effect(key, sector=None, industry=None):
            mapping = self.mock_mapper.mappings.get(key, {})
            return mapping.get("tags", [])

        self.mock_mapper.get_tags = Mock(side_effect=get_tags_side_effect)

    def test_evaluate_formula_safe(self):
        """Test safe formula evaluation."""
        orchestrator = MetricExtractionOrchestrator(
            canonical_mapper=self.mock_mapper,
            enable_audit=False,
        )

        components = {
            "operating_cash_flow": 8_000_000_000,
            "capital_expenditures": 3_000_000_000,
        }

        result = orchestrator._evaluate_formula("operating_cash_flow - capital_expenditures", components)

        assert result == 5_000_000_000

    def test_evaluate_formula_rejects_unsafe(self):
        """Test that unsafe formulas are rejected."""
        orchestrator = MetricExtractionOrchestrator(
            canonical_mapper=self.mock_mapper,
            enable_audit=False,
        )

        components = {"x": 1}

        # Formula with unsafe characters
        result = orchestrator._evaluate_formula("import os; x", components)

        assert result is None
