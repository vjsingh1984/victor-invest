"""
Unit tests for profitability_classifier module.

Tests multi-indicator profitability classification.
"""

import pytest
from investigator.domain.services.profitability_classifier import (
    ProfitabilityClassifier,
    ProfitabilityStage,
    ProfitabilityClassification,
    get_profitability_classifier,
)


class TestProfitabilityStage:
    """Tests for ProfitabilityStage enum."""

    def test_all_stages_exist(self):
        """Test all profitability stages are defined."""
        assert ProfitabilityStage.PROFITABLE
        assert ProfitabilityStage.MARGINALLY_PROFITABLE
        assert ProfitabilityStage.TRANSITIONING
        assert ProfitabilityStage.PRE_PROFIT
        assert ProfitabilityStage.UNKNOWN


class TestProfitabilityClassification:
    """Tests for ProfitabilityClassification dataclass."""

    def test_creation(self):
        """Test classification dataclass creation."""
        classification = ProfitabilityClassification(
            stage=ProfitabilityStage.PROFITABLE,
            confidence=0.90,
            indicators_checked=[],
            indicators_positive=2,
            indicators_total=2,
            primary_indicator='net_income',
            applicable_models=['dcf', 'pe'],
            model_adjustments={'dcf': 1.0, 'pe': 1.0},
            notes=[]
        )
        assert classification.stage == ProfitabilityStage.PROFITABLE
        assert classification.confidence == 0.90


class TestProfitabilityClassifier:
    """Tests for ProfitabilityClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create default classifier."""
        return ProfitabilityClassifier()

    def test_profitable_company(self, classifier):
        """Test classification of clearly profitable company."""
        financials = {
            'net_income': 1_000_000_000,
            'operating_income': 1_200_000_000,
            'ebitda': 1_500_000_000,
            'free_cash_flow': 800_000_000,
            'total_revenue': 10_000_000_000,
        }
        ratios = {
            'net_margin': 10.0,  # 10% margin (as percentage, not decimal)
            'operating_margin': 12.0,
            'fcf_margin': 8.0,
        }

        result = classifier.classify(financials, ratios)

        assert result.stage == ProfitabilityStage.PROFITABLE
        assert result.confidence >= 0.8

    def test_pre_profit_company(self, classifier):
        """Test classification of pre-profit company."""
        financials = {
            'net_income': -500_000_000,
            'operating_income': -400_000_000,
            'ebitda': -200_000_000,
            'free_cash_flow': -600_000_000,
            'total_revenue': 1_000_000_000,
        }
        ratios = {
            'net_margin': -0.50,
            'operating_margin': -0.40,
            'fcf_margin': -0.60,
        }

        result = classifier.classify(financials, ratios)

        assert result.stage == ProfitabilityStage.PRE_PROFIT

    def test_marginally_profitable(self, classifier):
        """Test classification of marginally profitable company."""
        financials = {
            'net_income': 10_000_000,  # Small positive
            'operating_income': 15_000_000,
            'total_revenue': 1_000_000_000,
        }
        ratios = {
            'net_margin': 0.01,  # 1% margin - marginal
            'operating_margin': 0.015,
        }

        result = classifier.classify(financials, ratios)

        assert result.stage in [
            ProfitabilityStage.MARGINALLY_PROFITABLE,
            ProfitabilityStage.PROFITABLE
        ]

    def test_transitioning_company(self, classifier):
        """Test classification of transitioning company (mixed signals)."""
        financials = {
            'net_income': -100_000_000,  # Still negative
            'operating_income': 50_000_000,  # But operating income positive
            'ebitda': 200_000_000,  # Strong EBITDA
            'free_cash_flow': 100_000_000,  # Positive FCF
            'total_revenue': 2_000_000_000,
        }
        ratios = {
            'net_margin': -0.05,
            'operating_margin': 0.025,
            'fcf_margin': 0.05,
        }

        result = classifier.classify(financials, ratios)

        # Should recognize positive signals despite net loss
        assert result.stage in [
            ProfitabilityStage.TRANSITIONING,
            ProfitabilityStage.MARGINALLY_PROFITABLE
        ]
        assert len(result.notes) > 0  # Should have notes about mixed signals

    def test_missing_data(self, classifier):
        """Test classification with missing financial data."""
        financials = {
            'total_revenue': 1_000_000_000,
            # All profit indicators missing
        }
        ratios = {}

        result = classifier.classify(financials, ratios)

        assert result.stage == ProfitabilityStage.UNKNOWN
        assert result.confidence < 0.5

    def test_only_ebitda_available(self, classifier):
        """Test classification when only EBITDA is available."""
        financials = {
            'ebitda': 500_000_000,
            'total_revenue': 2_000_000_000,
        }
        ratios = {}

        result = classifier.classify(financials, ratios)

        # Should still classify based on EBITDA
        assert result.stage != ProfitabilityStage.UNKNOWN
        assert any(i.name == 'ebitda' and i.is_positive for i in result.indicators_checked)

    def test_only_fcf_available(self, classifier):
        """Test classification when only FCF is available."""
        financials = {
            'free_cash_flow': 300_000_000,
            'total_revenue': 3_000_000_000,
        }
        ratios = {
            'fcf_margin': 10.0,  # 10% margin as percentage
        }

        result = classifier.classify(financials, ratios)

        # With only 1 positive indicator out of 4, it's transitioning
        # (needs majority positive for PROFITABLE/MARGINALLY_PROFITABLE)
        assert result.stage in [
            ProfitabilityStage.PROFITABLE,
            ProfitabilityStage.MARGINALLY_PROFITABLE,
            ProfitabilityStage.TRANSITIONING
        ]

    def test_none_values_handled(self, classifier):
        """Test that None values are handled gracefully."""
        financials = {
            'net_income': None,
            'operating_income': None,
            'ebitda': 100_000_000,
            'total_revenue': 1_000_000_000,
        }
        ratios = {
            'net_margin': None,
            'operating_margin': None,
        }

        result = classifier.classify(financials, ratios)

        # Should not crash, should use available data
        assert result is not None
        assert result.stage != ProfitabilityStage.UNKNOWN or result.confidence < 0.5

    def test_zero_revenue(self, classifier):
        """Test classification with zero revenue."""
        financials = {
            'net_income': -1_000_000,
            'total_revenue': 0,
            'revenue': 0,
        }
        ratios = {}

        result = classifier.classify(financials, ratios)

        # Zero revenue and negative net income = pre-profit
        assert result.stage == ProfitabilityStage.PRE_PROFIT

    def test_negative_ebitda_positive_fcf(self, classifier):
        """Test case where EBITDA negative but FCF positive."""
        financials = {
            'net_income': -50_000_000,
            'ebitda': -20_000_000,
            'free_cash_flow': 30_000_000,  # Positive due to working capital
            'total_revenue': 500_000_000,
        }
        ratios = {
            'fcf_margin': 0.06,
        }

        result = classifier.classify(financials, ratios)

        # Should note the discrepancy
        assert len(result.notes) > 0

    def test_singleton_classifier(self):
        """Test singleton get_profitability_classifier function."""
        classifier1 = get_profitability_classifier()
        classifier2 = get_profitability_classifier()
        assert classifier1 is classifier2

    def test_high_confidence_for_clear_signals(self, classifier):
        """Test confidence is high when all indicators agree."""
        financials = {
            'net_income': 500_000_000,
            'operating_income': 600_000_000,
            'ebitda': 800_000_000,
            'free_cash_flow': 400_000_000,
            'total_revenue': 5_000_000_000,
        }
        ratios = {
            'net_margin': 0.10,
            'operating_margin': 0.12,
            'fcf_margin': 0.08,
        }

        result = classifier.classify(financials, ratios)

        assert result.confidence >= 0.85
        assert result.indicators_positive >= 3
