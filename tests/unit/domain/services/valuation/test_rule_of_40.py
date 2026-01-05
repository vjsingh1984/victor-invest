"""
Unit tests for rule_of_40_valuation module.

Tests Rule of 40 based P/S valuation.
"""

import pytest

from investigator.domain.services.valuation.models.base import (
    ModelNotApplicable,
    ValuationModelResult,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile
from investigator.domain.services.valuation.rule_of_40_valuation import (
    Rule40Benchmarks,
    RuleOf40Valuation,
    calculate_rule_of_40_score,
)


class TestCalculateRuleOf40Score:
    """Tests for calculate_rule_of_40_score helper function."""

    def test_decimal_inputs(self):
        """Test score calculation with decimal inputs."""
        score = calculate_rule_of_40_score(0.25, 0.15)
        assert score == 40.0  # 25% + 15%

    def test_percentage_inputs(self):
        """Test score calculation with percentage inputs."""
        score = calculate_rule_of_40_score(25.0, 15.0)
        assert score == 40.0

    def test_mixed_inputs(self):
        """Test score calculation with mixed inputs."""
        score = calculate_rule_of_40_score(0.25, 15.0)
        # Should normalize correctly
        assert score == 40.0

    def test_negative_margin(self):
        """Test score calculation with negative margin."""
        score = calculate_rule_of_40_score(0.50, -0.10)
        assert score == 40.0  # 50% - 10%

    def test_negative_growth(self):
        """Test score calculation with negative growth."""
        score = calculate_rule_of_40_score(-0.10, 0.30)
        assert score == 20.0  # -10% + 30%


class TestRuleOf40Valuation:
    """Tests for RuleOf40Valuation class."""

    @pytest.fixture
    def company_profile(self):
        """Create test company profile."""
        return CompanyProfile(
            symbol="ZS",
            sector="Technology",
            industry="Software - Infrastructure",
        )

    @pytest.fixture
    def model(self, company_profile):
        """Create Rule of 40 valuation model."""
        return RuleOf40Valuation(company_profile)

    def test_exceptional_score(self, model):
        """Test valuation with exceptional Rule of 40 score (>=60)."""
        result = model.calculate(
            revenue_growth=0.40,  # 40%
            fcf_margin=0.25,  # 25%
            current_revenue=2_000_000_000,
            shares_outstanding=150_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.fair_value > 0
        assert result.assumptions["rule_40_score"] == 65.0
        assert result.assumptions["score_classification"] == "exceptional"

    def test_healthy_score(self, model):
        """Test valuation with healthy Rule of 40 score (>=40)."""
        result = model.calculate(
            revenue_growth=0.25,  # 25%
            fcf_margin=0.15,  # 15%
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions["rule_40_score"] == 40.0
        assert result.assumptions["score_classification"] == "healthy"

    def test_weak_score(self, model):
        """Test valuation with weak Rule of 40 score (20-30)."""
        result = model.calculate(
            revenue_growth=0.15,  # 15%
            fcf_margin=0.10,  # 10%
            current_revenue=500_000_000,
            shares_outstanding=50_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions["rule_40_score"] == 25.0
        assert result.assumptions["score_classification"] == "weak"

    def test_distressed_score(self, model):
        """Test valuation with very low Rule of 40 score (<20)."""
        result = model.calculate(
            revenue_growth=0.05,
            fcf_margin=-0.10,  # Negative margin
            current_revenue=500_000_000,
            shares_outstanding=50_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions["rule_40_score"] < 0
        assert result.assumptions["score_classification"] == "distressed"

    def test_missing_revenue_growth(self, model):
        """Test handling of missing revenue growth."""
        result = model.calculate(
            revenue_growth=None,
            fcf_margin=0.15,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ModelNotApplicable)
        assert "revenue growth" in result.reason.lower()

    def test_missing_fcf_margin(self, model):
        """Test handling of missing FCF margin."""
        result = model.calculate(
            revenue_growth=0.25,
            fcf_margin=None,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ModelNotApplicable)
        assert "fcf" in result.reason.lower()

    def test_missing_revenue(self, model):
        """Test handling of missing revenue."""
        result = model.calculate(
            revenue_growth=0.25,
            fcf_margin=0.15,
            current_revenue=None,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ModelNotApplicable)
        assert "revenue" in result.reason.lower()

    def test_missing_shares(self, model):
        """Test handling of missing shares outstanding."""
        result = model.calculate(
            revenue_growth=0.25,
            fcf_margin=0.15,
            current_revenue=1_000_000_000,
            shares_outstanding=None,
        )

        assert isinstance(result, ModelNotApplicable)
        assert "shares" in result.reason.lower()

    def test_upside_calculation(self, model):
        """Test upside potential calculation."""
        result = model.calculate(
            revenue_growth=0.25,
            fcf_margin=0.15,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
            current_price=50.0,
        )

        assert isinstance(result, ValuationModelResult)
        assert "upside_potential_pct" in result.metadata

    def test_confidence_varies_with_score(self, model):
        """Test that confidence varies appropriately with score."""
        # High score should have higher confidence
        result_high = model.calculate(
            revenue_growth=0.30,
            fcf_margin=0.15,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        # Extreme score should have lower confidence
        result_extreme = model.calculate(
            revenue_growth=0.80,  # Very high
            fcf_margin=0.30,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        assert result_high.confidence_score >= result_extreme.confidence_score

    def test_industry_benchmarks(self, model):
        """Test that industry benchmarks are used."""
        result = model.calculate(
            revenue_growth=0.25,
            fcf_margin=0.15,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        # Should include benchmark info
        assert "benchmark_median_score" in result.assumptions


class TestIndustryBenchmarks:
    """Tests for industry-specific benchmarks."""

    def test_enterprise_saas_benchmarks(self):
        """Test Enterprise SaaS benchmarks."""
        profile = CompanyProfile(
            symbol="CRM",
            sector="Technology",
            industry="SaaS - Enterprise",
        )
        model = RuleOf40Valuation(profile)

        result = model.calculate(
            revenue_growth=0.20,
            fcf_margin=0.25,
            current_revenue=30_000_000_000,
            shares_outstanding=1_000_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        # Enterprise SaaS should have higher benchmark
        assert result.assumptions["benchmark_median_ps"] >= 6.0

    def test_smb_saas_benchmarks(self):
        """Test SMB SaaS benchmarks."""
        profile = CompanyProfile(
            symbol="BILL",
            sector="Technology",
            industry="SaaS - SMB",
        )
        model = RuleOf40Valuation(profile)

        result = model.calculate(
            revenue_growth=0.25,
            fcf_margin=0.10,
            current_revenue=1_000_000_000,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ValuationModelResult)
