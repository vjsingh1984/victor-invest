"""
Unit tests for saas_valuation module.

Tests SaaS-specific metrics valuation.
"""

import pytest
from investigator.domain.services.valuation.models.saas_valuation import (
    SaaSValuationModel,
    SaaSMetrics,
)
from investigator.domain.services.valuation.models.base import (
    ValuationModelResult,
    ModelNotApplicable,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile


class TestSaaSMetrics:
    """Tests for SaaSMetrics dataclass."""

    def test_default_values(self):
        """Test default SaaSMetrics values."""
        metrics = SaaSMetrics()
        assert metrics.nrr is None
        assert metrics.ltv_cac is None
        assert metrics.gross_margin is None

    def test_custom_values(self):
        """Test SaaSMetrics with custom values."""
        metrics = SaaSMetrics(
            nrr=1.20,
            ltv_cac=5.0,
            gross_margin=0.80
        )
        assert metrics.nrr == 1.20
        assert metrics.ltv_cac == 5.0
        assert metrics.gross_margin == 0.80


class TestSaaSValuationModel:
    """Tests for SaaSValuationModel class."""

    @pytest.fixture
    def company_profile(self):
        """Create test company profile."""
        return CompanyProfile(
            symbol='CRWD',
            sector='Technology',
            industry='Software - Infrastructure',
        )

    @pytest.fixture
    def model(self, company_profile):
        """Create SaaS valuation model."""
        return SaaSValuationModel(company_profile)

    def test_high_growth_tier(self, model):
        """Test valuation for hyper-growth SaaS."""
        result = model.calculate(
            current_revenue=3_000_000_000,
            revenue_growth=0.60,  # 60% growth
            gross_margin=0.80,
            nrr=1.25,            # 125% NRR
            ltv_cac=5.0,
            fcf_margin=0.15,
            shares_outstanding=250_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions['growth_tier'] == 'hyper_growth'
        assert result.assumptions['base_ps_multiple'] == 15.0

    def test_growth_tier(self, model):
        """Test valuation for growth-stage SaaS."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.25,  # 25% growth
            gross_margin=0.75,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions['growth_tier'] == 'growth'
        assert result.assumptions['base_ps_multiple'] == 7.0

    def test_moderate_tier(self, model):
        """Test valuation for moderate-growth SaaS."""
        result = model.calculate(
            current_revenue=2_000_000_000,
            revenue_growth=0.10,  # 10% growth
            gross_margin=0.70,
            shares_outstanding=200_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions['growth_tier'] == 'moderate'
        assert result.assumptions['base_ps_multiple'] == 5.0

    def test_nrr_positive_adjustment(self, model):
        """Test positive adjustment for excellent NRR."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            nrr=1.35,  # Excellent NRR (135%)
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        # Should have positive NRR adjustment
        assert result.assumptions['adjustments']['nrr']['adjustment'] > 0

    def test_nrr_negative_adjustment(self, model):
        """Test negative adjustment for poor NRR."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            nrr=0.85,  # Poor NRR (85% - net churn)
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        # Should have negative NRR adjustment
        assert result.assumptions['adjustments']['nrr']['adjustment'] < 0

    def test_ltv_cac_adjustment(self, model):
        """Test LTV/CAC ratio adjustment."""
        # Good LTV/CAC
        result_good = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            ltv_cac=6.0,  # Excellent
            shares_outstanding=100_000_000,
        )

        # Poor LTV/CAC
        result_poor = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            ltv_cac=1.5,  # Poor
            shares_outstanding=100_000_000,
        )

        assert result_good.assumptions['adjustments']['ltv_cac']['adjustment'] > 0
        assert result_poor.assumptions['adjustments']['ltv_cac']['adjustment'] < 0

    def test_rule_of_40_adjustment(self, model):
        """Test Rule of 40 adjustment."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.35,  # 35%
            fcf_margin=0.15,      # 15%
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ValuationModelResult)
        # Rule of 40 score = 50, should have positive adjustment
        assert 'rule_of_40' in result.assumptions['adjustments']
        assert result.assumptions['adjustments']['rule_of_40']['value'] == 50.0

    def test_gross_margin_adjustment(self, model):
        """Test gross margin adjustment."""
        # High margin
        result_high = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            gross_margin=0.85,  # 85% - excellent
            shares_outstanding=100_000_000,
        )

        # Low margin
        result_low = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            gross_margin=0.55,  # 55% - low for SaaS
            shares_outstanding=100_000_000,
        )

        assert result_high.assumptions['adjustments']['gross_margin']['adjustment'] > 0
        assert result_low.assumptions['adjustments']['gross_margin']['adjustment'] < 0

    def test_adjustment_cap(self, model):
        """Test that total adjustment is capped."""
        # Maximum positive adjustments
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.60,  # Hyper growth
            nrr=1.40,            # Maximum NRR
            ltv_cac=7.0,         # Maximum LTV/CAC
            gross_margin=0.90,   # Maximum margin
            fcf_margin=0.30,     # Excellent margin
            shares_outstanding=100_000_000,
        )

        # Total adjustment should be capped at 100%
        assert result.assumptions['total_adjustment'] <= 1.00

    def test_saas_metrics_container(self, model):
        """Test using SaaSMetrics container."""
        metrics = SaaSMetrics(
            nrr=1.20,
            ltv_cac=4.5,
            gross_margin=0.78
        )

        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.25,
            shares_outstanding=100_000_000,
            saas_metrics=metrics,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions['adjustments']['nrr']['value'] == 1.20
        assert result.assumptions['adjustments']['ltv_cac']['value'] == 4.5

    def test_missing_revenue(self, model):
        """Test handling of missing revenue."""
        result = model.calculate(
            current_revenue=None,
            revenue_growth=0.25,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ModelNotApplicable)
        assert 'revenue' in result.reason.lower()

    def test_missing_growth(self, model):
        """Test handling of missing revenue growth."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=None,
            shares_outstanding=100_000_000,
        )

        assert isinstance(result, ModelNotApplicable)
        assert 'growth' in result.reason.lower()

    def test_missing_shares(self, model):
        """Test handling of missing shares outstanding."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.25,
            shares_outstanding=None,
        )

        assert isinstance(result, ModelNotApplicable)
        assert 'shares' in result.reason.lower()

    def test_upside_potential(self, model):
        """Test upside potential calculation."""
        result = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.30,
            shares_outstanding=100_000_000,
            current_price=50.0,
        )

        assert isinstance(result, ValuationModelResult)
        assert 'upside_potential_pct' in result.metadata
        assert 'current_price' in result.metadata

    def test_confidence_varies_with_metrics(self, model):
        """Test that confidence increases with more metrics."""
        # Minimal metrics
        result_minimal = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.25,
            shares_outstanding=100_000_000,
        )

        # Full metrics
        result_full = model.calculate(
            current_revenue=1_000_000_000,
            revenue_growth=0.25,
            nrr=1.15,
            ltv_cac=4.0,
            gross_margin=0.75,
            fcf_margin=0.15,
            shares_outstanding=100_000_000,
        )

        assert result_full.confidence_score > result_minimal.confidence_score
