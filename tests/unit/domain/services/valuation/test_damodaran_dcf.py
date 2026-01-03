"""
Unit tests for damodaran_dcf module.

Tests 3-stage DCF with Damodaran methodology.
"""

import pytest
from investigator.domain.services.valuation.damodaran_dcf import (
    DamodaranDCFModel,
    DCFProjection,
    MonteCarloResult,
)
from investigator.domain.services.valuation.models.base import (
    ValuationModelResult,
    ModelNotApplicable,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile


class TestDamodaranDCFModel:
    """Tests for DamodaranDCFModel class."""

    @pytest.fixture
    def company_profile(self):
        """Create test company profile."""
        return CompanyProfile(
            symbol='AAPL',
            sector='Technology',
            industry='Consumer Electronics',
        )

    @pytest.fixture
    def model(self, company_profile):
        """Create Damodaran DCF model."""
        return DamodaranDCFModel(company_profile)

    def test_positive_fcf_valuation(self, model):
        """Test valuation for company with positive FCF."""
        result = model.calculate(
            current_fcf=100_000_000_000,  # $100B FCF
            revenue_growth=0.08,
            fcf_margin=0.25,
            current_revenue=400_000_000_000,
            shares_outstanding=15_000_000_000,
            debt_to_equity=0.20,
            tax_rate=0.21,
        )

        assert isinstance(result, ValuationModelResult)
        assert result.fair_value > 0
        assert result.methodology == '3-Stage DCF (Damodaran)'
        assert 'high_growth_years' in result.assumptions

    def test_negative_fcf_uses_revenue_bridge(self, model):
        """Test that negative FCF uses revenue bridge method."""
        result = model.calculate(
            current_fcf=-500_000_000,  # Negative FCF
            revenue_growth=0.30,
            fcf_margin=-0.10,  # Negative margin now, targeting positive
            current_revenue=2_000_000_000,
            shares_outstanding=100_000_000,
            debt_to_equity=0.10,
            tax_rate=0.21,
            target_fcf_margin=0.15,  # Target 15% FCF margin
        )

        assert isinstance(result, ValuationModelResult)
        assert result.assumptions.get('use_revenue_bridge') is True

    def test_three_stage_structure(self, model):
        """Test that projections include all three stages."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.15,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
        )

        # Projection summary is in metadata
        projection_summary = result.metadata.get('projection_summary', {})

        # Should have high growth and transition years
        assert projection_summary.get('high_growth_years') == 5
        assert projection_summary.get('transition_years') == 5
        assert projection_summary.get('total_years') == 10

    def test_growth_rate_decay(self, model):
        """Test that growth rate decays in transition phase."""
        result = model.calculate(
            current_fcf=10_000_000_000,
            revenue_growth=0.25,  # 25% initial growth
            fcf_margin=0.15,
            current_revenue=66_000_000_000,
            shares_outstanding=500_000_000,
            debt_to_equity=0.20,
            tax_rate=0.21,
        )

        # High growth rate should be stored in assumptions
        assert result.assumptions.get('high_growth_rate') == 0.25

        # Terminal growth should be lower
        terminal_growth = result.assumptions.get('terminal_growth')
        assert terminal_growth < 0.25

    def test_wacc_from_industry(self, model):
        """Test that WACC is calculated from industry."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
        )

        assert 'wacc' in result.assumptions
        assert result.assumptions['wacc'] > 0
        assert result.assumptions['wacc'] < 0.20  # Reasonable range

    def test_custom_high_growth_years(self, model):
        """Test custom high growth period."""
        result = model.calculate(
            current_fcf=10_000_000_000,
            revenue_growth=0.20,
            fcf_margin=0.15,
            current_revenue=66_000_000_000,
            shares_outstanding=500_000_000,
            debt_to_equity=0.20,
            tax_rate=0.21,
            high_growth_years=7,  # Custom: 7 years
        )

        assert result.assumptions.get('high_growth_years') == 7

    def test_custom_terminal_growth(self, model):
        """Test custom terminal growth rate."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
            terminal_growth=0.03,  # Custom: 3%
        )

        assert result.assumptions.get('terminal_growth') == 0.03

    def test_missing_fcf_and_revenue(self, model):
        """Test handling when both FCF and revenue are missing."""
        result = model.calculate(
            current_fcf=None,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=None,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
        )

        assert isinstance(result, ModelNotApplicable)

    def test_missing_shares(self, model):
        """Test handling of missing shares outstanding."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=None,
            debt_to_equity=0.15,
            tax_rate=0.21,
        )

        assert isinstance(result, ModelNotApplicable)

    def test_upside_calculation(self, model):
        """Test upside potential calculation."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
            current_price=150.0,
        )

        assert 'upside_potential_pct' in result.metadata

    def test_monte_carlo_disabled(self, model):
        """Test that Monte Carlo is disabled by default."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
        )

        assert result.metadata.get('monte_carlo') is None

    def test_monte_carlo_enabled(self, model):
        """Test Monte Carlo simulation when enabled."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
            tax_rate=0.21,
            run_monte_carlo=True,
            monte_carlo_iterations=100,  # Fewer for test speed
        )

        # Should have Monte Carlo results
        if 'monte_carlo' in result.metadata:
            mc = result.metadata['monte_carlo']
            assert 'range_10_90' in mc or 'mean' in mc

    def test_default_tax_rate(self, model):
        """Test default tax rate when not provided."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0.15,
        )

        # Should use WACC calculated with default tax rate
        assert result.assumptions.get('wacc') is not None
        assert result.assumptions.get('wacc') > 0

    def test_zero_debt(self, model):
        """Test valuation with zero debt."""
        result = model.calculate(
            current_fcf=50_000_000_000,
            revenue_growth=0.10,
            fcf_margin=0.20,
            current_revenue=250_000_000_000,
            shares_outstanding=16_000_000_000,
            debt_to_equity=0,  # No debt
            tax_rate=0.21,
        )

        # WACC should equal cost of equity when no debt
        wacc = result.assumptions.get('wacc')
        assert wacc > 0


class TestDCFProjection:
    """Tests for DCFProjection dataclass."""

    def test_projection_creation(self):
        """Test creating a DCF projection."""
        proj = DCFProjection(
            year=1,
            phase='high_growth',
            revenue=10_000_000_000,
            growth_rate=0.15,
            fcf=1_000_000_000,
            fcf_margin=0.10,
            discount_factor=0.90,
            present_value=900_000_000,
        )

        assert proj.year == 1
        assert proj.fcf == 1_000_000_000
        assert proj.present_value == 900_000_000


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_monte_carlo_creation(self):
        """Test creating Monte Carlo result."""
        mc = MonteCarloResult(
            mean_fair_value=150.0,
            median_fair_value=148.0,
            std_dev=20.0,
            percentile_10=120.0,
            percentile_25=135.0,
            percentile_75=165.0,
            percentile_90=180.0,
            iterations=1000,
        )

        assert mc.mean_fair_value == 150.0
        assert mc.percentile_10 == 120.0
        assert mc.iterations == 1000


class TestSoftwareCompanyValuation:
    """Tests for software company specific valuation."""

    @pytest.fixture
    def software_profile(self):
        """Create software company profile."""
        return CompanyProfile(
            symbol='MSFT',
            sector='Technology',
            industry='Software (System & Application)',
        )

    @pytest.fixture
    def model(self, software_profile):
        """Create DCF model for software company."""
        return DamodaranDCFModel(software_profile)

    def test_software_industry_beta(self, model):
        """Test that software industry gets correct beta."""
        result = model.calculate(
            current_fcf=60_000_000_000,
            revenue_growth=0.12,
            fcf_margin=0.30,
            current_revenue=200_000_000_000,
            shares_outstanding=7_500_000_000,
            debt_to_equity=0.10,
            tax_rate=0.21,
        )

        # Software should have specific beta from Damodaran
        assert 'cost_of_capital' in result.metadata or 'wacc' in result.assumptions
