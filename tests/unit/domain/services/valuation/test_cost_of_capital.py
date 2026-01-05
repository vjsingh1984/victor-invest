"""
Unit tests for cost_of_capital module.

Tests Damodaran industry betas and WACC calculations.
"""

import pytest

from investigator.domain.services.valuation.cost_of_capital import (
    CostOfCapitalResult,
    IndustryCostOfCapital,
    get_industry_cost_of_capital,
)


class TestIndustryCostOfCapital:
    """Tests for IndustryCostOfCapital class."""

    @pytest.fixture
    def coc(self):
        """Create default cost of capital calculator."""
        return IndustryCostOfCapital()

    def test_get_unlevered_beta_exact_match(self, coc):
        """Test exact industry match for beta."""
        beta, is_exact = coc.get_unlevered_beta("Software (System & Application)")
        assert beta == 1.08
        assert is_exact is True

    def test_get_unlevered_beta_case_insensitive(self, coc):
        """Test case insensitive matching."""
        beta1, _ = coc.get_unlevered_beta("software (system & application)")
        beta2, _ = coc.get_unlevered_beta("SOFTWARE (SYSTEM & APPLICATION)")
        assert beta1 == beta2

    def test_get_unlevered_beta_partial_match(self, coc):
        """Test partial industry name matching."""
        beta, is_exact = coc.get_unlevered_beta("Software")
        assert beta > 0
        assert is_exact is False  # Partial match

    def test_get_unlevered_beta_unknown_industry(self, coc):
        """Test default beta for unknown industry."""
        beta, is_exact = coc.get_unlevered_beta("Unknown Industry XYZ")
        assert beta == 1.00  # Default
        assert is_exact is False

    def test_calculate_levered_beta(self, coc):
        """Test levered beta calculation."""
        unlevered_beta = 1.0
        debt_to_equity = 0.5
        tax_rate = 0.21

        levered_beta = coc.calculate_levered_beta(unlevered_beta, debt_to_equity, tax_rate)

        # Levered beta = Unlevered * (1 + (1-t) * D/E)
        expected = 1.0 * (1 + (1 - 0.21) * 0.5)
        assert abs(levered_beta - expected) < 0.001

    def test_calculate_levered_beta_no_debt(self, coc):
        """Test levered beta equals unlevered when no debt."""
        unlevered_beta = 1.2
        levered_beta = coc.calculate_levered_beta(unlevered_beta, debt_to_equity=0, tax_rate=0.21)
        assert levered_beta == unlevered_beta

    def test_calculate_cost_of_equity(self, coc):
        """Test CAPM cost of equity calculation."""
        levered_beta = 1.2
        rf = 0.045
        erp = 0.055

        cost_of_equity = coc.calculate_cost_of_equity(levered_beta, rf, erp)

        # CoE = Rf + Beta * ERP
        expected = 0.045 + 1.2 * 0.055
        assert abs(cost_of_equity - expected) < 0.001

    def test_calculate_wacc_software_company(self, coc):
        """Test full WACC calculation for software company."""
        result = coc.calculate_wacc(industry="Software (System & Application)", debt_to_equity=0.25, tax_rate=0.21)

        assert isinstance(result, CostOfCapitalResult)
        assert result.wacc > 0
        assert result.wacc < 0.20  # Reasonable range
        assert result.unlevered_beta == 1.08
        assert result.levered_beta > result.unlevered_beta  # Due to leverage
        assert result.industry == "Software (System & Application)"

    def test_calculate_wacc_bank(self, coc):
        """Test WACC for bank (lower beta)."""
        result = coc.calculate_wacc(industry="Banks (Regional)", debt_to_equity=0.5, tax_rate=0.21)

        # Banks have lower beta
        assert result.unlevered_beta < 1.0
        assert result.wacc > 0

    def test_calculate_wacc_semiconductor(self, coc):
        """Test WACC for semiconductor (higher beta)."""
        result = coc.calculate_wacc(industry="Semiconductor", debt_to_equity=0.15, tax_rate=0.21)

        # Semiconductors have higher beta
        assert result.unlevered_beta > 1.0
        assert result.wacc > result.cost_of_equity * 0.5  # Equity dominated

    def test_calculate_wacc_no_debt(self, coc):
        """Test WACC with no debt equals cost of equity."""
        result = coc.calculate_wacc(industry="Software (System & Application)", debt_to_equity=0, tax_rate=0.21)

        # With no debt, WACC should equal cost of equity
        assert abs(result.wacc - result.cost_of_equity) < 0.001

    def test_calculate_wacc_custom_rates(self, coc):
        """Test WACC with custom risk-free rate and ERP."""
        result = coc.calculate_wacc(
            industry="Software (System & Application)",
            debt_to_equity=0.25,
            tax_rate=0.21,
            risk_free_rate=0.05,
            equity_risk_premium=0.06,
        )

        assert result.risk_free_rate == 0.05
        assert result.equity_risk_premium == 0.06

    def test_get_terminal_growth_rate_us(self, coc):
        """Test terminal growth rate for US company."""
        growth = coc.get_terminal_growth_rate("Software", "US")
        assert growth > 0
        assert growth < 0.05  # Should be reasonable

    def test_get_terminal_growth_rate_emerging(self, coc):
        """Test terminal growth rate for emerging market."""
        growth_cn = coc.get_terminal_growth_rate("Software", "CN")
        growth_us = coc.get_terminal_growth_rate("Software", "US")

        # Emerging market should have higher terminal growth
        assert growth_cn > growth_us

    def test_get_terminal_growth_rate_utility(self, coc):
        """Test terminal growth rate for utility (defensive)."""
        growth_utility = coc.get_terminal_growth_rate("Utility", "US")
        growth_software = coc.get_terminal_growth_rate("Software", "US")

        # Utilities should have lower growth
        assert growth_utility < growth_software

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        coc = IndustryCostOfCapital(risk_free_rate=0.05, equity_risk_premium=0.06, cost_of_debt=0.07)

        assert coc.risk_free_rate == 0.05
        assert coc.equity_risk_premium == 0.06
        assert coc.cost_of_debt == 0.07

    def test_singleton_instance(self):
        """Test singleton get_industry_cost_of_capital function."""
        coc1 = get_industry_cost_of_capital()
        coc2 = get_industry_cost_of_capital()
        assert coc1 is coc2

    def test_wacc_result_notes(self, coc):
        """Test that approximated betas include notes."""
        result = coc.calculate_wacc(industry="Unknown Industry XYZ", debt_to_equity=0.25, tax_rate=0.21)

        # Should have a note about approximation
        assert len(result.notes) > 0


class TestIndustryBetas:
    """Tests for industry beta coverage."""

    @pytest.fixture
    def coc(self):
        return IndustryCostOfCapital()

    def test_technology_industries(self, coc):
        """Test beta lookup for technology industries."""
        industries = [
            "Software (System & Application)",
            "Internet Software/Services",
            "Semiconductor",
            "Computer Services",
        ]

        for industry in industries:
            beta, is_exact = coc.get_unlevered_beta(industry)
            assert beta > 0
            assert is_exact is True

    def test_financial_industries(self, coc):
        """Test beta lookup for financial industries."""
        industries = [
            "Banks (Regional)",
            "Banks (Diversified)",
            "Insurance (Life)",
            "Brokerage & Investment Banking",
        ]

        for industry in industries:
            beta, is_exact = coc.get_unlevered_beta(industry)
            assert beta > 0
            assert is_exact is True

    def test_healthcare_industries(self, coc):
        """Test beta lookup for healthcare industries."""
        industries = [
            "Healthcare Services",
            "Drugs (Pharmaceutical)",
            "Biotechnology",
            "Medical Devices",
        ]

        for industry in industries:
            beta, is_exact = coc.get_unlevered_beta(industry)
            assert beta > 0
            assert is_exact is True

    def test_beta_ranges_reasonable(self, coc):
        """Test that all betas are in reasonable range."""
        for industry, beta in coc.INDUSTRY_BETAS.items():
            assert 0.3 < beta < 2.0, f"Beta for {industry} = {beta} out of range"
