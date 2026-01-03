"""
Industry Cost of Capital - Damodaran-style cost of capital calculations.

Provides:
1. Industry-specific unlevered betas
2. Cost of equity calculation (CAPM)
3. WACC calculation
4. Levered beta calculation

Based on Aswath Damodaran's industry beta data.

Usage:
    from investigator.domain.services.valuation.cost_of_capital import IndustryCostOfCapital

    coc = IndustryCostOfCapital()
    wacc = coc.calculate_wacc(
        industry='Software (System & Application)',
        debt_to_equity=0.25,
        tax_rate=0.21,
        risk_free_rate=0.045
    )
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CostOfCapitalResult:
    """Result of cost of capital calculation."""

    cost_of_equity: float
    cost_of_debt: float
    wacc: float
    unlevered_beta: float
    levered_beta: float
    equity_risk_premium: float
    risk_free_rate: float
    tax_rate: float
    debt_to_equity: float
    industry: str
    notes: list


class IndustryCostOfCapital:
    """
    Industry-specific cost of capital calculator.

    Uses Damodaran's industry betas and market risk premiums to
    calculate cost of equity and WACC.

    Betas are unlevered (asset) betas from Damodaran's Jan 2024 data.
    """

    # Unlevered betas by industry (Damodaran Jan 2024)
    # Source: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html
    INDUSTRY_BETAS = {
        # Technology
        "Software (System & Application)": 1.08,
        "Software - Application": 1.08,
        "Software - Infrastructure": 1.05,
        "Internet Software/Services": 1.12,
        "Information Services": 0.93,
        "Computer Services": 1.02,
        "Computers/Peripherals": 1.15,
        "Semiconductor": 1.32,
        "Semiconductors": 1.32,
        "Semiconductor Equipment & Materials": 1.28,
        "Electronics": 0.98,
        "Telecom Equipment": 1.05,
        "Telecom. Services": 0.62,
        "Broadcasting": 0.88,
        "Entertainment": 0.95,
        # Financial Services
        "Banks (Regional)": 0.59,
        "Banks - Regional": 0.59,
        "Banks (Diversified)": 0.68,
        "Banks - Diversified": 0.68,
        "Brokerage & Investment Banking": 0.95,
        "Financial Services (Non-bank & Insurance)": 0.75,
        "Insurance (Life)": 0.82,
        "Insurance - Life": 0.82,
        "Insurance (Prop/Cas.)": 0.65,
        "Insurance - Property & Casualty": 0.65,
        "Investments & Asset Management": 0.88,
        "REIT": 0.65,
        # Healthcare
        "Healthcare Services": 0.78,
        "Hospitals/Healthcare Facilities": 0.75,
        "Healthcare Products": 0.95,
        "Drugs (Pharmaceutical)": 0.88,
        "Drugs (Biotechnology)": 1.15,
        "Biotechnology": 1.15,
        "Medical Devices": 0.92,
        # Consumer
        "Retail (General)": 0.85,
        "Retail (Online)": 1.05,
        "Internet Retail": 1.05,
        "Retail (Special Lines)": 0.88,
        "Specialty Retail": 0.88,
        "Restaurant/Dining": 0.78,
        "Household Products": 0.65,
        "Consumer Defensive": 0.65,
        "Food Processing": 0.55,
        "Food/Beverage/Tobacco": 0.55,
        "Apparel": 0.92,
        "Auto & Truck": 0.88,
        "Auto Manufacturers": 0.88,
        "Auto Parts": 0.95,
        # Industrial
        "Aerospace/Defense": 0.88,
        "Electrical Equipment": 0.95,
        "Industrial Services": 0.85,
        "Machinery": 0.92,
        "Packaging & Container": 0.68,
        "Transportation": 0.82,
        "Trucking": 0.75,
        "Air Transport": 0.95,
        "Railroads": 0.78,
        # Energy & Materials
        "Oil/Gas (Integrated)": 0.95,
        "Oil & Gas Integrated": 0.95,
        "Oil/Gas (Production & Exploration)": 1.05,
        "Oil & Gas E&P": 1.05,
        "Oil/Gas Distribution": 0.72,
        "Coal & Related Energy": 0.92,
        "Metals & Mining": 1.08,
        "Basic Materials": 0.95,
        "Steel": 1.12,
        "Chemicals (Basic)": 0.85,
        "Chemicals (Diversified)": 0.88,
        "Paper/Forest Products": 0.78,
        # Utilities & Real Estate
        "Utilities (General)": 0.52,
        "Utilities": 0.52,
        "Power": 0.58,
        "Water Utility": 0.45,
        "Real Estate (Development)": 0.72,
        "Real Estate (General/Diversified)": 0.68,
        "Real Estate": 0.68,
        "REIT - Retail": 0.65,
        "REIT - Residential": 0.62,
        "REIT - Office": 0.72,
        # Other
        "Engineering/Construction": 0.92,
        "Environmental & Waste Services": 0.75,
        "Education": 0.82,
        "Recreation": 0.88,
        "Publishing & Newspapers": 0.75,
    }

    # Default beta when industry not found
    DEFAULT_BETA = 1.00

    # Default market parameters
    DEFAULT_RISK_FREE_RATE = 0.045  # 4.5% (10Y Treasury approximation)
    DEFAULT_EQUITY_RISK_PREMIUM = 0.055  # 5.5% (mature market ERP)
    DEFAULT_COST_OF_DEBT = 0.06  # 6% (BBB-rated corporate debt)
    DEFAULT_TAX_RATE = 0.21  # 21% US corporate tax rate

    def __init__(
        self,
        risk_free_rate: Optional[float] = None,
        equity_risk_premium: Optional[float] = None,
        cost_of_debt: Optional[float] = None,
    ):
        """
        Initialize with optional market parameters.

        Args:
            risk_free_rate: Risk-free rate (default: 4.5%)
            equity_risk_premium: Equity risk premium (default: 5.5%)
            cost_of_debt: Pre-tax cost of debt (default: 6%)
        """
        self.risk_free_rate = risk_free_rate or self.DEFAULT_RISK_FREE_RATE
        self.equity_risk_premium = equity_risk_premium or self.DEFAULT_EQUITY_RISK_PREMIUM
        self.cost_of_debt = cost_of_debt or self.DEFAULT_COST_OF_DEBT

    def get_unlevered_beta(self, industry: str) -> Tuple[float, bool]:
        """
        Get unlevered beta for an industry.

        Args:
            industry: Industry name

        Returns:
            Tuple of (beta, is_exact_match)
        """
        # Try exact match first
        if industry in self.INDUSTRY_BETAS:
            return (self.INDUSTRY_BETAS[industry], True)

        # Try case-insensitive match
        industry_lower = industry.lower()
        for key, beta in self.INDUSTRY_BETAS.items():
            if key.lower() == industry_lower:
                return (beta, True)

        # Try partial match
        for key, beta in self.INDUSTRY_BETAS.items():
            if industry_lower in key.lower() or key.lower() in industry_lower:
                logger.debug(f"Partial beta match: '{industry}' -> '{key}'")
                return (beta, False)

        logger.warning(f"No beta found for industry '{industry}', using default {self.DEFAULT_BETA}")
        return (self.DEFAULT_BETA, False)

    def calculate_levered_beta(self, unlevered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
        """
        Calculate levered (equity) beta from unlevered beta.

        Formula: Levered Beta = Unlevered Beta × (1 + (1 - Tax Rate) × D/E)

        Args:
            unlevered_beta: Asset beta (unlevered)
            debt_to_equity: Debt-to-equity ratio
            tax_rate: Corporate tax rate

        Returns:
            Levered (equity) beta
        """
        return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)

    def calculate_cost_of_equity(
        self, levered_beta: float, risk_free_rate: Optional[float] = None, equity_risk_premium: Optional[float] = None
    ) -> float:
        """
        Calculate cost of equity using CAPM.

        Formula: Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium

        Args:
            levered_beta: Levered (equity) beta
            risk_free_rate: Override for risk-free rate
            equity_risk_premium: Override for ERP

        Returns:
            Cost of equity (as decimal, e.g., 0.12 for 12%)
        """
        rf = risk_free_rate or self.risk_free_rate
        erp = equity_risk_premium or self.equity_risk_premium

        return rf + levered_beta * erp

    def calculate_wacc(
        self,
        industry: str,
        debt_to_equity: float = 0.0,
        tax_rate: float = DEFAULT_TAX_RATE,
        risk_free_rate: Optional[float] = None,
        equity_risk_premium: Optional[float] = None,
        cost_of_debt: Optional[float] = None,
    ) -> CostOfCapitalResult:
        """
        Calculate WACC for a company.

        Formula: WACC = (E/V) × Re + (D/V) × Rd × (1 - Tc)

        Where:
        - E/V = equity weight
        - D/V = debt weight
        - Re = cost of equity
        - Rd = cost of debt
        - Tc = corporate tax rate

        Args:
            industry: Industry for beta lookup
            debt_to_equity: Debt-to-equity ratio (D/E)
            tax_rate: Corporate tax rate
            risk_free_rate: Override for risk-free rate
            equity_risk_premium: Override for equity risk premium
            cost_of_debt: Override for pre-tax cost of debt

        Returns:
            CostOfCapitalResult with all components
        """
        rf = risk_free_rate or self.risk_free_rate
        erp = equity_risk_premium or self.equity_risk_premium
        rd = cost_of_debt or self.cost_of_debt
        notes = []

        # Get unlevered beta
        unlevered_beta, exact_match = self.get_unlevered_beta(industry)
        if not exact_match:
            notes.append(f"Beta approximated for '{industry}'")

        # Calculate levered beta
        levered_beta = self.calculate_levered_beta(unlevered_beta, debt_to_equity, tax_rate)

        # Calculate cost of equity
        cost_of_equity = self.calculate_cost_of_equity(levered_beta, rf, erp)

        # Calculate capital weights
        # D/E = D/E ratio, so E = 1, D = D/E
        # V = E + D = 1 + D/E
        # E/V = 1 / (1 + D/E)
        # D/V = D/E / (1 + D/E)
        equity_weight = 1 / (1 + debt_to_equity)
        debt_weight = debt_to_equity / (1 + debt_to_equity)

        # Calculate WACC
        wacc = equity_weight * cost_of_equity + debt_weight * rd * (1 - tax_rate)

        logger.debug(
            f"WACC calculation for {industry}: "
            f"Unlevered β={unlevered_beta:.2f}, Levered β={levered_beta:.2f}, "
            f"Re={cost_of_equity:.1%}, WACC={wacc:.1%}"
        )

        return CostOfCapitalResult(
            cost_of_equity=cost_of_equity,
            cost_of_debt=rd,
            wacc=wacc,
            unlevered_beta=unlevered_beta,
            levered_beta=levered_beta,
            equity_risk_premium=erp,
            risk_free_rate=rf,
            tax_rate=tax_rate,
            debt_to_equity=debt_to_equity,
            industry=industry,
            notes=notes,
        )

    def get_terminal_growth_rate(self, industry: str, country: str = "US") -> float:
        """
        Get appropriate terminal growth rate for DCF.

        Terminal growth should not exceed long-term GDP growth.
        Higher for emerging markets, lower for mature markets.

        Args:
            industry: Industry name
            country: Country code (default: US)

        Returns:
            Terminal growth rate (as decimal)
        """
        # Base rates by market maturity
        base_rates = {
            "US": 0.025,  # 2.5% - mature market
            "EU": 0.020,  # 2.0% - mature market
            "UK": 0.020,  # 2.0% - mature market
            "JP": 0.010,  # 1.0% - mature, low growth
            "CN": 0.040,  # 4.0% - emerging, higher growth
            "IN": 0.045,  # 4.5% - emerging, higher growth
        }

        base = base_rates.get(country, 0.025)

        # Industry adjustments (defensive vs cyclical)
        industry_lower = industry.lower()

        if any(term in industry_lower for term in ["utility", "water", "power"]):
            return base * 0.8  # Slower growth
        elif any(term in industry_lower for term in ["software", "tech", "internet"]):
            return min(base * 1.2, 0.035)  # Slightly higher, capped
        elif any(term in industry_lower for term in ["bank", "insurance"]):
            return base  # GDP growth

        return base


# Singleton instance
_coc: Optional[IndustryCostOfCapital] = None


def get_industry_cost_of_capital() -> IndustryCostOfCapital:
    """Get the singleton IndustryCostOfCapital instance."""
    global _coc
    if _coc is None:
        _coc = IndustryCostOfCapital()
    return _coc
