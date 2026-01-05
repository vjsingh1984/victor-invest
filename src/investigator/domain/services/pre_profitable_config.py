"""
Pre-Profitable Company Configuration

Provides industry-specific growth assumptions and valuation parameters for
companies with negative earnings or EBITDA.

For pre-profitable companies, historical growth may be:
1. Too volatile (early-stage)
2. Too few quarters available (< 12)
3. Not indicative of future growth (investment phase)

This module provides sensible defaults based on industry benchmarks.

Usage:
    from investigator.domain.services.pre_profitable_config import (
        get_growth_assumptions,
        should_use_industry_defaults
    )

    assumptions = get_growth_assumptions(sector, industry)
    if should_use_industry_defaults(num_quarters, net_income, ebitda):
        revenue_growth = assumptions['default_revenue_growth']
"""

from typing import Dict, Optional

# Industry-specific growth assumptions for pre-profitable companies
PRE_PROFITABLE_GROWTH_ASSUMPTIONS = {
    # Technology - Software & Services
    "Technology/Computer Software: Prepackaged Software": {
        "default_revenue_growth": 0.25,  # 25% for cloud SaaS (Snowflake, Datadog, etc.)
        "min_quarters_for_historical": 8,  # Relax from 12 for pre-profitable
        "margin_expansion_assumption": True,  # Expect operating leverage
        "terminal_growth_premium": 0.01,  # 3% + 1% = 4% terminal growth
        "revenue_quality_weight": 0.7,  # Higher weight on revenue growth vs profitability
    },
    "Technology/Computer Software: Programming, Data Processing": {
        "default_revenue_growth": 0.30,  # 30% for infrastructure/dev tools
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.015,  # 3% + 1.5% = 4.5%
        "revenue_quality_weight": 0.75,
    },
    "Technology/EDP Services": {
        "default_revenue_growth": 0.20,  # 20% for enterprise services
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.005,
        "revenue_quality_weight": 0.65,
    },
    # Technology - Internet & E-commerce
    "Technology/Services-Computer Programming, Data Processing, Etc.": {
        "default_revenue_growth": 0.35,  # 35% for marketplace/platform
        "min_quarters_for_historical": 6,  # Very early stage
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.02,  # Network effects
        "revenue_quality_weight": 0.8,
    },
    # Healthcare - Biotech
    "Health Care/Biotechnology: Biological Products (No Diagnostic Substances)": {
        "default_revenue_growth": 0.15,  # 15% (lumpy, depends on approvals)
        "min_quarters_for_historical": 12,  # Need more data due to volatility
        "margin_expansion_assumption": False,  # R&D intensive
        "terminal_growth_premium": 0.005,
        "revenue_quality_weight": 0.5,  # Lower weight, focus on pipeline
    },
    # Financial Services - Fintech
    "Finance/Security Brokers, Dealers & Flotation Companies": {
        "default_revenue_growth": 0.40,  # 40% for fintech disruptors
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.015,
        "revenue_quality_weight": 0.7,
    },
    # Consumer - E-commerce & Direct-to-Consumer
    "Consumer Discretionary/Catalog & Mail-Order Houses": {
        "default_revenue_growth": 0.30,  # 30% for DTC brands
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.01,
        "revenue_quality_weight": 0.65,
    },
    # Transportation - Ride-sharing, Delivery
    "Consumer Discretionary/Transportation Services": {
        "default_revenue_growth": 0.25,  # 25% for logistics/delivery
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.01,
        "revenue_quality_weight": 0.7,
    },
}


# Fallback defaults by sector (when specific industry not found)
PRE_PROFITABLE_SECTOR_DEFAULTS = {
    "Technology": {
        "default_revenue_growth": 0.25,
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.01,
        "revenue_quality_weight": 0.7,
    },
    "Health Care": {
        "default_revenue_growth": 0.15,
        "min_quarters_for_historical": 10,
        "margin_expansion_assumption": False,
        "terminal_growth_premium": 0.005,
        "revenue_quality_weight": 0.5,
    },
    "Consumer Discretionary": {
        "default_revenue_growth": 0.20,
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.005,
        "revenue_quality_weight": 0.6,
    },
    "Financials": {
        "default_revenue_growth": 0.30,
        "min_quarters_for_historical": 8,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.01,
        "revenue_quality_weight": 0.65,
    },
    "Industrials": {
        "default_revenue_growth": 0.15,
        "min_quarters_for_historical": 10,
        "margin_expansion_assumption": True,
        "terminal_growth_premium": 0.005,
        "revenue_quality_weight": 0.55,
    },
}


# Conservative fallback if no sector/industry match
CONSERVATIVE_DEFAULTS = {
    "default_revenue_growth": 0.10,  # 10% conservative
    "min_quarters_for_historical": 12,
    "margin_expansion_assumption": False,
    "terminal_growth_premium": 0.0,
    "revenue_quality_weight": 0.5,
}


def get_growth_assumptions(sector: Optional[str] = None, industry: Optional[str] = None) -> Dict[str, any]:
    """
    Get growth assumptions for a pre-profitable company.

    Lookup priority:
    1. Specific industry match
    2. Sector-level defaults
    3. Conservative fallback

    Args:
        sector: Company sector (e.g., "Technology")
        industry: Specific industry (e.g., "Technology/Computer Software: Prepackaged Software")

    Returns:
        Dictionary with growth assumptions:
        - default_revenue_growth: Annual revenue growth rate (decimal)
        - min_quarters_for_historical: Minimum quarters to use historical vs default
        - margin_expansion_assumption: Whether to expect margin expansion
        - terminal_growth_premium: Premium to add to base terminal growth (3%)
        - revenue_quality_weight: Weight on revenue growth vs profitability (0-1)

    Examples:
        >>> assumptions = get_growth_assumptions(
        ...     sector="Technology",
        ...     industry="Technology/Computer Software: Prepackaged Software"
        ... )
        >>> assumptions['default_revenue_growth']
        0.25

        >>> assumptions = get_growth_assumptions(sector="Technology")
        >>> assumptions['default_revenue_growth']
        0.25
    """
    # Try industry-specific match first
    if industry and industry in PRE_PROFITABLE_GROWTH_ASSUMPTIONS:
        return PRE_PROFITABLE_GROWTH_ASSUMPTIONS[industry].copy()

    # Try sector-level match
    if sector:
        # Extract first part of sector (e.g., "Technology" from "Technology/Software")
        sector_key = sector.split("/")[0].strip() if "/" in sector else sector.strip()
        if sector_key in PRE_PROFITABLE_SECTOR_DEFAULTS:
            return PRE_PROFITABLE_SECTOR_DEFAULTS[sector_key].copy()

    # Conservative fallback
    return CONSERVATIVE_DEFAULTS.copy()


def should_use_industry_defaults(
    num_quarters: int,
    net_income: Optional[float] = None,
    ebitda: Optional[float] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
) -> bool:
    """
    Determine if industry defaults should be used instead of historical growth.

    Use industry defaults when:
    1. Company is pre-profitable (negative net income or EBITDA)
    2. Insufficient historical data (< min_quarters_for_historical)

    Args:
        num_quarters: Number of consecutive quarters available
        net_income: TTM net income (None if unavailable)
        ebitda: TTM EBITDA (None if unavailable)
        sector: Company sector
        industry: Specific industry

    Returns:
        True if industry defaults should be used

    Examples:
        >>> should_use_industry_defaults(
        ...     num_quarters=10,
        ...     net_income=-1000000,
        ...     ebitda=-500000,
        ...     sector="Technology"
        ... )
        True

        >>> should_use_industry_defaults(
        ...     num_quarters=15,
        ...     net_income=1000000,
        ...     ebitda=2000000,
        ...     sector="Technology"
        ... )
        False
    """
    # Get industry assumptions to check min_quarters requirement
    assumptions = get_growth_assumptions(sector, industry)
    min_quarters = assumptions["min_quarters_for_historical"]

    # Check if pre-profitable
    is_pre_profitable = False
    if net_income is not None and net_income < 0:
        is_pre_profitable = True
    if ebitda is not None and ebitda < 0:
        is_pre_profitable = True

    # Use defaults if pre-profitable AND insufficient quarters
    if is_pre_profitable and num_quarters < min_quarters:
        return True

    # ALSO use defaults if pre-profitable with SEVERE ongoing losses
    # Companies burning significant cash are in investment/growth phase
    # Their historical growth reflects unsustainable cash burn, not operational efficiency
    if is_pre_profitable and net_income is not None:
        # Severe losses threshold: -$100M (captures growth-stage SaaS, biotech, etc.)
        if net_income < -100_000_000:
            return True

    return False


def get_terminal_growth_rate(
    base_terminal_growth: float,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    is_high_quality: bool = False,
) -> float:
    """
    Calculate terminal growth rate with industry-specific premium.

    Args:
        base_terminal_growth: Base terminal growth (typically 10Y Treasury rate)
        sector: Company sector
        industry: Specific industry
        is_high_quality: Whether company has high quality metrics

    Returns:
        Adjusted terminal growth rate

    Examples:
        >>> get_terminal_growth_rate(0.03, sector="Technology", is_high_quality=True)
        0.04  # 3% + 1% premium
    """
    assumptions = get_growth_assumptions(sector, industry)
    premium = assumptions["terminal_growth_premium"]

    # Add premium for high-quality companies
    if is_high_quality:
        premium *= 1.5  # 50% bonus for high quality

    return base_terminal_growth + premium


def format_assumptions_log(
    assumptions: Dict[str, any], sector: Optional[str] = None, industry: Optional[str] = None
) -> str:
    """
    Format growth assumptions for logging.

    Args:
        assumptions: Growth assumptions dictionary
        sector: Company sector
        industry: Specific industry

    Returns:
        Formatted string for logging

    Examples:
        >>> assumptions = get_growth_assumptions(sector="Technology")
        >>> print(format_assumptions_log(assumptions, sector="Technology"))
        [PRE_PROFITABLE] Sector: Technology
          Revenue Growth: 25.0%
          Min Quarters: 8
          Margin Expansion: Yes
          Terminal Premium: +1.0%
    """
    lines = []

    if industry:
        lines.append(f"[PRE_PROFITABLE] Industry: {industry}")
    elif sector:
        lines.append(f"[PRE_PROFITABLE] Sector: {sector}")
    else:
        lines.append("[PRE_PROFITABLE] Using conservative defaults")

    lines.append(f"  Revenue Growth: {assumptions['default_revenue_growth']*100:.1f}%")
    lines.append(f"  Min Quarters: {assumptions['min_quarters_for_historical']}")
    lines.append(f"  Margin Expansion: {'Yes' if assumptions['margin_expansion_assumption'] else 'No'}")
    lines.append(f"  Terminal Premium: +{assumptions['terminal_growth_premium']*100:.1f}%")
    lines.append(f"  Revenue Quality Weight: {assumptions['revenue_quality_weight']*100:.0f}%")

    return "\n".join(lines)
