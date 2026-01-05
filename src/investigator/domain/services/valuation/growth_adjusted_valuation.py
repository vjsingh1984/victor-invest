"""
Universal Growth-Adjusted Valuation Framework

This module provides growth-adjusted valuation that applies across ALL sectors,
not just semiconductors. Growth is a critical discriminator in valuations.

Key concepts:
1. Growth Profile Classification - Based on revenue/earnings growth
2. PEG-Based Valuation - P/E relative to growth rate
3. Rule of 40 - For SaaS/tech (growth + margin >= 40%)
4. Forward P/E Integration - When analyst estimates available
5. Sector-Specific Adjustments - Different sectors have different growth premiums

The framework adjusts P/E multiples based on:
- Growth rate (higher growth = higher justified P/E)
- Sector characteristics (SaaS vs Manufacturing vs Financials)
- Quality metrics (Rule of 40, margin stability)
- Cyclicality (semiconductors, autos, commodities)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================
# GROWTH PROFILE CLASSIFICATION
# ====================


class GrowthProfile(Enum):
    """Universal growth profile classification."""

    HYPER_GROWTH = "hyper_growth"  # >50% revenue growth
    HIGH_GROWTH = "high_growth"  # 25-50% revenue growth
    MODERATE_GROWTH = "moderate_growth"  # 10-25% revenue growth
    LOW_GROWTH = "low_growth"  # 0-10% revenue growth
    STABLE = "stable"  # ~0% but consistent
    DECLINING = "declining"  # Negative growth
    UNKNOWN = "unknown"


class QualityTier(Enum):
    """Quality tier based on Rule of 40 and other metrics."""

    EXCEPTIONAL = "exceptional"  # Rule of 40 > 60%
    HIGH_QUALITY = "high_quality"  # Rule of 40 40-60%
    AVERAGE = "average"  # Rule of 40 20-40%
    BELOW_AVERAGE = "below_average"  # Rule of 40 < 20%
    UNKNOWN = "unknown"


# ====================
# SECTOR GROWTH CHARACTERISTICS
# ====================

SECTOR_GROWTH_CHARACTERISTICS = {
    # Sector -> (base_pe_premium, max_pe, peg_ceiling, supports_rule_of_40)
    # Updated for current market environment (2024-2025)
    "Technology": {
        "base_pe_premium": 1.4,  # 40% premium for tech leadership
        "max_pe": 100.0,  # Allow high multiples for hyper-growth
        "peg_ceiling": 2.0,  # Can sustain high PEG
        "supports_rule_of_40": True,
        "growth_sensitivity": 1.3,  # High sensitivity to growth
        "quality_premium": 1.2,  # Premium for quality/moat
    },
    "Consumer Cyclical": {
        "base_pe_premium": 1.2,
        "max_pe": 60.0,
        "peg_ceiling": 1.8,
        "supports_rule_of_40": False,
        "growth_sensitivity": 1.2,
        "quality_premium": 1.15,
    },
    "Consumer Defensive": {
        "base_pe_premium": 1.1,
        "max_pe": 35.0,
        "peg_ceiling": 1.3,
        "supports_rule_of_40": False,
        "growth_sensitivity": 0.9,
        "quality_premium": 1.2,  # Premium for stable brands (KO, PG)
    },
    "Industrials": {
        "base_pe_premium": 1.1,
        "max_pe": 40.0,
        "peg_ceiling": 1.4,
        "supports_rule_of_40": False,
        "growth_sensitivity": 1.0,
        "quality_premium": 1.1,
    },
    "Healthcare": {
        "base_pe_premium": 1.25,
        "max_pe": 60.0,
        "peg_ceiling": 1.6,
        "supports_rule_of_40": True,  # Biotech/pharma pipelines
        "growth_sensitivity": 1.2,
        "quality_premium": 1.15,
    },
    "Financials": {
        "base_pe_premium": 1.0,  # Near market for quality banks
        "max_pe": 25.0,
        "peg_ceiling": 1.2,
        "supports_rule_of_40": False,
        "growth_sensitivity": 0.8,
        "quality_premium": 1.15,
    },
    "Real Estate": {
        "base_pe_premium": 1.0,
        "max_pe": 35.0,
        "peg_ceiling": 1.2,
        "supports_rule_of_40": False,
        "growth_sensitivity": 0.7,
        "quality_premium": 1.1,
    },
    "Energy": {
        "base_pe_premium": 0.9,
        "max_pe": 25.0,
        "peg_ceiling": 1.0,
        "supports_rule_of_40": False,
        "growth_sensitivity": 0.6,  # Cyclical, less growth-dependent
        "quality_premium": 1.05,
    },
    "Materials": {
        "base_pe_premium": 1.0,
        "max_pe": 30.0,
        "peg_ceiling": 1.2,
        "supports_rule_of_40": False,
        "growth_sensitivity": 0.8,
        "quality_premium": 1.05,
    },
    "Utilities": {
        "base_pe_premium": 0.9,
        "max_pe": 25.0,
        "peg_ceiling": 0.9,
        "supports_rule_of_40": False,
        "growth_sensitivity": 0.5,  # Low growth, dividend focus
        "quality_premium": 1.1,
    },
    "Communication Services": {
        "base_pe_premium": 1.2,
        "max_pe": 50.0,
        "peg_ceiling": 1.6,
        "supports_rule_of_40": True,
        "growth_sensitivity": 1.1,
        "quality_premium": 1.15,
    },
}

# Default for unknown sectors
DEFAULT_SECTOR_CHARACTERISTICS = {
    "base_pe_premium": 1.0,
    "max_pe": 40.0,
    "peg_ceiling": 1.3,
    "supports_rule_of_40": False,
    "growth_sensitivity": 1.0,
    "quality_premium": 1.1,
}


# ====================
# GROWTH PROFILE P/E MULTIPLES
# ====================

# Base P/E multiples by growth profile (before sector adjustments)
# These are calibrated to current market environment (S&P 500 P/E ~25x)
# Key insight: High growth rates are unsustainable, so we apply sustainability discounts
GROWTH_PROFILE_PE_MULTIPLES = {
    GrowthProfile.HYPER_GROWTH: {
        "base_pe": 35.0,  # Lower base - growth already reflected in PEG
        "peg_target": 0.8,  # Discount for unsustainability (was 1.0)
        "peg_premium": 0.2,  # Modest premium for exceptional (was 0.5)
        "peg_pe_cap": 50.0,  # Hard cap on PEG-implied P/E (was 100)
        "forward_pe_ratio": 0.70,  # Forward P/E ~70% of trailing
        "sustainability_discount": 0.85,  # 15% discount for growth deceleration
    },
    GrowthProfile.HIGH_GROWTH: {
        "base_pe": 30.0,
        "peg_target": 0.9,
        "peg_premium": 0.2,
        "peg_pe_cap": 45.0,
        "forward_pe_ratio": 0.75,
        "sustainability_discount": 0.90,
    },
    GrowthProfile.MODERATE_GROWTH: {
        "base_pe": 25.0,  # Near market P/E
        "peg_target": 1.0,
        "peg_premium": 0.15,
        "peg_pe_cap": 40.0,
        "forward_pe_ratio": 0.80,
        "sustainability_discount": 0.95,
    },
    GrowthProfile.LOW_GROWTH: {
        "base_pe": 20.0,  # Slight discount to market
        "peg_target": 1.2,  # Higher PEG target for low growth
        "peg_premium": 0.0,
        "peg_pe_cap": 30.0,
        "forward_pe_ratio": 0.88,
        "sustainability_discount": 1.0,  # No discount for stable growth
    },
    GrowthProfile.STABLE: {
        "base_pe": 18.0,
        "peg_target": None,  # PEG not meaningful for 0% growth
        "peg_premium": 0.0,
        "peg_pe_cap": 25.0,
        "forward_pe_ratio": 0.92,
        "sustainability_discount": 1.0,
    },
    GrowthProfile.DECLINING: {
        "base_pe": 12.0,
        "peg_target": None,
        "peg_premium": 0.0,
        "peg_pe_cap": 18.0,
        "forward_pe_ratio": 1.0,
        "sustainability_discount": 1.0,
    },
    GrowthProfile.UNKNOWN: {
        "base_pe": 18.0,
        "peg_target": 1.0,
        "peg_premium": 0.0,
        "peg_pe_cap": 30.0,
        "forward_pe_ratio": 0.88,
        "sustainability_discount": 0.95,
    },
}


# ====================
# DATA CLASSES
# ====================


@dataclass
class GrowthMetrics:
    """Container for growth-related metrics."""

    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    fcf_margin: Optional[float] = None
    rule_of_40: Optional[float] = None
    growth_profile: GrowthProfile = GrowthProfile.UNKNOWN
    quality_tier: QualityTier = QualityTier.UNKNOWN


@dataclass
class ConfidenceInterval:
    """Fair value confidence interval."""

    low: float  # Bear case / pessimistic
    mid: float  # Base case / expected
    high: float  # Bull case / optimistic
    confidence_pct: int  # Confidence level (e.g., 80 = 80% CI)

    def __str__(self) -> str:
        return f"${self.low:.2f} - ${self.mid:.2f} - ${self.high:.2f} ({self.confidence_pct}% CI)"


@dataclass
class GrowthAdjustedResult:
    """Result from growth-adjusted valuation."""

    # Fair value estimates (required fields first)
    base_fair_value: float  # Using sector baseline P/E
    peg_fair_value: float  # PEG-adjusted
    blended_fair_value: float  # Blended result
    confidence: str
    growth_profile: GrowthProfile
    quality_tier: QualityTier
    applied_pe: float
    weights: Dict[str, float]

    # Optional fields (with defaults) must come after required fields
    forward_fair_value: Optional[float] = None  # Forward P/E based
    rule_of_40_adjusted: Optional[float] = None  # Quality-adjusted
    fair_value_range: Optional[ConfidenceInterval] = None  # Confidence interval
    details: Dict[str, Any] = field(default_factory=dict)
    market_premium_pct: Optional[float] = None  # Market price vs fundamental
    market_premium_category: Optional[str] = None  # Category of premium


# ====================
# CONFIDENCE INTERVAL CALCULATION
# ====================

# Uncertainty ranges by growth profile (higher growth = more uncertainty)
GROWTH_PROFILE_UNCERTAINTY = {
    GrowthProfile.HYPER_GROWTH: 0.35,  # ±35% range (high uncertainty)
    GrowthProfile.HIGH_GROWTH: 0.25,  # ±25% range
    GrowthProfile.MODERATE_GROWTH: 0.18,  # ±18% range
    GrowthProfile.LOW_GROWTH: 0.12,  # ±12% range
    GrowthProfile.STABLE: 0.10,  # ±10% range (lowest uncertainty)
    GrowthProfile.DECLINING: 0.20,  # ±20% (turnaround uncertainty)
    GrowthProfile.UNKNOWN: 0.25,  # ±25% (default uncertainty)
}


def calculate_confidence_interval(
    blended_fair_value: float,
    component_values: List[float],
    growth_profile: GrowthProfile,
    has_forward_eps: bool,
    confidence_level: int = 80,
) -> ConfidenceInterval:
    """
    Calculate confidence interval for fair value estimate.

    Methodology:
    1. Base uncertainty from growth profile (higher growth = wider range)
    2. Adjust based on agreement between valuation methods
    3. Narrow interval if forward estimates available (more data = more confidence)

    Args:
        blended_fair_value: Base case fair value
        component_values: List of individual method fair values
        growth_profile: Company's growth classification
        has_forward_eps: Whether forward EPS is available
        confidence_level: Confidence level (default 80%)

    Returns:
        ConfidenceInterval with low/mid/high estimates
    """
    # 1. Base uncertainty from growth profile
    base_uncertainty = GROWTH_PROFILE_UNCERTAINTY.get(growth_profile, GROWTH_PROFILE_UNCERTAINTY[GrowthProfile.UNKNOWN])

    # 2. Calculate dispersion of component valuations
    valid_components = [v for v in component_values if v > 0]
    if len(valid_components) >= 2:
        component_range = max(valid_components) - min(valid_components)
        dispersion_pct = component_range / blended_fair_value if blended_fair_value > 0 else 0

        # If methods agree closely, reduce uncertainty
        if dispersion_pct < 0.15:
            base_uncertainty *= 0.80  # 20% reduction
        # If methods disagree significantly, increase uncertainty
        elif dispersion_pct > 0.40:
            base_uncertainty *= 1.20  # 20% increase

    # 3. Adjust for data availability
    if has_forward_eps:
        base_uncertainty *= 0.90  # 10% reduction (more data points)

    # 4. Calculate interval bounds
    # Use asymmetric bounds (larger downside than upside for conservatism)
    upside_factor = 1 + (base_uncertainty * 0.85)  # Smaller upside
    downside_factor = 1 - (base_uncertainty * 1.15)  # Larger downside

    low = blended_fair_value * max(0.1, downside_factor)  # Floor at 10% of mid
    high = blended_fair_value * upside_factor
    mid = blended_fair_value

    return ConfidenceInterval(
        low=low,
        mid=mid,
        high=high,
        confidence_pct=confidence_level,
    )


# ====================
# CLASSIFICATION FUNCTIONS
# ====================


def classify_growth_profile(
    revenue_growth: Optional[float] = None,
    earnings_growth: Optional[float] = None,
) -> GrowthProfile:
    """
    Classify company's growth profile.

    Args:
        revenue_growth: YoY revenue growth (e.g., 0.50 = 50%)
        earnings_growth: YoY earnings growth

    Returns:
        GrowthProfile enum
    """
    # Prefer revenue growth (more stable signal)
    growth = revenue_growth if revenue_growth is not None else earnings_growth

    if growth is None:
        return GrowthProfile.UNKNOWN

    if growth > 0.50:
        return GrowthProfile.HYPER_GROWTH
    elif growth > 0.25:
        return GrowthProfile.HIGH_GROWTH
    elif growth > 0.10:
        return GrowthProfile.MODERATE_GROWTH
    elif growth > 0.02:
        return GrowthProfile.LOW_GROWTH
    elif growth > -0.02:
        return GrowthProfile.STABLE
    else:
        return GrowthProfile.DECLINING


def calculate_rule_of_40(
    revenue_growth: Optional[float],
    fcf_margin: Optional[float] = None,
    operating_margin: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate Rule of 40 score.

    Rule of 40 = Revenue Growth (%) + Profit Margin (%)
    A score >= 40 indicates a healthy SaaS/growth company.

    Args:
        revenue_growth: YoY revenue growth rate
        fcf_margin: Free cash flow margin (preferred)
        operating_margin: Operating margin (fallback)

    Returns:
        Rule of 40 score as percentage, or None if not calculable
    """
    if revenue_growth is None:
        return None

    margin = fcf_margin if fcf_margin is not None else operating_margin
    if margin is None:
        return None

    # Convert to percentages and add
    rule_of_40 = (revenue_growth * 100) + (margin * 100)

    logger.debug(f"Rule of 40: {revenue_growth*100:.1f}% growth + {margin*100:.1f}% margin = {rule_of_40:.1f}")

    return rule_of_40


def classify_quality_tier(rule_of_40: Optional[float]) -> QualityTier:
    """
    Classify quality tier based on Rule of 40.

    Args:
        rule_of_40: Rule of 40 score

    Returns:
        QualityTier enum
    """
    if rule_of_40 is None:
        return QualityTier.UNKNOWN

    if rule_of_40 >= 60:
        return QualityTier.EXCEPTIONAL
    elif rule_of_40 >= 40:
        return QualityTier.HIGH_QUALITY
    elif rule_of_40 >= 20:
        return QualityTier.AVERAGE
    else:
        return QualityTier.BELOW_AVERAGE


# ====================
# VALUATION FUNCTIONS
# ====================


def calculate_peg_fair_value(
    eps: float,
    growth_rate: float,
    growth_profile: GrowthProfile,
    sector: Optional[str] = None,
) -> Tuple[float, float, str]:
    """
    Calculate PEG-adjusted fair value with sustainability adjustments.

    PEG = P/E / Growth Rate
    Fair P/E = Growth Rate × PEG Target × Sustainability Discount

    Key insight: High growth rates are unsustainable, so we apply:
    1. Lower PEG targets for hyper-growth (0.8 vs 1.0)
    2. Profile-specific P/E caps (50x for hyper-growth vs 100x)
    3. Sustainability discounts (15% for hyper-growth)

    Args:
        eps: Earnings per share
        growth_rate: Growth rate (e.g., 0.50 = 50%)
        growth_profile: Classified growth profile
        sector: Company sector for adjustments

    Returns:
        Tuple of (fair_value, applied_pe, explanation)
    """
    if eps <= 0:
        return 0.0, 0.0, "PEG not applicable: non-positive EPS"

    if growth_rate <= 0:
        return 0.0, 0.0, "PEG not applicable: non-positive growth"

    # Get growth profile parameters
    params = GROWTH_PROFILE_PE_MULTIPLES.get(growth_profile, GROWTH_PROFILE_PE_MULTIPLES[GrowthProfile.UNKNOWN])
    peg_target = params.get("peg_target")
    peg_premium = params.get("peg_premium", 0)
    peg_pe_cap = params.get("peg_pe_cap", 50.0)  # Profile-specific cap
    sustainability_discount = params.get("sustainability_discount", 1.0)

    if peg_target is None:
        return 0.0, 0.0, "PEG not applicable for this growth profile"

    # Get sector characteristics
    sector_chars = SECTOR_GROWTH_CHARACTERISTICS.get(sector, DEFAULT_SECTOR_CHARACTERISTICS)
    peg_ceiling = sector_chars["peg_ceiling"]
    growth_sensitivity = sector_chars["growth_sensitivity"]

    # Calculate PEG-implied P/E with constraints
    # Fair P/E = Growth Rate (%) × (PEG Target + Premium) × Sensitivity
    growth_pct = growth_rate * 100
    effective_peg = min(peg_target + peg_premium, peg_ceiling)
    raw_pe = growth_pct * effective_peg * growth_sensitivity

    # Apply profile-specific cap (more conservative than sector max)
    fair_pe = min(raw_pe, peg_pe_cap)

    # Apply sustainability discount for high-growth companies
    fair_pe *= sustainability_discount

    # Calculate fair value
    fair_value = eps * fair_pe

    explanation = (
        f"PEG: {growth_pct:.0f}% × {effective_peg:.2f} PEG × {growth_sensitivity:.1f} sens "
        f"= {raw_pe:.0f}x -> capped at {peg_pe_cap:.0f}x × {sustainability_discount:.0%} sustainability "
        f"= {fair_pe:.1f}x P/E"
    )

    logger.info(f"PEG valuation: {explanation} -> ${fair_value:.2f}")

    return fair_value, fair_pe, explanation


def calculate_forward_pe_fair_value(
    forward_eps: float,
    growth_profile: GrowthProfile,
    sector: Optional[str] = None,
    trailing_eps: Optional[float] = None,
    years_forward: int = 1,
) -> Tuple[float, float, str]:
    """
    Calculate fair value using forward P/E with trajectory adjustment.

    When forward EPS >> trailing EPS, this signals strong earnings momentum.
    We can project continued growth and discount back to get a higher fair value
    that accounts for the trajectory.

    Args:
        forward_eps: Analyst consensus forward EPS
        growth_profile: Classified growth profile
        sector: Company sector
        trailing_eps: Trailing EPS (for trajectory calculation)
        years_forward: Years to project forward (default 1)

    Returns:
        Tuple of (fair_value, forward_pe_target, explanation)
    """
    if forward_eps <= 0:
        return 0.0, 0.0, "Forward P/E not applicable: no positive forward EPS"

    # Get base parameters
    params = GROWTH_PROFILE_PE_MULTIPLES.get(growth_profile, GROWTH_PROFILE_PE_MULTIPLES[GrowthProfile.UNKNOWN])
    base_pe = params["base_pe"]
    forward_ratio = params["forward_pe_ratio"]

    # Get sector adjustment
    sector_chars = SECTOR_GROWTH_CHARACTERISTICS.get(sector, DEFAULT_SECTOR_CHARACTERISTICS)
    sector_premium = sector_chars["base_pe_premium"]
    max_pe = sector_chars["max_pe"]

    # Calculate forward P/E target
    forward_pe = base_pe * forward_ratio * sector_premium
    forward_pe = min(forward_pe, max_pe * forward_ratio)

    # Calculate base fair value
    fair_value = forward_eps * forward_pe
    explanation = f"Forward P/E: ${forward_eps:.2f} × {forward_pe:.1f}x"

    # Trajectory adjustment: if forward EPS >> trailing, project continued growth
    if trailing_eps and trailing_eps > 0:
        eps_growth_rate = (forward_eps / trailing_eps) - 1

        if eps_growth_rate > 0.30:  # >30% EPS growth
            # Multi-year projection with decay
            # Hyper-growth: Project 3 years forward with 30% annual decay
            # This captures the market's pricing of future growth
            discount_rate = 0.12 if growth_profile == GrowthProfile.HYPER_GROWTH else 0.10
            decay_factor = 0.70  # Growth slows 30% per year

            # Project EPS forward 3 years
            y1_growth = eps_growth_rate
            y2_growth = y1_growth * decay_factor
            y3_growth = y2_growth * decay_factor

            y1_eps = forward_eps  # Already have this
            y2_eps = y1_eps * (1 + y2_growth)
            y3_eps = y2_eps * (1 + y3_growth)

            # Terminal P/E at year 3 (as growth normalizes)
            # Higher terminal for hyper-growth reflecting market premium
            terminal_pe = max_pe * 0.60 if growth_profile == GrowthProfile.HYPER_GROWTH else max_pe * 0.50

            # Year 3 price, discounted to present
            y3_price = y3_eps * terminal_pe
            trajectory_fair_value = y3_price / ((1 + discount_rate) ** 3)

            # For hyper-growth with >40% EPS growth, use higher blend weight
            if eps_growth_rate > 0.40 and growth_profile == GrowthProfile.HYPER_GROWTH:
                # Strong momentum: 40% immediate, 60% trajectory
                blend_immediate = 0.40
                blend_trajectory = 0.60
            else:
                # Moderate momentum: 60% immediate, 40% trajectory
                blend_immediate = 0.60
                blend_trajectory = 0.40

            if trajectory_fair_value > fair_value:
                blended_fv = fair_value * blend_immediate + trajectory_fair_value * blend_trajectory
                trajectory_premium = (blended_fv / fair_value - 1) * 100
                explanation += (
                    f" + trajectory ({eps_growth_rate*100:.0f}% EPS growth "
                    f"→ Y3 EPS ${y3_eps:.2f} @ {terminal_pe:.0f}x discounted "
                    f"= +{trajectory_premium:.0f}%)"
                )
                fair_value = blended_fv

    logger.info(f"Forward P/E valuation: {explanation} -> ${fair_value:.2f}")

    return fair_value, forward_pe, explanation


def calculate_rule_of_40_adjustment(
    base_value: float,
    rule_of_40: Optional[float],
    quality_tier: QualityTier,
    growth_profile: GrowthProfile = GrowthProfile.UNKNOWN,
) -> Tuple[float, str]:
    """
    Adjust fair value based on Rule of 40 quality.

    Key insight: Don't add premium on top of already high-growth valuations.
    The Rule of 40 adjustment should only apply meaningfully to:
    - Below-average companies (discount)
    - Moderate/low growth companies with high quality (premium)

    For hyper-growth companies, the growth is already reflected in the
    PEG-based valuation, so we reduce/eliminate the R40 premium.

    Args:
        base_value: Base fair value
        rule_of_40: Rule of 40 score
        quality_tier: Classified quality tier
        growth_profile: Growth profile to avoid double-counting

    Returns:
        Tuple of (adjusted_value, explanation)
    """
    if rule_of_40 is None or quality_tier == QualityTier.UNKNOWN:
        return base_value, "No Rule of 40 adjustment (insufficient data)"

    # Base quality adjustment multipliers
    base_adjustments = {
        QualityTier.EXCEPTIONAL: 1.10,  # +10% for exceptional (reduced from 15%)
        QualityTier.HIGH_QUALITY: 1.05,  # +5% for high quality
        QualityTier.AVERAGE: 1.00,  # No adjustment
        QualityTier.BELOW_AVERAGE: 0.90,  # -10% for below average
    }

    multiplier = base_adjustments.get(quality_tier, 1.0)

    # Reduce premium for high-growth companies (growth already priced in)
    # Only apply full premium to moderate/low growth companies
    if growth_profile == GrowthProfile.HYPER_GROWTH:
        # Minimal adjustment - growth already reflected in PEG
        if multiplier > 1.0:
            multiplier = 1.0 + (multiplier - 1.0) * 0.25  # 25% of premium
    elif growth_profile == GrowthProfile.HIGH_GROWTH:
        if multiplier > 1.0:
            multiplier = 1.0 + (multiplier - 1.0) * 0.50  # 50% of premium

    adjusted_value = base_value * multiplier

    explanation = (
        f"Rule of 40 = {rule_of_40:.0f}: {quality_tier.value} quality, "
        f"{(multiplier-1)*100:+.1f}% adjustment (scaled for {growth_profile.value})"
    )

    logger.info(f"Rule of 40 adjustment: {explanation}")

    return adjusted_value, explanation


# ====================
# MAIN VALUATION FUNCTION
# ====================


def calculate_growth_adjusted_valuation(
    symbol: str,
    financials: Dict[str, Any],
    market_data: Dict[str, Any],
    sector: Optional[str] = None,
) -> GrowthAdjustedResult:
    """
    Calculate comprehensive growth-adjusted valuation.

    This is the main entry point for universal growth-adjusted valuation.
    It combines:
    1. Base P/E valuation (sector baseline)
    2. PEG-adjusted valuation (growth-based)
    3. Forward P/E valuation (if available)
    4. Rule of 40 quality adjustment (for SaaS/tech)

    Args:
        symbol: Stock ticker
        financials: Dict with eps, revenue_growth, fcf_margin, etc.
        market_data: Dict with current_price, forward_eps, shares, etc.
        sector: Company sector

    Returns:
        GrowthAdjustedResult with all estimates and blended value
    """
    # Extract inputs
    eps = financials.get("eps", 0)
    revenue_growth = financials.get("revenue_growth")
    earnings_growth = financials.get("earnings_growth")
    fcf_margin = financials.get("fcf_margin")
    operating_margin = financials.get("operating_margin")
    ebitda = financials.get("ebitda", 0)

    forward_eps = market_data.get("forward_eps", 0)
    shares = market_data.get("shares_outstanding", 0)
    current_price = market_data.get("current_price", 0)

    details = {}

    # 0. Detect EPS anomaly (trailing EPS significantly lower than forward)
    eps_anomaly = False
    eps_anomaly_boost = 0.0
    if eps > 0 and forward_eps > 0:
        eps_anomaly, eps_anomaly_boost, anomaly_explanation = _detect_eps_anomaly(
            trailing_eps=eps,
            forward_eps=forward_eps,
        )
        details["eps_anomaly"] = eps_anomaly
        details["eps_anomaly_boost"] = eps_anomaly_boost
        details["eps_anomaly_explanation"] = anomaly_explanation
        if eps_anomaly:
            logger.info(f"{symbol} - {anomaly_explanation}")

    # 1. Classify growth profile
    growth_profile = classify_growth_profile(revenue_growth, earnings_growth)
    details["growth_profile"] = growth_profile.value

    # 2. Calculate Rule of 40 and quality tier
    rule_of_40 = calculate_rule_of_40(revenue_growth, fcf_margin, operating_margin)
    quality_tier = classify_quality_tier(rule_of_40)
    details["rule_of_40"] = rule_of_40
    details["quality_tier"] = quality_tier.value

    # 3. Get sector characteristics
    sector_chars = SECTOR_GROWTH_CHARACTERISTICS.get(sector, DEFAULT_SECTOR_CHARACTERISTICS)
    supports_rule_of_40 = sector_chars.get("supports_rule_of_40", False)

    # 4. Calculate base fair value (sector baseline P/E)
    base_params = GROWTH_PROFILE_PE_MULTIPLES.get(growth_profile, GROWTH_PROFILE_PE_MULTIPLES[GrowthProfile.UNKNOWN])
    base_pe = base_params["base_pe"] * sector_chars["base_pe_premium"]
    base_fair_value = eps * base_pe if eps > 0 else 0
    details["base_pe"] = base_pe
    details["base_fair_value"] = base_fair_value

    # 5. Calculate PEG fair value
    peg_fair_value = 0
    peg_pe = 0
    if revenue_growth and revenue_growth > 0:
        peg_fair_value, peg_pe, peg_explanation = calculate_peg_fair_value(
            eps=eps,
            growth_rate=revenue_growth,
            growth_profile=growth_profile,
            sector=sector,
        )
        details["peg_explanation"] = peg_explanation
    details["peg_fair_value"] = peg_fair_value
    details["peg_pe"] = peg_pe

    # 6. Calculate forward P/E fair value (with trajectory if EPS anomaly detected)
    forward_fair_value = None
    if forward_eps and forward_eps > 0:
        forward_fair_value, forward_pe, forward_explanation = calculate_forward_pe_fair_value(
            forward_eps=forward_eps,
            growth_profile=growth_profile,
            sector=sector,
            trailing_eps=eps if eps_anomaly else None,  # Pass trailing for trajectory
        )
        details["forward_explanation"] = forward_explanation
        details["forward_fair_value"] = forward_fair_value
        details["forward_pe"] = forward_pe

    # 7. Calculate EV/EBITDA fair value (for diversification)
    ev_ebitda_fair_value = 0
    if ebitda > 0 and shares > 0:
        # EV/EBITDA multiples by growth profile
        ev_multiples = {
            GrowthProfile.HYPER_GROWTH: 35.0,
            GrowthProfile.HIGH_GROWTH: 25.0,
            GrowthProfile.MODERATE_GROWTH: 18.0,
            GrowthProfile.LOW_GROWTH: 12.0,
            GrowthProfile.STABLE: 10.0,
            GrowthProfile.DECLINING: 8.0,
            GrowthProfile.UNKNOWN: 15.0,
        }
        ev_multiple = ev_multiples.get(growth_profile, 15.0)
        enterprise_value = ebitda * ev_multiple
        ev_ebitda_fair_value = enterprise_value / shares
        details["ev_ebitda_multiple"] = ev_multiple
        details["ev_ebitda_fair_value"] = ev_ebitda_fair_value

    # 8. Determine blending weights (with EPS anomaly boost if applicable)
    weights = _get_blending_weights(
        growth_profile=growth_profile,
        has_forward_eps=forward_eps > 0,
        supports_rule_of_40=supports_rule_of_40,
        sector=sector,
        eps_anomaly_boost=eps_anomaly_boost,
    )
    details["weights"] = weights

    # 9. Calculate blended fair value
    blended_fv = 0
    total_weight = 0

    if base_fair_value > 0 and weights.get("base", 0) > 0:
        blended_fv += base_fair_value * weights["base"]
        total_weight += weights["base"]

    if peg_fair_value > 0 and weights.get("peg", 0) > 0:
        blended_fv += peg_fair_value * weights["peg"]
        total_weight += weights["peg"]

    if forward_fair_value and forward_fair_value > 0 and weights.get("forward", 0) > 0:
        blended_fv += forward_fair_value * weights["forward"]
        total_weight += weights["forward"]

    if ev_ebitda_fair_value > 0 and weights.get("ev_ebitda", 0) > 0:
        blended_fv += ev_ebitda_fair_value * weights["ev_ebitda"]
        total_weight += weights["ev_ebitda"]

    if total_weight > 0:
        blended_fv /= total_weight

    # 10. Apply Rule of 40 adjustment if applicable
    rule_of_40_adjusted = None
    if supports_rule_of_40 and rule_of_40 is not None:
        rule_of_40_adjusted, r40_explanation = calculate_rule_of_40_adjustment(
            base_value=blended_fv,
            rule_of_40=rule_of_40,
            quality_tier=quality_tier,
            growth_profile=growth_profile,  # Pass to avoid double-counting growth
        )
        details["rule_of_40_explanation"] = r40_explanation
        # Use adjusted value as final blended
        blended_fv = rule_of_40_adjusted

    # 11. Determine confidence
    confidence = _determine_confidence(
        growth_profile=growth_profile,
        has_forward_eps=forward_eps > 0,
        has_rule_of_40=rule_of_40 is not None,
    )

    # Get primary P/E used
    applied_pe = peg_pe if peg_pe > 0 else base_pe

    # Calculate market premium (how much market is paying above fundamental fair value)
    market_premium_pct = None
    market_premium_category = None
    if current_price > 0 and blended_fv > 0:
        market_premium_pct = ((current_price / blended_fv) - 1) * 100
        details["market_premium_pct"] = market_premium_pct

        # Categorize the premium
        if market_premium_pct < -30:
            market_premium_category = "deeply_undervalued"
        elif market_premium_pct < -10:
            market_premium_category = "undervalued"
        elif market_premium_pct < 20:
            market_premium_category = "fair_value"
        elif market_premium_pct < 50:
            market_premium_category = "slight_premium"
        elif market_premium_pct < 100:
            market_premium_category = "growth_premium"
        elif market_premium_pct < 200:
            market_premium_category = "high_growth_premium"
        elif market_premium_pct < 400:
            market_premium_category = "optionality_premium"
        else:
            market_premium_category = "extreme_speculation"

        details["market_premium_category"] = market_premium_category

        # Add implied market P/E for context
        if eps > 0:
            details["market_trailing_pe"] = current_price / eps
        if forward_eps > 0:
            details["market_forward_pe"] = current_price / forward_eps

    # Calculate confidence interval (addresses "point estimates only" limitation)
    component_values = [
        base_fair_value,
        peg_fair_value,
        forward_fair_value or 0,
        ev_ebitda_fair_value,
    ]
    fair_value_range = calculate_confidence_interval(
        blended_fair_value=blended_fv,
        component_values=component_values,
        growth_profile=growth_profile,
        has_forward_eps=forward_eps > 0,
    )
    details["fair_value_range"] = {
        "low": fair_value_range.low,
        "mid": fair_value_range.mid,
        "high": fair_value_range.high,
        "confidence_pct": fair_value_range.confidence_pct,
    }

    logger.info(
        f"{symbol} - Growth-adjusted valuation: "
        f"profile={growth_profile.value}, quality={quality_tier.value}, "
        f"base=${base_fair_value:.2f}, peg=${peg_fair_value:.2f}, "
        f"blended=${blended_fv:.2f} (range: ${fair_value_range.low:.2f}-${fair_value_range.high:.2f}), "
        f"confidence={confidence}"
        + (
            f", market_premium={market_premium_pct:+.0f}% ({market_premium_category})"
            if market_premium_pct is not None
            else ""
        )
    )

    return GrowthAdjustedResult(
        base_fair_value=base_fair_value,
        peg_fair_value=peg_fair_value,
        forward_fair_value=forward_fair_value,
        rule_of_40_adjusted=rule_of_40_adjusted,
        blended_fair_value=blended_fv,
        confidence=confidence,
        fair_value_range=fair_value_range,
        growth_profile=growth_profile,
        quality_tier=quality_tier,
        applied_pe=applied_pe,
        weights=weights,
        details=details,
        market_premium_pct=market_premium_pct,
        market_premium_category=market_premium_category,
    )


def _detect_eps_anomaly(
    trailing_eps: float,
    forward_eps: float,
) -> Tuple[bool, float, str]:
    """
    Detect if trailing EPS is abnormally low relative to forward EPS.

    When forward EPS is significantly higher than trailing (ratio > 1.3),
    it indicates trailing earnings are depressed due to:
    - One-time charges
    - Investment phase
    - Cyclical trough
    - Business transition

    In these cases, we should weight forward P/E more heavily.

    Args:
        trailing_eps: Trailing 12-month EPS
        forward_eps: Analyst consensus forward EPS

    Returns:
        Tuple of (is_anomaly, forward_weight_boost, explanation)
    """
    if trailing_eps <= 0 or forward_eps <= 0:
        return False, 0.0, "Cannot assess EPS anomaly with non-positive EPS"

    ratio = forward_eps / trailing_eps

    if ratio > 2.0:
        # Severely depressed trailing EPS - heavily weight forward
        return True, 0.40, f"Forward EPS {ratio:.1f}x higher - severe trailing depression"
    elif ratio > 1.5:
        # Significantly depressed - moderate boost to forward
        return True, 0.25, f"Forward EPS {ratio:.1f}x higher - significant trailing depression"
    elif ratio > 1.3:
        # Mildly depressed - slight boost to forward
        return True, 0.15, f"Forward EPS {ratio:.1f}x higher - mild trailing depression"
    else:
        # Normal relationship
        return False, 0.0, f"Normal EPS trajectory (ratio {ratio:.2f})"


def _get_blending_weights(
    growth_profile: GrowthProfile,
    has_forward_eps: bool,
    supports_rule_of_40: bool,
    sector: Optional[str] = None,
    eps_anomaly_boost: float = 0.0,
) -> Dict[str, float]:
    """
    Determine blending weights based on growth profile and data availability.

    For high-growth companies: Weight more towards PEG and Forward P/E
    For low-growth companies: Weight more towards base P/E and EV/EBITDA
    For SaaS/Tech with Rule of 40: Additional quality adjustment
    For EPS anomalies: Boost forward P/E weight

    Args:
        growth_profile: Company's growth profile
        has_forward_eps: Whether forward EPS available
        supports_rule_of_40: Whether sector supports R40
        sector: Company sector
        eps_anomaly_boost: Additional weight to shift to forward P/E (0-0.4)

    Returns:
        Dict of weights for each method
    """
    # Base weights by growth profile
    if has_forward_eps:
        weights_matrix = {
            GrowthProfile.HYPER_GROWTH: {"base": 0.10, "peg": 0.30, "forward": 0.40, "ev_ebitda": 0.20},
            GrowthProfile.HIGH_GROWTH: {"base": 0.15, "peg": 0.30, "forward": 0.35, "ev_ebitda": 0.20},
            GrowthProfile.MODERATE_GROWTH: {"base": 0.25, "peg": 0.25, "forward": 0.30, "ev_ebitda": 0.20},
            GrowthProfile.LOW_GROWTH: {"base": 0.30, "peg": 0.15, "forward": 0.25, "ev_ebitda": 0.30},
            GrowthProfile.STABLE: {"base": 0.35, "peg": 0.0, "forward": 0.30, "ev_ebitda": 0.35},
            GrowthProfile.DECLINING: {"base": 0.30, "peg": 0.0, "forward": 0.30, "ev_ebitda": 0.40},
            GrowthProfile.UNKNOWN: {"base": 0.30, "peg": 0.20, "forward": 0.25, "ev_ebitda": 0.25},
        }
    else:
        # Without forward EPS, redistribute weight
        weights_matrix = {
            GrowthProfile.HYPER_GROWTH: {"base": 0.15, "peg": 0.55, "forward": 0.0, "ev_ebitda": 0.30},
            GrowthProfile.HIGH_GROWTH: {"base": 0.20, "peg": 0.50, "forward": 0.0, "ev_ebitda": 0.30},
            GrowthProfile.MODERATE_GROWTH: {"base": 0.30, "peg": 0.35, "forward": 0.0, "ev_ebitda": 0.35},
            GrowthProfile.LOW_GROWTH: {"base": 0.40, "peg": 0.15, "forward": 0.0, "ev_ebitda": 0.45},
            GrowthProfile.STABLE: {"base": 0.45, "peg": 0.0, "forward": 0.0, "ev_ebitda": 0.55},
            GrowthProfile.DECLINING: {"base": 0.40, "peg": 0.0, "forward": 0.0, "ev_ebitda": 0.60},
            GrowthProfile.UNKNOWN: {"base": 0.35, "peg": 0.25, "forward": 0.0, "ev_ebitda": 0.40},
        }

    weights = weights_matrix.get(growth_profile, weights_matrix[GrowthProfile.UNKNOWN]).copy()

    # Apply EPS anomaly boost - shift weight from base/peg to forward P/E
    if eps_anomaly_boost > 0 and has_forward_eps:
        # Take weight proportionally from base and peg, give to forward
        base_reduction = eps_anomaly_boost * 0.5
        peg_reduction = eps_anomaly_boost * 0.5

        # Don't reduce below zero
        actual_base_reduction = min(base_reduction, weights.get("base", 0) * 0.5)
        actual_peg_reduction = min(peg_reduction, weights.get("peg", 0) * 0.5)

        weights["base"] = max(0, weights.get("base", 0) - actual_base_reduction)
        weights["peg"] = max(0, weights.get("peg", 0) - actual_peg_reduction)
        weights["forward"] = weights.get("forward", 0) + actual_base_reduction + actual_peg_reduction

        logger.info(
            f"EPS anomaly detected: boosting forward P/E weight by "
            f"{(actual_base_reduction + actual_peg_reduction)*100:.0f}% "
            f"(new forward weight: {weights['forward']*100:.0f}%)"
        )

    return weights


def _determine_confidence(
    growth_profile: GrowthProfile,
    has_forward_eps: bool,
    has_rule_of_40: bool,
) -> str:
    """Determine confidence level in valuation."""
    score = 0

    # Growth profile clarity
    if growth_profile in [GrowthProfile.MODERATE_GROWTH, GrowthProfile.LOW_GROWTH, GrowthProfile.STABLE]:
        score += 2  # Easier to value
    elif growth_profile in [GrowthProfile.HIGH_GROWTH]:
        score += 1
    elif growth_profile == GrowthProfile.HYPER_GROWTH:
        score += 0  # Hardest to value

    # Data availability
    if has_forward_eps:
        score += 2
    if has_rule_of_40:
        score += 1

    if score >= 4:
        return "high"
    elif score >= 2:
        return "medium"
    else:
        return "low"


# ====================
# CONVENIENCE FUNCTIONS
# ====================


def get_growth_metrics(
    revenue_growth: Optional[float] = None,
    earnings_growth: Optional[float] = None,
    fcf_margin: Optional[float] = None,
    operating_margin: Optional[float] = None,
) -> GrowthMetrics:
    """
    Calculate all growth-related metrics from inputs.

    Args:
        revenue_growth: YoY revenue growth
        earnings_growth: YoY earnings growth
        fcf_margin: Free cash flow margin
        operating_margin: Operating margin

    Returns:
        GrowthMetrics with all calculated values
    """
    growth_profile = classify_growth_profile(revenue_growth, earnings_growth)
    rule_of_40 = calculate_rule_of_40(revenue_growth, fcf_margin, operating_margin)
    quality_tier = classify_quality_tier(rule_of_40)

    return GrowthMetrics(
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        fcf_margin=fcf_margin,
        rule_of_40=rule_of_40,
        growth_profile=growth_profile,
        quality_tier=quality_tier,
    )
