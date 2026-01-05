"""
Earnings Quality Assessment Module

Detects non-recurring items and adjusts earnings for valuation purposes.
This addresses the limitation of using raw reported earnings without quality filters.

Key quality signals:
1. Large one-time gains/losses (>10% of net income)
2. Goodwill impairments
3. Restructuring charges
4. Asset write-downs
5. Unusual revenue volatility
6. Accruals vs cash flow divergence

Author: Claude Code
Date: 2025-12-30
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EarningsQualityTier(Enum):
    """Earnings quality classification."""

    HIGH = "high"  # Clean earnings, minimal adjustments needed
    MODERATE = "moderate"  # Some non-recurring items, minor adjustments
    LOW = "low"  # Significant non-recurring items
    UNRELIABLE = "unreliable"  # Earnings too distorted for P/E valuation


@dataclass
class NonRecurringItem:
    """Represents a detected non-recurring item."""

    name: str
    amount: float
    as_pct_of_net_income: float
    impact: str  # "positive" or "negative" on reported earnings
    adjustment_recommended: float  # Amount to add back/subtract
    source: str  # Where detected (XBRL tag, calculation, etc.)


@dataclass
class EarningsQualityResult:
    """Result from earnings quality assessment."""

    quality_tier: EarningsQualityTier
    reported_net_income: float
    adjusted_net_income: float
    adjustment_amount: float
    adjustment_pct: float

    non_recurring_items: List[NonRecurringItem] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100 score
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # For valuation weight adjustment
    pe_reliability: float = 1.0  # 0-1, how much to trust P/E valuation
    use_adjusted_eps: bool = False


# XBRL tags for non-recurring items
NON_RECURRING_XBRL_TAGS = {
    # Gains/Losses
    "gain_on_sale_of_assets": [
        "GainLossOnSaleOfPropertyPlantEquipment",
        "GainLossOnDispositionOfAssets",
        "GainOnSaleOfBusiness",
        "GainLossOnSaleOfInvestments",
    ],
    "impairment_charges": [
        "GoodwillImpairmentLoss",
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
        "ImpairmentOfLongLivedAssetsHeldForUse",
        "AssetImpairmentCharges",
        "ImpairmentLossesRelatedToRealEstatePartnerships",
    ],
    "restructuring_charges": [
        "RestructuringCharges",
        "RestructuringAndRelatedCostIncurredCost",
        "RestructuringCostsAndAssetImpairmentCharges",
        "SeveranceCosts",
    ],
    "litigation_settlements": [
        "LitigationSettlementExpense",
        "LossContingencyAccrualAtCarryingValue",
        "GainLossRelatedToLitigationSettlement",
    ],
    "debt_extinguishment": [
        "GainsLossesOnExtinguishmentOfDebt",
        "GainLossOnRepurchaseOfDebtInstrument",
    ],
    "discontinued_operations": [
        "IncomeLossFromDiscontinuedOperationsNetOfTax",
        "DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax",
    ],
    "tax_adjustments": [
        "IncomeTaxReconciliationOtherAdjustments",
        "IncomeTaxReconciliationChangeInDeferredTaxAssetsValuationAllowance",
        "TaxCutsAndJobsActOf2017TransitionTaxForAccumulatedForeignEarningsLiability",
    ],
}

# Thresholds for materiality
MATERIALITY_THRESHOLDS = {
    "significant": 0.10,  # >10% of net income
    "material": 0.05,  # >5% of net income
    "minor": 0.02,  # >2% of net income
}


def extract_non_recurring_items(
    xbrl_data: Dict[str, Any],
    net_income: float,
) -> List[NonRecurringItem]:
    """
    Extract non-recurring items from XBRL data.

    Args:
        xbrl_data: Dictionary of XBRL facts
        net_income: Reported net income for materiality calculation

    Returns:
        List of detected non-recurring items
    """
    items = []

    if net_income == 0:
        return items

    for category, tags in NON_RECURRING_XBRL_TAGS.items():
        for tag in tags:
            if tag in xbrl_data:
                amount = xbrl_data[tag]
                if isinstance(amount, (int, float)) and amount != 0:
                    pct_of_ni = abs(amount / net_income) if net_income != 0 else 0

                    # Only flag if material (>2% of net income)
                    if pct_of_ni >= MATERIALITY_THRESHOLDS["minor"]:
                        # Determine impact direction
                        if category in ["gain_on_sale_of_assets", "debt_extinguishment"]:
                            impact = "positive" if amount > 0 else "negative"
                            adjustment = -amount  # Remove gains, add back losses
                        else:  # impairments, restructuring, litigation
                            impact = "negative" if amount > 0 else "positive"
                            adjustment = amount  # Add back charges

                        items.append(
                            NonRecurringItem(
                                name=category.replace("_", " ").title(),
                                amount=amount,
                                as_pct_of_net_income=pct_of_ni,
                                impact=impact,
                                adjustment_recommended=adjustment,
                                source=tag,
                            )
                        )

                        logger.info(
                            f"Detected non-recurring: {category} = ${amount/1e6:.1f}M " f"({pct_of_ni:.1%} of NI)"
                        )

    return items


def detect_accrual_quality(
    net_income: float,
    operating_cash_flow: float,
    revenue: float,
) -> Tuple[float, str]:
    """
    Detect earnings quality based on accruals vs cash flow.

    High-quality earnings are backed by cash flow.
    Large divergence between earnings and cash flow suggests accrual manipulation.

    Args:
        net_income: Reported net income
        operating_cash_flow: Cash from operations
        revenue: Total revenue

    Returns:
        Tuple of (quality_score 0-1, explanation)
    """
    if net_income <= 0 or revenue <= 0:
        return 0.5, "Cannot assess accrual quality with non-positive earnings/revenue"

    # Accruals = Net Income - Operating Cash Flow
    accruals = net_income - operating_cash_flow
    accruals_to_revenue = accruals / revenue

    # Cash flow coverage of earnings
    cffo_coverage = operating_cash_flow / net_income if net_income > 0 else 0

    # Quality scoring
    if cffo_coverage >= 1.2:
        # Cash flow exceeds earnings - high quality
        score = 1.0
        explanation = f"Excellent: CFO {cffo_coverage:.1f}x earnings (cash-backed)"
    elif cffo_coverage >= 0.8:
        # Cash flow roughly matches earnings
        score = 0.8
        explanation = f"Good: CFO {cffo_coverage:.1f}x earnings"
    elif cffo_coverage >= 0.5:
        # Some divergence
        score = 0.6
        explanation = f"Moderate: CFO only {cffo_coverage:.1f}x earnings (accrual-heavy)"
    elif cffo_coverage >= 0:
        # Significant divergence
        score = 0.4
        explanation = f"Low: CFO only {cffo_coverage:.1f}x earnings (high accruals)"
    else:
        # Negative operating cash flow despite positive earnings
        score = 0.2
        explanation = f"Poor: Negative CFO despite positive earnings"

    # Additional check: accruals as % of revenue
    if abs(accruals_to_revenue) > 0.15:
        score -= 0.2
        explanation += f"; High accruals ({accruals_to_revenue:.1%} of revenue)"

    return max(0, min(1, score)), explanation


def detect_revenue_quality(
    current_revenue: float,
    prior_revenue: float,
    receivables_current: float,
    receivables_prior: float,
) -> Tuple[float, str]:
    """
    Detect revenue quality based on receivables growth vs revenue growth.

    If receivables grow faster than revenue, may indicate:
    - Aggressive revenue recognition
    - Channel stuffing
    - Collection problems

    Args:
        current_revenue: Current period revenue
        prior_revenue: Prior period revenue
        receivables_current: Current accounts receivable
        receivables_prior: Prior accounts receivable

    Returns:
        Tuple of (quality_score 0-1, explanation)
    """
    if prior_revenue <= 0 or receivables_prior <= 0:
        return 0.7, "Insufficient prior data for revenue quality assessment"

    revenue_growth = (current_revenue / prior_revenue) - 1
    receivables_growth = (receivables_current / receivables_prior) - 1

    # Days Sales Outstanding change
    dso_current = (receivables_current / current_revenue) * 365
    dso_prior = (receivables_prior / prior_revenue) * 365
    dso_change = dso_current - dso_prior

    # Quality scoring
    if receivables_growth <= revenue_growth:
        score = 1.0
        explanation = "Good: Receivables growth <= revenue growth"
    elif receivables_growth <= revenue_growth + 0.10:
        score = 0.8
        explanation = (
            f"Acceptable: Receivables growing slightly faster ({receivables_growth:.1%} vs {revenue_growth:.1%})"
        )
    elif receivables_growth <= revenue_growth + 0.25:
        score = 0.6
        explanation = f"Caution: Receivables growing faster ({receivables_growth:.1%} vs {revenue_growth:.1%})"
    else:
        score = 0.4
        explanation = f"Warning: Receivables growing much faster ({receivables_growth:.1%} vs {revenue_growth:.1%})"

    # DSO increase is concerning
    if dso_change > 10:
        score -= 0.2
        explanation += f"; DSO increased {dso_change:.0f} days"

    return max(0, min(1, score)), explanation


def assess_earnings_quality(
    financials: Dict[str, Any],
    xbrl_data: Optional[Dict[str, Any]] = None,
) -> EarningsQualityResult:
    """
    Comprehensive earnings quality assessment.

    Combines:
    1. Non-recurring item detection
    2. Accrual quality analysis
    3. Revenue quality analysis

    Args:
        financials: Dictionary with net_income, operating_cash_flow, revenue, etc.
        xbrl_data: Optional XBRL facts for non-recurring detection

    Returns:
        EarningsQualityResult with adjusted earnings and quality tier
    """
    warnings = []
    recommendations = []

    # Extract key metrics
    net_income = financials.get("net_income", 0)
    operating_cf = financials.get("operating_cash_flow", 0)
    revenue = financials.get("revenue", 0) or financials.get("total_revenue", 0)
    prior_revenue = financials.get("prior_revenue", 0)
    receivables = financials.get("accounts_receivable", 0)
    prior_receivables = financials.get("prior_receivables", 0)

    # Initialize
    adjusted_ni = net_income
    total_adjustment = 0
    non_recurring_items = []
    quality_components = []

    # 1. Detect non-recurring items from XBRL
    if xbrl_data:
        non_recurring_items = extract_non_recurring_items(xbrl_data, net_income)
        for item in non_recurring_items:
            total_adjustment += item.adjustment_recommended
            if item.as_pct_of_net_income >= MATERIALITY_THRESHOLDS["significant"]:
                warnings.append(
                    f"Significant {item.name}: ${item.amount/1e6:.1f}M " f"({item.as_pct_of_net_income:.1%} of NI)"
                )

    # Calculate adjusted net income
    adjusted_ni = net_income + total_adjustment

    # 2. Accrual quality
    accrual_score = 0.7
    if net_income > 0 and operating_cf != 0:
        accrual_score, accrual_explanation = detect_accrual_quality(net_income, operating_cf, revenue)
        quality_components.append(("accrual", accrual_score, accrual_explanation))
        if accrual_score < 0.6:
            warnings.append(f"Accrual quality: {accrual_explanation}")

    # 3. Revenue quality
    revenue_score = 0.7
    if revenue > 0 and prior_revenue > 0 and receivables > 0:
        revenue_score, revenue_explanation = detect_revenue_quality(
            revenue, prior_revenue, receivables, prior_receivables
        )
        quality_components.append(("revenue", revenue_score, revenue_explanation))
        if revenue_score < 0.6:
            warnings.append(f"Revenue quality: {revenue_explanation}")

    # 4. Calculate overall quality score
    adjustment_pct = abs(total_adjustment / net_income) if net_income != 0 else 0

    # Non-recurring penalty
    non_recurring_score = max(0, 1 - adjustment_pct)
    quality_components.append(
        ("non_recurring", non_recurring_score, f"Non-recurring items: {adjustment_pct:.1%} of NI")
    )

    # Weighted quality score
    weights = {"accrual": 0.40, "revenue": 0.25, "non_recurring": 0.35}
    quality_score = sum(score * weights.get(name, 0.33) for name, score, _ in quality_components)
    quality_score = min(100, max(0, quality_score * 100))

    # Determine quality tier
    if quality_score >= 80 and adjustment_pct < 0.05:
        quality_tier = EarningsQualityTier.HIGH
        pe_reliability = 1.0
        use_adjusted = False
    elif quality_score >= 60 and adjustment_pct < 0.15:
        quality_tier = EarningsQualityTier.MODERATE
        pe_reliability = 0.85
        use_adjusted = adjustment_pct >= 0.05
    elif quality_score >= 40 and adjustment_pct < 0.30:
        quality_tier = EarningsQualityTier.LOW
        pe_reliability = 0.70
        use_adjusted = True
        recommendations.append("Consider using adjusted EPS for valuation")
    else:
        quality_tier = EarningsQualityTier.UNRELIABLE
        pe_reliability = 0.50
        use_adjusted = True
        recommendations.append("P/E valuation unreliable - weight other methods more heavily")
        recommendations.append("Consider using normalized or forward earnings")

    # Log summary
    logger.info(
        f"Earnings quality: tier={quality_tier.value}, score={quality_score:.0f}, "
        f"adjustment={adjustment_pct:.1%}, pe_reliability={pe_reliability:.0%}"
    )

    return EarningsQualityResult(
        quality_tier=quality_tier,
        reported_net_income=net_income,
        adjusted_net_income=adjusted_ni,
        adjustment_amount=total_adjustment,
        adjustment_pct=adjustment_pct,
        non_recurring_items=non_recurring_items,
        quality_score=quality_score,
        warnings=warnings,
        recommendations=recommendations,
        pe_reliability=pe_reliability,
        use_adjusted_eps=use_adjusted,
    )


def get_quality_adjusted_eps(
    reported_eps: float,
    quality_result: EarningsQualityResult,
    shares_outstanding: float,
) -> Tuple[float, str]:
    """
    Get the appropriate EPS to use based on earnings quality.

    Args:
        reported_eps: Reported EPS
        quality_result: Earnings quality assessment result
        shares_outstanding: Shares outstanding

    Returns:
        Tuple of (eps_to_use, explanation)
    """
    if not quality_result.use_adjusted_eps:
        return reported_eps, f"Using reported EPS (quality: {quality_result.quality_tier.value})"

    # Calculate adjusted EPS
    adjusted_eps = quality_result.adjusted_net_income / shares_outstanding

    explanation = (
        f"Using adjusted EPS ${adjusted_eps:.2f} (reported ${reported_eps:.2f}, "
        f"adjustment {quality_result.adjustment_pct:.1%}, "
        f"quality: {quality_result.quality_tier.value})"
    )

    return adjusted_eps, explanation
