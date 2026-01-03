"""
Bank Valuation Module

Implements Price-to-Book (P/B) valuation for banks and financial institutions.

Banks are valued differently than industrial companies because:
1. Balance sheet quality is paramount (assets = loans + securities)
2. Net Interest Margin (NIM) measures core profitability
3. Credit quality measured by Non-Performing Loan (NPL) ratios
4. Capital adequacy measured by Tier 1 capital ratio
5. Operating efficiency measured by efficiency ratio
6. ROE is more important than revenue growth

Methodology:
- Calculate book value per share (using latest quarter)
- Calculate ROE using TTM (Trailing Twelve Months) net income
- Determine target P/B ratio based on ROE, efficiency, and credit quality
- Fair value = Book Value per Share x Target P/B Ratio

P/B Target Determination:
- ROE > 15%, efficiency < 55%: P/B = 1.5x (excellent bank)
- ROE > 12%, efficiency < 60%: P/B = 1.2x (good bank)
- ROE > 10%: P/B = 1.0x (average bank)
- ROE < 8%: P/B = 0.7x (weak bank)

Author: Claude Code
Date: 2025-12-30
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================
# BANK TYPE CLASSIFICATION
# ====================

class BankType(Enum):
    """Classification of bank types for valuation purposes."""
    REGIONAL = "regional"           # Regional banks focused on specific geographic areas
    DIVERSIFIED = "diversified"     # Large diversified banks (money center banks)
    INVESTMENT = "investment"       # Investment banks and broker-dealers
    CREDIT_UNION = "credit_union"   # Credit unions and thrifts
    UNKNOWN = "unknown"


# ====================
# TARGET METRICS BY BANK TYPE
# ====================

# Target Net Interest Margin (NIM) by bank type
# NIM = (Interest Income - Interest Expense) / Average Earning Assets
TARGET_NIM: Dict[BankType, float] = {
    BankType.REGIONAL: 0.030,      # 3.0% - Regional banks typically have higher NIM
    BankType.DIVERSIFIED: 0.025,   # 2.5% - Diversified banks have lower NIM due to mix
    BankType.INVESTMENT: 0.015,    # 1.5% - Investment banks rely less on NIM
    BankType.CREDIT_UNION: 0.032,  # 3.2% - Credit unions often have higher NIM
    BankType.UNKNOWN: 0.028,       # 2.8% - Conservative default
}

# Target Tier 1 Capital Ratio
# Well-capitalized banks should have Tier 1 > 10%
# Adequately capitalized: 8-10%
# Under-capitalized: < 8%
TARGET_TIER1: Dict[BankType, float] = {
    BankType.REGIONAL: 0.10,       # 10% - Regional banks need solid capital base
    BankType.DIVERSIFIED: 0.12,    # 12% - Systemically important banks need higher capital
    BankType.INVESTMENT: 0.12,     # 12% - Investment banks face higher capital requirements
    BankType.CREDIT_UNION: 0.10,   # 10% - Standard requirement
    BankType.UNKNOWN: 0.10,        # 10% - Regulatory minimum for well-capitalized
}

# Target Efficiency Ratio by bank type
# Efficiency Ratio = Non-Interest Expense / (Net Interest Income + Non-Interest Income)
# Lower is better (indicates better cost management)
TARGET_EFFICIENCY_RATIO: Dict[BankType, float] = {
    BankType.REGIONAL: 0.60,       # 60% - Regional banks typically 55-65%
    BankType.DIVERSIFIED: 0.58,    # 58% - Large banks benefit from scale
    BankType.INVESTMENT: 0.65,     # 65% - Investment banks have higher compensation costs
    BankType.CREDIT_UNION: 0.62,   # 62% - Credit unions less efficient but member-focused
    BankType.UNKNOWN: 0.60,        # 60% - Conservative default
}

# Thresholds for quality assessment
EFFICIENCY_THRESHOLDS = {
    'excellent': 0.50,    # < 50% = excellent efficiency
    'good': 0.55,         # < 55% = good efficiency
    'average': 0.60,      # < 60% = average efficiency
    'weak': 0.65,         # < 65% = weak efficiency
    # >= 65% = poor efficiency
}

NPL_THRESHOLDS = {
    'excellent': 0.005,   # < 0.5% = excellent credit quality
    'good': 0.01,         # < 1.0% = good credit quality
    'average': 0.02,      # < 2.0% = average credit quality
    'weak': 0.03,         # < 3.0% = weak credit quality
    # >= 3.0% = poor credit quality (distressed)
}

TIER1_THRESHOLDS = {
    'excellent': 0.14,    # > 14% = excellent capital
    'good': 0.12,         # > 12% = good capital
    'average': 0.10,      # > 10% = average (well-capitalized)
    'weak': 0.08,         # > 8% = weak (adequately capitalized)
    # <= 8% = undercapitalized
}


# ====================
# DATA CLASSES
# ====================

@dataclass
class BankMetrics:
    """
    Bank-specific financial metrics extracted from XBRL data.

    Attributes:
        net_interest_margin: NIM as decimal (e.g., 0.03 = 3%)
        tier_1_capital_ratio: Tier 1 capital / Risk-weighted assets
        efficiency_ratio: Non-interest expense / Revenue
        npl_ratio: Non-performing loans / Total loans
        loan_to_deposit: Total loans / Total deposits
        roa: Return on Assets
        roe: Return on Equity
    """
    net_interest_margin: Optional[float] = None
    tier_1_capital_ratio: Optional[float] = None
    efficiency_ratio: Optional[float] = None
    npl_ratio: Optional[float] = None
    loan_to_deposit: Optional[float] = None
    roa: Optional[float] = None
    roe: Optional[float] = None

    # Additional metrics for transparency
    net_interest_income: Optional[float] = None
    non_interest_income: Optional[float] = None
    non_interest_expense: Optional[float] = None
    total_loans: Optional[float] = None
    total_deposits: Optional[float] = None
    non_performing_loans: Optional[float] = None
    risk_weighted_assets: Optional[float] = None
    tier_1_capital: Optional[float] = None


@dataclass
class BankValuationResult:
    """
    Result of bank valuation calculation.

    Attributes:
        fair_value: Calculated fair value per share
        book_value_per_share: Book value per share
        target_pb_ratio: Target price-to-book ratio
        current_pb_ratio: Current price-to-book ratio
        roe: Return on Equity (%)
        confidence: Confidence level (high, medium, low)
        warnings: List of warning messages
        details: Additional valuation details
    """
    fair_value: float
    book_value_per_share: float
    target_pb_ratio: float
    current_pb_ratio: float
    roe: float
    confidence: str
    warnings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


# ====================
# XBRL METRIC EXTRACTION
# ====================

def extract_bank_metrics_from_xbrl(
    symbol: str,
    xbrl_data: Dict,
    database_url: Optional[str] = None
) -> BankMetrics:
    """
    Extract bank-specific metrics from XBRL data using bank tag aliases.

    This function uses the XBRLTagAliasMapper to extract bank-specific
    metrics needed for P/B valuation.

    Args:
        symbol: Stock ticker symbol
        xbrl_data: Raw XBRL data dictionary (us-gaap format)
        database_url: Optional database URL for fallback queries

    Returns:
        BankMetrics dataclass with extracted values

    Example:
        >>> metrics = extract_bank_metrics_from_xbrl('JPM', xbrl_data)
        >>> print(f"NIM: {metrics.net_interest_margin:.2%}")
    """
    from utils.xbrl_tag_aliases import XBRLTagAliasMapper

    mapper = XBRLTagAliasMapper()
    metrics = BankMetrics()

    # Define bank metrics to extract
    bank_metric_mapping = {
        'net_interest_margin': 'net_interest_margin',
        'tier_1_capital_ratio': 'tier_1_capital_ratio',
        'efficiency_ratio': 'efficiency_ratio',
        'npl_ratio': 'npl_ratio',
        'loan_to_deposit_ratio': 'loan_to_deposit',
    }

    us_gaap = xbrl_data.get('facts', {}).get('us-gaap', {})
    if not us_gaap:
        logger.warning(f"{symbol} - No us-gaap data available for bank metric extraction")
        return metrics

    extracted_values = {}

    for metric_name, attr_name in bank_metric_mapping.items():
        # Get all XBRL tag aliases for this metric
        aliases = mapper.get_xbrl_aliases(metric_name)

        for alias in aliases:
            if alias in us_gaap:
                concept = us_gaap[alias]
                units = concept.get('units', {})

                # Bank ratios might be in 'pure' (decimal) or 'USD' units
                ratio_data = units.get('pure', []) or units.get('USD', [])

                if ratio_data:
                    # Get the latest value (sorted by fiscal year/period)
                    sorted_data = sorted(
                        [d for d in ratio_data if d.get('form') in ['10-Q', '10-K', '20-F']],
                        key=lambda x: (
                            x.get('fy', 0),
                            {'FY': 5, 'Q4': 4, 'Q3': 3, 'Q2': 2, 'Q1': 1}.get(x.get('fp', ''), 0)
                        ),
                        reverse=True
                    )

                    if sorted_data:
                        value = sorted_data[0].get('val')
                        if value is not None:
                            extracted_values[attr_name] = float(value)
                            logger.debug(
                                f"{symbol} - Extracted {metric_name} from {alias}: {float(value):.4f}"
                            )
                            break  # Found value, move to next metric

    # Set attributes on metrics object
    for attr_name, value in extracted_values.items():
        setattr(metrics, attr_name, value)

    # Log extraction summary
    found = [k for k, v in extracted_values.items() if v is not None]
    logger.info(
        f"{symbol} - Bank metric extraction: Found {len(found)}/{len(bank_metric_mapping)} metrics. "
        f"Found: {found if found else 'none'}"
    )

    return metrics


# ====================
# QUALITY ASSESSMENT
# ====================

def assess_bank_quality(
    metrics: BankMetrics,
    bank_type: BankType = BankType.UNKNOWN
) -> Tuple[str, str]:
    """
    Assess the overall quality of a bank based on key metrics.

    Quality is assessed based on:
    - ROE (primary driver)
    - Efficiency ratio (cost management)
    - NPL ratio (credit quality)
    - Tier 1 capital ratio (capital adequacy)

    Args:
        metrics: BankMetrics dataclass with extracted values
        bank_type: Type of bank for context-appropriate targets

    Returns:
        Tuple of (quality_rating, description)

    Example:
        >>> quality, desc = assess_bank_quality(metrics, BankType.REGIONAL)
        >>> print(f"Quality: {quality} - {desc}")
    """
    scores = []
    issues = []

    # Assess efficiency ratio
    if metrics.efficiency_ratio is not None:
        if metrics.efficiency_ratio < EFFICIENCY_THRESHOLDS['excellent']:
            scores.append(5)
        elif metrics.efficiency_ratio < EFFICIENCY_THRESHOLDS['good']:
            scores.append(4)
        elif metrics.efficiency_ratio < EFFICIENCY_THRESHOLDS['average']:
            scores.append(3)
        elif metrics.efficiency_ratio < EFFICIENCY_THRESHOLDS['weak']:
            scores.append(2)
            issues.append(f"High efficiency ratio ({metrics.efficiency_ratio:.1%})")
        else:
            scores.append(1)
            issues.append(f"Poor efficiency ratio ({metrics.efficiency_ratio:.1%})")

    # Assess NPL ratio (credit quality)
    if metrics.npl_ratio is not None:
        if metrics.npl_ratio < NPL_THRESHOLDS['excellent']:
            scores.append(5)
        elif metrics.npl_ratio < NPL_THRESHOLDS['good']:
            scores.append(4)
        elif metrics.npl_ratio < NPL_THRESHOLDS['average']:
            scores.append(3)
        elif metrics.npl_ratio < NPL_THRESHOLDS['weak']:
            scores.append(2)
            issues.append(f"Elevated NPL ratio ({metrics.npl_ratio:.2%})")
        else:
            scores.append(1)
            issues.append(f"High NPL ratio ({metrics.npl_ratio:.2%}) - credit concerns")

    # Assess Tier 1 capital
    if metrics.tier_1_capital_ratio is not None:
        if metrics.tier_1_capital_ratio >= TIER1_THRESHOLDS['excellent']:
            scores.append(5)
        elif metrics.tier_1_capital_ratio >= TIER1_THRESHOLDS['good']:
            scores.append(4)
        elif metrics.tier_1_capital_ratio >= TIER1_THRESHOLDS['average']:
            scores.append(3)
        elif metrics.tier_1_capital_ratio >= TIER1_THRESHOLDS['weak']:
            scores.append(2)
            issues.append(f"Low Tier 1 capital ({metrics.tier_1_capital_ratio:.1%})")
        else:
            scores.append(1)
            issues.append(f"Undercapitalized (Tier 1: {metrics.tier_1_capital_ratio:.1%})")

    # Assess ROE
    if metrics.roe is not None:
        if metrics.roe >= 15:
            scores.append(5)
        elif metrics.roe >= 12:
            scores.append(4)
        elif metrics.roe >= 10:
            scores.append(3)
        elif metrics.roe >= 8:
            scores.append(2)
            issues.append(f"Below-average ROE ({metrics.roe:.1f}%)")
        else:
            scores.append(1)
            issues.append(f"Weak ROE ({metrics.roe:.1f}%)")

    # Calculate average score
    if not scores:
        return ("unknown", "Insufficient data for quality assessment")

    avg_score = sum(scores) / len(scores)

    # Determine quality rating
    if avg_score >= 4.5:
        quality = "excellent"
        description = f"Excellent bank quality (score: {avg_score:.1f}/5)"
    elif avg_score >= 3.5:
        quality = "good"
        description = f"Good bank quality (score: {avg_score:.1f}/5)"
    elif avg_score >= 2.5:
        quality = "average"
        description = f"Average bank quality (score: {avg_score:.1f}/5)"
    elif avg_score >= 1.5:
        quality = "weak"
        description = f"Weak bank quality (score: {avg_score:.1f}/5)"
        if issues:
            description += f" - Issues: {'; '.join(issues)}"
    else:
        quality = "poor"
        description = f"Poor bank quality (score: {avg_score:.1f}/5)"
        if issues:
            description += f" - Issues: {'; '.join(issues)}"

    return (quality, description)


# ====================
# TARGET P/B DETERMINATION
# ====================

def _determine_target_pb_for_bank(
    symbol: str,
    roe: float,
    efficiency_ratio: Optional[float],
    npl_ratio: Optional[float],
    warnings: List[str]
) -> Tuple[float, str]:
    """
    Determine target P/B ratio for a bank based on ROE and quality metrics.

    P/B Target Framework:
    - ROE > 15%, efficiency < 55%: P/B = 1.5x (excellent bank)
    - ROE > 12%, efficiency < 60%: P/B = 1.2x (good bank)
    - ROE > 10%: P/B = 1.0x (average bank)
    - ROE < 8%: P/B = 0.7x (weak bank)

    Additional adjustments based on NPL ratio and efficiency.

    Args:
        symbol: Stock ticker symbol
        roe: Return on Equity (as percentage, e.g., 15.0 for 15%)
        efficiency_ratio: Efficiency ratio (as decimal, e.g., 0.55 for 55%)
        npl_ratio: Non-performing loan ratio (as decimal, e.g., 0.01 for 1%)
        warnings: List to append warnings to

    Returns:
        Tuple of (target_pb_ratio, confidence_level)
    """
    # Default efficiency and NPL if not provided
    eff = efficiency_ratio if efficiency_ratio is not None else 0.60  # Assume average
    npl = npl_ratio if npl_ratio is not None else 0.015  # Assume average

    # Log if using defaults
    if efficiency_ratio is None:
        warnings.append("Efficiency ratio not available, using 60% default")
        logger.debug(f"{symbol} - Using default efficiency ratio of 60%")
    if npl_ratio is None:
        warnings.append("NPL ratio not available, using 1.5% default")
        logger.debug(f"{symbol} - Using default NPL ratio of 1.5%")

    # Determine base P/B based on ROE and efficiency
    if roe >= 15 and eff < 0.55:
        # Excellent bank: High ROE with excellent efficiency
        target_pb = 1.50
        confidence = "high"
        logger.info(
            f"{symbol} - Excellent bank (ROE={roe:.1f}%, Efficiency={eff:.1%}) -> P/B={target_pb:.2f}x"
        )

    elif roe >= 12 and eff < 0.60:
        # Good bank: Solid ROE with good efficiency
        target_pb = 1.20
        confidence = "high"
        logger.info(
            f"{symbol} - Good bank (ROE={roe:.1f}%, Efficiency={eff:.1%}) -> P/B={target_pb:.2f}x"
        )

    elif roe >= 10:
        # Average bank: Acceptable ROE
        target_pb = 1.00
        confidence = "medium"
        logger.info(
            f"{symbol} - Average bank (ROE={roe:.1f}%) -> P/B={target_pb:.2f}x"
        )

    elif roe >= 8:
        # Below-average bank
        target_pb = 0.85
        confidence = "medium"
        warnings.append(f"Below-average ROE ({roe:.1f}%)")
        logger.info(
            f"{symbol} - Below-average bank (ROE={roe:.1f}%) -> P/B={target_pb:.2f}x"
        )

    else:
        # Weak bank: Low ROE
        target_pb = 0.70
        confidence = "low"
        warnings.append(f"Weak ROE ({roe:.1f}%) suggests challenged bank")
        logger.warning(
            f"{symbol} - Weak bank (ROE={roe:.1f}%) -> P/B={target_pb:.2f}x"
        )

    # Apply adjustments for credit quality (NPL ratio)
    pb_adjustment = 0.0

    if npl >= NPL_THRESHOLDS['weak']:
        # High NPL - significant discount
        pb_adjustment = -0.15
        warnings.append(f"High NPL ratio ({npl:.2%}) - P/B reduced by 0.15x")
        confidence = "low"
    elif npl >= NPL_THRESHOLDS['average']:
        # Elevated NPL - modest discount
        pb_adjustment = -0.05
        warnings.append(f"Elevated NPL ratio ({npl:.2%}) - P/B reduced by 0.05x")
    elif npl < NPL_THRESHOLDS['excellent']:
        # Excellent credit quality - premium
        pb_adjustment = 0.05
        logger.debug(f"{symbol} - Excellent credit quality (NPL={npl:.2%}) - P/B increased by 0.05x")

    # Apply adjustment for efficiency (if significantly different from expectation)
    if eff > 0.65:
        # Poor efficiency - discount
        pb_adjustment -= 0.10
        warnings.append(f"Poor efficiency ratio ({eff:.1%}) - P/B reduced by 0.10x")
        if confidence == "high":
            confidence = "medium"

    # Apply total adjustment
    target_pb = max(0.50, target_pb + pb_adjustment)  # Floor at 0.5x book

    logger.info(
        f"{symbol} - Final target P/B: {target_pb:.2f}x (adjustment: {pb_adjustment:+.2f}x)"
    )

    return target_pb, confidence


# ====================
# MAIN VALUATION FUNCTION
# ====================

def value_bank(
    symbol: str,
    financials: Dict,
    current_price: float,
    xbrl_data: Optional[Dict] = None,
    bank_type: BankType = BankType.UNKNOWN,
    database_url: Optional[str] = None
) -> BankValuationResult:
    """
    Value a bank using Price-to-Book (P/B) methodology.

    Banks are valued primarily on book value and ROE, with adjustments
    for efficiency, credit quality, and capital adequacy.

    Args:
        symbol: Stock ticker symbol
        financials: Dictionary of financial metrics (latest quarter for book value)
        current_price: Current stock price
        xbrl_data: Optional raw XBRL data for bank metric extraction
        bank_type: Type of bank for target ratio selection
        database_url: Optional database connection string for TTM calculation

    Returns:
        BankValuationResult with fair value and details

    Example:
        >>> result = value_bank('JPM', financials, 150.0, xbrl_data, BankType.DIVERSIFIED)
        >>> print(f"Fair Value: ${result.fair_value:.2f}")
    """
    warnings: List[str] = []
    details: Dict = {}

    # Extract required metrics (from latest quarter)
    stockholders_equity = financials.get('stockholders_equity', 0)
    shares_outstanding = financials.get('shares_outstanding', 0)
    net_income = financials.get('net_income', 0)

    # Validate required data
    if not stockholders_equity or not shares_outstanding:
        raise ValueError(
            f"{symbol} - Missing required metrics for bank valuation: "
            f"stockholders_equity={stockholders_equity}, "
            f"shares_outstanding={shares_outstanding}"
        )

    if not net_income:
        raise ValueError(f"{symbol} - Missing net_income for ROE calculation")

    # Calculate book value per share
    book_value_per_share = stockholders_equity / shares_outstanding

    # Calculate ROE (annualized if quarterly)
    # TODO: Implement TTM ROE calculation from database
    roe = (net_income / stockholders_equity) * 100

    # Check if quarterly data needs annualization
    fiscal_period = financials.get('fiscal_period', '')
    if fiscal_period in ['Q1', 'Q2', 'Q3', 'Q4']:
        roe = roe * 4  # Annualize quarterly ROE
        warnings.append("ROE annualized from quarterly data (may be less accurate)")

    # Extract bank-specific metrics from XBRL if available
    bank_metrics = BankMetrics()
    if xbrl_data:
        bank_metrics = extract_bank_metrics_from_xbrl(symbol, xbrl_data, database_url)
        bank_metrics.roe = roe

        # Assess bank quality
        quality, quality_description = assess_bank_quality(bank_metrics, bank_type)
        details['quality_rating'] = quality
        details['quality_description'] = quality_description
    else:
        warnings.append("No XBRL data provided - using basic P/B valuation")

    # Determine target P/B ratio
    target_pb, confidence = _determine_target_pb_for_bank(
        symbol=symbol,
        roe=roe,
        efficiency_ratio=bank_metrics.efficiency_ratio,
        npl_ratio=bank_metrics.npl_ratio,
        warnings=warnings
    )

    # Calculate fair value
    fair_value = book_value_per_share * target_pb

    # Calculate current P/B ratio
    current_pb = current_price / book_value_per_share if book_value_per_share > 0 else 0

    # Populate details
    details.update({
        'bank_type': bank_type.value,
        'net_interest_margin': bank_metrics.net_interest_margin,
        'tier_1_capital_ratio': bank_metrics.tier_1_capital_ratio,
        'efficiency_ratio': bank_metrics.efficiency_ratio,
        'npl_ratio': bank_metrics.npl_ratio,
        'loan_to_deposit': bank_metrics.loan_to_deposit,
        'target_nim': TARGET_NIM.get(bank_type),
        'target_tier1': TARGET_TIER1.get(bank_type),
        'target_efficiency': TARGET_EFFICIENCY_RATIO.get(bank_type),
    })

    # Log valuation summary
    logger.info(
        f"{symbol} - Bank Valuation: "
        f"ROE={roe:.1f}%, BV/share=${book_value_per_share:.2f}, "
        f"Current P/B={current_pb:.2f}x, Target P/B={target_pb:.2f}x, "
        f"Fair value=${fair_value:.2f}"
    )

    return BankValuationResult(
        fair_value=fair_value,
        book_value_per_share=book_value_per_share,
        target_pb_ratio=target_pb,
        current_pb_ratio=current_pb,
        roe=roe,
        confidence=confidence,
        warnings=warnings,
        details=details
    )
