"""
Insurance Company Valuation Module

Implements Price-to-Book (P/BV) valuation for insurance companies.

Insurance companies are valued differently than industrial companies because:
1. Balance sheet quality is paramount (assets = liabilities + float)
2. Underwriting profitability measured by Combined Ratio
3. Investment returns on float are key value driver
4. ROE is more important than revenue growth

Methodology:
- Calculate book value per share (using latest quarter)
- Calculate ROE using TTM (Trailing Twelve Months) net income
- Determine target P/BV ratio based on ROE and underwriting quality
- Fair value = Book Value per Share × Target P/BV Ratio

Combined Ratio Calculation (P1-B Enhancement):
- Combined Ratio = (Claims + Expenses) / Premiums Earned
- Uses actual XBRL insurance tags instead of net margin proxy
- Target ratios vary by insurance type (P&C, Life, Health, Reinsurance)

Author: Claude Code
Date: 2025-11-10
Updated: 2025-12-29 - Added actual combined ratio extraction from XBRL
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================
# INSURANCE TYPE CLASSIFICATION AND TARGET RATIOS
# ====================

class InsuranceType(Enum):
    """Classification of insurance company types for valuation purposes."""
    PROPERTY_CASUALTY = "property_casualty"
    LIFE = "life"
    HEALTH = "health"
    REINSURANCE = "reinsurance"
    MULTI_LINE = "multi_line"
    UNKNOWN = "unknown"


# Target combined ratios by insurance type
# Combined Ratio < 1.0 indicates underwriting profit
# Combined Ratio > 1.0 indicates underwriting loss (must rely on investment income)
TARGET_COMBINED_RATIOS: Dict[InsuranceType, float] = {
    InsuranceType.PROPERTY_CASUALTY: 0.95,  # P&C typically targets 95%
    InsuranceType.LIFE: 0.85,               # Life insurance has lower target
    InsuranceType.HEALTH: 0.88,             # Health insurance target
    InsuranceType.REINSURANCE: 0.92,        # Reinsurance slightly lower
    InsuranceType.MULTI_LINE: 0.93,         # Multi-line average
    InsuranceType.UNKNOWN: 0.95,            # Conservative default
}

# Combined ratio thresholds for valuation quality assessment
COMBINED_RATIO_THRESHOLDS = {
    'excellent': 0.90,    # < 90% = excellent underwriting
    'good': 0.95,         # < 95% = good underwriting
    'acceptable': 1.00,   # < 100% = acceptable (underwriting profit)
    'weak': 1.05,         # < 105% = weak (small underwriting loss)
    # >= 105% = poor (significant underwriting loss)
}


# ====================
# COMBINED RATIO CALCULATION (P1-B3)
# ====================

def calculate_combined_ratio(metrics: Dict) -> Optional[float]:
    """
    Calculate actual combined ratio from XBRL insurance data.

    Combined Ratio = (Claims Incurred + Policy Acquisition Costs + Operating Expenses) / Premiums Earned

    The combined ratio measures underwriting profitability:
    - < 100%: Underwriting profit (claims + expenses less than premiums)
    - = 100%: Break-even on underwriting
    - > 100%: Underwriting loss (must rely on investment income for profit)

    Args:
        metrics: Dictionary containing extracted insurance metrics:
            - premiums_earned: Net premiums earned (required)
            - claims_incurred: Claims and policyholder benefits (required)
            - policy_acquisition_costs: Deferred acquisition cost amortization (optional)
            - insurance_operating_expenses: General operating expenses (optional)

    Returns:
        Combined ratio as a decimal (e.g., 0.95 = 95%) or None if calculation not possible

    Example:
        >>> metrics = {
        ...     'premiums_earned': 10_000_000_000,  # $10B
        ...     'claims_incurred': 7_000_000_000,   # $7B
        ...     'policy_acquisition_costs': 1_500_000_000,  # $1.5B
        ...     'insurance_operating_expenses': 1_000_000_000,  # $1B
        ... }
        >>> ratio = calculate_combined_ratio(metrics)
        >>> print(f"Combined Ratio: {ratio:.1%}")  # 95.0%
    """
    premiums = metrics.get('premiums_earned')
    claims = metrics.get('claims_incurred')

    # Premiums and claims are required
    if not premiums or premiums <= 0:
        logger.debug("Cannot calculate combined ratio: premiums_earned not available or zero")
        return None

    if claims is None:
        logger.debug("Cannot calculate combined ratio: claims_incurred not available")
        return None

    # Optional expense components (default to 0 if not available)
    acquisition_costs = metrics.get('policy_acquisition_costs', 0) or 0
    operating_expenses = metrics.get('insurance_operating_expenses', 0) or 0

    # Calculate combined ratio
    total_costs = claims + acquisition_costs + operating_expenses
    combined_ratio = total_costs / premiums

    logger.info(
        f"Combined ratio calculated: {combined_ratio:.2%} "
        f"(Claims: ${claims/1e9:.2f}B + Acquisition: ${acquisition_costs/1e9:.2f}B + "
        f"Operating: ${operating_expenses/1e9:.2f}B) / Premiums: ${premiums/1e9:.2f}B"
    )

    return combined_ratio


def calculate_loss_ratio(metrics: Dict) -> Optional[float]:
    """
    Calculate loss ratio (claims / premiums).

    The loss ratio is a component of the combined ratio measuring claims only.

    Args:
        metrics: Dictionary with premiums_earned and claims_incurred

    Returns:
        Loss ratio as decimal or None if calculation not possible
    """
    premiums = metrics.get('premiums_earned')
    claims = metrics.get('claims_incurred')

    if not premiums or premiums <= 0 or claims is None:
        return None

    return claims / premiums


def calculate_expense_ratio(metrics: Dict) -> Optional[float]:
    """
    Calculate expense ratio (expenses / premiums).

    The expense ratio is a component of the combined ratio measuring expenses only.

    Args:
        metrics: Dictionary with premiums_earned, policy_acquisition_costs,
                 and insurance_operating_expenses

    Returns:
        Expense ratio as decimal or None if calculation not possible
    """
    premiums = metrics.get('premiums_earned')

    if not premiums or premiums <= 0:
        return None

    acquisition_costs = metrics.get('policy_acquisition_costs', 0) or 0
    operating_expenses = metrics.get('insurance_operating_expenses', 0) or 0

    total_expenses = acquisition_costs + operating_expenses
    if total_expenses == 0:
        return None

    return total_expenses / premiums


def assess_combined_ratio_quality(
    combined_ratio: Optional[float],
    insurance_type: InsuranceType = InsuranceType.UNKNOWN
) -> Tuple[str, str]:
    """
    Assess the quality of underwriting based on combined ratio.

    Args:
        combined_ratio: Calculated combined ratio (decimal)
        insurance_type: Type of insurance company for context

    Returns:
        Tuple of (quality_rating, description)
    """
    if combined_ratio is None:
        return ("unknown", "Combined ratio not available")

    target = TARGET_COMBINED_RATIOS.get(insurance_type, 0.95)

    if combined_ratio < COMBINED_RATIO_THRESHOLDS['excellent']:
        return ("excellent", f"Excellent underwriting ({combined_ratio:.1%} vs {target:.0%} target)")
    elif combined_ratio < COMBINED_RATIO_THRESHOLDS['good']:
        return ("good", f"Good underwriting ({combined_ratio:.1%} vs {target:.0%} target)")
    elif combined_ratio < COMBINED_RATIO_THRESHOLDS['acceptable']:
        return ("acceptable", f"Acceptable underwriting ({combined_ratio:.1%} vs {target:.0%} target)")
    elif combined_ratio < COMBINED_RATIO_THRESHOLDS['weak']:
        return ("weak", f"Weak underwriting ({combined_ratio:.1%} - slight loss)")
    else:
        return ("poor", f"Poor underwriting ({combined_ratio:.1%} - significant loss)")


def extract_insurance_metrics_from_xbrl(
    symbol: str,
    xbrl_data: Dict,
    database_url: Optional[str] = None
) -> Dict:
    """
    Extract insurance-specific metrics from XBRL data using insurance tag aliases.

    This function uses the XBRLTagAliasMapper to extract insurance-specific
    metrics that are needed for combined ratio calculation.

    Args:
        symbol: Stock ticker symbol
        xbrl_data: Raw XBRL data dictionary (us-gaap format)
        database_url: Optional database URL for fallback queries

    Returns:
        Dictionary with extracted insurance metrics:
            - premiums_earned
            - claims_incurred
            - policy_acquisition_costs
            - insurance_operating_expenses
            - loss_reserves
            - reinsurance_recoverables
            - insurance_investment_income
    """
    from utils.xbrl_tag_aliases import XBRLTagAliasMapper

    mapper = XBRLTagAliasMapper()
    metrics = {}

    # Define insurance metrics to extract
    insurance_metrics = [
        'premiums_earned',
        'claims_incurred',
        'policy_acquisition_costs',
        'insurance_operating_expenses',
        'loss_reserves',
        'reinsurance_recoverables',
        'insurance_investment_income',
    ]

    us_gaap = xbrl_data.get('facts', {}).get('us-gaap', {})
    if not us_gaap:
        logger.warning(f"{symbol} - No us-gaap data available for insurance metric extraction")
        return metrics

    for metric_name in insurance_metrics:
        # Get all XBRL tag aliases for this metric
        aliases = mapper.get_xbrl_aliases(metric_name)

        for alias in aliases:
            if alias in us_gaap:
                concept = us_gaap[alias]
                units = concept.get('units', {})
                usd_data = units.get('USD', [])

                if usd_data:
                    # Get the latest value (sorted by fiscal year/period)
                    sorted_data = sorted(
                        [d for d in usd_data if d.get('form') in ['10-Q', '10-K', '20-F']],
                        key=lambda x: (x.get('fy', 0), {'FY': 5, 'Q4': 4, 'Q3': 3, 'Q2': 2, 'Q1': 1}.get(x.get('fp', ''), 0)),
                        reverse=True
                    )

                    if sorted_data:
                        value = sorted_data[0].get('val')
                        if value is not None:
                            metrics[metric_name] = float(value)
                            logger.debug(
                                f"{symbol} - Extracted {metric_name} from {alias}: "
                                f"${float(value)/1e9:.2f}B"
                            )
                            break  # Found value, move to next metric

    # Log extraction summary
    found = [k for k in insurance_metrics if k in metrics]
    missing = [k for k in insurance_metrics if k not in metrics]

    logger.info(
        f"{symbol} - Insurance metric extraction: "
        f"Found {len(found)}/{len(insurance_metrics)} metrics. "
        f"Missing: {missing if missing else 'none'}"
    )

    return metrics


def value_insurance_company(
    symbol: str,
    financials: Dict,
    current_price: float,
    database_url: Optional[str] = None,
    xbrl_data: Optional[Dict] = None,
    insurance_type: InsuranceType = InsuranceType.UNKNOWN
) -> Dict:
    """
    Value insurance company using Price-to-Book (P/BV) methodology

    Args:
        symbol: Stock symbol
        financials: Dictionary of financial metrics (latest quarter for book value)
        current_price: Current stock price
        database_url: Optional database connection string for TTM calculation
        xbrl_data: Optional raw XBRL data for insurance metric extraction (P1-B enhancement)
        insurance_type: Type of insurance company for target ratio selection

    Returns:
        Dictionary with fair_value and valuation details
    """
    warnings = []

    # Extract required metrics (from latest quarter)
    stockholders_equity = financials.get('stockholders_equity', 0)
    shares_outstanding = financials.get('shares_outstanding', 0)
    total_revenue = financials.get('total_revenue', 0) or financials.get('revenue', 0)

    # If stockholders_equity or shares_outstanding is missing, try to fetch from database
    if not stockholders_equity or not shares_outstanding:
        logger.info(f"{symbol} - Missing equity/shares data in financials, querying database...")
        db_equity, db_shares, db_revenue = _fetch_from_database(symbol, database_url)
        if not stockholders_equity and db_equity:
            stockholders_equity = db_equity
            warnings.append("stockholders_equity from database (not in financials dict)")
        if not shares_outstanding and db_shares:
            shares_outstanding = db_shares
            warnings.append("shares_outstanding from database (not in financials dict)")
        if not total_revenue and db_revenue:
            total_revenue = db_revenue

    # Validate required data
    if not all([stockholders_equity, shares_outstanding]):
        raise ValueError(
            f"{symbol} - Missing required metrics for insurance valuation: "
            f"stockholders_equity={stockholders_equity}, "
            f"shares_outstanding={shares_outstanding}"
        )

    # Calculate book value per share (latest quarter)
    book_value_per_share = stockholders_equity / shares_outstanding

    # Calculate TTM (Trailing Twelve Months) ROE
    ttm_net_income, avg_equity = _calculate_ttm_metrics(symbol, database_url, warnings)

    # Fallback to single quarter if TTM calculation fails
    if ttm_net_income and avg_equity:
        net_income = ttm_net_income
        roe = (net_income / avg_equity) * 100
        logger.info(f"{symbol} - Using TTM metrics: NI=${net_income/1e9:.2f}B, Avg Equity=${avg_equity/1e9:.2f}B")
    else:
        net_income = financials.get('net_income', 0)
        if not net_income:
            raise ValueError(f"{symbol} - No net income data available")
        roe = (net_income / stockholders_equity) * 100
        warnings.append("Using quarterly data instead of TTM (may be less accurate)")
        logger.warning(f"{symbol} - Falling back to quarterly ROE calculation")

    # ====================
    # COMBINED RATIO EXTRACTION (P1-B Enhancement)
    # ====================
    # Try to extract actual combined ratio from XBRL insurance tags
    # Falls back to net margin proxy if XBRL data not available
    combined_ratio = None
    loss_ratio = None
    expense_ratio = None
    insurance_metrics = {}
    underwriting_quality = None
    underwriting_description = None

    if xbrl_data:
        # Extract insurance-specific metrics from XBRL
        insurance_metrics = extract_insurance_metrics_from_xbrl(symbol, xbrl_data, database_url)

        # Calculate actual combined ratio from extracted metrics
        combined_ratio = calculate_combined_ratio(insurance_metrics)
        loss_ratio = calculate_loss_ratio(insurance_metrics)
        expense_ratio = calculate_expense_ratio(insurance_metrics)

        if combined_ratio is not None:
            # Assess underwriting quality based on combined ratio
            underwriting_quality, underwriting_description = assess_combined_ratio_quality(
                combined_ratio, insurance_type
            )
            logger.info(
                f"{symbol} - Actual combined ratio extracted: {combined_ratio:.2%} "
                f"(Loss: {loss_ratio:.2%}, Expense: {expense_ratio:.2%}) - {underwriting_description}"
            )
        else:
            warnings.append("Could not extract combined ratio from XBRL - using net margin proxy")
            logger.info(f"{symbol} - Combined ratio not available, falling back to net margin proxy")
    else:
        warnings.append("No XBRL data provided - using net margin as combined ratio proxy")
        logger.debug(f"{symbol} - No XBRL data provided for insurance metric extraction")

    # Calculate net margin (used as fallback or supplementary metric)
    net_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0

    # Determine target P/BV ratio based on ROE and underwriting quality
    # Use combined ratio if available, otherwise fall back to net margin
    if combined_ratio is not None:
        # Use actual combined ratio for P/B determination
        target_pb, confidence = _determine_target_pb_from_combined_ratio(
            symbol, roe, combined_ratio, insurance_type, warnings
        )
    else:
        # Fallback: Use net margin as proxy for underwriting quality
        target_pb, confidence = _determine_target_pb(symbol, roe, net_margin, warnings)

    # Calculate fair value
    fair_value = book_value_per_share * target_pb

    # Current P/BV ratio
    current_pb = current_price / book_value_per_share if book_value_per_share > 0 else 0

    # Log valuation summary
    if combined_ratio is not None:
        logger.info(
            f"{symbol} - Insurance Valuation: "
            f"ROE={roe:.1f}%, Combined Ratio={combined_ratio:.1%}, "
            f"BV/share=${book_value_per_share:.2f}, "
            f"Current P/BV={current_pb:.2f}x, Target P/BV={target_pb:.2f}x, "
            f"Fair value=${fair_value:.2f}"
        )
    else:
        logger.info(
            f"{symbol} - Insurance Valuation: "
            f"ROE={roe:.1f}%, Net Margin={net_margin:.1f}% (proxy), "
            f"BV/share=${book_value_per_share:.2f}, "
            f"Current P/BV={current_pb:.2f}x, Target P/BV={target_pb:.2f}x, "
            f"Fair value=${fair_value:.2f}"
        )

    return {
        'fair_value': fair_value,
        'book_value_per_share': book_value_per_share,
        'target_pb_ratio': target_pb,
        'current_pb_ratio': current_pb,
        'roe': roe,
        'net_margin': net_margin,
        # P1-B Enhanced: Actual insurance metrics
        'combined_ratio': combined_ratio,
        'loss_ratio': loss_ratio,
        'expense_ratio': expense_ratio,
        'underwriting_quality': underwriting_quality,
        'underwriting_description': underwriting_description,
        'insurance_type': insurance_type.value if insurance_type else None,
        'target_combined_ratio': TARGET_COMBINED_RATIOS.get(insurance_type),
        # Raw insurance metrics for transparency
        'insurance_metrics': insurance_metrics,
        'confidence': confidence,
        'warnings': warnings,
    }


def _determine_target_pb_from_combined_ratio(
    symbol: str,
    roe: float,
    combined_ratio: float,
    insurance_type: InsuranceType,
    warnings: List[str]
) -> Tuple[float, str]:
    """
    Determine target P/BV ratio based on ROE and actual combined ratio.

    This is the P1-B enhanced version that uses actual combined ratio
    instead of net margin proxy.

    High-quality insurers with strong ROE and low combined ratio trade at 1.2-1.8x book
    Average insurers trade at 0.8-1.2x book
    Weak insurers trade at 0.5-0.8x book

    Args:
        symbol: Stock symbol
        roe: Return on Equity (%)
        combined_ratio: Actual combined ratio from XBRL (decimal, e.g., 0.95 = 95%)
        insurance_type: Type of insurance for target ratio comparison
        warnings: List to append warnings to

    Returns:
        Tuple of (target_pb_ratio, confidence_level)
    """
    target = TARGET_COMBINED_RATIOS.get(insurance_type, 0.95)

    # Assess underwriting performance relative to target
    underwriting_score = target - combined_ratio  # Positive = better than target

    # Excellent: ROE > 15% AND combined ratio significantly better than target
    if roe >= 15 and underwriting_score >= 0.05:  # 5+ points better than target
        target_pb = 1.60
        confidence = "high"
        logger.info(
            f"{symbol} - Excellent insurer (ROE={roe:.1f}%, CR={combined_ratio:.1%} vs "
            f"{target:.0%} target) -> P/BV={target_pb:.2f}x"
        )

    # Very Good: ROE > 12% AND combined ratio better than target
    elif roe >= 12 and underwriting_score >= 0:
        target_pb = 1.35
        confidence = "high"
        logger.info(
            f"{symbol} - Very good insurer (ROE={roe:.1f}%, CR={combined_ratio:.1%} vs "
            f"{target:.0%} target) -> P/BV={target_pb:.2f}x"
        )

    # Good: ROE > 10% AND combined ratio at or near target
    elif roe >= 10 and combined_ratio <= 1.00:
        target_pb = 1.15
        confidence = "high"
        logger.info(
            f"{symbol} - Good insurer (ROE={roe:.1f}%, CR={combined_ratio:.1%}) -> P/BV={target_pb:.2f}x"
        )

    # Average: ROE > 8% AND combined ratio acceptable (< 100%)
    elif roe >= 8 and combined_ratio < 1.00:
        target_pb = 1.00
        confidence = "medium"
        logger.info(
            f"{symbol} - Average insurer (ROE={roe:.1f}%, CR={combined_ratio:.1%}) -> P/BV={target_pb:.2f}x"
        )

    # Below Average: Underwriting loss but manageable
    elif combined_ratio >= 1.00 and combined_ratio < 1.05:
        target_pb = 0.85
        confidence = "medium"
        warnings.append(
            f"Underwriting loss (CR={combined_ratio:.1%}), relies on investment income"
        )
        logger.info(
            f"{symbol} - Below-average insurer (ROE={roe:.1f}%, CR={combined_ratio:.1%}) -> P/BV={target_pb:.2f}x"
        )

    # Weak: Significant underwriting losses
    else:
        target_pb = 0.70
        confidence = "low"
        warnings.append(
            f"Significant underwriting loss (CR={combined_ratio:.1%}) - distressed valuation"
        )
        logger.warning(
            f"{symbol} - Weak insurer (ROE={roe:.1f}%, CR={combined_ratio:.1%}) -> P/BV={target_pb:.2f}x"
        )

    return target_pb, confidence


def _fetch_from_database(
    symbol: str,
    database_url: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fetch latest stockholders_equity, shares_outstanding, and revenue from database.

    Args:
        symbol: Stock symbol
        database_url: Database connection string

    Returns:
        Tuple of (stockholders_equity, shares_outstanding, total_revenue) or (None, None, None)
    """
    if not database_url:
        try:
            from investigator.config import get_config
            config = get_config()
            database_url = (
                f"postgresql://{config.database.username}:{config.database.password}"
                f"@{config.database.host}:{config.database.port}/{config.database.database}"
            )
        except Exception as e:
            logger.warning(f"{symbol} - Could not load database config: {e}")
            return None, None, None

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Query latest quarter with balance sheet data
            query = text("""
                SELECT
                    stockholders_equity,
                    shares_outstanding,
                    total_revenue,
                    fiscal_year,
                    fiscal_period
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
                  AND stockholders_equity IS NOT NULL
                ORDER BY period_end_date DESC
                LIMIT 1
            """)

            result = conn.execute(query, {'symbol': symbol}).fetchone()

            if result:
                logger.info(
                    f"{symbol} - Fetched from database: "
                    f"equity=${float(result.stockholders_equity)/1e9:.2f}B, "
                    f"shares={float(result.shares_outstanding or 0)/1e6:.1f}M, "
                    f"period={result.fiscal_year}-{result.fiscal_period}"
                )
                return (
                    float(result.stockholders_equity) if result.stockholders_equity else None,
                    float(result.shares_outstanding) if result.shares_outstanding else None,
                    float(result.total_revenue) if result.total_revenue else None,
                )

            logger.warning(f"{symbol} - No balance sheet data found in database")
            return None, None, None

    except Exception as e:
        logger.warning(f"{symbol} - Database fetch failed: {e}")
        return None, None, None


def _calculate_ttm_metrics(
    symbol: str,
    database_url: Optional[str],
    warnings: List[str]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate TTM (Trailing Twelve Months) net income and average equity

    Args:
        symbol: Stock symbol
        database_url: Database connection string
        warnings: List to append warnings to

    Returns:
        Tuple of (ttm_net_income, avg_equity) or (None, None) if calculation fails
    """
    if not database_url:
        # Try to get from config
        try:
            from investigator.config import get_config
            config = get_config()
            database_url = (
                f"postgresql://{config.database.username}:{config.database.password}"
                f"@{config.database.host}:{config.database.port}/{config.database.database}"
            )
        except Exception as e:
            logger.warning(f"{symbol} - Could not load database config: {e}")
            return None, None

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Query last 4 quarters sorted by period_end_date DESC
            query = text("""
                SELECT
                    net_income,
                    stockholders_equity,
                    fiscal_year,
                    fiscal_period,
                    period_end_date
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
                  AND net_income IS NOT NULL
                  AND stockholders_equity IS NOT NULL
                ORDER BY period_end_date DESC
                LIMIT 4
            """)

            results = conn.execute(query, {'symbol': symbol}).fetchall()

            if len(results) < 4:
                logger.warning(
                    f"{symbol} - Only found {len(results)} quarters for TTM calculation "
                    f"(need 4). Falling back to quarterly data."
                )
                warnings.append(f"Insufficient quarterly data ({len(results)}/4 quarters)")
                return None, None

            # Calculate TTM net income (sum) and average equity
            # Convert Decimal to float to avoid type issues
            ttm_net_income = float(sum(row.net_income for row in results))
            avg_equity = float(sum(row.stockholders_equity for row in results)) / len(results)

            logger.info(
                f"{symbol} - TTM calculation: "
                f"Quarters: {[f'{r.fiscal_year}-{r.fiscal_period}' for r in results]}, "
                f"TTM NI=${ttm_net_income/1e9:.2f}B, "
                f"Avg Equity=${avg_equity/1e9:.2f}B"
            )

            return ttm_net_income, avg_equity

    except Exception as e:
        logger.warning(f"{symbol} - TTM calculation failed: {e}")
        warnings.append(f"TTM calculation failed: {str(e)}")
        return None, None


def _determine_target_pb(
    symbol: str,
    roe: float,
    net_margin: float,
    warnings: List[str]
) -> Tuple[float, str]:
    """
    Determine target P/BV ratio based on ROE and underwriting quality

    High-quality insurers with strong ROE trade at 1.2-1.8x book
    Average insurers trade at 0.8-1.2x book
    Weak insurers trade at 0.5-0.8x book

    Args:
        symbol: Stock symbol
        roe: Return on Equity (%)
        net_margin: Net profit margin (%) - proxy for underwriting quality
        warnings: List to append warnings to

    Returns:
        Tuple of (target_pb_ratio, confidence_level)
    """

    # Excellent insurers: ROE > 15%, Margin > 10%
    if roe >= 15 and net_margin >= 10:
        target_pb = 1.50
        confidence = "high"
        logger.info(f"{symbol} - Excellent insurer profile (ROE={roe:.1f}%, Margin={net_margin:.1f}%) → P/BV={target_pb:.2f}x")

    # Good insurers: ROE > 12%, Margin > 8%
    elif roe >= 12 and net_margin >= 8:
        target_pb = 1.20
        confidence = "high"
        logger.info(f"{symbol} - Good insurer profile (ROE={roe:.1f}%, Margin={net_margin:.1f}%) → P/BV={target_pb:.2f}x")

    # Average insurers: ROE > 10%, Margin > 5%
    elif roe >= 10 and net_margin >= 5:
        target_pb = 1.00
        confidence = "medium"
        logger.info(f"{symbol} - Average insurer profile (ROE={roe:.1f}%, Margin={net_margin:.1f}%) → P/BV={target_pb:.2f}x")

    # Below-average insurers: ROE > 8%, Margin > 3%
    elif roe >= 8 and net_margin >= 3:
        target_pb = 0.85
        confidence = "medium"
        warnings.append(f"Below-average profitability (ROE={roe:.1f}%, Margin={net_margin:.1f}%)")
        logger.info(f"{symbol} - Below-average insurer profile → P/BV={target_pb:.2f}x")

    # Weak insurers: ROE < 8% or Margin < 3%
    else:
        target_pb = 0.70
        confidence = "low"
        warnings.append(f"Weak profitability (ROE={roe:.1f}%, Margin={net_margin:.1f}%) suggests distressed insurer")
        logger.warning(f"{symbol} - Weak insurer profile → P/BV={target_pb:.2f}x")

    return target_pb, confidence


def calculate_insurance_specific_metrics(
    symbol: str,
    financials: Dict,
    xbrl_data: Optional[Dict] = None
) -> Dict:
    """
    Calculate insurance-specific metrics from XBRL data or financial metrics.

    P1-B Enhanced: Now extracts actual insurance metrics from XBRL tags:
    - PremiumsEarnedNet (revenue)
    - PolicyholderBenefitsAndClaimsIncurred (claims)
    - DeferredPolicyAcquisitionCosts (DAC)
    - UnearnedPremiums (float)
    - LossAndLossAdjustmentExpenseReserve (reserves)

    Args:
        symbol: Stock symbol
        financials: Dictionary of financial metrics
        xbrl_data: Optional raw XBRL data for insurance metric extraction

    Returns:
        Dictionary with insurance-specific metrics including:
        - premiums_earned
        - claims_incurred
        - policy_acquisition_costs
        - combined_ratio
        - loss_ratio
        - expense_ratio
        - float (unearned premiums + loss reserves)
    """
    metrics = {}

    # If XBRL data is provided, extract actual insurance metrics
    if xbrl_data:
        insurance_metrics = extract_insurance_metrics_from_xbrl(symbol, xbrl_data)

        # Copy extracted metrics
        metrics.update(insurance_metrics)

        # Calculate ratios
        metrics['combined_ratio'] = calculate_combined_ratio(insurance_metrics)
        metrics['loss_ratio'] = calculate_loss_ratio(insurance_metrics)
        metrics['expense_ratio'] = calculate_expense_ratio(insurance_metrics)

        # Calculate insurance float (unearned premiums + loss reserves)
        loss_reserves = insurance_metrics.get('loss_reserves', 0) or 0
        # Note: Unearned premiums might be in loss_reserves or need separate extraction
        metrics['float'] = loss_reserves

        # Assess underwriting quality
        if metrics['combined_ratio'] is not None:
            quality, description = assess_combined_ratio_quality(metrics['combined_ratio'])
            metrics['underwriting_quality'] = quality
            metrics['underwriting_description'] = description

        logger.info(
            f"{symbol} - Insurance metrics calculated: "
            f"Premiums=${metrics.get('premiums_earned', 0)/1e9:.2f}B, "
            f"Combined Ratio={metrics.get('combined_ratio', 'N/A')}"
        )
    else:
        # Fallback: Use financials dict with placeholders
        metrics['premiums_earned'] = financials.get('total_revenue') or financials.get('premiums_earned')
        metrics['claims_incurred'] = financials.get('claims_incurred')
        metrics['policy_acquisition_costs'] = financials.get('policy_acquisition_costs')
        metrics['insurance_operating_expenses'] = financials.get('insurance_operating_expenses')

        # Try to calculate combined ratio if we have the data
        metrics['combined_ratio'] = calculate_combined_ratio(metrics)
        metrics['loss_ratio'] = calculate_loss_ratio(metrics)
        metrics['expense_ratio'] = calculate_expense_ratio(metrics)
        metrics['float'] = None  # Cannot calculate without XBRL data

        if metrics['combined_ratio'] is None:
            logger.debug(f"{symbol} - Combined ratio not available from financials dict")

    return metrics
