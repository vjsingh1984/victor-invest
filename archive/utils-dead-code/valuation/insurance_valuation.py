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

Author: Claude Code
Date: 2025-11-10
"""

import logging
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def value_insurance_company(
    symbol: str, financials: Dict, current_price: float, database_url: Optional[str] = None
) -> Dict:
    """
    Value insurance company using Price-to-Book (P/BV) methodology

    Args:
        symbol: Stock symbol
        financials: Dictionary of financial metrics (latest quarter for book value)
        current_price: Current stock price
        database_url: Optional database connection string for TTM calculation

    Returns:
        Dictionary with fair_value and valuation details
    """
    warnings = []

    # Extract required metrics (from latest quarter)
    stockholders_equity = financials.get("stockholders_equity", 0)
    shares_outstanding = financials.get("shares_outstanding", 0)
    total_revenue = financials.get("total_revenue", 0)

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
        net_income = financials.get("net_income", 0)
        if not net_income:
            raise ValueError(f"{symbol} - No net income data available")
        roe = (net_income / stockholders_equity) * 100
        warnings.append("Using quarterly data instead of TTM (may be less accurate)")
        logger.warning(f"{symbol} - Falling back to quarterly ROE calculation")

    # Estimate Combined Ratio (industry proxy if not available)
    # Note: Ideally we'd extract this from insurance-specific XBRL tags
    # For now, use net margin as proxy for underwriting quality
    net_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0

    # Determine target P/BV ratio based on ROE and underwriting quality
    target_pb, confidence = _determine_target_pb(symbol, roe, net_margin, warnings)

    # Calculate fair value
    fair_value = book_value_per_share * target_pb

    # Current P/BV ratio
    current_pb = current_price / book_value_per_share if book_value_per_share > 0 else 0

    logger.info(
        f"{symbol} - Insurance Valuation: "
        f"ROE={roe:.1f}%, Net Margin={net_margin:.1f}%, "
        f"BV/share=${book_value_per_share:.2f}, "
        f"Current P/BV={current_pb:.2f}x, Target P/BV={target_pb:.2f}x, "
        f"Fair value=${fair_value:.2f}"
    )

    return {
        "fair_value": fair_value,
        "book_value_per_share": book_value_per_share,
        "target_pb_ratio": target_pb,
        "current_pb_ratio": current_pb,
        "roe": roe,
        "net_margin": net_margin,
        "combined_ratio": None,  # Would need insurance-specific data
        "confidence": confidence,
        "warnings": warnings,
    }


def _calculate_ttm_metrics(
    symbol: str, database_url: Optional[str], warnings: List[str]
) -> tuple[Optional[float], Optional[float]]:
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
        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Query last 4 quarters sorted by period_end_date DESC
            query = text(
                """
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
            """
            )

            results = conn.execute(query, {"symbol": symbol}).fetchall()

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


def _determine_target_pb(symbol: str, roe: float, net_margin: float, warnings: List[str]) -> tuple[float, str]:
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
        logger.info(
            f"{symbol} - Excellent insurer profile (ROE={roe:.1f}%, Margin={net_margin:.1f}%) → P/BV={target_pb:.2f}x"
        )

    # Good insurers: ROE > 12%, Margin > 8%
    elif roe >= 12 and net_margin >= 8:
        target_pb = 1.20
        confidence = "high"
        logger.info(
            f"{symbol} - Good insurer profile (ROE={roe:.1f}%, Margin={net_margin:.1f}%) → P/BV={target_pb:.2f}x"
        )

    # Average insurers: ROE > 10%, Margin > 5%
    elif roe >= 10 and net_margin >= 5:
        target_pb = 1.00
        confidence = "medium"
        logger.info(
            f"{symbol} - Average insurer profile (ROE={roe:.1f}%, Margin={net_margin:.1f}%) → P/BV={target_pb:.2f}x"
        )

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


def calculate_insurance_specific_metrics(symbol: str, financials: Dict) -> Dict:
    """
    Calculate insurance-specific metrics

    Future enhancement: Extract from insurance-specific XBRL tags:
    - PremiumsEarnedNet (revenue)
    - PolicyholderBenefitsAndClaimsIncurred (claims)
    - DeferredPolicyAcquisitionCosts (DAC)
    - UnearneledPremiums (float)
    - LossAndLossAdjustmentExpenseReserve (reserves)

    Args:
        symbol: Stock symbol
        financials: Dictionary of financial metrics

    Returns:
        Dictionary with insurance-specific metrics
    """
    metrics = {}

    # These would come from insurance-specific XBRL tags
    # For now, return placeholders
    metrics["premiums_earned"] = financials.get("total_revenue")
    metrics["combined_ratio"] = None  # Would calculate: (claims + expenses) / premiums_earned
    metrics["loss_ratio"] = None
    metrics["expense_ratio"] = None
    metrics["float"] = None  # Unearned premiums + loss reserves

    logger.debug(f"{symbol} - Insurance metrics extraction not yet implemented")

    return metrics
