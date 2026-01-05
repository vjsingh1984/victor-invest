"""
Unified Terminal Growth Calculator

Centralizes terminal growth rate calculation with Rule of 40 quality adjustments.
Ensures all DCF calculations use identical logic to reward quality stocks appropriately.

Created: 2025-11-12
Author: Claude Code
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TerminalGrowthCalculator:
    """
    Unified terminal growth calculation service

    Provides consistent terminal growth rates across all valuation frameworks,
    ensuring quality stocks (high Rule of 40, strong FCF margins) receive
    appropriate credit in DCF valuations.

    Terminal Growth Logic:
    1. Quality Mature: FCF margin >25% + revenue growth >0 → Base + 0.5%
    2. High Growth: Rule of 40 >40 → Base + 1.0%
    3. Standard: No special characteristics → Base + 0.0%

    Example:
        >>> calc = TerminalGrowthCalculator(
        ...     symbol='ZS',
        ...     sector='Technology',
        ...     base_terminal_growth=0.035
        ... )
        >>> result = calc.calculate_terminal_growth(
        ...     rule_of_40_score=58.8,
        ...     revenue_growth_pct=28.6,
        ...     fcf_margin_pct=30.2
        ... )
        >>> result['terminal_growth_rate']
        0.040  # 3.5% base + 0.5% quality adjustment = 4.0%
    """

    # Terminal growth tiers
    TIER_QUALITY_MATURE = "quality_mature"
    TIER_HIGH_GROWTH = "high_growth"
    TIER_STANDARD = "standard"

    # Adjustment amounts (as decimals) - CONSERVATIVE
    # Terminal growth is PERPETUITY (forever), so must be conservative
    # Should approximate long-term GDP growth + inflation (2.5-3.5%)
    ADJUSTMENT_QUALITY_MATURE = 0.002  # +0.2% (was +0.5%, too aggressive)
    ADJUSTMENT_HIGH_GROWTH = 0.003  # +0.3% (was +1.0%, way too aggressive)
    ADJUSTMENT_STANDARD = 0.000  # +0.0% (base rate only)

    # Thresholds
    THRESHOLD_FCF_MARGIN = 25.0  # FCF margin for quality mature (%)
    THRESHOLD_RULE_OF_40 = 40.0  # Rule of 40 for high growth
    THRESHOLD_MIN_REVENUE_GROWTH = 0.0  # Minimum revenue growth for quality mature

    def __init__(self, symbol: str, sector: str, base_terminal_growth: float = 0.035):
        """
        Initialize terminal growth calculator

        Args:
            symbol: Stock symbol
            sector: Company sector (e.g., 'Technology', 'Healthcare')
            base_terminal_growth: Base terminal growth rate (default: 3.5%)
        """
        self.symbol = symbol
        self.sector = sector
        self.base_terminal_growth = base_terminal_growth
        self.last_calculation: Optional[Dict[str, Any]] = None

    def calculate_terminal_growth(
        self, rule_of_40_score: float, revenue_growth_pct: float, fcf_margin_pct: float
    ) -> Dict[str, Any]:
        """
        Calculate terminal growth rate with quality adjustments

        This is the SINGLE SOURCE OF TRUTH for terminal growth across all
        DCF calculations (sector-specific, blended, etc.).

        Args:
            rule_of_40_score: Company's Rule of 40 score (revenue growth % + profit margin %)
            revenue_growth_pct: Revenue growth percentage (e.g., 28.6 for 28.6%)
            fcf_margin_pct: Free cash flow margin percentage (e.g., 30.2 for 30.2%)

        Returns:
            Dictionary with:
                - terminal_growth_rate: Final terminal growth rate (decimal, e.g., 0.040 for 4.0%)
                - base_rate: Base sector terminal growth rate
                - adjustment: Quality adjustment applied
                - adjustment_pct: Quality adjustment as percentage (for logging)
                - reason: Human-readable explanation
                - tier: Classification tier (quality_mature/high_growth/standard)
                - metrics: Input metrics used in calculation

        Example:
            >>> result = calc.calculate_terminal_growth(58.8, 28.6, 30.2)
            >>> result
            {
                'terminal_growth_rate': 0.040,
                'base_rate': 0.035,
                'adjustment': 0.005,
                'adjustment_pct': 0.5,
                'reason': 'Mature, efficient (FCF margin 30.2% >25%, revenue growth 28.6% >0)',
                'tier': 'quality_mature',
                'metrics': {...}
            }
        """
        # Determine tier and adjustment based on company characteristics
        tier, adjustment, reason = self._classify_company(
            rule_of_40_score=rule_of_40_score, revenue_growth_pct=revenue_growth_pct, fcf_margin_pct=fcf_margin_pct
        )

        # Calculate final terminal growth rate
        final_rate = self.base_terminal_growth + adjustment

        # Create result dictionary
        result = {
            "terminal_growth_rate": final_rate,
            "base_rate": self.base_terminal_growth,
            "adjustment": adjustment,
            "adjustment_pct": adjustment * 100,  # For logging (e.g., 0.5%)
            "reason": reason,
            "tier": tier,
            "metrics": {
                "rule_of_40_score": rule_of_40_score,
                "revenue_growth_pct": revenue_growth_pct,
                "fcf_margin_pct": fcf_margin_pct,
            },
        }

        # Store for reference
        self.last_calculation = result

        # Log the calculation
        logger.info(
            f"{self.symbol} - Terminal Growth: {self.base_terminal_growth*100:.2f}% (base) "
            f"{adjustment*100:+.2f}% (quality) = {final_rate*100:.2f}% (final) | "
            f"Tier: {tier} | {reason}"
        )

        return result

    def _classify_company(
        self, rule_of_40_score: float, revenue_growth_pct: float, fcf_margin_pct: float
    ) -> tuple[str, float, str]:
        """
        Classify company into tier and determine adjustment

        Priority order:
        1. Quality Mature: High FCF margin + positive revenue growth
        2. High Growth: Strong Rule of 40 score
        3. Standard: Everything else

        Args:
            rule_of_40_score: Rule of 40 score
            revenue_growth_pct: Revenue growth percentage
            fcf_margin_pct: FCF margin percentage

        Returns:
            (tier, adjustment, reason) tuple
        """
        # Priority 1: Quality Mature Companies
        # High FCF margin shows efficiency, positive revenue growth shows stability
        if fcf_margin_pct > self.THRESHOLD_FCF_MARGIN and revenue_growth_pct > self.THRESHOLD_MIN_REVENUE_GROWTH:
            tier = self.TIER_QUALITY_MATURE
            adjustment = self.ADJUSTMENT_QUALITY_MATURE
            reason = (
                f"Mature, efficient (FCF margin {fcf_margin_pct:.1f}% >{self.THRESHOLD_FCF_MARGIN}%, "
                f"revenue growth {revenue_growth_pct:.1f}% >{self.THRESHOLD_MIN_REVENUE_GROWTH})"
            )

        # Priority 2: High Growth Companies
        # Strong Rule of 40 indicates excellent growth/profitability balance
        elif rule_of_40_score > self.THRESHOLD_RULE_OF_40:
            tier = self.TIER_HIGH_GROWTH
            adjustment = self.ADJUSTMENT_HIGH_GROWTH
            reason = f"High growth (Rule of 40: {rule_of_40_score:.1f}% >{self.THRESHOLD_RULE_OF_40})"

        # Priority 3: Standard Companies
        # No special characteristics warrant premium terminal growth
        else:
            tier = self.TIER_STANDARD
            adjustment = self.ADJUSTMENT_STANDARD
            reason = f"Standard (Rule of 40: {rule_of_40_score:.1f}%, " f"FCF margin {fcf_margin_pct:.1f}%)"

        return tier, adjustment, reason

    def get_last_calculation(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent terminal growth calculation

        Returns:
            Last calculation result, or None if no calculations performed
        """
        return self.last_calculation

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TerminalGrowthCalculator(symbol='{self.symbol}', "
            f"sector='{self.sector}', "
            f"base_rate={self.base_terminal_growth*100:.2f}%)"
        )
