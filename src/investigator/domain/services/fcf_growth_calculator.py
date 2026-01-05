"""
FCF Growth Calculator

Calculates geometric mean Free Cash Flow growth from quarterly metrics.
Used as input for fading growth DCF projections.

Created: 2025-11-12
Author: Claude Code
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FCFGrowthCalculator:
    """
    Calculate historical FCF growth rates from quarterly metrics

    Uses geometric mean (CAGR) methodology to calculate sustainable
    growth rates that can be used as starting points for DCF projections.
    """

    def __init__(self, symbol: str):
        """
        Initialize calculator

        Args:
            symbol: Stock symbol
        """
        self.symbol = symbol

    def calculate_geometric_mean_fcf_growth(self, quarterly_metrics: List[Dict[str, Any]], years: int = 3) -> float:
        """
        Calculate geometric mean FCF growth over N years using TTM data

        Args:
            quarterly_metrics: List of quarterly financial metrics (must have 'free_cash_flow' and 'fiscal_year')
            years: Number of years to look back (default: 3)

        Returns:
            Geometric mean FCF growth rate (decimal, e.g., 0.18 for 18%)
            Returns 0.0 if insufficient data or negative starting FCF

        Algorithm:
            1. Calculate TTM FCF for each fiscal year (sum of 4 quarters)
            2. Get most recent N years
            3. Calculate CAGR: (end_fcf / start_fcf)^(1/years) - 1
            4. Floor at 0% (no negative growth assumptions for projections)

        Example:
            >>> calc = FCFGrowthCalculator('ZS')
            >>> growth = calc.calculate_geometric_mean_fcf_growth(quarterly_metrics, years=3)
            >>> growth
            0.18  # 18% FCF CAGR
        """
        if not quarterly_metrics or len(quarterly_metrics) < 4:
            logger.warning(
                f"{self.symbol} - Insufficient quarterly data for FCF growth calculation (need at least 4 quarters)"
            )
            return 0.0

        # Step 1: Calculate TTM FCF for each fiscal year
        ttm_fcf_by_year = {}
        for q in quarterly_metrics:
            year = q.get("fiscal_year")
            fcf = q.get("free_cash_flow", 0)

            if year is None:
                continue

            if year not in ttm_fcf_by_year:
                ttm_fcf_by_year[year] = []

            ttm_fcf_by_year[year].append(fcf)

        # Step 2: Calculate TTM FCF (sum of 4 quarters) for each year
        yearly_fcf = {}
        for year, quarters in ttm_fcf_by_year.items():
            if len(quarters) >= 4:
                # Use most recent 4 quarters for this year
                yearly_fcf[year] = sum(quarters[-4:])

        if len(yearly_fcf) < 2:
            logger.warning(f"{self.symbol} - Insufficient years of FCF data (have {len(yearly_fcf)}, need at least 2)")
            return 0.0

        # Step 3: Get most recent N years
        sorted_years = sorted(yearly_fcf.keys(), reverse=True)[: years + 1]

        if len(sorted_years) < 2:
            logger.warning(f"{self.symbol} - Need at least 2 years for CAGR calculation")
            return 0.0

        # Step 4: Calculate CAGR
        start_fcf = yearly_fcf[sorted_years[-1]]
        end_fcf = yearly_fcf[sorted_years[0]]
        num_years = len(sorted_years) - 1

        if start_fcf <= 0:
            logger.warning(
                f"{self.symbol} - Cannot calculate FCF growth from negative/zero starting FCF "
                f"(FY {sorted_years[-1]}: ${start_fcf/1e6:.1f}M)"
            )
            return 0.0

        # CAGR formula: (end_value / start_value)^(1/years) - 1
        cagr = (end_fcf / start_fcf) ** (1 / num_years) - 1

        # Floor at 0% (no negative growth assumptions for forward projections)
        final_growth = max(0.0, cagr)

        logger.info(
            f"{self.symbol} - Historical FCF Growth ({num_years}Y CAGR): "
            f"{final_growth*100:.1f}% | "
            f"Start FCF (FY {sorted_years[-1]}): ${start_fcf/1e6:.1f}M | "
            f"End FCF (FY {sorted_years[0]}): ${end_fcf/1e6:.1f}M"
        )

        return final_growth

    def calculate_fcf_margin(self, quarterly_metrics: List[Dict[str, Any]], ttm: bool = True) -> float:
        """
        Calculate Free Cash Flow margin (FCF / Revenue)

        Args:
            quarterly_metrics: List of quarterly metrics
            ttm: If True, calculate TTM margin. If False, use latest quarter.

        Returns:
            FCF margin as percentage (e.g., 30.2 for 30.2%)
            Returns 0.0 if insufficient data
        """
        if not quarterly_metrics:
            return 0.0

        if ttm:
            # TTM calculation (sum of last 4 quarters)
            if len(quarterly_metrics) < 4:
                logger.warning(f"{self.symbol} - Insufficient data for TTM FCF margin (need 4 quarters)")
                return 0.0

            recent_4q = quarterly_metrics[-4:]
            ttm_fcf = sum(q.get("free_cash_flow", 0) for q in recent_4q)
            ttm_revenue = sum(q.get("total_revenue", 0) for q in recent_4q)

            if ttm_revenue == 0:
                return 0.0

            margin_pct = (ttm_fcf / ttm_revenue) * 100
        else:
            # Latest quarter only
            latest = quarterly_metrics[-1]
            fcf = latest.get("free_cash_flow", 0)
            revenue = latest.get("total_revenue", 0)

            if revenue == 0:
                return 0.0

            margin_pct = (fcf / revenue) * 100

        logger.debug(f"{self.symbol} - FCF Margin ({'TTM' if ttm else 'Q'}): {margin_pct:.1f}%")

        return margin_pct

    def __repr__(self) -> str:
        """String representation"""
        return f"FCFGrowthCalculator(symbol='{self.symbol}')"
