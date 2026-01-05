"""
Trend Analyzer - Extracted from FundamentalAnalysisAgent for SRP.

This module handles financial trend analysis:
- Revenue trend analysis (accelerating, stable, decelerating)
- Margin trend analysis (expanding, stable, contracting)
- Cash flow trend analysis (improving, stable, deteriorating)
- Quarterly comparisons (Q/Q and Y/Y)
- Cyclical pattern detection

Part of Phase 5 refactoring to break up monolithic agent.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class QuarterlyDataProtocol(Protocol):
    """Protocol for quarterly data objects."""

    fiscal_year: int
    fiscal_period: str
    financial_data: Dict[str, Any]
    ratios: Optional[Dict[str, Any]]


class TrendAnalyzer:
    """
    Analyzes financial trends from quarterly data.

    Extracted from FundamentalAnalysisAgent to follow Single Responsibility Principle.
    All trend analysis logic is centralized here.

    Trend types analyzed:
    - Revenue: accelerating, stable, decelerating
    - Margins: expanding, stable, contracting
    - Cash flow: improving, stable, deteriorating
    - Cyclical patterns: seasonal business patterns
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize trend analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_revenue_trend(self, quarterly_data: List[QuarterlyDataProtocol]) -> Dict:
        """
        Analyze revenue trend: accelerating, stable, or decelerating.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with:
                - trend: 'accelerating', 'stable', or 'decelerating'
                - q_over_q_growth: List of quarter-over-quarter growth rates (%)
                - y_over_y_growth: List of year-over-year growth rates (%)
                - average_growth: Average Q/Q growth rate (%)
                - volatility: Standard deviation of Q/Q growth (%)
                - consistency_score: 0-100 (higher = more consistent)
        """
        if len(quarterly_data) < 2:
            return {
                "trend": "insufficient_data",
                "q_over_q_growth": [],
                "y_over_y_growth": [],
                "average_growth": 0.0,
                "volatility": 0.0,
                "consistency_score": 0.0,
            }

        # Extract revenues
        revenues = [q.financial_data.get("revenues", 0) for q in quarterly_data]

        # Calculate Q/Q growth rates
        qoq_growth = []
        for i in range(1, len(revenues)):
            if revenues[i - 1] > 0:
                growth = ((float(revenues[i]) - float(revenues[i - 1])) / float(revenues[i - 1])) * 100
                qoq_growth.append(growth)
            else:
                qoq_growth.append(0.0)

        # Calculate Y/Y growth rates (4 quarters lag)
        yoy_growth = []
        for i in range(4, len(revenues)):
            if revenues[i - 4] > 0:
                growth = ((float(revenues[i]) - float(revenues[i - 4])) / float(revenues[i - 4])) * 100
                yoy_growth.append(growth)
            else:
                yoy_growth.append(0.0)

        # Calculate average growth
        avg_growth = sum(qoq_growth) / len(qoq_growth) if qoq_growth else 0.0

        # Calculate volatility (standard deviation)
        if len(qoq_growth) > 1:
            variance = sum((g - avg_growth) ** 2 for g in qoq_growth) / len(qoq_growth)
            volatility = variance**0.5
        else:
            volatility = 0.0

        # Calculate consistency score (0-100, inverse of volatility)
        if volatility > 0:
            consistency_score = max(0, min(100, 100 - (volatility * 5)))
        else:
            consistency_score = 100.0

        # Determine trend (accelerating, stable, or decelerating)
        if len(qoq_growth) >= 6:
            early_avg = sum(qoq_growth[:3]) / 3
            late_avg = sum(qoq_growth[-3:]) / 3

            # Threshold for acceleration/deceleration: 2 percentage points
            if late_avg > early_avg + 2.0:
                trend = "accelerating"
            elif late_avg < early_avg - 2.0:
                trend = "decelerating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "q_over_q_growth": [round(g, 2) for g in qoq_growth],
            "y_over_y_growth": [round(g, 2) for g in yoy_growth],
            "average_growth": round(avg_growth, 2),
            "volatility": round(volatility, 2),
            "consistency_score": round(consistency_score, 1),
        }

    def analyze_margin_trend(self, quarterly_data: List[QuarterlyDataProtocol]) -> Dict:
        """
        Analyze margin trends: expanding, stable, or contracting.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with margin trends and historical values
        """
        if len(quarterly_data) < 2:
            return {
                "gross_margin_trend": "insufficient_data",
                "operating_margin_trend": "insufficient_data",
                "net_margin_trend": "insufficient_data",
                "gross_margins": [],
                "operating_margins": [],
                "net_margins": [],
            }

        gross_margins = []
        operating_margins = []
        net_margins = []

        for q in quarterly_data:
            revenue = q.financial_data.get("revenues", 0)
            net_income = q.financial_data.get("net_income", 0)

            # Calculate margins
            if revenue > 0:
                # Net margin (we have this directly)
                net_margin = (float(net_income) / float(revenue)) * 100
                net_margins.append(net_margin)

                # For gross and operating margins, use ratios if available
                if q.ratios:
                    gross_margin = q.ratios.get("profit_margin", net_margin)
                    operating_margin = q.ratios.get("profit_margin", net_margin)
                else:
                    # Fallback to net margin as proxy
                    gross_margin = net_margin
                    operating_margin = net_margin

                gross_margins.append(gross_margin)
                operating_margins.append(operating_margin)
            else:
                gross_margins.append(0.0)
                operating_margins.append(0.0)
                net_margins.append(0.0)

        # Determine trends (compare early vs late quarters)
        def determine_margin_trend(margins):
            if len(margins) < 4:
                return "stable"

            early_avg = sum(float(m) for m in margins[: len(margins) // 2]) / (len(margins) // 2)
            late_avg = sum(float(m) for m in margins[len(margins) // 2 :]) / (len(margins) - len(margins) // 2)

            # Threshold: 1 percentage point
            if late_avg > early_avg + 1.0:
                return "expanding"
            elif late_avg < early_avg - 1.0:
                return "contracting"
            else:
                return "stable"

        return {
            "gross_margin_trend": determine_margin_trend(gross_margins),
            "operating_margin_trend": determine_margin_trend(operating_margins),
            "net_margin_trend": determine_margin_trend(net_margins),
            "gross_margins": [round(m, 2) for m in gross_margins],
            "operating_margins": [round(m, 2) for m in operating_margins],
            "net_margins": [round(m, 2) for m in net_margins],
        }

    def analyze_cash_flow_trend(self, quarterly_data: List[QuarterlyDataProtocol]) -> Dict:
        """
        Analyze cash flow quality and trend.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with cash flow metrics and quality score
        """
        if len(quarterly_data) < 2:
            return {
                "trend": "insufficient_data",
                "operating_cash_flow": [],
                "free_cash_flow": [],
                "cash_conversion_ratio": [],
                "quality_of_earnings": 0.0,
            }

        ocf_values = []
        fcf_values = []
        cash_conversion = []

        for q in quarterly_data:
            ocf = q.financial_data.get("operating_cash_flow", 0)
            capex = q.financial_data.get("capital_expenditures", 0)
            net_income = q.financial_data.get("net_income", 0)

            ocf_values.append(ocf)
            fcf = ocf - capex
            fcf_values.append(fcf)

            # Cash conversion ratio (OCF / Net Income)
            if net_income > 0:
                conversion = (ocf / net_income) * 100
                cash_conversion.append(conversion)
            else:
                cash_conversion.append(0.0)

        # Determine trend
        if len(ocf_values) >= 4:
            early_avg = sum(ocf_values[: len(ocf_values) // 2]) / (len(ocf_values) // 2)
            late_avg = sum(ocf_values[len(ocf_values) // 2 :]) / (len(ocf_values) - len(ocf_values) // 2)

            if late_avg > early_avg * 1.1:  # 10% improvement
                trend = "improving"
            elif late_avg < early_avg * 0.9:  # 10% decline
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Quality of earnings score (0-100)
        if cash_conversion:
            avg_conversion = sum(cash_conversion) / len(cash_conversion)
            if avg_conversion >= 100:
                quality_score = min(100, 95 + (avg_conversion - 100) / 20)
            elif avg_conversion >= 80:
                quality_score = 80 + (avg_conversion - 80)
            elif avg_conversion >= 50:
                quality_score = 50 + (avg_conversion - 50) * 0.6
            else:
                quality_score = avg_conversion
        else:
            quality_score = 0.0

        return {
            "trend": trend,
            "operating_cash_flow": [round(ocf, 0) for ocf in ocf_values],
            "free_cash_flow": [round(fcf, 0) for fcf in fcf_values],
            "cash_conversion_ratio": [round(cc, 1) for cc in cash_conversion],
            "quality_of_earnings": round(quality_score, 1),
        }

    def calculate_quarterly_comparisons(self, quarterly_data: List[QuarterlyDataProtocol]) -> Dict:
        """
        Calculate quarter-over-quarter and year-over-year comparisons.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with Q/Q and Y/Y comparisons for key metrics
        """
        if len(quarterly_data) < 2:
            return {
                "q_over_q": {"revenue": [], "net_income": [], "eps": []},
                "y_over_y": {"revenue": [], "net_income": [], "eps": []},
            }

        revenues = [q.financial_data.get("revenues", 0) for q in quarterly_data]
        net_incomes = [q.financial_data.get("net_income", 0) for q in quarterly_data]

        # Q/Q comparisons
        qoq_revenue = []
        qoq_net_income = []

        for i in range(1, len(revenues)):
            if revenues[i - 1] > 0:
                qoq_revenue.append(((float(revenues[i]) - float(revenues[i - 1])) / float(revenues[i - 1])) * 100)
            else:
                qoq_revenue.append(0.0)

            if net_incomes[i - 1] > 0:
                qoq_net_income.append(
                    ((float(net_incomes[i]) - float(net_incomes[i - 1])) / float(net_incomes[i - 1])) * 100
                )
            else:
                qoq_net_income.append(0.0)

        # Y/Y comparisons (4 quarters lag)
        yoy_revenue = []
        yoy_net_income = []

        for i in range(4, len(revenues)):
            if revenues[i - 4] > 0:
                yoy_revenue.append(((float(revenues[i]) - float(revenues[i - 4])) / float(revenues[i - 4])) * 100)
            else:
                yoy_revenue.append(0.0)

            if net_incomes[i - 4] > 0:
                yoy_net_income.append(
                    ((float(net_incomes[i]) - float(net_incomes[i - 4])) / float(net_incomes[i - 4])) * 100
                )
            else:
                yoy_net_income.append(0.0)

        return {
            "q_over_q": {
                "revenue": [round(g, 2) for g in qoq_revenue],
                "net_income": [round(g, 2) for g in qoq_net_income],
                "eps": [],  # EPS not yet calculated
            },
            "y_over_y": {
                "revenue": [round(g, 2) for g in yoy_revenue],
                "net_income": [round(g, 2) for g in yoy_net_income],
                "eps": [],  # EPS not yet calculated
            },
        }

    def detect_cyclical_patterns(self, quarterly_data: List[QuarterlyDataProtocol]) -> Dict:
        """
        Detect seasonal/cyclical business patterns.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with cyclical pattern analysis
        """
        if len(quarterly_data) < 8:
            return {
                "is_cyclical": False,
                "seasonal_pattern": "insufficient_data",
                "quarterly_strength": {},
                "pattern_confidence": 0.0,
            }

        # Group revenues by quarter (Q1, Q2, Q3, Q4)
        quarter_revenues: Dict[str, List[float]] = {
            "Q1": [],
            "Q2": [],
            "Q3": [],
            "Q4": [],
        }

        for q in quarterly_data:
            period = q.fiscal_period
            revenue = q.financial_data.get("revenues", 0)
            if period in quarter_revenues:
                quarter_revenues[period].append(revenue)

        # Calculate average revenue per quarter
        quarter_averages = {}
        for period, revenues in quarter_revenues.items():
            if revenues:
                quarter_averages[period] = sum(revenues) / len(revenues)
            else:
                quarter_averages[period] = 0

        # Calculate overall average
        overall_avg = sum(quarter_averages.values()) / len(quarter_averages) if quarter_averages else 0

        # Calculate strength (% above/below average)
        quarterly_strength = {}
        for period, avg in quarter_averages.items():
            if overall_avg > 0:
                strength = ((avg - overall_avg) / overall_avg) * 100
                quarterly_strength[period] = round(strength, 1)
            else:
                quarterly_strength[period] = 0.0

        # Determine if cyclical (any quarter > 15% different from average)
        max_deviation = max(abs(s) for s in quarterly_strength.values())
        is_cyclical = max_deviation > 15.0

        # Identify strongest quarter
        strongest_quarter = max(quarterly_strength, key=quarterly_strength.get)

        # Pattern confidence (based on consistency across years)
        if len(quarterly_data) >= 8:
            pattern_confidence = min(100, max_deviation * 3)
        else:
            pattern_confidence = max_deviation * 2

        # Determine seasonal pattern
        if is_cyclical:
            if quarterly_strength[strongest_quarter] > 15:
                seasonal_pattern = f"{strongest_quarter}_strong"
            else:
                seasonal_pattern = "moderate_cyclical"
        else:
            seasonal_pattern = "non_cyclical"

        return {
            "is_cyclical": is_cyclical,
            "seasonal_pattern": seasonal_pattern,
            "quarterly_strength": quarterly_strength,
            "pattern_confidence": round(pattern_confidence, 1),
        }


# Singleton instance
_analyzer_instance: Optional[TrendAnalyzer] = None


def get_trend_analyzer(logger: Optional[logging.Logger] = None) -> TrendAnalyzer:
    """
    Get singleton TrendAnalyzer instance.

    Args:
        logger: Optional logger (only used on first call)

    Returns:
        TrendAnalyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TrendAnalyzer(logger)
    return _analyzer_instance
