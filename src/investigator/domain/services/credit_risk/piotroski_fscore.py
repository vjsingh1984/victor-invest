# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piotroski F-Score Calculator.

The Piotroski F-Score is a financial strength indicator developed by Joseph
Piotroski in 2000. It uses nine binary criteria to assess a company's financial
health, specifically designed for value investing in high book-to-market stocks.

Criteria (9 points total):

PROFITABILITY (4 points):
1. ROA > 0 (positive net income)
2. CFO > 0 (positive operating cash flow)
3. ROA increasing (current ROA > prior ROA)
4. Quality of Earnings: CFO > Net Income (accruals)

LEVERAGE & LIQUIDITY (3 points):
5. Leverage decreasing (LT Debt/Assets decreased)
6. Current ratio increasing
7. No new equity issuance (shares not diluted)

OPERATING EFFICIENCY (2 points):
8. Gross margin increasing
9. Asset turnover increasing

Interpretation:
    8-9: Strong - Excellent financial strength
    5-7: Moderate - Average financial health
    0-4: Weak - Potential financial weakness

References:
    Piotroski, J. D. (2000). "Value Investing: The Use of Historical Financial
    Statement Information to Separate Winners from Losers". Journal of
    Accounting Research, 38, 1-41.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from investigator.domain.services.credit_risk.protocols import (
    CreditScoreResult,
    FinancialData,
)

logger = logging.getLogger(__name__)


class FinancialStrength(Enum):
    """Piotroski F-Score financial strength classification."""

    STRONG = "strong"  # 8-9 points
    MODERATE = "moderate"  # 5-7 points
    WEAK = "weak"  # 0-4 points


@dataclass
class PiotroskiFScoreResult(CreditScoreResult):
    """Result of Piotroski F-Score calculation.

    Attributes:
        strength: Financial strength classification
        profitability_score: Points from profitability criteria (0-4)
        leverage_score: Points from leverage/liquidity criteria (0-3)
        efficiency_score: Points from operating efficiency criteria (0-2)
        criteria_details: Individual criterion pass/fail details
    """

    strength: Optional[FinancialStrength] = None
    profitability_score: int = 0
    leverage_score: int = 0
    efficiency_score: int = 0
    criteria_details: Dict[str, bool] = field(default_factory=dict)
    score_name: str = "Piotroski F-Score"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "strength": self.strength.value if self.strength else None,
                "profitability_score": self.profitability_score,
                "leverage_score": self.leverage_score,
                "efficiency_score": self.efficiency_score,
                "criteria_details": self.criteria_details,
                "criteria_breakdown": {
                    "profitability": f"{self.profitability_score}/4",
                    "leverage_liquidity": f"{self.leverage_score}/3",
                    "operating_efficiency": f"{self.efficiency_score}/2",
                },
            }
        )
        return result


class PiotroskiFScoreCalculator:
    """Calculator for Piotroski F-Score financial strength assessment.

    The F-Score requires two periods of data for year-over-year comparisons
    of leverage, efficiency, and profitability trends.

    SOLID: Single Responsibility - only calculates Piotroski F-Score
    """

    # Score thresholds
    STRONG_THRESHOLD = 8  # >= 8 is strong
    WEAK_THRESHOLD = 5  # < 5 is weak

    def __init__(self):
        """Initialize the F-Score calculator."""
        self._name = "Piotroski F-Score Calculator"
        self._description = (
            "Financial strength assessment using 9 binary criteria. " "8-9 = Strong, 5-7 = Moderate, 0-4 = Weak."
        )

    @property
    def name(self) -> str:
        """Return calculator name."""
        return self._name

    @property
    def description(self) -> str:
        """Return calculator description."""
        return self._description

    def validate_data(self, data: FinancialData) -> List[str]:
        """Validate required financial data fields.

        Args:
            data: Financial data to validate

        Returns:
            List of missing field names
        """
        missing = []

        # Current period fields
        current_fields = [
            ("net_income", data.net_income),
            ("total_assets", data.total_assets),
            ("operating_cash_flow", data.operating_cash_flow),
            ("revenue", data.revenue),
            ("gross_profit", data.gross_profit),
            ("current_assets", data.current_assets),
            ("current_liabilities", data.current_liabilities),
            ("long_term_debt", data.long_term_debt),
            ("shares_outstanding", data.shares_outstanding),
        ]

        for field_name, value in current_fields:
            if value is None:
                missing.append(f"current.{field_name}")

        # Prior period for comparisons
        if data.prior_period is None:
            missing.append("prior_period (required for YoY comparison)")
        else:
            prior = data.prior_period
            prior_fields = [
                ("net_income", prior.net_income),
                ("total_assets", prior.total_assets),
                ("revenue", prior.revenue),
                ("gross_profit", prior.gross_profit),
                ("current_assets", prior.current_assets),
                ("current_liabilities", prior.current_liabilities),
                ("long_term_debt", prior.long_term_debt),
                ("shares_outstanding", prior.shares_outstanding),
            ]

            for field_name, value in prior_fields:
                if value is None:
                    missing.append(f"prior.{field_name}")

        return missing

    def calculate(self, data: FinancialData) -> PiotroskiFScoreResult:
        """Calculate Piotroski F-Score from financial data.

        Args:
            data: Standardized financial data with prior period

        Returns:
            PiotroskiFScoreResult with score, strength, and criteria breakdown
        """
        result = PiotroskiFScoreResult(
            symbol=data.symbol,
            calculation_date=date.today(),
            data_date=data.data_date,
        )

        # Validate data
        missing = self.validate_data(data)
        if missing:
            result.warnings.extend([f"Missing: {f}" for f in missing[:5]])
            if len(missing) > 5:
                result.warnings.append(f"...and {len(missing) - 5} more missing fields")

        try:
            # Calculate each criterion
            criteria = {}
            components = {}

            # PROFITABILITY (4 points)
            prof_score, prof_criteria, prof_components = self._calculate_profitability(data)
            result.profitability_score = prof_score
            criteria.update(prof_criteria)
            components.update(prof_components)

            # LEVERAGE & LIQUIDITY (3 points)
            lev_score, lev_criteria, lev_components = self._calculate_leverage(data)
            result.leverage_score = lev_score
            criteria.update(lev_criteria)
            components.update(lev_components)

            # OPERATING EFFICIENCY (2 points)
            eff_score, eff_criteria, eff_components = self._calculate_efficiency(data)
            result.efficiency_score = eff_score
            criteria.update(eff_criteria)
            components.update(eff_components)

            # Total score
            total_score = prof_score + lev_score + eff_score
            result.score = total_score
            result.criteria_details = criteria
            result.components = components

            # Classify strength
            result.strength = self._classify_strength(total_score)

            # Set interpretation
            result.interpretation = self._get_interpretation(result.strength, total_score, criteria)

            logger.info(
                f"{data.symbol}: Piotroski F-Score = {total_score}/9 "
                f"({result.strength.value if result.strength else 'N/A'})"
            )

        except Exception as e:
            logger.error(f"Error calculating Piotroski F-Score for {data.symbol}: {e}")
            result.warnings.append(f"Calculation error: {str(e)}")
            result.interpretation = "Calculation failed"

        return result

    def _calculate_profitability(self, data: FinancialData) -> tuple[int, Dict[str, bool], Dict[str, Any]]:
        """Calculate profitability criteria (4 points).

        1. Positive ROA (net income > 0)
        2. Positive CFO
        3. ROA improving (current > prior)
        4. Quality earnings (CFO > net income)
        """
        score = 0
        criteria = {}
        components = {}

        # 1. Positive ROA
        has_positive_roa = False
        if data.net_income is not None and data.total_assets and data.total_assets > 0:
            roa = data.net_income / data.total_assets
            components["current_roa"] = roa
            has_positive_roa = data.net_income > 0
            if has_positive_roa:
                score += 1
        criteria["F1_positive_roa"] = has_positive_roa

        # 2. Positive CFO
        has_positive_cfo = False
        if data.operating_cash_flow is not None:
            has_positive_cfo = data.operating_cash_flow > 0
            components["current_cfo"] = data.operating_cash_flow
            if has_positive_cfo:
                score += 1
        criteria["F2_positive_cfo"] = has_positive_cfo

        # 3. ROA improving (requires prior period)
        roa_improving = False
        if data.prior_period and data.prior_period.net_income is not None:
            if data.prior_period.total_assets and data.prior_period.total_assets > 0:
                prior_roa = data.prior_period.net_income / data.prior_period.total_assets
                components["prior_roa"] = prior_roa
                if "current_roa" in components:
                    roa_improving = components["current_roa"] > prior_roa
                    components["roa_change"] = components["current_roa"] - prior_roa
                    if roa_improving:
                        score += 1
        criteria["F3_roa_improving"] = roa_improving

        # 4. Quality of earnings (CFO > Net Income, i.e., positive accruals quality)
        quality_earnings = False
        if data.operating_cash_flow is not None and data.net_income is not None:
            quality_earnings = data.operating_cash_flow > data.net_income
            components["accruals"] = data.net_income - data.operating_cash_flow
            if quality_earnings:
                score += 1
        criteria["F4_quality_earnings"] = quality_earnings

        return score, criteria, components

    def _calculate_leverage(self, data: FinancialData) -> tuple[int, Dict[str, bool], Dict[str, Any]]:
        """Calculate leverage and liquidity criteria (3 points).

        5. Leverage decreasing (LTD/Assets lower)
        6. Current ratio improving
        7. No equity dilution (shares not increased)
        """
        score = 0
        criteria = {}
        components = {}

        # 5. Leverage decreasing
        leverage_decreasing = False
        if data.long_term_debt is not None and data.total_assets and data.total_assets > 0:
            current_leverage = data.long_term_debt / data.total_assets
            components["current_leverage"] = current_leverage

            if data.prior_period:
                if data.prior_period.long_term_debt is not None and data.prior_period.total_assets:
                    if data.prior_period.total_assets > 0:
                        prior_leverage = data.prior_period.long_term_debt / data.prior_period.total_assets
                        components["prior_leverage"] = prior_leverage
                        leverage_decreasing = current_leverage < prior_leverage
                        components["leverage_change"] = current_leverage - prior_leverage
                        if leverage_decreasing:
                            score += 1
        criteria["F5_leverage_decreasing"] = leverage_decreasing

        # 6. Current ratio improving
        current_ratio_improving = False
        if data.current_assets is not None and data.current_liabilities:
            if data.current_liabilities > 0:
                current_ratio = data.current_assets / data.current_liabilities
                components["current_ratio"] = current_ratio

                if data.prior_period:
                    if data.prior_period.current_assets is not None and data.prior_period.current_liabilities:
                        if data.prior_period.current_liabilities > 0:
                            prior_ratio = data.prior_period.current_assets / data.prior_period.current_liabilities
                            components["prior_current_ratio"] = prior_ratio
                            current_ratio_improving = current_ratio > prior_ratio
                            components["current_ratio_change"] = current_ratio - prior_ratio
                            if current_ratio_improving:
                                score += 1
        criteria["F6_current_ratio_improving"] = current_ratio_improving

        # 7. No equity dilution (shares not increased)
        no_dilution = False
        if data.shares_outstanding is not None and data.prior_period:
            if data.prior_period.shares_outstanding is not None:
                no_dilution = data.shares_outstanding <= data.prior_period.shares_outstanding
                components["current_shares"] = data.shares_outstanding
                components["prior_shares"] = data.prior_period.shares_outstanding
                components["share_change"] = data.shares_outstanding - data.prior_period.shares_outstanding
                if no_dilution:
                    score += 1
        criteria["F7_no_dilution"] = no_dilution

        return score, criteria, components

    def _calculate_efficiency(self, data: FinancialData) -> tuple[int, Dict[str, bool], Dict[str, Any]]:
        """Calculate operating efficiency criteria (2 points).

        8. Gross margin improving
        9. Asset turnover improving
        """
        score = 0
        criteria = {}
        components = {}

        # 8. Gross margin improving
        margin_improving = False
        if data.gross_profit is not None and data.revenue and data.revenue > 0:
            current_margin = data.gross_profit / data.revenue
            components["current_gross_margin"] = current_margin

            if data.prior_period:
                if data.prior_period.gross_profit is not None and data.prior_period.revenue:
                    if data.prior_period.revenue > 0:
                        prior_margin = data.prior_period.gross_profit / data.prior_period.revenue
                        components["prior_gross_margin"] = prior_margin
                        margin_improving = current_margin > prior_margin
                        components["gross_margin_change"] = current_margin - prior_margin
                        if margin_improving:
                            score += 1
        criteria["F8_gross_margin_improving"] = margin_improving

        # 9. Asset turnover improving
        turnover_improving = False
        if data.revenue is not None and data.total_assets and data.total_assets > 0:
            current_turnover = data.revenue / data.total_assets
            components["current_asset_turnover"] = current_turnover

            if data.prior_period:
                if data.prior_period.revenue is not None and data.prior_period.total_assets:
                    if data.prior_period.total_assets > 0:
                        prior_turnover = data.prior_period.revenue / data.prior_period.total_assets
                        components["prior_asset_turnover"] = prior_turnover
                        turnover_improving = current_turnover > prior_turnover
                        components["asset_turnover_change"] = current_turnover - prior_turnover
                        if turnover_improving:
                            score += 1
        criteria["F9_asset_turnover_improving"] = turnover_improving

        return score, criteria, components

    def _classify_strength(self, score: int) -> FinancialStrength:
        """Classify F-Score into strength levels."""
        if score >= self.STRONG_THRESHOLD:
            return FinancialStrength.STRONG
        elif score < self.WEAK_THRESHOLD:
            return FinancialStrength.WEAK
        else:
            return FinancialStrength.MODERATE

    def _get_interpretation(self, strength: FinancialStrength, score: int, criteria: Dict[str, bool]) -> str:
        """Generate human-readable interpretation."""
        passed = sum(1 for v in criteria.values() if v)
        failed = len(criteria) - passed

        # Count by category
        prof_passed = sum(1 for k, v in criteria.items() if k.startswith("F") and int(k[1]) <= 4 and v)
        lev_passed = sum(1 for k, v in criteria.items() if k.startswith("F") and 5 <= int(k[1]) <= 7 and v)
        eff_passed = sum(1 for k, v in criteria.items() if k.startswith("F") and int(k[1]) >= 8 and v)

        base_msg = f"F-Score {score}/9: "

        if strength == FinancialStrength.STRONG:
            return (
                f"{base_msg}Strong financial position. "
                f"Profitability {prof_passed}/4, Leverage {lev_passed}/3, Efficiency {eff_passed}/2. "
                "Company demonstrates robust fundamentals across most criteria."
            )
        elif strength == FinancialStrength.WEAK:
            return (
                f"{base_msg}Weak financial position. "
                f"Profitability {prof_passed}/4, Leverage {lev_passed}/3, Efficiency {eff_passed}/2. "
                "Company shows financial weakness in multiple areas requiring attention."
            )
        else:
            return (
                f"{base_msg}Moderate financial position. "
                f"Profitability {prof_passed}/4, Leverage {lev_passed}/3, Efficiency {eff_passed}/2. "
                "Mixed signals - some strengths but also areas of concern."
            )
