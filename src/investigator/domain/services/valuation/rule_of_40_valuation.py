"""
Rule of 40 Valuation - Growth + Profitability balanced valuation for SaaS.

The Rule of 40 states that a healthy SaaS company should have:
Revenue Growth % + FCF Margin % >= 40%

This model uses the Rule of 40 score to adjust P/S multiples:
- Score >= 60: Premium P/S (1.5x median)
- Score >= 40: Fair P/S (1.0x median)
- Score >= 30: Discount P/S (0.8x median)
- Score < 30: Deep discount P/S (0.6x median)

Usage:
    from investigator.domain.services.valuation.rule_of_40_valuation import RuleOf40Valuation

    valuation = RuleOf40Valuation(company_profile)
    result = valuation.calculate(
        revenue_growth=0.25,
        fcf_margin=0.18,
        current_revenue=500e6,
        shares_outstanding=100e6
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from investigator.domain.services.valuation.models.base import (
    BaseValuationModel,
    ModelDiagnostics,
    ModelNotApplicable,
    ValuationModelResult,
    ValuationOutput,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile

logger = logging.getLogger(__name__)


@dataclass
class Rule40Benchmarks:
    """Industry benchmarks for Rule of 40 scoring."""

    median_score: float
    top_quartile_score: float
    median_ps_multiple: float
    top_quartile_ps_multiple: float


class RuleOf40Valuation(BaseValuationModel):
    """
    Rule of 40 based valuation for SaaS/software companies.

    The Rule of 40 balances growth and profitability:
    Score = Revenue Growth % + FCF Margin %

    Higher scores justify higher P/S multiples.
    """

    model_name = "rule_of_40"
    methodology = "Rule of 40 P/S Multiple Adjustment"

    # Industry benchmarks
    INDUSTRY_BENCHMARKS = {
        "SaaS - Enterprise": Rule40Benchmarks(
            median_score=35.0, top_quartile_score=55.0, median_ps_multiple=8.0, top_quartile_ps_multiple=15.0
        ),
        "SaaS - SMB": Rule40Benchmarks(
            median_score=30.0, top_quartile_score=45.0, median_ps_multiple=6.0, top_quartile_ps_multiple=12.0
        ),
        "Software - Application": Rule40Benchmarks(
            median_score=32.0, top_quartile_score=50.0, median_ps_multiple=7.0, top_quartile_ps_multiple=14.0
        ),
        "Software - Infrastructure": Rule40Benchmarks(
            median_score=35.0, top_quartile_score=55.0, median_ps_multiple=9.0, top_quartile_ps_multiple=18.0
        ),
        "Internet Software/Services": Rule40Benchmarks(
            median_score=30.0, top_quartile_score=50.0, median_ps_multiple=6.0, top_quartile_ps_multiple=12.0
        ),
        "default": Rule40Benchmarks(
            median_score=30.0, top_quartile_score=45.0, median_ps_multiple=5.0, top_quartile_ps_multiple=10.0
        ),
    }

    # Score thresholds for multiple adjustments
    SCORE_THRESHOLDS = {
        "exceptional": 60,  # 1.5x median multiple
        "strong": 45,  # 1.2x median multiple
        "healthy": 40,  # 1.0x median multiple
        "moderate": 30,  # 0.8x median multiple
        "weak": 20,  # 0.6x median multiple
    }

    def calculate(
        self,
        revenue_growth: Optional[float] = None,
        fcf_margin: Optional[float] = None,
        current_revenue: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        current_price: Optional[float] = None,
        **kwargs: Any,
    ) -> ValuationOutput:
        """
        Calculate fair value using Rule of 40.

        Args:
            revenue_growth: YoY revenue growth rate (as decimal, e.g., 0.25 for 25%)
            fcf_margin: Free cash flow margin (as decimal, e.g., 0.15 for 15%)
            current_revenue: TTM revenue
            shares_outstanding: Shares outstanding
            current_price: Optional current stock price for context

        Returns:
            ValuationModelResult with fair value or ModelNotApplicable
        """
        # Validate required inputs
        if revenue_growth is None:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Revenue growth rate not available",
                diagnostics=ModelDiagnostics(flags=["missing_revenue_growth"]),
            )

        if fcf_margin is None:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="FCF margin not available",
                diagnostics=ModelDiagnostics(flags=["missing_fcf_margin"]),
            )

        if current_revenue is None or current_revenue <= 0:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Current revenue not available or invalid",
                diagnostics=ModelDiagnostics(flags=["missing_revenue"]),
            )

        if shares_outstanding is None or shares_outstanding <= 0:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Shares outstanding not available",
                diagnostics=ModelDiagnostics(flags=["missing_shares"]),
            )

        # Calculate Rule of 40 score
        # Convert decimals to percentages for the calculation
        revenue_growth_pct = revenue_growth * 100 if abs(revenue_growth) < 5 else revenue_growth
        fcf_margin_pct = fcf_margin * 100 if abs(fcf_margin) < 5 else fcf_margin

        rule_40_score = revenue_growth_pct + fcf_margin_pct

        # Get industry benchmarks
        industry = self.company_profile.industry or "default"
        benchmarks = self._get_benchmarks(industry)

        # Determine P/S multiple based on score
        ps_multiple, score_classification = self._get_ps_multiple(rule_40_score, benchmarks)

        # Calculate fair value
        fair_market_cap = current_revenue * ps_multiple
        fair_value = fair_market_cap / shares_outstanding

        # Calculate upside/downside if current price available
        upside_potential = None
        if current_price and current_price > 0:
            upside_potential = (fair_value / current_price - 1) * 100

        # Estimate confidence
        confidence = self._calculate_confidence(rule_40_score, score_classification, revenue_growth_pct)

        # Build assumptions
        assumptions = {
            "revenue_growth_pct": revenue_growth_pct,
            "fcf_margin_pct": fcf_margin_pct,
            "rule_40_score": rule_40_score,
            "score_classification": score_classification,
            "ps_multiple_applied": ps_multiple,
            "benchmark_median_score": benchmarks.median_score,
            "benchmark_median_ps": benchmarks.median_ps_multiple,
        }

        # Build metadata
        metadata = {
            "fair_market_cap": fair_market_cap,
            "current_revenue": current_revenue,
            "shares_outstanding": shares_outstanding,
        }

        if upside_potential is not None:
            metadata["upside_potential_pct"] = upside_potential
            metadata["current_price"] = current_price

        logger.info(
            f"[{self.company_profile.symbol}] Rule of 40: Score={rule_40_score:.1f} "
            f"({score_classification}), P/S={ps_multiple:.1f}x, "
            f"Fair Value=${fair_value:.2f}"
        )

        return ValuationModelResult(
            model_name=self.model_name,
            fair_value=fair_value,
            confidence_score=confidence,
            methodology=self.methodology,
            assumptions=assumptions,
            diagnostics=ModelDiagnostics(
                data_quality_score=0.85 if rule_40_score > 0 else 0.6, flags=[f"rule40_score_{score_classification}"]
            ),
            metadata=metadata,
        )

    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        """Estimate confidence for the Rule of 40 model."""
        score = raw_output.get("rule_40_score", 0)
        classification = raw_output.get("score_classification", "unknown")
        growth = raw_output.get("revenue_growth_pct", 0)

        return self._calculate_confidence(score, classification, growth)

    def _get_benchmarks(self, industry: str) -> Rule40Benchmarks:
        """Get benchmarks for industry."""
        # Try exact match
        if industry in self.INDUSTRY_BENCHMARKS:
            return self.INDUSTRY_BENCHMARKS[industry]

        # Try partial match
        industry_lower = industry.lower()
        for key, benchmarks in self.INDUSTRY_BENCHMARKS.items():
            if key != "default" and (industry_lower in key.lower() or key.lower() in industry_lower):
                return benchmarks

        return self.INDUSTRY_BENCHMARKS["default"]

    def _get_ps_multiple(self, rule_40_score: float, benchmarks: Rule40Benchmarks) -> tuple:
        """
        Determine P/S multiple based on Rule of 40 score.

        Returns:
            Tuple of (ps_multiple, score_classification)
        """
        median_ps = benchmarks.median_ps_multiple
        top_ps = benchmarks.top_quartile_ps_multiple

        if rule_40_score >= self.SCORE_THRESHOLDS["exceptional"]:
            # Exceptional: Use top quartile multiple
            return (top_ps, "exceptional")

        elif rule_40_score >= self.SCORE_THRESHOLDS["strong"]:
            # Strong: Blend between median and top quartile
            blend = (rule_40_score - 45) / 15  # 0 to 1
            multiple = median_ps + blend * (top_ps - median_ps)
            return (multiple, "strong")

        elif rule_40_score >= self.SCORE_THRESHOLDS["healthy"]:
            # Healthy: Use median multiple
            return (median_ps, "healthy")

        elif rule_40_score >= self.SCORE_THRESHOLDS["moderate"]:
            # Moderate: Discount from median
            discount = (40 - rule_40_score) / 10  # 0 to 1
            multiple = median_ps * (1 - discount * 0.2)
            return (multiple, "moderate")

        elif rule_40_score >= self.SCORE_THRESHOLDS["weak"]:
            # Weak: Deeper discount
            multiple = median_ps * 0.6
            return (multiple, "weak")

        else:
            # Very weak: Significant discount
            multiple = median_ps * 0.4
            return (multiple, "distressed")

    def _calculate_confidence(self, score: float, classification: str, revenue_growth: float) -> float:
        """Calculate confidence score for the valuation."""
        base_confidence = 0.70

        # Higher confidence for scores near thresholds
        if 35 <= score <= 50:
            base_confidence += 0.10  # Most reliable range

        # Lower confidence for extreme values
        if score > 80 or score < 10:
            base_confidence -= 0.15

        # Adjust for growth rate reasonableness
        if 10 <= revenue_growth <= 50:
            base_confidence += 0.05
        elif revenue_growth > 100 or revenue_growth < -20:
            base_confidence -= 0.10

        return max(0.30, min(0.90, base_confidence))


def calculate_rule_of_40_score(revenue_growth: float, fcf_margin: float) -> float:
    """
    Calculate Rule of 40 score.

    Args:
        revenue_growth: Revenue growth rate (as decimal or percentage)
        fcf_margin: FCF margin (as decimal or percentage)

    Returns:
        Rule of 40 score (growth % + margin %)
    """
    # Normalize to percentage
    growth_pct = revenue_growth * 100 if abs(revenue_growth) < 5 else revenue_growth
    margin_pct = fcf_margin * 100 if abs(fcf_margin) < 5 else fcf_margin

    return growth_pct + margin_pct
