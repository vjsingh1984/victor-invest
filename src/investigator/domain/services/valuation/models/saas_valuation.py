"""
SaaS Valuation Model - Comprehensive SaaS/software company valuation.

Incorporates SaaS-specific metrics:
- Net Revenue Retention (NRR)
- LTV/CAC ratio
- Gross Margin
- Rule of 40
- Growth efficiency

These metrics adjust the base P/S multiple to arrive at fair value.

Usage:
    from investigator.domain.services.valuation.models.saas_valuation import SaaSValuationModel

    valuation = SaaSValuationModel(company_profile)
    result = valuation.calculate(
        current_revenue=500e6,
        revenue_growth=0.25,
        gross_margin=0.75,
        nrr=1.15,
        ltv_cac=4.0,
        fcf_margin=0.18,
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
class SaaSMetrics:
    """Container for SaaS-specific metrics."""

    nrr: Optional[float] = None  # Net Revenue Retention (e.g., 1.15 for 115%)
    ltv_cac: Optional[float] = None  # LTV/CAC ratio
    gross_margin: Optional[float] = None  # Gross margin (as decimal)
    cac_payback_months: Optional[float] = None  # CAC payback in months
    magic_number: Optional[float] = None  # Net new ARR / S&M spend
    burn_multiple: Optional[float] = None  # Net burn / Net new ARR


class SaaSValuationModel(BaseValuationModel):
    """
    SaaS-specific valuation model.

    Adjusts base P/S multiple using SaaS efficiency metrics:
    - NRR: +/-30% adjustment (-30% to +50%)
    - LTV/CAC: +/-20% adjustment (-40% to +30%)
    - Rule of 40: +/-15% adjustment (-20% to +20%)
    - Gross Margin: +/-10% adjustment (-10% to +10%)

    Base P/S multiple is determined by growth rate and market conditions.
    """

    model_name = "saas_multiples"
    methodology = "SaaS Efficiency-Adjusted P/S Multiple"

    # Base P/S multiples by growth rate tier
    BASE_PS_MULTIPLES = {
        "hyper_growth": 15.0,  # >50% growth
        "high_growth": 10.0,  # 30-50% growth
        "growth": 7.0,  # 15-30% growth
        "moderate": 5.0,  # 5-15% growth
        "low": 3.0,  # <5% growth
    }

    # Adjustment ranges for each metric
    METRIC_ADJUSTMENTS = {
        "nrr": {
            "min_value": 0.80,  # 80% retention
            "max_value": 1.40,  # 140% retention
            "min_adjustment": -0.30,
            "max_adjustment": 0.50,
            "benchmark": 1.10,  # 110% is benchmark
        },
        "ltv_cac": {
            "min_value": 1.0,
            "max_value": 7.0,
            "min_adjustment": -0.40,
            "max_adjustment": 0.30,
            "benchmark": 3.0,
        },
        "rule_of_40": {
            "min_value": 0,
            "max_value": 70,
            "min_adjustment": -0.20,
            "max_adjustment": 0.20,
            "benchmark": 40,
        },
        "gross_margin": {
            "min_value": 0.50,
            "max_value": 0.90,
            "min_adjustment": -0.10,
            "max_adjustment": 0.10,
            "benchmark": 0.75,
        },
    }

    def calculate(
        self,
        current_revenue: Optional[float] = None,
        revenue_growth: Optional[float] = None,
        gross_margin: Optional[float] = None,
        fcf_margin: Optional[float] = None,
        nrr: Optional[float] = None,
        ltv_cac: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        current_price: Optional[float] = None,
        saas_metrics: Optional[SaaSMetrics] = None,
        **kwargs: Any,
    ) -> ValuationOutput:
        """
        Calculate fair value using SaaS metrics.

        Args:
            current_revenue: TTM revenue
            revenue_growth: YoY revenue growth (as decimal)
            gross_margin: Gross margin (as decimal)
            fcf_margin: FCF margin (as decimal)
            nrr: Net revenue retention (as ratio, e.g., 1.15)
            ltv_cac: LTV/CAC ratio
            shares_outstanding: Shares outstanding
            current_price: Optional current price
            saas_metrics: Optional SaaSMetrics container

        Returns:
            ValuationModelResult or ModelNotApplicable
        """
        # Extract metrics from container if provided
        if saas_metrics:
            nrr = nrr or saas_metrics.nrr
            ltv_cac = ltv_cac or saas_metrics.ltv_cac
            gross_margin = gross_margin or saas_metrics.gross_margin

        # Validate required inputs
        if current_revenue is None or current_revenue <= 0:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Revenue not available",
                diagnostics=ModelDiagnostics(flags=["missing_revenue"]),
            )

        if revenue_growth is None:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Revenue growth not available",
                diagnostics=ModelDiagnostics(flags=["missing_growth"]),
            )

        if shares_outstanding is None or shares_outstanding <= 0:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Shares outstanding not available",
                diagnostics=ModelDiagnostics(flags=["missing_shares"]),
            )

        # Normalize growth rate to percentage
        growth_pct = revenue_growth * 100 if abs(revenue_growth) < 5 else revenue_growth

        # Get base P/S multiple from growth tier
        base_ps, growth_tier = self._get_base_ps(growth_pct)

        # Calculate adjustments
        adjustments = {}
        total_adjustment = 0

        # NRR adjustment
        if nrr is not None:
            adj = self._calculate_adjustment("nrr", nrr)
            adjustments["nrr"] = {
                "value": nrr,
                "adjustment": adj,
                "benchmark": self.METRIC_ADJUSTMENTS["nrr"]["benchmark"],
            }
            total_adjustment += adj

        # LTV/CAC adjustment
        if ltv_cac is not None:
            adj = self._calculate_adjustment("ltv_cac", ltv_cac)
            adjustments["ltv_cac"] = {
                "value": ltv_cac,
                "adjustment": adj,
                "benchmark": self.METRIC_ADJUSTMENTS["ltv_cac"]["benchmark"],
            }
            total_adjustment += adj

        # Rule of 40 adjustment
        if revenue_growth is not None and fcf_margin is not None:
            growth_for_r40 = revenue_growth * 100 if abs(revenue_growth) < 5 else revenue_growth
            margin_for_r40 = fcf_margin * 100 if abs(fcf_margin) < 5 else fcf_margin
            rule_40_score = growth_for_r40 + margin_for_r40

            adj = self._calculate_adjustment("rule_of_40", rule_40_score)
            adjustments["rule_of_40"] = {
                "value": rule_40_score,
                "adjustment": adj,
                "benchmark": self.METRIC_ADJUSTMENTS["rule_of_40"]["benchmark"],
            }
            total_adjustment += adj

        # Gross margin adjustment
        if gross_margin is not None:
            # Normalize to decimal if percentage
            gm = gross_margin / 100 if gross_margin > 1 else gross_margin
            adj = self._calculate_adjustment("gross_margin", gm)
            adjustments["gross_margin"] = {
                "value": gm,
                "adjustment": adj,
                "benchmark": self.METRIC_ADJUSTMENTS["gross_margin"]["benchmark"],
            }
            total_adjustment += adj

        # Cap total adjustment at -50% to +100%
        total_adjustment = max(-0.50, min(1.00, total_adjustment))

        # Apply adjustments to base P/S
        adjusted_ps = base_ps * (1 + total_adjustment)

        # Calculate fair value
        fair_market_cap = current_revenue * adjusted_ps
        fair_value = fair_market_cap / shares_outstanding

        # Calculate upside/downside
        upside_potential = None
        if current_price and current_price > 0:
            upside_potential = (fair_value / current_price - 1) * 100

        # Estimate confidence
        metrics_available = sum(1 for v in [nrr, ltv_cac, gross_margin, fcf_margin] if v is not None)
        confidence = self._calculate_confidence(metrics_available, total_adjustment, growth_pct)

        # Build assumptions
        assumptions = {
            "revenue_growth_pct": growth_pct,
            "growth_tier": growth_tier,
            "base_ps_multiple": base_ps,
            "total_adjustment": total_adjustment,
            "adjusted_ps_multiple": adjusted_ps,
            "metrics_available": metrics_available,
            "adjustments": adjustments,
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
            f"[{self.company_profile.symbol}] SaaS Valuation: "
            f"Base P/S={base_ps:.1f}x ({growth_tier}), "
            f"Adj={total_adjustment:+.0%}, Final P/S={adjusted_ps:.1f}x, "
            f"Fair Value=${fair_value:.2f}"
        )

        return ValuationModelResult(
            model_name=self.model_name,
            fair_value=fair_value,
            confidence_score=confidence,
            methodology=self.methodology,
            assumptions=assumptions,
            diagnostics=ModelDiagnostics(
                data_quality_score=0.50 + 0.125 * metrics_available, flags=[f"saas_tier_{growth_tier}"]
            ),
            metadata=metadata,
        )

    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        """Estimate confidence for SaaS model."""
        metrics = raw_output.get("metrics_available", 0)
        adj = raw_output.get("total_adjustment", 0)
        growth = raw_output.get("revenue_growth_pct", 0)

        return self._calculate_confidence(metrics, adj, growth)

    def _get_base_ps(self, growth_pct: float) -> tuple:
        """Get base P/S multiple from growth rate."""
        if growth_pct > 50:
            return (self.BASE_PS_MULTIPLES["hyper_growth"], "hyper_growth")
        elif growth_pct > 30:
            return (self.BASE_PS_MULTIPLES["high_growth"], "high_growth")
        elif growth_pct > 15:
            return (self.BASE_PS_MULTIPLES["growth"], "growth")
        elif growth_pct > 5:
            return (self.BASE_PS_MULTIPLES["moderate"], "moderate")
        else:
            return (self.BASE_PS_MULTIPLES["low"], "low")

    def _calculate_adjustment(self, metric: str, value: float) -> float:
        """Calculate adjustment for a metric based on its range."""
        if metric not in self.METRIC_ADJUSTMENTS:
            return 0

        config = self.METRIC_ADJUSTMENTS[metric]
        min_val = config["min_value"]
        max_val = config["max_value"]
        min_adj = config["min_adjustment"]
        max_adj = config["max_adjustment"]
        benchmark = config["benchmark"]

        # Clamp value to range
        value = max(min_val, min(max_val, value))

        # Linear interpolation from min to max
        if value < benchmark:
            # Below benchmark: interpolate from min_adj to 0
            ratio = (value - min_val) / (benchmark - min_val) if benchmark > min_val else 0
            return min_adj * (1 - ratio)
        else:
            # Above benchmark: interpolate from 0 to max_adj
            ratio = (value - benchmark) / (max_val - benchmark) if max_val > benchmark else 0
            return max_adj * ratio

    def _calculate_confidence(self, metrics_available: int, total_adjustment: float, growth_pct: float) -> float:
        """Calculate confidence score."""
        # Base confidence from metrics availability
        base = 0.50 + 0.10 * metrics_available

        # Reduce for extreme adjustments
        if abs(total_adjustment) > 0.50:
            base -= 0.10

        # Reduce for extreme growth (less reliable)
        if growth_pct > 100 or growth_pct < -20:
            base -= 0.10

        return max(0.30, min(0.90, base))
