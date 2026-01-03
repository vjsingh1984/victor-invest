"""
Valuation Adjustment Utilities
Smart price target calculations with both upward and downward revisions
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValuationBias(Enum):
    """Direction of valuation bias"""

    OVERVALUED = "overvalued"
    UNDERVALUED = "undervalued"
    FAIRLY_VALUED = "fairly_valued"


class QualityTier(Enum):
    """Company quality tiers"""

    EXCEPTIONAL = "exceptional"  # 90-100 quality score
    HIGH = "high"  # 80-89 quality score
    AVERAGE = "average"  # 60-79 quality score
    BELOW_AVERAGE = "below_average"  # 40-59 quality score
    POOR = "poor"  # <40 quality score


@dataclass
class ValuationMetrics:
    """Container for valuation metrics"""

    current_price: float
    fair_value: float
    technical_target: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    quality_score: Optional[float] = None
    dividend_yield: Optional[float] = None

    @property
    def discount_to_fair_value(self) -> float:
        """Calculate discount/premium to fair value"""
        if not self.fair_value or self.fair_value <= 0:
            return 0.0
        return (self.fair_value - self.current_price) / self.current_price

    @property
    def valuation_bias(self) -> ValuationBias:
        """Determine valuation bias"""
        discount = self.discount_to_fair_value
        if discount > 0.10:  # More than 10% undervalued
            return ValuationBias.UNDERVALUED
        elif discount < -0.10:  # More than 10% overvalued
            return ValuationBias.OVERVALUED
        else:
            return ValuationBias.FAIRLY_VALUED

    @property
    def quality_tier(self) -> QualityTier:
        """Determine quality tier"""
        if not self.quality_score:
            return QualityTier.AVERAGE

        if self.quality_score >= 90:
            return QualityTier.EXCEPTIONAL
        elif self.quality_score >= 80:
            return QualityTier.HIGH
        elif self.quality_score >= 60:
            return QualityTier.AVERAGE
        elif self.quality_score >= 40:
            return QualityTier.BELOW_AVERAGE
        else:
            return QualityTier.POOR


class SmartValuationAdjuster:
    """
    Smart valuation adjuster that applies both upward and downward revisions
    based on comprehensive analysis of fundamentals, technical, and market conditions
    """

    def __init__(self):
        # Quality-based valuation multiples
        self.quality_multipliers = {
            QualityTier.EXCEPTIONAL: 1.15,  # 15% premium for exceptional quality
            QualityTier.HIGH: 1.05,  # 5% premium for high quality
            QualityTier.AVERAGE: 1.00,  # No adjustment for average quality
            QualityTier.BELOW_AVERAGE: 0.90,  # 10% discount for below average
            QualityTier.POOR: 0.75,  # 25% discount for poor quality
        }

        # Risk-based adjustments
        self.risk_adjustments = {
            "very_low": 1.10,  # 10% premium for very low risk
            "low": 1.05,  # 5% premium for low risk
            "medium": 1.00,  # No adjustment for medium risk
            "high": 0.90,  # 10% discount for high risk
            "very_high": 0.75,  # 25% discount for very high risk
        }

        # Technical trend adjustments
        self.trend_adjustments = {
            "strong_uptrend": 1.08,  # 8% premium for strong uptrend
            "uptrend": 1.04,  # 4% premium for uptrend
            "sideways": 1.00,  # No adjustment for sideways
            "downtrend": 0.95,  # 5% discount for downtrend
            "strong_downtrend": 0.85,  # 15% discount for strong downtrend
        }

        # Market sentiment adjustments
        self.sentiment_adjustments = {
            "very_bullish": 1.06,  # 6% premium for very bullish sentiment
            "bullish": 1.03,  # 3% premium for bullish sentiment
            "neutral": 1.00,  # No adjustment for neutral sentiment
            "bearish": 0.95,  # 5% discount for bearish sentiment
            "very_bearish": 0.88,  # 12% discount for very bearish sentiment
        }

        # Sector-specific adjustments
        self.sector_adjustments = {
            "technology": 1.10,  # Tech premium
            "healthcare": 1.05,  # Healthcare premium
            "utilities": 0.95,  # Utilities discount
            "energy": 0.90,  # Energy discount (volatility)
            "financials": 0.95,  # Financials discount (regulation)
            "default": 1.00,  # Default for other sectors
        }

    def calculate_adjusted_target(
        self, metrics: ValuationMetrics, analysis_context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate intelligently adjusted price target

        Args:
            metrics: Valuation metrics
            analysis_context: Context from technical, fundamental, and other analyses

        Returns:
            Tuple of (adjusted_target, adjustment_details)
        """
        # Start with weighted average of fair value and technical target
        base_target = self._calculate_base_target(metrics)

        # Apply systematic adjustments
        adjustment_details = {}
        adjusted_target = base_target

        # 1. Quality-based adjustment
        quality_mult = self.quality_multipliers.get(metrics.quality_tier, 1.0)
        adjusted_target *= quality_mult
        adjustment_details["quality"] = {
            "tier": metrics.quality_tier.value,
            "multiplier": quality_mult,
            "rationale": self._get_quality_rationale(metrics.quality_tier),
        }

        # 2. Valuation bias adjustment (KEY: This handles overvaluation!)
        bias_mult = self._calculate_valuation_bias_adjustment(metrics)
        adjusted_target *= bias_mult
        adjustment_details["valuation_bias"] = {
            "bias": metrics.valuation_bias.value,
            "discount_to_fair": metrics.discount_to_fair_value,
            "multiplier": bias_mult,
            "rationale": self._get_bias_rationale(metrics.valuation_bias, metrics.discount_to_fair_value),
        }

        # 3. Risk-based adjustment
        risk_level = analysis_context.get("risk_level", "medium")
        risk_mult = self.risk_adjustments.get(risk_level, 1.0)
        adjusted_target *= risk_mult
        adjustment_details["risk"] = {
            "level": risk_level,
            "multiplier": risk_mult,
            "rationale": f"Risk level: {risk_level}",
        }

        # 4. Technical trend adjustment
        trend = analysis_context.get("technical_trend", "sideways")
        trend_mult = self.trend_adjustments.get(trend, 1.0)
        adjusted_target *= trend_mult
        adjustment_details["trend"] = {
            "direction": trend,
            "multiplier": trend_mult,
            "rationale": f"Technical trend: {trend}",
        }

        # 5. Market sentiment adjustment
        sentiment = analysis_context.get("market_sentiment", "neutral")
        sentiment_mult = self.sentiment_adjustments.get(sentiment, 1.0)
        adjusted_target *= sentiment_mult
        adjustment_details["sentiment"] = {
            "sentiment": sentiment,
            "multiplier": sentiment_mult,
            "rationale": f"Market sentiment: {sentiment}",
        }

        # 6. Sector adjustment
        sector = analysis_context.get("sector", "default").lower()
        sector_mult = self.sector_adjustments.get(sector, self.sector_adjustments["default"])
        adjusted_target *= sector_mult
        adjustment_details["sector"] = {"sector": sector, "multiplier": sector_mult, "rationale": f"Sector: {sector}"}

        # 7. Extreme valuation protection
        adjusted_target = self._apply_extreme_valuation_protection(adjusted_target, metrics, adjustment_details)

        # Calculate overall adjustment
        overall_adjustment = (adjusted_target - base_target) / base_target if base_target > 0 else 0
        adjustment_details["summary"] = {
            "base_target": base_target,
            "adjusted_target": adjusted_target,
            "overall_adjustment": overall_adjustment,
            "adjustment_direction": (
                "upward" if overall_adjustment > 0 else "downward" if overall_adjustment < 0 else "neutral"
            ),
        }

        logger.info(
            f"Target price adjustment: {base_target:.2f} -> {adjusted_target:.2f} "
            f"({overall_adjustment:+.1%}) - {adjustment_details['summary']['adjustment_direction']}"
        )

        return adjusted_target, adjustment_details

    def _calculate_base_target(self, metrics: ValuationMetrics) -> float:
        """Calculate base target from fair value and technical target"""
        # Weight fair value higher if we have good fundamental data
        if metrics.fair_value and metrics.technical_target:
            # 70% fundamental, 30% technical for base calculation
            return metrics.fair_value * 0.7 + metrics.technical_target * 0.3
        elif metrics.fair_value:
            return metrics.fair_value
        elif metrics.technical_target:
            return metrics.technical_target
        else:
            return metrics.current_price

    def _calculate_valuation_bias_adjustment(self, metrics: ValuationMetrics) -> float:
        """
        Calculate adjustment based on valuation bias
        THIS IS WHERE WE HANDLE OVERVALUATION DISCOUNTS
        """
        bias = metrics.valuation_bias
        discount = metrics.discount_to_fair_value

        if bias == ValuationBias.OVERVALUED:
            # Apply progressive discounts for overvaluation
            if discount < -0.50:  # More than 50% overvalued
                return 0.70  # 30% discount
            elif discount < -0.30:  # 30-50% overvalued
                return 0.80  # 20% discount
            elif discount < -0.20:  # 20-30% overvalued
                return 0.90  # 10% discount
            else:  # 10-20% overvalued
                return 0.95  # 5% discount

        elif bias == ValuationBias.UNDERVALUED:
            # Apply progressive premiums for undervaluation
            if discount > 0.50:  # More than 50% undervalued
                return 1.20  # 20% premium (capped to avoid excessive targets)
            elif discount > 0.30:  # 30-50% undervalued
                return 1.15  # 15% premium
            elif discount > 0.20:  # 20-30% undervalued
                return 1.10  # 10% premium
            else:  # 10-20% undervalued
                return 1.05  # 5% premium

        else:  # Fairly valued
            return 1.00  # No adjustment

    def _apply_extreme_valuation_protection(
        self, adjusted_target: float, metrics: ValuationMetrics, adjustment_details: Dict
    ) -> float:
        """Apply protection against extreme valuations"""
        current_price = metrics.current_price

        # Maximum allowed adjustment from current price
        max_upward = current_price * 1.50  # 50% maximum upward
        max_downward = current_price * 0.60  # 40% maximum downward

        original_target = adjusted_target

        if adjusted_target > max_upward:
            adjusted_target = max_upward
            adjustment_details["extreme_protection"] = {
                "applied": True,
                "type": "upward_cap",
                "original_target": original_target,
                "capped_target": adjusted_target,
                "rationale": "Capped at 50% above current price to prevent extreme targets",
            }
        elif adjusted_target < max_downward:
            adjusted_target = max_downward
            adjustment_details["extreme_protection"] = {
                "applied": True,
                "type": "downward_cap",
                "original_target": original_target,
                "capped_target": adjusted_target,
                "rationale": "Floored at 40% below current price to prevent extreme pessimism",
            }
        else:
            adjustment_details["extreme_protection"] = {"applied": False, "rationale": "No extreme protection needed"}

        return adjusted_target

    def _get_quality_rationale(self, quality_tier: QualityTier) -> str:
        """Get rationale for quality adjustment"""
        rationales = {
            QualityTier.EXCEPTIONAL: "Exceptional quality justifies premium valuation",
            QualityTier.HIGH: "High quality supports modest premium",
            QualityTier.AVERAGE: "Average quality, no adjustment needed",
            QualityTier.BELOW_AVERAGE: "Below-average quality warrants valuation discount",
            QualityTier.POOR: "Poor quality requires significant valuation discount",
        }
        return rationales.get(quality_tier, "Quality tier assessment")

    def _get_bias_rationale(self, bias: ValuationBias, discount: float) -> str:
        """Get rationale for valuation bias adjustment"""
        if bias == ValuationBias.OVERVALUED:
            return f"Stock appears {abs(discount):.1%} overvalued, applying conservative discount"
        elif bias == ValuationBias.UNDERVALUED:
            return f"Stock appears {discount:.1%} undervalued, applying modest premium"
        else:
            return "Stock appears fairly valued, no bias adjustment"

    def generate_valuation_summary(
        self, metrics: ValuationMetrics, adjusted_target: float, adjustment_details: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive valuation summary"""
        current_to_target = (
            (adjusted_target - metrics.current_price) / metrics.current_price if metrics.current_price > 0 else 0
        )

        # Determine investment appeal
        if current_to_target > 0.20:
            appeal = "Strong Buy"
        elif current_to_target > 0.10:
            appeal = "Buy"
        elif current_to_target > -0.10:
            appeal = "Hold"
        elif current_to_target > -0.20:
            appeal = "Weak Hold"
        else:
            appeal = "Sell"

        return {
            "current_price": metrics.current_price,
            "adjusted_target": adjusted_target,
            "upside_potential": current_to_target,
            "investment_appeal": appeal,
            "valuation_bias": metrics.valuation_bias.value,
            "quality_tier": metrics.quality_tier.value,
            "margin_of_safety": max(0, current_to_target) if current_to_target > 0 else 0,
            "risk_of_loss": max(0, -current_to_target) if current_to_target < 0 else 0,
            "adjustment_summary": adjustment_details.get("summary", {}),
            "key_adjustments": [
                adj
                for key, adj in adjustment_details.items()
                if key in ["quality", "valuation_bias", "risk", "trend"] and adj.get("multiplier", 1.0) != 1.0
            ],
        }
