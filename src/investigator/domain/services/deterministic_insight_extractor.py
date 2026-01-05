"""
Deterministic Insight Extractor - Replaces LLM-based key insight extraction.

This module provides rule-based extraction of key insights that replaces
the LLM call in `_extract_key_insights()`. It uses pattern matching and
metric thresholds to identify positive/negative factors.

Benefits:
- Zero token cost
- Instant response
- Consistent, reproducible insights
- Threshold-based objectivity

Design Principles (SOLID):
- Single Responsibility: Each extractor handles one analysis type
- Open/Closed: New insight patterns can be added via registry
- Liskov Substitution: All extractors implement common protocol
- Interface Segregation: Focused interfaces for positive/negative extraction
- Dependency Inversion: Depends on abstractions, not concretions
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SourceInsight:
    """Insights extracted from a single analysis source."""

    source: str
    positive_factors: List[str]
    negative_factors: List[str]
    critical_metric: str
    unique_insight: str
    confidence: int  # 0-100


@dataclass
class ExtractedInsights:
    """Complete extracted insights from all sources."""

    fundamental: Optional[SourceInsight] = None
    technical: Optional[SourceInsight] = None
    sec: Optional[SourceInsight] = None
    market_context: Optional[SourceInsight] = None
    quantitative: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        result = {}
        if self.fundamental:
            result["fundamental"] = self._source_to_dict(self.fundamental)
        if self.technical:
            result["technical"] = self._source_to_dict(self.technical)
        if self.sec:
            result["sec"] = self._source_to_dict(self.sec)
        if self.market_context:
            result["market_context"] = self._source_to_dict(self.market_context)
        if self.quantitative:
            result["quantitative"] = self.quantitative
        return result

    def _source_to_dict(self, insight: SourceInsight) -> Dict[str, Any]:
        return {
            "positive_factors": insight.positive_factors,
            "negative_factors": insight.negative_factors,
            "critical_metric": insight.critical_metric,
            "unique_insight": insight.unique_insight,
            "confidence": insight.confidence,
        }


class SourceExtractor(Protocol):
    """Protocol for extracting insights from a specific analysis source."""

    def extract(self, data: Dict[str, Any]) -> Optional[SourceInsight]:
        """Extract insights from source data."""
        ...


# ============================================================================
# Metric Thresholds
# ============================================================================


class MetricThresholds:
    """Configurable thresholds for metric evaluation."""

    # Revenue growth thresholds
    REVENUE_GROWTH_EXCELLENT = 0.20  # >20% = excellent
    REVENUE_GROWTH_GOOD = 0.10  # >10% = good
    REVENUE_GROWTH_POOR = 0.0  # <0% = poor

    # Margin thresholds
    PROFIT_MARGIN_EXCELLENT = 0.20  # >20% = excellent
    PROFIT_MARGIN_GOOD = 0.10  # >10% = good
    PROFIT_MARGIN_POOR = 0.05  # <5% = poor

    # ROE thresholds
    ROE_EXCELLENT = 0.20  # >20% = excellent
    ROE_GOOD = 0.12  # >12% = good
    ROE_POOR = 0.08  # <8% = poor

    # Debt thresholds
    DEBT_TO_EQUITY_LOW = 0.5  # <0.5 = conservative
    DEBT_TO_EQUITY_HIGH = 1.5  # >1.5 = high leverage

    # P/E thresholds
    PE_LOW = 15  # <15 = potentially undervalued
    PE_HIGH = 35  # >35 = potentially overvalued

    # Data quality
    DATA_QUALITY_EXCELLENT = 80
    DATA_QUALITY_GOOD = 60
    DATA_QUALITY_POOR = 40


# ============================================================================
# Source Extractors
# ============================================================================


class FundamentalInsightExtractor:
    """Extracts insights from fundamental analysis."""

    def extract(self, data: Dict[str, Any]) -> Optional[SourceInsight]:
        """Extract insights from fundamental analysis data."""
        if not data:
            return None

        positive = []
        negative = []

        # Extract valuation insights
        valuation = data.get("valuation", {})
        fair_value = valuation.get("fair_value", 0)
        current_price = valuation.get("current_price", 0)

        if fair_value > 0 and current_price > 0:
            upside = (fair_value - current_price) / current_price
            if upside > 0.15:
                positive.append(f"Attractive valuation with {upside:.0%} upside to fair value")
            elif upside < -0.15:
                negative.append(f"Valuation stretched with {abs(upside):.0%} downside to fair value")

        # Extract ratio insights
        ratios = data.get("ratios", data.get("analysis", {}).get("ratios", {}))

        # Revenue growth
        rev_growth = ratios.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth > MetricThresholds.REVENUE_GROWTH_EXCELLENT:
                positive.append(f"Strong revenue growth of {rev_growth:.0%}")
            elif rev_growth < MetricThresholds.REVENUE_GROWTH_POOR:
                negative.append(f"Declining revenue ({rev_growth:.0%})")

        # Profit margin
        margin = ratios.get("profit_margin") or ratios.get("operating_margin")
        if margin is not None:
            if margin > MetricThresholds.PROFIT_MARGIN_EXCELLENT:
                positive.append(f"Excellent profit margins of {margin:.0%}")
            elif margin < MetricThresholds.PROFIT_MARGIN_POOR:
                negative.append(f"Thin profit margins of {margin:.0%}")

        # ROE
        roe = ratios.get("roe") or ratios.get("return_on_equity")
        if roe is not None:
            if roe > MetricThresholds.ROE_EXCELLENT:
                positive.append(f"High return on equity of {roe:.0%}")
            elif roe < MetricThresholds.ROE_POOR:
                negative.append(f"Below-average return on equity of {roe:.0%}")

        # Debt
        debt_equity = ratios.get("debt_to_equity")
        if debt_equity is not None:
            if debt_equity < MetricThresholds.DEBT_TO_EQUITY_LOW:
                positive.append("Conservative balance sheet with low leverage")
            elif debt_equity > MetricThresholds.DEBT_TO_EQUITY_HIGH:
                negative.append(f"High financial leverage (D/E: {debt_equity:.1f}x)")

        # Extract from analysis conclusions if available
        analysis = data.get("analysis", {})
        if isinstance(analysis, dict):
            strengths = analysis.get("strengths", analysis.get("bull_case", []))
            if isinstance(strengths, list):
                for strength in strengths[:2]:
                    if strength and strength not in positive:
                        positive.append(str(strength))

            weaknesses = analysis.get("weaknesses", analysis.get("bear_case", []))
            if isinstance(weaknesses, list):
                for weakness in weaknesses[:2]:
                    if weakness and weakness not in negative:
                        negative.append(str(weakness))

        # Determine critical metric
        critical_metric = self._determine_critical_metric(valuation, ratios)

        # Determine unique insight
        unique_insight = self._determine_unique_insight(data, positive, negative)

        # Calculate confidence
        confidence = self._calculate_confidence(data)

        return SourceInsight(
            source="fundamental",
            positive_factors=positive[:3],
            negative_factors=negative[:3],
            critical_metric=critical_metric,
            unique_insight=unique_insight,
            confidence=confidence,
        )

    def _determine_critical_metric(self, valuation: Dict[str, Any], ratios: Dict[str, Any]) -> str:
        """Determine the most critical metric from fundamental analysis."""
        # Priority: valuation upside > ROE > revenue growth > margins

        fair_value = valuation.get("fair_value", 0)
        current_price = valuation.get("current_price", 0)

        if fair_value > 0 and current_price > 0:
            upside = (fair_value - current_price) / current_price
            return f"Implied upside/downside: {upside:+.0%} to fair value of ${fair_value:.2f}"

        roe = ratios.get("roe")
        if roe is not None:
            return f"Return on equity: {roe:.1%}"

        rev_growth = ratios.get("revenue_growth")
        if rev_growth is not None:
            return f"Revenue growth: {rev_growth:.1%}"

        return "Financial metrics require further analysis"

    def _determine_unique_insight(self, data: Dict[str, Any], positive: List[str], negative: List[str]) -> str:
        """Determine a unique insight not commonly found elsewhere."""
        # Check for multi-model valuation insights
        multi_model = data.get("multi_model_summary", {})
        if multi_model:
            divergence = multi_model.get("divergence_flag", False)
            if divergence:
                return "Significant model divergence suggests valuation uncertainty - approach with margin of safety"

            agreement = multi_model.get("model_agreement_score", 0)
            if agreement > 0.8:
                return "High model agreement provides confidence in fair value estimate"

        # Check for quality score extremes
        quality_score = data.get("quality_score", 50)
        if quality_score > 90:
            return "Exceptional business quality metrics suggest durable competitive advantage"
        elif quality_score < 30:
            return "Quality concerns warrant additional due diligence before investing"

        # Default based on balance of factors
        if len(positive) > len(negative):
            return "Fundamental analysis supports constructive outlook on risk-adjusted basis"
        elif len(negative) > len(positive):
            return "Fundamental analysis suggests caution with limited margin of safety"
        else:
            return "Mixed fundamental picture requires careful position sizing"

    def _calculate_confidence(self, data: Dict[str, Any]) -> int:
        """Calculate confidence in fundamental analysis."""
        base_confidence = 70

        # Adjust for data quality
        dq = data.get("data_quality", {})
        dq_score = dq.get("data_quality_score", 50)
        if dq_score >= MetricThresholds.DATA_QUALITY_EXCELLENT:
            base_confidence += 15
        elif dq_score >= MetricThresholds.DATA_QUALITY_GOOD:
            base_confidence += 5
        elif dq_score < MetricThresholds.DATA_QUALITY_POOR:
            base_confidence -= 20

        # Adjust for model agreement
        multi_model = data.get("multi_model_summary", {})
        agreement = multi_model.get("model_agreement_score", 0.5)
        if agreement > 0.7:
            base_confidence += 10
        elif agreement < 0.4:
            base_confidence -= 15

        return max(20, min(95, base_confidence))


class TechnicalInsightExtractor:
    """Extracts insights from technical analysis."""

    def extract(self, data: Dict[str, Any]) -> Optional[SourceInsight]:
        """Extract insights from technical analysis data."""
        if not data:
            return None

        positive = []
        negative = []

        # Extract signals
        signals = data.get("signals", data.get("analysis", {}))
        if isinstance(signals, dict):
            trend = signals.get("trend", "").lower()
            if "bullish" in trend:
                positive.append("Technical trend is bullish")
            elif "bearish" in trend:
                negative.append("Technical trend is bearish")

            # Momentum
            rsi = signals.get("rsi", signals.get("momentum", {}).get("rsi"))
            if rsi is not None:
                if rsi < 30:
                    positive.append(f"Oversold RSI ({rsi:.0f}) suggests potential bounce")
                elif rsi > 70:
                    negative.append(f"Overbought RSI ({rsi:.0f}) suggests potential pullback")

            # Moving averages
            ma_signal = signals.get("ma_signal", signals.get("moving_averages", {}).get("signal"))
            if ma_signal:
                if "golden" in str(ma_signal).lower() or "above" in str(ma_signal).lower():
                    positive.append("Price above key moving averages")
                elif "death" in str(ma_signal).lower() or "below" in str(ma_signal).lower():
                    negative.append("Price below key moving averages")

            # Volume
            volume_signal = signals.get("volume_signal", signals.get("volume", {}).get("signal"))
            if volume_signal:
                if "increasing" in str(volume_signal).lower() or "high" in str(volume_signal).lower():
                    positive.append("Strong volume supports price action")
                elif "declining" in str(volume_signal).lower():
                    negative.append("Weak volume raises sustainability concerns")

        # Support/resistance
        levels = data.get("levels", {})
        support = levels.get("support")
        resistance = levels.get("resistance")
        current = data.get("current_price", levels.get("current_price"))

        if support and resistance and current:
            if isinstance(support, (int, float)) and isinstance(current, (int, float)):
                support_distance = (current - support) / current
                if support_distance < 0.05:
                    positive.append("Price near strong support level")

            if isinstance(resistance, (int, float)) and isinstance(current, (int, float)):
                resistance_distance = (resistance - current) / current
                if resistance_distance < 0.05:
                    negative.append("Price approaching resistance level")

        # Critical metric
        critical_metric = self._determine_critical_metric(data)

        # Unique insight
        unique_insight = self._determine_unique_insight(data, positive, negative)

        # Confidence
        confidence = self._calculate_confidence(data)

        return SourceInsight(
            source="technical",
            positive_factors=positive[:3],
            negative_factors=negative[:3],
            critical_metric=critical_metric,
            unique_insight=unique_insight,
            confidence=confidence,
        )

    def _determine_critical_metric(self, data: Dict[str, Any]) -> str:
        """Determine critical technical metric."""
        signals = data.get("signals", data.get("analysis", {}))

        # Priority: trend > RSI > MA signal
        trend = signals.get("trend")
        if trend:
            return f"Primary trend: {trend}"

        rsi = signals.get("rsi", signals.get("momentum", {}).get("rsi"))
        if rsi is not None:
            return f"RSI: {rsi:.0f} (oversold <30, overbought >70)"

        return "Technical signals mixed - await confirmation"

    def _determine_unique_insight(self, data: Dict[str, Any], positive: List[str], negative: List[str]) -> str:
        """Determine unique technical insight."""
        signals = data.get("signals", {})

        # Check for pattern recognition
        pattern = signals.get("pattern")
        if pattern:
            return f"Technical pattern detected: {pattern}"

        # Check for divergences
        if signals.get("divergence"):
            return "Price-indicator divergence may signal trend reversal"

        # Default
        if len(positive) > len(negative):
            return "Technical momentum favors continuation of current trend"
        else:
            return "Technical caution warranted - await clearer signals"

    def _calculate_confidence(self, data: Dict[str, Any]) -> int:
        """Calculate confidence in technical analysis."""
        base_confidence = 65

        # Technical confidence affected by signal clarity
        signals = data.get("signals", {})

        # Strong trend = higher confidence
        trend = str(signals.get("trend", "")).lower()
        if "strong" in trend:
            base_confidence += 15
        elif "weak" in trend or "neutral" in trend:
            base_confidence -= 10

        # Multiple confirming signals = higher confidence
        confirming = 0
        if signals.get("rsi"):
            confirming += 1
        if signals.get("ma_signal"):
            confirming += 1
        if signals.get("volume_signal"):
            confirming += 1

        base_confidence += confirming * 5

        return max(30, min(90, base_confidence))


class SECInsightExtractor:
    """Extracts insights from SEC filing analysis."""

    def extract(self, data: Dict[str, Any]) -> Optional[SourceInsight]:
        """Extract insights from SEC analysis data."""
        if not data:
            return None

        positive = []
        negative = []

        analysis = data.get("analysis", data)

        # Revenue trends
        revenue_trend = analysis.get("revenue_trend", analysis.get("trends", {}).get("revenue"))
        if revenue_trend:
            if "growing" in str(revenue_trend).lower() or "positive" in str(revenue_trend).lower():
                positive.append("SEC filings show consistent revenue growth")
            elif "declining" in str(revenue_trend).lower() or "negative" in str(revenue_trend).lower():
                negative.append("SEC filings show revenue decline trend")

        # Profitability
        profit_trend = analysis.get("profit_trend", analysis.get("trends", {}).get("profit"))
        if profit_trend:
            if "improving" in str(profit_trend).lower():
                positive.append("Profitability trend improving per SEC filings")
            elif "declining" in str(profit_trend).lower():
                negative.append("Profitability pressure evident in SEC filings")

        # Risk factors
        risks = analysis.get("risks", analysis.get("risk_factors", []))
        if isinstance(risks, list) and risks:
            for risk in risks[:2]:
                if isinstance(risk, dict):
                    risk_desc = risk.get("description", str(risk))
                else:
                    risk_desc = str(risk)
                if len(risk_desc) > 10:
                    negative.append(risk_desc[:100])

        # Quality metrics from SEC data
        dq = data.get("data_quality", {})
        completeness = dq.get("completeness", dq.get("data_quality_score", 50))
        if completeness > 80:
            positive.append("Comprehensive SEC disclosure supports analysis confidence")
        elif completeness < 50:
            negative.append("Limited SEC data availability reduces analysis confidence")

        # Overall rating
        rating = analysis.get("overall_rating", 5)
        if rating >= 8:
            positive.append(f"SEC analysis overall rating: {rating}/10 (strong)")
        elif rating <= 4:
            negative.append(f"SEC analysis overall rating: {rating}/10 (weak)")

        # Critical metric
        critical_metric = self._determine_critical_metric(analysis)

        # Unique insight
        unique_insight = self._determine_unique_insight(data, analysis)

        # Confidence
        confidence = self._calculate_confidence(data)

        return SourceInsight(
            source="sec",
            positive_factors=positive[:3],
            negative_factors=negative[:3],
            critical_metric=critical_metric,
            unique_insight=unique_insight,
            confidence=confidence,
        )

    def _determine_critical_metric(self, analysis: Dict[str, Any]) -> str:
        """Determine critical SEC-derived metric."""
        rating = analysis.get("overall_rating")
        if rating is not None:
            return f"SEC filing quality rating: {rating}/10"

        fiscal_period = analysis.get("fiscal_period")
        if fiscal_period:
            return f"Latest SEC data: {fiscal_period}"

        return "SEC filing analysis in progress"

    def _determine_unique_insight(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Determine unique SEC insight."""
        # Check for accounting quality flags
        accounting_flags = analysis.get("accounting_quality", {}).get("flags", [])
        if accounting_flags:
            return f"Accounting quality flag: {accounting_flags[0]}"

        # Check for material changes
        material_changes = analysis.get("material_changes")
        if material_changes:
            return f"Material SEC disclosure: {str(material_changes)[:100]}"

        return "SEC filings provide foundational data for fundamental analysis"

    def _calculate_confidence(self, data: Dict[str, Any]) -> int:
        """Calculate confidence in SEC analysis."""
        base_confidence = 75  # SEC data is authoritative

        dq = data.get("data_quality", {})
        dq_score = dq.get("data_quality_score", 50)

        if dq_score >= 80:
            base_confidence += 10
        elif dq_score < 50:
            base_confidence -= 20

        return max(40, min(95, base_confidence))


# ============================================================================
# Quantitative Insight Extractor
# ============================================================================


class QuantitativeInsightExtractor:
    """Extracts quantitative insights from numerical data."""

    def extract(
        self,
        fundamental: Optional[Dict[str, Any]] = None,
        technical: Optional[Dict[str, Any]] = None,
        sec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract quantitative metrics summary."""
        quant = {}

        # Valuation metrics
        if fundamental:
            valuation = fundamental.get("valuation", {})
            quant["valuation"] = {
                "current_price": valuation.get("current_price"),
                "fair_value": valuation.get("fair_value"),
                "pe_ratio": valuation.get("pe_ratio"),
                "ps_ratio": valuation.get("ps_ratio"),
                "dividend_yield": valuation.get("dividend_yield"),
            }

            # Quality metrics
            quant["quality_score"] = fundamental.get("quality_score", 50)

            # Model summary
            multi_model = fundamental.get("multi_model_summary", {})
            if multi_model:
                quant["model_agreement"] = multi_model.get("model_agreement_score")
                quant["overall_confidence"] = multi_model.get("overall_confidence")

        # Technical metrics
        if technical:
            signals = technical.get("signals", {})
            quant["technical"] = {
                "trend": signals.get("trend"),
                "rsi": signals.get("rsi", signals.get("momentum", {}).get("rsi")),
                "signal": signals.get("overall_signal"),
            }

        # SEC metrics
        if sec:
            analysis = sec.get("analysis", {})
            quant["sec_rating"] = analysis.get("overall_rating")

        # Filter out None values
        return {k: v for k, v in quant.items() if v is not None}


# ============================================================================
# Main Extractor
# ============================================================================


class DeterministicInsightExtractor:
    """
    Main orchestrator for deterministic insight extraction.

    Replaces LLM-based `_extract_key_insights()` with rule-based extraction.

    Usage:
        extractor = DeterministicInsightExtractor()
        insights = extractor.extract(
            fundamental=...,
            technical=...,
            sec=...,
            market_context=...
        )
        output_dict = insights.to_dict()
    """

    def __init__(
        self,
        fundamental_extractor: Optional[SourceExtractor] = None,
        technical_extractor: Optional[SourceExtractor] = None,
        sec_extractor: Optional[SourceExtractor] = None,
        quantitative_extractor: Optional[QuantitativeInsightExtractor] = None,
    ):
        self.fundamental_extractor = fundamental_extractor or FundamentalInsightExtractor()
        self.technical_extractor = technical_extractor or TechnicalInsightExtractor()
        self.sec_extractor = sec_extractor or SECInsightExtractor()
        self.quantitative_extractor = quantitative_extractor or QuantitativeInsightExtractor()

    def extract(
        self,
        fundamental: Optional[Dict[str, Any]] = None,
        technical: Optional[Dict[str, Any]] = None,
        sec: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractedInsights:
        """
        Extract key insights from all analysis sources.

        Args:
            fundamental: Fundamental analysis results
            technical: Technical analysis results
            sec: SEC analysis results
            market_context: Market context data

        Returns:
            ExtractedInsights with all source insights
        """
        insights = ExtractedInsights()

        # Extract from each source
        if fundamental:
            insights.fundamental = self.fundamental_extractor.extract(fundamental)

        if technical:
            insights.technical = self.technical_extractor.extract(technical)

        if sec:
            insights.sec = self.sec_extractor.extract(sec)

        # Extract quantitative summary
        insights.quantitative = self.quantitative_extractor.extract(
            fundamental=fundamental, technical=technical, sec=sec
        )

        return insights


# ============================================================================
# Convenience function for drop-in replacement
# ============================================================================


def extract_key_insights(
    fundamental: Optional[Dict[str, Any]] = None,
    technical: Optional[Dict[str, Any]] = None,
    sec: Optional[Dict[str, Any]] = None,
    market_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for LLM-based key insight extraction.

    Returns dict with same structure as LLM response for API compatibility.

    Example:
        # Before (LLM):
        response = await self.ollama.generate(model=..., prompt=...)

        # After (deterministic):
        response = extract_key_insights(
            fundamental=synthesis_input.fundamental_analysis,
            technical=synthesis_input.technical_analysis,
            sec=synthesis_input.sec_analysis
        )
    """
    extractor = DeterministicInsightExtractor()
    insights = extractor.extract(fundamental=fundamental, technical=technical, sec=sec, market_context=market_context)
    return insights.to_dict()


__all__ = [
    "DeterministicInsightExtractor",
    "ExtractedInsights",
    "SourceInsight",
    "FundamentalInsightExtractor",
    "TechnicalInsightExtractor",
    "SECInsightExtractor",
    "QuantitativeInsightExtractor",
    "MetricThresholds",
    "extract_key_insights",
]
