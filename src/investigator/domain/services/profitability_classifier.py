"""
Profitability Classifier - Multi-indicator classification for pre-profit detection.

Provides:
1. Multi-indicator profitability check (net income, operating income, EBITDA, FCF)
2. Classification into profitability stages
3. Confidence scoring for classification
4. Valuation model applicability guidance

Problem being solved:
- Current pre-profit detection relies solely on EBITDA
- Missing EBITDA causes 100% P/S allocation even for profitable companies
- Single-indicator approach is fragile to data gaps

Solution:
- Check multiple profitability indicators in priority order
- Classify into stages: profitable, marginally_profitable, transitioning, pre_profit
- Return confidence and valuation model recommendations

Usage:
    from investigator.domain.services.profitability_classifier import ProfitabilityClassifier

    classifier = ProfitabilityClassifier()
    result = classifier.classify(financials, ratios)
    print(f"Classification: {result.stage.value}")
    print(f"Applicable models: {result.applicable_models}")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProfitabilityStage(Enum):
    """Profitability classification stages."""

    PROFITABLE = "profitable"  # Clear profitability
    MARGINALLY_PROFITABLE = "marginally_profitable"  # Thin margins
    TRANSITIONING = "transitioning"  # Mixed signals
    PRE_PROFIT = "pre_profit"  # Not yet profitable
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class ProfitabilityIndicator:
    """Single profitability indicator result."""

    name: str
    value: Optional[float]
    is_positive: bool
    margin: Optional[float]  # As percentage
    confidence: float  # 0-1, how reliable is this indicator


@dataclass
class ProfitabilityClassification:
    """Result of profitability classification."""

    stage: ProfitabilityStage
    confidence: float  # 0-1
    indicators_checked: List[ProfitabilityIndicator]
    indicators_positive: int
    indicators_total: int
    primary_indicator: Optional[str]  # Which indicator drove the classification
    applicable_models: List[str]  # Which valuation models are appropriate
    model_adjustments: Dict[str, float]  # Suggested weight adjustments
    notes: List[str] = field(default_factory=list)

    def is_profitable(self) -> bool:
        """Check if company is classified as profitable."""
        return self.stage in [ProfitabilityStage.PROFITABLE, ProfitabilityStage.MARGINALLY_PROFITABLE]

    def summary(self) -> str:
        """Get a summary of the classification."""
        return (
            f"Stage: {self.stage.value}, "
            f"Confidence: {self.confidence:.0%}, "
            f"Indicators: {self.indicators_positive}/{self.indicators_total} positive, "
            f"Models: {', '.join(self.applicable_models)}"
        )


class ProfitabilityClassifier:
    """
    Classifies companies by profitability using multiple indicators.

    Uses a priority chain of indicators to determine profitability:
    1. Net Income (most authoritative for P/E)
    2. Operating Income (core business profitability)
    3. EBITDA (cash generation before capital allocation)
    4. Free Cash Flow (actual cash generation)

    Example:
        classifier = ProfitabilityClassifier()

        financials = {
            'net_income': 15000000000,
            'operating_income': 20000000000,
            'ebitda': 25000000000,
            'free_cash_flow': 18000000000,
            'revenue': 100000000000,
        }

        ratios = {
            'net_margin': 15.0,
            'operating_margin': 20.0,
            'fcf_margin': 18.0,
        }

        result = classifier.classify(financials, ratios)
        print(f"Stage: {result.stage.value}")  # "profitable"
        print(f"Applicable models: {result.applicable_models}")  # ["dcf", "pe", "ps", ...]
    """

    # Priority order for checking profitability
    INDICATORS_PRIORITY = [
        ("net_income", "net_margin", 1.0),  # Highest priority
        ("operating_income", "operating_margin", 0.9),
        ("ebitda", None, 0.8),  # EBITDA often doesn't have margin
        ("free_cash_flow", "fcf_margin", 0.85),
    ]

    # Margin thresholds for classification
    MARGIN_THRESHOLDS = {
        "profitable": 5.0,  # >= 5% margin is clearly profitable
        "marginal": 0.0,  # > 0% but < 5% is marginally profitable
        "transitioning": -5.0,  # > -5% may be transitioning
    }

    # Model applicability by stage
    MODEL_APPLICABILITY = {
        ProfitabilityStage.PROFITABLE: {
            "models": ["dcf", "pe", "ps", "pb", "ev_ebitda", "ggm"],
            "adjustments": {"dcf": 1.0, "pe": 1.0, "ps": 0.8, "pb": 1.0, "ev_ebitda": 1.0, "ggm": 1.0},
        },
        ProfitabilityStage.MARGINALLY_PROFITABLE: {
            "models": ["dcf", "pe", "ps", "ev_ebitda"],
            "adjustments": {"dcf": 0.9, "pe": 0.8, "ps": 1.0, "ev_ebitda": 1.0},
        },
        ProfitabilityStage.TRANSITIONING: {
            "models": ["dcf", "ps", "ev_ebitda"],
            "adjustments": {"dcf": 0.7, "ps": 1.1, "ev_ebitda": 0.9},
        },
        ProfitabilityStage.PRE_PROFIT: {
            "models": ["ps", "ev_revenue", "rule_of_40"],
            "adjustments": {"ps": 1.2, "ev_revenue": 1.0, "rule_of_40": 1.0},
        },
        ProfitabilityStage.UNKNOWN: {
            "models": ["ps"],
            "adjustments": {"ps": 1.0},
        },
    }

    def __init__(self, margin_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize classifier with optional custom thresholds.

        Args:
            margin_thresholds: Custom margin thresholds for classification
        """
        self.margin_thresholds = margin_thresholds or self.MARGIN_THRESHOLDS

    def classify(
        self, financials: Dict[str, Any], ratios: Optional[Dict[str, Any]] = None
    ) -> ProfitabilityClassification:
        """
        Classify company profitability using multiple indicators.

        Args:
            financials: Financial data dict with net_income, operating_income, etc.
            ratios: Optional ratio data with margins

        Returns:
            ProfitabilityClassification with stage, confidence, and model guidance
        """
        ratios = ratios or {}
        indicators_checked: List[ProfitabilityIndicator] = []
        notes: List[str] = []

        # Check each indicator in priority order
        for value_key, margin_key, weight in self.INDICATORS_PRIORITY:
            value = self._get_value(financials, value_key)
            margin = self._get_value(ratios, margin_key) if margin_key else None

            # Also try to get margin from financials if not in ratios
            if margin is None and margin_key:
                margin = self._get_value(financials, margin_key)

            # Calculate margin from value and revenue if available
            if margin is None and value is not None:
                revenue = self._get_value(financials, "revenue") or self._get_value(financials, "total_revenue")
                if revenue and revenue > 0:
                    margin = (value / revenue) * 100

            is_positive = value is not None and value > 0
            indicator_confidence = weight if value is not None else 0

            indicators_checked.append(
                ProfitabilityIndicator(
                    name=value_key, value=value, is_positive=is_positive, margin=margin, confidence=indicator_confidence
                )
            )

            if value is not None:
                margin_str = f"{margin:.1f}%" if margin is not None else "N/A"
                logger.debug(
                    f"Indicator {value_key}: value={value/1e6:.1f}M, "
                    f"margin={margin_str}, "
                    f"positive={is_positive}"
                )

        # Count positive indicators
        indicators_positive = sum(1 for i in indicators_checked if i.is_positive)
        indicators_total = sum(1 for i in indicators_checked if i.value is not None)

        # Determine stage based on indicators
        stage, primary_indicator, confidence = self._determine_stage(
            indicators_checked, indicators_positive, indicators_total
        )

        # Get model applicability for this stage
        applicability = self.MODEL_APPLICABILITY.get(stage, self.MODEL_APPLICABILITY[ProfitabilityStage.UNKNOWN])

        # Add notes
        if indicators_total == 0:
            notes.append("No profitability indicators available")
        elif indicators_positive == 0 and indicators_total > 0:
            notes.append("All available indicators show losses")
        elif indicators_positive < indicators_total:
            negative_indicators = [i.name for i in indicators_checked if i.value is not None and not i.is_positive]
            notes.append(f"Mixed signals: {', '.join(negative_indicators)} negative")

        return ProfitabilityClassification(
            stage=stage,
            confidence=confidence,
            indicators_checked=indicators_checked,
            indicators_positive=indicators_positive,
            indicators_total=indicators_total,
            primary_indicator=primary_indicator,
            applicable_models=applicability["models"],
            model_adjustments=applicability["adjustments"],
            notes=notes,
        )

    def _get_value(self, data: Dict[str, Any], key: str) -> Optional[float]:
        """Get a numeric value from a dict, handling None and invalid values."""
        if data is None:
            return None

        value = data.get(key)
        if value is None:
            return None

        try:
            num = float(value)
            if not (num != num):  # Check for NaN
                return num
        except (TypeError, ValueError):
            pass

        return None

    def _determine_stage(
        self, indicators: List[ProfitabilityIndicator], positive_count: int, total_count: int
    ) -> Tuple[ProfitabilityStage, Optional[str], float]:
        """
        Determine profitability stage from indicator results.

        Returns:
            Tuple of (stage, primary_indicator_name, confidence)
        """
        if total_count == 0:
            return (ProfitabilityStage.UNKNOWN, None, 0.0)

        # Find the highest-priority positive indicator
        primary_indicator = None
        primary_margin = None

        for indicator in indicators:
            if indicator.is_positive:
                primary_indicator = indicator.name
                primary_margin = indicator.margin
                break

        # Calculate confidence based on indicator agreement
        agreement_ratio = positive_count / total_count
        base_confidence = 0.5 + (agreement_ratio * 0.5)  # 0.5 to 1.0

        # Determine stage
        if positive_count == total_count and total_count >= 2:
            # All indicators positive
            if primary_margin is not None and primary_margin >= self.margin_thresholds["profitable"]:
                return (ProfitabilityStage.PROFITABLE, primary_indicator, min(0.95, base_confidence))
            elif primary_margin is not None and primary_margin >= self.margin_thresholds["marginal"]:
                return (ProfitabilityStage.MARGINALLY_PROFITABLE, primary_indicator, min(0.85, base_confidence))
            else:
                return (ProfitabilityStage.PROFITABLE, primary_indicator, min(0.80, base_confidence))

        elif positive_count >= total_count / 2:
            # Majority positive
            if primary_margin is not None and primary_margin >= self.margin_thresholds["profitable"]:
                return (ProfitabilityStage.MARGINALLY_PROFITABLE, primary_indicator, min(0.75, base_confidence))
            else:
                return (ProfitabilityStage.TRANSITIONING, primary_indicator, min(0.70, base_confidence))

        elif positive_count > 0:
            # Some positive, but minority
            return (ProfitabilityStage.TRANSITIONING, primary_indicator, min(0.60, base_confidence))

        else:
            # All negative
            return (ProfitabilityStage.PRE_PROFIT, None, min(0.80, base_confidence))

    def get_model_weight_adjustments(
        self,
        financials: Dict[str, Any],
        ratios: Optional[Dict[str, Any]] = None,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], ProfitabilityClassification]:
        """
        Get adjusted model weights based on profitability classification.

        Args:
            financials: Financial data
            ratios: Ratio data
            base_weights: Starting weights to adjust

        Returns:
            Tuple of (adjusted_weights, classification)
        """
        classification = self.classify(financials, ratios)

        if base_weights is None:
            return ({}, classification)

        adjusted_weights = {}
        for model, weight in base_weights.items():
            adjustment = classification.model_adjustments.get(model, 1.0)
            adjusted_weights[model] = weight * adjustment

            if adjustment != 1.0:
                logger.debug(
                    f"[{classification.stage.value}] {model}: "
                    f"{weight:.1f}% → {adjusted_weights[model]:.1f}% (×{adjustment:.2f})"
                )

        return (adjusted_weights, classification)


# Singleton instance
_classifier: Optional[ProfitabilityClassifier] = None


def get_profitability_classifier() -> ProfitabilityClassifier:
    """Get the singleton ProfitabilityClassifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = ProfitabilityClassifier()
    return _classifier
