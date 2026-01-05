"""
Model Agreement Scorer - Analyze divergence between valuation models.

Provides:
1. Agreement scoring (0-1) based on coefficient of variation
2. Outlier detection using z-scores
3. Confidence adjustments for divergence
4. Outlier weight penalties

Problem being solved:
- Multiple valuation models can produce widely different fair values
- Divergence flags set but no action taken
- No confidence reduction when models disagree
- Outlier models not penalized in final weighting

Solution:
- Calculate agreement score based on coefficient of variation
- Identify outlier models (>2 sigma from weighted mean)
- Apply confidence penalties for divergence
- Reduce weight of outlier models

Usage:
    from investigator.domain.services.model_agreement_scorer import ModelAgreementScorer

    scorer = ModelAgreementScorer()
    agreement = scorer.analyze(model_fair_values, 'AAPL')

    if agreement.divergence_flag:
        print(f"Model divergence detected: CV={agreement.cv:.0%}")
        print(f"Outlier models: {agreement.outlier_models}")
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgreementLevel(Enum):
    """Model agreement level classification."""

    HIGH = "high"  # CV < 15%, models closely agree
    MODERATE = "moderate"  # CV 15-25%, reasonable agreement
    LOW = "low"  # CV 25-35%, notable divergence
    DIVERGENT = "divergent"  # CV > 35%, significant disagreement


@dataclass
class AgreementScore:
    """Result of model agreement analysis."""

    agreement_score: float  # 0-1, higher = more agreement
    cv: float  # Coefficient of variation (std/mean)
    divergence_flag: bool  # True if significant divergence
    outlier_models: List[str]  # Models with z-score > threshold
    confidence_adjustment: float  # -0.15 to +0.10
    agreement_level: AgreementLevel
    weighted_mean: float  # Weighted average fair value
    simple_mean: float  # Simple average fair value
    std_dev: float  # Standard deviation
    model_z_scores: Dict[str, float]  # Z-score for each model
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Get a summary of the agreement analysis."""
        return (
            f"Agreement: {self.agreement_level.value} ({self.agreement_score:.0%}), "
            f"CV: {self.cv:.1%}, "
            f"Confidence adj: {self.confidence_adjustment:+.0%}, "
            f"Outliers: {len(self.outlier_models)}"
        )


@dataclass
class AgreementConfig:
    """Configuration for model agreement scoring."""

    divergence_threshold: float = 0.35  # CV > 35% = divergence
    high_agreement_threshold: float = 0.15  # CV < 15% = high agreement
    zscore_threshold: float = 2.0  # Z-score > 2 = outlier
    outlier_weight_penalty: float = 0.50  # Reduce outlier weight by 50%
    divergence_penalty: float = -0.15  # Max confidence reduction
    high_agreement_bonus: float = 0.10  # Max confidence increase


class ModelAgreementScorer:
    """
    Analyzes agreement between valuation model outputs.

    Measures divergence using coefficient of variation and identifies
    outlier models using z-scores. Provides confidence adjustments
    and weight penalties based on agreement level.

    Example:
        scorer = ModelAgreementScorer()

        fair_values = {
            'dcf': 150.0,
            'pe': 145.0,
            'ps': 200.0,  # Outlier
            'ev_ebitda': 148.0,
        }

        weights = {
            'dcf': 30.0,
            'pe': 25.0,
            'ps': 35.0,
            'ev_ebitda': 10.0,
        }

        agreement = scorer.analyze(fair_values, 'AAPL', weights)
        if agreement.divergence_flag:
            print(f"Divergence detected: {agreement.outlier_models}")

        # Apply outlier penalties
        adjusted_weights = scorer.apply_outlier_penalty(weights, agreement.outlier_models)
    """

    def __init__(self, config: Optional[AgreementConfig] = None):
        """
        Initialize scorer with optional custom configuration.

        Args:
            config: AgreementConfig with custom thresholds
        """
        self.config = config or AgreementConfig()

    def analyze(
        self, model_fair_values: Dict[str, float], symbol: str, model_weights: Optional[Dict[str, float]] = None
    ) -> AgreementScore:
        """
        Analyze agreement between model fair values.

        Args:
            model_fair_values: Dict of model_name -> fair_value
            symbol: Stock symbol for logging
            model_weights: Optional weights for weighted mean calculation

        Returns:
            AgreementScore with analysis results
        """
        notes: List[str] = []

        # Filter out invalid values
        valid_values = {
            model: value for model, value in model_fair_values.items() if self._is_valid_number(value) and value > 0
        }

        if len(valid_values) < 2:
            return AgreementScore(
                agreement_score=0.0,
                cv=0.0,
                divergence_flag=False,
                outlier_models=[],
                confidence_adjustment=0.0,
                agreement_level=AgreementLevel.LOW,
                weighted_mean=0.0,
                simple_mean=0.0,
                std_dev=0.0,
                model_z_scores={},
                notes=["Insufficient valid model outputs for agreement analysis"],
            )

        # Calculate statistics
        values = list(valid_values.values())
        models = list(valid_values.keys())

        # Simple mean
        simple_mean = sum(values) / len(values)

        # Weighted mean (if weights provided)
        if model_weights:
            weighted_sum = sum(value * model_weights.get(model, 1.0) for model, value in valid_values.items())
            weight_sum = sum(model_weights.get(model, 1.0) for model in valid_values)
            weighted_mean = weighted_sum / weight_sum if weight_sum > 0 else simple_mean
        else:
            weighted_mean = simple_mean

        # Standard deviation
        variance = sum((v - simple_mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)

        # Coefficient of variation
        cv = std_dev / simple_mean if simple_mean > 0 else 0

        # Calculate z-scores for each model
        model_z_scores = {}
        outlier_models = []

        for model, value in valid_values.items():
            if std_dev > 0:
                z_score = (value - simple_mean) / std_dev
            else:
                z_score = 0

            model_z_scores[model] = z_score

            if abs(z_score) > self.config.zscore_threshold:
                outlier_models.append(model)
                direction = "above" if z_score > 0 else "below"
                notes.append(f"{model} is outlier ({direction} mean by {abs(z_score):.1f} sigma)")

        # Determine agreement level
        if cv < self.config.high_agreement_threshold:
            agreement_level = AgreementLevel.HIGH
        elif cv < 0.25:
            agreement_level = AgreementLevel.MODERATE
        elif cv < self.config.divergence_threshold:
            agreement_level = AgreementLevel.LOW
        else:
            agreement_level = AgreementLevel.DIVERGENT

        # Determine divergence flag
        divergence_flag = cv > self.config.divergence_threshold

        # Calculate agreement score (inverse of CV, bounded 0-1)
        # CV = 0 -> score = 1.0 (perfect agreement)
        # CV = 0.5 -> score = 0.5
        # CV >= 1.0 -> score = 0.0
        agreement_score = max(0, min(1, 1 - cv))

        # Calculate confidence adjustment
        if agreement_level == AgreementLevel.HIGH:
            confidence_adjustment = self.config.high_agreement_bonus
            notes.append(f"High agreement bonus: {confidence_adjustment:+.0%}")
        elif agreement_level == AgreementLevel.DIVERGENT:
            confidence_adjustment = self.config.divergence_penalty
            notes.append(f"Divergence penalty: {confidence_adjustment:+.0%}")
        elif agreement_level == AgreementLevel.LOW:
            # Graduated penalty
            confidence_adjustment = self.config.divergence_penalty * 0.5
        else:
            confidence_adjustment = 0.0

        # Log analysis
        logger.info(
            f"[{symbol}] Model Agreement Analysis: "
            f"{agreement_level.value} (CV: {cv:.1%}, {len(valid_values)} models)"
        )

        if outlier_models:
            logger.warning(f"[{symbol}] Outlier models detected: {outlier_models}")

        return AgreementScore(
            agreement_score=agreement_score,
            cv=cv,
            divergence_flag=divergence_flag,
            outlier_models=outlier_models,
            confidence_adjustment=confidence_adjustment,
            agreement_level=agreement_level,
            weighted_mean=weighted_mean,
            simple_mean=simple_mean,
            std_dev=std_dev,
            model_z_scores=model_z_scores,
            notes=notes,
        )

    def apply_outlier_penalty(
        self, weights: Dict[str, float], outlier_models: List[str], normalize: bool = True
    ) -> Dict[str, float]:
        """
        Apply weight penalties to outlier models.

        Args:
            weights: Original model weights
            outlier_models: List of outlier model names
            normalize: Whether to normalize weights to 100%

        Returns:
            Adjusted weights with outlier penalties applied
        """
        adjusted = {}

        for model, weight in weights.items():
            if model in outlier_models:
                adjusted[model] = weight * (1 - self.config.outlier_weight_penalty)
                logger.debug(f"Outlier penalty applied to {model}: " f"{weight:.1f}% -> {adjusted[model]:.1f}%")
            else:
                adjusted[model] = weight

        # Normalize to 100% if requested
        if normalize:
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {model: weight / total * 100 for model, weight in adjusted.items()}

        return adjusted

    def get_weighted_fair_value(
        self, model_fair_values: Dict[str, float], model_weights: Dict[str, float], apply_outlier_penalty: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted fair value with optional outlier handling.

        Args:
            model_fair_values: Fair values from each model
            model_weights: Weights for each model
            apply_outlier_penalty: Whether to penalize outliers

        Returns:
            Tuple of (weighted_fair_value, effective_weights)
        """
        if apply_outlier_penalty:
            # Analyze agreement first
            agreement = self.analyze(model_fair_values, "weighted_calc", model_weights)
            effective_weights = self.apply_outlier_penalty(model_weights, agreement.outlier_models)
        else:
            effective_weights = model_weights

        # Calculate weighted fair value
        weighted_sum = 0
        weight_sum = 0

        for model, value in model_fair_values.items():
            if self._is_valid_number(value) and value > 0:
                weight = effective_weights.get(model, 0)
                weighted_sum += value * weight
                weight_sum += weight

        weighted_fair_value = weighted_sum / weight_sum if weight_sum > 0 else 0

        return (weighted_fair_value, effective_weights)

    def _is_valid_number(self, value: Any) -> bool:
        """Check if value is a valid, finite number."""
        if value is None:
            return False
        try:
            num = float(value)
            return not (math.isnan(num) or math.isinf(num))
        except (TypeError, ValueError):
            return False


# Singleton instance
_scorer: Optional[ModelAgreementScorer] = None


def get_model_agreement_scorer() -> ModelAgreementScorer:
    """Get the singleton ModelAgreementScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ModelAgreementScorer()
    return _scorer
