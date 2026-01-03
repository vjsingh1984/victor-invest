"""
Weight Normalizer Utility

Standardized weight normalization for multi-model valuation blending.
Ensures weights sum to 100% and round to specified increments (default: 5%).

Author: InvestiGator Team
Date: 2025-11-07
"""

import logging
from typing import Dict, List


logger = logging.getLogger(__name__)


class WeightNormalizer:
    """
    Utility for normalizing and rounding model weights.

    Features:
    - Normalize weights to sum to 100%
    - Round to specified increments (5%, 1%, etc.)
    - Handle edge cases (all zeros, negatives)
    - Ensure exact 100% sum (adjust largest weight if needed)
    """

    def __init__(self, rounding_increment: int = 5):
        """
        Initialize WeightNormalizer.

        Args:
            rounding_increment: Percentage increment for rounding (default: 5)
                               - 5 = round to nearest 5% (0, 5, 10, 15, ...)
                               - 1 = round to nearest 1% (0, 1, 2, 3, ...)
        """
        if rounding_increment <= 0 or rounding_increment > 100:
            raise ValueError(f"Rounding increment must be 1-100, got {rounding_increment}")

        self.increment = rounding_increment

    def normalize(
        self,
        weights: Dict[str, float],
        model_order: List[str] = None
    ) -> Dict[str, float]:
        """
        Normalize weights to sum to 100% and round to increment.

        Args:
            weights: Dict mapping model_name → weight
                    Example: {"dcf": 45.3, "pe": 32.1, "ps": 12.6}
            model_order: Optional list defining complete model order
                        Missing models will be added with weight=0
                        Example: ["dcf", "ggm", "pe", "ps", "pb", "ev_ebitda"]

        Returns:
            Normalized and rounded weights summing to 100%
            Example: {"dcf": 50, "pe": 35, "ps": 15}

        Raises:
            ValueError: If all weights are zero or negative
        """
        if not weights:
            raise ValueError("Cannot normalize empty weights dict")

        # Remove zero/negative weights
        non_zero = {k: max(0, v) for k, v in weights.items() if v > 0}

        if not non_zero:
            raise ValueError("All weights are zero or negative, cannot normalize")

        # Normalize to 100%
        total = sum(non_zero.values())
        normalized = {k: (v / total * 100) for k, v in non_zero.items()}

        # Round to nearest increment
        rounded = {}
        for model, weight in normalized.items():
            rounded_weight = round(weight / self.increment) * self.increment
            rounded[model] = max(0, rounded_weight)  # No negatives

        # Adjust to ensure sum = 100%
        current_sum = sum(rounded.values())

        if current_sum != 100:
            diff = 100 - current_sum

            # Add/subtract diff from largest weight
            if rounded:
                max_model = max(rounded, key=rounded.get)
                new_value = rounded[max_model] + diff

                # If adjustment makes largest weight negative, redistribute
                if new_value < 0:
                    logger.warning(
                        f"Adjustment would make {max_model} negative ({new_value}%), "
                        f"redistributing difference"
                    )
                    # Reset and try equal distribution
                    num_models = len(rounded)
                    base_weight = (100 // num_models // self.increment) * self.increment
                    remainder = 100 - (base_weight * num_models)

                    rounded = {k: base_weight for k in rounded.keys()}
                    # Add remainder to first model
                    first_model = list(rounded.keys())[0]
                    rounded[first_model] += remainder
                else:
                    rounded[max_model] = new_value

        # Add zero weights for models in model_order but not in result
        if model_order:
            final = {model: rounded.get(model, 0) for model in model_order}
        else:
            final = rounded

        # Final validation
        final_sum = sum(final.values())
        if abs(final_sum - 100) > 0.01:  # Allow tiny floating point errors
            logger.warning(
                f"Final weights sum to {final_sum}% (not exactly 100%), "
                f"this may cause issues"
            )

        return final

    def normalize_with_fallback(
        self,
        weights: Dict[str, float],
        fallback_weights: Dict[str, float],
        model_order: List[str] = None
    ) -> Dict[str, float]:
        """
        Normalize weights with fallback to default weights if all zeros.

        Args:
            weights: Primary weights to normalize
            fallback_weights: Weights to use if primary weights are all zero
            model_order: Optional complete model order

        Returns:
            Normalized weights (either primary or fallback)
        """
        try:
            return self.normalize(weights, model_order=model_order)
        except ValueError as e:
            if "All weights are zero" in str(e):
                logger.info("All primary weights are zero, using fallback weights")
                try:
                    return self.normalize(fallback_weights, model_order=model_order)
                except ValueError:
                    logger.error("Fallback weights also all zero, cannot normalize")
                    raise
            else:
                raise

    def apply_confidence_weighting(
        self,
        base_weights: Dict[str, float],
        confidences: Dict[str, float],
        model_order: List[str] = None
    ) -> Dict[str, float]:
        """
        Adjust weights based on model confidence scores.

        Multiplies base weights by confidence scores, then normalizes.

        Args:
            base_weights: Base model weights (e.g., from tier classification)
            confidences: Confidence scores per model (0.0-1.0)
            model_order: Optional complete model order

        Returns:
            Confidence-adjusted normalized weights

        Example:
            base_weights = {"dcf": 50, "pe": 30, "ps": 20}
            confidences = {"dcf": 0.8, "pe": 0.6, "ps": 0.3}
            Result: DCF gets boosted relative to others due to higher confidence
        """
        # Multiply base weights by confidence
        adjusted = {}
        for model, weight in base_weights.items():
            if weight > 0:
                confidence = confidences.get(model, 0.0)
                adjusted[model] = weight * max(0.0, min(1.0, confidence))

        # If all adjusted weights are zero, fall back to base weights
        if sum(adjusted.values()) == 0:
            logger.warning("Confidence adjustment resulted in all zero weights, using base weights")
            adjusted = base_weights

        # Normalize
        return self.normalize(adjusted, model_order=model_order)

    def validate_weights(self, weights: Dict[str, float], tolerance: float = 0.01) -> bool:
        """
        Validate that weights are properly normalized.

        Args:
            weights: Weights to validate
            tolerance: Allowed deviation from 100% (default: 0.01%)

        Returns:
            True if weights sum to ~100%, False otherwise
        """
        total = sum(weights.values())
        is_valid = abs(total - 100.0) <= tolerance

        if not is_valid:
            logger.error(f"Weights sum to {total}%, not 100% (tolerance: {tolerance}%)")

        return is_valid

    @staticmethod
    def format_weights_string(weights: Dict[str, float]) -> str:
        """
        Format weights as human-readable string.

        Args:
            weights: Dict of model weights

        Returns:
            Formatted string
            Example: "DCF=50%, PE=30%, PS=20%"
        """
        non_zero = {k: v for k, v in weights.items() if v > 0}
        if not non_zero:
            return "No weights"

        parts = [f"{model.upper()}={weight:.0f}%" for model, weight in sorted(
            non_zero.items(), key=lambda x: x[1], reverse=True
        )]
        return ", ".join(parts)
