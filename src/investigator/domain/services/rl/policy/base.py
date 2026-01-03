"""
Base RL Policy Interface

Abstract base class defining the policy interface for RL-based model weighting.
All policy implementations must inherit from this class.

Follows Open/Closed Principle: Extend by creating new policy classes,
not by modifying existing ones.

Usage:
    from investigator.domain.services.rl.policy import RLPolicy

    class MyCustomPolicy(RLPolicy):
        def predict(self, context):
            # Custom prediction logic
            pass

        def update(self, context, action, reward):
            # Custom learning logic
            pass
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from investigator.domain.services.rl.models import (
    ValuationContext,
    Experience,
)
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)


# Standard valuation models
VALUATION_MODELS = ["dcf", "pe", "ps", "ev_ebitda", "pb", "ggm"]


class RLPolicy(ABC):
    """
    Abstract base class for RL policies.

    A policy maps context (state) to actions (model weights).
    It can be updated based on observed rewards.

    Subclasses must implement:
    - predict(): Get model weights for a context
    - update(): Update policy based on reward signal
    - save(): Persist policy state
    - load(): Load policy state
    """

    def __init__(
        self,
        name: str = "base_policy",
        version: str = "1.0",
        model_names: Optional[List[str]] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ):
        """
        Initialize base policy.

        Args:
            name: Policy name for identification.
            version: Policy version string.
            model_names: List of valuation model names.
            normalizer: Feature normalizer for preprocessing.
        """
        self.name = name
        self.version = version
        self.model_names = model_names or VALUATION_MODELS
        self.normalizer = normalizer
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._update_count = 0
        self._ready = False

    @abstractmethod
    def predict(
        self,
        context: ValuationContext,
    ) -> Dict[str, float]:
        """
        Predict optimal model weights given context.

        Args:
            context: ValuationContext with all features.

        Returns:
            Dict mapping model names to weights (should sum to 100).
            Example: {"dcf": 40, "pe": 30, "ps": 30}
        """
        pass

    @abstractmethod
    def update(
        self,
        context: ValuationContext,
        action: Dict[str, float],
        reward: float,
    ) -> None:
        """
        Update policy based on observed reward.

        Args:
            context: ValuationContext when prediction was made.
            action: Model weights that were used.
            reward: Observed reward signal (-1 to 1).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Persist policy state to file.

        Args:
            path: File path to save to.

        Returns:
            True if successful.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load policy state from file.

        Args:
            path: File path to load from.

        Returns:
            True if successful.
        """
        pass

    def batch_update(
        self,
        experiences: List[Experience],
    ) -> int:
        """
        Update policy with batch of experiences.

        Default implementation calls update() for each experience.
        Subclasses may override for more efficient batch updates.

        Args:
            experiences: List of (context, action, reward) experiences.

        Returns:
            Number of experiences processed.
        """
        processed = 0
        for exp in experiences:
            if exp.reward.primary_reward is not None:
                self.update(
                    context=exp.context,
                    action=exp.weights_used,
                    reward=exp.reward.primary_reward,
                )
                processed += 1
        return processed

    def predict_with_confidence(
        self,
        context: ValuationContext,
    ) -> Tuple[Dict[str, float], float]:
        """
        Predict weights with confidence estimate.

        Default implementation returns weights with fixed confidence.
        Subclasses may override to provide actual uncertainty estimates.

        Args:
            context: ValuationContext with features.

        Returns:
            Tuple of (weights_dict, confidence).
            Confidence is 0-1 representing prediction certainty.
        """
        weights = self.predict(context)
        confidence = 0.5  # Default confidence
        return weights, confidence

    def get_exploration_bonus(
        self,
        context: ValuationContext,
    ) -> Dict[str, float]:
        """
        Get exploration bonus for each model.

        Used for exploration-exploitation tradeoff.
        Default returns zeros (no exploration bonus).

        Args:
            context: ValuationContext with features.

        Returns:
            Dict mapping model names to exploration bonuses.
        """
        return {model: 0.0 for model in self.model_names}

    def is_ready(self) -> bool:
        """
        Check if policy is ready for predictions.

        Returns:
            True if policy can make predictions.
        """
        return self._ready

    def get_state(self) -> Dict[str, Any]:
        """
        Get policy state for inspection/debugging.

        Returns:
            Dict with policy state information.
        """
        return {
            "name": self.name,
            "version": self.version,
            "model_names": self.model_names,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "update_count": self._update_count,
            "ready": self._ready,
        }

    def normalize_weights(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize weights to sum to 100.

        Args:
            weights: Unnormalized weights.

        Returns:
            Normalized weights summing to 100.
        """
        total = sum(w for w in weights.values() if w > 0)
        if total <= 0:
            # Return equal weights if all zero
            n = len(self.model_names)
            return {m: 100 / n for m in self.model_names}

        return {m: (w / total) * 100 for m, w in weights.items()}

    def apply_applicability_mask(
        self,
        weights: Dict[str, float],
        context: ValuationContext,
    ) -> Dict[str, float]:
        """
        Zero out weights for inapplicable models.

        Args:
            weights: Original weights.
            context: Context with applicability flags.

        Returns:
            Weights with inapplicable models zeroed.
        """
        # Handle both ValuationContext objects and dicts
        if isinstance(context, dict):
            applicability = {
                "dcf": context.get("dcf_applicable", True),
                "pe": context.get("pe_applicable", True),
                "ps": context.get("ps_applicable", True),
                "ev_ebitda": context.get("evebitda_applicable", True),
                "pb": context.get("pb_applicable", True),
                "ggm": context.get("ggm_applicable", False),
            }
        else:
            applicability = {
                "dcf": context.dcf_applicable,
                "pe": context.pe_applicable,
                "ps": context.ps_applicable,
                "ev_ebitda": context.evebitda_applicable,
                "pb": context.pb_applicable,
                "ggm": context.ggm_applicable,
            }

        masked = {}
        for model, weight in weights.items():
            if applicability.get(model, True):
                masked[model] = weight
            else:
                masked[model] = 0.0

        return self.normalize_weights(masked)

    def _extract_features(
        self,
        context: ValuationContext,
    ) -> np.ndarray:
        """
        Extract normalized feature vector from context.

        Args:
            context: ValuationContext to process.

        Returns:
            Normalized feature array.
        """
        if self.normalizer and self.normalizer.is_fitted:
            return self.normalizer.transform(context)
        else:
            # Fall back to extractor's raw tensor
            from investigator.domain.services.rl.feature_extractor import (
                ValuationContextExtractor,
            )

            extractor = ValuationContextExtractor()
            return extractor.to_tensor(context)


class UniformPolicy(RLPolicy):
    """
    Simple baseline policy that returns equal weights.

    Useful for testing and as a baseline for comparison.
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
    ):
        super().__init__(
            name="uniform_policy",
            version="1.0",
            model_names=model_names,
        )
        self._ready = True

    def predict(self, context: ValuationContext) -> Dict[str, float]:
        """Return equal weights for all applicable models."""
        weights = {m: 100 / len(self.model_names) for m in self.model_names}
        return self.apply_applicability_mask(weights, context)

    def update(
        self,
        context: ValuationContext,
        action: Dict[str, float],
        reward: float,
    ) -> None:
        """No-op for uniform policy (no learning)."""
        pass

    def save(self, path: str) -> bool:
        """Save policy (minimal state for uniform)."""
        import pickle
        import os

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"name": self.name, "version": self.version}, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save uniform policy: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load policy state."""
        import pickle

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.name = state.get("name", self.name)
            self.version = state.get("version", self.version)
            self._ready = True
            return True
        except Exception as e:
            logger.error(f"Failed to load uniform policy: {e}")
            return False
