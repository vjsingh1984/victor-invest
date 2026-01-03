"""
Dual RL Policy - Combines Technical and Fundamental Policies

This composite policy separates concerns:
1. TechnicalRLPolicy: Handles TIMING (when to trade, position signals)
2. FundamentalRLPolicy: Handles WEIGHTS (which models to use) and HOLDING PERIOD

The dual approach allows each policy to specialize:
- Technical policy learns from price action, momentum, volatility
- Fundamental policy learns from financial metrics, sector patterns

Usage:
    from investigator.domain.services.rl.policy import DualRLPolicy

    policy = DualRLPolicy()

    # Get complete prediction
    result = policy.predict_full(context)
    # result = {
    #     "position": 1,  # Long
    #     "position_confidence": 0.75,
    #     "weights": {"dcf": 30.0, "pe": 25.0, ...},
    #     "holding_period": "6m",
    # }

    # Update both policies with observed reward
    policy.update(context, result, reward, holding_reward_by_period)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from investigator.domain.services.rl.models import ValuationContext
from investigator.domain.services.rl.policy.technical_policy import TechnicalRLPolicy
from investigator.domain.services.rl.policy.fundamental_policy import FundamentalRLPolicy
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)


class DualRLPolicy:
    """
    Dual RL Policy combining technical timing and fundamental analysis.

    Separation of concerns:
    - Technical Policy: Position signal (Long/Short/Skip), entry/exit timing
    - Fundamental Policy: Model weights, holding period optimization
    """

    def __init__(
        self,
        base_weighting_service: Optional[Any] = None,
        technical_policy: Optional[TechnicalRLPolicy] = None,
        fundamental_policy: Optional[FundamentalRLPolicy] = None,
        technical_path: str = "data/rl_models/technical_policy.pkl",
        fundamental_path: str = "data/rl_models/fundamental_policy.pkl",
    ):
        """
        Initialize dual policy.

        Args:
            base_weighting_service: Rule-based weighting service for fundamentals
            technical_policy: Pre-initialized technical policy (or None to create new)
            fundamental_policy: Pre-initialized fundamental policy (or None to create new)
            technical_path: Path to load/save technical policy
            fundamental_path: Path to load/save fundamental policy
        """
        self.technical_path = technical_path
        self.fundamental_path = fundamental_path

        # Initialize or use provided policies
        self.technical = technical_policy or TechnicalRLPolicy()
        self.fundamental = fundamental_policy or FundamentalRLPolicy(
            base_weighting_service=base_weighting_service
        )

        # Try to load existing policies
        self._load_policies()

        self._created_at = datetime.now()
        self._updated_at = datetime.now()

    def _load_policies(self) -> None:
        """Load policies from disk if available."""
        if os.path.exists(self.technical_path):
            self.technical.load(self.technical_path)
            logger.info(f"Loaded technical policy: {self.technical._update_count} updates")

        if os.path.exists(self.fundamental_path):
            self.fundamental.load(self.fundamental_path)
            logger.info(f"Loaded fundamental policy: {self.fundamental._update_count} updates")

    def predict_position(self, context: ValuationContext) -> Tuple[int, float]:
        """
        Predict position signal using technical policy.

        Returns:
            Tuple of (position_signal, confidence)
            position_signal: 1 (Long), -1 (Short), 0 (Skip)
        """
        return self.technical.predict_position(context)

    def predict_weights(self, context: ValuationContext) -> Dict[str, float]:
        """
        Predict model weights using fundamental policy.

        Returns:
            Dict of model weights summing to 100%
        """
        return self.fundamental.predict(context)

    def predict_holding_period(self, context: ValuationContext) -> str:
        """
        Predict optimal holding period using fundamental policy.

        Returns:
            Holding period string (1m, 3m, 6m, 12m, 18m, 24m, 36m)
        """
        return self.fundamental.predict_holding_period(context)

    def predict_full(self, context: ValuationContext) -> Dict[str, Any]:
        """
        Get complete prediction from both policies.

        Returns:
            Dict with:
            - position: int (1=Long, -1=Short, 0=Skip)
            - position_confidence: float (0-1)
            - position_probs: Dict[str, float] (skip/long/short probabilities)
            - weights: Dict[str, float] (model weights)
            - holding_period: str (recommended holding period)
        """
        # Technical policy: position and timing
        position, confidence = self.technical.predict_position(context)
        position_probs = self.technical.predict(context)

        # Fundamental policy: weights and holding period
        weights = self.fundamental.predict(context)
        holding_period = self.fundamental.predict_holding_period(context)

        return {
            "position": position,
            "position_confidence": confidence,
            "position_probs": position_probs,
            "weights": weights,
            "holding_period": holding_period,
        }

    def update(
        self,
        context: ValuationContext,
        prediction: Dict[str, Any],
        position_reward: float,
        holding_period_rewards: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update both policies based on observed rewards.

        Args:
            context: The context when prediction was made
            prediction: The prediction that was made (from predict_full)
            position_reward: Reward for the position decision (-1 to 1)
            holding_period_rewards: Dict mapping holding periods to rewards
        """
        # Update technical policy with position reward
        position = prediction.get("position", 0)
        # Convert position to action index: 0=skip, 1=long, 2=short
        if position == 0:
            action = 0
        elif position == 1:
            action = 1
        else:
            action = 2

        self.technical.update(context, action, position_reward)

        # Update fundamental policy with weights reward
        weights = prediction.get("weights", {})
        self.fundamental.update_weights(context, weights, position_reward)

        # Update holding period learning if we have period-specific rewards
        if holding_period_rewards:
            for period, reward in holding_period_rewards.items():
                if reward is not None:
                    self.fundamental.update_holding_period(context, period, reward)

        self._updated_at = datetime.now()

    def update_position(
        self,
        context: ValuationContext,
        position: int,
        reward: float,
    ) -> None:
        """Update only the technical (position) policy."""
        if position == 0:
            action = 0
        elif position == 1:
            action = 1
        else:
            action = 2
        self.technical.update(context, action, reward)

    def update_weights(
        self,
        context: ValuationContext,
        weights: Dict[str, float],
        reward: float,
    ) -> None:
        """Update only the fundamental (weights) policy."""
        self.fundamental.update_weights(context, weights, reward)

    def update_holding_period(
        self,
        context: ValuationContext,
        period: str,
        reward: float,
    ) -> None:
        """Update only the holding period learning."""
        self.fundamental.update_holding_period(context, period, reward)

    def save(self) -> bool:
        """Save both policies to disk."""
        os.makedirs(os.path.dirname(self.technical_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.fundamental_path) or ".", exist_ok=True)

        tech_ok = self.technical.save(self.technical_path)
        fund_ok = self.fundamental.save(self.fundamental_path)

        if tech_ok and fund_ok:
            logger.info("Saved dual policy successfully")
            return True
        return False

    def load(self) -> bool:
        """Load both policies from disk."""
        tech_ok = self.technical.load(self.technical_path)
        fund_ok = self.fundamental.load(self.fundamental_path)
        return tech_ok and fund_ok

    def get_state(self) -> Dict[str, Any]:
        """Get combined state from both policies."""
        return {
            "technical": self.technical.get_state(),
            "fundamental": self.fundamental.get_state(),
            "technical_updates": self.technical._update_count,
            "fundamental_updates": self.fundamental._update_count,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
        }

    def get_action_stats(self) -> Dict[str, Any]:
        """Get action statistics from technical policy."""
        return self.technical.get_action_stats()

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics from fundamental policy."""
        return self.fundamental.get_model_stats()

    def get_holding_period_stats(self) -> Dict[str, Any]:
        """Get holding period statistics from fundamental policy."""
        return self.fundamental.get_holding_period_stats()

    def is_ready(self) -> bool:
        """Check if both policies are ready."""
        return self.technical._ready and self.fundamental._ready


def load_dual_policy(
    technical_path: str = "data/rl_models/technical_policy.pkl",
    fundamental_path: str = "data/rl_models/fundamental_policy.pkl",
    base_weighting_service: Optional[Any] = None,
) -> DualRLPolicy:
    """
    Load or create a dual RL policy.

    Args:
        technical_path: Path to technical policy file
        fundamental_path: Path to fundamental policy file
        base_weighting_service: Optional rule-based weighting service

    Returns:
        Initialized DualRLPolicy
    """
    policy = DualRLPolicy(
        base_weighting_service=base_weighting_service,
        technical_path=technical_path,
        fundamental_path=fundamental_path,
    )
    return policy
