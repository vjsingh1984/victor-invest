"""
Technical RL Policy - Timing and Position Decisions

This policy focuses on WHEN to trade, not WHAT weights to use.
It learns from technical indicators to make:
1. Position decisions (Long/Short/Skip)
2. Entry timing signals
3. Exit timing signals

Features used:
- Technical indicators (RSI, MACD, OBV, ADX, Stochastic, MFI)
- Trend signals (technical_trend, market_sentiment)
- Volatility measures
- Entry/exit signal features
- Valuation gap and confidence

Output:
- Position signal: 1 (Long), -1 (Short), 0 (Skip)
- Entry confidence: 0-1 score
- Exit urgency: 0-1 score
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import ValuationContext
from investigator.domain.services.rl.policy.base import RLPolicy

logger = logging.getLogger(__name__)

# Technical features used by this policy
TECHNICAL_FEATURES = [
    "technical_trend",
    "market_sentiment",
    "volatility",
    "rsi_14",
    "macd_histogram",
    "obv_trend",
    "adx_14",
    "stoch_k",
    "mfi_14",
    "entry_signal_strength",
    "exit_signal_strength",
    "signal_confluence",
    "days_from_support",
    "risk_reward_ratio",
    "valuation_gap",
    "valuation_confidence",
]

# Position actions: Skip, Long, Short
POSITION_ACTIONS = ["skip", "long", "short"]


class TechnicalRLPolicy(RLPolicy):
    """
    RL Policy for timing and position decisions based on technical analysis.

    Uses a contextual bandit approach to learn:
    - When to go Long (price expected to rise)
    - When to go Short (price expected to fall)
    - When to Skip (unclear signal, avoid trade)
    """

    def __init__(
        self,
        normalizer: Optional[FeatureNormalizer] = None,
        prior_variance: float = 1.0,
        noise_variance: float = 0.1,
        exploration_weight: float = 0.5,
        min_gap_for_signal: float = 0.05,
        min_gap_for_short: float = 0.15,  # Asymmetric: require larger gap for shorts
        min_confidence: float = 0.60,
        high_confidence_penalty: float = 0.15,  # Penalize overconfident predictions
        high_confidence_threshold: float = 0.80,  # Confidence level to start penalizing
    ):
        """
        Initialize technical policy.

        Args:
            normalizer: Feature normalizer for technical features
            prior_variance: Prior variance for Bayesian updates
            noise_variance: Observation noise variance
            exploration_weight: UCB exploration parameter
            min_gap_for_signal: Minimum valuation gap to trigger LONG position (5%)
            min_gap_for_short: Minimum valuation gap to trigger SHORT position (15%)
            min_confidence: Minimum confidence to trigger position (60%)
            high_confidence_penalty: Penalty for predictions above threshold
            high_confidence_threshold: Confidence level above which to apply penalty
        """
        super().__init__(
            name="technical_policy",
            version="1.1",  # Version bump for asymmetric signals
            model_names=POSITION_ACTIONS,
            normalizer=normalizer,
        )

        self.n_features = len(TECHNICAL_FEATURES)
        self.n_actions = len(POSITION_ACTIONS)
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self.exploration_weight = exploration_weight
        self.min_gap_for_signal = min_gap_for_signal
        self.min_gap_for_short = min_gap_for_short  # Higher bar for shorts
        self.min_confidence = min_confidence
        self.high_confidence_penalty = high_confidence_penalty
        self.high_confidence_threshold = high_confidence_threshold

        # Bayesian linear regression parameters for each action
        # mu: mean of weight vector
        # Lambda: precision matrix (inverse covariance)
        self.mu = np.zeros((self.n_actions, self.n_features))
        self.Lambda = np.array([np.eye(self.n_features) / prior_variance for _ in range(self.n_actions)])
        self.Sigma = np.array([np.eye(self.n_features) * prior_variance for _ in range(self.n_actions)])

        # Action statistics
        self.action_counts = np.zeros(self.n_actions)
        self.action_rewards = np.zeros(self.n_actions)

        # Per-context tracking for learning
        self._position_history: List[Dict] = []

        self._ready = True

    def _extract_features(self, context: ValuationContext) -> np.ndarray:
        """Extract technical features from context."""
        if isinstance(context, dict):
            features = np.array(
                [
                    context.get("technical_trend", 0.0),
                    context.get("market_sentiment", 0.0),
                    context.get("volatility", 0.5),
                    context.get("rsi_14", 50.0) / 100.0,  # Normalize to 0-1
                    context.get("macd_histogram", 0.0),
                    context.get("obv_trend", 0.0),
                    context.get("adx_14", 25.0) / 100.0,  # Normalize to 0-1
                    context.get("stoch_k", 50.0) / 100.0,  # Normalize to 0-1
                    context.get("mfi_14", 50.0) / 100.0,  # Normalize to 0-1
                    context.get("entry_signal_strength", 0.0),
                    context.get("exit_signal_strength", 0.0),
                    context.get("signal_confluence", 0.0),
                    context.get("days_from_support", 0.5),
                    context.get("risk_reward_ratio", 2.0) / 5.0,  # Normalize
                    context.get("valuation_gap", 0.0),
                    context.get("valuation_confidence", 0.5),
                ]
            )
        else:
            features = np.array(
                [
                    context.technical_trend,
                    context.market_sentiment,
                    context.volatility,
                    context.rsi_14 / 100.0,
                    context.macd_histogram,
                    context.obv_trend,
                    context.adx_14 / 100.0,
                    context.stoch_k / 100.0,
                    context.mfi_14 / 100.0,
                    context.entry_signal_strength,
                    context.exit_signal_strength,
                    context.signal_confluence,
                    context.days_from_support,
                    context.risk_reward_ratio / 5.0,
                    context.valuation_gap,
                    context.valuation_confidence,
                ]
            )

        return features

    def predict(self, context: ValuationContext) -> Dict[str, float]:
        """
        Predict position probabilities.

        Returns dict with probabilities for each position:
        {"skip": 0.3, "long": 0.5, "short": 0.2}
        """
        features = self._extract_features(context)

        # Calculate expected reward for each action using Thompson Sampling
        action_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            # Mean prediction
            mean = np.dot(self.mu[a], features)
            # Uncertainty (for exploration)
            var = np.dot(features, np.dot(self.Sigma[a], features))
            std = np.sqrt(max(var, 1e-6))
            # UCB-style value
            action_values[a] = mean + self.exploration_weight * std

        # Convert to probabilities via softmax
        exp_values = np.exp(action_values - np.max(action_values))
        probs = exp_values / exp_values.sum()

        return {
            "skip": float(probs[0]),
            "long": float(probs[1]),
            "short": float(probs[2]),
        }

    def predict_position(self, context: ValuationContext) -> Tuple[int, float]:
        """
        Predict optimal position signal with asymmetric thresholds.

        Key adjustments based on backtest analysis:
        1. ASYMMETRIC GAP: Require larger gap for shorts (15%) vs longs (5%)
        2. CONFIDENCE INVERSION: Penalize overconfident predictions (>80%)
        3. Conservative short bias: More skeptical of short signals

        Returns:
            Tuple of (position_signal, confidence)
            position_signal: 1 (Long), -1 (Short), 0 (Skip)
            confidence: 0-1 confidence in the decision
        """
        probs = self.predict(context)

        # Get valuation gap and confidence for rule-based filtering
        if isinstance(context, dict):
            val_gap = context.get("valuation_gap", 0.0)
            val_conf = context.get("valuation_confidence", 0.5)
            sector = context.get("sector", "")
            industry = context.get("industry", "")
        else:
            val_gap = getattr(context, "valuation_gap", 0.0)
            val_conf = getattr(context, "valuation_confidence", 0.5)
            sector = getattr(context, "sector", "")
            industry = getattr(context, "industry", "")

        # Semiconductor penalty - skip volatile sectors
        is_semiconductor = (
            "semiconductor" in industry.lower() or "chip" in industry.lower() or sector.lower() == "semiconductors"
        )
        if is_semiconductor:
            # Semiconductors have -0.13 avg reward - be very conservative
            return 0, probs["skip"]

        # ASYMMETRIC GAP THRESHOLDS
        # Longs: 5% gap is enough (they perform well)
        # Shorts: Require 15% gap (they underperform badly)
        if val_gap > 0 and val_gap < self.min_gap_for_signal:
            return 0, probs["skip"]  # Gap too small for long
        if val_gap < 0 and abs(val_gap) < self.min_gap_for_short:
            return 0, probs["skip"]  # Gap too small for short (higher bar)

        if val_conf < self.min_confidence:
            return 0, probs["skip"]  # Models disagree

        # RL-based decision
        best_action = max(probs, key=probs.get)
        confidence = probs[best_action]

        # CONFIDENCE INVERSION: Penalize overconfident predictions
        # Backtest showed: higher confidence = worse performance
        if confidence > self.high_confidence_threshold:
            # Apply penalty to confidence, making it less likely to trigger
            adjusted_confidence = confidence - self.high_confidence_penalty
            if adjusted_confidence < self.min_confidence:
                return 0, probs["skip"]  # Too confident = skip
            confidence = adjusted_confidence

        # Additional short signal skepticism
        # Shorts have -0.1471 avg reward vs +0.2347 for longs
        if best_action == "short":
            # Reduce short probability by 20% (shift toward skip)
            probs["short"] *= 0.8
            probs["skip"] += probs["short"] * 0.2
            # Recalculate best action
            best_action = max(probs, key=probs.get)
            confidence = probs[best_action]

        if best_action == "long" and val_gap > 0:
            return 1, confidence
        elif best_action == "short" and val_gap < 0:
            return -1, confidence
        else:
            # RL suggests opposite of valuation gap - skip to be safe
            return 0, probs["skip"]

    def update(
        self,
        context: ValuationContext,
        action: int,  # 0=skip, 1=long, 2=short
        reward: float,
    ) -> None:
        """
        Update policy based on observed reward.

        Args:
            context: The context when decision was made
            action: Action taken (0=skip, 1=long, 2=short)
            reward: Observed reward (-1 to 1)
        """
        features = self._extract_features(context)

        # Bayesian update for the chosen action
        a = action

        # Update precision matrix
        self.Lambda[a] += np.outer(features, features) / self.noise_variance

        # Update mean
        self.Sigma[a] = np.linalg.inv(self.Lambda[a])
        self.mu[a] = np.dot(self.Sigma[a], np.dot(self.Lambda[a], self.mu[a]) + features * reward / self.noise_variance)

        # Track statistics
        self.action_counts[a] += 1
        self.action_rewards[a] += reward

        self._update_count += 1
        self._updated_at = datetime.now()

    def get_action_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each action."""
        stats = {}
        for i, action in enumerate(POSITION_ACTIONS):
            count = self.action_counts[i]
            total_reward = self.action_rewards[i]
            stats[action] = {
                "count": int(count),
                "total_reward": float(total_reward),
                "avg_reward": float(total_reward / count) if count > 0 else 0.0,
            }
        return stats

    def save(self, path: str) -> bool:
        """Save policy state."""
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

            state = {
                "name": self.name,
                "version": self.version,
                "n_features": self.n_features,
                "n_actions": self.n_actions,
                "prior_variance": self.prior_variance,
                "noise_variance": self.noise_variance,
                "exploration_weight": self.exploration_weight,
                "min_gap_for_signal": self.min_gap_for_signal,
                "min_confidence": self.min_confidence,
                "mu": self.mu,
                "Lambda": self.Lambda,
                "Sigma": self.Sigma,
                "action_counts": self.action_counts,
                "action_rewards": self.action_rewards,
                "update_count": self._update_count,
                "created_at": self._created_at.isoformat(),
                "updated_at": self._updated_at.isoformat(),
            }

            with open(path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved technical policy to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save technical policy: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load policy state."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.name = state.get("name", self.name)
            self.version = state.get("version", self.version)
            self.n_features = state.get("n_features", self.n_features)
            self.n_actions = state.get("n_actions", self.n_actions)
            self.prior_variance = state.get("prior_variance", self.prior_variance)
            self.noise_variance = state.get("noise_variance", self.noise_variance)
            self.exploration_weight = state.get("exploration_weight", self.exploration_weight)
            self.min_gap_for_signal = state.get("min_gap_for_signal", self.min_gap_for_signal)
            self.min_confidence = state.get("min_confidence", self.min_confidence)
            self.mu = state.get("mu", self.mu)
            self.Lambda = state.get("Lambda", self.Lambda)
            self.Sigma = state.get("Sigma", self.Sigma)
            self.action_counts = state.get("action_counts", self.action_counts)
            self.action_rewards = state.get("action_rewards", self.action_rewards)
            self._update_count = state.get("update_count", 0)

            if "created_at" in state:
                self._created_at = datetime.fromisoformat(state["created_at"])
            if "updated_at" in state:
                self._updated_at = datetime.fromisoformat(state["updated_at"])

            self._ready = True
            logger.info(f"Loaded technical policy from {path}")
            return True

        except FileNotFoundError:
            logger.warning(f"Technical policy file not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load technical policy: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get policy state for inspection."""
        return {
            "name": self.name,
            "version": self.version,
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "update_count": self._update_count,
            "action_stats": self.get_action_stats(),
            "min_gap_for_signal": self.min_gap_for_signal,
            "min_confidence": self.min_confidence,
        }
