"""
Contextual Bandit Policy with Thompson Sampling

Implements a contextual multi-armed bandit for model weight selection.
Uses Thompson Sampling for exploration/exploitation balance.

Good for:
- Exploration/exploitation balance out of the box
- Uncertainty quantification (knows when it doesn't know)
- Works well with limited data
- Interpretable (linear model per arm)

Architecture:
- Each "arm" is a tier classification (or direct weight vector)
- Features are the ValuationContext
- Uses Bayesian linear regression for reward estimation
- Thompson Sampling for action selection

Usage:
    from investigator.domain.services.rl.policy import ContextualBanditPolicy

    policy = ContextualBanditPolicy(n_features=50)

    # Predict weights
    weights = policy.predict(context)

    # Update after outcome known
    policy.update(context, weights, reward=0.75)
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg

from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import ValuationContext
from investigator.domain.services.rl.policy.base import VALUATION_MODELS, RLPolicy

logger = logging.getLogger(__name__)


# Tier classifications that map to weight templates
TIER_CLASSIFICATIONS = [
    "pre_profit_negative_ebitda",
    "pre_profit_positive_ebitda",
    "pre_profit_high_growth",
    "dividend_aristocrat_pure",
    "dividend_aristocrat_hybrid",
    "dividend_high_yield",
    "high_growth_saas",
    "high_growth_emerging",
    "high_growth_established",
    "transitioning_to_profit",
    "transitioning_to_mature",
    "mature_value",
    "mature_garp",
    "mature_quality",
    "balanced_default",
]


# Weight templates for each tier (maps tier to model weights)
TIER_WEIGHT_TEMPLATES = {
    "pre_profit_negative_ebitda": {"dcf": 0, "pe": 0, "ps": 70, "ev_ebitda": 0, "pb": 30, "ggm": 0},
    "pre_profit_positive_ebitda": {"dcf": 0, "pe": 0, "ps": 50, "ev_ebitda": 30, "pb": 20, "ggm": 0},
    "pre_profit_high_growth": {"dcf": 10, "pe": 0, "ps": 60, "ev_ebitda": 20, "pb": 10, "ggm": 0},
    "dividend_aristocrat_pure": {"dcf": 25, "pe": 25, "ps": 10, "ev_ebitda": 15, "pb": 5, "ggm": 20},
    "dividend_aristocrat_hybrid": {"dcf": 30, "pe": 25, "ps": 10, "ev_ebitda": 20, "pb": 5, "ggm": 10},
    "dividend_high_yield": {"dcf": 20, "pe": 20, "ps": 15, "ev_ebitda": 20, "pb": 10, "ggm": 15},
    "high_growth_saas": {"dcf": 20, "pe": 15, "ps": 40, "ev_ebitda": 15, "pb": 10, "ggm": 0},
    "high_growth_emerging": {"dcf": 15, "pe": 20, "ps": 35, "ev_ebitda": 20, "pb": 10, "ggm": 0},
    "high_growth_established": {"dcf": 25, "pe": 25, "ps": 25, "ev_ebitda": 20, "pb": 5, "ggm": 0},
    "transitioning_to_profit": {"dcf": 20, "pe": 25, "ps": 30, "ev_ebitda": 20, "pb": 5, "ggm": 0},
    "transitioning_to_mature": {"dcf": 30, "pe": 30, "ps": 15, "ev_ebitda": 20, "pb": 5, "ggm": 0},
    "mature_value": {"dcf": 35, "pe": 30, "ps": 10, "ev_ebitda": 20, "pb": 5, "ggm": 0},
    "mature_garp": {"dcf": 30, "pe": 30, "ps": 15, "ev_ebitda": 20, "pb": 5, "ggm": 0},
    "mature_quality": {"dcf": 30, "pe": 25, "ps": 15, "ev_ebitda": 25, "pb": 5, "ggm": 0},
    "balanced_default": {"dcf": 25, "pe": 25, "ps": 20, "ev_ebitda": 25, "pb": 5, "ggm": 0},
}


class ContextualBanditPolicy(RLPolicy):
    """
    Contextual Bandit with Thompson Sampling.

    Uses Bayesian linear regression to estimate expected reward for each
    tier classification given the context. Thompson Sampling samples from
    posterior to select action.

    The policy learns which tier works best for different company contexts.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        n_actions: int = len(TIER_CLASSIFICATIONS),
        prior_variance: float = 1.0,
        noise_variance: float = 0.1,
        exploration_weight: float = 1.0,
        normalizer: Optional[FeatureNormalizer] = None,
    ):
        """
        Initialize contextual bandit policy.

        Args:
            n_features: Feature dimension (auto-detected if not provided).
            n_actions: Number of tier classifications.
            prior_variance: Prior variance for Bayesian regression.
            noise_variance: Observation noise variance.
            exploration_weight: Scaling factor for exploration.
            normalizer: Feature normalizer.
        """
        super().__init__(
            name="contextual_bandit",
            version="1.0",
            normalizer=normalizer,
        )

        # Will be set on first observation if n_features is None
        self.n_features = n_features
        self.n_actions = n_actions
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self.exploration_weight = exploration_weight

        # Bayesian linear regression parameters for each action
        # For each action a: y = x^T theta_a + noise
        # Posterior: theta_a ~ N(mu_a, Sigma_a)
        self._initialized = False
        self._mu: Optional[np.ndarray] = None  # Shape: (n_actions, n_features)
        self._Sigma: Optional[np.ndarray] = None  # Shape: (n_actions, n_features, n_features)
        self._Lambda: Optional[np.ndarray] = None  # Precision matrices

        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)

        if n_features is not None:
            self._initialize_parameters(n_features)

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize Bayesian regression parameters."""
        self.n_features = n_features

        # Initialize prior: theta ~ N(0, prior_variance * I)
        self._mu = np.zeros((self.n_actions, n_features))

        # Precision matrix (inverse covariance)
        # Lambda = (1/prior_variance) * I
        prior_precision = 1.0 / self.prior_variance
        self._Lambda = np.array([np.eye(n_features) * prior_precision for _ in range(self.n_actions)])

        # Covariance is inverse of precision
        self._Sigma = np.array([np.eye(n_features) * self.prior_variance for _ in range(self.n_actions)])

        self._initialized = True
        self._ready = True
        logger.info(f"Initialized bandit with {n_features} features, {self.n_actions} actions")

    def predict(self, context: ValuationContext) -> Dict[str, float]:
        """
        Predict model weights using Thompson Sampling.

        1. Extract features from context
        2. Sample theta from posterior for each action
        3. Compute expected reward for each action
        4. Select action with highest sampled reward
        5. Return weight template for selected action
        """
        features = self._extract_features(context)

        # Initialize on first prediction if needed
        if not self._initialized:
            self._initialize_parameters(len(features))

        # Thompson Sampling: sample theta from posterior for each action
        sampled_rewards = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            # Sample theta_a ~ N(mu_a, Sigma_a)
            try:
                theta_sample = np.random.multivariate_normal(self._mu[a], self._Sigma[a] * self.exploration_weight)
            except np.linalg.LinAlgError:
                # Fallback if covariance is singular
                theta_sample = self._mu[a] + np.random.randn(self.n_features) * 0.1

            # Expected reward = x^T theta
            sampled_rewards[a] = np.dot(features, theta_sample)

        # Select action with highest sampled reward
        best_action = np.argmax(sampled_rewards)
        tier = TIER_CLASSIFICATIONS[best_action]

        # Get weight template for this tier
        weights = TIER_WEIGHT_TEMPLATES.get(tier, TIER_WEIGHT_TEMPLATES["balanced_default"])

        # Apply applicability mask
        weights = self.apply_applicability_mask(dict(weights), context)

        return weights

    def predict_with_confidence(
        self,
        context: ValuationContext,
    ) -> Tuple[Dict[str, float], float]:
        """
        Predict with confidence estimate based on posterior uncertainty.
        """
        features = self._extract_features(context)

        if not self._initialized:
            self._initialize_parameters(len(features))

        # Compute expected reward and uncertainty for each action
        expected_rewards = np.zeros(self.n_actions)
        uncertainties = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            expected_rewards[a] = np.dot(features, self._mu[a])
            # Uncertainty: sqrt(x^T Sigma x)
            uncertainties[a] = np.sqrt(np.dot(features, np.dot(self._Sigma[a], features)))

        best_action = np.argmax(expected_rewards)
        tier = TIER_CLASSIFICATIONS[best_action]
        weights = TIER_WEIGHT_TEMPLATES.get(tier, TIER_WEIGHT_TEMPLATES["balanced_default"])
        weights = self.apply_applicability_mask(dict(weights), context)

        # Confidence: inverse of uncertainty (normalized)
        avg_uncertainty = np.mean(uncertainties)
        confidence = 1.0 / (1.0 + avg_uncertainty)

        return weights, confidence

    def update(
        self,
        context: ValuationContext,
        action: Dict[str, float],
        reward: float,
    ) -> None:
        """
        Update policy using Bayesian linear regression update.

        Args:
            context: Context when prediction was made.
            action: Weights that were used (used to identify tier).
            reward: Observed reward (-1 to 1).
        """
        features = self._extract_features(context)

        if not self._initialized:
            self._initialize_parameters(len(features))

        # Identify which action was taken based on weights
        action_idx = self._identify_action(action)

        # Bayesian update for action a:
        # Lambda_new = Lambda + (1/noise_var) * x x^T
        # mu_new = Sigma_new @ (Lambda @ mu + (1/noise_var) * x * r)
        noise_precision = 1.0 / self.noise_variance

        # Update precision matrix
        outer_prod = np.outer(features, features)
        self._Lambda[action_idx] += noise_precision * outer_prod

        # Update covariance (inverse of precision)
        try:
            self._Sigma[action_idx] = linalg.inv(self._Lambda[action_idx])
        except linalg.LinAlgError:
            # Add small regularization if singular
            self._Lambda[action_idx] += np.eye(self.n_features) * 1e-6
            self._Sigma[action_idx] = linalg.inv(self._Lambda[action_idx])

        # Update mean
        old_precision_mean = np.dot(self._Lambda[action_idx] - noise_precision * outer_prod, self._mu[action_idx])
        self._mu[action_idx] = np.dot(self._Sigma[action_idx], old_precision_mean + noise_precision * features * reward)

        # Track statistics
        self.action_counts[action_idx] += 1
        self.action_rewards[action_idx] += reward
        self._update_count += 1
        self._updated_at = datetime.now()

    def _identify_action(self, weights: Dict[str, float]) -> int:
        """
        Identify which tier action corresponds to the given weights.

        Uses cosine similarity to find closest tier.
        """
        # Convert weights to vector
        weight_vec = np.array([weights.get(m, 0) for m in VALUATION_MODELS])
        weight_vec = weight_vec / (np.linalg.norm(weight_vec) + 1e-8)

        best_similarity = -1
        best_action = len(TIER_CLASSIFICATIONS) - 1  # Default to balanced

        for i, tier in enumerate(TIER_CLASSIFICATIONS):
            template = TIER_WEIGHT_TEMPLATES[tier]
            template_vec = np.array([template.get(m, 0) for m in VALUATION_MODELS])
            template_vec = template_vec / (np.linalg.norm(template_vec) + 1e-8)

            similarity = np.dot(weight_vec, template_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = i

        return best_action

    def get_exploration_bonus(
        self,
        context: ValuationContext,
    ) -> Dict[str, float]:
        """
        Get exploration bonus based on posterior uncertainty.
        """
        if not self._initialized:
            return {m: 1.0 for m in self.model_names}

        features = self._extract_features(context)

        # Compute uncertainty for each action
        bonuses = {}
        for a, tier in enumerate(TIER_CLASSIFICATIONS):
            uncertainty = np.sqrt(np.dot(features, np.dot(self._Sigma[a], features)))
            # Map uncertainty to model bonuses based on tier weights
            template = TIER_WEIGHT_TEMPLATES[tier]
            for model in self.model_names:
                model_bonus = uncertainty * template.get(model, 0) / 100
                if model not in bonuses:
                    bonuses[model] = 0
                bonuses[model] += model_bonus / self.n_actions

        return bonuses

    def save(self, path: str) -> bool:
        """Save policy state to file."""
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
                "mu": self._mu,
                "Lambda": self._Lambda,
                "Sigma": self._Sigma,
                "action_counts": self.action_counts,
                "action_rewards": self.action_rewards,
                "update_count": self._update_count,
                "created_at": self._created_at.isoformat(),
                "updated_at": self._updated_at.isoformat(),
            }

            with open(path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved contextual bandit policy to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save policy: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load policy state from file."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.name = state.get("name", self.name)
            self.version = state.get("version", self.version)
            self.n_features = state["n_features"]
            self.n_actions = state["n_actions"]
            self.prior_variance = state.get("prior_variance", self.prior_variance)
            self.noise_variance = state.get("noise_variance", self.noise_variance)
            self.exploration_weight = state.get("exploration_weight", self.exploration_weight)
            self._mu = state["mu"]
            self._Lambda = state["Lambda"]
            self._Sigma = state["Sigma"]
            self.action_counts = state.get("action_counts", np.zeros(self.n_actions))
            self.action_rewards = state.get("action_rewards", np.zeros(self.n_actions))
            self._update_count = state.get("update_count", 0)

            if "created_at" in state:
                self._created_at = datetime.fromisoformat(state["created_at"])
            if "updated_at" in state:
                self._updated_at = datetime.fromisoformat(state["updated_at"])

            self._initialized = True
            self._ready = True
            logger.info(f"Loaded contextual bandit policy from {path}")
            return True

        except FileNotFoundError:
            logger.warning(f"Policy file not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get policy state for inspection."""
        state = super().get_state()
        state.update(
            {
                "n_features": self.n_features,
                "n_actions": self.n_actions,
                "action_counts": self.action_counts.tolist() if self.action_counts is not None else [],
                "action_rewards": self.action_rewards.tolist() if self.action_rewards is not None else [],
                "total_updates": self._update_count,
            }
        )
        return state

    def get_action_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each action."""
        stats = {}
        for i, tier in enumerate(TIER_CLASSIFICATIONS):
            count = self.action_counts[i]
            stats[tier] = {
                "count": int(count),
                "total_reward": float(self.action_rewards[i]),
                "avg_reward": float(self.action_rewards[i] / count) if count > 0 else 0.0,
            }
        return stats
