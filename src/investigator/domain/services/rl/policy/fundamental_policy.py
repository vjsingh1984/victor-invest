"""
Fundamental RL Policy - Model Weights and Holding Period Decisions

This policy focuses on WHAT weights to use and HOW LONG to hold.
It learns from fundamental metrics and sector/stage/industry to determine:
1. Valuation model weights (DCF, PE, PS, EV/EBITDA, PB, GGM)
2. Optimal holding period (1m to 36m)

Features used:
- Fundamental metrics (profitability, margins, growth rates)
- Sector classification (one-hot encoded)
- Industry classification (one-hot encoded - 60+ categories)
- Growth stage classification
- Company size classification
- Industry characteristics (volatility, cyclicality, orientation)
- Data quality metrics

Output:
- Model weights: Dict[str, float] summing to 100%
- Recommended holding period: str (1m, 3m, 6m, 12m, 18m, 24m, 36m)

Enhanced with industry-level granularity for better model weight selection.
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize
from investigator.domain.services.rl.policy.base import RLPolicy, VALUATION_MODELS
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.industry_weights import (
    IndustryCategory,
    INDUSTRY_PROFILES,
    classify_industry,
    get_industry_profile,
    get_industry_weights,
    get_industry_holding_period,
)

logger = logging.getLogger(__name__)

# Fundamental features used by this policy
FUNDAMENTAL_FEATURES = [
    "profitability_score",
    "pe_level",
    "revenue_growth",
    "fcf_margin",
    "rule_of_40_score",
    "payout_ratio",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "data_quality_score",
    "quarters_available",
]

# Sector categories for one-hot encoding
SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Energy",
    "Materials",
    "Real Estate",
    "Utilities",
    "Communication Services",
]

# Growth stages for one-hot encoding
STAGES = [
    "pre_profit",
    "early_growth",
    "high_growth",
    "transitioning",
    "mature",
    "dividend_paying",
]

# Company sizes for one-hot encoding
SIZES = [
    "micro_cap",
    "small_cap",
    "mid_cap",
    "large_cap",
    "mega_cap",
]

# Holding periods
HOLDING_PERIODS = ["1m", "3m", "6m", "12m", "18m", "24m", "36m"]

# Default sector-based holding periods (fallback if industry not matched)
SECTOR_DEFAULT_HOLDING = {
    "Technology": "6m",
    "Communication Services": "6m",
    "Consumer Discretionary": "12m",
    "Industrials": "12m",
    "Materials": "12m",
    "Energy": "12m",
    "Healthcare": "18m",
    "Financials": "18m",
    "Consumer Staples": "18m",
    "Utilities": "24m",
    "Real Estate": "24m",
}

# Industry categories for one-hot encoding (60+ categories)
INDUSTRY_CATEGORIES = [cat.value for cat in IndustryCategory]

# Volatility levels for feature encoding
VOLATILITY_LEVELS = ["low", "medium", "high", "very_high"]

# Orientation types for feature encoding
ORIENTATION_TYPES = ["growth", "value", "blend"]


class FundamentalRLPolicy(RLPolicy):
    """
    RL Policy for model weights and holding period based on fundamentals.

    Uses a contextual bandit approach to learn:
    - Optimal model weights for each sector/stage combination
    - Optimal holding period for each context
    """

    def __init__(
        self,
        base_weighting_service: Optional[Any] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        prior_variance: float = 1.0,
        noise_variance: float = 0.1,
        exploration_weight: float = 0.3,
        max_adjustment: float = 0.5,  # Max 50% adjustment from base weights
        use_industry_granularity: bool = True,  # Enable industry-level features
    ):
        """
        Initialize fundamental policy.

        Args:
            base_weighting_service: Rule-based weighting service (DynamicModelWeightingService)
            normalizer: Feature normalizer
            prior_variance: Prior variance for Bayesian updates
            noise_variance: Observation noise variance
            exploration_weight: UCB exploration parameter
            max_adjustment: Maximum adjustment from base weights (0.5 = 50%)
            use_industry_granularity: Use industry-level weights (60+ categories)
        """
        super().__init__(
            name="fundamental_policy",
            version="2.0",  # Updated version for industry granularity
            model_names=VALUATION_MODELS,
            normalizer=normalizer,
        )

        self.base_service = base_weighting_service
        self.max_adjustment = max_adjustment
        self.use_industry_granularity = use_industry_granularity

        # Calculate feature dimensions
        self.n_fundamental = len(FUNDAMENTAL_FEATURES)
        self.n_sectors = len(SECTORS)
        self.n_stages = len(STAGES)
        self.n_sizes = len(SIZES)

        # Industry-level features (new)
        self.n_industries = len(INDUSTRY_CATEGORIES) if use_industry_granularity else 0
        self.n_volatility = len(VOLATILITY_LEVELS) if use_industry_granularity else 0
        self.n_orientation = len(ORIENTATION_TYPES) if use_industry_granularity else 0
        self.n_industry_flags = 2 if use_industry_granularity else 0  # cyclical, is_known_industry

        # Total features: fundamentals + sector + stage + size + industry + volatility + orientation + flags
        self.n_features = (
            self.n_fundamental + self.n_sectors + self.n_stages + self.n_sizes +
            self.n_industries + self.n_volatility + self.n_orientation + self.n_industry_flags
        )

        self.n_models = len(VALUATION_MODELS)
        self.n_holding_periods = len(HOLDING_PERIODS)

        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self.exploration_weight = exploration_weight

        # Industry-specific learning tracking
        self._industry_update_counts: Dict[str, int] = {}
        self._industry_rewards: Dict[str, List[float]] = {}
        self._industry_optimal_weights: Dict[str, Dict[str, float]] = {}

        # Bayesian parameters for model weight adjustments
        # Each model has its own weight adjustment learned
        self.weight_mu = np.zeros((self.n_models, self.n_features))
        self.weight_Lambda = np.array([
            np.eye(self.n_features) / prior_variance
            for _ in range(self.n_models)
        ])
        self.weight_Sigma = np.array([
            np.eye(self.n_features) * prior_variance
            for _ in range(self.n_models)
        ])

        # Bayesian parameters for holding period
        self.holding_mu = np.zeros((self.n_holding_periods, self.n_features))
        self.holding_Lambda = np.array([
            np.eye(self.n_features) / prior_variance
            for _ in range(self.n_holding_periods)
        ])
        self.holding_Sigma = np.array([
            np.eye(self.n_features) * prior_variance
            for _ in range(self.n_holding_periods)
        ])

        # Statistics tracking
        self.model_update_counts = np.zeros(self.n_models)
        self.model_rewards = np.zeros(self.n_models)
        self.holding_update_counts = np.zeros(self.n_holding_periods)
        self.holding_rewards = np.zeros(self.n_holding_periods)

        # Sector-specific learned holding periods
        self._sector_optimal_periods: Dict[str, str] = {}
        self._sector_period_rewards: Dict[str, Dict[str, List[float]]] = {}

        self._ready = True

    def _extract_features(self, context: ValuationContext) -> np.ndarray:
        """Extract fundamental features from context including industry-level granularity."""
        if isinstance(context, dict):
            # Fundamental features
            fundamental = np.array([
                context.get("profitability_score", 0.5),
                context.get("pe_level", 0.5),
                context.get("revenue_growth", 0.0),
                context.get("fcf_margin", 0.0),
                context.get("rule_of_40_score", 0.0),
                context.get("payout_ratio", 0.0),
                context.get("debt_to_equity", 0.0),
                context.get("gross_margin", 0.0),
                context.get("operating_margin", 0.0),
                context.get("data_quality_score", 50.0) / 100.0,
                min(context.get("quarters_available", 0) / 20.0, 1.0),
            ])
            sector = context.get("sector", "Unknown")
            industry = context.get("industry", "Unknown")
            stage = context.get("growth_stage", "mature")
            if hasattr(stage, "value"):
                stage = stage.value
            size = context.get("company_size", "mid_cap")
            if hasattr(size, "value"):
                size = size.value
        else:
            fundamental = np.array([
                context.profitability_score,
                context.pe_level,
                context.revenue_growth,
                context.fcf_margin,
                context.rule_of_40_score,
                context.payout_ratio,
                context.debt_to_equity,
                context.gross_margin,
                context.operating_margin,
                context.data_quality_score / 100.0,
                min(context.quarters_available / 20.0, 1.0),
            ])
            sector = context.sector
            industry = getattr(context, "industry", "Unknown")
            stage = context.growth_stage.value if hasattr(context.growth_stage, "value") else context.growth_stage
            size = context.company_size.value if hasattr(context.company_size, "value") else context.company_size

        # One-hot encode sector
        sector_onehot = np.zeros(self.n_sectors)
        if sector in SECTORS:
            sector_onehot[SECTORS.index(sector)] = 1.0

        # One-hot encode stage
        stage_onehot = np.zeros(self.n_stages)
        if stage in STAGES:
            stage_onehot[STAGES.index(stage)] = 1.0

        # One-hot encode size
        size_onehot = np.zeros(self.n_sizes)
        if size in SIZES:
            size_onehot[SIZES.index(size)] = 1.0

        # Base features
        base_features = [fundamental, sector_onehot, stage_onehot, size_onehot]

        # Industry-level features (if enabled)
        if self.use_industry_granularity and self.n_industries > 0:
            # Classify industry
            industry_category = classify_industry(sector, industry)
            industry_profile = get_industry_profile(sector, industry)

            # One-hot encode industry category
            industry_onehot = np.zeros(self.n_industries)
            if industry_category.value in INDUSTRY_CATEGORIES:
                industry_onehot[INDUSTRY_CATEGORIES.index(industry_category.value)] = 1.0

            # One-hot encode volatility
            volatility_onehot = np.zeros(self.n_volatility)
            if industry_profile.volatility in VOLATILITY_LEVELS:
                volatility_onehot[VOLATILITY_LEVELS.index(industry_profile.volatility)] = 1.0

            # One-hot encode orientation
            orientation_onehot = np.zeros(self.n_orientation)
            if industry_profile.orientation in ORIENTATION_TYPES:
                orientation_onehot[ORIENTATION_TYPES.index(industry_profile.orientation)] = 1.0

            # Binary flags
            industry_flags = np.array([
                1.0 if industry_profile.cyclical else 0.0,
                1.0 if industry_category != IndustryCategory.UNKNOWN else 0.0,  # is_known_industry
            ])

            base_features.extend([industry_onehot, volatility_onehot, orientation_onehot, industry_flags])

        # Concatenate all features
        features = np.concatenate(base_features)

        return features

    def _get_industry_info(self, context: ValuationContext) -> Tuple[str, str, IndustryCategory]:
        """Extract industry information from context."""
        if isinstance(context, dict):
            sector = context.get("sector", "Unknown")
            industry = context.get("industry", "Unknown")
        else:
            sector = context.sector
            industry = getattr(context, "industry", "Unknown")

        category = classify_industry(sector, industry)
        return sector, industry, category

    def predict(self, context: ValuationContext) -> Dict[str, float]:
        """
        Predict model weights.

        Returns dict with weights for each model summing to 100%.
        """
        features = self._extract_features(context)

        # Get base weights from rule-based service (if available)
        base_weights = self._get_base_weights(context)

        # Calculate adjustment for each model
        adjustments = {}
        for i, model in enumerate(VALUATION_MODELS):
            # Mean prediction for adjustment
            mean = np.dot(self.weight_mu[i], features)
            # Clip adjustment to max_adjustment
            adj = np.clip(mean, -self.max_adjustment, self.max_adjustment)
            adjustments[model] = adj

        # Apply adjustments to base weights
        adjusted_weights = {}
        for model in VALUATION_MODELS:
            base = base_weights.get(model, 100.0 / self.n_models)
            # Multiplicative adjustment: base * (1 + adjustment)
            adjusted = base * (1.0 + adjustments[model])
            adjusted_weights[model] = max(0.0, adjusted)

        # Normalize to 100%
        total = sum(adjusted_weights.values())
        if total > 0:
            for model in adjusted_weights:
                adjusted_weights[model] = adjusted_weights[model] * 100.0 / total

        return adjusted_weights

    def predict_holding_period(self, context: ValuationContext) -> str:
        """
        Predict optimal holding period.

        Returns one of: "1m", "3m", "6m", "12m", "18m", "24m", "36m"

        Priority:
        1. Learned optimal period for specific industry (if enough data)
        2. Learned optimal period for sector (if enough data)
        3. Industry-profile default holding period
        4. Sector default holding period
        """
        features = self._extract_features(context)

        # Get sector/industry info
        sector, industry, category = self._get_industry_info(context)

        # Check if we have a learned optimal period for this sector
        if sector in self._sector_optimal_periods:
            return self._sector_optimal_periods[sector]

        # Calculate expected reward for each holding period
        period_values = np.zeros(self.n_holding_periods)
        for i in range(self.n_holding_periods):
            mean = np.dot(self.holding_mu[i], features)
            var = np.dot(features, np.dot(self.holding_Sigma[i], features))
            std = np.sqrt(max(var, 1e-6))
            period_values[i] = mean + self.exploration_weight * std

        # Select best period
        best_idx = np.argmax(period_values)

        # If not enough data, use industry-level default
        if self.holding_update_counts[best_idx] < 10:
            # Try industry-level holding period first
            if self.use_industry_granularity and category != IndustryCategory.UNKNOWN:
                return get_industry_holding_period(sector, industry)
            # Fall back to sector default
            return SECTOR_DEFAULT_HOLDING.get(sector, "12m")

        return HOLDING_PERIODS[best_idx]

    def predict_with_holding_period(
        self,
        context: ValuationContext,
    ) -> Tuple[Dict[str, float], str]:
        """
        Predict both model weights and holding period.

        Returns:
            Tuple of (weights dict, holding period string)
        """
        weights = self.predict(context)
        holding = self.predict_holding_period(context)
        return weights, holding

    def _get_base_weights(self, context: ValuationContext) -> Dict[str, float]:
        """
        Get base weights from industry profile or rule-based service.

        Priority:
        1. Industry-specific weights from INDUSTRY_PROFILES (if enabled)
        2. Rule-based weighting service (DynamicModelWeightingService)
        3. Equal weights as fallback
        """
        # Extract sector/industry
        sector, industry, category = self._get_industry_info(context)

        # Use industry-level weights if enabled and industry is known
        if self.use_industry_granularity and category != IndustryCategory.UNKNOWN:
            industry_weights = get_industry_weights(sector, industry)
            return industry_weights

        # Fall back to rule-based service
        if self.base_service is not None:
            try:
                if isinstance(context, dict):
                    symbol = context.get("symbol", "")
                    financials = {"sector": context.get("sector", "Unknown")}
                    ratios = {"revenue_growth": context.get("revenue_growth", 0.0)}
                else:
                    symbol = context.symbol
                    financials = {"sector": context.sector}
                    ratios = {"revenue_growth": context.revenue_growth}

                weights, _, _ = self.base_service.determine_weights(
                    symbol=symbol,
                    financials=financials,
                    ratios=ratios,
                    market_context=None,
                )
                return weights
            except Exception as e:
                logger.warning(f"Base service failed: {e}")

        # Equal weights as last fallback
        return {m: 100.0 / self.n_models for m in VALUATION_MODELS}

    def update_weights(
        self,
        context: ValuationContext,
        weights_used: Dict[str, float],
        reward: float,
    ) -> None:
        """
        Update weight prediction based on observed reward.

        Args:
            context: The context when prediction was made
            weights_used: Weights that were used
            reward: Observed reward
        """
        features = self._extract_features(context)

        # Update each model based on its contribution
        for i, model in enumerate(VALUATION_MODELS):
            weight = weights_used.get(model, 0.0)
            if weight > 0:
                # Scale reward by weight contribution
                model_reward = reward * (weight / 100.0)

                # Bayesian update
                self.weight_Lambda[i] += np.outer(features, features) / self.noise_variance
                self.weight_Sigma[i] = np.linalg.inv(self.weight_Lambda[i])
                self.weight_mu[i] = np.dot(
                    self.weight_Sigma[i],
                    np.dot(self.weight_Lambda[i], self.weight_mu[i]) +
                    features * model_reward / self.noise_variance
                )

                self.model_update_counts[i] += 1
                self.model_rewards[i] += model_reward

        self._update_count += 1
        self._updated_at = datetime.now()

    def update_holding_period(
        self,
        context: ValuationContext,
        holding_period_used: str,
        reward: float,
    ) -> None:
        """
        Update holding period prediction based on observed reward.

        Args:
            context: The context when prediction was made
            holding_period_used: Holding period that was used
            reward: Observed reward at that holding period
        """
        if holding_period_used not in HOLDING_PERIODS:
            return

        features = self._extract_features(context)
        period_idx = HOLDING_PERIODS.index(holding_period_used)

        # Bayesian update for holding period
        self.holding_Lambda[period_idx] += np.outer(features, features) / self.noise_variance
        self.holding_Sigma[period_idx] = np.linalg.inv(self.holding_Lambda[period_idx])
        self.holding_mu[period_idx] = np.dot(
            self.holding_Sigma[period_idx],
            np.dot(self.holding_Lambda[period_idx], self.holding_mu[period_idx]) +
            features * reward / self.noise_variance
        )

        self.holding_update_counts[period_idx] += 1
        self.holding_rewards[period_idx] += reward

        # Update sector-specific tracking
        if isinstance(context, dict):
            sector = context.get("sector", "Unknown")
        else:
            sector = context.sector

        if sector not in self._sector_period_rewards:
            self._sector_period_rewards[sector] = {p: [] for p in HOLDING_PERIODS}

        self._sector_period_rewards[sector][holding_period_used].append(reward)

        # Update sector optimal period if we have enough data
        sector_data = self._sector_period_rewards[sector]
        min_samples = min(len(sector_data[p]) for p in HOLDING_PERIODS)
        if min_samples >= 5:
            best_period = None
            best_avg = -float("inf")
            for period in HOLDING_PERIODS:
                rewards = sector_data[period]
                if rewards:
                    avg = sum(rewards) / len(rewards)
                    if avg > best_avg:
                        best_avg = avg
                        best_period = period
            if best_period:
                self._sector_optimal_periods[sector] = best_period

    def update(
        self,
        context: ValuationContext,
        action: Dict[str, float],
        reward: float,
    ) -> None:
        """Combined update for weights (backward compatibility)."""
        self.update_weights(context, action, reward)

    def get_model_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each model."""
        stats = {}
        for i, model in enumerate(VALUATION_MODELS):
            count = self.model_update_counts[i]
            total_reward = self.model_rewards[i]
            stats[model] = {
                "count": int(count),
                "total_reward": float(total_reward),
                "avg_reward": float(total_reward / count) if count > 0 else 0.0,
            }
        return stats

    def get_holding_period_stats(self) -> Dict[str, Any]:
        """Get holding period learning statistics."""
        period_stats = {}
        for i, period in enumerate(HOLDING_PERIODS):
            count = self.holding_update_counts[i]
            total_reward = self.holding_rewards[i]
            period_stats[period] = {
                "count": int(count),
                "avg_reward": float(total_reward / count) if count > 0 else 0.0,
            }

        return {
            "period_stats": period_stats,
            "sector_optimal_periods": self._sector_optimal_periods,
            "sectors_learned": len(self._sector_optimal_periods),
        }

    def get_industry_stats(self) -> Dict[str, Any]:
        """Get industry-level learning statistics."""
        if not self.use_industry_granularity:
            return {"enabled": False}

        industry_stats = {}
        for industry_cat, count in self._industry_update_counts.items():
            rewards = self._industry_rewards.get(industry_cat, [])
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            industry_stats[industry_cat] = {
                "count": count,
                "avg_reward": avg_reward,
            }

        # Get top/bottom performing industries
        sorted_industries = sorted(
            industry_stats.items(),
            key=lambda x: x[1]["avg_reward"],
            reverse=True
        )

        return {
            "enabled": True,
            "n_industries_seen": len(industry_stats),
            "total_industry_categories": len(INDUSTRY_CATEGORIES),
            "industry_stats": industry_stats,
            "top_5_industries": sorted_industries[:5],
            "bottom_5_industries": sorted_industries[-5:] if len(sorted_industries) >= 5 else [],
            "industry_optimal_weights": self._industry_optimal_weights,
        }

    def save(self, path: str) -> bool:
        """Save policy state."""
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

            state = {
                "name": self.name,
                "version": self.version,
                "n_features": self.n_features,
                "n_models": self.n_models,
                "n_holding_periods": self.n_holding_periods,
                "prior_variance": self.prior_variance,
                "noise_variance": self.noise_variance,
                "exploration_weight": self.exploration_weight,
                "max_adjustment": self.max_adjustment,
                # Industry granularity settings
                "use_industry_granularity": self.use_industry_granularity,
                "n_industries": self.n_industries,
                "n_volatility": self.n_volatility,
                "n_orientation": self.n_orientation,
                "n_industry_flags": self.n_industry_flags,
                # Weight parameters
                "weight_mu": self.weight_mu,
                "weight_Lambda": self.weight_Lambda,
                "weight_Sigma": self.weight_Sigma,
                "model_update_counts": self.model_update_counts,
                "model_rewards": self.model_rewards,
                # Holding period parameters
                "holding_mu": self.holding_mu,
                "holding_Lambda": self.holding_Lambda,
                "holding_Sigma": self.holding_Sigma,
                "holding_update_counts": self.holding_update_counts,
                "holding_rewards": self.holding_rewards,
                "sector_optimal_periods": self._sector_optimal_periods,
                "sector_period_rewards": self._sector_period_rewards,
                # Industry-level tracking
                "industry_update_counts": self._industry_update_counts,
                "industry_rewards": self._industry_rewards,
                "industry_optimal_weights": self._industry_optimal_weights,
                # Metadata
                "update_count": self._update_count,
                "created_at": self._created_at.isoformat(),
                "updated_at": self._updated_at.isoformat(),
            }

            with open(path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved fundamental policy to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save fundamental policy: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load policy state."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.name = state.get("name", self.name)
            self.version = state.get("version", self.version)
            self.max_adjustment = state.get("max_adjustment", self.max_adjustment)

            # Industry granularity settings (load but don't override constructor setting)
            loaded_granularity = state.get("use_industry_granularity", False)
            if loaded_granularity != self.use_industry_granularity:
                logger.warning(
                    f"Industry granularity mismatch: loaded={loaded_granularity}, "
                    f"current={self.use_industry_granularity}. Using current setting."
                )

            # Weight parameters - handle dimension mismatch gracefully
            loaded_n_features = state.get("n_features", self.n_features)
            if loaded_n_features == self.n_features:
                self.weight_mu = state.get("weight_mu", self.weight_mu)
                self.weight_Lambda = state.get("weight_Lambda", self.weight_Lambda)
                self.weight_Sigma = state.get("weight_Sigma", self.weight_Sigma)
                self.holding_mu = state.get("holding_mu", self.holding_mu)
                self.holding_Lambda = state.get("holding_Lambda", self.holding_Lambda)
                self.holding_Sigma = state.get("holding_Sigma", self.holding_Sigma)
            else:
                logger.warning(
                    f"Feature dimension mismatch: loaded={loaded_n_features}, "
                    f"current={self.n_features}. Keeping new dimensions (retraining needed)."
                )

            self.model_update_counts = state.get("model_update_counts", self.model_update_counts)
            self.model_rewards = state.get("model_rewards", self.model_rewards)

            # Holding period parameters
            self.holding_update_counts = state.get("holding_update_counts", self.holding_update_counts)
            self.holding_rewards = state.get("holding_rewards", self.holding_rewards)
            self._sector_optimal_periods = state.get("sector_optimal_periods", {})
            self._sector_period_rewards = state.get("sector_period_rewards", {})

            # Industry-level tracking
            self._industry_update_counts = state.get("industry_update_counts", {})
            self._industry_rewards = state.get("industry_rewards", {})
            self._industry_optimal_weights = state.get("industry_optimal_weights", {})

            self._update_count = state.get("update_count", 0)

            if "created_at" in state:
                self._created_at = datetime.fromisoformat(state["created_at"])
            if "updated_at" in state:
                self._updated_at = datetime.fromisoformat(state["updated_at"])

            self._ready = True
            logger.info(f"Loaded fundamental policy from {path}")
            return True

        except FileNotFoundError:
            logger.warning(f"Fundamental policy file not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load fundamental policy: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get policy state for inspection."""
        return {
            "name": self.name,
            "version": self.version,
            "n_features": self.n_features,
            "update_count": self._update_count,
            "model_stats": self.get_model_stats(),
            "holding_period_stats": self.get_holding_period_stats(),
            "industry_stats": self.get_industry_stats(),
            "max_adjustment": self.max_adjustment,
            "use_industry_granularity": self.use_industry_granularity,
            "n_industries": self.n_industries,
        }
