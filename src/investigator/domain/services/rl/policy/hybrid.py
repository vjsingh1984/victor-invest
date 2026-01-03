"""
Hybrid Policy - Combines Rule-Based and RL

Combines the existing DynamicModelWeightingService (rule-based tier selection)
with RL-learned adjustments. This provides:

1. Preserves domain expertise embedded in tier system
2. RL learns corrections/improvements on top
3. Graceful degradation if RL fails (fallback to rules)
4. Configurable adjustment bounds (default: full 0-100% range)
5. Easier to interpret and debug
6. Optimal holding period prediction based on sector/context

Strategy:
1. Get base weights from DynamicModelWeightingService
2. Get adjustment multipliers from RL policy
3. Apply bounded adjustments to base weights
4. Normalize to 100%
5. Predict optimal holding period based on learned patterns

Configuration:
- max_adjustment=1.0: Allow 0-200% of base weight (can zero out or double)
- max_adjustment=0.3: Conservative ±30% adjustments only
- min_weight=0.0: Allow complete model disabling
- min_weight=5.0: Ensure every model has at least 5% weight

Usage:
    from investigator.domain.services.rl.policy import HybridPolicy

    # Full flexibility (default) - RL can set any model to 0% or 100%
    policy = HybridPolicy(
        base_weighting_service=dynamic_model_weighting_service,
        adjustment_policy=contextual_bandit,
    )

    # Conservative mode - limit adjustments to ±30%
    policy = HybridPolicy(
        base_weighting_service=dynamic_model_weighting_service,
        adjustment_policy=contextual_bandit,
        max_adjustment=0.30,
        min_weight=5.0,
    )

    weights = policy.predict(context)
    weights, holding_period = policy.predict_with_holding_period(context)
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.services.rl.models import ValuationContext
from investigator.domain.services.rl.policy.base import RLPolicy, VALUATION_MODELS
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)

# Holding periods in order from shortest to longest
HOLDING_PERIODS = ["1m", "3m", "6m", "12m", "18m", "24m", "36m"]

# Default sector-based holding period recommendations (based on business cycle sensitivity)
SECTOR_HOLDING_PERIODS = {
    # Short-term sectors (momentum-driven, high volatility)
    "Technology": "6m",
    "Information Technology": "6m",
    "Communication Services": "6m",
    # Medium-term sectors (cyclical)
    "Consumer Discretionary": "12m",
    "Industrials": "12m",
    "Materials": "12m",
    "Energy": "12m",
    # Long-term sectors (stable, value-oriented)
    "Consumer Staples": "18m",
    "Health Care": "18m",
    "Healthcare": "18m",
    "Financials": "18m",
    "Utilities": "24m",
    "Real Estate": "24m",
    # Default
    "Unknown": "12m",
}


class HybridPolicy(RLPolicy):
    """
    Hybrid policy combining rule-based weights with RL adjustments.

    The base weighting service provides domain-expert weights based on
    tier classification. The adjustment policy (RL) learns multipliers
    to improve upon the base weights.

    Final weight = base_weight * adjustment_multiplier

    Adjustments are bounded to [1 - max_adjustment, 1 + max_adjustment]
    to prevent extreme deviations from expert knowledge.
    """

    def __init__(
        self,
        base_weighting_service: Optional[Any] = None,
        adjustment_policy: Optional[RLPolicy] = None,
        max_adjustment: float = 1.0,
        min_weight: float = 0.0,
        model_names: Optional[List[str]] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        learn_adjustments: bool = True,
    ):
        """
        Initialize hybrid policy.

        Args:
            base_weighting_service: DynamicModelWeightingService instance.
            adjustment_policy: RL policy for learning adjustments.
            max_adjustment: Maximum adjustment multiplier (e.g., 1.0 = 0-200% of base).
                           Set to 1.0 to allow RL to zero out or double any model weight.
            min_weight: Minimum weight for any model after adjustment (default 0.0).
                       Set to 0.0 to allow RL to completely disable models.
            model_names: List of valuation model names.
            normalizer: Feature normalizer.
            learn_adjustments: If True, learn adjustments from rewards.
        """
        super().__init__(
            name="hybrid_policy",
            version="1.0",
            model_names=model_names,
            normalizer=normalizer,
        )

        self.base_service = base_weighting_service
        self.adjustment_policy = adjustment_policy
        self.max_adjustment = max_adjustment
        self.min_weight = min_weight
        self.learn_adjustments = learn_adjustments

        # Track adjustment statistics
        self._adjustment_history: List[Dict[str, float]] = []
        self._reward_history: List[float] = []

        # Per-model adjustment accumulators (for learning model-specific adjustments)
        self._model_adjustment_sum = {m: 0.0 for m in self.model_names}
        self._model_adjustment_count = {m: 0 for m in self.model_names}
        self._model_reward_correlation = {m: 0.0 for m in self.model_names}

        # Holding period learning (per-sector optimal holding periods)
        self._sector_period_rewards: Dict[str, Dict[str, List[float]]] = {}
        self._sector_optimal_periods: Dict[str, str] = {}
        self._symbol_optimal_periods: Dict[str, str] = {}

        # Ready if we have at least the base service
        self._ready = base_weighting_service is not None

    def predict(self, context: ValuationContext) -> Dict[str, float]:
        """
        Predict weights combining base service and RL adjustments.

        Steps:
        1. Get base weights from DynamicModelWeightingService
        2. Get adjustment multipliers from RL policy (if available)
        3. Apply bounded adjustments
        4. Normalize to 100%
        """
        # Step 1: Get base weights from tier-based system
        base_weights, tier, audit = self._get_base_weights(context)

        # Step 2: Get adjustment multipliers from RL
        adjustments = self._get_adjustments(context)

        # Step 3: Apply bounded adjustments
        adjusted_weights = self._apply_adjustments(base_weights, adjustments)

        # Step 4: Normalize and apply applicability mask
        final_weights = self.apply_applicability_mask(adjusted_weights, context)

        return final_weights

    def predict_with_confidence(
        self,
        context: ValuationContext,
    ) -> Tuple[Dict[str, float], float]:
        """
        Predict with confidence based on RL policy uncertainty.
        """
        weights = self.predict(context)

        # Get confidence from adjustment policy if available
        confidence = 0.7  # Base confidence from expert rules

        if self.adjustment_policy and hasattr(self.adjustment_policy, "predict_with_confidence"):
            _, rl_confidence = self.adjustment_policy.predict_with_confidence(context)
            # Blend confidences (expert rules get more weight)
            confidence = 0.6 * confidence + 0.4 * rl_confidence

        return weights, confidence

    def predict_with_holding_period(
        self,
        context: ValuationContext,
    ) -> Tuple[Dict[str, float], str]:
        """
        Predict weights and optimal holding period.

        Returns:
            Tuple of (weights dict, recommended holding period)
        """
        weights = self.predict(context)
        holding_period = self.predict_holding_period(context)
        return weights, holding_period

    def predict_holding_period(self, context: ValuationContext) -> str:
        """
        Predict optimal holding period based on context.

        Priority:
        1. Symbol-specific learned period (if available)
        2. Sector-specific learned period (if available)
        3. Default sector-based period
        """
        # Handle both ValuationContext objects and dicts
        if isinstance(context, dict):
            symbol = context.get("symbol", "")
            sector = context.get("sector", "Unknown")
            volatility = context.get("volatility", 0.5)
        else:
            symbol = context.symbol
            sector = context.sector
            volatility = context.volatility

        # Check symbol-specific learned period
        if symbol in self._symbol_optimal_periods:
            return self._symbol_optimal_periods[symbol]

        # Check sector-specific learned period
        if sector in self._sector_optimal_periods:
            return self._sector_optimal_periods[sector]

        # Default sector-based period (adjusted for volatility)
        base_period = SECTOR_HOLDING_PERIODS.get(sector, "12m")

        # High volatility -> shorter holding period
        # Low volatility -> longer holding period
        if volatility > 0.7:
            # Shorten by one step
            idx = HOLDING_PERIODS.index(base_period)
            return HOLDING_PERIODS[max(0, idx - 1)]
        elif volatility < 0.3:
            # Lengthen by one step
            idx = HOLDING_PERIODS.index(base_period)
            return HOLDING_PERIODS[min(len(HOLDING_PERIODS) - 1, idx + 1)]

        return base_period

    def update_holding_period_learning(
        self,
        context: ValuationContext,
        multi_period_rewards: Dict[str, float],
    ) -> None:
        """
        Update holding period learning from observed rewards.

        Args:
            context: The valuation context
            multi_period_rewards: Dict mapping period (1m, 3m, etc.) to reward
        """
        if isinstance(context, dict):
            symbol = context.get("symbol", "")
            sector = context.get("sector", "Unknown")
        else:
            symbol = context.symbol
            sector = context.sector

        # Initialize sector tracking if needed
        if sector not in self._sector_period_rewards:
            self._sector_period_rewards[sector] = {p: [] for p in HOLDING_PERIODS}

        # Add rewards for each period
        for period, reward in multi_period_rewards.items():
            if reward is not None and period in HOLDING_PERIODS:
                self._sector_period_rewards[sector][period].append(reward)

        # Update sector optimal period (need at least 10 samples per period)
        sector_data = self._sector_period_rewards[sector]
        min_samples = min(len(sector_data[p]) for p in HOLDING_PERIODS)
        if min_samples >= 10:
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
                logger.info(f"Updated optimal holding period for {sector}: {best_period} (avg reward: {best_avg:.3f})")

        # Update symbol optimal period if we have enough data
        best_symbol_period = None
        best_symbol_reward = -float("inf")
        for period, reward in multi_period_rewards.items():
            if reward is not None and reward > best_symbol_reward:
                best_symbol_reward = reward
                best_symbol_period = period

        if best_symbol_period and best_symbol_reward > 0.1:  # Only store if significantly positive
            self._symbol_optimal_periods[symbol] = best_symbol_period

    def update(
        self,
        context: ValuationContext,
        action: Dict[str, float],
        reward: float,
    ) -> None:
        """
        Update adjustment policy based on observed reward.

        Also tracks model-specific performance for adaptive learning.
        """
        if not self.learn_adjustments:
            return

        # Get base weights to compute what adjustments were made
        base_weights, _, _ = self._get_base_weights(context)

        # Calculate actual adjustments that were applied
        actual_adjustments = {}
        for model in self.model_names:
            base = base_weights.get(model, 0)
            actual = action.get(model, 0)
            if base > 0:
                actual_adjustments[model] = actual / base
            else:
                actual_adjustments[model] = 1.0

        # Update adjustment policy if available
        if self.adjustment_policy:
            self.adjustment_policy.update(context, action, reward)

        # Track adjustment history
        self._adjustment_history.append(actual_adjustments)
        self._reward_history.append(reward)

        # Update model-specific statistics
        for model, adj in actual_adjustments.items():
            self._model_adjustment_sum[model] += adj
            self._model_adjustment_count[model] += 1

            # Simple correlation tracking (adjustment vs reward)
            # Positive correlation means higher adjustment -> higher reward
            count = self._model_adjustment_count[model]
            if count > 1:
                avg_adj = self._model_adjustment_sum[model] / count
                # Running correlation update
                delta = (adj - avg_adj) * reward
                self._model_reward_correlation[model] = (
                    self._model_reward_correlation[model] * (count - 1) + delta
                ) / count

        self._update_count += 1
        self._updated_at = datetime.now()

    def _get_base_weights(
        self,
        context: ValuationContext,
    ) -> Tuple[Dict[str, float], str, Any]:
        """Get base weights from the tier-based service."""
        if self.base_service is None:
            # Fallback to equal weights
            n = len(self.model_names)
            return {m: 100 / n for m in self.model_names}, "fallback_equal", None

        try:
            # Convert context to format expected by base service
            financials = self._context_to_financials(context)
            ratios = self._context_to_ratios(context)

            # Handle both ValuationContext objects and dicts for symbol
            symbol = context.get("symbol") if isinstance(context, dict) else context.symbol

            # Note: We pass market_context=None because DynamicModelWeightingService
            # expects a MarketContext object (with enum attributes), not a dict.
            # The RL policy already accounts for market context through ValuationContext.
            weights, tier, audit = self.base_service.determine_weights(
                symbol=symbol,
                financials=financials,
                ratios=ratios,
                market_context=None,
            )
            return weights, tier, audit

        except Exception as e:
            logger.warning(f"Base service failed, using fallback: {e}")
            n = len(self.model_names)
            return {m: 100 / n for m in self.model_names}, "fallback_error", None

    def _get_adjustments(
        self,
        context: ValuationContext,
    ) -> Dict[str, float]:
        """Get adjustment multipliers from RL policy."""
        # Default: no adjustment (multiplier = 1.0)
        default_adjustments = {m: 1.0 for m in self.model_names}

        if self.adjustment_policy is None or not self.adjustment_policy.is_ready():
            return default_adjustments

        try:
            # The adjustment policy predicts weights, we convert to multipliers
            # by comparing to equal weights
            predicted = self.adjustment_policy.predict(context)

            # Convert predicted weights to adjustment multipliers
            # If policy predicts 40% for DCF (vs 16.7% equal), multiplier = 40/16.7 = 2.4
            # We'll cap this with max_adjustment
            base_equal = 100 / len(self.model_names)

            adjustments = {}
            for model in self.model_names:
                predicted_weight = predicted.get(model, base_equal)
                raw_multiplier = predicted_weight / base_equal

                # Center around 1.0 and apply bounds
                # If predicted is higher, multiplier > 1
                # If predicted is lower, multiplier < 1
                bounded = max(1 - self.max_adjustment, min(1 + self.max_adjustment, raw_multiplier))
                adjustments[model] = bounded

            return adjustments

        except Exception as e:
            logger.warning(f"Adjustment policy failed: {e}")
            return default_adjustments

    def _apply_adjustments(
        self,
        base_weights: Dict[str, float],
        adjustments: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply adjustment multipliers to base weights.

        With max_adjustment=1.0 and min_weight=0.0:
        - A model can be completely disabled (weight=0)
        - A model can be doubled (weight=2x base)
        - After normalization, weights sum to 100%

        This allows RL to learn extreme allocations like:
        - 100% DCF for stable dividend payers
        - 0% DCF for pre-profit tech (use PS/EV instead)
        """
        adjusted = {}
        for model in self.model_names:
            base = base_weights.get(model, 0)
            mult = adjustments.get(model, 1.0)
            weight = base * mult
            # Apply min_weight floor (default 0.0 allows complete disabling)
            adjusted[model] = max(self.min_weight, weight)

        return self.normalize_weights(adjusted)

    def _context_to_financials(self, context: ValuationContext) -> Dict[str, Any]:
        """Convert context back to financials dict for base service."""
        # Handle both ValuationContext objects and dicts
        if isinstance(context, dict):
            return {
                "sector": context.get("sector", "Unknown"),
                "industry": context.get("industry", "Unknown"),
                "fiscal_period": context.get("fiscal_period"),
                "market_cap": None,
                "free_cash_flow": None,
                "net_income": None,
                "ebitda": None,
            }
        return {
            "sector": context.sector,
            "industry": context.industry,
            "fiscal_period": context.fiscal_period,
            "market_cap": None,  # Not stored in context
            "free_cash_flow": None,
            "net_income": None,
            "ebitda": None,
        }

    def _context_to_ratios(self, context: ValuationContext) -> Dict[str, Any]:
        """Convert context back to ratios dict for base service."""
        # Handle both ValuationContext objects and dicts
        if isinstance(context, dict):
            return {
                "revenue_growth": context.get("revenue_growth", 0.0),
                "fcf_margin": context.get("fcf_margin", 0.0),
                "gross_margin": context.get("gross_margin", 0.0),
                "operating_margin": context.get("operating_margin", 0.0),
                "payout_ratio": context.get("payout_ratio", 0.0),
                "debt_to_equity": context.get("debt_to_equity", 0.0),
                "pe_ratio": None,
            }
        return {
            "revenue_growth": context.revenue_growth,
            "fcf_margin": context.fcf_margin,
            "gross_margin": context.gross_margin,
            "operating_margin": context.operating_margin,
            "payout_ratio": context.payout_ratio,
            "debt_to_equity": context.debt_to_equity,
            "pe_ratio": None,  # We have pe_level but not raw PE
        }

    def _context_to_market_context(self, context: ValuationContext) -> Optional[Dict[str, Any]]:
        """Convert context to market context dict."""
        # Handle both ValuationContext objects and dicts
        logger.debug(
            f"_context_to_market_context: context type = {type(context)}, is dict = {isinstance(context, dict)}"
        )
        if isinstance(context, dict):
            logger.debug("Using dict path")
            return {
                "trend_score": context.get("technical_trend", 0.0),
                "sentiment_score": context.get("market_sentiment", 0.0),
                "volatility": context.get("volatility", 0.5),
            }
        logger.debug("Using ValuationContext path")
        return {
            "trend_score": context.technical_trend,
            "sentiment_score": context.market_sentiment,
            "volatility": context.volatility,
        }

    def get_adjustment_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics on model adjustments."""
        stats = {}
        for model in self.model_names:
            count = self._model_adjustment_count[model]
            if count > 0:
                avg_adj = self._model_adjustment_sum[model] / count
                stats[model] = {
                    "count": count,
                    "avg_adjustment": avg_adj,
                    "reward_correlation": self._model_reward_correlation[model],
                }
            else:
                stats[model] = {
                    "count": 0,
                    "avg_adjustment": 1.0,
                    "reward_correlation": 0.0,
                }
        return stats

    def get_learned_adjustments(self) -> Dict[str, float]:
        """
        Get learned adjustment multipliers based on historical performance.

        Returns average adjustments that have worked well.
        """
        adjustments = {}
        for model in self.model_names:
            if self._model_adjustment_count[model] > 10:
                # Use correlation to determine direction
                corr = self._model_reward_correlation[model]
                avg_adj = self._model_adjustment_sum[model] / self._model_adjustment_count[model]

                # If positive correlation, adjusting up helps
                # If negative correlation, adjusting down helps
                if corr > 0.1:
                    adjustments[model] = min(1 + self.max_adjustment, avg_adj)
                elif corr < -0.1:
                    adjustments[model] = max(1 - self.max_adjustment, 2 - avg_adj)
                else:
                    adjustments[model] = 1.0
            else:
                adjustments[model] = 1.0

        return adjustments

    def save(self, path: str) -> bool:
        """Save hybrid policy state."""
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

            state = {
                "name": self.name,
                "version": self.version,
                "max_adjustment": self.max_adjustment,
                "learn_adjustments": self.learn_adjustments,
                "model_adjustment_sum": self._model_adjustment_sum,
                "model_adjustment_count": self._model_adjustment_count,
                "model_reward_correlation": self._model_reward_correlation,
                "update_count": self._update_count,
                "created_at": self._created_at.isoformat(),
                "updated_at": self._updated_at.isoformat(),
                # Holding period learning
                "sector_period_rewards": self._sector_period_rewards,
                "sector_optimal_periods": self._sector_optimal_periods,
                "symbol_optimal_periods": self._symbol_optimal_periods,
            }

            # Save adjustment policy separately if it exists
            if self.adjustment_policy:
                adj_policy_path = path.replace(".pkl", "_adjustment.pkl")
                self.adjustment_policy.save(adj_policy_path)
                state["adjustment_policy_path"] = adj_policy_path

            with open(path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved hybrid policy to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save hybrid policy: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load hybrid policy state."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.name = state.get("name", self.name)
            self.version = state.get("version", self.version)
            self.max_adjustment = state.get("max_adjustment", self.max_adjustment)
            self.learn_adjustments = state.get("learn_adjustments", self.learn_adjustments)
            self._model_adjustment_sum = state.get("model_adjustment_sum", {m: 0.0 for m in self.model_names})
            self._model_adjustment_count = state.get("model_adjustment_count", {m: 0 for m in self.model_names})
            self._model_reward_correlation = state.get("model_reward_correlation", {m: 0.0 for m in self.model_names})
            self._update_count = state.get("update_count", 0)

            # Load holding period learning
            self._sector_period_rewards = state.get("sector_period_rewards", {})
            self._sector_optimal_periods = state.get("sector_optimal_periods", {})
            self._symbol_optimal_periods = state.get("symbol_optimal_periods", {})

            if "created_at" in state:
                self._created_at = datetime.fromisoformat(state["created_at"])
            if "updated_at" in state:
                self._updated_at = datetime.fromisoformat(state["updated_at"])

            # Load adjustment policy if path specified
            if "adjustment_policy_path" in state and self.adjustment_policy:
                self.adjustment_policy.load(state["adjustment_policy_path"])

            self._ready = self.base_service is not None
            logger.info(f"Loaded hybrid policy from {path}")
            return True

        except FileNotFoundError:
            logger.warning(f"Policy file not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load hybrid policy: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get policy state for inspection."""
        state = super().get_state()
        state.update(
            {
                "max_adjustment": self.max_adjustment,
                "learn_adjustments": self.learn_adjustments,
                "has_base_service": self.base_service is not None,
                "has_adjustment_policy": self.adjustment_policy is not None,
                "adjustment_stats": self.get_adjustment_stats(),
                "holding_period_stats": self.get_holding_period_stats(),
            }
        )
        return state

    def get_holding_period_stats(self) -> Dict[str, Any]:
        """Get holding period learning statistics."""
        sector_stats = {}
        for sector, period_data in self._sector_period_rewards.items():
            sector_stats[sector] = {
                "samples_per_period": {p: len(r) for p, r in period_data.items()},
                "avg_reward_per_period": {
                    p: (sum(r) / len(r) if r else 0) for p, r in period_data.items()
                },
                "optimal_period": self._sector_optimal_periods.get(sector),
            }

        return {
            "sectors_learned": len(self._sector_optimal_periods),
            "symbols_learned": len(self._symbol_optimal_periods),
            "sector_stats": sector_stats,
        }
