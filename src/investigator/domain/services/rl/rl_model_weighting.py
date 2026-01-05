"""
RL Model Weighting Service

Drop-in replacement for DynamicModelWeightingService that uses RL-based
weight prediction. Falls back to rule-based weights when RL is not ready.

Usage:
    from investigator.domain.services.rl import RLModelWeightingService

    # Create service (will use RL when ready, fallback otherwise)
    service = RLModelWeightingService(
        rl_enabled=True,
        fallback_service=dynamic_model_weighting_service,
    )

    # Use exactly like DynamicModelWeightingService
    weights, tier, audit = service.determine_weights(
        symbol="AAPL",
        financials=financials_dict,
        ratios=ratios_dict,
        market_context=market_context,
    )
"""

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from investigator.domain.services.rl.feature_extractor import ValuationContextExtractor
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import (
    ABTestGroup,
    ValuationContext,
)
from investigator.domain.services.rl.outcome_tracker import OutcomeTracker
from investigator.domain.services.rl.policy.base import RLPolicy
from investigator.domain.services.rl.policy.contextual_bandit import ContextualBanditPolicy
from investigator.domain.services.rl.policy.dual_policy import DualRLPolicy
from investigator.domain.services.rl.policy.hybrid import HybridPolicy
from investigator.domain.services.weight_audit_trail import WeightAuditTrail

logger = logging.getLogger(__name__)


class RLModelWeightingService:
    """
    RL-enabled model weighting service.

    Drop-in replacement for DynamicModelWeightingService that:
    1. Uses RL policy for weight prediction when available
    2. Falls back to rule-based weights when RL not ready
    3. Records predictions for outcome tracking
    4. Supports A/B testing between RL and baseline

    Interface matches DynamicModelWeightingService.determine_weights()
    for seamless integration.
    """

    def __init__(
        self,
        rl_enabled: bool = True,
        fallback_service: Optional[Any] = None,
        policy: Optional[RLPolicy] = None,
        policy_path: str = "data/rl_models/policy.pkl",
        normalizer_path: str = "data/rl_models/normalizer.pkl",
        outcome_tracker: Optional[OutcomeTracker] = None,
        ab_test_enabled: bool = False,
        ab_test_rl_fraction: float = 0.20,
        # Dual policy support
        use_dual_policy: bool = True,
        technical_policy_path: str = "data/rl_models/active_technical_policy.pkl",
        fundamental_policy_path: str = "data/rl_models/active_fundamental_policy.pkl",
    ):
        """
        Initialize RL model weighting service.

        Args:
            rl_enabled: Whether to use RL policy.
            fallback_service: DynamicModelWeightingService for fallback.
            policy: Pre-created RL policy (optional).
            policy_path: Path to load policy from (legacy single policy).
            normalizer_path: Path to load normalizer from.
            outcome_tracker: OutcomeTracker for recording predictions.
            ab_test_enabled: Whether to run A/B test.
            ab_test_rl_fraction: Fraction of requests to use RL.
            use_dual_policy: Whether to use dual policy (Technical + Fundamental).
            technical_policy_path: Path to technical policy for dual mode.
            fundamental_policy_path: Path to fundamental policy for dual mode.
        """
        self.rl_enabled = rl_enabled
        self.fallback_service = fallback_service
        self.policy = policy
        self.policy_path = policy_path
        self.normalizer_path = normalizer_path
        self.outcome_tracker = outcome_tracker or OutcomeTracker()
        self.ab_test_enabled = ab_test_enabled
        self.ab_test_rl_fraction = ab_test_rl_fraction

        # Dual policy support
        self.use_dual_policy = use_dual_policy
        self.technical_policy_path = technical_policy_path
        self.fundamental_policy_path = fundamental_policy_path
        self.dual_policy: Optional[DualRLPolicy] = None

        # Feature extraction
        self.extractor = ValuationContextExtractor()
        self.normalizer = FeatureNormalizer()

        # Load existing policy if available
        self._load_policy()

        # Track usage statistics
        self._rl_predictions = 0
        self._fallback_predictions = 0

    def _load_policy(self) -> bool:
        """Load policy and normalizer from disk."""
        if not self.rl_enabled:
            return False

        try:
            # Load normalizer first
            if os.path.exists(self.normalizer_path):
                self.normalizer.load(self.normalizer_path)
                logger.info(f"Loaded normalizer from {self.normalizer_path}")

            # Try to load dual policy first (preferred)
            if self.use_dual_policy:
                tech_exists = os.path.exists(self.technical_policy_path)
                fund_exists = os.path.exists(self.fundamental_policy_path)

                if tech_exists and fund_exists:
                    self.dual_policy = DualRLPolicy(
                        base_weighting_service=self.fallback_service,
                        technical_path=self.technical_policy_path,
                        fundamental_path=self.fundamental_policy_path,
                    )
                    logger.info(
                        f"Loaded dual RL policy: technical={self.technical_policy_path}, "
                        f"fundamental={self.fundamental_policy_path}"
                    )
                    return True
                else:
                    logger.info(
                        f"Dual policy files not found (tech={tech_exists}, fund={fund_exists}), "
                        "falling back to single policy mode"
                    )

            # Fall back to single policy mode
            if self.policy is None:
                if self.fallback_service:
                    # Use hybrid policy
                    adjustment_policy = ContextualBanditPolicy(normalizer=self.normalizer)
                    self.policy = HybridPolicy(
                        base_weighting_service=self.fallback_service,
                        adjustment_policy=adjustment_policy,
                        normalizer=self.normalizer,
                    )
                else:
                    # Use standalone bandit
                    self.policy = ContextualBanditPolicy(normalizer=self.normalizer)

            # Load policy weights
            if os.path.exists(self.policy_path):
                self.policy.load(self.policy_path)
                logger.info(f"Loaded RL policy from {self.policy_path}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to load RL policy: {e}")
            return False

    def is_dual_policy_active(self) -> bool:
        """Check if dual policy is loaded and active."""
        return self.dual_policy is not None

    def determine_weights(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], str, Optional[WeightAuditTrail]]:
        """
        Determine model weights using RL or fallback.

        Interface matches DynamicModelWeightingService for drop-in replacement.

        Args:
            symbol: Stock ticker symbol.
            financials: Financial statement data.
            ratios: Calculated financial ratios.
            data_quality: Data quality scores.
            market_context: Market context data.

        Returns:
            Tuple of (weights_dict, tier_classification, audit_trail).
        """
        # Determine A/B test group
        ab_group = self._get_ab_test_group(symbol)

        # Decide whether to use RL (dual policy takes precedence)
        use_dual = self.dual_policy is not None
        use_single = self.policy is not None and self.policy.is_ready()
        use_rl = (
            self.rl_enabled and (use_dual or use_single) and (not self.ab_test_enabled or ab_group == ABTestGroup.RL)
        )

        if use_rl:
            weights, tier, audit = self._predict_with_rl(symbol, financials, ratios, data_quality, market_context)
            self._rl_predictions += 1
        else:
            weights, tier, audit = self._predict_with_fallback(symbol, financials, ratios, data_quality, market_context)
            self._fallback_predictions += 1

        # Record prediction for outcome tracking
        self._record_prediction(
            symbol=symbol,
            weights=weights,
            tier=tier,
            financials=financials,
            ratios=ratios,
            data_quality=data_quality,
            market_context=market_context,
            ab_group=ab_group if self.ab_test_enabled else None,
            used_rl=use_rl,
        )

        return weights, tier, audit

    def _predict_with_rl(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], str, Optional[WeightAuditTrail]]:
        """Make prediction using RL policy (dual or single)."""
        try:
            # Extract context
            context = self.extractor.extract(
                symbol=symbol,
                financials=financials,
                ratios=ratios,
                market_context=market_context,
                data_quality=data_quality,
            )

            # Use dual policy if available (includes industry granularity)
            if self.dual_policy is not None:
                result = self.dual_policy.predict_full(context)
                weights = result.get("weights", {})
                position = result.get("position", 0)
                position_conf = result.get("position_confidence", 0.0)
                holding_period = result.get("holding_period", "3m")

                # Create audit trail with dual policy metadata
                audit = WeightAuditTrail(symbol=symbol)
                audit.metadata = {
                    "policy_type": "dual",
                    "position": position,  # -1=short, 0=skip, 1=long
                    "position_confidence": position_conf,
                    "holding_period": holding_period,
                    "industry_category": result.get("industry_category", "unknown"),
                }

                tier = f"dual_rl_{'long' if position == 1 else 'short' if position == -1 else 'skip'}"
                return weights, tier, audit

            # Fall back to single policy
            weights = self.policy.predict(context)

            # Create simple audit trail
            audit = WeightAuditTrail(symbol=symbol)
            audit.metadata = {"policy_type": "single"}

            return weights, "rl_predicted", audit

        except Exception as e:
            logger.warning(f"RL prediction failed for {symbol}, using fallback: {e}")
            return self._predict_with_fallback(symbol, financials, ratios, data_quality, market_context)

    def _predict_with_fallback(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], str, Optional[WeightAuditTrail]]:
        """Make prediction using fallback service."""
        if self.fallback_service is None:
            # Return default equal weights
            default_weights = {
                "dcf": 16.67,
                "pe": 16.67,
                "ps": 16.67,
                "ev_ebitda": 16.67,
                "pb": 16.66,
                "ggm": 16.66,
            }
            return default_weights, "fallback_equal", None

        try:
            # Check if fallback service has the determine_weights method
            if hasattr(self.fallback_service, "determine_weights"):
                return self.fallback_service.determine_weights(
                    symbol=symbol,
                    financials=financials,
                    ratios=ratios,
                    data_quality=data_quality,
                    market_context=market_context,
                )
            else:
                # Try legacy interface
                weights, tier, _ = self.fallback_service.determine_weights(
                    symbol=symbol,
                    financials=financials,
                    ratios=ratios,
                    market_context=market_context,
                )
                return weights, tier, None

        except Exception as e:
            logger.error(f"Fallback service failed for {symbol}: {e}")
            default_weights = {
                "dcf": 16.67,
                "pe": 16.67,
                "ps": 16.67,
                "ev_ebitda": 16.67,
                "pb": 16.66,
                "ggm": 16.66,
            }
            return default_weights, "fallback_error", None

    def _get_ab_test_group(self, symbol: str) -> ABTestGroup:
        """
        Determine A/B test group for symbol.

        Uses consistent hashing so same symbol always gets same group.
        """
        if not self.ab_test_enabled:
            return ABTestGroup.BASELINE

        # Hash symbol for consistent assignment
        hash_val = hash(symbol) % 100
        if hash_val < self.ab_test_rl_fraction * 100:
            return ABTestGroup.RL
        else:
            return ABTestGroup.BASELINE

    def _record_prediction(
        self,
        symbol: str,
        weights: Dict[str, float],
        tier: str,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
        ab_group: Optional[ABTestGroup],
        used_rl: bool,
    ) -> None:
        """Record prediction for outcome tracking."""
        try:
            # Extract context for storage
            context = self.extractor.extract(
                symbol=symbol,
                financials=financials,
                ratios=ratios,
                market_context=market_context,
                data_quality=data_quality,
            )

            # Get current price from ratios or financials
            current_price = ratios.get("current_price") or financials.get("current_price") or 0.0

            # Placeholder for blended fair value (will be updated by caller)
            blended_fair_value = 0.0

            # Get model fair values if available
            model_fair_values = financials.get("model_fair_values", {})

            self.outcome_tracker.record_prediction(
                symbol=symbol,
                analysis_date=date.today(),
                blended_fair_value=blended_fair_value,
                current_price=current_price,
                model_fair_values=model_fair_values,
                model_weights=weights,
                tier_classification=tier,
                context_features=context,
                fiscal_period=financials.get("fiscal_period"),
                ab_test_group=ab_group,
                policy_version=self.policy.version if self.policy else None,
            )

        except Exception as e:
            logger.warning(f"Failed to record prediction for {symbol}: {e}")

    def update_blended_value(
        self,
        symbol: str,
        analysis_date: date,
        blended_fair_value: float,
        model_fair_values: Dict[str, float],
    ) -> None:
        """
        Update recorded prediction with blended fair value.

        Called after valuation models have computed their fair values.

        Args:
            symbol: Stock ticker.
            analysis_date: Date of analysis.
            blended_fair_value: Final blended fair value.
            model_fair_values: Individual model fair values.
        """
        # This would update the existing record with the actual values
        # For now, we record at prediction time with placeholders
        pass

    def update_from_outcome(
        self,
        symbol: str,
        analysis_date: date,
        weights_used: Dict[str, float],
        reward: float,
    ) -> None:
        """
        Update RL policy based on observed outcome.

        Called when outcome data becomes available.

        Args:
            symbol: Stock ticker.
            analysis_date: Date of original prediction.
            weights_used: Weights that were used.
            reward: Calculated reward.
        """
        if self.policy and self.rl_enabled:
            try:
                # Reconstruct context from stored features
                # This would require looking up the stored prediction
                # For now, we rely on batch training from experience database
                pass
            except Exception as e:
                logger.warning(f"Failed to update policy: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = self._rl_predictions + self._fallback_predictions
        return {
            "rl_enabled": self.rl_enabled,
            "policy_ready": self.policy.is_ready() if self.policy else False,
            "ab_test_enabled": self.ab_test_enabled,
            "rl_predictions": self._rl_predictions,
            "fallback_predictions": self._fallback_predictions,
            "total_predictions": total,
            "rl_fraction": self._rl_predictions / total if total > 0 else 0,
        }

    def reload_policy(self) -> bool:
        """
        Reload policy from disk.

        Useful after retraining.

        Returns:
            True if successfully reloaded.
        """
        return self._load_policy()


# Factory function
def get_rl_model_weighting_service(
    rl_enabled: bool = True,
    fallback_service: Optional[Any] = None,
) -> RLModelWeightingService:
    """Get RLModelWeightingService instance."""
    return RLModelWeightingService(
        rl_enabled=rl_enabled,
        fallback_service=fallback_service,
    )
