"""
RL Policy Package

Provides policy implementations for valuation model weight prediction:

Single Policy Approaches:
- RLPolicy: Abstract base class defining the policy interface
- ContextualBanditPolicy: Thompson Sampling for discrete tier selection
- HybridPolicy: Combines rule-based tiers with RL adjustments

Dual Policy Approach (Recommended):
- TechnicalRLPolicy: Timing and position decisions (Long/Short/Skip)
- FundamentalRLPolicy: Model weights and holding periods
- DualRLPolicy: Combines both for complete prediction

Usage (Dual Policy - Recommended):
    from investigator.domain.services.rl.policy import DualRLPolicy, load_dual_policy

    # Load or create dual policy
    policy = load_dual_policy()

    # Get complete prediction
    result = policy.predict_full(context)
    # result = {
    #     "position": 1,  # Long
    #     "position_confidence": 0.75,
    #     "weights": {"dcf": 30.0, "pe": 25.0, ...},
    #     "holding_period": "6m",
    # }

    # Update after outcome known
    policy.update(context, result, position_reward, holding_rewards)

Usage (Single Policy):
    from investigator.domain.services.rl.policy import HybridPolicy

    policy = HybridPolicy(
        base_weighting_service=dynamic_model_weighting_service,
        adjustment_policy=contextual_bandit,
    )
    weights = policy.predict(context)
"""

from investigator.domain.services.rl.policy.base import RLPolicy
from investigator.domain.services.rl.policy.contextual_bandit import ContextualBanditPolicy
from investigator.domain.services.rl.policy.dual_policy import DualRLPolicy, load_dual_policy
from investigator.domain.services.rl.policy.fundamental_policy import FundamentalRLPolicy
from investigator.domain.services.rl.policy.hybrid import HybridPolicy
from investigator.domain.services.rl.policy.technical_policy import TechnicalRLPolicy

__all__ = [
    # Base
    "RLPolicy",
    # Single policies
    "ContextualBanditPolicy",
    "HybridPolicy",
    # Dual policy system (recommended)
    "TechnicalRLPolicy",
    "FundamentalRLPolicy",
    "DualRLPolicy",
    "load_dual_policy",
]
