"""
Reinforcement Learning Module for Adaptive Model Weighting

This module provides RL-based learning for optimal valuation model weights.
It learns from historical prediction outcomes to improve accuracy over time.

Architecture:
- OutcomeTracker: Records predictions and updates with actual outcomes
- FeatureExtractor: Extracts context features for RL state representation
- Policy: RL policies (ContextualBandit, Hybrid) for weight prediction
- Training: Experience collection and policy training
- Monitoring: Performance metrics and A/B testing

Usage:
    from investigator.domain.services.rl import (
        OutcomeTracker,
        ValuationContextExtractor,
        HybridPolicy,
        RLModelWeightingService,
    )

    # Record prediction for future outcome tracking
    outcome_tracker.record_prediction(
        symbol="AAPL",
        analysis_date=date.today(),
        blended_fair_value=175.50,
        current_price=170.00,
        model_fair_values={"dcf": 180.0, "pe": 170.0, "ps": 175.0},
        model_weights={"dcf": 40, "pe": 35, "ps": 25},
        tier_classification="balanced_high_quality",
        context_features=context,
    )

    # Use RL-enabled weighting (drop-in replacement for DynamicModelWeightingService)
    weights, tier, audit = rl_weighting_service.determine_weights(
        symbol="AAPL",
        financials=financials_dict,
        ratios=ratios_dict,
        market_context=market_context,
    )
"""

from investigator.domain.services.rl.models import (
    ValuationContext,
    Experience,
    TrainingMetrics,
    EvaluationMetrics,
    ABTestResults,
    RewardSignal,
)
from investigator.domain.services.rl.outcome_tracker import OutcomeTracker
from investigator.domain.services.rl.price_history import PriceHistoryService
from investigator.domain.services.rl.feature_extractor import ValuationContextExtractor
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer

# Policy imports
from investigator.domain.services.rl.policy import (
    RLPolicy,
    ContextualBanditPolicy,
    HybridPolicy,
)

# Training imports
from investigator.domain.services.rl.training import (
    ExperienceCollector,
    RLTrainer,
    RLTrainingPipeline,
)

# Monitoring imports
from investigator.domain.services.rl.monitoring import (
    RLMetrics,
    ABTestingFramework,
)

# Reward calculation
from investigator.domain.services.rl.reward_calculator import (
    RewardCalculator,
    RewardComponents,
    get_reward_calculator,
    calculate_reward,
)

# Main integration service
from investigator.domain.services.rl.rl_model_weighting import RLModelWeightingService

__all__ = [
    # Models
    "ValuationContext",
    "Experience",
    "TrainingMetrics",
    "EvaluationMetrics",
    "ABTestResults",
    "RewardSignal",
    # Core services
    "OutcomeTracker",
    "PriceHistoryService",
    "ValuationContextExtractor",
    "FeatureNormalizer",
    # Policies
    "RLPolicy",
    "ContextualBanditPolicy",
    "HybridPolicy",
    # Training
    "ExperienceCollector",
    "RLTrainer",
    "RLTrainingPipeline",
    # Monitoring
    "RLMetrics",
    "ABTestingFramework",
    # Integration
    "RLModelWeightingService",
    # Reward calculation
    "RewardCalculator",
    "RewardComponents",
    "get_reward_calculator",
    "calculate_reward",
]
