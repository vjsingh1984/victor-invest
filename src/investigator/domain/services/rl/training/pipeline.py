"""
RL Training Pipeline

End-to-end training pipeline that coordinates:
1. Experience collection from outcome database
2. Feature extraction and normalization
3. Policy training
4. Model evaluation and checkpointing
5. A/B testing against baseline

Usage:
    from investigator.domain.services.rl.training import RLTrainingPipeline

    pipeline = RLTrainingPipeline(config)
    result = pipeline.run()
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import (
    EvaluationMetrics,
    TrainingMetrics,
)
from investigator.domain.services.rl.policy.base import RLPolicy, UniformPolicy
from investigator.domain.services.rl.policy.contextual_bandit import ContextualBanditPolicy
from investigator.domain.services.rl.policy.hybrid import HybridPolicy
from investigator.domain.services.rl.training.experience_collector import ExperienceCollector
from investigator.domain.services.rl.training.trainer import RLTrainer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for training pipeline."""

    # Policy settings
    policy_type: str = "contextual_bandit"  # contextual_bandit, hybrid
    policy_path: str = "data/rl_models/policy.pkl"
    normalizer_path: str = "data/rl_models/normalizer.pkl"

    # Training settings
    min_experiences: int = 100
    max_experiences: int = 10000
    epochs: int = 10
    batch_size: int = 32
    early_stopping_patience: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Experience collection
    min_days_ago: int = 90
    exclude_used: bool = False

    # Hybrid policy settings
    max_adjustment: float = 0.30

    # Output settings
    checkpoint_dir: str = "data/rl_models/checkpoints"
    save_metrics: bool = True


@dataclass
class PipelineResult:
    """Result of training pipeline execution."""

    success: bool
    training_metrics: Optional[TrainingMetrics]
    evaluation_metrics: Optional[EvaluationMetrics]
    baseline_comparison: Optional[Dict[str, float]]
    policy_path: Optional[str]
    normalizer_path: Optional[str]
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0


class RLTrainingPipeline:
    """
    End-to-end RL training pipeline.

    Orchestrates the full training workflow:
    1. Collect experiences from database
    2. Split into train/val/test
    3. Fit normalizer
    4. Train policy
    5. Evaluate on test set
    6. Compare to baseline
    7. Save artifacts
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        base_weighting_service: Optional[Any] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration.
            base_weighting_service: DynamicModelWeightingService for hybrid policy.
        """
        self.config = config or PipelineConfig()
        self.base_weighting_service = base_weighting_service

        # Initialize components
        self.collector = ExperienceCollector()
        self.normalizer = FeatureNormalizer()
        self.policy: Optional[RLPolicy] = None
        self.trainer: Optional[RLTrainer] = None

    def run(self) -> PipelineResult:
        """
        Execute the full training pipeline.

        Returns:
            PipelineResult with training outcomes.
        """
        start_time = datetime.now()

        try:
            # Step 1: Collect experiences
            logger.info("Step 1: Collecting experiences...")
            experiences = self.collector.collect_experiences(
                min_days_ago=self.config.min_days_ago,
                max_experiences=self.config.max_experiences,
                exclude_used=self.config.exclude_used,
            )

            if len(experiences) < self.config.min_experiences:
                return PipelineResult(
                    success=False,
                    training_metrics=None,
                    evaluation_metrics=None,
                    baseline_comparison=None,
                    policy_path=None,
                    normalizer_path=None,
                    error_message=f"Insufficient experiences: {len(experiences)} < {self.config.min_experiences}",
                )

            logger.info(f"Collected {len(experiences)} experiences")

            # Step 2: Split data
            logger.info("Step 2: Splitting data...")
            train, val, test = self.collector.stratified_split(
                experiences,
                stratify_by="sector",
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
            )

            # Step 3: Create policy
            logger.info("Step 3: Creating policy...")
            self.policy = self._create_policy()

            # Step 4: Create trainer and train
            logger.info("Step 4: Training policy...")
            self.trainer = RLTrainer(
                policy=self.policy,
                normalizer=self.normalizer,
                checkpoint_dir=self.config.checkpoint_dir,
            )

            training_metrics = self.trainer.train_batch(
                experiences=train + val,  # Trainer handles internal val split
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                early_stopping_patience=self.config.early_stopping_patience,
            )

            # Step 5: Evaluate on test set
            logger.info("Step 5: Evaluating on test set...")
            evaluation_metrics = self.trainer.evaluate(test)

            # Step 6: Compare to baseline
            logger.info("Step 6: Comparing to baseline...")
            baseline = UniformPolicy()
            baseline_comparison = self.trainer.compare_to_baseline(test, baseline)

            # Step 7: Save artifacts
            logger.info("Step 7: Saving artifacts...")
            self._save_artifacts()

            execution_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Pipeline complete: reward={evaluation_metrics.mean_reward:.3f}, "
                f"improvement={baseline_comparison.get('reward_improvement_pct', 0):.1f}%"
            )

            return PipelineResult(
                success=True,
                training_metrics=training_metrics,
                evaluation_metrics=evaluation_metrics,
                baseline_comparison=baseline_comparison,
                policy_path=self.config.policy_path,
                normalizer_path=self.config.normalizer_path,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                training_metrics=None,
                evaluation_metrics=None,
                baseline_comparison=None,
                policy_path=None,
                normalizer_path=None,
                error_message=str(e),
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _create_policy(self) -> RLPolicy:
        """Create policy based on configuration."""
        if self.config.policy_type == "hybrid":
            adjustment_policy = ContextualBanditPolicy(normalizer=self.normalizer)
            return HybridPolicy(
                base_weighting_service=self.base_weighting_service,
                adjustment_policy=adjustment_policy,
                max_adjustment=self.config.max_adjustment,
                normalizer=self.normalizer,
            )
        else:  # contextual_bandit (default)
            return ContextualBanditPolicy(normalizer=self.normalizer)

    def _save_artifacts(self) -> None:
        """Save trained policy and normalizer."""
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config.policy_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.config.normalizer_path) or ".", exist_ok=True)

        # Save policy
        if self.policy:
            self.policy.save(self.config.policy_path)
            logger.info(f"Saved policy to {self.config.policy_path}")

        # Save normalizer
        if self.normalizer:
            self.normalizer.save(self.config.normalizer_path)
            logger.info(f"Saved normalizer to {self.config.normalizer_path}")

    def load_existing(self) -> bool:
        """
        Load existing policy and normalizer.

        Returns:
            True if loaded successfully.
        """
        try:
            self.policy = self._create_policy()
            policy_loaded = self.policy.load(self.config.policy_path)

            self.normalizer = FeatureNormalizer()
            normalizer_loaded = self.normalizer.load(self.config.normalizer_path)

            if policy_loaded:
                self.policy.normalizer = self.normalizer

            return policy_loaded and normalizer_loaded

        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
            return False

    def incremental_update(
        self,
        new_experiences_limit: int = 100,
    ) -> Optional[Dict[str, float]]:
        """
        Perform incremental update with recent experiences.

        For online learning: update policy with new experiences
        without full retraining.

        Args:
            new_experiences_limit: Maximum new experiences to process.

        Returns:
            Dict with update statistics, or None if failed.
        """
        if not self.policy:
            if not self.load_existing():
                logger.warning("No existing policy to update")
                return None

        # Get recent unused experiences
        experiences = self.collector.collect_recent(
            days=30,
            max_experiences=new_experiences_limit,
        )

        if not experiences:
            return {"updated": 0, "message": "No new experiences"}

        # Online update
        updated = 0
        for exp in experiences:
            if exp.reward.primary_reward is not None:
                self.trainer = self.trainer or RLTrainer(self.policy)
                self.trainer.train_online(exp)
                updated += 1

        # Save updated policy
        if updated > 0:
            self._save_artifacts()

        return {
            "updated": updated,
            "total_available": len(experiences),
        }


# Factory function
def get_training_pipeline(
    config: Optional[PipelineConfig] = None,
) -> RLTrainingPipeline:
    """Get RLTrainingPipeline instance."""
    return RLTrainingPipeline(config)
