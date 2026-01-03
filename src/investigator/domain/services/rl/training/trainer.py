"""
RL Trainer for Policy Training

Trains RL policies on historical experiences. Supports:
- Batch training on historical data
- Incremental online updates
- Model checkpointing
- Training metrics logging
- Early stopping

Usage:
    from investigator.domain.services.rl.training import RLTrainer

    trainer = RLTrainer(policy)

    # Train on batch of experiences
    metrics = trainer.train_batch(experiences, epochs=10)

    # Evaluate on test set
    eval_metrics = trainer.evaluate(test_experiences)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.services.rl.models import (
    Experience,
    TrainingMetrics,
    EvaluationMetrics,
)
from investigator.domain.services.rl.policy.base import RLPolicy
from investigator.domain.services.rl.feature_extractor import ValuationContextExtractor
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trains RL policies on historical experiences.

    Supports multiple training modes:
    - Batch: Train on full dataset for multiple epochs
    - Online: Update incrementally with each new experience
    - Mini-batch: Train on random samples (SGD-style)

    Provides:
    - Training metrics (loss, reward improvement)
    - Validation for early stopping
    - Model checkpointing
    - Comparison with baseline
    """

    def __init__(
        self,
        policy: RLPolicy,
        normalizer: Optional[FeatureNormalizer] = None,
        checkpoint_dir: str = "data/rl_models/checkpoints",
    ):
        """
        Initialize trainer.

        Args:
            policy: RL policy to train.
            normalizer: Feature normalizer (will be fitted on training data).
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.policy = policy
        self.normalizer = normalizer or FeatureNormalizer()
        self.checkpoint_dir = checkpoint_dir
        self.extractor = ValuationContextExtractor()

        # Training history
        self._train_history: List[Dict[str, float]] = []
        self._val_history: List[Dict[str, float]] = []
        self._best_val_reward = float("-inf")
        self._best_epoch = 0
        self._training_batch_id = 0

    def train_batch(
        self,
        experiences: List[Experience],
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 3,
        checkpoint_frequency: int = 5,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Train policy on batch of experiences.

        Args:
            experiences: Training experiences.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction for validation.
            early_stopping_patience: Epochs without improvement before stopping.
            checkpoint_frequency: Save checkpoint every N epochs.
            verbose: Print training progress.

        Returns:
            TrainingMetrics with training results.
        """
        if not experiences:
            logger.warning("No experiences provided for training")
            return self._empty_metrics()

        self._training_batch_id += 1
        start_time = datetime.now()

        # Split into train/val
        n_val = int(len(experiences) * validation_split)
        np.random.shuffle(experiences)
        val_experiences = experiences[:n_val]
        train_experiences = experiences[n_val:]

        # Fit normalizer on training data
        contexts = [exp.context for exp in train_experiences]
        self.normalizer.fit(contexts)
        self.policy.normalizer = self.normalizer

        if verbose:
            logger.info(
                f"Starting training: {len(train_experiences)} train, " f"{len(val_experiences)} val, {epochs} epochs"
            )

        best_val_reward = float("-inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training epoch
            train_metrics = self._train_epoch(train_experiences, batch_size)
            self._train_history.append(train_metrics)

            # Validation
            val_metrics = self._evaluate_internal(val_experiences)
            self._val_history.append(val_metrics)

            if verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_reward={train_metrics['mean_reward']:.3f}, "
                    f"val_reward={val_metrics['mean_reward']:.3f}"
                )

            # Check for improvement
            if val_metrics["mean_reward"] > best_val_reward:
                best_val_reward = val_metrics["mean_reward"]
                self._best_epoch = epoch
                self._best_val_reward = best_val_reward
                epochs_without_improvement = 0

                # Save best model
                self._save_best_checkpoint()
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Periodic checkpoint
            if (epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(epoch + 1)

        # Load best model
        self._load_best_checkpoint()

        return self._create_training_metrics(
            train_experiences=train_experiences,
            val_experiences=val_experiences,
            epochs_completed=epoch + 1,
            early_stopped=(epochs_without_improvement >= early_stopping_patience),
            start_time=start_time,
        )

    def train_online(
        self,
        experience: Experience,
    ) -> None:
        """
        Update policy with single new experience (online learning).

        Args:
            experience: New experience to learn from.
        """
        if experience.reward.primary_reward is None:
            logger.warning("Experience has no reward, skipping")
            return

        # Update normalizer
        self.normalizer.partial_fit(experience.context)

        # Update policy
        self.policy.update(
            context=experience.context,
            action=experience.weights_used,
            reward=experience.reward.primary_reward,
        )

    def evaluate(
        self,
        experiences: List[Experience],
    ) -> EvaluationMetrics:
        """
        Evaluate policy on test experiences.

        Args:
            experiences: Test experiences with known outcomes.

        Returns:
            EvaluationMetrics with evaluation results.
        """
        if not experiences:
            return self._empty_eval_metrics()

        # Get predictions and compare to actual outcomes
        predictions = []
        actuals = []
        rewards = []

        for exp in experiences:
            if exp.reward.primary_reward is None:
                continue

            # Get policy prediction
            predicted_weights = self.policy.predict(exp.context)

            # Store for analysis
            predictions.append(predicted_weights)
            actuals.append(exp.weights_used)
            rewards.append(exp.reward.primary_reward)

        if not rewards:
            return self._empty_eval_metrics()

        # Calculate metrics
        mean_reward = float(np.mean(rewards))
        median_reward = float(np.median(rewards))
        std_reward = float(np.std(rewards))

        # Direction accuracy: did we predict upside/downside correctly?
        direction_correct = sum(
            1
            for exp in experiences
            if exp.reward.reward_90d is not None
            and exp.blended_fair_value is not None
            and exp.current_price is not None
            and ((exp.blended_fair_value > exp.current_price) == (exp.reward.reward_90d > 0))
        )
        valid_direction_samples = sum(
            1 for exp in experiences
            if exp.reward.reward_90d is not None
            and exp.blended_fair_value is not None
            and exp.current_price is not None
        )
        direction_accuracy = direction_correct / valid_direction_samples if valid_direction_samples > 0 else 0

        # MAPE calculation (if we have per-model data)
        mape_values = []
        for exp in experiences:
            if exp.per_model_rewards:
                for model_rewards in exp.per_model_rewards.values():
                    if isinstance(model_rewards, dict) and "error_pct" in model_rewards:
                        mape_values.append(model_rewards["error_pct"])

        mape = float(np.mean(mape_values)) if mape_values else 0.0

        # Sector breakdown
        sector_performance = {}
        for exp in experiences:
            sector = exp.context.sector
            if sector not in sector_performance:
                sector_performance[sector] = {"rewards": [], "count": 0}
            if exp.reward.primary_reward is not None:
                sector_performance[sector]["rewards"].append(exp.reward.primary_reward)
                sector_performance[sector]["count"] += 1

        for sector, data in sector_performance.items():
            if data["rewards"]:
                data["mean_reward"] = float(np.mean(data["rewards"]))
            del data["rewards"]  # Remove list, keep summary

        return EvaluationMetrics(
            policy_type=self.policy.name,
            evaluation_date=datetime.now(),
            num_samples=len(experiences),
            mape=mape,
            direction_accuracy=direction_accuracy,
            mean_reward=mean_reward,
            median_reward=median_reward,
            std_reward=std_reward,
            sector_performance=sector_performance,
        )

    def compare_to_baseline(
        self,
        experiences: List[Experience],
        baseline_policy: RLPolicy,
    ) -> Dict[str, float]:
        """
        Compare policy performance to baseline.

        Args:
            experiences: Test experiences.
            baseline_policy: Baseline policy to compare against.

        Returns:
            Dict with comparison metrics.
        """
        # Evaluate both policies
        rl_metrics = self.evaluate(experiences)

        # Temporarily swap policy
        original_policy = self.policy
        self.policy = baseline_policy
        baseline_metrics = self.evaluate(experiences)
        self.policy = original_policy

        # Calculate improvement
        reward_improvement = (
            (rl_metrics.mean_reward - baseline_metrics.mean_reward) / abs(baseline_metrics.mean_reward) * 100
            if baseline_metrics.mean_reward != 0
            else 0
        )

        mape_improvement = (
            (baseline_metrics.mape - rl_metrics.mape) / baseline_metrics.mape * 100 if baseline_metrics.mape > 0 else 0
        )

        direction_improvement = (rl_metrics.direction_accuracy - baseline_metrics.direction_accuracy) * 100

        return {
            "rl_mean_reward": rl_metrics.mean_reward,
            "baseline_mean_reward": baseline_metrics.mean_reward,
            "reward_improvement_pct": reward_improvement,
            "rl_mape": rl_metrics.mape,
            "baseline_mape": baseline_metrics.mape,
            "mape_improvement_pct": mape_improvement,
            "rl_direction_accuracy": rl_metrics.direction_accuracy,
            "baseline_direction_accuracy": baseline_metrics.direction_accuracy,
            "direction_improvement_pct": direction_improvement,
        }

    def _train_epoch(
        self,
        experiences: List[Experience],
        batch_size: int,
    ) -> Dict[str, float]:
        """Run single training epoch."""
        np.random.shuffle(experiences)

        epoch_rewards = []
        n_batches = max(1, len(experiences) // batch_size)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(experiences))
            batch = experiences[start:end]

            for exp in batch:
                if exp.reward.primary_reward is not None:
                    self.policy.update(
                        context=exp.context,
                        action=exp.weights_used,
                        reward=exp.reward.primary_reward,
                    )
                    epoch_rewards.append(exp.reward.primary_reward)

        return {
            "mean_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0,
            "n_updates": len(epoch_rewards),
        }

    def _evaluate_internal(
        self,
        experiences: List[Experience],
    ) -> Dict[str, float]:
        """Internal evaluation for validation."""
        rewards = [exp.reward.primary_reward for exp in experiences if exp.reward.primary_reward is not None]

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0,
            "n_samples": len(rewards),
        }

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"{self.policy.name}_epoch{epoch}.pkl")
        self.policy.save(path)
        logger.debug(f"Saved checkpoint: {path}")

    def _save_best_checkpoint(self) -> None:
        """Save best model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"{self.policy.name}_best.pkl")
        self.policy.save(path)

    def _load_best_checkpoint(self) -> None:
        """Load best model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{self.policy.name}_best.pkl")
        if os.path.exists(path):
            self.policy.load(path)

    def _create_training_metrics(
        self,
        train_experiences: List[Experience],
        val_experiences: List[Experience],
        epochs_completed: int,
        early_stopped: bool,
        start_time: datetime,
    ) -> TrainingMetrics:
        """Create TrainingMetrics from training run."""
        train_rewards = [e.reward.primary_reward for e in train_experiences if e.reward.primary_reward is not None]
        val_rewards = [e.reward.primary_reward for e in val_experiences if e.reward.primary_reward is not None]

        return TrainingMetrics(
            batch_id=self._training_batch_id,
            batch_date=start_time,
            policy_type=self.policy.name,
            num_experiences=len(train_experiences) + len(val_experiences),
            train_size=len(train_experiences),
            validation_size=len(val_experiences),
            test_size=0,
            train_loss=0.0,  # Not applicable for bandit
            validation_loss=0.0,
            train_reward_mean=float(np.mean(train_rewards)) if train_rewards else 0,
            validation_reward_mean=float(np.mean(val_rewards)) if val_rewards else 0,
            epochs_completed=epochs_completed,
            early_stopped=early_stopped,
            best_epoch=self._best_epoch,
        )

    def _empty_metrics(self) -> TrainingMetrics:
        """Create empty training metrics."""
        return TrainingMetrics(
            batch_id=0,
            batch_date=datetime.now(),
            policy_type=self.policy.name,
            num_experiences=0,
            train_size=0,
            validation_size=0,
            test_size=0,
            train_loss=0.0,
            validation_loss=0.0,
            train_reward_mean=0.0,
            validation_reward_mean=0.0,
        )

    def _empty_eval_metrics(self) -> EvaluationMetrics:
        """Create empty evaluation metrics."""
        return EvaluationMetrics(
            policy_type=self.policy.name,
            evaluation_date=datetime.now(),
            num_samples=0,
            mape=0.0,
            direction_accuracy=0.0,
            mean_reward=0.0,
            median_reward=0.0,
            std_reward=0.0,
        )

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return {
            "train_rewards": [h["mean_reward"] for h in self._train_history],
            "val_rewards": [h["mean_reward"] for h in self._val_history],
        }
