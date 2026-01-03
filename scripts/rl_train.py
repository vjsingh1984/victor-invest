#!/usr/bin/env python3
"""
RL Policy Training Script

Train the RL policy on historical valuation outcomes.
Run this script periodically (e.g., weekly) after new outcome data is available.

Usage:
    python scripts/rl_train.py                    # Train with defaults
    python scripts/rl_train.py --epochs 30        # More epochs
    python scripts/rl_train.py --min-samples 100  # Require more samples
    python scripts/rl_train.py --deploy           # Deploy after training

Environment:
    PYTHONPATH=./src:. python scripts/rl_train.py
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from investigator.domain.services.rl.outcome_tracker import OutcomeTracker
from investigator.domain.services.rl.policy.contextual_bandit import ContextualBanditPolicy
from investigator.domain.services.rl.training.trainer import RLTrainer
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path("data/rl_models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
POLICY_PATH = MODEL_DIR / "policy.pkl"
NORMALIZER_PATH = MODEL_DIR / "normalizer.pkl"
TRAINING_LOG_PATH = MODEL_DIR / "training_log.json"


def load_experiences(min_samples: int = 50) -> list:
    """Load training experiences from database."""
    logger.info("Loading training experiences...")
    tracker = OutcomeTracker()
    experiences = tracker.get_training_experiences(limit=50000, exclude_used=False)

    if len(experiences) < min_samples:
        logger.error(f"Not enough experiences: {len(experiences)} < {min_samples}")
        sys.exit(1)

    logger.info(f"Loaded {len(experiences)} experiences")
    return experiences


def analyze_experiences(experiences: list) -> dict:
    """Analyze experience distribution."""
    tier_counts = {}
    tier_rewards = {}

    for exp in experiences:
        tier = exp.tier_classification
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if tier not in tier_rewards:
            tier_rewards[tier] = []
        if exp.reward.primary_reward is not None:
            tier_rewards[tier].append(exp.reward.primary_reward)

    analysis = {
        "total_experiences": len(experiences),
        "tier_distribution": {},
    }

    for tier in sorted(tier_counts.keys(), key=lambda t: tier_counts[t], reverse=True):
        avg = np.mean(tier_rewards[tier]) if tier_rewards[tier] else 0
        analysis["tier_distribution"][tier] = {
            "count": tier_counts[tier],
            "avg_reward": round(avg, 3),
        }

    return analysis


def train_policy(
    experiences: list,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.15,
    early_stopping_patience: int = 5,
    resume_from: str = None,
) -> tuple:
    """Train the RL policy.

    Args:
        experiences: List of training experiences
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Fraction for validation
        early_stopping_patience: Epochs to wait for improvement
        resume_from: Path to existing policy to resume training from (incremental learning)
    """
    logger.info("Initializing policy and trainer...")

    normalizer = FeatureNormalizer()

    if resume_from and Path(resume_from).exists():
        # Load existing policy for incremental training
        logger.info(f"Loading existing policy from {resume_from} for incremental training...")
        policy = ContextualBanditPolicy(
            n_features=None,
            prior_variance=1.0,
            noise_variance=0.1,
            exploration_weight=0.5,  # Lower exploration for fine-tuning
            normalizer=normalizer,
        )
        policy.load(resume_from)

        # Also load the normalizer if it exists
        normalizer_path = resume_from.replace("policy.pkl", "normalizer.pkl")
        if Path(normalizer_path).exists():
            normalizer.load(normalizer_path)
            logger.info(f"Loaded normalizer from {normalizer_path}")
        logger.info("Resuming training from existing policy (incremental learning)")
    else:
        # Create new policy from scratch
        policy = ContextualBanditPolicy(
            n_features=None,
            prior_variance=1.0,
            noise_variance=0.1,
            exploration_weight=1.0,
            normalizer=normalizer,
        )

    trainer = RLTrainer(
        policy=policy,
        normalizer=normalizer,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    logger.info(f"Training policy for {epochs} epochs...")
    metrics = trainer.train_batch(
        experiences=experiences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        checkpoint_frequency=5,
        verbose=True,
    )

    logger.info("Evaluating trained policy...")
    eval_metrics = trainer.evaluate(experiences)

    return policy, normalizer, metrics, eval_metrics


def save_policy(policy, normalizer, metrics, eval_metrics, analysis: dict):
    """Save trained policy and training log."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save policy and normalizer
    policy.save(str(POLICY_PATH))
    normalizer.save(str(NORMALIZER_PATH))
    logger.info(f"Saved policy to {POLICY_PATH}")
    logger.info(f"Saved normalizer to {NORMALIZER_PATH}")

    # Save training log
    training_log = {
        "training_date": datetime.now().isoformat(),
        "num_experiences": analysis["total_experiences"],
        "tier_distribution": analysis["tier_distribution"],
        "training_metrics": {
            "epochs_completed": metrics.epochs_completed,
            "early_stopped": metrics.early_stopped,
            "best_epoch": metrics.best_epoch,
            "train_reward_mean": round(metrics.train_reward_mean, 4),
            "validation_reward_mean": round(metrics.validation_reward_mean, 4),
        },
        "evaluation_metrics": {
            "mape": round(eval_metrics.mape, 2),
            "direction_accuracy": round(eval_metrics.direction_accuracy, 4),
            "mean_reward": round(eval_metrics.mean_reward, 4),
            "median_reward": round(eval_metrics.median_reward, 4),
        },
        "action_stats": policy.get_action_stats(),
    }

    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Saved training log to {TRAINING_LOG_PATH}")

    return training_log


def deploy_policy():
    """Deploy the trained policy (copy to active location)."""
    active_policy_path = MODEL_DIR / "active_policy.pkl"
    active_normalizer_path = MODEL_DIR / "active_normalizer.pkl"

    if not POLICY_PATH.exists():
        logger.error(f"No trained policy found at {POLICY_PATH}")
        return False

    import shutil

    shutil.copy(POLICY_PATH, active_policy_path)
    shutil.copy(NORMALIZER_PATH, active_normalizer_path)

    logger.info(f"Deployed policy to {active_policy_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train RL policy on valuation outcomes")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples required")
    parser.add_argument("--deploy", action="store_true", help="Deploy after training")
    parser.add_argument("--validation-split", type=float, default=0.15, help="Validation split")
    parser.add_argument("--resume", action="store_true", help="Resume training from deployed active policy (incremental learning)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to policy file to resume from")
    args = parser.parse_args()

    print("=" * 70)
    print("RL POLICY TRAINING")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load and analyze experiences
    experiences = load_experiences(args.min_samples)
    analysis = analyze_experiences(experiences)

    print(f"\nExperience Distribution (top 10):")
    for tier, data in list(analysis["tier_distribution"].items())[:10]:
        print(f"  {tier}: {data['count']} samples, avg_reward={data['avg_reward']}")

    # Train policy
    # Determine resume path
    resume_path = None
    if args.resume:
        resume_path = str(MODEL_DIR / "active_policy.pkl")
        print(f"Resuming from active policy: {resume_path}")
    elif args.resume_from:
        resume_path = args.resume_from
        print(f"Resuming from: {resume_path}")

    policy, normalizer, metrics, eval_metrics = train_policy(
        experiences=experiences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        resume_from=resume_path,
    )

    # Save results
    training_log = save_policy(policy, normalizer, metrics, eval_metrics, analysis)

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Epochs completed: {metrics.epochs_completed}")
    print(f"Early stopped: {metrics.early_stopped}")
    print(f"Train reward mean: {metrics.train_reward_mean:.3f}")
    print(f"Validation reward mean: {metrics.validation_reward_mean:.3f}")
    print(f"Evaluation MAPE: {eval_metrics.mape:.1f}%")
    print(f"Direction accuracy: {eval_metrics.direction_accuracy:.1%}")

    # Deploy if requested
    if args.deploy:
        print("\nDeploying trained policy...")
        if deploy_policy():
            print("Policy deployed successfully!")
        else:
            print("Deployment failed!")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
