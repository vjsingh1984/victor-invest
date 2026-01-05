#!/usr/bin/env python3
"""
Dual RL Policy Training Script

Trains two separate policies:
1. TechnicalRLPolicy: For timing and position decisions (Long/Short/Skip)
2. FundamentalRLPolicy: For model weights and holding periods

This separation allows each policy to specialize on its domain.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Check for MPS/PyTorch availability
try:
    import torch
    HAS_TORCH = True
    if torch.backends.mps.is_available():
        MPS_DEVICE = torch.device("mps")
        logger_init = logging.getLogger(__name__)
        logger_init.info("MPS (Apple Metal) backend available")
    else:
        MPS_DEVICE = torch.device("cpu")
except ImportError:
    HAS_TORCH = False
    MPS_DEVICE = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from investigator.domain.services.rl.policy import (
    TechnicalRLPolicy,
    FundamentalRLPolicy,
    DualRLPolicy,
)
from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize
from investigator.domain.services.rl.outcome_tracker import OutcomeTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data() -> List[Dict[str, Any]]:
    """Load training experiences from database using OutcomeTracker."""
    logger.info("Loading training experiences...")

    tracker = OutcomeTracker()
    raw_experiences = tracker.get_training_experiences(limit=50000, exclude_used=False)

    experiences = []
    for exp in raw_experiences:
        try:
            # Get context as dict
            context_dict = exp.context.to_dict() if hasattr(exp.context, 'to_dict') else {}

            # Calculate position signal from valuation gap
            if exp.current_price and exp.current_price > 0:
                val_gap = (exp.blended_fair_value - exp.current_price) / exp.current_price
            else:
                val_gap = 0

            # Determine actual position that was taken
            if val_gap > 0.05:
                position = 1  # Long
            elif val_gap < -0.05:
                position = -1  # Short
            else:
                position = 0  # Skip

            experiences.append({
                "symbol": exp.symbol,
                "analysis_date": exp.analysis_date,
                "fair_value": exp.blended_fair_value,
                "price": exp.current_price,
                "valuation_gap": val_gap,
                "position": position,
                "weights": exp.weights_used,
                "tier": exp.tier_classification,
                "context": context_dict,
                "reward_30d": exp.reward.reward_30d if exp.reward else None,
                "reward_90d": exp.reward.reward_90d if exp.reward else None,
                "reward_180d": exp.reward.reward_365d if exp.reward else None,  # Using 365 as proxy
                "reward_365d": exp.reward.reward_365d if exp.reward else None,
            })
        except Exception as e:
            logger.warning(f"Failed to parse experience: {e}")
            continue

    logger.info(f"Loaded {len(experiences)} experiences")
    return experiences


def create_context(exp: Dict[str, Any]) -> ValuationContext:
    """Create ValuationContext from experience dict."""
    ctx = exp.get("context", {})

    # Parse growth stage
    stage_str = ctx.get("growth_stage", "mature")
    try:
        growth_stage = GrowthStage(stage_str)
    except ValueError:
        growth_stage = GrowthStage.MATURE

    # Parse company size
    size_str = ctx.get("company_size", "mid_cap")
    try:
        company_size = CompanySize(size_str)
    except ValueError:
        company_size = CompanySize.MID_CAP

    return ValuationContext(
        symbol=exp.get("symbol", ""),
        analysis_date=exp.get("analysis_date"),
        sector=ctx.get("sector", "Unknown"),
        industry=ctx.get("industry", "Unknown"),
        growth_stage=growth_stage,
        company_size=company_size,
        profitability_score=ctx.get("profitability_score", 0.5),
        pe_level=ctx.get("pe_level", 0.5),
        revenue_growth=ctx.get("revenue_growth", 0.0),
        fcf_margin=ctx.get("fcf_margin", 0.0),
        rule_of_40_score=ctx.get("rule_of_40_score", 0.0),
        payout_ratio=ctx.get("payout_ratio", 0.0),
        debt_to_equity=ctx.get("debt_to_equity", 0.0),
        gross_margin=ctx.get("gross_margin", 0.0),
        operating_margin=ctx.get("operating_margin", 0.0),
        data_quality_score=ctx.get("data_quality_score", 50.0),
        quarters_available=ctx.get("quarters_available", 0),
        technical_trend=ctx.get("technical_trend", 0.0),
        market_sentiment=ctx.get("market_sentiment", 0.0),
        volatility=ctx.get("volatility", 0.5),
        rsi_14=ctx.get("rsi_14", 50.0),
        macd_histogram=ctx.get("macd_histogram", 0.0),
        obv_trend=ctx.get("obv_trend", 0.0),
        adx_14=ctx.get("adx_14", 25.0),
        stoch_k=ctx.get("stoch_k", 50.0),
        mfi_14=ctx.get("mfi_14", 50.0),
        entry_signal_strength=ctx.get("entry_signal_strength", 0.0),
        exit_signal_strength=ctx.get("exit_signal_strength", 0.0),
        signal_confluence=ctx.get("signal_confluence", 0.0),
        days_from_support=ctx.get("days_from_support", 0.5),
        risk_reward_ratio=ctx.get("risk_reward_ratio", 2.0),
        valuation_gap=exp.get("valuation_gap", 0.0),
        valuation_confidence=ctx.get("valuation_confidence", 0.5),
        position_signal=exp.get("position", 0),
    )


def train_technical_policy(
    experiences: List[Dict[str, Any]],
    epochs: int = 10,
) -> TechnicalRLPolicy:
    """Train the technical policy on position decisions."""
    logger.info("Training Technical Policy...")

    policy = TechnicalRLPolicy()

    # Track weight changes across epochs for learning diagnostics
    prev_mu_norm = np.linalg.norm(policy.mu)

    # Train for multiple epochs
    for epoch in range(epochs):
        np.random.shuffle(experiences)
        epoch_rewards = []
        action_counts = {0: 0, 1: 0, 2: 0}  # skip, long, short
        action_rewards = {0: [], 1: [], 2: []}

        for exp in experiences:
            context = create_context(exp)
            position = exp.get("position", 0)
            reward = exp.get("reward_90d", 0) or 0

            # Convert position to action: 0=skip, 1=long, 2=short
            if position == 0:
                action = 0
            elif position == 1:
                action = 1
            else:
                action = 2

            # Update policy
            policy.update(context, action, reward)
            epoch_rewards.append(reward)
            action_counts[action] += 1
            action_rewards[action].append(reward)

        # Calculate learning diagnostics
        avg_reward = np.mean(epoch_rewards)
        current_mu_norm = np.linalg.norm(policy.mu)
        weight_change = current_mu_norm - prev_mu_norm
        prev_mu_norm = current_mu_norm

        # Per-action reward means (model's actual learning signal)
        action_avgs = {
            a: np.mean(r) if r else 0.0
            for a, r in action_rewards.items()
        }

        logger.info(
            f"Technical Epoch {epoch + 1}/{epochs}: "
            f"dataset_avg={avg_reward:.4f}, "
            f"weight_delta={weight_change:.6f}, "
            f"skip_avg={action_avgs[0]:.4f}, "
            f"long_avg={action_avgs[1]:.4f}, "
            f"short_avg={action_avgs[2]:.4f}"
        )

    # Log action statistics
    stats = policy.get_action_stats()
    logger.info("Technical Policy Action Stats:")
    for action, data in stats.items():
        logger.info(f"  {action}: count={data['count']}, avg_reward={data['avg_reward']:.4f}")

    return policy


def train_fundamental_policy_batched_mps(
    experiences: List[Dict[str, Any]],
    epochs: int = 10,
    batch_size: int = 512,
) -> FundamentalRLPolicy:
    """Train fundamental policy using batched PyTorch with MPS acceleration."""
    if not HAS_TORCH:
        logger.warning("PyTorch not available, falling back to NumPy training")
        return train_fundamental_policy_numpy(experiences, epochs)

    logger.info(f"Training Fundamental Policy with MPS (batch_size={batch_size})...")
    logger.info(f"Device: {MPS_DEVICE}")

    policy = FundamentalRLPolicy()

    # Map reward periods to holding periods
    reward_to_holding = {
        "reward_30d": "1m",
        "reward_90d": "3m",
        "reward_180d": "6m",
        "reward_365d": "12m",
    }

    # Pre-extract all features and rewards to tensors
    logger.info("Pre-extracting features to tensors...")
    all_features = []
    all_rewards = []
    all_weights = []  # Store weight dicts for per-model updates
    all_contexts = []  # Store contexts for holding period updates

    for exp in experiences:
        context = create_context(exp)
        features = policy._extract_features(context)
        reward = exp.get("reward_90d", 0) or 0
        weights = exp.get("weights", {})

        all_features.append(features)
        all_rewards.append(reward)
        all_weights.append(weights)
        all_contexts.append((context, exp))  # Store for holding period updates

    # Convert to tensors on MPS device
    features_tensor = torch.tensor(np.array(all_features), dtype=torch.float32, device=MPS_DEVICE)
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=MPS_DEVICE)

    n_samples = len(experiences)
    n_batches = (n_samples + batch_size - 1) // batch_size

    logger.info(f"Training {n_samples} samples in {n_batches} batches")

    # Track weight norms for learning diagnostics
    prev_weight_mu_norm = np.linalg.norm(policy.weight_mu)

    for epoch in range(epochs):
        # Shuffle indices
        indices = torch.randperm(n_samples, device=MPS_DEVICE)

        epoch_rewards = []
        model_updates = {m: 0 for m in ["dcf", "pe", "ps", "ev_ebitda", "pb", "ggm"]}

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx].cpu().numpy()

            # Batched feature extraction (already on GPU)
            batch_features = features_tensor[batch_indices]
            batch_rewards = rewards_tensor[batch_indices]

            # Batch outer products on MPS (main computation)
            # outer_products: (batch_size, n_features, n_features)
            outer_products = torch.einsum('bi,bj->bij', batch_features, batch_features)

            # Sum outer products for batch update
            batch_outer_sum = outer_products.sum(dim=0).cpu().numpy()
            batch_feature_reward_sum = (batch_features * batch_rewards.unsqueeze(1)).sum(dim=0).cpu().numpy()

            # Update policy parameters for each model based on weights used
            for i, model in enumerate(["dcf", "pe", "ps", "ev_ebitda", "pb", "ggm"]):
                # Calculate weight contribution for this model in batch
                model_weight_sum = 0.0
                for idx in batch_indices:
                    weight = all_weights[idx].get(model, 0.0)
                    if weight > 0:
                        model_weight_sum += weight / 100.0
                        model_updates[model] += 1

                if model_weight_sum > 0:
                    scale = model_weight_sum / len(batch_indices)
                    # Bayesian update with scaled batch outer product
                    policy.weight_Lambda[i] += batch_outer_sum * scale / policy.noise_variance
                    policy.weight_Sigma[i] = np.linalg.inv(policy.weight_Lambda[i])
                    policy.weight_mu[i] = np.dot(
                        policy.weight_Sigma[i],
                        np.dot(policy.weight_Lambda[i], policy.weight_mu[i]) +
                        batch_feature_reward_sum * scale / policy.noise_variance
                    )

            # Update holding periods for each sample in batch
            for idx in batch_indices:
                context, exp = all_contexts[idx]
                for reward_key, holding_period in reward_to_holding.items():
                    reward = exp.get(reward_key)
                    if reward is not None:
                        policy.update_holding_period(context, holding_period, reward)

            epoch_rewards.extend(batch_rewards.cpu().numpy().tolist())

        # Calculate learning diagnostics
        avg_reward = np.mean(epoch_rewards)
        current_weight_mu_norm = np.linalg.norm(policy.weight_mu)
        weight_delta = current_weight_mu_norm - prev_weight_mu_norm
        prev_weight_mu_norm = current_weight_mu_norm

        # Log with more informative metrics
        top_models = sorted(model_updates.items(), key=lambda x: x[1], reverse=True)[:3]
        top_models_str = ", ".join([f"{m}:{c}" for m, c in top_models])

        logger.info(
            f"Fundamental Epoch {epoch + 1}/{epochs}: "
            f"dataset_avg={avg_reward:.4f}, "
            f"weight_delta={weight_delta:.6f}, "
            f"top_models=[{top_models_str}]"
        )

    policy._update_count = n_samples * epochs

    # Log model statistics
    stats = policy.get_model_stats()
    logger.info("Fundamental Policy Model Stats:")
    for model, data in stats.items():
        logger.info(f"  {model}: count={data['count']}, avg_reward={data['avg_reward']:.4f}")

    # Log holding period statistics
    hp_stats = policy.get_holding_period_stats()
    logger.info("Holding Period Stats:")
    for period, data in hp_stats.get("period_stats", {}).items():
        logger.info(f"  {period}: count={data['count']}, avg_reward={data['avg_reward']:.4f}")

    return policy


def train_fundamental_policy_numpy(
    experiences: List[Dict[str, Any]],
    epochs: int = 10,
) -> FundamentalRLPolicy:
    """Train fundamental policy using NumPy (fallback if no PyTorch)."""
    logger.info("Training Fundamental Policy (NumPy)...")

    policy = FundamentalRLPolicy()

    # Map reward periods to holding periods
    reward_to_holding = {
        "reward_30d": "1m",
        "reward_90d": "3m",
        "reward_180d": "6m",
        "reward_365d": "12m",
    }

    # Track weight changes
    prev_weight_mu_norm = np.linalg.norm(policy.weight_mu)

    # Train for multiple epochs
    for epoch in range(epochs):
        np.random.shuffle(experiences)
        epoch_rewards = []
        model_updates = {m: 0 for m in ["dcf", "pe", "ps", "ev_ebitda", "pb", "ggm"]}

        for exp in experiences:
            context = create_context(exp)
            weights = exp.get("weights", {})
            reward_90d = exp.get("reward_90d", 0) or 0

            # Track which models were updated
            for model, weight in weights.items():
                if weight > 0:
                    model_updates[model] = model_updates.get(model, 0) + 1

            # Update weights
            policy.update_weights(context, weights, reward_90d)

            # Update holding periods with all available rewards
            for reward_key, holding_period in reward_to_holding.items():
                reward = exp.get(reward_key)
                if reward is not None:
                    policy.update_holding_period(context, holding_period, reward)

            epoch_rewards.append(reward_90d)

        # Calculate learning diagnostics
        avg_reward = np.mean(epoch_rewards)
        current_weight_mu_norm = np.linalg.norm(policy.weight_mu)
        weight_delta = current_weight_mu_norm - prev_weight_mu_norm
        prev_weight_mu_norm = current_weight_mu_norm

        top_models = sorted(model_updates.items(), key=lambda x: x[1], reverse=True)[:3]
        top_models_str = ", ".join([f"{m}:{c}" for m, c in top_models])

        logger.info(
            f"Fundamental Epoch {epoch + 1}/{epochs}: "
            f"dataset_avg={avg_reward:.4f}, "
            f"weight_delta={weight_delta:.6f}, "
            f"top_models=[{top_models_str}]"
        )

    # Log model statistics
    stats = policy.get_model_stats()
    logger.info("Fundamental Policy Model Stats:")
    for model, data in stats.items():
        logger.info(f"  {model}: count={data['count']}, avg_reward={data['avg_reward']:.4f}")

    # Log holding period statistics
    hp_stats = policy.get_holding_period_stats()
    logger.info("Holding Period Stats:")
    for period, data in hp_stats.get("period_stats", {}).items():
        logger.info(f"  {period}: count={data['count']}, avg_reward={data['avg_reward']:.4f}")

    if hp_stats.get("sector_optimal_periods"):
        logger.info("Learned Sector Optimal Periods:")
        for sector, period in hp_stats["sector_optimal_periods"].items():
            logger.info(f"  {sector}: {period}")

    return policy


def train_fundamental_policy(
    experiences: List[Dict[str, Any]],
    epochs: int = 10,
    use_mps: bool = True,
    batch_size: int = 512,
) -> FundamentalRLPolicy:
    """Train fundamental policy with optional MPS acceleration."""
    if use_mps and HAS_TORCH:
        return train_fundamental_policy_batched_mps(experiences, epochs, batch_size)
    else:
        return train_fundamental_policy_numpy(experiences, epochs)


def evaluate_dual_policy(
    policy: DualRLPolicy,
    experiences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate the dual policy on held-out data."""
    logger.info("Evaluating Dual Policy...")

    position_correct = 0
    position_total = 0
    position_rewards = []
    weight_rewards = []

    for exp in experiences:
        context = create_context(exp)
        actual_position = exp.get("position", 0)
        actual_reward = exp.get("reward_90d", 0) or 0

        # Get prediction
        result = policy.predict_full(context)
        predicted_position = result["position"]

        # Evaluate position accuracy
        if predicted_position == actual_position:
            position_correct += 1
        position_total += 1

        # Track rewards
        if actual_position != 0:  # Only for non-skip positions
            position_rewards.append(actual_reward)
        weight_rewards.append(actual_reward)

    accuracy = position_correct / position_total if position_total > 0 else 0
    avg_position_reward = np.mean(position_rewards) if position_rewards else 0
    avg_weight_reward = np.mean(weight_rewards) if weight_rewards else 0

    return {
        "position_accuracy": accuracy,
        "avg_position_reward": avg_position_reward,
        "avg_weight_reward": avg_weight_reward,
        "total_samples": position_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dual RL Policy")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--eval-split", type=float, default=0.15, help="Evaluation split ratio")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rl_models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        help="Disable MPS (Apple Metal) acceleration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for MPS training (default: 512)",
    )
    args = parser.parse_args()

    # Log MPS availability
    if HAS_TORCH:
        logger.info(f"PyTorch available, MPS device: {MPS_DEVICE}")
    else:
        logger.info("PyTorch not available, using NumPy")

    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("DUAL RL POLICY TRAINING")
    logger.info(f"Started: {start_time}")
    logger.info("=" * 70)

    # Load data
    experiences = load_training_data()

    if len(experiences) < 100:
        logger.error(f"Not enough training data: {len(experiences)} < 100")
        return

    # Split into train/eval
    np.random.shuffle(experiences)
    split_idx = int(len(experiences) * (1 - args.eval_split))
    train_exp = experiences[:split_idx]
    eval_exp = experiences[split_idx:]

    logger.info(f"Training set: {len(train_exp)} samples")
    logger.info(f"Evaluation set: {len(eval_exp)} samples")

    # Train both policies
    technical_policy = train_technical_policy(train_exp, epochs=args.epochs)
    fundamental_policy = train_fundamental_policy(
        train_exp,
        epochs=args.epochs,
        use_mps=not args.no_mps,
        batch_size=args.batch_size,
    )

    # Create combined dual policy
    dual_policy = DualRLPolicy(
        technical_policy=technical_policy,
        fundamental_policy=fundamental_policy,
        technical_path=os.path.join(args.output_dir, "technical_policy.pkl"),
        fundamental_path=os.path.join(args.output_dir, "fundamental_policy.pkl"),
    )

    # Evaluate
    eval_results = evaluate_dual_policy(dual_policy, eval_exp)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Position Accuracy: {eval_results['position_accuracy']:.1%}")
    logger.info(f"Avg Position Reward: {eval_results['avg_position_reward']:.4f}")
    logger.info(f"Avg Weight Reward: {eval_results['avg_weight_reward']:.4f}")

    # Save policies
    os.makedirs(args.output_dir, exist_ok=True)
    dual_policy.save()

    # Save training log
    log_path = os.path.join(args.output_dir, "dual_training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "epochs": args.epochs,
            "train_size": len(train_exp),
            "eval_size": len(eval_exp),
            "technical_updates": technical_policy._update_count,
            "fundamental_updates": fundamental_policy._update_count,
            "evaluation": eval_results,
            "technical_action_stats": technical_policy.get_action_stats(),
            "fundamental_model_stats": fundamental_policy.get_model_stats(),
            "holding_period_stats": fundamental_policy.get_holding_period_stats(),
        }, f, indent=2, default=str)

    logger.info(f"Saved training log to {log_path}")

    # Summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Technical Policy: {technical_policy._update_count} updates")
    logger.info(f"Fundamental Policy: {fundamental_policy._update_count} updates")
    logger.info(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
