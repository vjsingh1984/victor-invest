"""
Experience Collector for RL Training

Collects training experiences from the valuation outcomes database.
Experiences consist of (state, action, reward) tuples used to train
RL policies.

Usage:
    from investigator.domain.services.rl.training import ExperienceCollector

    collector = ExperienceCollector()

    # Collect experiences with 90+ days of outcome data
    experiences = collector.collect_experiences(min_days_ago=90)

    # Split for training
    train, val, test = collector.train_val_test_split(
        experiences,
        train_ratio=0.7,
        val_ratio=0.15,
    )
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.services.rl.models import (
    Experience,
    RewardSignal,
    ValuationContext,
)
from investigator.domain.services.rl.outcome_tracker import (
    OutcomeTracker,
    ValuationOutcomesDAO,
)

logger = logging.getLogger(__name__)


class ExperienceCollector:
    """
    Collects experiences from the outcome database for RL training.

    Experiences are (state, action, reward) tuples where:
    - State: ValuationContext features
    - Action: Model weights that were used
    - Reward: Calculated reward based on prediction accuracy

    Provides methods for:
    - Collecting all available experiences
    - Filtering by sector, tier, date range
    - Splitting into train/val/test sets
    - Sampling strategies (uniform, weighted by sector, etc.)
    """

    def __init__(
        self,
        outcome_tracker: Optional[OutcomeTracker] = None,
        dao: Optional[ValuationOutcomesDAO] = None,
    ):
        """
        Initialize experience collector.

        Args:
            outcome_tracker: OutcomeTracker instance.
            dao: Direct DAO access (if tracker not provided).
        """
        self.outcome_tracker = outcome_tracker or OutcomeTracker()
        self.dao = dao or ValuationOutcomesDAO()

    def collect_experiences(
        self,
        min_days_ago: int = 90,
        max_experiences: int = 10000,
        exclude_used: bool = False,
        min_reward: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        tiers: Optional[List[str]] = None,
    ) -> List[Experience]:
        """
        Collect experiences from the outcome database.

        Args:
            min_days_ago: Only include predictions at least this old.
            max_experiences: Maximum number of experiences to return.
            exclude_used: If True, exclude experiences already used for training.
            min_reward: Only include experiences with reward >= this value.
            sectors: Filter to specific sectors.
            tiers: Filter to specific tier classifications.

        Returns:
            List of Experience objects.
        """
        # Get raw experiences from tracker
        experiences = self.outcome_tracker.get_training_experiences(
            limit=max_experiences,
            exclude_used=exclude_used,
        )

        # Filter by criteria
        filtered = []
        for exp in experiences:
            # Check date
            days_old = (date.today() - exp.analysis_date).days
            if days_old < min_days_ago:
                continue

            # Check reward threshold
            if min_reward is not None and exp.reward.primary_reward is not None:
                if exp.reward.primary_reward < min_reward:
                    continue

            # Check sector filter
            if sectors and exp.context.sector not in sectors:
                continue

            # Check tier filter
            if tiers and exp.tier_classification not in tiers:
                continue

            filtered.append(exp)

        logger.info(
            f"Collected {len(filtered)} experiences " f"(filtered from {len(experiences)}, min_days={min_days_ago})"
        )

        return filtered

    def collect_by_sector(
        self,
        min_per_sector: int = 10,
        max_per_sector: int = 500,
    ) -> Dict[str, List[Experience]]:
        """
        Collect experiences grouped by sector.

        Useful for sector-specific model training or analysis.

        Args:
            min_per_sector: Minimum experiences required per sector.
            max_per_sector: Maximum experiences per sector.

        Returns:
            Dict mapping sector names to experience lists.
        """
        all_experiences = self.collect_experiences(max_experiences=10000)

        by_sector: Dict[str, List[Experience]] = {}
        for exp in all_experiences:
            sector = exp.context.sector
            if sector not in by_sector:
                by_sector[sector] = []
            if len(by_sector[sector]) < max_per_sector:
                by_sector[sector].append(exp)

        # Filter sectors with too few experiences
        result = {sector: exps for sector, exps in by_sector.items() if len(exps) >= min_per_sector}

        logger.info(f"Collected experiences for {len(result)} sectors " f"(min {min_per_sector} per sector)")

        return result

    def collect_recent(
        self,
        days: int = 30,
        max_experiences: int = 1000,
    ) -> List[Experience]:
        """
        Collect recent experiences for online learning.

        Note: Recent experiences may not have full outcome data yet.
        Use with caution - rewards may only be from 30-day outcomes.

        Args:
            days: Collect experiences from last N days.
            max_experiences: Maximum to return.

        Returns:
            List of recent experiences.
        """
        experiences = self.outcome_tracker.get_training_experiences(
            limit=max_experiences,
            exclude_used=False,
        )

        cutoff = date.today() - timedelta(days=days)
        recent = [exp for exp in experiences if exp.analysis_date >= cutoff]

        logger.info(f"Collected {len(recent)} recent experiences (last {days} days)")
        return recent

    def train_val_test_split(
        self,
        experiences: List[Experience],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[Experience], List[Experience], List[Experience]]:
        """
        Split experiences into train/validation/test sets.

        Args:
            experiences: Full list of experiences.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            shuffle: Whether to shuffle before splitting.
            random_seed: Random seed for reproducibility.

        Returns:
            Tuple of (train, validation, test) experience lists.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n = len(experiences)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train = [experiences[i] for i in train_indices]
        val = [experiences[i] for i in val_indices]
        test = [experiences[i] for i in test_indices]

        logger.info(f"Split {n} experiences: train={len(train)}, " f"val={len(val)}, test={len(test)}")

        return train, val, test

    def stratified_split(
        self,
        experiences: List[Experience],
        stratify_by: str = "sector",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[Experience], List[Experience], List[Experience]]:
        """
        Stratified split maintaining distribution of stratify_by field.

        Args:
            experiences: Full list of experiences.
            stratify_by: Field to stratify by ("sector" or "tier").
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            random_seed: Random seed for reproducibility.

        Returns:
            Tuple of (train, validation, test) experience lists.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Group by stratification key
        groups: Dict[str, List[Experience]] = {}
        for exp in experiences:
            if stratify_by == "sector":
                key = exp.context.sector
            elif stratify_by == "tier":
                key = exp.tier_classification
            else:
                key = "default"

            if key not in groups:
                groups[key] = []
            groups[key].append(exp)

        train, val, test = [], [], []

        # Split each group proportionally
        for group_exps in groups.values():
            n = len(group_exps)
            indices = np.random.permutation(n)

            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train.extend([group_exps[i] for i in indices[:train_end]])
            val.extend([group_exps[i] for i in indices[train_end:val_end]])
            test.extend([group_exps[i] for i in indices[val_end:]])

        # Shuffle final sets
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)

        logger.info(f"Stratified split by {stratify_by}: train={len(train)}, " f"val={len(val)}, test={len(test)}")

        return train, val, test

    def sample_balanced(
        self,
        experiences: List[Experience],
        n_samples: int,
        balance_by: str = "sector",
    ) -> List[Experience]:
        """
        Sample experiences with balanced representation.

        Args:
            experiences: Full list to sample from.
            n_samples: Number of samples to return.
            balance_by: Field to balance ("sector" or "tier").

        Returns:
            Balanced sample of experiences.
        """
        # Group by balance key
        groups: Dict[str, List[Experience]] = {}
        for exp in experiences:
            if balance_by == "sector":
                key = exp.context.sector
            elif balance_by == "tier":
                key = exp.tier_classification
            else:
                key = "default"

            if key not in groups:
                groups[key] = []
            groups[key].append(exp)

        # Calculate samples per group
        n_groups = len(groups)
        samples_per_group = max(1, n_samples // n_groups)

        sampled = []
        for group_exps in groups.values():
            n_take = min(samples_per_group, len(group_exps))
            indices = np.random.choice(len(group_exps), n_take, replace=False)
            sampled.extend([group_exps[i] for i in indices])

        # If we need more samples, take randomly from all
        if len(sampled) < n_samples:
            remaining = [e for e in experiences if e not in sampled]
            n_more = min(n_samples - len(sampled), len(remaining))
            if n_more > 0:
                indices = np.random.choice(len(remaining), n_more, replace=False)
                sampled.extend([remaining[i] for i in indices])

        np.random.shuffle(sampled)
        return sampled[:n_samples]

    def get_statistics(
        self,
        experiences: List[Experience],
    ) -> Dict[str, Any]:
        """
        Get statistics about the experience set.

        Returns:
            Dict with statistics about sectors, tiers, rewards, etc.
        """
        if not experiences:
            return {"count": 0}

        rewards = [e.reward.primary_reward for e in experiences if e.reward.primary_reward is not None]

        sectors = [e.context.sector for e in experiences]
        tiers = [e.tier_classification for e in experiences]

        sector_counts = {}
        for s in sectors:
            sector_counts[s] = sector_counts.get(s, 0) + 1

        tier_counts = {}
        for t in tiers:
            tier_counts[t] = tier_counts.get(t, 0) + 1

        return {
            "count": len(experiences),
            "reward_mean": float(np.mean(rewards)) if rewards else None,
            "reward_std": float(np.std(rewards)) if rewards else None,
            "reward_min": float(np.min(rewards)) if rewards else None,
            "reward_max": float(np.max(rewards)) if rewards else None,
            "sector_counts": sector_counts,
            "tier_counts": tier_counts,
            "unique_sectors": len(sector_counts),
            "unique_tiers": len(tier_counts),
            "date_range": {
                "earliest": min(e.analysis_date for e in experiences).isoformat(),
                "latest": max(e.analysis_date for e in experiences).isoformat(),
            },
        }


# Factory function
def get_experience_collector() -> ExperienceCollector:
    """Get ExperienceCollector instance."""
    return ExperienceCollector()
