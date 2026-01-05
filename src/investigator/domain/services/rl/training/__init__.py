"""
RL Training Package

Provides training infrastructure for RL policies:
- ExperienceCollector: Collects training experiences from outcome database
- RLTrainer: Trains policies on collected experiences
- RLTrainingPipeline: End-to-end training pipeline

Usage:
    from investigator.domain.services.rl.training import (
        ExperienceCollector,
        RLTrainer,
        RLTrainingPipeline,
    )

    # Collect experiences
    collector = ExperienceCollector(db_connection)
    experiences = collector.collect_experiences(min_days_ago=90)

    # Train policy
    trainer = RLTrainer(policy)
    metrics = trainer.train_batch(experiences, epochs=10)

    # Or run full pipeline
    pipeline = RLTrainingPipeline(config)
    result = pipeline.run()
"""

from investigator.domain.services.rl.training.experience_collector import ExperienceCollector
from investigator.domain.services.rl.training.pipeline import RLTrainingPipeline
from investigator.domain.services.rl.training.trainer import RLTrainer

__all__ = [
    "ExperienceCollector",
    "RLTrainer",
    "RLTrainingPipeline",
]
