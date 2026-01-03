"""
RL Monitoring Package

Provides monitoring and evaluation infrastructure:
- RLMetrics: Performance metrics tracking and reporting
- ABTestingFramework: A/B testing RL policy vs baseline

Usage:
    from investigator.domain.services.rl.monitoring import (
        RLMetrics,
        ABTestingFramework,
    )

    # Track metrics
    metrics = RLMetrics(db_connection)
    sector_performance = metrics.get_accuracy_by_sector()
    baseline_comparison = metrics.compare_to_baseline()

    # Run A/B test
    ab_test = ABTestingFramework(
        rl_policy=hybrid_policy,
        baseline_service=dynamic_weighting_service,
        rl_traffic_pct=0.20,
    )
    results = ab_test.get_test_results()
"""

from investigator.domain.services.rl.monitoring.metrics import RLMetrics
from investigator.domain.services.rl.monitoring.ab_testing import ABTestingFramework

__all__ = [
    "RLMetrics",
    "ABTestingFramework",
]
