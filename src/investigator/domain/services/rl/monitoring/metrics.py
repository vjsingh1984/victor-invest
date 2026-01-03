"""
RL Metrics Tracking

Tracks and reports RL system performance metrics.
Provides dashboards for monitoring accuracy, model contribution,
and comparison to baseline.

Usage:
    from investigator.domain.services.rl.monitoring import RLMetrics

    metrics = RLMetrics()

    # Get accuracy by sector
    sector_accuracy = metrics.get_accuracy_by_sector()

    # Get model contribution
    contribution = metrics.get_model_contribution()

    # Compare to baseline
    comparison = metrics.compare_to_baseline()
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text

from investigator.infrastructure.database.db import get_db_manager

logger = logging.getLogger(__name__)


class RLMetrics:
    """
    Tracks and reports RL policy performance metrics.

    Provides:
    - Accuracy breakdown by sector, industry, tier
    - Model-specific contribution analysis
    - Baseline comparison
    - Trend analysis over time
    """

    def __init__(self):
        """Initialize RLMetrics."""
        self.db = get_db_manager()

    def get_accuracy_by_sector(
        self,
        min_samples: int = 10,
        days: int = 365,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy breakdown by GICS sector.

        Args:
            min_samples: Minimum samples required per sector.
            days: Look back period in days.

        Returns:
            Dict mapping sector to accuracy metrics.
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT
                            context_features->>'sector' as sector,
                            COUNT(*) as num_predictions,
                            AVG(reward_90d) as avg_reward,
                            STDDEV(reward_90d) as std_reward,
                            AVG(ABS(blended_fair_value - actual_price_90d) /
                                NULLIF(actual_price_90d, 0) * 100) as avg_error_pct,
                            SUM(CASE WHEN
                                (blended_fair_value > current_price AND actual_price_90d > current_price) OR
                                (blended_fair_value <= current_price AND actual_price_90d <= current_price)
                                THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as direction_accuracy
                        FROM valuation_outcomes
                        WHERE reward_90d IS NOT NULL
                          AND analysis_date >= CURRENT_DATE - :days
                        GROUP BY context_features->>'sector'
                        HAVING COUNT(*) >= :min_samples
                        ORDER BY avg_reward DESC
                    """
                    ),
                    {"days": days, "min_samples": min_samples},
                ).fetchall()

                return {
                    row[0]
                    or "Unknown": {
                        "num_predictions": int(row[1]),
                        "avg_reward": float(row[2]) if row[2] else 0,
                        "std_reward": float(row[3]) if row[3] else 0,
                        "avg_error_pct": float(row[4]) if row[4] else 0,
                        "direction_accuracy": float(row[5]) if row[5] else 0,
                    }
                    for row in result
                }

        except Exception as e:
            logger.error(f"Failed to get accuracy by sector: {e}")
            return {}

    def get_accuracy_by_tier(
        self,
        min_samples: int = 10,
        days: int = 365,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy breakdown by tier classification.

        Args:
            min_samples: Minimum samples required per tier.
            days: Look back period in days.

        Returns:
            Dict mapping tier to accuracy metrics.
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT
                            tier_classification,
                            COUNT(*) as num_predictions,
                            AVG(reward_90d) as avg_reward,
                            STDDEV(reward_90d) as std_reward,
                            AVG(ABS(blended_fair_value - actual_price_90d) /
                                NULLIF(actual_price_90d, 0) * 100) as avg_error_pct
                        FROM valuation_outcomes
                        WHERE reward_90d IS NOT NULL
                          AND analysis_date >= CURRENT_DATE - :days
                        GROUP BY tier_classification
                        HAVING COUNT(*) >= :min_samples
                        ORDER BY avg_reward DESC
                    """
                    ),
                    {"days": days, "min_samples": min_samples},
                ).fetchall()

                return {
                    row[0]
                    or "Unknown": {
                        "num_predictions": int(row[1]),
                        "avg_reward": float(row[2]) if row[2] else 0,
                        "std_reward": float(row[3]) if row[3] else 0,
                        "avg_error_pct": float(row[4]) if row[4] else 0,
                    }
                    for row in result
                }

        except Exception as e:
            logger.error(f"Failed to get accuracy by tier: {e}")
            return {}

    def get_model_contribution(
        self,
        days: int = 365,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which models contribute most to accurate predictions.

        Returns:
            Dict mapping model name to contribution metrics.
        """
        try:
            # This would analyze per_model_rewards JSONB column
            # For now, return placeholder structure
            models = ["dcf", "pe", "ps", "ev_ebitda", "pb", "ggm"]

            with self.db.get_session() as session:
                # Get average weights used
                result = session.execute(
                    text(
                        """
                        SELECT
                            AVG((model_weights->>'dcf')::float) as dcf_avg,
                            AVG((model_weights->>'pe')::float) as pe_avg,
                            AVG((model_weights->>'ps')::float) as ps_avg,
                            AVG((model_weights->>'ev_ebitda')::float) as ev_ebitda_avg,
                            AVG((model_weights->>'pb')::float) as pb_avg,
                            AVG((model_weights->>'ggm')::float) as ggm_avg
                        FROM valuation_outcomes
                        WHERE reward_90d IS NOT NULL
                          AND analysis_date >= CURRENT_DATE - :days
                    """
                    ),
                    {"days": days},
                ).fetchone()

                if result:
                    return {
                        "dcf": {"avg_weight": float(result[0] or 0)},
                        "pe": {"avg_weight": float(result[1] or 0)},
                        "ps": {"avg_weight": float(result[2] or 0)},
                        "ev_ebitda": {"avg_weight": float(result[3] or 0)},
                        "pb": {"avg_weight": float(result[4] or 0)},
                        "ggm": {"avg_weight": float(result[5] or 0)},
                    }

                return {}

        except Exception as e:
            logger.error(f"Failed to get model contribution: {e}")
            return {}

    def compare_to_baseline(
        self,
        days: int = 90,
    ) -> Dict[str, float]:
        """
        Compare RL policy performance to baseline (rule-based).

        Args:
            days: Look back period.

        Returns:
            Dict with comparison metrics.
        """
        try:
            with self.db.get_session() as session:
                # Compare RL vs baseline A/B test results
                result = session.execute(
                    text(
                        """
                        SELECT
                            ab_test_group,
                            COUNT(*) as num_predictions,
                            AVG(reward_90d) as avg_reward,
                            AVG(ABS(blended_fair_value - actual_price_90d) /
                                NULLIF(actual_price_90d, 0) * 100) as avg_error_pct,
                            SUM(CASE WHEN
                                (blended_fair_value > current_price AND actual_price_90d > current_price) OR
                                (blended_fair_value <= current_price AND actual_price_90d <= current_price)
                                THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as direction_accuracy
                        FROM valuation_outcomes
                        WHERE reward_90d IS NOT NULL
                          AND ab_test_group IS NOT NULL
                          AND analysis_date >= CURRENT_DATE - :days
                        GROUP BY ab_test_group
                    """
                    ),
                    {"days": days},
                ).fetchall()

                comparison = {}
                for row in result:
                    group = row[0]
                    comparison[f"{group}_count"] = int(row[1])
                    comparison[f"{group}_avg_reward"] = float(row[2]) if row[2] else 0
                    comparison[f"{group}_avg_error"] = float(row[3]) if row[3] else 0
                    comparison[f"{group}_direction_accuracy"] = float(row[4]) if row[4] else 0

                # Calculate improvement
                if "rl_avg_reward" in comparison and "baseline_avg_reward" in comparison:
                    baseline = comparison["baseline_avg_reward"]
                    rl = comparison["rl_avg_reward"]
                    if baseline != 0:
                        comparison["reward_improvement_pct"] = (rl - baseline) / abs(baseline) * 100

                return comparison

        except Exception as e:
            logger.error(f"Failed to compare to baseline: {e}")
            return {}

    def get_trend(
        self,
        days: int = 90,
        bucket: str = "week",
    ) -> List[Dict[str, Any]]:
        """
        Get performance trend over time.

        Args:
            days: Look back period.
            bucket: Aggregation bucket ("day", "week", "month").

        Returns:
            List of time buckets with metrics.
        """
        try:
            bucket_sql = {
                "day": "date_trunc('day', analysis_date)",
                "week": "date_trunc('week', analysis_date)",
                "month": "date_trunc('month', analysis_date)",
            }.get(bucket, "date_trunc('week', analysis_date)")

            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        f"""
                        SELECT
                            {bucket_sql} as period,
                            COUNT(*) as num_predictions,
                            AVG(reward_90d) as avg_reward,
                            AVG(ABS(blended_fair_value - actual_price_90d) /
                                NULLIF(actual_price_90d, 0) * 100) as avg_error_pct
                        FROM valuation_outcomes
                        WHERE reward_90d IS NOT NULL
                          AND analysis_date >= CURRENT_DATE - :days
                        GROUP BY {bucket_sql}
                        ORDER BY period ASC
                    """
                    ),
                    {"days": days},
                ).fetchall()

                return [
                    {
                        "period": row[0].isoformat() if row[0] else None,
                        "num_predictions": int(row[1]),
                        "avg_reward": float(row[2]) if row[2] else 0,
                        "avg_error_pct": float(row[3]) if row[3] else 0,
                    }
                    for row in result
                ]

        except Exception as e:
            logger.error(f"Failed to get trend: {e}")
            return []

    def get_summary(
        self,
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        Get summary of RL system performance.

        Returns:
            Dict with summary metrics.
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT
                            COUNT(*) as total_predictions,
                            COUNT(CASE WHEN reward_90d IS NOT NULL THEN 1 END) as with_outcomes,
                            AVG(reward_90d) as avg_reward,
                            STDDEV(reward_90d) as std_reward,
                            MIN(reward_90d) as min_reward,
                            MAX(reward_90d) as max_reward,
                            AVG(ABS(blended_fair_value - actual_price_90d) /
                                NULLIF(actual_price_90d, 0) * 100) as avg_mape,
                            SUM(CASE WHEN
                                (blended_fair_value > current_price AND actual_price_90d > current_price) OR
                                (blended_fair_value <= current_price AND actual_price_90d <= current_price)
                                THEN 1 ELSE 0 END)::FLOAT /
                                NULLIF(COUNT(CASE WHEN reward_90d IS NOT NULL THEN 1 END), 0)
                                as direction_accuracy
                        FROM valuation_outcomes
                        WHERE analysis_date >= CURRENT_DATE - :days
                    """
                    ),
                    {"days": days},
                ).fetchone()

                if result:
                    return {
                        "period_days": days,
                        "total_predictions": int(result[0]) if result[0] else 0,
                        "predictions_with_outcomes": int(result[1]) if result[1] else 0,
                        "avg_reward": float(result[2]) if result[2] else 0,
                        "std_reward": float(result[3]) if result[3] else 0,
                        "min_reward": float(result[4]) if result[4] else 0,
                        "max_reward": float(result[5]) if result[5] else 0,
                        "avg_mape": float(result[6]) if result[6] else 0,
                        "direction_accuracy": float(result[7]) if result[7] else 0,
                    }

                return {}

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {}


# Factory function
def get_rl_metrics() -> RLMetrics:
    """Get RLMetrics instance."""
    return RLMetrics()
