#!/usr/bin/env python3
"""
RL Outcome Updater - Cron Job Script

Updates valuation predictions with actual price outcomes and calculates rewards.

Run daily via cron:
    0 6 * * * cd /path/to/InvestiGator && PYTHONPATH="$PWD/src:$PYTHONPATH" python3 scripts/rl_outcome_updater.py

This script:
1. Finds predictions that are 30+ days old and need price updates
2. Fetches actual prices from the stock database
3. Calculates rewards based on prediction accuracy
4. Updates the valuation_outcomes table

Author: InvestiGator Team
Date: 2025-12-28
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

# Add src to path
sys.path.insert(0, "src")

from investigator.infrastructure.database.db import get_db_manager, safe_json_dumps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class OutcomeUpdater:
    """Updates valuation predictions with actual price outcomes."""

    def __init__(self):
        self.db = get_db_manager()  # SEC database

        # Create separate connection for stock database
        self.stock_engine = create_engine("postgresql://stockuser:${STOCK_DB_PASSWORD}@${STOCK_DB_HOST}:5432/stock")
        self.StockSession = sessionmaker(bind=self.stock_engine)

    def get_pending_30d_updates(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get predictions needing 30-day outcome update."""
        with self.db.get_session() as session:
            results = session.execute(
                text(
                    """
                    SELECT id, symbol, analysis_date, blended_fair_value, current_price
                    FROM valuation_outcomes
                    WHERE actual_price_30d IS NULL
                      AND analysis_date <= CURRENT_DATE - INTERVAL '30 days'
                    ORDER BY analysis_date ASC
                    LIMIT :limit
                """
                ),
                {"limit": limit},
            ).fetchall()

            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "analysis_date": r[2],
                    "blended_fair_value": float(r[3]) if r[3] else 0,
                    "current_price": float(r[4]) if r[4] else 0,
                }
                for r in results
            ]

    def get_pending_90d_updates(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get predictions needing 90-day outcome update (and reward calculation)."""
        with self.db.get_session() as session:
            results = session.execute(
                text(
                    """
                    SELECT id, symbol, analysis_date, blended_fair_value, current_price,
                           dcf_fair_value, pe_fair_value, ps_fair_value,
                           evebitda_fair_value, pb_fair_value, ggm_fair_value
                    FROM valuation_outcomes
                    WHERE actual_price_90d IS NULL
                      AND analysis_date <= CURRENT_DATE - INTERVAL '90 days'
                    ORDER BY analysis_date ASC
                    LIMIT :limit
                """
                ),
                {"limit": limit},
            ).fetchall()

            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "analysis_date": r[2],
                    "blended_fair_value": float(r[3]) if r[3] else 0,
                    "current_price": float(r[4]) if r[4] else 0,
                    "model_fair_values": {
                        "dcf": float(r[5]) if r[5] else None,
                        "pe": float(r[6]) if r[6] else None,
                        "ps": float(r[7]) if r[7] else None,
                        "ev_ebitda": float(r[8]) if r[8] else None,
                        "pb": float(r[9]) if r[9] else None,
                        "ggm": float(r[10]) if r[10] else None,
                    },
                }
                for r in results
            ]

    def get_historical_price(self, symbol: str, target_date: date) -> Optional[float]:
        """Get stock price on or near target_date from stock database."""
        session = self.StockSession()
        try:
            query = """
                SELECT close
                FROM tickerdata
                WHERE ticker = :symbol
                  AND date <= :target_date
                ORDER BY date DESC
                LIMIT 1
            """
            result = session.execute(text(query), {"symbol": symbol, "target_date": target_date}).fetchone()

            if result:
                return float(result[0])
            return None
        finally:
            session.close()

    def calculate_reward(
        self,
        blended_fv: float,
        current_price: float,
        actual_price: float,
    ) -> float:
        """
        Calculate reward signal.

        Reward = 0.6 * accuracy + 0.4 * direction
        """
        if actual_price <= 0 or blended_fv <= 0:
            return 0

        # Accuracy component (1 - error, min 0)
        error = abs(blended_fv - actual_price) / actual_price
        accuracy = max(0, 1 - error)

        # Direction component
        predicted_direction = 1 if blended_fv > current_price else -1
        actual_direction = 1 if actual_price > current_price else -1
        direction_correct = predicted_direction == actual_direction
        direction_reward = 1.0 if direction_correct else -0.5

        # Combined reward
        reward = 0.6 * accuracy + 0.4 * direction_reward
        return max(-1.0, min(1.0, reward))

    def calculate_per_model_rewards(
        self,
        model_fair_values: Dict[str, Optional[float]],
        current_price: float,
        actual_price: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate rewards for each individual model."""
        per_model = {}

        actual_direction = 1 if actual_price > current_price else -1

        for model, fv in model_fair_values.items():
            if fv is not None and fv > 0 and actual_price > 0:
                error = abs(fv - actual_price) / actual_price
                accuracy = max(0, 1 - error)

                model_direction = 1 if fv > current_price else -1
                direction_correct = model_direction == actual_direction
                direction_reward = 1.0 if direction_correct else -0.5

                reward = 0.6 * accuracy + 0.4 * direction_reward

                per_model[model] = {
                    "fair_value": round(fv, 2),
                    "error_pct": round(error * 100, 2),
                    "accuracy": round(accuracy, 3),
                    "direction_correct": direction_correct,
                    "reward_90d": round(max(-1.0, min(1.0, reward)), 3),
                }

        return per_model

    def update_30d_outcomes(self) -> Tuple[int, int]:
        """Update 30-day outcomes for eligible predictions."""
        pending = self.get_pending_30d_updates()
        logger.info(f"Found {len(pending)} predictions pending 30-day updates")

        updated = 0
        errors = 0

        for record in pending:
            try:
                target_date = record["analysis_date"] + timedelta(days=30)
                price = self.get_historical_price(record["symbol"], target_date)

                if price:
                    # Update price
                    with self.db.get_session() as session:
                        session.execute(
                            text(
                                """
                                UPDATE valuation_outcomes
                                SET actual_price_30d = :price,
                                    outcome_updated_at = CURRENT_TIMESTAMP
                                WHERE id = :id
                            """
                            ),
                            {"id": record["id"], "price": price},
                        )
                        session.commit()

                    # Calculate and update 30d reward
                    reward_30d = self.calculate_reward(record["blended_fair_value"], record["current_price"], price)
                    with self.db.get_session() as session:
                        session.execute(
                            text(
                                """
                                UPDATE valuation_outcomes
                                SET reward_30d = :reward
                                WHERE id = :id
                            """
                            ),
                            {"id": record["id"], "reward": reward_30d},
                        )
                        session.commit()

                    updated += 1
                    logger.info(
                        f"{record['symbol']} - 30d update: "
                        f"FV=${record['blended_fair_value']:.2f}, "
                        f"Actual=${price:.2f}, Reward={reward_30d:.3f}"
                    )
                else:
                    logger.warning(f"{record['symbol']} - No price data for {target_date}")
                    errors += 1

            except Exception as e:
                logger.error(f"Error updating 30d for {record['symbol']}: {e}")
                errors += 1

        return updated, errors

    def update_90d_outcomes(self) -> Tuple[int, int]:
        """Update 90-day outcomes and calculate rewards."""
        pending = self.get_pending_90d_updates()
        logger.info(f"Found {len(pending)} predictions pending 90-day updates")

        updated = 0
        errors = 0

        for record in pending:
            try:
                target_date = record["analysis_date"] + timedelta(days=90)
                price = self.get_historical_price(record["symbol"], target_date)

                if price:
                    # Calculate rewards
                    reward_90d = self.calculate_reward(record["blended_fair_value"], record["current_price"], price)

                    per_model_rewards = self.calculate_per_model_rewards(
                        record["model_fair_values"], record["current_price"], price
                    )

                    # Update database
                    with self.db.get_session() as session:
                        session.execute(
                            text(
                                """
                                UPDATE valuation_outcomes
                                SET actual_price_90d = :price,
                                    reward_90d = :reward_90d,
                                    per_model_rewards = :per_model_rewards,
                                    outcome_updated_at = CURRENT_TIMESTAMP
                                WHERE id = :id
                            """
                            ),
                            {
                                "id": record["id"],
                                "price": price,
                                "reward_90d": reward_90d,
                                "per_model_rewards": safe_json_dumps(per_model_rewards),
                            },
                        )
                        session.commit()

                    updated += 1
                    logger.info(
                        f"{record['symbol']} - 90d update: "
                        f"FV=${record['blended_fair_value']:.2f}, "
                        f"Actual=${price:.2f}, Reward={reward_90d:.3f}"
                    )
                else:
                    logger.warning(f"{record['symbol']} - No price data for {target_date}")
                    errors += 1

            except Exception as e:
                logger.error(f"Error updating 90d for {record['symbol']}: {e}")
                errors += 1

        return updated, errors

    def run_full_update(self) -> Dict[str, Any]:
        """Run full outcome update cycle."""
        logger.info("Starting outcome update cycle...")

        # Update 30-day outcomes
        updated_30d, errors_30d = self.update_30d_outcomes()

        # Update 90-day outcomes (with rewards)
        updated_90d, errors_90d = self.update_90d_outcomes()

        results = {
            "30d_updated": updated_30d,
            "30d_errors": errors_30d,
            "90d_updated": updated_90d,
            "90d_errors": errors_90d,
            "total_updated": updated_30d + updated_90d,
            "total_errors": errors_30d + errors_90d,
        }

        logger.info(
            f"Outcome update complete: " f"{results['total_updated']} updated, {results['total_errors']} errors"
        )

        return results


def main():
    parser = argparse.ArgumentParser(description="RL Outcome Updater")
    parser.add_argument("--30d-only", action="store_true", help="Only update 30-day outcomes")
    parser.add_argument("--90d-only", action="store_true", help="Only update 90-day outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Show pending updates without applying")

    args = parser.parse_args()

    updater = OutcomeUpdater()

    if args.dry_run:
        pending_30d = updater.get_pending_30d_updates()
        pending_90d = updater.get_pending_90d_updates()
        print(f"\n=== DRY RUN ===")
        print(f"Pending 30-day updates: {len(pending_30d)}")
        for p in pending_30d[:10]:
            print(f"  - {p['symbol']} ({p['analysis_date']})")
        if len(pending_30d) > 10:
            print(f"  ... and {len(pending_30d) - 10} more")

        print(f"\nPending 90-day updates: {len(pending_90d)}")
        for p in pending_90d[:10]:
            print(f"  - {p['symbol']} ({p['analysis_date']})")
        if len(pending_90d) > 10:
            print(f"  ... and {len(pending_90d) - 10} more")
        return

    if args._30d_only:
        updated, errors = updater.update_30d_outcomes()
        print(f"\n30-day updates: {updated} updated, {errors} errors")
    elif args._90d_only:
        updated, errors = updater.update_90d_outcomes()
        print(f"\n90-day updates: {updated} updated, {errors} errors")
    else:
        results = updater.run_full_update()
        print("\n" + "=" * 50)
        print("OUTCOME UPDATE SUMMARY")
        print("=" * 50)
        print(f"30-day updates: {results['30d_updated']} updated, {results['30d_errors']} errors")
        print(f"90-day updates: {results['90d_updated']} updated, {results['90d_errors']} errors")
        print(f"Total: {results['total_updated']} updated, {results['total_errors']} errors")
        print("=" * 50)


if __name__ == "__main__":
    main()
