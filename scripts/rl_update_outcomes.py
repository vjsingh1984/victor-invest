#!/usr/bin/env python3
"""
RL Outcome Update Script

Updates valuation outcomes with actual prices after 30/90/365 days.
Run this script daily via cron to keep outcome data current.

Usage:
    python scripts/rl_update_outcomes.py              # Update all pending
    python scripts/rl_update_outcomes.py --batch 500  # Limit batch size
    python scripts/rl_update_outcomes.py --dry-run    # Preview without updating

Cron Example (daily at 6 AM):
    0 6 * * * cd /path/to/victor-invest && PYTHONPATH=./src:. python scripts/rl_update_outcomes.py >> logs/rl_outcomes.log 2>&1

Environment:
    PYTHONPATH=./src:. python scripts/rl_update_outcomes.py
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text
from investigator.infrastructure.database.db import get_db_manager
from investigator.domain.services.rl.price_history import PriceHistoryService
from investigator.domain.services.rl.reward_calculator import calculate_reward

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_pending_updates(db, days: int, batch_size: int) -> list:
    """Get predictions needing outcome updates."""
    with db.get_session() as session:
        if days == 30:
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
                {"limit": batch_size},
            ).fetchall()
        elif days == 90:
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
                {"limit": batch_size},
            ).fetchall()
        else:  # 365 days
            results = session.execute(
                text(
                    """
                    SELECT id, symbol, analysis_date, blended_fair_value, current_price
                    FROM valuation_outcomes
                    WHERE actual_price_365d IS NULL
                      AND analysis_date <= CURRENT_DATE - INTERVAL '365 days'
                    ORDER BY analysis_date ASC
                    LIMIT :limit
                """
                ),
                {"limit": batch_size},
            ).fetchall()

        return results


def get_price_on_date(price_service: PriceHistoryService, symbol: str, target_date: date) -> float | None:
    """Get stock price on a specific date from tickerdata table."""
    return price_service._get_price_sync(symbol, target_date, use_adj_close=True, search_days=5)


# Note: calculate_reward is now imported from investigator.domain.services.rl.reward_calculator
# This ensures consistency with rl_backtest.py and outcome_tracker.py


def update_30d_outcomes(db, price_service: PriceHistoryService, batch_size: int, dry_run: bool) -> tuple:
    """Update 30-day outcomes."""
    pending = get_pending_updates(db, 30, batch_size)
    updated = 0
    errors = 0

    for record in pending:
        record_id, symbol, analysis_date, fv, current_price = record
        target_date = analysis_date + timedelta(days=30)

        price = get_price_on_date(price_service, symbol, target_date)
        if price is None:
            errors += 1
            continue

        if not dry_run:
            with db.get_session() as session:
                session.execute(
                    text(
                        """
                        UPDATE valuation_outcomes
                        SET actual_price_30d = :price,
                            outcome_updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                    """
                    ),
                    {"id": record_id, "price": price},
                )
                session.commit()

        updated += 1

    return updated, errors, len(pending)


def update_90d_outcomes(db, price_service: PriceHistoryService, batch_size: int, dry_run: bool) -> tuple:
    """Update 90-day outcomes and calculate rewards."""
    pending = get_pending_updates(db, 90, batch_size)
    updated = 0
    errors = 0

    for record in pending:
        record_id = record[0]
        symbol = record[1]
        analysis_date = record[2]
        fv = float(record[3]) if record[3] else 0
        current_price = float(record[4]) if record[4] else 0
        target_date = analysis_date + timedelta(days=90)

        price = get_price_on_date(price_service, symbol, target_date)
        if price is None or fv == 0 or current_price == 0:
            errors += 1
            continue

        # Use shared reward calculator (annualized, risk-adjusted, asymmetric)
        reward = calculate_reward(
            predicted_fv=fv,
            price_at_prediction=current_price,
            actual_price=price,
            days=90,
            beta=1.0,  # TODO: Could fetch actual beta from database
        )

        if not dry_run:
            with db.get_session() as session:
                session.execute(
                    text(
                        """
                        UPDATE valuation_outcomes
                        SET actual_price_90d = :price,
                            reward_90d = :reward,
                            outcome_updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                    """
                    ),
                    {"id": record_id, "price": price, "reward": reward},
                )
                session.commit()

        updated += 1

    return updated, errors, len(pending)


def update_365d_outcomes(db, price_service: PriceHistoryService, batch_size: int, dry_run: bool) -> tuple:
    """Update 365-day outcomes."""
    pending = get_pending_updates(db, 365, batch_size)
    updated = 0
    errors = 0

    for record in pending:
        record_id, symbol, analysis_date, fv, current_price = record
        target_date = analysis_date + timedelta(days=365)

        price = get_price_on_date(price_service, symbol, target_date)
        if price is None:
            errors += 1
            continue

        # Use shared reward calculator (annualized, risk-adjusted, asymmetric)
        reward = None
        if fv and current_price:
            reward = calculate_reward(
                predicted_fv=float(fv),
                price_at_prediction=float(current_price),
                actual_price=price,
                days=365,
                beta=1.0,
            )

        if not dry_run:
            with db.get_session() as session:
                session.execute(
                    text(
                        """
                        UPDATE valuation_outcomes
                        SET actual_price_365d = :price,
                            reward_365d = :reward,
                            outcome_updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                    """
                    ),
                    {"id": record_id, "price": price, "reward": reward},
                )
                session.commit()

        updated += 1

    return updated, errors, len(pending)


def get_stats(db) -> dict:
    """Get outcome statistics."""
    with db.get_session() as session:
        total = session.execute(text("SELECT COUNT(*) FROM valuation_outcomes")).scalar()
        with_30d = session.execute(
            text("SELECT COUNT(*) FROM valuation_outcomes WHERE actual_price_30d IS NOT NULL")
        ).scalar()
        with_90d = session.execute(
            text("SELECT COUNT(*) FROM valuation_outcomes WHERE actual_price_90d IS NOT NULL")
        ).scalar()
        with_365d = session.execute(
            text("SELECT COUNT(*) FROM valuation_outcomes WHERE actual_price_365d IS NOT NULL")
        ).scalar()

        pending_30d = session.execute(
            text(
                """
                SELECT COUNT(*) FROM valuation_outcomes
                WHERE actual_price_30d IS NULL
                  AND analysis_date <= CURRENT_DATE - INTERVAL '30 days'
            """
            )
        ).scalar()
        pending_90d = session.execute(
            text(
                """
                SELECT COUNT(*) FROM valuation_outcomes
                WHERE actual_price_90d IS NULL
                  AND analysis_date <= CURRENT_DATE - INTERVAL '90 days'
            """
            )
        ).scalar()
        pending_365d = session.execute(
            text(
                """
                SELECT COUNT(*) FROM valuation_outcomes
                WHERE actual_price_365d IS NULL
                  AND analysis_date <= CURRENT_DATE - INTERVAL '365 days'
            """
            )
        ).scalar()

        return {
            "total": total,
            "with_30d": with_30d,
            "with_90d": with_90d,
            "with_365d": with_365d,
            "pending_30d": pending_30d,
            "pending_90d": pending_90d,
            "pending_365d": pending_365d,
        }


def main():
    parser = argparse.ArgumentParser(description="Update valuation outcomes with actual prices")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size per update type")
    parser.add_argument("--dry-run", action="store_true", help="Preview without updating")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics")
    args = parser.parse_args()

    print("=" * 70)
    print("RL OUTCOME UPDATE")
    print(f"Started: {datetime.now().isoformat()}")
    if args.dry_run:
        print("MODE: DRY RUN (no updates)")
    print("=" * 70)

    db = get_db_manager()
    price_service = PriceHistoryService()

    # Show current stats
    stats = get_stats(db)
    print(f"\nCurrent Statistics:")
    print(f"  Total predictions: {stats['total']}")
    print(f"  With 30d outcomes: {stats['with_30d']} (pending: {stats['pending_30d']})")
    print(f"  With 90d outcomes: {stats['with_90d']} (pending: {stats['pending_90d']})")
    print(f"  With 365d outcomes: {stats['with_365d']} (pending: {stats['pending_365d']})")

    if args.stats_only:
        return

    # Update outcomes
    print(f"\nUpdating outcomes (batch size: {args.batch})...")

    print("\n30-day outcomes:")
    updated, errors, total = update_30d_outcomes(db, price_service, args.batch, args.dry_run)
    print(f"  Pending: {total}, Updated: {updated}, Errors: {errors}")

    print("\n90-day outcomes (with rewards):")
    updated, errors, total = update_90d_outcomes(db, price_service, args.batch, args.dry_run)
    print(f"  Pending: {total}, Updated: {updated}, Errors: {errors}")

    print("\n365-day outcomes:")
    updated, errors, total = update_365d_outcomes(db, price_service, args.batch, args.dry_run)
    print(f"  Pending: {total}, Updated: {updated}, Errors: {errors}")

    # Show updated stats
    if not args.dry_run:
        stats = get_stats(db)
        print(f"\nUpdated Statistics:")
        print(f"  With 90d outcomes (training-ready): {stats['with_90d']}")

    print("\n" + "=" * 70)
    print("UPDATE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
