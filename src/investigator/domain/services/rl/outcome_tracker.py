"""
Outcome Tracker Service

Tracks valuation predictions and updates with actual outcomes for RL training.
This is the foundation for the RL system - without outcome data, we cannot
calculate rewards or train policies.

Responsibilities:
- Store predictions immediately after analysis
- Update predictions with actual prices after 30/90/365 days
- Calculate rewards based on prediction accuracy
- Provide training-ready experiences

Usage:
    from investigator.domain.services.rl import OutcomeTracker

    tracker = OutcomeTracker()

    # Record prediction
    record_id = tracker.record_prediction(
        symbol="AAPL",
        analysis_date=date.today(),
        blended_fair_value=175.50,
        current_price=170.00,
        model_fair_values={"dcf": 180.0, "pe": 170.0, "ps": 175.0},
        model_weights={"dcf": 40, "pe": 35, "ps": 25},
        tier_classification="balanced_high_quality",
        context_features=context,
    )

    # Update outcomes (run via cron job)
    updated_count = await tracker.update_outcomes(lookback_days=90)
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from investigator.domain.services.rl.models import (
    ABTestGroup,
    Experience,
    PerModelReward,
    RewardSignal,
    ValuationContext,
)
from investigator.domain.services.rl.reward_calculator import (
    RewardCalculator,
    get_reward_calculator,
)
from investigator.infrastructure.database.db import (
    DatabaseManager,
    get_db_manager,
    safe_json_dumps,
    safe_json_loads,
)

logger = logging.getLogger(__name__)


class ValuationOutcomesDAO:
    """
    Data Access Object for valuation_outcomes table.

    Handles all database operations for prediction/outcome storage.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def insert_prediction(
        self,
        symbol: str,
        analysis_date: date,
        fiscal_period: Optional[str],
        blended_fair_value: float,
        current_price: float,
        predicted_upside_pct: float,
        model_fair_values: Dict[str, float],
        model_weights: Dict[str, float],
        tier_classification: str,
        context_features: Dict[str, Any],
        ab_test_group: Optional[str] = None,
        policy_version: Optional[str] = None,
        position_type: str = "inferred",
        entry_date: Optional[date] = None,
        exit_date_30d: Optional[date] = None,
        exit_date_90d: Optional[date] = None,
    ) -> Optional[int]:
        """
        Insert new prediction record.

        Args:
            entry_date: Date when position was entered (defaults to analysis_date)
            exit_date_30d: Date 30 days after entry
            exit_date_90d: Date 90 days after entry

        Returns:
            Record ID if successful, None otherwise.
        """
        # Default entry_date to analysis_date
        effective_entry_date = entry_date or analysis_date
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        INSERT INTO valuation_outcomes (
                            symbol, analysis_date, fiscal_period,
                            blended_fair_value, current_price, predicted_upside_pct,
                            dcf_fair_value, pe_fair_value, ps_fair_value,
                            evebitda_fair_value, pb_fair_value, ggm_fair_value,
                            model_weights, tier_classification, context_features,
                            ab_test_group, policy_version, position_type,
                            entry_date, exit_date_30d, exit_date_90d
                        ) VALUES (
                            :symbol, :analysis_date, :fiscal_period,
                            :blended_fair_value, :current_price, :predicted_upside_pct,
                            :dcf_fair_value, :pe_fair_value, :ps_fair_value,
                            :evebitda_fair_value, :pb_fair_value, :ggm_fair_value,
                            :model_weights, :tier_classification, :context_features,
                            :ab_test_group, :policy_version, :position_type,
                            :entry_date, :exit_date_30d, :exit_date_90d
                        )
                        ON CONFLICT (symbol, analysis_date, position_type) DO UPDATE SET
                            fiscal_period = EXCLUDED.fiscal_period,
                            blended_fair_value = EXCLUDED.blended_fair_value,
                            current_price = EXCLUDED.current_price,
                            predicted_upside_pct = EXCLUDED.predicted_upside_pct,
                            dcf_fair_value = EXCLUDED.dcf_fair_value,
                            pe_fair_value = EXCLUDED.pe_fair_value,
                            ps_fair_value = EXCLUDED.ps_fair_value,
                            evebitda_fair_value = EXCLUDED.evebitda_fair_value,
                            pb_fair_value = EXCLUDED.pb_fair_value,
                            ggm_fair_value = EXCLUDED.ggm_fair_value,
                            model_weights = EXCLUDED.model_weights,
                            tier_classification = EXCLUDED.tier_classification,
                            context_features = EXCLUDED.context_features,
                            ab_test_group = EXCLUDED.ab_test_group,
                            policy_version = EXCLUDED.policy_version,
                            entry_date = EXCLUDED.entry_date,
                            exit_date_30d = EXCLUDED.exit_date_30d,
                            exit_date_90d = EXCLUDED.exit_date_90d,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """
                    ),
                    {
                        "symbol": symbol,
                        "analysis_date": analysis_date,
                        "fiscal_period": fiscal_period,
                        "blended_fair_value": blended_fair_value,
                        "current_price": current_price,
                        "predicted_upside_pct": predicted_upside_pct,
                        "dcf_fair_value": model_fair_values.get("dcf"),
                        "pe_fair_value": model_fair_values.get("pe"),
                        "ps_fair_value": model_fair_values.get("ps"),
                        "evebitda_fair_value": model_fair_values.get("ev_ebitda"),
                        "pb_fair_value": model_fair_values.get("pb"),
                        "ggm_fair_value": model_fair_values.get("ggm"),
                        "model_weights": safe_json_dumps(model_weights),
                        "tier_classification": tier_classification,
                        "context_features": safe_json_dumps(context_features),
                        "ab_test_group": ab_test_group,
                        "policy_version": policy_version,
                        "position_type": position_type,
                        "entry_date": effective_entry_date,
                        "exit_date_30d": exit_date_30d,
                        "exit_date_90d": exit_date_90d,
                    },
                )
                row = result.fetchone()
                session.commit()

                record_id = row[0] if row else None
                logger.info(
                    f"Recorded prediction for {symbol} on {analysis_date}: "
                    f"FV=${blended_fair_value:.2f}, price=${current_price:.2f}, "
                    f"upside={predicted_upside_pct:.1f}%"
                )
                return record_id

        except SQLAlchemyError as e:
            logger.error(f"Failed to insert prediction for {symbol}: {e}")
            return None

    def update_outcome_prices(
        self,
        record_id: int,
        actual_price_30d: Optional[float] = None,
        actual_price_90d: Optional[float] = None,
        actual_price_365d: Optional[float] = None,
    ) -> bool:
        """Update outcome prices for a prediction record."""
        try:
            with self.db.get_session() as session:
                # Build dynamic update
                updates = []
                params = {"id": record_id}

                if actual_price_30d is not None:
                    updates.append("actual_price_30d = :actual_price_30d")
                    params["actual_price_30d"] = actual_price_30d
                if actual_price_90d is not None:
                    updates.append("actual_price_90d = :actual_price_90d")
                    params["actual_price_90d"] = actual_price_90d
                if actual_price_365d is not None:
                    updates.append("actual_price_365d = :actual_price_365d")
                    params["actual_price_365d"] = actual_price_365d

                if not updates:
                    return True  # Nothing to update

                updates.append("outcome_updated_at = CURRENT_TIMESTAMP")

                query = f"""
                    UPDATE valuation_outcomes
                    SET {', '.join(updates)}
                    WHERE id = :id
                """
                session.execute(text(query), params)
                session.commit()
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to update outcome prices for record {record_id}: {e}")
            return False

    def update_rewards(
        self,
        record_id: int,
        reward_30d: Optional[float] = None,
        reward_90d: Optional[float] = None,
        reward_365d: Optional[float] = None,
        per_model_rewards: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update calculated rewards for a prediction record."""
        try:
            with self.db.get_session() as session:
                updates = []
                params = {"id": record_id}

                if reward_30d is not None:
                    updates.append("reward_30d = :reward_30d")
                    params["reward_30d"] = reward_30d
                if reward_90d is not None:
                    updates.append("reward_90d = :reward_90d")
                    params["reward_90d"] = reward_90d
                if reward_365d is not None:
                    updates.append("reward_365d = :reward_365d")
                    params["reward_365d"] = reward_365d
                if per_model_rewards is not None:
                    updates.append("per_model_rewards = :per_model_rewards")
                    params["per_model_rewards"] = safe_json_dumps(per_model_rewards)

                if not updates:
                    return True

                query = f"""
                    UPDATE valuation_outcomes
                    SET {', '.join(updates)}
                    WHERE id = :id
                """
                session.execute(text(query), params)
                session.commit()
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to update rewards for record {record_id}: {e}")
            return False

    def mark_used_for_training(
        self,
        record_ids: List[int],
        training_batch_id: int,
    ) -> int:
        """Mark records as used for training."""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        UPDATE valuation_outcomes
                        SET used_for_training = TRUE,
                            training_batch_id = :batch_id
                        WHERE id = ANY(:ids)
                    """
                    ),
                    {"ids": record_ids, "batch_id": training_batch_id},
                )
                session.commit()
                return result.rowcount

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark records as used for training: {e}")
            return 0

    def get_pending_30d_updates(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get predictions needing 30-day outcome update."""
        try:
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
                        "blended_fair_value": r[3],
                        "current_price": r[4],
                    }
                    for r in results
                ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get pending 30d updates: {e}")
            return []

    def get_pending_90d_updates(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get predictions needing 90-day outcome update."""
        try:
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
                        "blended_fair_value": r[3],
                        "current_price": r[4],
                        "model_fair_values": {
                            "dcf": r[5],
                            "pe": r[6],
                            "ps": r[7],
                            "ev_ebitda": r[8],
                            "pb": r[9],
                            "ggm": r[10],
                        },
                    }
                    for r in results
                ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get pending 90d updates: {e}")
            return []

    def get_training_ready_experiences(
        self,
        limit: int = 10000,
        exclude_used: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get experiences ready for training (have 90d outcomes)."""
        try:
            with self.db.get_session() as session:
                used_clause = "AND used_for_training = FALSE" if exclude_used else ""
                results = session.execute(
                    text(
                        f"""
                        SELECT id, symbol, analysis_date, fiscal_period,
                               blended_fair_value, current_price, predicted_upside_pct,
                               dcf_fair_value, pe_fair_value, ps_fair_value,
                               evebitda_fair_value, pb_fair_value, ggm_fair_value,
                               model_weights, tier_classification, context_features,
                               actual_price_30d, actual_price_90d, actual_price_365d,
                               reward_30d, reward_90d, reward_365d,
                               per_model_rewards, ab_test_group, policy_version
                        FROM valuation_outcomes
                        WHERE reward_90d IS NOT NULL
                              {used_clause}
                        ORDER BY analysis_date DESC
                        LIMIT :limit
                    """
                    ),
                    {"limit": limit},
                ).fetchall()

                return [self._row_to_dict(r) for r in results]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get training-ready experiences: {e}")
            return []

    def get_by_symbol(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all predictions for a symbol."""
        try:
            with self.db.get_session() as session:
                results = session.execute(
                    text(
                        """
                        SELECT id, symbol, analysis_date, fiscal_period,
                               blended_fair_value, current_price, predicted_upside_pct,
                               dcf_fair_value, pe_fair_value, ps_fair_value,
                               evebitda_fair_value, pb_fair_value, ggm_fair_value,
                               model_weights, tier_classification, context_features,
                               actual_price_30d, actual_price_90d, actual_price_365d,
                               reward_30d, reward_90d, reward_365d,
                               per_model_rewards, ab_test_group, policy_version
                        FROM valuation_outcomes
                        WHERE symbol = :symbol
                        ORDER BY analysis_date DESC
                        LIMIT :limit
                    """
                    ),
                    {"symbol": symbol, "limit": limit},
                ).fetchall()

                return [self._row_to_dict(r) for r in results]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get predictions for {symbol}: {e}")
            return []

    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        return {
            "id": row[0],
            "symbol": row[1],
            "analysis_date": row[2],
            "fiscal_period": row[3],
            "blended_fair_value": float(row[4]) if row[4] else None,
            "current_price": float(row[5]) if row[5] else None,
            "predicted_upside_pct": float(row[6]) if row[6] else None,
            "model_fair_values": {
                "dcf": float(row[7]) if row[7] else None,
                "pe": float(row[8]) if row[8] else None,
                "ps": float(row[9]) if row[9] else None,
                "ev_ebitda": float(row[10]) if row[10] else None,
                "pb": float(row[11]) if row[11] else None,
                "ggm": float(row[12]) if row[12] else None,
            },
            "model_weights": safe_json_loads(row[13]) if row[13] else {},
            "tier_classification": row[14],
            "context_features": safe_json_loads(row[15]) if row[15] else {},
            "actual_price_30d": float(row[16]) if row[16] else None,
            "actual_price_90d": float(row[17]) if row[17] else None,
            "actual_price_365d": float(row[18]) if row[18] else None,
            "reward_30d": float(row[19]) if row[19] else None,
            "reward_90d": float(row[20]) if row[20] else None,
            "reward_365d": float(row[21]) if row[21] else None,
            "per_model_rewards": safe_json_loads(row[22]) if row[22] else {},
            "ab_test_group": row[23],
            "policy_version": row[24],
        }


class OutcomeTracker:
    """
    Tracks valuation predictions and updates with actual outcomes.

    This is the core service for the RL feedback loop. It:
    1. Records predictions when analyses are run
    2. Updates predictions with actual prices after time passes
    3. Calculates rewards for RL training

    The service is designed to be called from the valuation pipeline
    and periodically via batch jobs.
    """

    def __init__(
        self,
        dao: Optional[ValuationOutcomesDAO] = None,
        price_history_service: Optional[Any] = None,  # PriceHistoryService
    ):
        """
        Initialize OutcomeTracker.

        Args:
            dao: Data access object for database operations.
            price_history_service: Service for fetching historical prices.
        """
        self.dao = dao or ValuationOutcomesDAO()
        self.price_service = price_history_service

    def record_prediction(
        self,
        symbol: str,
        analysis_date: date,
        blended_fair_value: float,
        current_price: float,
        model_fair_values: Dict[str, float],
        model_weights: Dict[str, float],
        tier_classification: str,
        context_features: ValuationContext,
        fiscal_period: Optional[str] = None,
        ab_test_group: Optional[ABTestGroup] = None,
        policy_version: Optional[str] = None,
        position_type: str = "inferred",
        entry_date: Optional[date] = None,
        exit_date_30d: Optional[date] = None,
        exit_date_90d: Optional[date] = None,
    ) -> Optional[int]:
        """
        Record a valuation prediction for future outcome tracking.

        Should be called immediately after a valuation analysis completes.

        Args:
            symbol: Stock ticker symbol.
            analysis_date: Date of the analysis.
            blended_fair_value: Final blended fair value prediction.
            current_price: Current stock price at analysis time.
            model_fair_values: Fair values from each valuation model.
            model_weights: Weights used for each model.
            tier_classification: Tier classification used.
            context_features: ValuationContext with all features.
            fiscal_period: Fiscal period being analyzed.
            ab_test_group: A/B test group assignment.
            policy_version: RL policy version if RL was used.
            position_type: Position signal type ('LONG', 'SHORT', or 'inferred').
            entry_date: Date when position was entered (defaults to analysis_date).
            exit_date_30d: Date 30 days after entry.
            exit_date_90d: Date 90 days after entry.

        Returns:
            Database record ID if successful, None otherwise.
        """
        # Calculate predicted upside
        predicted_upside_pct = 0.0
        if current_price and current_price > 0:
            predicted_upside_pct = ((blended_fair_value - current_price) / current_price) * 100

        # Convert context to dict if needed
        context_dict = (
            context_features.to_dict() if isinstance(context_features, ValuationContext) else context_features
        )

        # Get AB test group string
        ab_group_str = ab_test_group.value if ab_test_group else None

        return self.dao.insert_prediction(
            symbol=symbol,
            analysis_date=analysis_date,
            fiscal_period=fiscal_period,
            blended_fair_value=blended_fair_value,
            current_price=current_price,
            predicted_upside_pct=predicted_upside_pct,
            model_fair_values=model_fair_values,
            model_weights=model_weights,
            tier_classification=tier_classification,
            context_features=context_dict,
            ab_test_group=ab_group_str,
            policy_version=policy_version,
            position_type=position_type,
            entry_date=entry_date,
            exit_date_30d=exit_date_30d,
            exit_date_90d=exit_date_90d,
        )

    async def update_outcomes(
        self,
        lookback_days: int = 90,
        batch_size: int = 100,
    ) -> Tuple[int, int]:
        """
        Batch update outcomes with actual prices.

        Should be run via cron job or background task.

        Args:
            lookback_days: Only update predictions at least this old.
            batch_size: Number of records to process per batch.

        Returns:
            Tuple of (updated_count, error_count).
        """
        if not self.price_service:
            logger.warning("No price history service configured, skipping outcome update")
            return 0, 0

        updated_count = 0
        error_count = 0

        # Update 30-day outcomes
        pending_30d = self.dao.get_pending_30d_updates(limit=batch_size)
        for record in pending_30d:
            try:
                target_date = record["analysis_date"] + timedelta(days=30)
                price = await self.price_service.get_price_on_date(record["symbol"], target_date)
                if price:
                    self.dao.update_outcome_prices(record["id"], actual_price_30d=price)
                    updated_count += 1
            except Exception as e:
                logger.error(f"Error updating 30d outcome for {record['symbol']}: {e}")
                error_count += 1

        # Update 90-day outcomes and calculate rewards
        pending_90d = self.dao.get_pending_90d_updates(limit=batch_size)
        for record in pending_90d:
            try:
                target_date = record["analysis_date"] + timedelta(days=90)
                price = await self.price_service.get_price_on_date(record["symbol"], target_date)
                if price:
                    self.dao.update_outcome_prices(record["id"], actual_price_90d=price)

                    # Calculate rewards
                    rewards = self._calculate_rewards(
                        blended_fair_value=record["blended_fair_value"],
                        current_price=record["current_price"],
                        actual_price_90d=price,
                        model_fair_values=record["model_fair_values"],
                    )

                    self.dao.update_rewards(
                        record["id"],
                        reward_90d=rewards["reward_90d"],
                        per_model_rewards=rewards["per_model_rewards"],
                    )
                    updated_count += 1
            except Exception as e:
                logger.error(f"Error updating 90d outcome for {record['symbol']}: {e}")
                error_count += 1

        logger.info(f"Outcome update complete: {updated_count} updated, {error_count} errors")
        return updated_count, error_count

    def _calculate_rewards(
        self,
        blended_fair_value: float,
        current_price: float,
        actual_price_90d: float,
        model_fair_values: Dict[str, Optional[float]],
        actual_price_30d: Optional[float] = None,
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate reward signals from prediction outcomes.

        Uses shared RewardCalculator for consistency across all code paths:
        - Risk-adjusted, annualized ROI-weighted rewards
        - Sharpe-like beta adjustment
        - Asymmetric penalties for short vs long errors

        Args:
            blended_fair_value: Predicted fair value.
            current_price: Price at prediction time.
            actual_price_90d: Actual price 90 days later.
            model_fair_values: Individual model predictions.
            actual_price_30d: Actual price 30 days later (optional).
            beta: Stock beta for risk adjustment (default 1.0).

        Returns:
            Dict with reward signals and per-model rewards.
        """
        calculator = get_reward_calculator()

        # Calculate 90d reward using shared calculator
        result_90d = calculator.calculate(
            predicted_fv=blended_fair_value,
            price_at_prediction=current_price,
            actual_price=actual_price_90d,
            days=90,
            beta=beta,
        )

        # Calculate per-model rewards
        per_model_rewards = calculator.calculate_per_model_rewards(
            model_fair_values=model_fair_values,
            price_at_prediction=current_price,
            actual_price=actual_price_90d,
            days=90,
            beta=beta,
        )

        # Calculate 30d reward if available
        reward_30d = None
        if actual_price_30d:
            result_30d = calculator.calculate(
                predicted_fv=blended_fair_value,
                price_at_prediction=current_price,
                actual_price=actual_price_30d,
                days=30,
                beta=beta,
            )
            reward_30d = result_30d.reward

        # Calculate error for reporting
        error_90d = abs(blended_fair_value - actual_price_90d) / actual_price_90d if actual_price_90d > 0 else 0

        return {
            "reward_90d": round(result_90d.reward, 4),
            "reward_30d": round(reward_30d, 4) if reward_30d is not None else None,
            "error_pct_90d": round(error_90d * 100, 2),
            "direction_correct": result_90d.direction_correct,
            "annualized_return_90d": round(result_90d.annualized_return * 100, 2),
            "position_return_90d": round(result_90d.position_return * 100, 2),
            "per_model_rewards": per_model_rewards,
        }

    def get_training_experiences(
        self,
        limit: int = 10000,
        exclude_used: bool = True,
    ) -> List[Experience]:
        """
        Get experiences ready for training.

        Returns experiences that have 90-day outcomes and calculated rewards.

        Args:
            limit: Maximum number of experiences to return.
            exclude_used: If True, exclude experiences already used for training.

        Returns:
            List of Experience objects ready for training.
        """
        records = self.dao.get_training_ready_experiences(limit=limit, exclude_used=exclude_used)

        experiences = []
        for record in records:
            try:
                context = ValuationContext.from_dict(record["context_features"])

                reward = RewardSignal(
                    reward_30d=record["reward_30d"],
                    reward_90d=record["reward_90d"],
                    reward_365d=record["reward_365d"],
                    direction_correct_90d=record.get("per_model_rewards", {}).get("direction_correct"),
                )

                experience = Experience(
                    id=record["id"],
                    symbol=record["symbol"],
                    analysis_date=record["analysis_date"],
                    context=context,
                    weights_used=record["model_weights"],
                    tier_classification=record["tier_classification"],
                    blended_fair_value=record["blended_fair_value"],
                    current_price=record["current_price"],
                    reward=reward,
                    per_model_rewards=record.get("per_model_rewards"),
                )
                experiences.append(experience)
            except Exception as e:
                logger.warning(f"Failed to parse experience {record['id']}: {e}")

        return experiences

    def mark_experiences_used(
        self,
        experience_ids: List[int],
        training_batch_id: int,
    ) -> int:
        """
        Mark experiences as used for training.

        Args:
            experience_ids: List of experience IDs to mark.
            training_batch_id: Training batch ID they were used in.

        Returns:
            Number of records updated.
        """
        return self.dao.mark_used_for_training(experience_ids, training_batch_id)

    def get_symbol_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get prediction history for a symbol.

        Args:
            symbol: Stock ticker symbol.
            limit: Maximum records to return.

        Returns:
            List of prediction records with outcomes.
        """
        return self.dao.get_by_symbol(symbol, limit=limit)


# Factory function for dependency injection
def get_outcome_tracker() -> OutcomeTracker:
    """Get OutcomeTracker instance."""
    return OutcomeTracker()
