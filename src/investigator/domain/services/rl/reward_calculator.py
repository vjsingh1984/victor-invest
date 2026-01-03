"""
Shared Reward Calculator Service

Provides a single source of truth for reward calculation across:
- rl_backtest.py (historical backtesting)
- outcome_tracker.py (production outcome tracking)
- rl_update_outcomes.py (cron job updates)

This prevents drift between training and production reward signals.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Breakdown of reward calculation components."""
    reward: float  # Final reward in [-1, 1]
    annualized_return: float  # Annualized position return
    risk_adjusted_return: float  # After beta adjustment
    direction_correct: bool  # Was direction prediction correct
    position_return: float  # Raw position return (profit/loss)
    direction_factor: float  # Multiplier applied for direction
    predicted_direction: int  # 1 = long, -1 = short
    actual_direction: int  # 1 = up, -1 = down


class RewardCalculator:
    """
    Calculates risk-adjusted, annualized ROI-weighted rewards.

    Features:
    - Annualized returns for time normalization
    - Sharpe-like beta risk adjustment
    - Asymmetric penalties for short vs long errors
    - Consistent across all code paths

    Example:
        calculator = RewardCalculator()
        result = calculator.calculate(
            predicted_fv=300.0,
            price_at_prediction=250.0,
            actual_price=280.0,
            days=90,
            beta=1.2
        )
        print(f"Reward: {result.reward}, Direction: {'correct' if result.direction_correct else 'wrong'}")
    """

    def __init__(
        self,
        beta_min: float = 0.3,
        beta_max: float = 3.0,
        gain_risk_power: float = 0.5,  # sqrt(beta) for gains
        loss_risk_power: float = 0.75,  # beta^0.75 for losses
        short_wrong_base_multiplier: float = 1.5,  # Base penalty for short wrong
        short_squeeze_sensitivity: float = 2.0,  # How much squeeze amplifies penalty
        long_wrong_dampening: float = 0.7,  # Long wrong is recoverable
        direction_correct_bonus: float = 1.2,  # Bonus for correct direction
        normalization_scale: float = 1.0,  # Scale for tanh normalization
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gain_risk_power = gain_risk_power
        self.loss_risk_power = loss_risk_power
        self.short_wrong_base_multiplier = short_wrong_base_multiplier
        self.short_squeeze_sensitivity = short_squeeze_sensitivity
        self.long_wrong_dampening = long_wrong_dampening
        self.direction_correct_bonus = direction_correct_bonus
        self.normalization_scale = normalization_scale

    def calculate(
        self,
        predicted_fv: float,
        price_at_prediction: float,
        actual_price: float,
        days: int = 90,
        beta: float = 1.0,
    ) -> RewardComponents:
        """
        Calculate risk-adjusted, annualized reward signal.

        Args:
            predicted_fv: Predicted fair value
            price_at_prediction: Stock price when prediction was made
            actual_price: Stock price after `days` elapsed
            days: Number of days between prediction and outcome
            beta: Stock beta for risk adjustment (default 1.0)

        Returns:
            RewardComponents with full breakdown of calculation
        """
        # Handle invalid inputs
        if actual_price <= 0 or predicted_fv <= 0 or price_at_prediction <= 0:
            return RewardComponents(
                reward=0.0,
                annualized_return=0.0,
                risk_adjusted_return=0.0,
                direction_correct=False,
                position_return=0.0,
                direction_factor=1.0,
                predicted_direction=0,
                actual_direction=0,
            )

        # Clamp beta to reasonable range
        beta = max(self.beta_min, min(self.beta_max, beta))

        # Raw price return
        raw_return = (actual_price - price_at_prediction) / price_at_prediction

        # Determine directions
        # FV > Price → LONG (expect price to rise to FV)
        # FV < Price → SHORT (expect price to fall to FV)
        predicted_direction = 1 if predicted_fv > price_at_prediction else -1
        actual_direction = 1 if actual_price > price_at_prediction else -1
        direction_correct = predicted_direction == actual_direction

        # Position return from trader's perspective
        # Long profits when stock goes up (raw_return positive)
        # Short profits when stock goes down (raw_return negative)
        if predicted_direction == 1:  # Long
            position_return = raw_return
        else:  # Short
            position_return = -raw_return  # Profit when stock drops

        # Annualized position return
        # For gains: (1 + r)^(365/days) - 1
        # For losses: -((1 / (1 + r))^(365/days) - 1) to handle compounding losses
        if position_return > -1:
            if position_return >= 0:
                annualized_return = (1 + position_return) ** (365 / days) - 1
            else:
                annualized_return = -((1 / (1 + position_return)) ** (365 / days) - 1)
        else:
            annualized_return = -10.0  # Total loss

        # Risk adjustment (Sharpe-like for gains, Sortino-like for losses)
        if annualized_return >= 0:
            # Divide gains by sqrt(beta) - higher beta = less reward per unit gain
            risk_adjusted = annualized_return / np.power(beta, self.gain_risk_power)
        else:
            # Multiply losses by beta^0.75 - higher beta = more penalty for losses
            risk_adjusted = annualized_return * np.power(beta, self.loss_risk_power)

        # Asymmetric direction factor
        # Key insight: Short squeeze can liquidate, long losses are recoverable
        if direction_correct:
            direction_factor = self.direction_correct_bonus
        else:
            if predicted_direction == -1:  # Short wrong - AMPLIFY
                # Stock went up when we expected down - squeeze risk
                squeeze_multiplier = 1.0 + max(0, raw_return) * self.short_squeeze_sensitivity
                direction_factor = self.short_wrong_base_multiplier * squeeze_multiplier
            else:  # Long wrong - DAMPEN
                # Stock went down when we expected up - losses are recoverable over time
                direction_factor = self.long_wrong_dampening

        # Apply direction factor
        weighted_return = risk_adjusted * direction_factor

        # Normalize to [-1, 1] using tanh
        reward = float(np.tanh(weighted_return / self.normalization_scale))

        return RewardComponents(
            reward=reward,
            annualized_return=float(annualized_return),
            risk_adjusted_return=float(risk_adjusted),
            direction_correct=direction_correct,
            position_return=float(position_return),
            direction_factor=float(direction_factor),
            predicted_direction=predicted_direction,
            actual_direction=actual_direction,
        )

    def calculate_simple(
        self,
        predicted_fv: float,
        price_at_prediction: float,
        actual_price: float,
        days: int = 90,
        beta: float = 1.0,
    ) -> float:
        """
        Calculate reward and return just the scalar value.

        Convenience method for cases where only the final reward is needed.
        """
        return self.calculate(predicted_fv, price_at_prediction, actual_price, days, beta).reward

    def calculate_per_model_rewards(
        self,
        model_fair_values: Dict[str, Optional[float]],
        price_at_prediction: float,
        actual_price: float,
        days: int = 90,
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate rewards for each valuation model individually.

        Args:
            model_fair_values: Dict of model_name -> fair_value
            price_at_prediction: Price when prediction was made
            actual_price: Actual price after days elapsed
            days: Days between prediction and outcome
            beta: Stock beta for risk adjustment

        Returns:
            Dict with reward breakdown per model
        """
        per_model = {}

        for model_name, fv in model_fair_values.items():
            if fv is not None and fv > 0:
                result = self.calculate(
                    predicted_fv=fv,
                    price_at_prediction=price_at_prediction,
                    actual_price=actual_price,
                    days=days,
                    beta=beta,
                )
                per_model[model_name] = {
                    "reward": round(result.reward, 4),
                    "direction_correct": result.direction_correct,
                    "position_return": round(result.position_return * 100, 2),  # As percentage
                    "annualized_return": round(result.annualized_return * 100, 2),
                }

        return per_model


# Singleton instance for convenience
_calculator: Optional[RewardCalculator] = None


def get_reward_calculator() -> RewardCalculator:
    """Get shared RewardCalculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = RewardCalculator()
    return _calculator


def calculate_reward(
    predicted_fv: float,
    price_at_prediction: float,
    actual_price: float,
    days: int = 90,
    beta: float = 1.0,
) -> float:
    """
    Convenience function to calculate reward.

    Use this instead of implementing reward calculation locally.
    """
    return get_reward_calculator().calculate_simple(
        predicted_fv, price_at_prediction, actual_price, days, beta
    )
