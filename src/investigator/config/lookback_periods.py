# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard Lookback Periods Configuration.

This module defines consistent lookback periods for all data sources
and analysis tools. Use these constants instead of hardcoding values
to ensure consistency across CLI, batch processing, and RL backtesting.

Rationale for default periods:
- Insider Trading: 90 days captures quarterly patterns and cluster signals
- Institutional (13F): 90 days aligns with SEC quarterly reporting
- Short Interest: 30 days for recent sentiment, 90 days for trends
- Credit Risk: 365 days for annual financial health assessment
- Macro Indicators: 30 days for current regime, 90 days for trends
- Technical Analysis: Multiple periods for different signal types
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class InsiderTradingPeriods:
    """Lookback periods for insider transaction analysis."""

    # Recent activity (cluster detection, urgency signals)
    recent_days: int = 14
    # Short-term sentiment (recent quarter)
    short_term_days: int = 30
    # Standard sentiment calculation period
    standard_days: int = 90
    # Long-term accumulation patterns
    long_term_days: int = 365
    # Scheduled collector default
    collector_hours: int = 6


@dataclass(frozen=True)
class InstitutionalPeriods:
    """Lookback periods for institutional holdings (13F) analysis."""

    # Recent quarter filings
    recent_days: int = 90
    # Year-over-year comparison
    annual_days: int = 365
    # Full cycle analysis (2 years)
    cycle_days: int = 730


@dataclass(frozen=True)
class ShortInterestPeriods:
    """Lookback periods for short interest analysis."""

    # Recent changes
    recent_days: int = 14
    # Standard analysis period
    standard_days: int = 30
    # Trend analysis
    trend_days: int = 90


@dataclass(frozen=True)
class CreditRiskPeriods:
    """Lookback periods for credit risk analysis."""

    # Refresh interval (weekly recalculation)
    refresh_days: int = 7
    # Standard calculation uses annual data
    calculation_days: int = 365


@dataclass(frozen=True)
class MacroIndicatorPeriods:
    """Lookback periods for macro/market regime analysis."""

    # Current regime assessment
    current_days: int = 30
    # Trend analysis
    trend_days: int = 90
    # Cycle analysis
    cycle_days: int = 365


@dataclass(frozen=True)
class TechnicalAnalysisPeriods:
    """Lookback periods for technical analysis."""

    # Short-term momentum
    momentum_short_days: int = 14
    # Medium-term trend
    momentum_medium_days: int = 50
    # Long-term trend
    momentum_long_days: int = 200
    # Volatility calculation
    volatility_days: int = 20
    # Support/resistance analysis
    support_resistance_days: int = 252


@dataclass(frozen=True)
class RLBacktestPeriods:
    """Lookback periods for RL backtesting and training.

    These periods are used for:
    - Feature extraction lookback
    - Holding period evaluation
    - Training data windowing
    """

    # Standard lookback months for analysis (quarterly snapshots)
    standard_lookback_months: List[int] = (3, 6, 9, 12)
    # Extended lookback for full cycle analysis
    extended_lookback_months: List[int] = (3, 6, 9, 12, 18, 24, 36)
    # Holding periods for reward calculation (days)
    holding_periods: Dict[str, int] = None
    # Minimum training examples per symbol
    min_training_examples: int = 8
    # Outcome evaluation lookback
    outcome_lookback_days: int = 90

    def __post_init__(self):
        if self.holding_periods is None:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(
                self,
                "holding_periods",
                {
                    "1m": 30,
                    "3m": 90,
                    "6m": 180,
                    "12m": 365,
                    "18m": 548,
                    "24m": 730,
                    "36m": 1095,
                },
            )


# Singleton instances for easy import
INSIDER_PERIODS = InsiderTradingPeriods()
INSTITUTIONAL_PERIODS = InstitutionalPeriods()
SHORT_INTEREST_PERIODS = ShortInterestPeriods()
CREDIT_RISK_PERIODS = CreditRiskPeriods()
MACRO_PERIODS = MacroIndicatorPeriods()
TECHNICAL_PERIODS = TechnicalAnalysisPeriods()
RL_BACKTEST_PERIODS = RLBacktestPeriods()


def get_all_periods() -> Dict[str, object]:
    """Get all period configurations as a dictionary."""
    return {
        "insider": INSIDER_PERIODS,
        "institutional": INSTITUTIONAL_PERIODS,
        "short_interest": SHORT_INTEREST_PERIODS,
        "credit_risk": CREDIT_RISK_PERIODS,
        "macro": MACRO_PERIODS,
        "technical": TECHNICAL_PERIODS,
        "rl_backtest": RL_BACKTEST_PERIODS,
    }
