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

"""Yield Curve Analyzer Service.

This module provides analysis of the Treasury yield curve to derive
market signals for investment decisions.

Key Insights:
- Yield curve shape indicates market expectations for growth and inflation
- Inverted curves (10Y < 2Y) historically precede recessions
- Steep curves indicate economic expansion expectations
- Flat curves suggest uncertainty or late-cycle conditions

Signal Mappings:
- STEEP: Risk-on, favor growth/cyclicals, equity overweight
- NORMAL: Balanced positioning, moderate risk
- FLAT: Cautious, favor quality, reduce duration risk
- INVERTED: Defensive, favor defensives/bonds, reduce equity
- DEEPLY_INVERTED: Strong defensive, recession imminent

Example:
    analyzer = get_yield_curve_analyzer()

    # Get current analysis
    analysis = await analyzer.analyze()
    print(f"Curve shape: {analysis.shape}")
    print(f"Investment signal: {analysis.investment_signal}")

    # Get historical shape transitions
    history = await analyzer.get_shape_history(days=365)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class YieldCurveShape(Enum):
    """Classification of yield curve shapes."""

    STEEP = "steep"  # Spread > 150 bps
    NORMAL = "normal"  # Spread 50-150 bps
    FLAT = "flat"  # Spread 0-50 bps
    INVERTED = "inverted"  # Spread -50 to 0 bps
    DEEPLY_INVERTED = "deeply_inverted"  # Spread < -50 bps
    UNKNOWN = "unknown"


class InvestmentSignal(Enum):
    """Investment signals derived from yield curve."""

    RISK_ON = "risk_on"  # Favor growth, cyclicals
    MODERATE_RISK = "moderate_risk"  # Balanced positioning
    CAUTIOUS = "cautious"  # Favor quality, reduce risk
    DEFENSIVE = "defensive"  # Favor defensives, reduce equity
    STRONGLY_DEFENSIVE = "strongly_defensive"  # Max defensive


@dataclass
class YieldCurveAnalysis:
    """Complete yield curve analysis result.

    Attributes:
        date: Analysis date
        shape: Classified curve shape
        spread_10y_2y_bps: 10Y-2Y spread in basis points
        spread_10y_3m_bps: 10Y-3M spread in basis points
        investment_signal: Derived investment signal
        yield_10y: Current 10-year yield
        yield_2y: Current 2-year yield
        yield_3m: Current 3-month yield
        risk_free_rate: Suggested risk-free rate for WACC
        days_inverted: Consecutive days inverted (if applicable)
        historical_context: Context vs historical averages
        warnings: Any data quality warnings
    """

    date: date
    shape: YieldCurveShape = YieldCurveShape.UNKNOWN
    spread_10y_2y_bps: Optional[float] = None
    spread_10y_3m_bps: Optional[float] = None
    investment_signal: InvestmentSignal = InvestmentSignal.MODERATE_RISK
    yield_10y: Optional[float] = None
    yield_2y: Optional[float] = None
    yield_3m: Optional[float] = None
    risk_free_rate: Optional[float] = None
    days_inverted: int = 0
    historical_context: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived values."""
        # Set suggested risk-free rate (use 10Y Treasury)
        if self.yield_10y is not None:
            self.risk_free_rate = self.yield_10y

        # Determine investment signal from shape
        self._set_investment_signal()

    def _set_investment_signal(self):
        """Set investment signal based on curve shape."""
        signal_map = {
            YieldCurveShape.STEEP: InvestmentSignal.RISK_ON,
            YieldCurveShape.NORMAL: InvestmentSignal.MODERATE_RISK,
            YieldCurveShape.FLAT: InvestmentSignal.CAUTIOUS,
            YieldCurveShape.INVERTED: InvestmentSignal.DEFENSIVE,
            YieldCurveShape.DEEPLY_INVERTED: InvestmentSignal.STRONGLY_DEFENSIVE,
            YieldCurveShape.UNKNOWN: InvestmentSignal.MODERATE_RISK,
        }
        self.investment_signal = signal_map.get(self.shape, InvestmentSignal.MODERATE_RISK)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": str(self.date),
            "shape": self.shape.value,
            "spreads": {
                "10y_2y_bps": self.spread_10y_2y_bps,
                "10y_3m_bps": self.spread_10y_3m_bps,
            },
            "yields": {
                "10y": self.yield_10y,
                "2y": self.yield_2y,
                "3m": self.yield_3m,
            },
            "signals": {
                "investment_signal": self.investment_signal.value,
                "risk_free_rate": self.risk_free_rate,
            },
            "inversion": {
                "is_inverted": self.shape in (YieldCurveShape.INVERTED, YieldCurveShape.DEEPLY_INVERTED),
                "days_inverted": self.days_inverted,
            },
            "historical_context": self.historical_context,
            "interpretation": self._get_interpretation(),
            "warnings": self.warnings,
        }

    def _get_interpretation(self) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            YieldCurveShape.STEEP: (
                "Steep curve signals strong growth expectations. "
                "Favor cyclical sectors and growth stocks. "
                "Banks typically benefit from wider margins."
            ),
            YieldCurveShape.NORMAL: (
                "Normal curve indicates balanced growth outlook. " "Maintain diversified positioning across sectors."
            ),
            YieldCurveShape.FLAT: (
                "Flat curve suggests economic uncertainty or late-cycle. "
                "Reduce duration risk, favor quality names. "
                "Consider increasing cash allocation."
            ),
            YieldCurveShape.INVERTED: (
                "Inverted curve historically precedes recessions (12-18 months). "
                "Favor defensive sectors (utilities, staples, healthcare). "
                "Consider reducing equity exposure."
            ),
            YieldCurveShape.DEEPLY_INVERTED: (
                "Deeply inverted curve signals high recession probability. "
                "Strong defensive positioning recommended. "
                "Favor high-quality bonds and defensive equities."
            ),
        }
        return interpretations.get(self.shape, "Yield curve data unavailable")

    @property
    def equity_adjustment(self) -> float:
        """Suggested equity allocation adjustment (-20% to +10%)."""
        adjustments = {
            YieldCurveShape.STEEP: 0.10,  # +10%
            YieldCurveShape.NORMAL: 0.0,  # No change
            YieldCurveShape.FLAT: -0.05,  # -5%
            YieldCurveShape.INVERTED: -0.10,  # -10%
            YieldCurveShape.DEEPLY_INVERTED: -0.20,  # -20%
        }
        return adjustments.get(self.shape, 0.0)

    @property
    def wacc_spread_adjustment(self) -> float:
        """Suggested WACC spread adjustment (bps)."""
        # Add spread to risk-free rate based on curve shape
        adjustments = {
            YieldCurveShape.STEEP: 0,  # Normal conditions
            YieldCurveShape.NORMAL: 0,  # Normal conditions
            YieldCurveShape.FLAT: 25,  # Add 25 bps
            YieldCurveShape.INVERTED: 50,  # Add 50 bps
            YieldCurveShape.DEEPLY_INVERTED: 100,  # Add 100 bps
        }
        return adjustments.get(self.shape, 0)


class YieldCurveAnalyzer:
    """Service for analyzing Treasury yield curve.

    Provides yield curve shape classification and investment signals
    based on Treasury yield data.

    SOLID: Single Responsibility - only handles yield curve analysis
    """

    # Historical average spread (10Y-2Y) is approximately 90 bps
    HISTORICAL_AVG_SPREAD = 90

    def __init__(self, treasury_client=None):
        """Initialize analyzer with optional treasury client.

        Args:
            treasury_client: TreasuryApiClient instance (created if not provided)
        """
        self._treasury_client = treasury_client

    def _get_treasury_client(self):
        """Lazy-load treasury client to avoid circular imports."""
        if self._treasury_client is None:
            from investigator.infrastructure.external.treasury import get_treasury_client

            self._treasury_client = get_treasury_client()
        return self._treasury_client

    async def analyze(self, as_of_date: Optional[date] = None) -> YieldCurveAnalysis:
        """Analyze current yield curve.

        Args:
            as_of_date: Date for analysis (default: latest available)

        Returns:
            YieldCurveAnalysis with shape and signals
        """
        try:
            client = self._get_treasury_client()
            curve = await client.get_yield_curve(as_of_date)

            if curve is None:
                analysis = YieldCurveAnalysis(date=as_of_date or date.today())
                analysis.warnings.append("Could not retrieve yield curve data")
                return analysis

            # Classify shape
            shape = self._classify_shape(curve.spread_10y_2y)

            # Build analysis
            analysis = YieldCurveAnalysis(
                date=curve.date,
                shape=shape,
                spread_10y_2y_bps=curve.spread_10y_2y,
                spread_10y_3m_bps=curve.spread_10y_3m,
                yield_10y=curve.yield_10y,
                yield_2y=curve.yield_2y,
                yield_3m=curve.yield_3m,
            )

            # Add historical context
            analysis.historical_context = {
                "vs_historical_avg_bps": (
                    curve.spread_10y_2y - self.HISTORICAL_AVG_SPREAD if curve.spread_10y_2y else None
                ),
                "historical_avg_spread_bps": self.HISTORICAL_AVG_SPREAD,
            }

            # Count days inverted if applicable
            if shape in (YieldCurveShape.INVERTED, YieldCurveShape.DEEPLY_INVERTED):
                analysis.days_inverted = await self._count_inversion_days()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing yield curve: {e}")
            analysis = YieldCurveAnalysis(date=as_of_date or date.today())
            analysis.warnings.append(f"Analysis error: {str(e)}")
            return analysis

    def _classify_shape(self, spread_bps: Optional[float]) -> YieldCurveShape:
        """Classify yield curve shape from 10Y-2Y spread.

        Args:
            spread_bps: 10Y-2Y spread in basis points

        Returns:
            YieldCurveShape classification
        """
        if spread_bps is None:
            return YieldCurveShape.UNKNOWN
        if spread_bps < -50:
            return YieldCurveShape.DEEPLY_INVERTED
        if spread_bps < 0:
            return YieldCurveShape.INVERTED
        if spread_bps < 50:
            return YieldCurveShape.FLAT
        if spread_bps < 150:
            return YieldCurveShape.NORMAL
        return YieldCurveShape.STEEP

    async def _count_inversion_days(self) -> int:
        """Count consecutive days of yield curve inversion.

        Returns:
            Number of consecutive inverted days
        """
        try:
            client = self._get_treasury_client()
            history = await client.get_spread_history(days=365, spread_type="10y_2y")

            if not history:
                return 0

            # Count consecutive inverted days from most recent
            count = 0
            for entry in history:
                if entry.get("is_inverted", False):
                    count += 1
                else:
                    break

            return count

        except Exception as e:
            logger.debug(f"Error counting inversion days: {e}")
            return 0

    async def get_shape_history(self, days: int = 365) -> List[Dict[str, Any]]:
        """Get historical yield curve shapes.

        Args:
            days: Number of days of history

        Returns:
            List of {date, shape, spread_bps} dictionaries
        """
        try:
            client = self._get_treasury_client()
            history = await client.get_spread_history(days=days, spread_type="10y_2y")

            result = []
            for entry in history:
                spread = entry.get("spread_bps")
                shape = self._classify_shape(spread)
                result.append(
                    {
                        "date": entry.get("date"),
                        "shape": shape.value,
                        "spread_bps": spread,
                        "is_inverted": entry.get("is_inverted", False),
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error fetching shape history: {e}")
            return []

    async def get_valuation_adjustments(self) -> Dict[str, Any]:
        """Get valuation model adjustments based on yield curve.

        Returns:
            Dictionary with suggested adjustments for valuation models
        """
        analysis = await self.analyze()

        return {
            "risk_free_rate": analysis.risk_free_rate,
            "wacc_spread_adjustment_bps": analysis.wacc_spread_adjustment,
            "equity_allocation_adjustment": analysis.equity_adjustment,
            "discount_rate_adjustment": (
                analysis.wacc_spread_adjustment / 100 if analysis.wacc_spread_adjustment else 0
            ),
            "curve_shape": analysis.shape.value,
            "investment_signal": analysis.investment_signal.value,
        }


# Singleton instance
_yield_curve_analyzer: Optional[YieldCurveAnalyzer] = None


def get_yield_curve_analyzer() -> YieldCurveAnalyzer:
    """Get or create singleton analyzer instance."""
    global _yield_curve_analyzer
    if _yield_curve_analyzer is None:
        _yield_curve_analyzer = YieldCurveAnalyzer()
    return _yield_curve_analyzer
