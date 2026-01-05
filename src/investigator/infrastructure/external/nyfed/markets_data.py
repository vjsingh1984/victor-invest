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

"""NY Fed Markets Data Client.

This module provides access to New York Federal Reserve economic indicators
via their public data downloads.

Key Data Series:
- Recession Probability: Probability of recession 12 months ahead based on yield curve
- GSCPI: Global Supply Chain Pressure Index
- Term Premia: ACM Term Premia estimates

Data Sources:
- https://www.newyorkfed.org/research/capital_markets/ycfaq.html (Recession Probability)
- https://www.newyorkfed.org/research/policy/gscpi (GSCPI)

Example:
    client = get_nyfed_client()

    # Get recession probability
    prob = await client.get_recession_probability()
    print(f"12-month recession probability: {prob.probability}%")

    # Get GSCPI
    gscpi = await client.get_gscpi()
    print(f"GSCPI: {gscpi.value} ({gscpi.interpretation})")
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# NY Fed data URLs
RECESSION_PROB_URL = "https://www.newyorkfed.org/medialibrary/media/research/capital_markets/Prob_Rec.xlsx"
GSCPI_URL = "https://www.newyorkfed.org/medialibrary/media/research/policy/gscpi_data.xlsx"


class RecessionRiskLevel(Enum):
    """Classification of recession risk."""

    VERY_LOW = "very_low"  # < 10%
    LOW = "low"  # 10-20%
    MODERATE = "moderate"  # 20-35%
    ELEVATED = "elevated"  # 35-50%
    HIGH = "high"  # 50-70%
    VERY_HIGH = "very_high"  # > 70%


@dataclass
class RecessionProbability:
    """NY Fed recession probability data.

    The NY Fed model estimates the probability of recession 12 months ahead
    using the slope of the yield curve (10Y-3M spread).

    Attributes:
        date: Date of the estimate
        probability: Probability as percentage (0-100)
        spread_10y_3m: Yield spread used in calculation (bps)
        risk_level: Classified risk level
        historical_avg: Historical average probability
    """

    date: date
    probability: float
    spread_10y_3m: Optional[float] = None
    risk_level: RecessionRiskLevel = RecessionRiskLevel.LOW
    historical_avg: float = 14.5  # Long-term average ~14.5%

    def __post_init__(self):
        """Classify risk level based on probability."""
        if self.probability < 10:
            self.risk_level = RecessionRiskLevel.VERY_LOW
        elif self.probability < 20:
            self.risk_level = RecessionRiskLevel.LOW
        elif self.probability < 35:
            self.risk_level = RecessionRiskLevel.MODERATE
        elif self.probability < 50:
            self.risk_level = RecessionRiskLevel.ELEVATED
        elif self.probability < 70:
            self.risk_level = RecessionRiskLevel.HIGH
        else:
            self.risk_level = RecessionRiskLevel.VERY_HIGH

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": str(self.date),
            "probability_pct": round(self.probability, 2),
            "risk_level": self.risk_level.value,
            "spread_10y_3m_bps": self.spread_10y_3m,
            "vs_historical_avg": round(self.probability - self.historical_avg, 2),
            "interpretation": self._get_interpretation(),
        }

    def _get_interpretation(self) -> str:
        """Get human-readable interpretation."""
        if self.risk_level == RecessionRiskLevel.VERY_LOW:
            return "Recession very unlikely in next 12 months"
        elif self.risk_level == RecessionRiskLevel.LOW:
            return "Low recession risk, economy appears stable"
        elif self.risk_level == RecessionRiskLevel.MODERATE:
            return "Moderate recession risk, monitor closely"
        elif self.risk_level == RecessionRiskLevel.ELEVATED:
            return "Elevated recession risk, caution advised"
        elif self.risk_level == RecessionRiskLevel.HIGH:
            return "High recession risk, defensive positioning recommended"
        else:
            return "Very high recession risk, significant defensive measures advised"


class GSCPILevel(Enum):
    """Classification of GSCPI levels."""

    VERY_LOW = "very_low"  # < -1.5 std dev
    LOW = "low"  # -1.5 to -0.5 std dev
    NORMAL = "normal"  # -0.5 to 0.5 std dev
    ELEVATED = "elevated"  # 0.5 to 1.5 std dev
    HIGH = "high"  # 1.5 to 2.5 std dev
    VERY_HIGH = "very_high"  # > 2.5 std dev


@dataclass
class GSCPIData:
    """Global Supply Chain Pressure Index data.

    The GSCPI measures global supply chain conditions using multiple
    indicators including shipping costs, delivery times, and backlogs.
    Values are expressed in standard deviations from the historical mean.

    Attributes:
        date: Date of the reading
        value: GSCPI value (standard deviations from mean)
        level: Classified pressure level
        one_month_change: Change from prior month
        yoy_change: Year-over-year change
    """

    date: date
    value: float
    level: GSCPILevel = GSCPILevel.NORMAL
    one_month_change: Optional[float] = None
    yoy_change: Optional[float] = None

    def __post_init__(self):
        """Classify pressure level based on value."""
        if self.value < -1.5:
            self.level = GSCPILevel.VERY_LOW
        elif self.value < -0.5:
            self.level = GSCPILevel.LOW
        elif self.value < 0.5:
            self.level = GSCPILevel.NORMAL
        elif self.value < 1.5:
            self.level = GSCPILevel.ELEVATED
        elif self.value < 2.5:
            self.level = GSCPILevel.HIGH
        else:
            self.level = GSCPILevel.VERY_HIGH

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": str(self.date),
            "value": round(self.value, 3),
            "level": self.level.value,
            "one_month_change": round(self.one_month_change, 3) if self.one_month_change else None,
            "yoy_change": round(self.yoy_change, 3) if self.yoy_change else None,
            "interpretation": self._get_interpretation(),
        }

    def _get_interpretation(self) -> str:
        """Get human-readable interpretation."""
        if self.level == GSCPILevel.VERY_LOW:
            return "Supply chains very loose, deflationary pressure"
        elif self.level == GSCPILevel.LOW:
            return "Supply chains loose, favorable conditions"
        elif self.level == GSCPILevel.NORMAL:
            return "Supply chain conditions normal"
        elif self.level == GSCPILevel.ELEVATED:
            return "Supply chain stress elevated, potential inflation pressure"
        elif self.level == GSCPILevel.HIGH:
            return "Significant supply chain stress, inflation concern"
        else:
            return "Severe supply chain disruption, major inflation risk"


class NYFedDataClient:
    """Client for NY Fed economic data.

    Provides async access to NY Fed economic indicators including
    recession probability and GSCPI.

    SOLID: Single Responsibility - only handles NY Fed data access
    """

    def __init__(self, timeout: int = 30):
        """Initialize NY Fed client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_recession_probability(self) -> Optional[RecessionProbability]:
        """Get latest NY Fed recession probability.

        The NY Fed model uses the 10Y-3M Treasury spread to estimate
        the probability of recession 12 months ahead.

        Returns:
            RecessionProbability or None if unavailable
        """
        try:
            # Try to get from FRED first (more reliable)
            prob = await self._get_recession_prob_from_fred()
            if prob:
                return prob

            # Fallback: Calculate from yield spread
            prob = await self._calculate_recession_prob_from_spread()
            if prob:
                return prob

            logger.warning("Could not retrieve recession probability")
            return None

        except Exception as e:
            logger.error(f"Error fetching recession probability: {e}")
            return None

    async def _get_recession_prob_from_fred(self) -> Optional[RecessionProbability]:
        """Get recession probability from FRED if available.

        FRED series: RECPROUSM156N (Smoothed U.S. Recession Probabilities)
        """
        try:
            from investigator.infrastructure.external.fred.macro_indicators import get_macro_indicator_service

            service = get_macro_indicator_service()

            # Get recession probability from FRED
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(None, lambda: service.get_latest_value("RECPROUSM156N"))

            if value is not None:
                return RecessionProbability(
                    date=date.today(),
                    probability=float(value),
                )

        except ImportError:
            logger.debug("FRED service not available")
        except Exception as e:
            logger.debug(f"FRED recession prob failed: {e}")

        return None

    async def _calculate_recession_prob_from_spread(self) -> Optional[RecessionProbability]:
        """Calculate recession probability from yield spread.

        Uses the NY Fed's probit model approximation:
        P(recession) = Normal_CDF(-0.5333 - 0.6330 * spread)

        where spread is the 10Y-3M Treasury spread in percentage points.
        """
        try:
            import math

            from investigator.infrastructure.external.treasury import get_treasury_client

            treasury = get_treasury_client()
            curve = await treasury.get_yield_curve()

            if curve is None or curve.yield_10y is None or curve.yield_3m is None:
                return None

            # Calculate spread in percentage points
            spread = curve.yield_10y - curve.yield_3m

            # NY Fed probit model coefficients (approximation)
            # Actual model uses more sophisticated approach
            z = -0.5333 - 0.6330 * spread

            # Approximate normal CDF using logistic function
            # P = 1 / (1 + exp(-1.702 * z))
            prob = 1 / (1 + math.exp(-1.702 * z)) * 100

            return RecessionProbability(
                date=curve.date,
                probability=prob,
                spread_10y_3m=curve.spread_10y_3m,
            )

        except Exception as e:
            logger.debug(f"Spread-based calculation failed: {e}")
            return None

    async def get_gscpi(self) -> Optional[GSCPIData]:
        """Get latest Global Supply Chain Pressure Index.

        The GSCPI is published monthly by the NY Fed and measures
        global supply chain conditions.

        Returns:
            GSCPIData or None if unavailable
        """
        try:
            # Try to get from FRED first
            gscpi = await self._get_gscpi_from_fred()
            if gscpi:
                return gscpi

            logger.warning("Could not retrieve GSCPI")
            return None

        except Exception as e:
            logger.error(f"Error fetching GSCPI: {e}")
            return None

    async def _get_gscpi_from_fred(self) -> Optional[GSCPIData]:
        """Get GSCPI from FRED if available.

        FRED series: GSCPI (Global Supply Chain Pressure Index)
        """
        try:
            from investigator.infrastructure.external.fred.macro_indicators import get_macro_indicator_service

            service = get_macro_indicator_service()
            loop = asyncio.get_event_loop()

            # Get current value
            value = await loop.run_in_executor(None, lambda: service.get_latest_value("GSCPI"))

            if value is None:
                return None

            # Try to get historical for change calculations
            history = await loop.run_in_executor(None, lambda: service.get_time_series("GSCPI", days=400))

            one_month_change = None
            yoy_change = None

            if history and len(history) > 1:
                # Calculate month-over-month change
                if len(history) >= 2:
                    one_month_change = value - history[1].get("value", value)

                # Calculate year-over-year change
                if len(history) >= 13:
                    yoy_change = value - history[12].get("value", value)

            return GSCPIData(
                date=date.today(),
                value=float(value),
                one_month_change=one_month_change,
                yoy_change=yoy_change,
            )

        except ImportError:
            logger.debug("FRED service not available for GSCPI")
        except Exception as e:
            logger.debug(f"FRED GSCPI failed: {e}")

        return None

    async def get_recession_probability_history(self, months: int = 24) -> List[Dict[str, Any]]:
        """Get historical recession probability.

        Args:
            months: Number of months of history

        Returns:
            List of {date, probability, risk_level} dictionaries
        """
        try:
            from investigator.infrastructure.external.fred.macro_indicators import get_macro_indicator_service

            service = get_macro_indicator_service()
            loop = asyncio.get_event_loop()

            history = await loop.run_in_executor(
                None, lambda: service.get_time_series("RECPROUSM156N", days=months * 31)
            )

            if not history:
                return []

            result = []
            for entry in history[:months]:
                prob = RecessionProbability(
                    date=(
                        datetime.strptime(entry["date"], "%Y-%m-%d").date()
                        if isinstance(entry.get("date"), str)
                        else entry.get("date", date.today())
                    ),
                    probability=float(entry.get("value", 0)),
                )
                result.append(
                    {
                        "date": str(prob.date),
                        "probability": prob.probability,
                        "risk_level": prob.risk_level.value,
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error fetching recession probability history: {e}")
            return []

    async def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive NY Fed market summary.

        Returns:
            Dictionary with recession probability and GSCPI
        """
        recession = await self.get_recession_probability()
        gscpi = await self.get_gscpi()

        return {
            "recession_probability": recession.to_dict() if recession else None,
            "gscpi": gscpi.to_dict() if gscpi else None,
            "summary": {
                "recession_risk": recession.risk_level.value if recession else "unknown",
                "supply_chain_pressure": gscpi.level.value if gscpi else "unknown",
            },
        }


# Singleton instance
_nyfed_client: Optional[NYFedDataClient] = None


def get_nyfed_client() -> NYFedDataClient:
    """Get or create singleton NY Fed client instance."""
    global _nyfed_client
    if _nyfed_client is None:
        _nyfed_client = NYFedDataClient()
    return _nyfed_client
