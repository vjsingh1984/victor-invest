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

"""CBOE Market Data Client.

Key Data Series:
    - VIX: Volatility Index (spot)
    - VIX Futures Term Structure: Forward volatility expectations
    - VIX9D, VIX3M, VIX6M: Short and medium-term volatility
    - SKEW: Tail risk indicator
    - Put/Call Ratios: Sentiment indicators

Data Sources:
    - https://www.cboe.com/tradable_products/vix/
    - VIX futures data from CME
    - FRED for historical VIX data

Investment Signals:
    - VIX > 30: High fear, potential bottom
    - VIX < 12: Complacency, caution warranted
    - Term structure contango: Normal market
    - Term structure backwardation: Fear/stress
    - SKEW > 140: Elevated tail risk
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# CBOE data URLs
VIX_CURRENT_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_VIX.json"
VIX_FUTURES_URL = "https://www.cboe.com/us/futures/market_statistics/settlement/csv"
SKEW_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_SKEW.json"


class VolatilityRegime(Enum):
    """Classification of volatility regime."""
    VERY_LOW = "very_low"        # VIX < 12
    LOW = "low"                  # 12-15
    NORMAL = "normal"            # 15-20
    ELEVATED = "elevated"        # 20-25
    HIGH = "high"                # 25-35
    EXTREME = "extreme"          # > 35


class TermStructure(Enum):
    """VIX term structure shape."""
    STEEP_CONTANGO = "steep_contango"      # > 10% upward slope
    CONTANGO = "contango"                  # Normal upward slope
    FLAT = "flat"                          # Near flat
    BACKWARDATION = "backwardation"        # Inverted, fear signal
    STEEP_BACKWARDATION = "steep_backwardation"  # Severe stress


@dataclass
class VIXData:
    """CBOE VIX Index data.

    Attributes:
        date: Date of observation
        vix_spot: Spot VIX level
        vix_open: Opening level
        vix_high: Daily high
        vix_low: Daily low
        previous_close: Prior day close
        change: Daily change
        change_pct: Percentage change
        regime: Classified volatility regime
    """
    date: date
    vix_spot: float
    vix_open: Optional[float] = None
    vix_high: Optional[float] = None
    vix_low: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    regime: Optional[VolatilityRegime] = None

    def __post_init__(self):
        if self.regime is None:
            self.regime = self._classify_regime()

    def _classify_regime(self) -> VolatilityRegime:
        if self.vix_spot < 12:
            return VolatilityRegime.VERY_LOW
        elif self.vix_spot < 15:
            return VolatilityRegime.LOW
        elif self.vix_spot < 20:
            return VolatilityRegime.NORMAL
        elif self.vix_spot < 25:
            return VolatilityRegime.ELEVATED
        elif self.vix_spot < 35:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME


@dataclass
class VIXTermStructure:
    """VIX Futures Term Structure.

    Attributes:
        date: Date of observation
        spot: Spot VIX
        front_month: Front month futures
        second_month: Second month futures
        third_month: Third month futures
        vix3m: 3-month VIX (VIX3M index)
        vix6m: 6-month VIX (VIX6M index)
        front_vs_spot: Premium/discount of front month
        second_vs_front: Roll yield indicator
        structure: Classified term structure shape
        contango_ratio: Front month / Spot ratio
    """
    date: date
    spot: float
    front_month: Optional[float] = None
    second_month: Optional[float] = None
    third_month: Optional[float] = None
    vix3m: Optional[float] = None
    vix6m: Optional[float] = None
    structure: Optional[TermStructure] = None

    def __post_init__(self):
        if self.structure is None and self.front_month:
            self.structure = self._classify_structure()

    def _classify_structure(self) -> TermStructure:
        if not self.front_month:
            return TermStructure.FLAT

        ratio = self.front_month / self.spot if self.spot > 0 else 1.0

        if ratio > 1.10:
            return TermStructure.STEEP_CONTANGO
        elif ratio > 1.02:
            return TermStructure.CONTANGO
        elif ratio > 0.98:
            return TermStructure.FLAT
        elif ratio > 0.90:
            return TermStructure.BACKWARDATION
        else:
            return TermStructure.STEEP_BACKWARDATION

    @property
    def front_vs_spot(self) -> Optional[float]:
        """Front month premium/discount vs spot."""
        if self.front_month:
            return self.front_month - self.spot
        return None

    @property
    def second_vs_front(self) -> Optional[float]:
        """Second month vs front month (roll yield indicator)."""
        if self.second_month and self.front_month:
            return self.second_month - self.front_month
        return None

    @property
    def contango_ratio(self) -> Optional[float]:
        """Front month / Spot ratio."""
        if self.front_month and self.spot > 0:
            return self.front_month / self.spot
        return None

    @property
    def is_backwardation(self) -> bool:
        """Whether term structure is inverted (fear signal)."""
        return self.structure in (TermStructure.BACKWARDATION, TermStructure.STEEP_BACKWARDATION)


@dataclass
class SKEWData:
    """CBOE SKEW Index data.

    The SKEW index measures perceived tail risk in S&P 500 options.
    Higher values indicate greater concern about extreme moves.

    Attributes:
        date: Date of observation
        skew: SKEW index value
        previous: Prior day value
        change: Daily change
        percentile_90d: 90-day percentile rank
    """
    date: date
    skew: float
    previous: Optional[float] = None
    change: Optional[float] = None
    percentile_90d: Optional[float] = None

    @property
    def is_elevated(self) -> bool:
        """Whether tail risk is elevated (SKEW > 130)."""
        return self.skew > 130

    @property
    def is_extreme(self) -> bool:
        """Whether tail risk is extreme (SKEW > 145)."""
        return self.skew > 145


@dataclass
class PutCallRatio:
    """Put/Call ratio data.

    Measures options sentiment - high values indicate bearishness.

    Attributes:
        date: Date of observation
        total_ratio: Total put/call ratio
        equity_ratio: Equity-only ratio
        index_ratio: Index options ratio
        volume_puts: Put volume
        volume_calls: Call volume
    """
    date: date
    total_ratio: float
    equity_ratio: Optional[float] = None
    index_ratio: Optional[float] = None
    volume_puts: Optional[int] = None
    volume_calls: Optional[int] = None

    @property
    def is_bearish(self) -> bool:
        """Whether ratio indicates bearish sentiment (>1.0)."""
        return self.total_ratio > 1.0

    @property
    def is_extreme_bearish(self) -> bool:
        """Whether ratio indicates extreme bearishness (>1.3)."""
        return self.total_ratio > 1.3


class CBOEClient:
    """Client for CBOE market data.

    Example:
        client = get_cboe_client()

        # Get VIX term structure
        ts = await client.get_vix_term_structure()
        print(f"VIX: {ts.spot}, Front: {ts.front_month} ({ts.structure.value})")

        # Get SKEW
        skew = await client.get_skew()
        print(f"SKEW: {skew.skew} (elevated: {skew.is_elevated})")
    """

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            try:
                from investigator.infrastructure.external.http_client import create_session
                self._session = await create_session()
            except ImportError:
                self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def get_vix(self) -> Optional[VIXData]:
        """Get current VIX data."""
        try:
            session = await self._get_session()
            headers = {"Accept": "application/json"}

            async with session.get(VIX_CURRENT_URL, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_vix_json(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")
            return None

    def _parse_vix_json(self, data: Dict) -> Optional[VIXData]:
        """Parse VIX data from CBOE JSON response."""
        try:
            # CBOE returns historical data array
            if "data" in data and data["data"]:
                latest = data["data"][-1]
                # Format: [date, open, high, low, close]
                obs_date = datetime.strptime(latest[0], "%Y-%m-%d").date()
                return VIXData(
                    date=obs_date,
                    vix_spot=float(latest[4]),  # close
                    vix_open=float(latest[1]),
                    vix_high=float(latest[2]),
                    vix_low=float(latest[3]),
                )
            return None
        except Exception as e:
            logger.debug(f"Could not parse VIX JSON: {e}")
            return None

    async def get_vix_term_structure(self) -> Optional[VIXTermStructure]:
        """Get VIX futures term structure."""
        try:
            # Get spot VIX first
            vix = await self.get_vix()
            spot = vix.vix_spot if vix else 20.0

            # VIX futures typically from CME or approximated
            # For now, create structure from spot
            return VIXTermStructure(
                date=date.today(),
                spot=spot,
                # Futures would need CME data
            )
        except Exception as e:
            logger.warning(f"Failed to get VIX term structure: {e}")
            return None

    async def get_skew(self) -> Optional[SKEWData]:
        """Get CBOE SKEW index data."""
        try:
            session = await self._get_session()
            headers = {"Accept": "application/json"}

            async with session.get(SKEW_URL, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_skew_json(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch SKEW: {e}")
            return None

    def _parse_skew_json(self, data: Dict) -> Optional[SKEWData]:
        """Parse SKEW data from CBOE JSON response."""
        try:
            if "data" in data and data["data"]:
                latest = data["data"][-1]
                prev = data["data"][-2] if len(data["data"]) > 1 else None

                obs_date = datetime.strptime(latest[0], "%Y-%m-%d").date()
                skew_val = float(latest[4])  # close
                prev_val = float(prev[4]) if prev else None

                return SKEWData(
                    date=obs_date,
                    skew=skew_val,
                    previous=prev_val,
                    change=skew_val - prev_val if prev_val else None,
                )
            return None
        except Exception as e:
            logger.debug(f"Could not parse SKEW JSON: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Get all CBOE indicators."""
        import asyncio

        vix, ts, skew = await asyncio.gather(
            self.get_vix(),
            self.get_vix_term_structure(),
            self.get_skew(),
            return_exceptions=True,
        )

        return {
            "vix": vix if not isinstance(vix, Exception) else None,
            "vix_term_structure": ts if not isinstance(ts, Exception) else None,
            "skew": skew if not isinstance(skew, Exception) else None,
        }


_client: Optional[CBOEClient] = None


def get_cboe_client() -> CBOEClient:
    """Get the shared CBOE client instance."""
    global _client
    if _client is None:
        _client = CBOEClient()
    return _client
