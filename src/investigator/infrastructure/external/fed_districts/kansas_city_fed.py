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

"""Kansas City Federal Reserve Data Client.

Key Data Series:
    - Manufacturing Survey: Tenth District manufacturing
    - Services Survey: Tenth District services
    - Labor Market Conditions Index (LMCI)
    - Financial Stress Index (KCFSI)

Data Sources:
    - https://www.kansascityfed.org/research/indicatorsdata/mfg/
    - https://www.kansascityfed.org/research/indicatorsdata/kcfsi/

Investment Signals:
    - KCFSI > 0: Financial stress above normal
    - Manufacturing composite > 0: Expansion
    - LMCI: Leading indicator for employment
"""

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Kansas City Fed data URLs
MFG_SURVEY_URL = "https://www.kansascityfed.org/~/media/files/publicat/research/indicatorsdata/mfg/mfgdata.xlsx"
KCFSI_URL = "https://www.kansascityfed.org/~/media/files/publicat/research/indicatorsdata/kcfsi/kcfsi.xlsx"
LMCI_URL = "https://www.kansascityfed.org/~/media/files/publicat/research/indicatorsdata/lmci/lmci.xlsx"


class FinancialStressLevel(Enum):
    """Classification of financial stress."""
    VERY_LOW = "very_low"      # < -1
    LOW = "low"                # -1 to -0.5
    BELOW_NORMAL = "below_normal"  # -0.5 to 0
    NORMAL = "normal"          # 0 to 0.5
    ELEVATED = "elevated"      # 0.5 to 1
    HIGH = "high"              # 1 to 2
    SEVERE = "severe"          # > 2


@dataclass
class KCManufacturing:
    """Kansas City Fed Manufacturing Survey.

    Monthly survey of Tenth District manufacturers covering CO, KS, NE,
    OK, WY, and parts of MO and NM.

    Attributes:
        date: Month of the survey
        composite_index: Overall composite index
        production: Production index
        shipments: Shipments index
        new_orders: New orders index
        employment: Employment index
        workweek: Average workweek index
        prices_paid: Input prices index
        prices_received: Output prices index
        future_composite: 6-month expectations composite
        future_production: 6-month production expectations
    """
    date: date
    composite_index: float
    production: Optional[float] = None
    shipments: Optional[float] = None
    new_orders: Optional[float] = None
    employment: Optional[float] = None
    workweek: Optional[float] = None
    prices_paid: Optional[float] = None
    prices_received: Optional[float] = None
    future_composite: Optional[float] = None
    future_production: Optional[float] = None

    @property
    def is_expanding(self) -> bool:
        return self.composite_index > 0


@dataclass
class KCFinancialStressIndex:
    """Kansas City Fed Financial Stress Index (KCFSI).

    Weekly index of financial market stress based on 11 indicators
    including yield spreads, volatility, and market correlations.

    Attributes:
        date: Week of the observation
        kcfsi: Financial stress index value
        previous_week: Prior week value
        one_month_change: Change over past month
        stress_level: Classified stress level
    """
    date: date
    kcfsi: float
    previous_week: Optional[float] = None
    one_month_change: Optional[float] = None
    stress_level: Optional[FinancialStressLevel] = None

    def __post_init__(self):
        if self.stress_level is None:
            if self.kcfsi < -1:
                self.stress_level = FinancialStressLevel.VERY_LOW
            elif self.kcfsi < -0.5:
                self.stress_level = FinancialStressLevel.LOW
            elif self.kcfsi < 0:
                self.stress_level = FinancialStressLevel.BELOW_NORMAL
            elif self.kcfsi < 0.5:
                self.stress_level = FinancialStressLevel.NORMAL
            elif self.kcfsi < 1:
                self.stress_level = FinancialStressLevel.ELEVATED
            elif self.kcfsi < 2:
                self.stress_level = FinancialStressLevel.HIGH
            else:
                self.stress_level = FinancialStressLevel.SEVERE

    @property
    def is_stressed(self) -> bool:
        return self.kcfsi > 0.5


@dataclass
class LaborMarketConditions:
    """Kansas City Fed Labor Market Conditions Index.

    Summarizes information from 24 labor market indicators.

    Attributes:
        date: Month of the observation
        lmci_level: Level of activity indicator
        lmci_momentum: Momentum indicator (rate of change)
    """
    date: date
    lmci_level: float
    lmci_momentum: Optional[float] = None

    @property
    def is_strong(self) -> bool:
        return self.lmci_level > 0

    @property
    def is_improving(self) -> bool:
        return self.lmci_momentum is not None and self.lmci_momentum > 0


class KansasCityFedClient:
    """Client for Kansas City Federal Reserve economic data."""

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

    async def get_manufacturing_survey(self) -> Optional[KCManufacturing]:
        """Get the latest Manufacturing Survey."""
        try:
            session = await self._get_session()
            async with session.get(MFG_SURVEY_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_mfg_excel(content)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch KC manufacturing: {e}")
            return None

    def _parse_mfg_excel(self, content: bytes) -> Optional[KCManufacturing]:
        try:
            import io
            import pandas as pd
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
            if df.empty:
                return None
            latest = df.iloc[-1]
            obs_date = pd.to_datetime(latest[df.columns[0]]).date()
            return KCManufacturing(
                date=obs_date,
                composite_index=float(latest[df.columns[1]]),
            )
        except Exception as e:
            logger.debug(f"Could not parse KC mfg Excel: {e}")
            return None

    async def get_financial_stress_index(self) -> Optional[KCFinancialStressIndex]:
        """Get the latest KCFSI."""
        try:
            session = await self._get_session()
            async with session.get(KCFSI_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_kcfsi_excel(content)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch KCFSI: {e}")
            return None

    def _parse_kcfsi_excel(self, content: bytes) -> Optional[KCFinancialStressIndex]:
        try:
            import io
            import pandas as pd
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
            if df.empty:
                return None
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            obs_date = pd.to_datetime(latest[df.columns[0]]).date()
            return KCFinancialStressIndex(
                date=obs_date,
                kcfsi=float(latest[df.columns[1]]),
                previous_week=float(prev[df.columns[1]]) if prev is not None else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse KCFSI Excel: {e}")
            return None

    async def get_labor_market_conditions(self) -> Optional[LaborMarketConditions]:
        """Get the latest LMCI."""
        try:
            session = await self._get_session()
            async with session.get(LMCI_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_lmci_excel(content)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch LMCI: {e}")
            return None

    def _parse_lmci_excel(self, content: bytes) -> Optional[LaborMarketConditions]:
        try:
            import io
            import pandas as pd
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
            if df.empty:
                return None
            latest = df.iloc[-1]
            obs_date = pd.to_datetime(latest[df.columns[0]]).date()
            return LaborMarketConditions(
                date=obs_date,
                lmci_level=float(latest[df.columns[1]]),
                lmci_momentum=float(latest[df.columns[2]]) if len(df.columns) > 2 else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse LMCI Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        import asyncio
        mfg, kcfsi, lmci = await asyncio.gather(
            self.get_manufacturing_survey(),
            self.get_financial_stress_index(),
            self.get_labor_market_conditions(),
            return_exceptions=True,
        )
        return {
            "manufacturing_survey": mfg if not isinstance(mfg, Exception) else None,
            "financial_stress_index": kcfsi if not isinstance(kcfsi, Exception) else None,
            "labor_market_conditions": lmci if not isinstance(lmci, Exception) else None,
        }


_client: Optional[KansasCityFedClient] = None


def get_kc_fed_client() -> KansasCityFedClient:
    global _client
    if _client is None:
        _client = KansasCityFedClient()
    return _client
