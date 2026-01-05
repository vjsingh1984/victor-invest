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

"""Chicago Federal Reserve Data Client.

Key Data Series:
    - CFNAI: Chicago Fed National Activity Index (85 indicators)
    - CFNAI-MA3: 3-month moving average (recession signal)
    - NFCI: National Financial Conditions Index
    - ANFCI: Adjusted NFCI (removing economic conditions)
    - Midwest Economy Index

Data Sources:
    - https://www.chicagofed.org/research/data/cfnai/current-data
    - https://www.chicagofed.org/research/data/nfci/current-data

Investment Signals:
    - CFNAI-MA3 < -0.7: Recession signal (70% probability)
    - CFNAI-MA3 > 0.7: End of recession signal
    - NFCI > 0: Tighter than average financial conditions
    - ANFCI: Financial stress independent of economy
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Chicago Fed data URLs
CFNAI_URL = "https://www.chicagofed.org/~/media/publications/cfnai/cfnai-data-series-xlsx.xlsx"
CFNAI_HISTORICAL_URL = "https://www.chicagofed.org/~/media/publications/cfnai/cfnai-history-xlsx.xlsx"
NFCI_URL = "https://www.chicagofed.org/~/media/publications/nfci/nfci-data-series-xlsx.xlsx"


class EconomicCondition(Enum):
    """Classification of economic conditions from CFNAI."""

    RECESSION = "recession"  # MA3 < -0.7
    CONTRACTION_RISK = "contraction_risk"  # MA3 < -0.35
    BELOW_TREND = "below_trend"  # MA3 < 0
    TREND_GROWTH = "trend_growth"  # MA3 0 to 0.35
    ABOVE_TREND = "above_trend"  # MA3 0.35 to 0.7
    RECOVERY_END = "recovery_end"  # MA3 > 0.7


class FinancialCondition(Enum):
    """Classification of financial conditions from NFCI."""

    VERY_LOOSE = "very_loose"  # < -0.5
    LOOSE = "loose"  # -0.5 to 0
    NEUTRAL = "neutral"  # 0 to 0.5
    TIGHT = "tight"  # 0.5 to 1.0
    VERY_TIGHT = "very_tight"  # > 1.0


@dataclass
class CFNAIData:
    """Chicago Fed National Activity Index.

    The CFNAI is a weighted average of 85 existing monthly indicators
    of national economic activity. It's designed to have an average
    value of zero and a standard deviation of one.

    Categories:
    1. Production and Income (23 indicators)
    2. Employment, Unemployment, and Hours (24 indicators)
    3. Personal Consumption and Housing (15 indicators)
    4. Sales, Orders, and Inventories (23 indicators)

    Attributes:
        date: Month of the observation
        cfnai: Chicago Fed National Activity Index
        cfnai_ma3: 3-month moving average (primary signal)
        production_income: Production/income contribution
        employment: Employment contribution
        consumption_housing: Consumption/housing contribution
        sales_orders_inventories: Sales/orders contribution
        condition: Classified economic condition
        recession_probability: Implied recession probability
    """

    date: date
    cfnai: float
    cfnai_ma3: float
    production_income: Optional[float] = None
    employment: Optional[float] = None
    consumption_housing: Optional[float] = None
    sales_orders_inventories: Optional[float] = None
    condition: Optional[EconomicCondition] = None
    recession_probability: Optional[float] = None

    def __post_init__(self):
        if self.condition is None:
            self.condition = self._classify_condition()
        if self.recession_probability is None:
            self.recession_probability = self._calc_recession_prob()

    def _classify_condition(self) -> EconomicCondition:
        if self.cfnai_ma3 < -0.7:
            return EconomicCondition.RECESSION
        elif self.cfnai_ma3 < -0.35:
            return EconomicCondition.CONTRACTION_RISK
        elif self.cfnai_ma3 < 0:
            return EconomicCondition.BELOW_TREND
        elif self.cfnai_ma3 < 0.35:
            return EconomicCondition.TREND_GROWTH
        elif self.cfnai_ma3 < 0.7:
            return EconomicCondition.ABOVE_TREND
        else:
            return EconomicCondition.RECOVERY_END

    def _calc_recession_prob(self) -> float:
        """Estimate recession probability from CFNAI-MA3.

        Based on historical relationship:
        - MA3 = -0.7: ~70% probability
        - MA3 = -1.0: ~90% probability
        - MA3 = 0: ~10% probability
        """
        # Logistic approximation
        import math

        prob = 1 / (1 + math.exp(-((-self.cfnai_ma3 - 0.3) * 3)))
        return round(prob * 100, 1)

    @property
    def is_recession_signal(self) -> bool:
        """Whether MA3 is below recession threshold."""
        return self.cfnai_ma3 < -0.7

    @property
    def is_recovery_signal(self) -> bool:
        """Whether MA3 suggests recovery ending."""
        return self.cfnai_ma3 > 0.7


@dataclass
class NFCIData:
    """National Financial Conditions Index.

    The NFCI measures financial conditions (risk, credit, leverage)
    using over 100 indicators. Zero represents average conditions.

    Attributes:
        date: Week of the observation
        nfci: National Financial Conditions Index
        anfci: Adjusted NFCI (removing economic activity)
        risk_subindex: Risk subindex
        credit_subindex: Credit subindex
        leverage_subindex: Leverage subindex
        condition: Classified financial condition
    """

    date: date
    nfci: float
    anfci: Optional[float] = None
    risk_subindex: Optional[float] = None
    credit_subindex: Optional[float] = None
    leverage_subindex: Optional[float] = None
    condition: Optional[FinancialCondition] = None

    def __post_init__(self):
        if self.condition is None:
            self.condition = self._classify_condition()

    def _classify_condition(self) -> FinancialCondition:
        if self.nfci < -0.5:
            return FinancialCondition.VERY_LOOSE
        elif self.nfci < 0:
            return FinancialCondition.LOOSE
        elif self.nfci < 0.5:
            return FinancialCondition.NEUTRAL
        elif self.nfci < 1.0:
            return FinancialCondition.TIGHT
        else:
            return FinancialCondition.VERY_TIGHT

    @property
    def is_stress_signal(self) -> bool:
        """Whether NFCI indicates financial stress."""
        return self.nfci > 0.5


class ChicagoFedClient:
    """Client for Chicago Federal Reserve economic data.

    Example:
        client = get_chicago_fed_client()

        # Get CFNAI
        cfnai = await client.get_cfnai()
        print(f"CFNAI-MA3: {cfnai.cfnai_ma3} ({cfnai.condition.value})")
        print(f"Recession probability: {cfnai.recession_probability}%")

        # Get NFCI
        nfci = await client.get_nfci()
        print(f"NFCI: {nfci.nfci} ({nfci.condition.value})")
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

    async def get_cfnai(self) -> Optional[CFNAIData]:
        """Get the latest Chicago Fed National Activity Index.

        Returns:
            CFNAIData with current values, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(CFNAI_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_cfnai_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch CFNAI: {e}")
            return None

    def _parse_cfnai_excel(self, content: bytes) -> Optional[CFNAIData]:
        """Parse CFNAI data from Excel file."""
        try:
            import io

            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            # Find columns
            def find_col(keywords: List[str]) -> Optional[str]:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if all(k in col_lower for k in keywords):
                        return col
                return None

            cfnai_col = find_col(["cfnai"]) or df.columns[1]
            ma3_col = find_col(["ma3"]) or find_col(["moving"])
            prod_col = find_col(["production"])
            emp_col = find_col(["employ"])
            cons_col = find_col(["consumption"]) or find_col(["housing"])
            sales_col = find_col(["sales"]) or find_col(["orders"])

            cfnai = float(latest[cfnai_col])
            cfnai_ma3 = float(latest[ma3_col]) if ma3_col else cfnai  # Fallback to current

            return CFNAIData(
                date=obs_date,
                cfnai=cfnai,
                cfnai_ma3=cfnai_ma3,
                production_income=float(latest[prod_col]) if prod_col else None,
                employment=float(latest[emp_col]) if emp_col else None,
                consumption_housing=float(latest[cons_col]) if cons_col else None,
                sales_orders_inventories=float(latest[sales_col]) if sales_col else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse CFNAI Excel: {e}")
            return None

    async def get_nfci(self) -> Optional[NFCIData]:
        """Get the latest National Financial Conditions Index.

        Returns:
            NFCIData with current values, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(NFCI_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_nfci_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch NFCI: {e}")
            return None

    def _parse_nfci_excel(self, content: bytes) -> Optional[NFCIData]:
        """Parse NFCI data from Excel file."""
        try:
            import io

            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            # Find columns
            def find_col(keywords: List[str]) -> Optional[str]:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if all(k in col_lower for k in keywords):
                        return col
                return None

            nfci_col = find_col(["nfci"]) or df.columns[1]
            anfci_col = find_col(["anfci"]) or find_col(["adjusted"])
            risk_col = find_col(["risk"])
            credit_col = find_col(["credit"])
            leverage_col = find_col(["leverage"])

            return NFCIData(
                date=obs_date,
                nfci=float(latest[nfci_col]),
                anfci=float(latest[anfci_col]) if anfci_col else None,
                risk_subindex=float(latest[risk_col]) if risk_col else None,
                credit_subindex=float(latest[credit_col]) if credit_col else None,
                leverage_subindex=float(latest[leverage_col]) if leverage_col else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse NFCI Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Get all Chicago Fed indicators.

        Returns:
            Dictionary with all available indicators
        """
        import asyncio

        cfnai, nfci = await asyncio.gather(
            self.get_cfnai(),
            self.get_nfci(),
            return_exceptions=True,
        )

        return {
            "cfnai": cfnai if not isinstance(cfnai, Exception) else None,
            "nfci": nfci if not isinstance(nfci, Exception) else None,
        }


# Singleton instance
_client: Optional[ChicagoFedClient] = None


def get_chicago_fed_client() -> ChicagoFedClient:
    """Get the shared Chicago Fed client instance."""
    global _client
    if _client is None:
        _client = ChicagoFedClient()
    return _client
