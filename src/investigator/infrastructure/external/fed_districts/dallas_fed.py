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

"""Dallas Federal Reserve Data Client.

Key Data Series:
    - Texas Manufacturing Outlook Survey: Monthly manufacturing activity
    - Texas Service Sector Outlook Survey: Monthly services activity
    - Texas Retail Outlook Survey: Monthly retail activity
    - Texas Leading Index: State leading indicator
    - Trimmed Mean PCE: Alternative inflation measure

Data Sources:
    - https://www.dallasfed.org/research/surveys/tmos
    - https://www.dallasfed.org/research/pce

Investment Signals:
    - Texas surveys often lead national data (large, diverse economy)
    - Trimmed Mean PCE: Fed's preferred inflation measure foundation
    - Energy sector sensitivity: Oil price correlation
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Dallas Fed data URLs
TMOS_URL = "https://www.dallasfed.org/-/media/Documents/research/surveys/tmos/tmos.xlsx"
TSSOS_URL = "https://www.dallasfed.org/-/media/Documents/research/surveys/tssos/tssos.xlsx"
TRIMMED_PCE_URL = "https://www.dallasfed.org/-/media/Documents/research/pce/pce_data.xlsx"


class RegionalOutlook(Enum):
    """Classification of regional activity."""
    STRONG_CONTRACTION = "strong_contraction"  # < -15
    CONTRACTION = "contraction"                # -15 to 0
    WEAK_EXPANSION = "weak_expansion"          # 0 to 10
    MODERATE_EXPANSION = "moderate_expansion"  # 10 to 20
    STRONG_EXPANSION = "strong_expansion"      # > 20


@dataclass
class TexasManufacturing:
    """Dallas Fed Texas Manufacturing Outlook Survey.

    Monthly survey of Texas manufacturers on business conditions.
    Important as Texas is a leading indicator for national manufacturing.

    Attributes:
        date: Month of the survey
        production_index: Production activity index
        new_orders_index: New orders index
        capacity_utilization: Capacity utilization index
        shipments_index: Shipments index
        employment_index: Employment index
        hours_worked: Hours worked index
        wages_benefits: Wages and benefits index
        raw_materials_prices: Input price index
        finished_goods_prices: Output price index
        general_business_activity: Overall outlook index
        company_outlook: Company-specific outlook
        future_production: 6-month production expectations
        future_employment: 6-month employment expectations
        outlook: Classified activity level
    """
    date: date
    production_index: float
    new_orders_index: Optional[float] = None
    capacity_utilization: Optional[float] = None
    shipments_index: Optional[float] = None
    employment_index: Optional[float] = None
    hours_worked: Optional[float] = None
    wages_benefits: Optional[float] = None
    raw_materials_prices: Optional[float] = None
    finished_goods_prices: Optional[float] = None
    general_business_activity: Optional[float] = None
    company_outlook: Optional[float] = None
    future_production: Optional[float] = None
    future_employment: Optional[float] = None
    outlook: Optional[RegionalOutlook] = None

    def __post_init__(self):
        if self.outlook is None:
            self.outlook = self._classify_outlook()

    def _classify_outlook(self) -> RegionalOutlook:
        idx = self.general_business_activity or self.production_index
        if idx < -15:
            return RegionalOutlook.STRONG_CONTRACTION
        elif idx < 0:
            return RegionalOutlook.CONTRACTION
        elif idx < 10:
            return RegionalOutlook.WEAK_EXPANSION
        elif idx < 20:
            return RegionalOutlook.MODERATE_EXPANSION
        else:
            return RegionalOutlook.STRONG_EXPANSION

    @property
    def is_expanding(self) -> bool:
        """Whether manufacturing is expanding."""
        idx = self.general_business_activity or self.production_index
        return idx > 0


@dataclass
class TexasServices:
    """Dallas Fed Texas Service Sector Outlook Survey.

    Monthly survey of Texas service companies on business conditions.

    Attributes:
        date: Month of the survey
        revenue_index: Revenue index
        employment_index: Employment index
        hours_worked: Hours worked index
        wages_benefits: Wages and benefits index
        input_prices: Input price index
        selling_prices: Selling price index
        general_business_activity: Overall outlook index
        company_outlook: Company-specific outlook
        future_revenue: 6-month revenue expectations
        outlook: Classified activity level
    """
    date: date
    revenue_index: float
    employment_index: Optional[float] = None
    hours_worked: Optional[float] = None
    wages_benefits: Optional[float] = None
    input_prices: Optional[float] = None
    selling_prices: Optional[float] = None
    general_business_activity: Optional[float] = None
    company_outlook: Optional[float] = None
    future_revenue: Optional[float] = None
    outlook: Optional[RegionalOutlook] = None

    def __post_init__(self):
        if self.outlook is None:
            idx = self.general_business_activity or self.revenue_index
            if idx < -15:
                self.outlook = RegionalOutlook.STRONG_CONTRACTION
            elif idx < 0:
                self.outlook = RegionalOutlook.CONTRACTION
            elif idx < 10:
                self.outlook = RegionalOutlook.WEAK_EXPANSION
            elif idx < 20:
                self.outlook = RegionalOutlook.MODERATE_EXPANSION
            else:
                self.outlook = RegionalOutlook.STRONG_EXPANSION


@dataclass
class TrimmedMeanPCE:
    """Dallas Fed Trimmed Mean PCE Inflation.

    Alternative core inflation measure that trims extreme price changes.
    Often used as input for Fed policy decisions.

    Attributes:
        date: Month of the observation
        one_month_annualized: 1-month change (annualized)
        six_month_annualized: 6-month change (annualized)
        twelve_month: 12-month change
    """
    date: date
    one_month_annualized: float
    six_month_annualized: Optional[float] = None
    twelve_month: Optional[float] = None


class DallasFedClient:
    """Client for Dallas Federal Reserve economic data.

    Example:
        client = get_dallas_fed_client()

        # Get Texas manufacturing survey
        mfg = await client.get_texas_manufacturing()
        print(f"Production: {mfg.production_index} ({mfg.outlook.value})")

        # Get trimmed mean PCE
        pce = await client.get_trimmed_mean_pce()
        print(f"12M Trimmed PCE: {pce.twelve_month}%")
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

    async def get_texas_manufacturing(self) -> Optional[TexasManufacturing]:
        """Get the latest Texas Manufacturing Outlook Survey.

        Returns:
            TexasManufacturing data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(TMOS_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_tmos_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch TMOS: {e}")
            return None

    def _parse_tmos_excel(self, content: bytes) -> Optional[TexasManufacturing]:
        """Parse Texas Manufacturing data from Excel."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            def find_col(pattern: str) -> Optional[str]:
                for col in df.columns:
                    if pattern.lower() in str(col).lower():
                        return col
                return None

            prod_col = find_col('production') or df.columns[1]
            orders_col = find_col('new order')
            emp_col = find_col('employment')
            gba_col = find_col('general business')
            future_col = find_col('future') and find_col('production')

            return TexasManufacturing(
                date=obs_date,
                production_index=float(latest[prod_col]),
                new_orders_index=float(latest[orders_col]) if orders_col else None,
                employment_index=float(latest[emp_col]) if emp_col else None,
                general_business_activity=float(latest[gba_col]) if gba_col else None,
                future_production=float(latest[future_col]) if future_col else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse TMOS Excel: {e}")
            return None

    async def get_texas_services(self) -> Optional[TexasServices]:
        """Get the latest Texas Service Sector Outlook Survey.

        Returns:
            TexasServices data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(TSSOS_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_tssos_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch TSSOS: {e}")
            return None

    def _parse_tssos_excel(self, content: bytes) -> Optional[TexasServices]:
        """Parse Texas Services data from Excel."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            def find_col(pattern: str) -> Optional[str]:
                for col in df.columns:
                    if pattern.lower() in str(col).lower():
                        return col
                return None

            rev_col = find_col('revenue') or df.columns[1]
            emp_col = find_col('employment')
            gba_col = find_col('general business')

            return TexasServices(
                date=obs_date,
                revenue_index=float(latest[rev_col]),
                employment_index=float(latest[emp_col]) if emp_col else None,
                general_business_activity=float(latest[gba_col]) if gba_col else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse TSSOS Excel: {e}")
            return None

    async def get_trimmed_mean_pce(self) -> Optional[TrimmedMeanPCE]:
        """Get the latest Trimmed Mean PCE inflation.

        Returns:
            TrimmedMeanPCE data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(TRIMMED_PCE_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_pce_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch Trimmed Mean PCE: {e}")
            return None

    def _parse_pce_excel(self, content: bytes) -> Optional[TrimmedMeanPCE]:
        """Parse Trimmed Mean PCE from Excel."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            def find_col(pattern: str) -> Optional[str]:
                for col in df.columns:
                    if pattern.lower() in str(col).lower():
                        return col
                return None

            one_m = find_col('1-month') or find_col('1 month')
            six_m = find_col('6-month') or find_col('6 month')
            twelve_m = find_col('12-month') or find_col('12 month') or find_col('year')

            return TrimmedMeanPCE(
                date=obs_date,
                one_month_annualized=float(latest[one_m or df.columns[1]]),
                six_month_annualized=float(latest[six_m]) if six_m else None,
                twelve_month=float(latest[twelve_m]) if twelve_m else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse PCE Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Get all Dallas Fed indicators.

        Returns:
            Dictionary with all available indicators
        """
        import asyncio

        mfg, svc, pce = await asyncio.gather(
            self.get_texas_manufacturing(),
            self.get_texas_services(),
            self.get_trimmed_mean_pce(),
            return_exceptions=True,
        )

        return {
            "texas_manufacturing": mfg if not isinstance(mfg, Exception) else None,
            "texas_services": svc if not isinstance(svc, Exception) else None,
            "trimmed_mean_pce": pce if not isinstance(pce, Exception) else None,
        }


# Singleton instance
_client: Optional[DallasFedClient] = None


def get_dallas_fed_client() -> DallasFedClient:
    """Get the shared Dallas Fed client instance."""
    global _client
    if _client is None:
        _client = DallasFedClient()
    return _client
