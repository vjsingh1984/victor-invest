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

"""Richmond Federal Reserve Data Client.

Key Data Series:
    - Fifth District Manufacturing Survey: Regional manufacturing
    - Fifth District Services Survey: Regional services
    - CFO Survey: Business outlook from CFOs

Data Sources:
    - https://www.richmondfed.org/research/regional_economy/surveys_of_business_conditions

Investment Signals:
    - Composite index > 0: Expansion
    - Employment component: Labor market tightness
    - Price components: Regional inflation pressure
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Richmond Fed data URLs
MFG_SURVEY_URL = "https://www.richmondfed.org/-/media/RichmondFedOrg/research/regional_economy/surveys_of_business_conditions/manufacturing/xls/manufacturing_all.xlsx"
SERVICES_SURVEY_URL = "https://www.richmondfed.org/-/media/RichmondFedOrg/research/regional_economy/surveys_of_business_conditions/services/xls/services_all.xlsx"


@dataclass
class FifthDistrictSurvey:
    """Richmond Fed Fifth District Survey.

    Monthly survey covering DC, MD, NC, SC, VA, and most of WV.

    Attributes:
        date: Month of the survey
        composite_index: Overall composite index
        shipments: Shipments index
        new_orders: New orders index
        employment: Employment index
        wages: Wages index
        prices_paid: Input prices index
        prices_received: Output prices index
        capacity_utilization: Capacity utilization index
        backlog: Backlog of orders index
        vendor_lead_time: Vendor lead time index
        inventories: Inventory levels index
        future_shipments: 6-month shipments expectations
        future_new_orders: 6-month orders expectations
        future_employment: 6-month employment expectations
        survey_type: 'manufacturing' or 'services'
    """

    date: date
    composite_index: float
    shipments: Optional[float] = None
    new_orders: Optional[float] = None
    employment: Optional[float] = None
    wages: Optional[float] = None
    prices_paid: Optional[float] = None
    prices_received: Optional[float] = None
    capacity_utilization: Optional[float] = None
    backlog: Optional[float] = None
    vendor_lead_time: Optional[float] = None
    inventories: Optional[float] = None
    future_shipments: Optional[float] = None
    future_new_orders: Optional[float] = None
    future_employment: Optional[float] = None
    survey_type: str = "manufacturing"

    @property
    def is_expanding(self) -> bool:
        return self.composite_index > 0

    @property
    def has_price_pressure(self) -> bool:
        if self.prices_paid:
            return self.prices_paid > 25
        return False


class RichmondFedClient:
    """Client for Richmond Federal Reserve economic data."""

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

    async def get_manufacturing_survey(self) -> Optional[FifthDistrictSurvey]:
        """Get the latest Fifth District Manufacturing Survey."""
        try:
            session = await self._get_session()
            async with session.get(MFG_SURVEY_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_survey_excel(content, "manufacturing")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch Richmond mfg survey: {e}")
            return None

    async def get_services_survey(self) -> Optional[FifthDistrictSurvey]:
        """Get the latest Fifth District Services Survey."""
        try:
            session = await self._get_session()
            async with session.get(SERVICES_SURVEY_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_survey_excel(content, "services")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch Richmond services survey: {e}")
            return None

    def _parse_survey_excel(self, content: bytes, survey_type: str) -> Optional[FifthDistrictSurvey]:
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

            comp_col = find_col("composite") or find_col("index") or df.columns[1]
            ship_col = find_col("shipment")
            orders_col = find_col("new order")
            emp_col = find_col("employment")
            wages_col = find_col("wage")
            pp_col = find_col("prices paid")
            pr_col = find_col("prices received")

            return FifthDistrictSurvey(
                date=obs_date,
                composite_index=float(latest[comp_col]),
                shipments=float(latest[ship_col]) if ship_col else None,
                new_orders=float(latest[orders_col]) if orders_col else None,
                employment=float(latest[emp_col]) if emp_col else None,
                wages=float(latest[wages_col]) if wages_col else None,
                prices_paid=float(latest[pp_col]) if pp_col else None,
                prices_received=float(latest[pr_col]) if pr_col else None,
                survey_type=survey_type,
            )
        except Exception as e:
            logger.debug(f"Could not parse Richmond survey Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        import asyncio

        mfg, svc = await asyncio.gather(
            self.get_manufacturing_survey(),
            self.get_services_survey(),
            return_exceptions=True,
        )
        return {
            "manufacturing_survey": mfg if not isinstance(mfg, Exception) else None,
            "services_survey": svc if not isinstance(svc, Exception) else None,
        }


_client: Optional[RichmondFedClient] = None


def get_richmond_fed_client() -> RichmondFedClient:
    global _client
    if _client is None:
        _client = RichmondFedClient()
    return _client
