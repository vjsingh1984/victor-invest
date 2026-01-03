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

"""Philadelphia Federal Reserve Data Client.

Key Data Series:
    - Manufacturing Business Outlook Survey (BOS): Third District manufacturing
    - State Leading Indexes: 50-state leading indicator
    - State Coincident Indexes: Current economic activity
    - Nonmanufacturing Business Outlook Survey
    - ADS Business Conditions Index: Daily real-time tracking

Data Sources:
    - https://www.philadelphiafed.org/surveys-and-data
    - https://www.philadelphiafed.org/surveys-and-data/regional-economic-analysis/state-leading-indexes

Investment Signals:
    - Manufacturing BOS diffusion index > 0: Expansion
    - Leading index: 6-month forward outlook
    - New orders component: Demand signal
    - Prices paid: Inflation pressure
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Philadelphia Fed data URLs
MANUFACTURING_URL = "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/mbos/historical-data/bos_historical_data.xlsx"
LEADING_INDEX_URL = "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/sli/sli_historical_data.xlsx"
COINCIDENT_INDEX_URL = "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/sci/sci_historical_data.xlsx"
ADS_URL = "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/ads/ads_index.xlsx"


class ManufacturingOutlook(Enum):
    """Classification of manufacturing activity."""
    STRONG_CONTRACTION = "strong_contraction"  # < -20
    CONTRACTION = "contraction"                # -20 to 0
    WEAK_EXPANSION = "weak_expansion"          # 0 to 10
    MODERATE_EXPANSION = "moderate_expansion"  # 10 to 25
    STRONG_EXPANSION = "strong_expansion"      # > 25


@dataclass
class ManufacturingSurvey:
    """Philadelphia Fed Manufacturing Business Outlook Survey.

    The diffusion index measures the breadth of changes in manufacturing
    activity. Positive values indicate expansion, negative indicate contraction.

    Attributes:
        date: Month of the survey
        diffusion_index: Overall diffusion index (headline)
        new_orders: New orders diffusion index
        shipments: Shipments diffusion index
        unfilled_orders: Unfilled orders diffusion index
        employment: Employment diffusion index
        avg_employee_workweek: Average workweek diffusion
        prices_paid: Prices paid diffusion (input costs)
        prices_received: Prices received diffusion (output prices)
        future_activity: 6-month ahead expectations
        future_employment: Expected employment changes
        future_capex: Expected capital expenditures
        outlook: Classified activity level
    """
    date: date
    diffusion_index: float
    new_orders: Optional[float] = None
    shipments: Optional[float] = None
    unfilled_orders: Optional[float] = None
    employment: Optional[float] = None
    avg_employee_workweek: Optional[float] = None
    prices_paid: Optional[float] = None
    prices_received: Optional[float] = None
    future_activity: Optional[float] = None
    future_employment: Optional[float] = None
    future_capex: Optional[float] = None
    outlook: Optional[ManufacturingOutlook] = None

    def __post_init__(self):
        if self.outlook is None:
            self.outlook = self._classify_outlook()

    def _classify_outlook(self) -> ManufacturingOutlook:
        if self.diffusion_index < -20:
            return ManufacturingOutlook.STRONG_CONTRACTION
        elif self.diffusion_index < 0:
            return ManufacturingOutlook.CONTRACTION
        elif self.diffusion_index < 10:
            return ManufacturingOutlook.WEAK_EXPANSION
        elif self.diffusion_index < 25:
            return ManufacturingOutlook.MODERATE_EXPANSION
        else:
            return ManufacturingOutlook.STRONG_EXPANSION

    @property
    def is_expanding(self) -> bool:
        """Whether manufacturing is expanding."""
        return self.diffusion_index > 0

    @property
    def has_inflation_pressure(self) -> bool:
        """Whether prices are accelerating."""
        if self.prices_paid is not None:
            return self.prices_paid > 30
        return False


@dataclass
class LeadingIndex:
    """Philadelphia Fed State Leading Index.

    Predicts 6-month growth in the state coincident index.
    Useful for regional economic outlook and national aggregation.

    Attributes:
        date: Month of the observation
        state: State abbreviation
        leading_index: Leading index value
        previous_value: Prior month value
        six_month_change: Change over 6 months
        interpretation: Text interpretation
    """
    date: date
    state: str
    leading_index: float
    previous_value: Optional[float] = None
    six_month_change: Optional[float] = None

    @property
    def interpretation(self) -> str:
        if self.leading_index > 1:
            return "Strong growth expected"
        elif self.leading_index > 0:
            return "Moderate growth expected"
        elif self.leading_index > -1:
            return "Weak growth expected"
        else:
            return "Contraction likely"


@dataclass
class CoincidentIndex:
    """Philadelphia Fed State Coincident Index.

    Measures current economic conditions using employment,
    unemployment, hours worked, and wages.

    Attributes:
        date: Month of the observation
        state: State abbreviation
        coincident_index: Coincident index value
        one_month_change: Month-over-month change
        three_month_change: 3-month change (annualized)
        twelve_month_change: Year-over-year change
    """
    date: date
    state: str
    coincident_index: float
    one_month_change: Optional[float] = None
    three_month_change: Optional[float] = None
    twelve_month_change: Optional[float] = None


@dataclass
class ADSIndex:
    """Aruoba-Diebold-Scotti Business Conditions Index.

    Daily index of real business conditions based on:
    - Weekly initial jobless claims
    - Monthly payroll employment
    - Industrial production
    - Personal income less transfer payments
    - Manufacturing and trade sales
    - Quarterly real GDP

    Attributes:
        date: Date of the observation
        ads_index: ADS index value
        interpretation: Text interpretation
    """
    date: date
    ads_index: float

    @property
    def interpretation(self) -> str:
        if self.ads_index > 0.5:
            return "Above average growth"
        elif self.ads_index > 0:
            return "Average growth"
        elif self.ads_index > -0.5:
            return "Below average growth"
        elif self.ads_index > -2:
            return "Weak conditions"
        else:
            return "Recessionary conditions"


class PhiladelphiaFedClient:
    """Client for Philadelphia Federal Reserve economic data.

    Example:
        client = get_philly_fed_client()

        # Get manufacturing survey
        mfg = await client.get_manufacturing_survey()
        print(f"Diffusion index: {mfg.diffusion_index} ({mfg.outlook.value})")

        # Get national leading index
        leading = await client.get_leading_index("US")
        print(f"US Leading: {leading.leading_index}")
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

    async def get_manufacturing_survey(self) -> Optional[ManufacturingSurvey]:
        """Get the latest Manufacturing Business Outlook Survey.

        Returns:
            ManufacturingSurvey with current data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(MANUFACTURING_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_manufacturing_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch manufacturing survey: {e}")
            return None

    def _parse_manufacturing_excel(self, content: bytes) -> Optional[ManufacturingSurvey]:
        """Parse manufacturing survey from Excel file."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            # Find key columns
            def find_col(keywords: List[str]) -> Optional[str]:
                for col in df.columns:
                    col_lower = col.lower() if isinstance(col, str) else str(col).lower()
                    if all(k in col_lower for k in keywords):
                        return col
                return None

            diffusion_col = find_col(['diffusion']) or find_col(['index']) or df.columns[1]
            orders_col = find_col(['new', 'order'])
            ship_col = find_col(['shipment'])
            emp_col = find_col(['employment'])
            prices_paid_col = find_col(['prices', 'paid'])
            future_col = find_col(['future', 'activity']) or find_col(['6', 'month'])

            return ManufacturingSurvey(
                date=obs_date,
                diffusion_index=float(latest[diffusion_col]),
                new_orders=float(latest[orders_col]) if orders_col else None,
                shipments=float(latest[ship_col]) if ship_col else None,
                employment=float(latest[emp_col]) if emp_col else None,
                prices_paid=float(latest[prices_paid_col]) if prices_paid_col else None,
                future_activity=float(latest[future_col]) if future_col else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse manufacturing Excel: {e}")
            return None

    async def get_leading_index(self, state: str = "US") -> Optional[LeadingIndex]:
        """Get the leading index for a state.

        Args:
            state: Two-letter state code or "US" for national

        Returns:
            LeadingIndex data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(LEADING_INDEX_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_leading_index_excel(content, state)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch leading index: {e}")
            return None

    def _parse_leading_index_excel(self, content: bytes, state: str) -> Optional[LeadingIndex]:
        """Parse leading index from Excel file."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            # Find state column
            state_col = None
            for col in df.columns:
                if state.upper() in str(col).upper():
                    state_col = col
                    break

            if state_col is None:
                logger.debug(f"State {state} not found in leading index data")
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            six_month_ago = df.iloc[-7] if len(df) > 6 else None

            date_col = df.columns[0]
            obs_date = pd.to_datetime(latest[date_col]).date()

            value = float(latest[state_col])
            prev_value = float(prev[state_col]) if prev is not None else None
            six_month_value = float(six_month_ago[state_col]) if six_month_ago is not None else None

            return LeadingIndex(
                date=obs_date,
                state=state.upper(),
                leading_index=value,
                previous_value=prev_value,
                six_month_change=value - six_month_value if six_month_value else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse leading index Excel: {e}")
            return None

    async def get_coincident_index(self, state: str = "US") -> Optional[CoincidentIndex]:
        """Get the coincident index for a state.

        Args:
            state: Two-letter state code or "US" for national

        Returns:
            CoincidentIndex data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(COINCIDENT_INDEX_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_coincident_index_excel(content, state)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch coincident index: {e}")
            return None

    def _parse_coincident_index_excel(self, content: bytes, state: str) -> Optional[CoincidentIndex]:
        """Parse coincident index from Excel file."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            state_col = None
            for col in df.columns:
                if state.upper() in str(col).upper():
                    state_col = col
                    break

            if state_col is None:
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            three_month_ago = df.iloc[-4] if len(df) > 3 else None
            year_ago = df.iloc[-13] if len(df) > 12 else None

            date_col = df.columns[0]
            obs_date = pd.to_datetime(latest[date_col]).date()

            value = float(latest[state_col])

            return CoincidentIndex(
                date=obs_date,
                state=state.upper(),
                coincident_index=value,
                one_month_change=value - float(prev[state_col]) if prev is not None else None,
                three_month_change=(value - float(three_month_ago[state_col])) * 4 if three_month_ago is not None else None,
                twelve_month_change=value - float(year_ago[state_col]) if year_ago is not None else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse coincident index Excel: {e}")
            return None

    async def get_ads_index(self) -> Optional[ADSIndex]:
        """Get the latest ADS Business Conditions Index.

        Returns:
            ADSIndex with latest value, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(ADS_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_ads_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch ADS index: {e}")
            return None

    def _parse_ads_excel(self, content: bytes) -> Optional[ADSIndex]:
        """Parse ADS index from Excel file."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]
            ads_col = df.columns[1]

            obs_date = pd.to_datetime(latest[date_col]).date()
            ads_value = float(latest[ads_col])

            return ADSIndex(
                date=obs_date,
                ads_index=ads_value,
            )
        except Exception as e:
            logger.debug(f"Could not parse ADS Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Get all Philadelphia Fed indicators.

        Returns:
            Dictionary with all available indicators
        """
        import asyncio

        mfg, leading, coincident, ads = await asyncio.gather(
            self.get_manufacturing_survey(),
            self.get_leading_index("US"),
            self.get_coincident_index("US"),
            self.get_ads_index(),
            return_exceptions=True,
        )

        return {
            "manufacturing_survey": mfg if not isinstance(mfg, Exception) else None,
            "leading_index_us": leading if not isinstance(leading, Exception) else None,
            "coincident_index_us": coincident if not isinstance(coincident, Exception) else None,
            "ads_index": ads if not isinstance(ads, Exception) else None,
        }


# Singleton instance
_client: Optional[PhiladelphiaFedClient] = None


def get_philly_fed_client() -> PhiladelphiaFedClient:
    """Get the shared Philadelphia Fed client instance."""
    global _client
    if _client is None:
        _client = PhiladelphiaFedClient()
    return _client
