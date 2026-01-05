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

"""Atlanta Federal Reserve Data Client.

Key Data Series:
    - GDPNow: Real-time GDP growth estimate (updated ~6x per quarter)
    - Wage Growth Tracker: Median wage growth from job-stayers
    - Business Inflation Expectations: Survey-based inflation outlook
    - Taylor Rule Utility: Implied policy rate

Data Sources:
    - https://www.atlantafed.org/cqer/research/gdpnow
    - https://www.atlantafed.org/chcs/wage-growth-tracker

Investment Signals:
    - GDPNow vs consensus: Leading indicator of GDP surprises
    - GDPNow trend: Acceleration/deceleration of growth
    - Wage Growth Tracker: Inflation pressure indicator
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Atlanta Fed data URLs
GDPNOW_URL = "https://www.atlantafed.org/cqer/research/gdpnow"
GDPNOW_DATA_URL = "https://www.atlantafed.org/-/media/documents/cqer/researchcq/gdpnow/GDPNowForecast.xlsx"
WAGE_TRACKER_URL = (
    "https://www.atlantafed.org/-/media/documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx"
)
BIE_URL = "https://www.atlantafed.org/-/media/documents/research/inflationproject/bie/BIEData.xlsx"


class GDPOutlook(Enum):
    """Classification of GDP growth outlook."""

    STRONG_CONTRACTION = "strong_contraction"  # < -2%
    CONTRACTION = "contraction"  # -2% to 0%
    WEAK_GROWTH = "weak_growth"  # 0% to 1%
    MODERATE_GROWTH = "moderate_growth"  # 1% to 2.5%
    STRONG_GROWTH = "strong_growth"  # 2.5% to 4%
    VERY_STRONG = "very_strong"  # > 4%


@dataclass
class GDPNowData:
    """Atlanta Fed GDPNow real-time GDP estimate.

    GDPNow is a "nowcasting" model that provides a running estimate of
    real GDP growth based on available economic data. It's updated
    multiple times as new data releases occur.

    Attributes:
        date: Date of the estimate
        quarter: Quarter being estimated (e.g., "2025Q1")
        gdp_estimate: Real GDP growth estimate (annualized %)
        previous_estimate: Prior estimate before latest data
        change_from_previous: Change in estimate
        blue_chip_consensus: Blue Chip consensus forecast (if available)
        components: Contribution breakdown by category
        outlook: Classified growth outlook
        data_releases_incorporated: List of data releases in estimate
    """

    date: date
    quarter: str
    gdp_estimate: float
    previous_estimate: Optional[float] = None
    change_from_previous: Optional[float] = None
    blue_chip_consensus: Optional[float] = None
    components: Optional[Dict[str, float]] = None
    outlook: Optional[GDPOutlook] = None
    data_releases_incorporated: Optional[List[str]] = None

    def __post_init__(self):
        if self.outlook is None:
            self.outlook = self._classify_outlook()

    def _classify_outlook(self) -> GDPOutlook:
        if self.gdp_estimate < -2:
            return GDPOutlook.STRONG_CONTRACTION
        elif self.gdp_estimate < 0:
            return GDPOutlook.CONTRACTION
        elif self.gdp_estimate < 1:
            return GDPOutlook.WEAK_GROWTH
        elif self.gdp_estimate < 2.5:
            return GDPOutlook.MODERATE_GROWTH
        elif self.gdp_estimate < 4:
            return GDPOutlook.STRONG_GROWTH
        else:
            return GDPOutlook.VERY_STRONG

    @property
    def vs_consensus(self) -> Optional[float]:
        """Difference from Blue Chip consensus (positive = above consensus)."""
        if self.blue_chip_consensus is not None:
            return self.gdp_estimate - self.blue_chip_consensus
        return None

    @property
    def is_accelerating(self) -> Optional[bool]:
        """Whether growth estimate is increasing."""
        if self.change_from_previous is not None:
            return self.change_from_previous > 0
        return None


@dataclass
class WageGrowthData:
    """Atlanta Fed Wage Growth Tracker data.

    Measures median wage growth of continuously employed workers,
    providing a cleaner signal of labor market tightness than
    average wage measures.

    Attributes:
        date: Month of the observation
        overall: Overall median wage growth (12-month moving avg)
        job_stayers: Wage growth for workers staying in same job
        job_switchers: Wage growth for workers changing jobs
        full_time: Full-time workers wage growth
        part_time: Part-time workers wage growth
        hourly: Hourly workers wage growth
        non_hourly: Non-hourly workers wage growth
    """

    date: date
    overall: float
    job_stayers: Optional[float] = None
    job_switchers: Optional[float] = None
    full_time: Optional[float] = None
    part_time: Optional[float] = None
    hourly: Optional[float] = None
    non_hourly: Optional[float] = None

    @property
    def switcher_premium(self) -> Optional[float]:
        """Premium for switching jobs vs staying."""
        if self.job_stayers and self.job_switchers:
            return self.job_switchers - self.job_stayers
        return None


@dataclass
class BusinessInflationExpectations:
    """Atlanta Fed Business Inflation Expectations survey.

    Monthly survey of business leaders on their inflation expectations.

    Attributes:
        date: Month of the survey
        year_ahead: Expected inflation 1 year ahead
        year_ahead_uncertainty: Uncertainty around 1-year forecast
        unit_cost_growth: Expected unit cost growth
        sales_growth: Expected sales growth
    """

    date: date
    year_ahead: float
    year_ahead_uncertainty: Optional[float] = None
    unit_cost_growth: Optional[float] = None
    sales_growth: Optional[float] = None


class AtlantaFedClient:
    """Client for Atlanta Federal Reserve economic data.

    This client fetches real-time economic indicators from the
    Atlanta Fed, most notably the GDPNow estimate.

    Example:
        client = get_atlanta_fed_client()

        # Get current GDPNow estimate
        gdpnow = await client.get_gdpnow()
        print(f"GDP estimate: {gdpnow.gdp_estimate}% ({gdpnow.outlook.value})")

        # Get wage growth tracker
        wages = await client.get_wage_growth()
        print(f"Wage growth: {wages.overall}%")
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

    async def get_gdpnow(self) -> Optional[GDPNowData]:
        """Get the latest GDPNow GDP estimate.

        Returns:
            GDPNowData with current estimate, or None if unavailable
        """
        try:
            session = await self._get_session()

            # Try to get the data from the Excel file first
            try:
                async with session.get(GDPNOW_DATA_URL, timeout=30) as response:
                    if response.status == 200:
                        content = await response.read()
                        result = self._parse_gdpnow_excel(content)
                        if result:
                            return result
            except Exception as e:
                logger.debug(f"Excel fetch failed: {e}")

            # Fallback: scrape the main page
            async with session.get(GDPNOW_URL, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_gdpnow_html(html)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch GDPNow: {e}")
            return None

    def _parse_gdpnow_excel(self, content: bytes) -> Optional[GDPNowData]:
        """Parse GDPNow data from Excel file."""
        try:
            import io

            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            # Get the latest row
            if df.empty:
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None

            # Column names vary - try common patterns
            date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            gdp_col = next((c for c in df.columns if "gdpnow" in c.lower() or "forecast" in c.lower()), df.columns[1])

            estimate_date = pd.to_datetime(latest[date_col]).date()
            gdp_estimate = float(latest[gdp_col])
            prev_estimate = float(prev[gdp_col]) if prev is not None else None

            # Determine quarter from date
            q = (estimate_date.month - 1) // 3 + 1
            quarter = f"{estimate_date.year}Q{q}"

            return GDPNowData(
                date=estimate_date,
                quarter=quarter,
                gdp_estimate=gdp_estimate,
                previous_estimate=prev_estimate,
                change_from_previous=gdp_estimate - prev_estimate if prev_estimate else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse GDPNow Excel: {e}")
            return None

    def _parse_gdpnow_html(self, html: str) -> Optional[GDPNowData]:
        """Parse GDPNow data from HTML page."""
        try:
            # Look for GDP growth patterns - try multiple approaches
            patterns = [
                r"GDP.*?(-?\d+\.\d+).*?percent",  # "GDP growth is 3.0 percent"
                r"GDPNow.*?(-?\d+\.\d+)",  # "GDPNow model estimate... 3.0"
                r"estimate.*?(-?\d+\.\d+).*?percent",  # "estimate of 3.0 percent"
            ]

            gdp_estimate = None
            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    # Sanity check: GDP growth should be between -15% and +15%
                    if -15 <= val <= 15:
                        gdp_estimate = val
                        break

            if gdp_estimate is not None:
                today = date.today()
                q = (today.month - 1) // 3 + 1
                quarter = f"{today.year}Q{q}"

                return GDPNowData(
                    date=today,
                    quarter=quarter,
                    gdp_estimate=gdp_estimate,
                )
            return None
        except Exception as e:
            logger.debug(f"Could not parse GDPNow HTML: {e}")
            return None

    async def get_wage_growth(self) -> Optional[WageGrowthData]:
        """Get the latest Wage Growth Tracker data.

        Returns:
            WageGrowthData with current metrics, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(WAGE_TRACKER_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_wage_growth_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch wage growth data: {e}")
            return None

    def _parse_wage_growth_excel(self, content: bytes) -> Optional[WageGrowthData]:
        """Parse wage growth data from Excel file."""
        try:
            import io

            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]

            # Find columns
            date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            overall_col = next((c for c in df.columns if "overall" in c.lower() or "total" in c.lower()), None)

            obs_date = pd.to_datetime(latest[date_col]).date()
            overall = float(latest[overall_col]) if overall_col else float(latest.iloc[1])

            return WageGrowthData(
                date=obs_date,
                overall=overall,
            )
        except Exception as e:
            logger.debug(f"Could not parse wage growth Excel: {e}")
            return None

    async def get_business_inflation_expectations(self) -> Optional[BusinessInflationExpectations]:
        """Get the latest Business Inflation Expectations survey.

        Returns:
            BusinessInflationExpectations data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(BIE_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_bie_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch BIE data: {e}")
            return None

    def _parse_bie_excel(self, content: bytes) -> Optional[BusinessInflationExpectations]:
        """Parse Business Inflation Expectations from Excel."""
        try:
            import io

            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]
            obs_date = pd.to_datetime(latest[date_col]).date()

            # Find year-ahead column
            ya_col = next((c for c in df.columns if "year" in c.lower() and "ahead" in c.lower()), None)
            year_ahead = float(latest[ya_col]) if ya_col else float(latest.iloc[1])

            return BusinessInflationExpectations(
                date=obs_date,
                year_ahead=year_ahead,
            )
        except Exception as e:
            logger.debug(f"Could not parse BIE Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Get all Atlanta Fed indicators.

        Returns:
            Dictionary with all available indicators
        """
        import asyncio

        gdpnow, wages, bie = await asyncio.gather(
            self.get_gdpnow(),
            self.get_wage_growth(),
            self.get_business_inflation_expectations(),
            return_exceptions=True,
        )

        return {
            "gdpnow": gdpnow if not isinstance(gdpnow, Exception) else None,
            "wage_growth": wages if not isinstance(wages, Exception) else None,
            "business_inflation_expectations": bie if not isinstance(bie, Exception) else None,
        }


# Singleton instance
_client: Optional[AtlantaFedClient] = None


def get_atlanta_fed_client() -> AtlantaFedClient:
    """Get the shared Atlanta Fed client instance."""
    global _client
    if _client is None:
        _client = AtlantaFedClient()
    return _client
