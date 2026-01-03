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

"""Cleveland Federal Reserve Data Client.

Key Data Series:
    - Inflation Expectations: Model-based inflation forecasts
    - Yield Curve Model: Recession probability from term structure
    - Median CPI: Core inflation measure
    - 16% Trimmed-Mean CPI: Alternative core measure

Data Sources:
    - https://www.clevelandfed.org/indicators-and-data/inflation-expectations
    - https://www.clevelandfed.org/indicators-and-data/yield-curve-and-predicted-gdp-growth

Investment Signals:
    - 10-year inflation expectations: Long-term inflation outlook
    - Expected vs actual inflation gap: Surprise risk
    - Yield curve recession probability: Complementary to NY Fed model
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Cleveland Fed data URLs
INFLATION_EXPECTATIONS_URL = "https://www.clevelandfed.org/-/media/files/charts/inflation-expectations/ie_data.xlsx"
YIELD_CURVE_URL = "https://www.clevelandfed.org/-/media/files/charts/yield-curve/yield_curve_data.xlsx"
MEDIAN_CPI_URL = "https://www.clevelandfed.org/-/media/files/charts/median-cpi/median-cpi.xlsx"


class InflationOutlook(Enum):
    """Classification of inflation expectations."""
    DEFLATION_RISK = "deflation_risk"      # < 1%
    VERY_LOW = "very_low"                  # 1-1.5%
    LOW = "low"                            # 1.5-2%
    TARGET = "target"                      # 2-2.5%
    ELEVATED = "elevated"                  # 2.5-3%
    HIGH = "high"                          # 3-4%
    VERY_HIGH = "very_high"                # > 4%


@dataclass
class InflationExpectations:
    """Cleveland Fed Model-Based Inflation Expectations.

    Uses financial market data (Treasury yields, inflation swaps, surveys)
    to estimate expected inflation at various horizons.

    Attributes:
        date: Date of the estimate
        one_year: 1-year expected inflation
        two_year: 2-year expected inflation
        five_year: 5-year expected inflation
        ten_year: 10-year expected inflation (key for long-term)
        five_year_five_year: 5Y5Y forward inflation (Fed's preferred)
        inflation_risk_premium: Compensation for inflation uncertainty
        outlook: Classified inflation outlook
    """
    date: date
    one_year: float
    two_year: Optional[float] = None
    five_year: Optional[float] = None
    ten_year: Optional[float] = None
    five_year_five_year: Optional[float] = None
    inflation_risk_premium: Optional[float] = None
    outlook: Optional[InflationOutlook] = None

    def __post_init__(self):
        if self.outlook is None:
            self.outlook = self._classify_outlook()

    def _classify_outlook(self) -> InflationOutlook:
        # Use 10-year or 5-year as primary
        ref = self.ten_year or self.five_year or self.one_year

        if ref < 1:
            return InflationOutlook.DEFLATION_RISK
        elif ref < 1.5:
            return InflationOutlook.VERY_LOW
        elif ref < 2:
            return InflationOutlook.LOW
        elif ref < 2.5:
            return InflationOutlook.TARGET
        elif ref < 3:
            return InflationOutlook.ELEVATED
        elif ref < 4:
            return InflationOutlook.HIGH
        else:
            return InflationOutlook.VERY_HIGH

    @property
    def is_anchored(self) -> bool:
        """Whether long-term inflation is near 2% target."""
        if self.ten_year:
            return 1.5 <= self.ten_year <= 2.5
        return False

    @property
    def term_structure_slope(self) -> Optional[float]:
        """Difference between 10Y and 1Y expectations (steepness)."""
        if self.ten_year and self.one_year:
            return self.ten_year - self.one_year
        return None


@dataclass
class YieldCurveModel:
    """Cleveland Fed Yield Curve Model.

    Alternative recession probability model using the full term structure,
    not just 10Y-3M spread like NY Fed.

    Attributes:
        date: Date of the estimate
        recession_prob_12m: 12-month ahead recession probability
        recession_prob_24m: 24-month ahead recession probability
        expected_gdp_growth: Expected real GDP growth
        term_spread_10y_3m: 10Y-3M spread used
        term_spread_10y_2y: 10Y-2Y spread
    """
    date: date
    recession_prob_12m: float
    recession_prob_24m: Optional[float] = None
    expected_gdp_growth: Optional[float] = None
    term_spread_10y_3m: Optional[float] = None
    term_spread_10y_2y: Optional[float] = None

    @property
    def is_warning(self) -> bool:
        """Whether recession probability is elevated (>30%)."""
        return self.recession_prob_12m > 30

    @property
    def is_high_risk(self) -> bool:
        """Whether recession probability is high (>50%)."""
        return self.recession_prob_12m > 50


@dataclass
class MedianCPI:
    """Cleveland Fed Median CPI.

    The median CPI calculates the median price change of CPI components,
    providing a cleaner measure of underlying inflation than headline.

    Attributes:
        date: Month of the observation
        median_cpi: Median CPI (annualized monthly change)
        median_cpi_yoy: Year-over-year change
        trimmed_mean_16: 16% trimmed-mean CPI
        trimmed_mean_16_yoy: 16% trimmed-mean year-over-year
    """
    date: date
    median_cpi: float
    median_cpi_yoy: Optional[float] = None
    trimmed_mean_16: Optional[float] = None
    trimmed_mean_16_yoy: Optional[float] = None


class ClevelandFedClient:
    """Client for Cleveland Federal Reserve economic data.

    Example:
        client = get_cleveland_fed_client()

        # Get inflation expectations
        infl = await client.get_inflation_expectations()
        print(f"10Y inflation: {infl.ten_year}% ({infl.outlook.value})")

        # Get yield curve recession probability
        yc = await client.get_yield_curve_model()
        print(f"12M recession prob: {yc.recession_prob_12m}%")
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

    async def get_inflation_expectations(self) -> Optional[InflationExpectations]:
        """Get the latest model-based inflation expectations.

        Returns:
            InflationExpectations data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(INFLATION_EXPECTATIONS_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_inflation_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch inflation expectations: {e}")
            return None

    def _parse_inflation_excel(self, content: bytes) -> Optional[InflationExpectations]:
        """Parse inflation expectations from Excel file."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            # Find columns by pattern
            def find_col(pattern: str) -> Optional[str]:
                for col in df.columns:
                    if pattern.lower() in str(col).lower():
                        return col
                return None

            one_yr = find_col('1-year') or find_col('1 year') or find_col('1yr')
            two_yr = find_col('2-year') or find_col('2 year') or find_col('2yr')
            five_yr = find_col('5-year') or find_col('5 year') or find_col('5yr')
            ten_yr = find_col('10-year') or find_col('10 year') or find_col('10yr')
            five_five = find_col('5y5y') or find_col('5-year, 5-year')

            # Use first numeric column as fallback
            one_year_val = float(latest[one_yr]) if one_yr else float(latest[df.columns[1]])

            return InflationExpectations(
                date=obs_date,
                one_year=one_year_val,
                two_year=float(latest[two_yr]) if two_yr else None,
                five_year=float(latest[five_yr]) if five_yr else None,
                ten_year=float(latest[ten_yr]) if ten_yr else None,
                five_year_five_year=float(latest[five_five]) if five_five else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse inflation Excel: {e}")
            return None

    async def get_yield_curve_model(self) -> Optional[YieldCurveModel]:
        """Get the latest yield curve recession probability.

        Returns:
            YieldCurveModel data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(YIELD_CURVE_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_yield_curve_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch yield curve model: {e}")
            return None

    def _parse_yield_curve_excel(self, content: bytes) -> Optional[YieldCurveModel]:
        """Parse yield curve model from Excel file."""
        try:
            import io
            import pandas as pd

            df = pd.read_excel(io.BytesIO(content), sheet_name=0)

            if df.empty:
                return None

            latest = df.iloc[-1]
            date_col = df.columns[0]

            obs_date = pd.to_datetime(latest[date_col]).date()

            # Find recession probability column
            def find_col(pattern: str) -> Optional[str]:
                for col in df.columns:
                    if pattern.lower() in str(col).lower():
                        return col
                return None

            prob_col = find_col('recession') or find_col('probability') or df.columns[1]
            gdp_col = find_col('gdp')
            spread_10_3 = find_col('10y-3m') or find_col('10-3')
            spread_10_2 = find_col('10y-2y') or find_col('10-2')

            return YieldCurveModel(
                date=obs_date,
                recession_prob_12m=float(latest[prob_col]),
                expected_gdp_growth=float(latest[gdp_col]) if gdp_col else None,
                term_spread_10y_3m=float(latest[spread_10_3]) if spread_10_3 else None,
                term_spread_10y_2y=float(latest[spread_10_2]) if spread_10_2 else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse yield curve Excel: {e}")
            return None

    async def get_median_cpi(self) -> Optional[MedianCPI]:
        """Get the latest Median CPI data.

        Returns:
            MedianCPI data, or None if unavailable
        """
        try:
            session = await self._get_session()

            async with session.get(MEDIAN_CPI_URL, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    return self._parse_median_cpi_excel(content)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch median CPI: {e}")
            return None

    def _parse_median_cpi_excel(self, content: bytes) -> Optional[MedianCPI]:
        """Parse median CPI from Excel file."""
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
            def find_col(pattern: str) -> Optional[str]:
                for col in df.columns:
                    if pattern.lower() in str(col).lower():
                        return col
                return None

            median_col = find_col('median') or df.columns[1]
            trimmed_col = find_col('trimmed') or find_col('16%')

            return MedianCPI(
                date=obs_date,
                median_cpi=float(latest[median_col]),
                trimmed_mean_16=float(latest[trimmed_col]) if trimmed_col else None,
            )
        except Exception as e:
            logger.debug(f"Could not parse median CPI Excel: {e}")
            return None

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Get all Cleveland Fed indicators.

        Returns:
            Dictionary with all available indicators
        """
        import asyncio

        infl, yc, cpi = await asyncio.gather(
            self.get_inflation_expectations(),
            self.get_yield_curve_model(),
            self.get_median_cpi(),
            return_exceptions=True,
        )

        return {
            "inflation_expectations": infl if not isinstance(infl, Exception) else None,
            "yield_curve_model": yc if not isinstance(yc, Exception) else None,
            "median_cpi": cpi if not isinstance(cpi, Exception) else None,
        }


# Singleton instance
_client: Optional[ClevelandFedClient] = None


def get_cleveland_fed_client() -> ClevelandFedClient:
    """Get the shared Cleveland Fed client instance."""
    global _client
    if _client is None:
        _client = ClevelandFedClient()
    return _client
