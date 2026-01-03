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

"""Treasury Fiscal Data API Client.

This module provides access to U.S. Treasury yield curve data via the
Treasury Fiscal Data API (https://fiscaldata.treasury.gov/api-documentation/).

The API is FREE and requires no authentication.

Key Endpoints:
- Daily Treasury Par Yield Curve Rates
- Average Interest Rates on U.S. Treasury Securities

Example:
    client = get_treasury_client()

    # Get current yield curve
    curve = await client.get_yield_curve()
    print(f"10Y yield: {curve.yield_10y}%")
    print(f"2Y-10Y spread: {curve.spread_10y_2y} bps")
    print(f"Inverted: {curve.is_inverted}")

    # Get historical yields
    history = await client.get_yield_history(days=365)
"""

import asyncio
import logging
import ssl
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp

# Try to use certifi for SSL certificates (fixes macOS issues)
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

logger = logging.getLogger(__name__)

# Treasury API base URL
TREASURY_API_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

# Daily Treasury Par Yield Curve Rates endpoint
YIELD_CURVE_ENDPOINT = "/v2/accounting/od/avg_interest_rates"
DAILY_YIELD_ENDPOINT = "/v2/accounting/od/rates_of_exchange"

# Alternative: Treasury.gov XML feed for daily rates
TREASURY_XML_BASE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"


@dataclass
class TreasuryYield:
    """Single treasury yield data point.

    Attributes:
        date: Date of the yield
        maturity: Maturity term (e.g., "1 Month", "10 Year")
        yield_pct: Yield percentage
        maturity_months: Maturity in months for sorting
    """
    date: date
    maturity: str
    yield_pct: Optional[float]
    maturity_months: int = 0

    def __post_init__(self):
        """Calculate maturity in months for sorting."""
        if "Month" in self.maturity:
            try:
                self.maturity_months = int(self.maturity.split()[0])
            except (ValueError, IndexError):
                pass
        elif "Year" in self.maturity:
            try:
                years = int(self.maturity.split()[0])
                self.maturity_months = years * 12
            except (ValueError, IndexError):
                pass


@dataclass
class YieldCurveData:
    """Complete yield curve snapshot.

    Attributes:
        date: Date of the yield curve
        yield_1m: 1-month yield
        yield_2m: 2-month yield
        yield_3m: 3-month yield
        yield_4m: 4-month yield
        yield_6m: 6-month yield
        yield_1y: 1-year yield
        yield_2y: 2-year yield
        yield_3y: 3-year yield
        yield_5y: 5-year yield
        yield_7y: 7-year yield
        yield_10y: 10-year yield
        yield_20y: 20-year yield
        yield_30y: 30-year yield
        spread_10y_2y: 10Y minus 2Y spread in basis points
        spread_10y_3m: 10Y minus 3M spread in basis points
        is_inverted: Whether 10Y-2Y spread is negative
        is_deeply_inverted: Whether spread < -50 bps
    """
    date: date
    yield_1m: Optional[float] = None
    yield_2m: Optional[float] = None
    yield_3m: Optional[float] = None
    yield_4m: Optional[float] = None
    yield_6m: Optional[float] = None
    yield_1y: Optional[float] = None
    yield_2y: Optional[float] = None
    yield_3y: Optional[float] = None
    yield_5y: Optional[float] = None
    yield_7y: Optional[float] = None
    yield_10y: Optional[float] = None
    yield_20y: Optional[float] = None
    yield_30y: Optional[float] = None
    spread_10y_2y: Optional[float] = None
    spread_10y_3m: Optional[float] = None
    is_inverted: bool = False
    is_deeply_inverted: bool = False

    def __post_init__(self):
        """Calculate spreads and inversion status."""
        # Calculate 10Y-2Y spread
        if self.yield_10y is not None and self.yield_2y is not None:
            self.spread_10y_2y = round((self.yield_10y - self.yield_2y) * 100, 2)  # bps
            self.is_inverted = self.spread_10y_2y < 0
            self.is_deeply_inverted = self.spread_10y_2y < -50

        # Calculate 10Y-3M spread (alternative recession indicator)
        if self.yield_10y is not None and self.yield_3m is not None:
            self.spread_10y_3m = round((self.yield_10y - self.yield_3m) * 100, 2)  # bps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": str(self.date),
            "yields": {
                "1m": self.yield_1m,
                "2m": self.yield_2m,
                "3m": self.yield_3m,
                "4m": self.yield_4m,
                "6m": self.yield_6m,
                "1y": self.yield_1y,
                "2y": self.yield_2y,
                "3y": self.yield_3y,
                "5y": self.yield_5y,
                "7y": self.yield_7y,
                "10y": self.yield_10y,
                "20y": self.yield_20y,
                "30y": self.yield_30y,
            },
            "spreads": {
                "10y_2y_bps": self.spread_10y_2y,
                "10y_3m_bps": self.spread_10y_3m,
            },
            "inversion": {
                "is_inverted": self.is_inverted,
                "is_deeply_inverted": self.is_deeply_inverted,
            },
        }

    @property
    def curve_shape(self) -> str:
        """Classify the yield curve shape."""
        if self.spread_10y_2y is None:
            return "unknown"
        if self.spread_10y_2y < -50:
            return "deeply_inverted"
        if self.spread_10y_2y < 0:
            return "inverted"
        if self.spread_10y_2y < 50:
            return "flat"
        if self.spread_10y_2y < 150:
            return "normal"
        return "steep"


class TreasuryApiClient:
    """Client for Treasury Fiscal Data API.

    Provides async access to U.S. Treasury yield curve data.
    The API is FREE and requires no authentication.

    SOLID: Single Responsibility - only handles Treasury API communication
    """

    # Maturity mappings for parsing
    MATURITY_MAP = {
        "1 Mo": "1m",
        "2 Mo": "2m",
        "3 Mo": "3m",
        "4 Mo": "4m",
        "6 Mo": "6m",
        "1 Yr": "1y",
        "2 Yr": "2y",
        "3 Yr": "3y",
        "5 Yr": "5y",
        "7 Yr": "7y",
        "10 Yr": "10y",
        "20 Yr": "20y",
        "30 Yr": "30y",
    }

    def __init__(self, timeout: int = 30):
        """Initialize Treasury API client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper SSL context."""
        if self._session is None or self._session.closed:
            # Use TCPConnector with SSL context for proper certificate handling
            connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector
            )
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_yield_curve(self, as_of_date: Optional[date] = None) -> Optional[YieldCurveData]:
        """Get yield curve for a specific date.

        Args:
            as_of_date: Date for yield curve (default: latest available)

        Returns:
            YieldCurveData or None if not available
        """
        try:
            # Use Treasury.gov XML/CSV feed for daily par yield curve rates
            # This is more reliable than the API for current yields
            yields = await self._fetch_treasury_yields(as_of_date)

            if not yields:
                logger.warning("No yield data available")
                return None

            return self._build_yield_curve(yields)

        except Exception as e:
            logger.error(f"Error fetching yield curve: {e}")
            return None

    async def _fetch_treasury_yields(
        self,
        as_of_date: Optional[date] = None,
        days_back: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch treasury yields from Treasury.gov.

        Uses the Treasury.gov XML feed which provides daily par yield curve rates.

        Args:
            as_of_date: Target date (will search back days_back days if not available)
            days_back: Number of days to search back

        Returns:
            List of yield dictionaries
        """
        session = await self._get_session()

        target_date = as_of_date or date.today()

        # Try fetching from Treasury.gov XML feed
        # Format: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/YYYY/all
        year = target_date.year
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_treasury_csv(text, target_date, days_back)
        except Exception as e:
            logger.warning(f"Failed to fetch from Treasury.gov CSV: {e}")

        # Fallback: Try FRED for Treasury yields
        return await self._fetch_from_fred_fallback(target_date)

    def _parse_treasury_csv(
        self,
        csv_text: str,
        target_date: date,
        days_back: int
    ) -> List[Dict[str, Any]]:
        """Parse Treasury CSV response.

        Args:
            csv_text: CSV text content
            target_date: Target date to find
            days_back: Days to search back

        Returns:
            List of yield dictionaries for the target date
        """
        lines = csv_text.strip().split('\n')
        if len(lines) < 2:
            return []

        # Parse header
        header = lines[0].split(',')
        header = [h.strip().strip('"') for h in header]

        # Find the date column and yield columns
        date_col_idx = None
        yield_cols = {}

        for i, col in enumerate(header):
            col_lower = col.lower()
            if 'date' in col_lower:
                date_col_idx = i
            elif 'mo' in col_lower or 'yr' in col_lower:
                # Map column names to standard keys
                if '1 mo' in col_lower or '1mo' in col_lower:
                    yield_cols['1m'] = i
                elif '2 mo' in col_lower or '2mo' in col_lower:
                    yield_cols['2m'] = i
                elif '3 mo' in col_lower or '3mo' in col_lower:
                    yield_cols['3m'] = i
                elif '4 mo' in col_lower or '4mo' in col_lower:
                    yield_cols['4m'] = i
                elif '6 mo' in col_lower or '6mo' in col_lower:
                    yield_cols['6m'] = i
                elif '1 yr' in col_lower or '1yr' in col_lower:
                    yield_cols['1y'] = i
                elif '2 yr' in col_lower or '2yr' in col_lower:
                    yield_cols['2y'] = i
                elif '3 yr' in col_lower or '3yr' in col_lower:
                    yield_cols['3y'] = i
                elif '5 yr' in col_lower or '5yr' in col_lower:
                    yield_cols['5y'] = i
                elif '7 yr' in col_lower or '7yr' in col_lower:
                    yield_cols['7y'] = i
                elif '10 yr' in col_lower or '10yr' in col_lower:
                    yield_cols['10y'] = i
                elif '20 yr' in col_lower or '20yr' in col_lower:
                    yield_cols['20y'] = i
                elif '30 yr' in col_lower or '30yr' in col_lower:
                    yield_cols['30y'] = i

        if date_col_idx is None:
            logger.warning("Could not find date column in Treasury CSV")
            return []

        # Parse data rows, looking for target date or closest before
        min_date = target_date - timedelta(days=days_back)
        best_row = None
        best_date = None

        for line in lines[1:]:
            if not line.strip():
                continue

            cols = line.split(',')
            cols = [c.strip().strip('"') for c in cols]

            if len(cols) <= date_col_idx:
                continue

            # Parse date
            date_str = cols[date_col_idx]
            try:
                row_date = datetime.strptime(date_str, '%m/%d/%Y').date()
            except ValueError:
                try:
                    row_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    continue

            # Check if this date is valid and better than current best
            if row_date <= target_date and row_date >= min_date:
                if best_date is None or row_date > best_date:
                    best_date = row_date
                    best_row = cols

        if best_row is None:
            return []

        # Extract yields from best row
        yields = [{'date': best_date}]
        for maturity, col_idx in yield_cols.items():
            if col_idx < len(best_row):
                try:
                    value = best_row[col_idx].strip()
                    if value and value.lower() != 'n/a':
                        yields[0][maturity] = float(value)
                except (ValueError, IndexError):
                    pass

        return yields

    async def _fetch_from_fred_fallback(self, target_date: date) -> List[Dict[str, Any]]:
        """Fallback to FRED for treasury yields if Treasury.gov fails.

        Args:
            target_date: Target date

        Returns:
            List of yield dictionaries
        """
        # FRED series IDs for treasury yields
        fred_series = {
            '1m': 'DGS1MO',
            '3m': 'DGS3MO',
            '6m': 'DGS6MO',
            '1y': 'DGS1',
            '2y': 'DGS2',
            '3y': 'DGS3',
            '5y': 'DGS5',
            '7y': 'DGS7',
            '10y': 'DGS10',
            '20y': 'DGS20',
            '30y': 'DGS30',
        }

        try:
            # Try to get from local FRED cache if available
            from investigator.infrastructure.external.fred.macro_indicators import (
                get_macro_indicator_service
            )

            service = get_macro_indicator_service()
            yields = {'date': target_date}

            for maturity, series_id in fred_series.items():
                try:
                    # Get latest value from FRED service
                    value = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: service.get_latest_value(series_id)
                    )
                    if value is not None:
                        yields[maturity] = value
                except Exception:
                    pass

            if len(yields) > 1:  # Has at least one yield besides date
                return [yields]

        except ImportError:
            logger.debug("FRED service not available for fallback")
        except Exception as e:
            logger.warning(f"FRED fallback failed: {e}")

        return []

    def _build_yield_curve(self, yields: List[Dict[str, Any]]) -> Optional[YieldCurveData]:
        """Build YieldCurveData from parsed yields.

        Args:
            yields: List of yield dictionaries

        Returns:
            YieldCurveData instance
        """
        if not yields:
            return None

        data = yields[0]
        yield_date = data.get('date', date.today())

        return YieldCurveData(
            date=yield_date,
            yield_1m=data.get('1m'),
            yield_2m=data.get('2m'),
            yield_3m=data.get('3m'),
            yield_4m=data.get('4m'),
            yield_6m=data.get('6m'),
            yield_1y=data.get('1y'),
            yield_2y=data.get('2y'),
            yield_3y=data.get('3y'),
            yield_5y=data.get('5y'),
            yield_7y=data.get('7y'),
            yield_10y=data.get('10y'),
            yield_20y=data.get('20y'),
            yield_30y=data.get('30y'),
        )

    async def get_yield_history(
        self,
        days: int = 365,
        maturity: str = "10y"
    ) -> List[Dict[str, Any]]:
        """Get historical yields for a specific maturity.

        Args:
            days: Number of days of history
            maturity: Maturity to get (e.g., "10y", "2y")

        Returns:
            List of {date, yield} dictionaries
        """
        try:
            session = await self._get_session()

            # Fetch full year CSV
            year = date.today().year
            url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"

            history = []
            cutoff_date = date.today() - timedelta(days=days)

            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    lines = text.strip().split('\n')

                    if len(lines) < 2:
                        return []

                    # Parse header
                    header = lines[0].split(',')
                    header = [h.strip().strip('"').lower() for h in header]

                    # Find date and target maturity columns
                    date_col = None
                    yield_col = None

                    for i, col in enumerate(header):
                        if 'date' in col:
                            date_col = i
                        elif maturity.replace('y', ' yr') in col or maturity.replace('m', ' mo') in col:
                            yield_col = i

                    if date_col is None or yield_col is None:
                        return []

                    # Parse data
                    for line in lines[1:]:
                        if not line.strip():
                            continue

                        cols = line.split(',')
                        cols = [c.strip().strip('"') for c in cols]

                        if len(cols) <= max(date_col, yield_col):
                            continue

                        try:
                            row_date = datetime.strptime(cols[date_col], '%m/%d/%Y').date()
                            if row_date < cutoff_date:
                                continue

                            yield_val = cols[yield_col]
                            if yield_val and yield_val.lower() != 'n/a':
                                history.append({
                                    'date': str(row_date),
                                    'yield': float(yield_val)
                                })
                        except (ValueError, IndexError):
                            continue

            # Sort by date
            history.sort(key=lambda x: x['date'], reverse=True)
            return history

        except Exception as e:
            logger.error(f"Error fetching yield history: {e}")
            return []

    async def get_spread_history(
        self,
        days: int = 365,
        spread_type: str = "10y_2y"
    ) -> List[Dict[str, Any]]:
        """Get historical spread data.

        Args:
            days: Number of days of history
            spread_type: Spread type ("10y_2y" or "10y_3m")

        Returns:
            List of {date, spread_bps, is_inverted} dictionaries
        """
        try:
            session = await self._get_session()

            year = date.today().year
            url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"

            history = []
            cutoff_date = date.today() - timedelta(days=days)

            # Determine which maturities to use
            if spread_type == "10y_2y":
                long_term = "10 yr"
                short_term = "2 yr"
            else:  # 10y_3m
                long_term = "10 yr"
                short_term = "3 mo"

            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    lines = text.strip().split('\n')

                    if len(lines) < 2:
                        return []

                    # Parse header
                    header = lines[0].split(',')
                    header = [h.strip().strip('"').lower() for h in header]

                    # Find columns
                    date_col = None
                    long_col = None
                    short_col = None

                    for i, col in enumerate(header):
                        if 'date' in col:
                            date_col = i
                        elif long_term in col:
                            long_col = i
                        elif short_term in col:
                            short_col = i

                    if None in (date_col, long_col, short_col):
                        return []

                    # Parse data
                    for line in lines[1:]:
                        if not line.strip():
                            continue

                        cols = line.split(',')
                        cols = [c.strip().strip('"') for c in cols]

                        if len(cols) <= max(date_col, long_col, short_col):
                            continue

                        try:
                            row_date = datetime.strptime(cols[date_col], '%m/%d/%Y').date()
                            if row_date < cutoff_date:
                                continue

                            long_val = cols[long_col]
                            short_val = cols[short_col]

                            if (long_val and long_val.lower() != 'n/a' and
                                short_val and short_val.lower() != 'n/a'):

                                spread = (float(long_val) - float(short_val)) * 100  # bps

                                history.append({
                                    'date': str(row_date),
                                    'spread_bps': round(spread, 2),
                                    'is_inverted': spread < 0
                                })
                        except (ValueError, IndexError):
                            continue

            # Sort by date
            history.sort(key=lambda x: x['date'], reverse=True)
            return history

        except Exception as e:
            logger.error(f"Error fetching spread history: {e}")
            return []


# Singleton instance
_treasury_client: Optional[TreasuryApiClient] = None


def get_treasury_client() -> TreasuryApiClient:
    """Get or create singleton Treasury client instance."""
    global _treasury_client
    if _treasury_client is None:
        _treasury_client = TreasuryApiClient()
    return _treasury_client


class TreasuryFetcher:
    """High-level Treasury data fetcher for scheduled jobs.

    Provides a synchronous-friendly interface for the scheduler.
    """

    def __init__(self):
        self.client = get_treasury_client()

    async def get_yield_curve_history(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """Get yield curve history between dates.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of yield curve dictionaries
        """
        try:
            from datetime import datetime

            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date()
            days = (end - start).days

            # Fetch full yield curves for the date range
            results = []
            current = end

            while current >= start:
                curve = await self.client.get_yield_curve(as_of_date=current)
                if curve:
                    results.append({
                        "date": str(curve.date),
                        "yield_1m": curve.yield_1m,
                        "yield_3m": curve.yield_3m,
                        "yield_6m": curve.yield_6m,
                        "yield_1y": curve.yield_1y,
                        "yield_2y": curve.yield_2y,
                        "yield_5y": curve.yield_5y,
                        "yield_10y": curve.yield_10y,
                        "yield_20y": curve.yield_20y,
                        "yield_30y": curve.yield_30y,
                    })
                    # Skip to avoid fetching same data repeatedly
                    current = curve.date - timedelta(days=1)
                else:
                    current -= timedelta(days=1)

                # Limit to prevent excessive API calls
                if len(results) >= days:
                    break

            return results

        except Exception as e:
            logger.error(f"Error fetching yield curve history: {e}")
            return []


# Singleton fetcher instance
_treasury_fetcher: Optional[TreasuryFetcher] = None


def get_treasury_fetcher() -> TreasuryFetcher:
    """Get or create singleton Treasury fetcher instance.

    This is the preferred interface for scheduled jobs.
    """
    global _treasury_fetcher
    if _treasury_fetcher is None:
        _treasury_fetcher = TreasuryFetcher()
    return _treasury_fetcher
