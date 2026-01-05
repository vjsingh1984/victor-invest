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

"""FINRA Short Interest Data Fetcher.

This module fetches short interest and short volume data from FINRA's public API.
Short interest is reported twice monthly (mid-month and end-of-month).

Data Sources:
- FINRA API: https://api.finra.org/data/
- FINRA Short Interest: Bi-monthly settlement dates
- FINRA Short Volume: Daily short sale volume

Investment Signals:
- High short interest (>10% of float): Potential squeeze or bearish sentiment
- Rising short interest: Increasing bearish conviction
- Falling short interest: Short covering, potentially bullish
- Days to cover >5: Extended squeeze potential
- Short ratio spike: Contrarian buy signal if fundamentals strong

Example:
    fetcher = get_short_interest_fetcher()

    # Get current short interest
    data = await fetcher.get_short_interest("AAPL")

    # Get short interest history
    history = await fetcher.get_short_interest_history("AAPL", periods=12)

    # Get daily short volume
    volume = await fetcher.get_short_volume("AAPL", days=30)
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# FINRA API endpoints
FINRA_API_BASE = "https://api.finra.org/data"
FINRA_SHORT_INTEREST_ENDPOINT = "/group/otcMarket/name/shortInterest"
FINRA_SHORT_VOLUME_ENDPOINT = "/group/otcMarket/name/regShoDaily"

# Alternative data sources
NASDAQ_SHORT_INTEREST_URL = "https://www.nasdaq.com/market-activity/stocks/{symbol}/short-interest"


@dataclass
class ShortInterestData:
    """Short interest data for a security.

    Attributes:
        symbol: Stock ticker symbol
        settlement_date: Settlement date for the short interest report
        short_interest: Total shares sold short
        avg_daily_volume: Average daily trading volume
        days_to_cover: Short interest / avg daily volume
        short_percent_float: Short interest as % of float
        short_percent_outstanding: Short interest as % of shares outstanding
        previous_short_interest: Prior period short interest
        change_from_previous: Change from prior period
        change_percent: Percentage change from prior period
    """

    symbol: str
    settlement_date: date
    short_interest: int = 0
    avg_daily_volume: int = 0
    days_to_cover: float = 0.0
    short_percent_float: Optional[float] = None
    short_percent_outstanding: Optional[float] = None
    previous_short_interest: Optional[int] = None
    change_from_previous: Optional[int] = None
    change_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "settlement_date": str(self.settlement_date),
            "short_interest": self.short_interest,
            "avg_daily_volume": self.avg_daily_volume,
            "days_to_cover": round(self.days_to_cover, 2),
            "short_percent_float": round(self.short_percent_float, 2) if self.short_percent_float else None,
            "short_percent_outstanding": (
                round(self.short_percent_outstanding, 2) if self.short_percent_outstanding else None
            ),
            "previous": (
                {
                    "short_interest": self.previous_short_interest,
                    "change": self.change_from_previous,
                    "change_percent": round(self.change_percent, 2) if self.change_percent else None,
                }
                if self.previous_short_interest
                else None
            ),
        }


@dataclass
class ShortVolumeData:
    """Daily short volume data.

    Attributes:
        symbol: Stock ticker symbol
        trade_date: Trading date
        short_volume: Number of shares sold short
        short_exempt_volume: Short exempt volume
        total_volume: Total trading volume
        short_percent: Short volume as % of total volume
    """

    symbol: str
    trade_date: date
    short_volume: int = 0
    short_exempt_volume: int = 0
    total_volume: int = 0
    short_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "trade_date": str(self.trade_date),
            "short_volume": self.short_volume,
            "short_exempt_volume": self.short_exempt_volume,
            "total_volume": self.total_volume,
            "short_percent": round(self.short_percent, 2),
        }


@dataclass
class ShortSqueezeRisk:
    """Short squeeze risk assessment.

    Attributes:
        symbol: Stock ticker symbol
        squeeze_score: 0-100 score indicating squeeze potential
        risk_level: low, moderate, elevated, high, extreme
        factors: Contributing risk factors
        interpretation: Human-readable assessment
    """

    symbol: str
    squeeze_score: float = 0.0
    risk_level: str = "low"
    factors: List[str] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "squeeze_score": round(self.squeeze_score, 1),
            "risk_level": self.risk_level,
            "factors": self.factors,
            "interpretation": self.interpretation,
        }


class ShortInterestFetcher:
    """Fetches short interest data from FINRA and alternative sources.

    Provides access to:
    - Bi-monthly short interest reports
    - Daily short volume data
    - Short squeeze risk assessment
    - Historical short interest trends

    SOLID: Single Responsibility - only handles short interest data fetching
    """

    def __init__(self, timeout: int = 60):
        """Initialize fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "User-Agent": "Victor-Invest/1.0 (Investment Research; contact@example.com)",
                "Accept": "application/json",
            }
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_short_interest(
        self,
        symbol: str,
    ) -> Optional[ShortInterestData]:
        """Get current short interest for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ShortInterestData or None if not found
        """
        symbol = symbol.upper()

        # Try database first
        data = await self._get_from_database(symbol)
        if data:
            return data

        # Try FINRA API
        data = await self._fetch_from_finra(symbol)
        if data:
            return data

        # Try alternative sources
        data = await self._fetch_from_alternative(symbol)
        if data:
            return data

        return None

    async def _get_from_database(self, symbol: str) -> Optional[ShortInterestData]:
        """Get short interest from local database."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            query = text(
                """
                SELECT
                    symbol,
                    settlement_date,
                    short_interest,
                    days_to_cover,
                    short_percent_float,
                    avg_daily_volume
                FROM short_interest
                WHERE symbol = :symbol
                ORDER BY settlement_date DESC
                LIMIT 2
            """
            )

            with engine.connect() as conn:
                results = conn.execute(query, {"symbol": symbol}).fetchall()

                if not results:
                    return None

                current = results[0]
                previous = results[1] if len(results) > 1 else None

                change = None
                change_pct = None
                if previous and previous[2] and current[2]:
                    change = current[2] - previous[2]
                    if previous[2] > 0:
                        change_pct = (change / previous[2]) * 100

                return ShortInterestData(
                    symbol=current[0],
                    settlement_date=current[1],
                    short_interest=current[2] or 0,
                    days_to_cover=float(current[3]) if current[3] else 0.0,
                    short_percent_float=float(current[4]) if current[4] else None,
                    avg_daily_volume=current[5] or 0,
                    previous_short_interest=previous[2] if previous else None,
                    change_from_previous=change,
                    change_percent=change_pct,
                )

        except Exception as e:
            logger.debug(f"Database short interest lookup failed: {e}")
            return None

    async def _fetch_from_finra(self, symbol: str) -> Optional[ShortInterestData]:
        """Fetch short interest from FINRA API."""
        try:
            session = await self._get_session()

            # FINRA API requires specific query format
            url = f"{FINRA_API_BASE}{FINRA_SHORT_INTEREST_ENDPOINT}"
            params = {
                "symbol": symbol,
                "limit": 2,
                "sortField": "settlementDate",
                "sortDir": "desc",
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.debug(f"FINRA API returned status {response.status}")
                    return None

                data = await response.json()

                if not data or len(data) == 0:
                    return None

                current = data[0]
                previous = data[1] if len(data) > 1 else None

                settlement_date = datetime.strptime(current.get("settlementDate", ""), "%Y-%m-%d").date()

                short_interest = current.get("shortInterest", 0)
                avg_volume = current.get("avgDailyShareVolume", 0)
                days_to_cover = short_interest / avg_volume if avg_volume > 0 else 0

                prev_short = previous.get("shortInterest") if previous else None
                change = short_interest - prev_short if prev_short else None
                change_pct = (change / prev_short * 100) if prev_short and prev_short > 0 else None

                return ShortInterestData(
                    symbol=symbol,
                    settlement_date=settlement_date,
                    short_interest=short_interest,
                    avg_daily_volume=avg_volume,
                    days_to_cover=days_to_cover,
                    short_percent_float=current.get("shortPercentOfFloat"),
                    previous_short_interest=prev_short,
                    change_from_previous=change,
                    change_percent=change_pct,
                )

        except Exception as e:
            logger.debug(f"FINRA API fetch failed: {e}")
            return None

    async def _fetch_from_alternative(self, symbol: str) -> Optional[ShortInterestData]:
        """Fetch from alternative sources (NASDAQ, etc.)."""
        # For now, return synthetic data based on typical patterns
        # In production, this would scrape NASDAQ or other sources
        logger.debug(f"Using synthetic short interest data for {symbol}")

        # Generate reasonable synthetic data for demo purposes
        today = date.today()
        # Settlement dates are typically mid-month or end-of-month
        if today.day <= 15:
            settlement = date(today.year, today.month, 15)
        else:
            import calendar

            last_day = calendar.monthrange(today.year, today.month)[1]
            settlement = date(today.year, today.month, last_day)

        if settlement > today:
            # Use previous settlement
            if settlement.day == 15:
                import calendar

                prev_month = today.month - 1 if today.month > 1 else 12
                prev_year = today.year if today.month > 1 else today.year - 1
                last_day = calendar.monthrange(prev_year, prev_month)[1]
                settlement = date(prev_year, prev_month, last_day)
            else:
                settlement = date(today.year, today.month, 15)

        return None  # Return None if no real data available

    async def get_short_interest_history(self, symbol: str, periods: int = 12) -> List[ShortInterestData]:
        """Get historical short interest data.

        Args:
            symbol: Stock ticker symbol
            periods: Number of bi-monthly periods (default 12 = 6 months)

        Returns:
            List of ShortInterestData sorted by date descending
        """
        symbol = symbol.upper()

        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            query = text(
                """
                SELECT
                    symbol,
                    settlement_date,
                    short_interest,
                    days_to_cover,
                    short_percent_float,
                    avg_daily_volume
                FROM short_interest
                WHERE symbol = :symbol
                ORDER BY settlement_date DESC
                LIMIT :periods
            """
            )

            with engine.connect() as conn:
                results = conn.execute(query, {"symbol": symbol, "periods": periods}).fetchall()

                history = []
                prev_short = None

                for row in reversed(results):
                    change = None
                    change_pct = None
                    if prev_short and row[2]:
                        change = row[2] - prev_short
                        if prev_short > 0:
                            change_pct = (change / prev_short) * 100

                    history.append(
                        ShortInterestData(
                            symbol=row[0],
                            settlement_date=row[1],
                            short_interest=row[2] or 0,
                            days_to_cover=float(row[3]) if row[3] else 0.0,
                            short_percent_float=float(row[4]) if row[4] else None,
                            avg_daily_volume=row[5] or 0,
                            previous_short_interest=prev_short,
                            change_from_previous=change,
                            change_percent=change_pct,
                        )
                    )
                    prev_short = row[2]

                return list(reversed(history))

        except Exception as e:
            logger.error(f"Error getting short interest history: {e}")
            return []

    async def get_short_volume(self, symbol: str, days: int = 30) -> List[ShortVolumeData]:
        """Get daily short volume data.

        Args:
            symbol: Stock ticker symbol
            days: Number of trading days

        Returns:
            List of ShortVolumeData sorted by date descending
        """
        symbol = symbol.upper()

        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            query = text(
                """
                SELECT
                    symbol,
                    trade_date,
                    short_volume,
                    short_exempt_volume,
                    total_volume
                FROM short_volume
                WHERE symbol = :symbol
                ORDER BY trade_date DESC
                LIMIT :days
            """
            )

            with engine.connect() as conn:
                results = conn.execute(query, {"symbol": symbol, "days": days}).fetchall()

                return [
                    ShortVolumeData(
                        symbol=row[0],
                        trade_date=row[1],
                        short_volume=row[2] or 0,
                        short_exempt_volume=row[3] or 0,
                        total_volume=row[4] or 0,
                        short_percent=(row[2] / row[4] * 100) if row[4] and row[4] > 0 else 0.0,
                    )
                    for row in results
                ]

        except Exception as e:
            logger.debug(f"Error getting short volume: {e}")
            return []

    async def calculate_squeeze_risk(self, symbol: str) -> ShortSqueezeRisk:
        """Calculate short squeeze risk for a symbol.

        Factors considered:
        - Short percent of float (>20% = high risk)
        - Days to cover (>5 = elevated risk)
        - Short interest trend (increasing = higher risk)
        - Recent price action (price spike + high short = squeeze potential)

        Args:
            symbol: Stock ticker symbol

        Returns:
            ShortSqueezeRisk assessment
        """
        symbol = symbol.upper()

        # Get current short interest
        current = await self.get_short_interest(symbol)
        if not current:
            return ShortSqueezeRisk(
                symbol=symbol,
                squeeze_score=0,
                risk_level="unknown",
                factors=["No short interest data available"],
                interpretation="Unable to assess squeeze risk without data",
            )

        # Get history for trend
        history = await self.get_short_interest_history(symbol, periods=6)

        score = 0.0
        factors = []

        # Factor 1: Short percent of float (0-30 points)
        if current.short_percent_float:
            spf = current.short_percent_float
            if spf >= 40:
                score += 30
                factors.append(f"Extreme short interest: {spf:.1f}% of float")
            elif spf >= 25:
                score += 25
                factors.append(f"Very high short interest: {spf:.1f}% of float")
            elif spf >= 15:
                score += 20
                factors.append(f"High short interest: {spf:.1f}% of float")
            elif spf >= 10:
                score += 15
                factors.append(f"Elevated short interest: {spf:.1f}% of float")
            elif spf >= 5:
                score += 8
                factors.append(f"Moderate short interest: {spf:.1f}% of float")
            else:
                factors.append(f"Low short interest: {spf:.1f}% of float")

        # Factor 2: Days to cover (0-25 points)
        dtc = current.days_to_cover
        if dtc >= 10:
            score += 25
            factors.append(f"Very high days to cover: {dtc:.1f}")
        elif dtc >= 7:
            score += 20
            factors.append(f"High days to cover: {dtc:.1f}")
        elif dtc >= 5:
            score += 15
            factors.append(f"Elevated days to cover: {dtc:.1f}")
        elif dtc >= 3:
            score += 8
            factors.append(f"Moderate days to cover: {dtc:.1f}")
        else:
            factors.append(f"Low days to cover: {dtc:.1f}")

        # Factor 3: Short interest trend (0-25 points)
        if len(history) >= 3:
            changes = [h.change_percent for h in history[-3:] if h.change_percent is not None]
            if changes:
                avg_change = sum(changes) / len(changes)
                if avg_change > 20:
                    score += 25
                    factors.append(f"Rapidly increasing short interest (+{avg_change:.1f}% avg)")
                elif avg_change > 10:
                    score += 18
                    factors.append(f"Increasing short interest (+{avg_change:.1f}% avg)")
                elif avg_change > 5:
                    score += 10
                    factors.append(f"Slightly increasing short interest (+{avg_change:.1f}% avg)")
                elif avg_change < -10:
                    score -= 10
                    factors.append(f"Short covering in progress ({avg_change:.1f}% avg)")
                elif avg_change < -5:
                    score -= 5
                    factors.append(f"Slight short covering ({avg_change:.1f}% avg)")

        # Factor 4: Recent change spike (0-20 points)
        if current.change_percent:
            if current.change_percent > 30:
                score += 20
                factors.append(f"Massive short increase: +{current.change_percent:.1f}%")
            elif current.change_percent > 20:
                score += 15
                factors.append(f"Large short increase: +{current.change_percent:.1f}%")
            elif current.change_percent > 10:
                score += 10
                factors.append(f"Notable short increase: +{current.change_percent:.1f}%")

        # Determine risk level
        score = max(0, min(100, score))  # Clamp to 0-100

        if score >= 75:
            risk_level = "extreme"
            interpretation = (
                "Extreme short squeeze potential. Very high short interest combined with "
                "increasing trend and extended days to cover creates significant squeeze risk. "
                "Any positive catalyst could trigger rapid covering."
            )
        elif score >= 55:
            risk_level = "high"
            interpretation = (
                "High short squeeze potential. Elevated short interest and unfavorable "
                "positioning for shorts. Positive news or price momentum could force covering."
            )
        elif score >= 35:
            risk_level = "elevated"
            interpretation = (
                "Elevated squeeze potential. Short positioning is notable but not extreme. "
                "Monitor for changes in short interest and price action."
            )
        elif score >= 20:
            risk_level = "moderate"
            interpretation = (
                "Moderate squeeze potential. Some short interest but not at concerning levels. "
                "Standard market dynamics likely to prevail."
            )
        else:
            risk_level = "low"
            interpretation = (
                "Low squeeze potential. Short interest is manageable and shorts can cover "
                "without significant price impact."
            )

        return ShortSqueezeRisk(
            symbol=symbol,
            squeeze_score=score,
            risk_level=risk_level,
            factors=factors,
            interpretation=interpretation,
        )

    async def get_most_shorted(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most shorted stocks.

        Args:
            limit: Number of stocks to return

        Returns:
            List of most shorted stocks with key metrics
        """
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            query = text(
                """
                SELECT DISTINCT ON (symbol)
                    symbol,
                    settlement_date,
                    short_interest,
                    days_to_cover,
                    short_percent_float
                FROM short_interest
                WHERE short_percent_float IS NOT NULL
                ORDER BY symbol, settlement_date DESC
            """
            )

            with engine.connect() as conn:
                results = conn.execute(query).fetchall()

                # Sort by short percent of float descending
                sorted_results = sorted(results, key=lambda x: x[4] if x[4] else 0, reverse=True)[:limit]

                return [
                    {
                        "symbol": row[0],
                        "settlement_date": str(row[1]),
                        "short_interest": row[2],
                        "days_to_cover": float(row[3]) if row[3] else 0,
                        "short_percent_float": float(row[4]) if row[4] else 0,
                    }
                    for row in sorted_results
                ]

        except Exception as e:
            logger.error(f"Error getting most shorted stocks: {e}")
            return []


# Singleton instance
_short_interest_fetcher: Optional[ShortInterestFetcher] = None


def get_short_interest_fetcher() -> ShortInterestFetcher:
    """Get or create singleton fetcher instance."""
    global _short_interest_fetcher
    if _short_interest_fetcher is None:
        _short_interest_fetcher = ShortInterestFetcher()
    return _short_interest_fetcher
