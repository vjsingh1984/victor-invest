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

"""SEC Form 13F Institutional Holdings Fetcher.

This module fetches institutional holdings data from SEC EDGAR Form 13F filings.
Form 13F is filed quarterly by institutional investment managers with over
$100M in qualifying assets under management.

Data Source: https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets

Key Concepts:
- CUSIP: 9-character identifier for securities
- 13F-HR: Holdings Report (quarterly filing)
- Reporting Period: End of calendar quarter

Investment Signals:
- Increasing institutional ownership: Bullish signal
- Decreasing institutional ownership: Bearish signal
- New positions by top funds: Often precede price appreciation
- Cluster buying by multiple institutions: Strong bullish signal

Example:
    fetcher = get_institutional_holdings_fetcher()

    # Get holdings for a symbol
    holdings = await fetcher.get_holdings_by_symbol("AAPL")

    # Get top holders
    top_holders = await fetcher.get_top_holders("AAPL", limit=20)

    # Get ownership changes
    changes = await fetcher.get_ownership_changes("AAPL", quarters=4)
"""

import asyncio
import csv
import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# SEC EDGAR base URLs
SEC_EDGAR_BASE = "https://www.sec.gov"
FORM_13F_DATA_URL = "https://www.sec.gov/files/structureddata/data/13f-data-sets"


@dataclass
class InstitutionalHolder:
    """Represents an institutional holder from 13F filing.

    Attributes:
        cik: SEC Central Index Key of the filer
        name: Institution name
        filing_date: Date of 13F filing
        report_date: Quarter end date reported
        form_type: Filing type (13F-HR, 13F-HR/A)
    """
    cik: str
    name: str
    filing_date: Optional[date] = None
    report_date: Optional[date] = None
    form_type: str = "13F-HR"
    total_value: float = 0.0
    num_holdings: int = 0


@dataclass
class Holding:
    """Individual holding from 13F filing.

    Attributes:
        cusip: 9-character CUSIP identifier
        symbol: Ticker symbol (if mapped)
        issuer_name: Name of the issuer
        class_title: Class of security (e.g., "COM", "CL A")
        shares: Number of shares held
        value: Value in thousands of dollars
        investment_discretion: SOLE, SHARED, or NONE
        voting_authority_sole: Sole voting authority shares
        voting_authority_shared: Shared voting authority shares
        voting_authority_none: No voting authority shares
    """
    cusip: str
    symbol: Optional[str] = None
    issuer_name: str = ""
    class_title: str = ""
    shares: int = 0
    value: float = 0.0  # In thousands
    investment_discretion: str = "SOLE"
    voting_authority_sole: int = 0
    voting_authority_shared: int = 0
    voting_authority_none: int = 0
    put_call: Optional[str] = None  # PUT, CALL, or None

    @property
    def value_dollars(self) -> float:
        """Value in dollars (not thousands)."""
        return self.value * 1000


@dataclass
class InstitutionalOwnership:
    """Aggregated institutional ownership for a symbol.

    Attributes:
        symbol: Stock ticker symbol
        report_quarter: Quarter end date
        total_shares: Total shares held by institutions
        total_value: Total value held (in thousands)
        num_institutions: Number of institutional holders
        top_holders: List of top institutional holders
        ownership_pct: Institutional ownership percentage (if known)
        qoq_change_pct: Quarter-over-quarter change in shares
    """
    symbol: str
    report_quarter: date
    total_shares: int = 0
    total_value: float = 0.0
    num_institutions: int = 0
    top_holders: List[Dict[str, Any]] = field(default_factory=list)
    ownership_pct: Optional[float] = None
    qoq_change_pct: Optional[float] = None
    qoq_change_shares: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "report_quarter": str(self.report_quarter),
            "total_shares": self.total_shares,
            "total_value_thousands": self.total_value,
            "total_value_dollars": self.total_value * 1000,
            "num_institutions": self.num_institutions,
            "ownership_pct": self.ownership_pct,
            "changes": {
                "qoq_change_pct": self.qoq_change_pct,
                "qoq_change_shares": self.qoq_change_shares,
            },
            "top_holders": self.top_holders[:10],
        }


class CUSIPMapper:
    """Maps CUSIP identifiers to ticker symbols.

    Uses multiple sources to map CUSIPs:
    1. Local database cache
    2. SEC company tickers list
    3. OpenFIGI API (if available)
    """

    def __init__(self):
        """Initialize CUSIP mapper."""
        self._cache: Dict[str, str] = {}
        self._reverse_cache: Dict[str, str] = {}  # Symbol -> CUSIP

    async def get_symbol(self, cusip: str) -> Optional[str]:
        """Get ticker symbol for a CUSIP.

        Args:
            cusip: 9-character CUSIP identifier

        Returns:
            Ticker symbol or None if not found
        """
        # Check cache first
        if cusip in self._cache:
            return self._cache[cusip]

        # Try database lookup
        symbol = await self._lookup_from_database(cusip)
        if symbol:
            self._cache[cusip] = symbol
            return symbol

        return None

    async def get_cusip(self, symbol: str) -> Optional[str]:
        """Get CUSIP for a ticker symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CUSIP or None if not found
        """
        symbol = symbol.upper()

        # Check reverse cache
        if symbol in self._reverse_cache:
            return self._reverse_cache[symbol]

        # Try database lookup
        cusip = await self._lookup_cusip_from_database(symbol)
        if cusip:
            self._reverse_cache[symbol] = cusip
            self._cache[cusip] = symbol
            return cusip

        return None

    async def _lookup_from_database(self, cusip: str) -> Optional[str]:
        """Look up CUSIP in local database."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database"
            )

            query = text("""
                SELECT ticker FROM cusip_mapping
                WHERE cusip = :cusip
                LIMIT 1
            """)

            with engine.connect() as conn:
                result = conn.execute(query, {"cusip": cusip}).fetchone()
                if result:
                    return result[0]

        except Exception as e:
            logger.debug(f"Database CUSIP lookup failed: {e}")

        return None

    async def _lookup_cusip_from_database(self, symbol: str) -> Optional[str]:
        """Look up ticker in local database to get CUSIP."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database"
            )

            query = text("""
                SELECT cusip FROM cusip_mapping
                WHERE ticker = :symbol
                LIMIT 1
            """)

            with engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol}).fetchone()
                if result:
                    return result[0]

        except Exception as e:
            logger.debug(f"Database ticker lookup failed: {e}")

        return None


class InstitutionalHoldingsFetcher:
    """Fetches institutional holdings from SEC EDGAR 13F filings.

    Provides access to quarterly institutional holdings data including:
    - Holdings by symbol (which institutions hold a stock)
    - Holdings by institution (what stocks an institution holds)
    - Ownership changes over time
    - Top holders for a symbol

    SOLID: Single Responsibility - only handles 13F data fetching
    """

    def __init__(self, timeout: int = 60):
        """Initialize fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cusip_mapper = CUSIPMapper()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "User-Agent": "Victor-Invest/1.0 (Investment Research; contact@example.com)",
                "Accept": "application/json, text/html, text/csv, */*",
            }
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_holdings_by_symbol(
        self,
        symbol: str,
        quarter: Optional[str] = None
    ) -> InstitutionalOwnership:
        """Get institutional holdings for a symbol.

        Args:
            symbol: Stock ticker symbol
            quarter: Quarter in format "YYYY-Q#" (e.g., "2024-Q4"), defaults to latest

        Returns:
            InstitutionalOwnership with aggregated data
        """
        symbol = symbol.upper()

        # Get CUSIP for symbol
        cusip = await self._cusip_mapper.get_cusip(symbol)

        # Try database first
        ownership = await self._get_holdings_from_database(symbol, cusip, quarter)
        if ownership and ownership.num_institutions > 0:
            return ownership

        # Return empty ownership if no data
        return InstitutionalOwnership(
            symbol=symbol,
            report_quarter=self._get_quarter_date(quarter),
        )

    async def _get_holdings_from_database(
        self,
        symbol: str,
        cusip: Optional[str],
        quarter: Optional[str]
    ) -> Optional[InstitutionalOwnership]:
        """Get holdings from local database."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database"
            )

            # Build query based on available identifiers
            if cusip:
                query = text("""
                    SELECT
                        filer_name,
                        shares,
                        value_thousands,
                        report_date
                    FROM form13f_holdings h
                    JOIN form13f_filers f ON h.filer_cik = f.cik
                    WHERE h.cusip = :cusip
                    AND h.report_date = (
                        SELECT MAX(report_date) FROM form13f_holdings
                        WHERE cusip = :cusip
                    )
                    ORDER BY value_thousands DESC
                    LIMIT 50
                """)
                params = {"cusip": cusip}
            else:
                # Try matching by issuer name
                query = text("""
                    SELECT
                        filer_name,
                        shares,
                        value_thousands,
                        report_date
                    FROM form13f_holdings h
                    JOIN form13f_filers f ON h.filer_cik = f.cik
                    WHERE UPPER(h.issuer_name) LIKE :symbol_pattern
                    AND h.report_date = (
                        SELECT MAX(report_date) FROM form13f_holdings
                        WHERE UPPER(issuer_name) LIKE :symbol_pattern
                    )
                    ORDER BY value_thousands DESC
                    LIMIT 50
                """)
                params = {"symbol_pattern": f"%{symbol}%"}

            with engine.connect() as conn:
                results = conn.execute(query, params).fetchall()

                if not results:
                    return None

                total_shares = sum(r[1] or 0 for r in results)
                total_value = sum(r[2] or 0 for r in results)
                report_date = results[0][3] if results else date.today()

                top_holders = [
                    {
                        "name": r[0],
                        "shares": r[1],
                        "value_thousands": r[2],
                    }
                    for r in results[:20]
                ]

                return InstitutionalOwnership(
                    symbol=symbol,
                    report_quarter=report_date,
                    total_shares=total_shares,
                    total_value=total_value,
                    num_institutions=len(results),
                    top_holders=top_holders,
                )

        except Exception as e:
            logger.debug(f"Database holdings lookup failed: {e}")
            return None

    async def get_top_holders(
        self,
        symbol: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top institutional holders for a symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of holders to return

        Returns:
            List of top holders with shares and values
        """
        ownership = await self.get_holdings_by_symbol(symbol)
        return ownership.top_holders[:limit]

    async def get_ownership_changes(
        self,
        symbol: str,
        quarters: int = 4
    ) -> List[Dict[str, Any]]:
        """Get ownership changes over multiple quarters.

        Args:
            symbol: Stock ticker symbol
            quarters: Number of quarters to analyze

        Returns:
            List of quarterly changes
        """
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database"
            )

            # Get CUSIP
            cusip = await self._cusip_mapper.get_cusip(symbol)
            if not cusip:
                return []

            query = text("""
                SELECT
                    report_date,
                    SUM(shares) as total_shares,
                    SUM(value_thousands) as total_value,
                    COUNT(DISTINCT filer_cik) as num_institutions
                FROM form13f_holdings
                WHERE cusip = :cusip
                GROUP BY report_date
                ORDER BY report_date DESC
                LIMIT :quarters
            """)

            with engine.connect() as conn:
                results = conn.execute(query, {
                    "cusip": cusip,
                    "quarters": quarters
                }).fetchall()

                changes = []
                prev_shares = None

                for r in reversed(results):
                    change = {
                        "quarter": str(r[0]),
                        "total_shares": r[1],
                        "total_value_thousands": r[2],
                        "num_institutions": r[3],
                    }

                    if prev_shares is not None and prev_shares > 0:
                        change["qoq_change_pct"] = round(
                            ((r[1] - prev_shares) / prev_shares) * 100, 2
                        )
                        change["qoq_change_shares"] = r[1] - prev_shares

                    changes.append(change)
                    prev_shares = r[1]

                return list(reversed(changes))

        except Exception as e:
            logger.error(f"Error getting ownership changes: {e}")
            return []

    async def get_institution_holdings(
        self,
        institution_cik: str,
        quarter: Optional[str] = None
    ) -> List[Holding]:
        """Get all holdings for an institution.

        Args:
            institution_cik: CIK of the institution
            quarter: Quarter in format "YYYY-Q#"

        Returns:
            List of holdings
        """
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database"
            )

            query = text("""
                SELECT
                    cusip,
                    issuer_name,
                    class_title,
                    shares,
                    value_thousands,
                    investment_discretion,
                    put_call
                FROM form13f_holdings
                WHERE filer_cik = :cik
                AND report_date = (
                    SELECT MAX(report_date) FROM form13f_holdings
                    WHERE filer_cik = :cik
                )
                ORDER BY value_thousands DESC
            """)

            with engine.connect() as conn:
                results = conn.execute(query, {"cik": institution_cik}).fetchall()

                holdings = []
                for r in results:
                    symbol = await self._cusip_mapper.get_symbol(r[0])
                    holdings.append(Holding(
                        cusip=r[0],
                        symbol=symbol,
                        issuer_name=r[1],
                        class_title=r[2],
                        shares=r[3] or 0,
                        value=r[4] or 0,
                        investment_discretion=r[5] or "SOLE",
                        put_call=r[6],
                    ))

                return holdings

        except Exception as e:
            logger.error(f"Error getting institution holdings: {e}")
            return []

    async def search_institutions(
        self,
        query: str,
        limit: int = 20
    ) -> List[InstitutionalHolder]:
        """Search for institutions by name.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching institutions
        """
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database"
            )

            sql = text("""
                SELECT DISTINCT
                    cik,
                    name,
                    MAX(filing_date) as latest_filing
                FROM form13f_filers
                WHERE LOWER(name) LIKE :query
                GROUP BY cik, name
                ORDER BY latest_filing DESC
                LIMIT :limit
            """)

            with engine.connect() as conn:
                results = conn.execute(sql, {
                    "query": f"%{query.lower()}%",
                    "limit": limit
                }).fetchall()

                return [
                    InstitutionalHolder(
                        cik=r[0],
                        name=r[1],
                        filing_date=r[2],
                    )
                    for r in results
                ]

        except Exception as e:
            logger.error(f"Error searching institutions: {e}")
            return []

    def _get_quarter_date(self, quarter: Optional[str] = None) -> date:
        """Get quarter end date from quarter string.

        Args:
            quarter: Quarter string like "2024-Q4" or None for latest

        Returns:
            Quarter end date
        """
        if quarter:
            match = re.match(r"(\d{4})-Q(\d)", quarter)
            if match:
                year = int(match.group(1))
                q = int(match.group(2))
                month = q * 3
                if month == 3:
                    return date(year, 3, 31)
                elif month == 6:
                    return date(year, 6, 30)
                elif month == 9:
                    return date(year, 9, 30)
                else:
                    return date(year, 12, 31)

        # Default to end of last complete quarter
        today = date.today()
        if today.month <= 3:
            return date(today.year - 1, 12, 31)
        elif today.month <= 6:
            return date(today.year, 3, 31)
        elif today.month <= 9:
            return date(today.year, 6, 30)
        else:
            return date(today.year, 9, 30)


# Singleton instance
_holdings_fetcher: Optional[InstitutionalHoldingsFetcher] = None


def get_institutional_holdings_fetcher() -> InstitutionalHoldingsFetcher:
    """Get or create singleton fetcher instance."""
    global _holdings_fetcher
    if _holdings_fetcher is None:
        _holdings_fetcher = InstitutionalHoldingsFetcher()
    return _holdings_fetcher
