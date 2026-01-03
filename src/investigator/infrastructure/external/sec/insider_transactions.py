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

"""SEC Form 4 Insider Transaction Fetcher.

This module provides infrastructure for fetching and parsing SEC Form 3, 4, and 5
insider transaction filings. These forms report:

- Form 3: Initial statement of beneficial ownership
- Form 4: Changes in beneficial ownership (most common)
- Form 5: Annual statement of beneficial ownership

Data Source: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

The SEC provides bulk data in tab-delimited format that can be downloaded quarterly.
This module also supports real-time fetching via SEC EDGAR API.

Example:
    fetcher = InsiderTransactionFetcher()

    # Fetch recent Form 4 filings
    filings = await fetcher.fetch_recent_filings("AAPL", days=30)

    # Parse a Form 4 XML document
    parsed = fetcher.parse_form4_xml(xml_content)
"""

import asyncio
import logging
import ssl
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import aiohttp
import requests

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """SEC Form 4 transaction codes."""
    PURCHASE = "P"           # Open market or private purchase
    SALE = "S"               # Open market or private sale
    GRANT = "A"              # Grant, award, or other acquisition
    DISPOSITION = "D"        # Disposition to issuer
    GIFT = "G"               # Gift
    EXERCISE = "M"           # Exercise or conversion
    CONVERSION = "C"         # Conversion of derivative
    TAX_PAYMENT = "F"        # Payment of tax withholding
    DISCRETIONARY = "I"      # Discretionary transaction
    OTHER_ACQUISITION = "J"  # Other acquisition
    EQUITY_SWAP = "K"        # Equity swap or similar
    SMALL_ACQUISITION = "L"  # Small acquisition under Rule 16a-6
    WILL_TRUST = "W"         # Acquisition or disposition by will or laws of descent
    EXERCISE_OOM = "X"       # Exercise of out-of-money derivative
    OTHER = "Z"              # Other

    @classmethod
    def from_code(cls, code: str) -> Optional["TransactionType"]:
        """Get transaction type from SEC code."""
        for t in cls:
            if t.value == code:
                return t
        return None

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy-side transaction."""
        return self in [
            TransactionType.PURCHASE,
            TransactionType.GRANT,
            TransactionType.EXERCISE,
            TransactionType.OTHER_ACQUISITION,
            TransactionType.SMALL_ACQUISITION,
        ]

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell-side transaction."""
        return self in [
            TransactionType.SALE,
            TransactionType.DISPOSITION,
            TransactionType.GIFT,
            TransactionType.TAX_PAYMENT,
        ]


@dataclass
class InsiderTransaction:
    """Represents a single insider transaction."""
    transaction_date: Optional[date] = None
    transaction_code: Optional[str] = None
    transaction_type: Optional[TransactionType] = None
    shares: float = 0.0
    price_per_share: float = 0.0
    total_value: float = 0.0
    is_derivative: bool = False
    security_title: Optional[str] = None
    acquired_disposed: Optional[str] = None  # "A" or "D"
    post_transaction_shares: Optional[float] = None

    def __post_init__(self):
        if self.transaction_code and not self.transaction_type:
            self.transaction_type = TransactionType.from_code(self.transaction_code)
        if self.shares and self.price_per_share:
            self.total_value = self.shares * self.price_per_share


@dataclass
class ReportingOwner:
    """Represents the insider filing the form."""
    cik: Optional[str] = None
    name: str = ""
    title: str = ""
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    is_other: bool = False

    @property
    def is_key_insider(self) -> bool:
        """Check if this is a key insider (CEO, CFO, Director)."""
        if self.is_director:
            return True

        title_lower = self.title.lower()
        key_titles = [
            "chief executive", "ceo",
            "chief financial", "cfo",
            "chief operating", "coo",
            "chief technology", "cto",
            "president",
        ]

        # Check for C-level but exclude VP
        for key in key_titles:
            if key in title_lower:
                if key == "president" and "vice" in title_lower:
                    continue
                return True

        return False


@dataclass
class Form4Filing:
    """Represents a parsed Form 4 filing."""
    accession_number: str = ""
    filing_date: Optional[date] = None
    issuer_cik: Optional[str] = None
    issuer_name: str = ""
    issuer_symbol: str = ""
    reporting_owner: Optional[ReportingOwner] = None
    transactions: List[InsiderTransaction] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)

    @property
    def total_purchase_value(self) -> float:
        """Total value of purchase transactions."""
        return sum(
            t.total_value for t in self.transactions
            if t.transaction_type and t.transaction_type.is_buy
        )

    @property
    def total_sale_value(self) -> float:
        """Total value of sale transactions."""
        return sum(
            t.total_value for t in self.transactions
            if t.transaction_type and t.transaction_type.is_sell
        )

    @property
    def net_value(self) -> float:
        """Net transaction value (positive = net buying)."""
        return self.total_purchase_value - self.total_sale_value

    @property
    def is_significant(self) -> bool:
        """Check if this filing is significant."""
        # Significant if key insider or large transaction
        if self.reporting_owner and self.reporting_owner.is_key_insider:
            return True
        if abs(self.net_value) > 500000:  # $500K threshold
            return True
        if self.total_purchase_value > 1000000:  # $1M purchase
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accession_number": self.accession_number,
            "filing_date": str(self.filing_date) if self.filing_date else None,
            "issuer": {
                "cik": self.issuer_cik,
                "name": self.issuer_name,
                "symbol": self.issuer_symbol,
            },
            "reporting_owner": {
                "cik": self.reporting_owner.cik if self.reporting_owner else None,
                "name": self.reporting_owner.name if self.reporting_owner else None,
                "title": self.reporting_owner.title if self.reporting_owner else None,
                "is_director": self.reporting_owner.is_director if self.reporting_owner else False,
                "is_officer": self.reporting_owner.is_officer if self.reporting_owner else False,
                "is_ten_percent_owner": self.reporting_owner.is_ten_percent_owner if self.reporting_owner else False,
                "is_key_insider": self.reporting_owner.is_key_insider if self.reporting_owner else False,
            },
            "transactions": [
                {
                    "date": str(t.transaction_date) if t.transaction_date else None,
                    "code": t.transaction_code,
                    "type": t.transaction_type.name if t.transaction_type else None,
                    "shares": t.shares,
                    "price": t.price_per_share,
                    "value": t.total_value,
                    "is_derivative": t.is_derivative,
                    "security_title": t.security_title,
                }
                for t in self.transactions
            ],
            "summary": {
                "total_purchase_value": self.total_purchase_value,
                "total_sale_value": self.total_sale_value,
                "net_value": self.net_value,
                "is_significant": self.is_significant,
                "transaction_count": len(self.transactions),
            },
        }


class InsiderTransactionFetcher:
    """Fetches and parses SEC Form 3/4/5 insider transaction filings.

    Uses the SEC EDGAR API to fetch real-time filings and parses the XML
    content to extract transaction details.
    """

    SEC_BASE_URL = "https://www.sec.gov"
    SEC_DATA_URL = "https://data.sec.gov"
    # SEC requires: AppName/Version (email) - see https://www.sec.gov/os/accessing-edgar-data
    USER_AGENT = "Victor-Invest/1.0 (singhvjd@gmail.com)"

    def __init__(self, rate_limit: float = 0.1):
        """Initialize fetcher with rate limiting.

        Args:
            rate_limit: Minimum seconds between requests (SEC requires 10 req/sec max)
        """
        self._rate_limit = rate_limit
        self._last_request = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept-Encoding": "gzip, deflate",
                },
                connector=connector,
            )
        return self._session

    async def _rate_limited_request(self, url: str) -> Optional[str]:
        """Make a rate-limited request to SEC API."""
        import time

        # Enforce rate limit
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._rate_limit:
            await asyncio.sleep(self._rate_limit - elapsed)

        self._last_request = time.time()

        session = await self._ensure_session()
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"SEC API returned {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _sync_request(self, url: str) -> Optional[str]:
        """Synchronous request for non-async contexts."""
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=30
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    async def fetch_recent_filings(
        self,
        symbol: str,
        days: int = 30,
        cik: Optional[str] = None
    ) -> List[Form4Filing]:
        """Fetch recent Form 4 filings for a symbol.

        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            cik: CIK if known (avoids lookup)

        Returns:
            List of parsed Form4Filing objects
        """
        if not cik:
            cik = await self._get_cik(symbol)
            if not cik:
                logger.warning(f"Could not find CIK for {symbol}")
                return []

        # Fetch submissions from SEC
        padded_cik = cik.zfill(10)
        url = f"{self.SEC_DATA_URL}/submissions/CIK{padded_cik}.json"

        content = await self._rate_limited_request(url)
        if not content:
            return []

        try:
            import json
            data = json.loads(content)

            filings = []
            recent = data.get("filings", {}).get("recent", {})

            if not recent:
                return []

            cutoff_date = datetime.now() - timedelta(days=days)

            form_types = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            accession_numbers = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            for i in range(min(len(form_types), 100)):  # Limit to 100 filings
                if form_types[i] in ["4", "3", "5"]:
                    try:
                        filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d")
                        if filing_date >= cutoff_date:
                            # Fetch and parse the XML
                            accession = accession_numbers[i].replace("-", "")
                            # Use www.sec.gov for XML files (data.sec.gov returns 403)
                            base_url = (
                                f"{self.SEC_BASE_URL}/Archives/edgar/data/"
                                f"{int(cik)}/{accession}"
                            )

                            # Get the actual Form 4 XML (not XSLT-transformed view)
                            xml_url = await self._find_form4_xml_url(
                                base_url, primary_docs[i]
                            )

                            if xml_url:
                                xml_content = await self._rate_limited_request(xml_url)
                                if xml_content:
                                    parsed = self.parse_form4_xml(xml_content)
                                    if parsed:
                                        parsed.accession_number = accession_numbers[i]
                                        parsed.filing_date = filing_date.date()
                                        filings.append(parsed)
                    except Exception as e:
                        logger.debug(f"Error parsing filing {i}: {e}")
                        continue

            logger.info(f"Found {len(filings)} Form 3/4/5 filings for {symbol}")
            return filings

        except Exception as e:
            logger.error(f"Error processing submissions for {symbol}: {e}")
            return []

    def parse_form4_xml(self, xml_content: str) -> Optional[Form4Filing]:
        """Parse Form 4 XML content.

        Args:
            xml_content: XML content of Form 4

        Returns:
            Parsed Form4Filing or None if parsing fails
        """
        try:
            # Handle namespace issues
            xml_content = xml_content.replace('xmlns="', 'ns="')

            root = ET.fromstring(xml_content)
            filing = Form4Filing()

            # Parse issuer information
            issuer = root.find(".//issuer")
            if issuer is not None:
                filing.issuer_cik = self._get_xml_text(issuer, "issuerCik")
                filing.issuer_name = self._get_xml_text(issuer, "issuerName")
                filing.issuer_symbol = self._get_xml_text(issuer, "issuerTradingSymbol")

            # Parse reporting owner
            reporting_owner_elem = root.find(".//reportingOwner")
            if reporting_owner_elem is not None:
                owner = ReportingOwner()

                owner_id = reporting_owner_elem.find(".//reportingOwnerId")
                if owner_id is not None:
                    owner.cik = self._get_xml_text(owner_id, "rptOwnerCik")
                    owner.name = self._get_xml_text(owner_id, "rptOwnerName")

                relationship = reporting_owner_elem.find(".//reportingOwnerRelationship")
                if relationship is not None:
                    owner.is_director = self._get_xml_text(relationship, "isDirector") == "1"
                    owner.is_officer = self._get_xml_text(relationship, "isOfficer") == "1"
                    owner.is_ten_percent_owner = self._get_xml_text(relationship, "isTenPercentOwner") == "1"
                    owner.is_other = self._get_xml_text(relationship, "isOther") == "1"
                    owner.title = self._get_xml_text(relationship, "officerTitle")

                filing.reporting_owner = owner

            # Parse non-derivative transactions
            for trans_elem in root.findall(".//nonDerivativeTransaction"):
                trans = self._parse_transaction(trans_elem, is_derivative=False)
                if trans:
                    filing.transactions.append(trans)

            # Parse derivative transactions
            for trans_elem in root.findall(".//derivativeTransaction"):
                trans = self._parse_transaction(trans_elem, is_derivative=True)
                if trans:
                    filing.transactions.append(trans)

            # Parse footnotes
            for footnote in root.findall(".//footnote"):
                text = footnote.text
                if text:
                    filing.footnotes.append(text.strip())

            return filing

        except Exception as e:
            logger.error(f"Error parsing Form 4 XML: {e}")
            return None

    def _parse_transaction(
        self,
        elem: ET.Element,
        is_derivative: bool = False
    ) -> Optional[InsiderTransaction]:
        """Parse a single transaction element."""
        try:
            trans = InsiderTransaction(is_derivative=is_derivative)

            # Transaction date
            date_str = self._get_xml_text(elem, ".//transactionDate/value")
            if date_str:
                trans.transaction_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Transaction code
            trans.transaction_code = self._get_xml_text(elem, ".//transactionCoding/transactionCode")
            if trans.transaction_code:
                trans.transaction_type = TransactionType.from_code(trans.transaction_code)

            # Transaction amounts
            amounts = elem.find(".//transactionAmounts")
            if amounts is not None:
                shares_str = self._get_xml_text(amounts, "transactionShares/value")
                price_str = self._get_xml_text(amounts, "transactionPricePerShare/value")
                acq_disp = self._get_xml_text(amounts, "transactionAcquiredDisposedCode/value")

                trans.shares = float(shares_str) if shares_str else 0.0
                trans.price_per_share = float(price_str) if price_str else 0.0
                trans.acquired_disposed = acq_disp
                trans.total_value = trans.shares * trans.price_per_share

            # Security title (for derivatives)
            if is_derivative:
                trans.security_title = self._get_xml_text(elem, ".//securityTitle/value")

            # Post-transaction holdings
            post_shares = self._get_xml_text(elem, ".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
            if post_shares:
                trans.post_transaction_shares = float(post_shares)

            return trans

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")
            return None

    def _get_xml_text(self, elem: ET.Element, path: str, default: str = "") -> str:
        """Safely get text from XML element."""
        if elem is None:
            return default

        found = elem.find(path) if path.startswith(".//") else elem.find(".//" + path)
        if found is not None and found.text:
            return found.text.strip()

        return default

    async def _find_form4_xml_url(
        self, base_url: str, primary_doc: str
    ) -> Optional[str]:
        """Find the actual Form 4 XML file URL.

        The primaryDocument field often points to XSLT-transformed views
        (e.g., xslF345X05/wk-form4_xxx.xml). The actual XML file has the
        same name but is in the root of the filing directory.

        Args:
            base_url: Base URL for the filing (without filename)
            primary_doc: Primary document path from submissions

        Returns:
            URL to the actual Form 4 XML file, or None if not found
        """
        # If primary doc doesn't start with xsl, it is the actual XML
        if not primary_doc.startswith("xsl"):
            return f"{base_url}/{primary_doc}"

        # Extract filename from XSLT path
        # e.g., "xslF345X05/wk-form4_1763163012.xml" -> "wk-form4_1763163012.xml"
        if "/" in primary_doc:
            xml_filename = primary_doc.split("/")[-1]
            return f"{base_url}/{xml_filename}"

        return None

    async def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK for a symbol from SEC ticker mapping."""
        try:
            # Use SEC's ticker.txt mapping
            url = "https://www.sec.gov/include/ticker.txt"
            content = await self._rate_limited_request(url)
            if content:
                for line in content.strip().split("\n"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2 and parts[0].upper() == symbol.upper():
                        return parts[1].zfill(10)
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")

        return None

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance
_insider_fetcher: Optional[InsiderTransactionFetcher] = None


def get_insider_transaction_fetcher() -> InsiderTransactionFetcher:
    """Get or create singleton fetcher instance."""
    global _insider_fetcher
    if _insider_fetcher is None:
        _insider_fetcher = InsiderTransactionFetcher()
    return _insider_fetcher


# Alias for scheduled collector compatibility
get_insider_fetcher = get_insider_transaction_fetcher
