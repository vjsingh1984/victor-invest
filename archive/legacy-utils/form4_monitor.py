"""
Form 4 Monitor Module

Monitors and parses SEC Form 4 insider trading filings.
Detects significant insider activity and generates alerts.
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class Form4Monitor:
    """Monitor SEC Form 4 insider trading filings"""

    def __init__(self, db_manager=None):
        """
        Initialize Form 4 monitor

        Args:
            db_manager: Database manager for storing filings
        """
        self.db_manager = db_manager
        self.sec_base_url = "https://www.sec.gov"
        self.sec_api_base = "https://data.sec.gov"

        # Transaction code mappings
        self.transaction_codes = {
            "P": "Purchase",
            "S": "Sale",
            "A": "Grant",
            "D": "Disposition",
            "G": "Gift",
            "M": "Exercise",
            "C": "Conversion",
            "F": "Tax Payment",
            "I": "Discretionary",
            "J": "Other",
            "K": "Equity Swap",
            "L": "Small Acquisition",
            "W": "Will or Trust",
            "X": "Exercise Out of Money",
            "Z": "Other",
        }

    def fetch_recent_filings(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch recent Form 4 filings for a symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of filing information
        """
        try:
            # Get CIK for symbol
            cik = self._get_cik(symbol)
            if not cik:
                logger.warning(f"Could not find CIK for {symbol}")
                return []

            # Fetch submissions from SEC
            logger.info(f"Fetching Form 4 filings for {symbol} (CIK: {cik})")

            headers = {
                "User-Agent": "InvestiGator Investment Analysis System (contact@example.com)",
                "Accept-Encoding": "gzip, deflate",
            }

            # Get recent filings via SEC API
            url = f"{self.sec_api_base}/submissions/CIK{cik}.json"

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Filter for Form 4 filings within date range
            filings = []
            recent = data.get("filings", {}).get("recent", {})

            if not recent:
                logger.info(f"No recent filings found for {symbol}")
                return []

            cutoff_date = datetime.now() - timedelta(days=days)

            form_types = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            accession_numbers = recent.get("accessionNumber", [])

            for i in range(len(form_types)):
                if form_types[i] == "4":
                    filing_date_str = filing_dates[i]
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")

                    if filing_date >= cutoff_date:
                        filings.append(
                            {
                                "accession_number": accession_numbers[i],
                                "filing_date": filing_date_str,
                                "cik": cik,
                                "symbol": symbol,
                            }
                        )

            logger.info(f"Found {len(filings)} Form 4 filings for {symbol}")
            return filings

        except Exception as e:
            logger.error(f"Error fetching Form 4 filings for {symbol}: {e}")
            return []

    def parse_form4_xml(self, xml_content: str) -> Optional[Dict]:
        """
        Parse Form 4 XML content

        Args:
            xml_content: XML content of Form 4

        Returns:
            Dictionary with parsed filing data
        """
        try:
            # Remove namespace prefixes for easier parsing
            xml_content = xml_content.replace("xmlns=", "ns=")

            root = ET.fromstring(xml_content)

            result = {"issuer": {}, "reporting_owner": {}, "transactions": []}

            # Parse issuer information
            issuer = root.find(".//issuer")
            if issuer is not None:
                result["issuer"] = {
                    "cik": self._get_text(issuer, "issuerCik"),
                    "name": self._get_text(issuer, "issuerName"),
                    "symbol": self._get_text(issuer, "issuerTradingSymbol"),
                }

            # Parse reporting owner
            reporting_owner = root.find(".//reportingOwner")
            if reporting_owner is not None:
                owner_id = reporting_owner.find(".//reportingOwnerId")
                relationship = reporting_owner.find(".//reportingOwnerRelationship")

                result["reporting_owner"] = {
                    "name": self._get_text(owner_id, "rptOwnerName") if owner_id is not None else "",
                    "is_director": (
                        self._get_text(relationship, "isDirector") == "1" if relationship is not None else False
                    ),
                    "is_officer": (
                        self._get_text(relationship, "isOfficer") == "1" if relationship is not None else False
                    ),
                    "is_ten_percent_owner": (
                        self._get_text(relationship, "isTenPercentOwner") == "1" if relationship is not None else False
                    ),
                    "title": self._get_text(relationship, "officerTitle") if relationship is not None else "",
                }

            # Parse non-derivative transactions
            for transaction in root.findall(".//nonDerivativeTransaction"):
                parsed = self._parse_transaction(transaction)
                if parsed:
                    result["transactions"].append(parsed)

            # Parse derivative transactions (options, RSUs, etc.)
            for transaction in root.findall(".//derivativeTransaction"):
                parsed = self._parse_derivative_transaction(transaction)
                if parsed:
                    result["transactions"].append(parsed)

            return result

        except Exception as e:
            logger.error(f"Error parsing Form 4 XML: {e}")
            return None

    def _parse_transaction(self, transaction_elem) -> Optional[Dict]:
        """Parse a single non-derivative transaction"""
        try:
            transaction_date = self._get_text(transaction_elem, ".//transactionDate/value")
            transaction_code = self._get_text(transaction_elem, ".//transactionCoding/transactionCode")

            amounts = transaction_elem.find(".//transactionAmounts")
            if amounts is None:
                return None

            shares_str = self._get_text(amounts, "transactionShares/value")
            price_str = self._get_text(amounts, "transactionPricePerShare/value")

            return {
                "transaction_date": transaction_date,
                "transaction_code": transaction_code,
                "transaction_type": self._classify_transaction_type(transaction_code),
                "shares": float(shares_str) if shares_str else 0,
                "price_per_share": float(price_str) if price_str else 0,
                "is_derivative": False,
            }

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")
            return None

    def _parse_derivative_transaction(self, transaction_elem) -> Optional[Dict]:
        """Parse a derivative transaction (options, RSUs)"""
        try:
            transaction_date = self._get_text(transaction_elem, ".//transactionDate/value")
            transaction_code = self._get_text(transaction_elem, ".//transactionCoding/transactionCode")
            security_title = self._get_text(transaction_elem, ".//securityTitle/value")

            amounts = transaction_elem.find(".//transactionAmounts")
            if amounts is None:
                return None

            shares_str = self._get_text(amounts, "transactionShares/value")

            return {
                "transaction_date": transaction_date,
                "transaction_code": transaction_code,
                "transaction_type": self._classify_transaction_type(transaction_code),
                "shares": float(shares_str) if shares_str else 0,
                "security_type": security_title,
                "is_derivative": True,
            }

        except Exception as e:
            logger.debug(f"Error parsing derivative transaction: {e}")
            return None

    def _get_text(self, element, path: str, default: str = "") -> str:
        """Safely get text from XML element"""
        if element is None:
            return default

        if path.startswith(".//"):
            found = element.find(path)
        else:
            found = element.find(".//" + path)

        if found is not None and found.text:
            return found.text.strip()

        return default

    def _classify_transaction_type(self, code: str) -> str:
        """Classify transaction type from code"""
        return self.transaction_codes.get(code, "Unknown")

    def assess_filing_significance(self, filing: Dict) -> Dict:
        """
        Assess the significance of a Form 4 filing

        Args:
            filing: Parsed filing data

        Returns:
            Significance assessment
        """
        reasons = []
        is_significant = False

        # Check if key insider
        reporting_owner = filing.get("reporting_owner", {})
        is_key = self._is_key_insider(
            reporting_owner.get("title", ""),
            reporting_owner.get("is_officer", False),
            reporting_owner.get("is_director", False),
        )

        if is_key:
            reasons.append("Key insider (CEO/CFO/Director)")
            is_significant = True

        # Check transaction size
        transactions = filing.get("transactions", [])
        total_value = self._calculate_transaction_value(transactions)

        if total_value > 1000000:  # $1M threshold
            reasons.append(f"Large transaction value: ${total_value:,.0f}")
            is_significant = True

        # Check for purchases (bullish signal)
        purchases = [t for t in transactions if t.get("transaction_code") == "P"]
        if purchases:
            total_purchase_value = sum(t.get("shares", 0) * t.get("price_per_share", 0) for t in purchases)
            if total_purchase_value > 500000:  # $500k threshold
                reasons.append(f"Significant insider purchase: ${total_purchase_value:,.0f}")
                is_significant = True

        return {
            "is_significant": is_significant,
            "reasons": reasons,
            "total_value": total_value,
            "is_key_insider": is_key,
        }

    def _is_key_insider(self, title: str, is_officer: bool, is_director: bool) -> bool:
        """Determine if insider is a key person"""
        if is_director:
            return True

        title_lower = title.lower()

        # Check for C-level executives
        c_level_titles = [
            "chief executive",
            "ceo",
            "chief financial",
            "cfo",
            "chief operating",
            "coo",
            "chief technology",
            "cto",
        ]

        if any(key in title_lower for key in c_level_titles):
            return True

        # President (but not Vice President)
        if "president" in title_lower and "vice" not in title_lower:
            return True

        return False

    def _calculate_transaction_value(self, transactions: List[Dict]) -> float:
        """Calculate total value of transactions"""
        total = 0
        for t in transactions:
            if not t.get("is_derivative", False):  # Only count actual stock transactions
                shares = t.get("shares", 0)
                price = t.get("price_per_share", 0)
                total += shares * price
        return total

    def _detect_purchase_cluster(self, filings: List[Dict], days: int = 30) -> bool:
        """Detect cluster of insider purchases"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_purchases = 0

        for filing in filings:
            filing_date_str = filing.get("filing_date", "")
            if filing_date_str:
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                if filing_date >= cutoff:
                    transactions = filing.get("transactions", [])
                    purchases = [t for t in transactions if t.get("transaction_code") == "P"]
                    if purchases:
                        recent_purchases += 1

        return recent_purchases >= 3  # 3+ purchases in window

    def generate_insider_alert(self, filing: Dict) -> Optional[Dict]:
        """
        Generate alert for significant insider activity

        Args:
            filing: Parsed Form 4 filing

        Returns:
            Alert dictionary if significant
        """
        significance = self.assess_filing_significance(filing)

        if not significance["is_significant"]:
            return None

        issuer = filing.get("issuer", {})
        reporting_owner = filing.get("reporting_owner", {})
        transactions = filing.get("transactions", [])

        # Determine primary transaction type
        purchase_value = sum(
            t.get("shares", 0) * t.get("price_per_share", 0) for t in transactions if t.get("transaction_code") == "P"
        )
        sale_value = sum(
            t.get("shares", 0) * t.get("price_per_share", 0) for t in transactions if t.get("transaction_code") == "S"
        )

        if purchase_value > sale_value:
            action = "Purchase"
            value = purchase_value
            severity = "medium"  # Purchases are generally positive
        else:
            action = "Sale"
            value = sale_value
            severity = "high"  # Sales can be concerning

        message = f"{reporting_owner.get('name', 'Insider')} ({reporting_owner.get('title', 'Unknown')}) - {action} of ${value:,.0f}"

        return {
            "symbol": issuer.get("symbol"),
            "type": "insider_transaction",
            "severity": severity,
            "message": message,
            "details": {
                "insider_name": reporting_owner.get("name"),
                "insider_title": reporting_owner.get("title"),
                "transaction_type": action,
                "total_value": value,
                "filing_date": filing.get("filing_date"),
                "reasons": significance["reasons"],
            },
            "timestamp": datetime.now(),
        }

    def save_filing(self, filing: Dict) -> bool:
        """
        Save Form 4 filing to database

        Args:
            filing: Parsed filing data

        Returns:
            True if saved successfully
        """
        if not self.db_manager:
            return False

        try:
            from sqlalchemy import text

            query = text(
                """
                INSERT INTO form4_filings (
                    symbol, cik, filing_date, accession_number,
                    reporting_owner_name, reporting_owner_title,
                    transaction_data, is_significant, created_at
                )
                VALUES (
                    :symbol, :cik, :filing_date, :accession_number,
                    :owner_name, :owner_title,
                    :transaction_data, :is_significant, :created_at
                )
                ON CONFLICT (accession_number) DO NOTHING
            """
            )

            issuer = filing.get("issuer", {})
            reporting_owner = filing.get("reporting_owner", {})
            significance = self.assess_filing_significance(filing)

            with self.db_manager.get_session() as session:
                session.execute(
                    query,
                    {
                        "symbol": issuer.get("symbol"),
                        "cik": issuer.get("cik"),
                        "filing_date": filing.get("filing_date"),
                        "accession_number": filing.get("accession_number"),
                        "owner_name": reporting_owner.get("name"),
                        "owner_title": reporting_owner.get("title"),
                        "transaction_data": str(filing.get("transactions", [])),
                        "is_significant": significance["is_significant"],
                        "created_at": datetime.now(),
                    },
                )
                session.commit()

            logger.info(f"Saved Form 4 filing: {filing.get('accession_number')}")
            return True

        except Exception as e:
            logger.error(f"Error saving Form 4 filing: {e}")
            return False

    def get_recent_insider_activity(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Get recent insider activity for a symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of recent filings
        """
        if not self.db_manager:
            return []

        try:
            from sqlalchemy import text

            cutoff_date = datetime.now() - timedelta(days=days)

            query = text(
                """
                SELECT filing_date, reporting_owner_name, reporting_owner_title,
                       transaction_data, is_significant
                FROM form4_filings
                WHERE symbol = :symbol
                AND filing_date >= :cutoff_date
                ORDER BY filing_date DESC
            """
            )

            with self.db_manager.get_session() as session:
                results = session.execute(
                    query, {"symbol": symbol, "cutoff_date": cutoff_date.strftime("%Y-%m-%d")}
                ).fetchall()

                filings = []
                for row in results:
                    filings.append(
                        {
                            "filing_date": row[0],
                            "reporting_owner_name": row[1],
                            "reporting_owner_title": row[2],
                            "transaction_data": row[3],
                            "is_significant": row[4],
                        }
                    )

                return filings

        except Exception as e:
            logger.error(f"Error retrieving insider activity: {e}")
            return []

    def monitor_symbols(self, symbols: List[str], days: int = 1) -> Dict[str, List[Dict]]:
        """
        Monitor multiple symbols for Form 4 filings

        Args:
            symbols: List of stock symbols
            days: Number of days to look back

        Returns:
            Dictionary mapping symbols to their filings
        """
        results = {}

        for symbol in symbols:
            try:
                filings = self.fetch_recent_filings(symbol, days=days)
                results[symbol] = filings
                logger.info(f"{symbol}: Found {len(filings)} Form 4 filings")
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                results[symbol] = []

        return results

    def _get_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK for symbol from database

        Args:
            symbol: Stock symbol

        Returns:
            CIK string or None
        """
        if not self.db_manager:
            return None

        try:
            from sqlalchemy import text

            query = text("SELECT cik FROM ticker_cik_mapping WHERE ticker = :symbol")

            with self.db_manager.get_session() as session:
                result = session.execute(query, {"symbol": symbol}).fetchone()
                if result:
                    return str(result[0]).zfill(10)  # Pad to 10 digits

        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")

        return None
