"""
SEC API Client Module
Handles interactions with SEC EDGAR API
"""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests

from investigator.config import get_config
from investigator.infrastructure.cache.cache_manager import get_cache_manager
from investigator.infrastructure.cache.cache_types import CacheType

# Lazy imports to avoid circular dependency:
# sec_api -> utils.submission_processor -> application -> domain.agents.sec -> sec
# These are imported inside methods or at first use
if TYPE_CHECKING:
    from utils.api_client import SECAPIClient as LegacySECAPIClient
    from utils.submission_processor import Filing
    from utils.ticker_cik_mapper import TickerCIKMapper

logger = logging.getLogger(__name__)

# Module-level cache for lazy imports
_lazy_imports_cache = {}


class SECApiClient:
    """
    Async-friendly client for interacting with the SEC EDGAR API.

    This wraps the mature synchronous utilities used by the legacy pipeline so
    the new agentic orchestrator can reuse the proven filing retrieval logic.
    """

    def __init__(self, user_agent: Optional[str] = None, config: Optional[Any] = None) -> None:
        self.config = config or get_config()
        sec_config = getattr(self.config, "sec", None)
        self.user_agent = user_agent or getattr(sec_config, "user_agent", "InvestiGator/1.0")

        self.base_url = "https://data.sec.gov"
        self.edgar_url = "https://www.sec.gov/Archives/edgar/data"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )

        self.cache_manager = get_cache_manager()
        cache_dir = getattr(sec_config, "cache_dir", "data")

        # Lazy imports to avoid circular dependency
        from utils.api_client import SECAPIClient as LegacySECAPIClient
        from utils.submission_processor import get_submission_processor
        from utils.ticker_cik_mapper import TickerCIKMapper

        self.ticker_mapper = TickerCIKMapper(data_dir=cache_dir, config=self.config)
        self.submission_processor = get_submission_processor()
        self.legacy_client = LegacySECAPIClient(self.user_agent, self.config)

    async def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """Fetch company facts, delegating to the legacy SECAPIClient."""
        cik_padded = str(cik).zfill(10)
        return await asyncio.to_thread(self.legacy_client.get_company_facts, cik_padded)

    async def get_submissions(self, cik: str) -> Dict[str, Any]:
        """Fetch raw submissions JSON from EDGAR."""
        cik_padded = str(cik).zfill(10)
        return await asyncio.to_thread(self.legacy_client.get_submissions, cik_padded)

    async def get_filing(self, accession_number: str, cik: str) -> str:
        """Download the raw filing text for a given accession number."""
        accession = accession_number.replace("-", "")
        cik_padded = str(cik).zfill(10)
        url = f"{self.edgar_url}/{cik_padded}/{accession}/{accession_number}.txt"

        def _fetch() -> str:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            return response.text

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:
            logger.error("Error fetching filing %s: %s", accession_number, exc)
            raise

    async def get_filing_by_symbol(
        self,
        symbol: str,
        form_type: str = "10-K",
        period: str = "latest",
    ) -> Dict[str, Any]:
        """
        Resolve the latest SEC filing for a symbol and return structured content.
        """
        try:
            cik = self._resolve_cik(symbol)
            if not cik:
                raise ValueError(f"Unable to resolve CIK for symbol {symbol}")

            submissions_data = await self._load_submissions(symbol, cik)
            parsed_submissions = self.submission_processor.parse_submissions(submissions_data)

            filings = parsed_submissions.get("filings", {}).get("all", [])
            target_filings = self._filter_filings(filings, form_type)
            if not target_filings:
                raise ValueError(f"No {form_type} filings available for {symbol}")

            selected_filing = self._select_filing(target_filings, period)
            if not selected_filing:
                raise ValueError(f"No filing matched period '{period}' for {symbol}")

            filing_text = await self.get_filing(selected_filing.accession_number, cik)
            period_end = selected_filing.report_date or selected_filing.filing_date

            return {
                "cik": cik,
                "symbol": symbol,
                "filing_type": selected_filing.base_form_type,
                "filing_date": selected_filing.filing_date,
                "period_end": period_end,
                "form_url": self._build_form_url(cik, selected_filing),
                "xbrl_url": self._build_xbrl_url(cik, submissions_data, selected_filing),
                "text": filing_text,
            }

        except Exception as exc:
            logger.error("Error getting filing for %s: %s", symbol, exc)
            raise

    async def search_filings(
        self,
        symbol: str,
        form_type: str = "10-K",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return recent filings metadata for a given symbol.
        """
        cik = self._resolve_cik(symbol)
        if not cik:
            return []

        submissions_data = await self._load_submissions(symbol, cik)
        parsed_submissions = self.submission_processor.parse_submissions(submissions_data)
        filings = self._filter_filings(parsed_submissions.get("filings", {}).get("all", []), form_type)

        results: List[Dict[str, Any]] = []
        for filing in filings[:limit]:
            results.append(
                {
                    "form_type": filing.base_form_type,
                    "filing_date": filing.filing_date,
                    "period_key": getattr(filing, "period_key", ""),
                    "accession_number": filing.accession_number,
                    "form_url": self._build_form_url(cik, filing),
                }
            )
        return results

    def close(self) -> None:
        """Close underlying HTTP sessions."""
        try:
            if self.session:
                self.session.close()
        finally:
            if hasattr(self.legacy_client, "close"):
                self.legacy_client.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_cik(self, symbol: str) -> Optional[str]:
        cik = self.ticker_mapper.get_cik_padded(symbol)
        if cik:
            return cik
        logger.error("Ticker mapper could not resolve CIK for %s", symbol)
        return None

    async def _load_submissions(self, symbol: str, cik: str) -> Dict[str, Any]:
        """
        Load submissions from cache or SEC API and persist them for reuse.
        """
        cache_key = {"symbol": symbol, "cik": cik}
        cached = self.cache_manager.get(CacheType.SUBMISSION_DATA, cache_key)
        if cached and "submissions_data" in cached:
            return cached["submissions_data"]

        submissions = await self.get_submissions(cik)
        cache_value = {
            "symbol": symbol,
            "cik": cik,
            "submissions_data": submissions,
            "cached_at": datetime.utcnow().isoformat(),
        }
        try:
            self.cache_manager.set(CacheType.SUBMISSION_DATA, cache_key, cache_value)
        except Exception as exc:
            logger.warning("Failed to cache submissions for %s: %s", symbol, exc)
        return submissions

    @staticmethod
    def _filter_filings(filings: List["Filing"], form_type: str) -> List["Filing"]:
        from utils.submission_processor import Filing

        base_type = (form_type or "").upper()
        results: List[Filing] = []
        for filing in filings:
            if isinstance(filing, Filing):
                if filing.base_form_type.upper() == base_type:
                    results.append(filing)
            else:
                candidate = SECApiClient._dict_to_filing(filing)
                if candidate.base_form_type.upper() == base_type:
                    results.append(candidate)
        # Sort newest first by filing date
        results.sort(key=lambda f: f.filing_date or "", reverse=True)
        return results

    @staticmethod
    def _dict_to_filing(data: Dict[str, Any]) -> "Filing":
        from utils.submission_processor import Filing

        return Filing(
            form_type=data.get("form_type", ""),
            filing_date=data.get("filing_date", ""),
            accession_number=data.get("accession_number", ""),
            primary_document=data.get("primary_document", ""),
            fiscal_year=data.get("fiscal_year"),
            fiscal_period=data.get("fiscal_period"),
            is_amended=data.get("is_amended", False),
            amendment_number=data.get("amendment_number"),
            report_date=data.get("report_date"),
        )

    @staticmethod
    def _select_filing(filings: List["Filing"], period: str) -> Optional["Filing"]:
        if not filings:
            return None

        normalized_period = (period or "latest").upper()
        if normalized_period not in ("LATEST", "RECENT"):
            for filing in filings:
                period_key = getattr(filing, "period_key", "").upper()
                if period_key == normalized_period:
                    return filing

        # Fallback to most recent by filing date
        def _parse_date(value: str) -> datetime:
            try:
                return datetime.strptime(value, "%Y-%m-%d")
            except Exception:
                return datetime.min

        return max(filings, key=lambda f: _parse_date(f.filing_date or "1900-01-01"))

    def _build_form_url(self, cik: str, filing: "Filing") -> str:
        accession = filing.accession_number.replace("-", "")
        primary_doc = getattr(filing, "primary_document", "") or f"{filing.accession_number}.txt"
        return f"{self.edgar_url}/{cik}/{accession}/{primary_doc}"

    def _build_xbrl_url(
        self,
        cik: str,
        submissions: Dict[str, Any],
        filing: "Filing",
    ) -> Optional[str]:
        """
        Attempt to build an XBRL URL if the submissions JSON lists a matching file.
        """
        files = submissions.get("filings", {}).get("files", [])
        accession = filing.accession_number.replace("-", "")

        for file_entry in files:
            if file_entry.get("accessionNumber") == filing.accession_number:
                filename = file_entry.get("filename")
                if filename:
                    return f"{self.edgar_url}/{cik}/{accession}/{filename}"

        # Fall back to the standard XBRL convention if the filename is provided
        primary_doc = getattr(filing, "primary_document", "")
        if primary_doc and primary_doc.endswith(".htm"):
            xbrl_candidate = primary_doc.replace(".htm", "_htm.xml")
            return f"{self.edgar_url}/{cik}/{accession}/{xbrl_candidate}"

        return None


# Backwards compatibility for legacy imports
SECAPIClient = SECApiClient
