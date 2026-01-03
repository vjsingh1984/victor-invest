#!/usr/bin/env python3
"""
InvestiGator - Data Collector Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Centralized data collection from external APIs with intelligent routing
Routes SEC submissions to appropriate storage pipelines based on form type
Implements submission-driven event processing with CIK-based storage
Manages staggered TTL strategy for Russell 1000 stocks
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from pathlib import Path

from utils.vector_db.vector_engine import FinancialVectorDB, VectorDocument
from utils.vector_db.event_analyzer import EventAnalyzer, FinancialEvent
from utils.vector_db.vector_cache_handler import VectorCacheStorageHandler
from utils.api_client import SECAPIClient
from utils.ticker_cik_mapper import TickerCIKMapper
from investigator.infrastructure.cache.cache_manager import get_cache_manager
from investigator.infrastructure.cache.cache_types import CacheType

logger = logging.getLogger(__name__)


@dataclass
class SubmissionProcessingRule:
    """Rules for processing different submission types"""

    form_types: List[str]
    target_storage: str  # 'vector', 'disk_cache', 'rdbms_cache'
    enable_event_extraction: bool
    cache_type: Optional[CacheType] = None
    ttl_hours: int = 168  # 7 days default


class RussellTTLManager:
    """Manages staggered TTL for Russell 1000 stocks across 7 days"""

    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Russell 1000 symbols (sample - in production, load from file/API)
        self.russell_1000_symbols = self._load_russell_symbols()

        # Divide symbols into 7 groups for staggered refresh
        self.symbol_groups = self._create_staggered_groups()

    def _load_russell_symbols(self) -> List[str]:
        """Load Russell 1000 symbols - placeholder implementation"""
        # In production, this would load from a file or API
        # For now, using major tech stocks as example
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "NFLX",
            "CRM",
            "SNOW",
            "PLTR",
            "UBER",
            "LYFT",
            "SPOT",
            "ZM",
            "DOCU",
            "SHOP",
            "SQ",
            "PYPL",
            "V",
            "MA",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "USB",
            "PNC",
            "TFC",
            "COF",
            "AXP",
            "BLK",
            "SCHW",
            # Add more symbols to reach 1000...
        ]

    def _create_staggered_groups(self) -> Dict[int, List[str]]:
        """Divide Russell 1000 symbols into 7 groups for staggered refresh"""
        groups = {i: [] for i in range(7)}

        for i, symbol in enumerate(self.russell_1000_symbols):
            group_id = i % 7
            groups[group_id].append(symbol)

        self.logger.info(f"Created 7 staggered groups with {[len(groups[i]) for i in range(7)]} symbols each")
        return groups

    def get_symbol_refresh_day(self, symbol: str) -> int:
        """Get the refresh day (0-6) for a given symbol"""
        for day, symbols in self.symbol_groups.items():
            if symbol in symbols:
                return day
        return 0  # Default to day 0 if not found

    def get_symbols_for_refresh_day(self, day: int) -> List[str]:
        """Get symbols that should be refreshed on a given day (0-6)"""
        return self.symbol_groups.get(day, [])

    def get_current_refresh_symbols(self) -> List[str]:
        """Get symbols that should be refreshed today"""
        current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
        return self.get_symbols_for_refresh_day(current_day)

    def calculate_ttl_for_symbol(self, symbol: str, base_ttl_hours: int = 168) -> int:
        """Calculate TTL hours for a symbol based on its refresh schedule"""
        refresh_day = self.get_symbol_refresh_day(symbol)
        current_day = datetime.now().weekday()

        # Calculate days until next refresh
        if refresh_day >= current_day:
            days_until_refresh = refresh_day - current_day
        else:
            days_until_refresh = 7 - (current_day - refresh_day)

        # Add some buffer to ensure data stays fresh
        ttl_hours = (days_until_refresh * 24) + 12  # 12 hour buffer
        return min(ttl_hours, base_ttl_hours)  # Cap at base TTL


class DataCollector:
    """
    Centralized data collector for external APIs with intelligent routing.

    Manages all external API calls and routes data to appropriate storage:
    - SEC API calls for submissions, company facts, consolidated frames
    - Yahoo Finance API for technical data
    - Vector database for event-driven narratives (8-K filings)
    - Traditional cache for structured financial data (10-Q, 10-K)
    """

    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize external API clients
        self.sec_client = SECAPIClient(
            user_agent=config.sec.user_agent if config else "InvestiGator/1.0", config=config
        )
        self.ticker_mapper = TickerCIKMapper(data_dir=str(config.sec.cache_dir) if config else "./data/sec_cache")

        # Initialize storage and routing components
        self.cache_manager = get_cache_manager()
        self.event_analyzer = EventAnalyzer()
        self.ttl_manager = RussellTTLManager(config)

        # Initialize vector database if enabled
        self.vector_db = None
        if config and hasattr(config, "vector_db") and config.vector_db.enabled:
            try:
                from .vector_engine import FinancialVectorDB

                self.vector_db = FinancialVectorDB(
                    config.vector_db.get_storage_path(), config.vector_db.embedding_model
                )
                self.logger.info("Vector database initialized for submission routing")
            except Exception as e:
                self.logger.warning(f"Failed to initialize vector database: {e}")

        # Define processing rules
        self.processing_rules = self._initialize_processing_rules()

    def collect_submissions_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Collect all submissions for a symbol from SEC API and route to appropriate storage.
        This is the main entry point for submission-driven data collection.
        """
        try:
            self.logger.info(f"Collecting submissions for {symbol}")

            # Get CIK for symbol
            cik = self.ticker_mapper.get_cik_padded(symbol)
            if not cik:
                return {"status": "error", "reason": f"CIK not found for symbol {symbol}"}

            # Calculate TTL for this symbol based on refresh schedule
            base_ttl = 168  # 7 days
            symbol_ttl = self.ttl_manager.calculate_ttl_for_symbol(symbol, base_ttl)

            # Check if we should refresh based on staggered schedule
            current_refresh_symbols = self.ttl_manager.get_current_refresh_symbols()
            should_refresh = symbol in current_refresh_symbols

            self.logger.info(f"Symbol {symbol} TTL: {symbol_ttl}h, Should refresh: {should_refresh}")

            # Fetch submissions from SEC API
            submissions_response = self._fetch_submissions_from_sec(cik)
            if not submissions_response:
                return {"status": "error", "reason": "Failed to fetch submissions from SEC API"}

            # Process and route submissions
            routing_results = self.process_symbol_submissions(symbol, cik, submissions_response.get("submissions", []))

            # Also collect company facts if this is a periodic filing refresh
            if should_refresh or routing_results.get("routed_to_rdbms", 0) > 0:
                company_facts_result = self._collect_company_facts(symbol, cik, symbol_ttl)
                routing_results["company_facts"] = company_facts_result

            routing_results["ttl_info"] = {
                "symbol_ttl_hours": symbol_ttl,
                "refresh_day": self.ttl_manager.get_symbol_refresh_day(symbol),
                "should_refresh_today": should_refresh,
            }

            return routing_results

        except Exception as e:
            self.logger.error(f"Error collecting submissions for {symbol}: {e}")
            return {"status": "error", "reason": str(e)}

    def collect_daily_refresh_batch(self) -> Dict[str, Any]:
        """
        Collect submissions for all symbols scheduled for refresh today.
        Implements the staggered refresh strategy across Russell 1000.
        """
        refresh_symbols = self.ttl_manager.get_current_refresh_symbols()
        self.logger.info(f"Starting daily refresh batch for {len(refresh_symbols)} symbols")

        results = {
            "refresh_date": datetime.now().isoformat(),
            "symbols_processed": 0,
            "symbols_successful": 0,
            "symbols_failed": 0,
            "total_submissions_routed": 0,
            "symbol_results": {},
        }

        for symbol in refresh_symbols:
            try:
                symbol_result = self.collect_submissions_for_symbol(symbol)
                results["symbol_results"][symbol] = symbol_result
                results["symbols_processed"] += 1

                if symbol_result.get("status") == "success":
                    results["symbols_successful"] += 1
                    results["total_submissions_routed"] += symbol_result.get("processed", 0)
                else:
                    results["symbols_failed"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {symbol} in daily batch: {e}")
                results["symbols_failed"] += 1
                results["symbol_results"][symbol] = {"status": "error", "reason": str(e)}

        self.logger.info(
            f"Daily refresh completed: {results['symbols_successful']}/{results['symbols_processed']} successful"
        )
        return results

    def _fetch_submissions_from_sec(self, cik: str) -> Optional[Dict[str, Any]]:
        """Fetch submissions from SEC API"""
        try:
            # Use SEC API client to fetch submissions
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.sec_client.session.get(url, timeout=30)
            response.raise_for_status()

            submissions_data = response.json()

            # Process recent filings
            recent_filings = submissions_data.get("filings", {}).get("recent", {})
            if not recent_filings:
                return None

            # Convert to list of submission dictionaries
            submissions = []
            form_types = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])
            accession_numbers = recent_filings.get("accessionNumber", [])

            for i, form_type in enumerate(form_types):
                if i < len(filing_dates) and i < len(accession_numbers):
                    submissions.append(
                        {
                            "form": form_type,
                            "filingDate": filing_dates[i],
                            "accessionNumber": accession_numbers[i],
                            "cik": cik,
                        }
                    )

            return {"submissions": submissions}

        except Exception as e:
            self.logger.error(f"Error fetching submissions for CIK {cik}: {e}")
            return None

    def _collect_company_facts(self, symbol: str, cik: str, ttl_hours: int) -> Dict[str, Any]:
        """Collect company facts and store in appropriate cache"""
        try:
            # Fetch company facts from SEC API
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            response = self.sec_client.session.get(url, timeout=30)
            response.raise_for_status()

            company_facts = response.json()

            # Store in RDBMS cache for efficient querying
            cache_key = {"symbol": symbol, "cik": cik}
            cache_data = {
                "companyfacts": company_facts,
                "symbol": symbol,
                "cik": cik,
                "collected_timestamp": datetime.utcnow().isoformat(),
                "ttl_hours": ttl_hours,
            }

            success = self.cache_manager.set(CacheType.COMPANY_FACTS, cache_key, cache_data)

            return {
                "status": "success" if success else "error",
                "storage_type": "rdbms_cache",
                "cache_type": "company_facts",
                "data_size": len(str(company_facts)),
            }

        except Exception as e:
            self.logger.error(f"Error collecting company facts for {symbol}: {e}")
            return {"status": "error", "reason": str(e)}

    def collect_technical_data_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Collect technical data from Yahoo Finance API.
        Routes to parquet cache for efficient time-series storage.
        """
        try:
            self.logger.info(f"Collecting technical data for {symbol}")

            # Calculate TTL (technical data refreshes daily)
            ttl_hours = 24

            # Import yahoo finance technical module
            from yahoo_technical import get_technical_data

            # Fetch technical data
            technical_data = get_technical_data(symbol)
            if not technical_data:
                return {"status": "error", "reason": "Failed to fetch technical data"}

            # Store in technical cache (parquet format)
            cache_key = {"symbol": symbol, "analysis_date": datetime.now().strftime("%Y-%m-%d")}

            cache_data = {
                "technical_data": technical_data,
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "ttl_hours": ttl_hours,
            }

            success = self.cache_manager.set(CacheType.TECHNICAL_DATA, cache_key, cache_data)

            return {
                "status": "success" if success else "error",
                "storage_type": "parquet_cache",
                "cache_type": "technical_data",
                "data_points": len(technical_data) if isinstance(technical_data, (list, dict)) else 1,
            }

        except Exception as e:
            self.logger.error(f"Error collecting technical data for {symbol}: {e}")
            return {"status": "error", "reason": str(e)}

    def get_collection_schedule_status(self) -> Dict[str, Any]:
        """Get comprehensive status of data collection schedules"""
        return {
            "staggered_refresh": self.ttl_manager.get_refresh_schedule_status(),
            "processing_rules": {
                rule_name: {
                    "form_types": rule.form_types,
                    "target_storage": rule.target_storage,
                    "enable_event_extraction": rule.enable_event_extraction,
                    "ttl_hours": rule.ttl_hours,
                }
                for rule_name, rule in self.processing_rules.items()
            },
            "api_clients": {
                "sec_api": {
                    "base_url": (
                        self.sec_client.base_url if hasattr(self.sec_client, "base_url") else "https://data.sec.gov"
                    ),
                    "user_agent": self.sec_client.user_agent if hasattr(self.sec_client, "user_agent") else "Unknown",
                },
                "vector_db_enabled": self.vector_db is not None,
            },
        }

    def _initialize_processing_rules(self) -> Dict[str, SubmissionProcessingRule]:
        """Initialize rules for processing different submission types"""
        return {
            "event_forms": SubmissionProcessingRule(
                form_types=["8-K", "8-K/A", "8-K12B", "8-K15D5"],
                target_storage="vector",
                enable_event_extraction=True,
                cache_type=None,  # Vector storage doesn't use cache types
                ttl_hours=168,  # 7 days
            ),
            "quarterly_forms": SubmissionProcessingRule(
                form_types=["10-Q", "10-Q/A"],
                target_storage="disk_cache",
                enable_event_extraction=False,
                cache_type=CacheType.SEC_RESPONSE,
                ttl_hours=2160,  # 90 days - quarterly data is stable
            ),
            "annual_forms": SubmissionProcessingRule(
                form_types=["10-K", "10-K/A"],
                target_storage="rdbms_cache",
                enable_event_extraction=False,
                cache_type=CacheType.SEC_RESPONSE,
                ttl_hours=8760,  # 1 year - annual data is very stable
            ),
            "proxy_forms": SubmissionProcessingRule(
                form_types=["DEF 14A", "DEFA14A", "DEFM14A"],
                target_storage="vector",
                enable_event_extraction=True,
                cache_type=None,
                ttl_hours=8760,  # 1 year
            ),
            "insider_forms": SubmissionProcessingRule(
                form_types=["4", "4/A", "3", "3/A", "5", "5/A"],
                target_storage="vector",
                enable_event_extraction=True,
                cache_type=None,
                ttl_hours=168,  # 7 days
            ),
            "ownership_forms": SubmissionProcessingRule(
                form_types=["13D", "13D/A", "13G", "13G/A"],
                target_storage="vector",
                enable_event_extraction=True,
                cache_type=None,
                ttl_hours=720,  # 30 days
            ),
        }

    def route_submission(self, submission_data: Dict[str, Any], symbol: str, cik: str) -> Dict[str, Any]:
        """
        Route a SEC submission to appropriate processing pipeline

        Args:
            submission_data: Raw submission data from SEC API
            symbol: Stock ticker symbol
            cik: Central Index Key

        Returns:
            Processing result with routing information
        """
        try:
            form_type = submission_data.get("form", "").strip()
            filing_date = self._parse_filing_date(submission_data.get("filingDate"))
            accession_number = submission_data.get("accessionNumber")

            self.logger.info(f"Routing {form_type} submission for {symbol} (CIK: {cik})")

            # Determine processing rule
            processing_rule = self._get_processing_rule(form_type)
            if not processing_rule:
                self.logger.warning(f"No processing rule found for form type: {form_type}")
                return {"status": "skipped", "reason": f"Unsupported form type: {form_type}"}

            # Calculate TTL based on symbol's refresh schedule
            ttl_hours = self.ttl_manager.calculate_ttl_for_symbol(symbol, processing_rule.ttl_hours)

            # Route based on target storage
            if processing_rule.target_storage == "vector":
                result = self._route_to_vector_storage(
                    submission_data, symbol, cik, form_type, filing_date, accession_number, processing_rule, ttl_hours
                )
            elif processing_rule.target_storage == "disk_cache":
                result = self._route_to_disk_cache(
                    submission_data, symbol, cik, form_type, filing_date, accession_number, processing_rule, ttl_hours
                )
            elif processing_rule.target_storage == "rdbms_cache":
                result = self._route_to_rdbms_cache(
                    submission_data, symbol, cik, form_type, filing_date, accession_number, processing_rule, ttl_hours
                )
            else:
                result = {"status": "error", "reason": f"Unknown target storage: {processing_rule.target_storage}"}

            # Add routing metadata
            result["routing_info"] = {
                "form_type": form_type,
                "target_storage": processing_rule.target_storage,
                "ttl_hours": ttl_hours,
                "refresh_day": self.ttl_manager.get_symbol_refresh_day(symbol),
                "event_extraction_enabled": processing_rule.enable_event_extraction,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error routing submission for {symbol}: {e}")
            return {"status": "error", "reason": str(e)}

    def _route_to_vector_storage(
        self,
        submission_data: Dict[str, Any],
        symbol: str,
        cik: str,
        form_type: str,
        filing_date: date,
        accession_number: str,
        processing_rule: SubmissionProcessingRule,
        ttl_hours: int,
    ) -> Dict[str, Any]:
        """Route submission to vector storage with event extraction"""
        if not self.vector_db:
            return {"status": "error", "reason": "Vector database not available"}

        try:
            # Extract filing content (this would normally come from SEC filing URL)
            filing_content = self._extract_filing_content(submission_data)

            documents_created = 0
            events_extracted = 0

            # Extract events if enabled
            if processing_rule.enable_event_extraction and filing_content:
                events = self.event_analyzer.analyze_filing(
                    filing_content=filing_content,
                    symbol=symbol,
                    cik=cik,
                    form_type=form_type,
                    filing_date=filing_date,
                    accession_number=accession_number,
                )

                # Convert events to vector documents
                for event in events:
                    vector_doc = event.to_vector_document()
                    if self.vector_db.add_document(vector_doc):
                        documents_created += 1
                        events_extracted += 1

            # Also store the raw filing as a document for general search
            if filing_content:
                filing_doc = VectorDocument(
                    doc_id=f"{symbol}_filing_{accession_number}",
                    content=filing_content[:5000],  # Truncate for embedding
                    doc_type="filing",
                    symbol=symbol,
                    fiscal_year=filing_date.year,
                    fiscal_period=f"Q{((filing_date.month - 1) // 3) + 1}",
                    form_type=form_type,
                    topics=[form_type.lower()],
                    metadata={
                        "accession_number": accession_number,
                        "filing_date": filing_date.isoformat(),
                        "cik": cik,
                        "ttl_hours": ttl_hours,
                    },
                )

                if self.vector_db.add_document(filing_doc):
                    documents_created += 1

            return {
                "status": "success",
                "documents_created": documents_created,
                "events_extracted": events_extracted,
                "storage_type": "vector",
            }

        except Exception as e:
            self.logger.error(f"Error routing to vector storage: {e}")
            return {"status": "error", "reason": str(e)}

    def _route_to_disk_cache(
        self,
        submission_data: Dict[str, Any],
        symbol: str,
        cik: str,
        form_type: str,
        filing_date: date,
        accession_number: str,
        processing_rule: SubmissionProcessingRule,
        ttl_hours: int,
    ) -> Dict[str, Any]:
        """Route submission to disk cache (for company facts and structured data)"""
        try:
            # Create cache key based on CIK (submissions are CIK-specific)
            cache_key = {
                "symbol": symbol,
                "cik": cik,
                "form_type": form_type,
                "fiscal_year": filing_date.year,
                "fiscal_period": f"Q{((filing_date.month - 1) // 3) + 1}",
            }

            # Prepare data for caching
            cache_data = {
                "submission_data": submission_data,
                "symbol": symbol,
                "cik": cik,
                "form_type": form_type,
                "filing_date": filing_date.isoformat(),
                "accession_number": accession_number,
                "ttl_hours": ttl_hours,
                "refresh_day": self.ttl_manager.get_symbol_refresh_day(symbol),
                "processed_timestamp": datetime.utcnow().isoformat(),
            }

            # Store in appropriate cache
            if processing_rule.cache_type:
                success = self.cache_manager.set(processing_rule.cache_type, cache_key, cache_data)

                return {
                    "status": "success" if success else "error",
                    "storage_type": "disk_cache",
                    "cache_type": processing_rule.cache_type.value,
                    "cache_key": cache_key,
                }
            else:
                return {"status": "error", "reason": "No cache type specified for disk storage"}

        except Exception as e:
            self.logger.error(f"Error routing to disk cache: {e}")
            return {"status": "error", "reason": str(e)}

    def _route_to_rdbms_cache(
        self,
        submission_data: Dict[str, Any],
        symbol: str,
        cik: str,
        form_type: str,
        filing_date: date,
        accession_number: str,
        processing_rule: SubmissionProcessingRule,
        ttl_hours: int,
    ) -> Dict[str, Any]:
        """Route submission to RDBMS cache (for consolidated company facts)"""
        try:
            # Create cache key for RDBMS storage
            cache_key = {
                "symbol": symbol,
                "cik": cik,
                "form_type": form_type,
                "fiscal_year": filing_date.year,
                "fiscal_period": f"Q{((filing_date.month - 1) // 3) + 1}",
            }

            # Prepare consolidated data for RDBMS
            cache_data = {
                "consolidated_submission": submission_data,
                "company_metadata": {
                    "symbol": symbol,
                    "cik": cik,
                    "form_type": form_type,
                    "filing_date": filing_date.isoformat(),
                    "accession_number": accession_number,
                },
                "cache_metadata": {
                    "ttl_hours": ttl_hours,
                    "refresh_day": self.ttl_manager.get_symbol_refresh_day(symbol),
                    "storage_type": "rdbms",
                    "processed_timestamp": datetime.utcnow().isoformat(),
                },
            }

            # Store in RDBMS cache
            if processing_rule.cache_type:
                success = self.cache_manager.set(processing_rule.cache_type, cache_key, cache_data)

                return {
                    "status": "success" if success else "error",
                    "storage_type": "rdbms_cache",
                    "cache_type": processing_rule.cache_type.value,
                    "cache_key": cache_key,
                }
            else:
                return {"status": "error", "reason": "No cache type specified for RDBMS storage"}

        except Exception as e:
            self.logger.error(f"Error routing to RDBMS cache: {e}")
            return {"status": "error", "reason": str(e)}

    def _get_processing_rule(self, form_type: str) -> Optional[SubmissionProcessingRule]:
        """Get processing rule for a form type"""
        for rule_name, rule in self.processing_rules.items():
            if form_type in rule.form_types:
                return rule
        return None

    def _parse_filing_date(self, date_str: str) -> date:
        """Parse filing date from SEC format"""
        try:
            if isinstance(date_str, str):
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            return date_str
        except:
            return date.today()

    def _extract_filing_content(self, submission_data: Dict[str, Any]) -> str:
        """Extract filing content - placeholder implementation"""
        # In production, this would fetch the actual filing content from SEC
        # For now, return a representative text based on submission data

        form_type = submission_data.get("form", "")
        filing_date = submission_data.get("filingDate", "")

        # Simulate filing content based on form type
        if form_type.startswith("8-K"):
            return f"""
            Item 2.02 Results of Operations and Financial Condition
            
            This Form 8-K filing dated {filing_date} contains material information about the Company's
            operations and financial condition. The Company has reported significant developments
            that may impact future performance and investor expectations.
            
            Item 7.01 Regulation FD Disclosure
            
            Additional regulatory disclosures and forward-looking statements are included
            in this filing to provide transparency to shareholders and the investment community.
            """
        elif form_type.startswith("10-Q"):
            return f"""
            Quarterly Report dated {filing_date}
            
            Management's Discussion and Analysis of Financial Condition and Results of Operations
            
            This quarterly report provides detailed analysis of the Company's financial performance,
            including revenue trends, cost management initiatives, and strategic developments
            during the quarter.
            """
        elif form_type.startswith("10-K"):
            return f"""
            Annual Report dated {filing_date}
            
            Business Overview and Strategy
            
            This annual report provides comprehensive information about the Company's business model,
            competitive position, risk factors, and long-term strategic initiatives.
            
            Management's Discussion and Analysis provides detailed commentary on financial performance
            and outlook for the upcoming fiscal year.
            """
        else:
            return f"SEC Filing {form_type} dated {filing_date} - Content not available for simulation"

    def get_refresh_schedule_status(self) -> Dict[str, Any]:
        """Get status of the staggered refresh schedule"""
        current_symbols = self.ttl_manager.get_current_refresh_symbols()

        return {
            "current_refresh_day": datetime.now().weekday(),
            "symbols_to_refresh_today": len(current_symbols),
            "sample_symbols": current_symbols[:10],  # First 10 for display
            "total_russell_symbols": len(self.ttl_manager.russell_1000_symbols),
            "group_sizes": [len(self.ttl_manager.symbol_groups[i]) for i in range(7)],
        }

    def process_symbol_submissions(self, symbol: str, cik: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process all submissions for a symbol based on routing rules"""
        results = {
            "symbol": symbol,
            "cik": cik,
            "total_submissions": len(submissions),
            "processed": 0,
            "routed_to_vector": 0,
            "routed_to_disk": 0,
            "routed_to_rdbms": 0,
            "errors": 0,
            "details": [],
        }

        for submission in submissions:
            result = self.route_submission(submission, symbol, cik)
            results["details"].append(result)

            if result["status"] == "success":
                results["processed"] += 1
                storage_type = result.get("routing_info", {}).get("target_storage")

                if storage_type == "vector":
                    results["routed_to_vector"] += 1
                elif storage_type == "disk_cache":
                    results["routed_to_disk"] += 1
                elif storage_type == "rdbms_cache":
                    results["routed_to_rdbms"] += 1
            else:
                results["errors"] += 1

        return results
