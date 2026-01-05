#!/usr/bin/env python3
"""
InvestiGator - RDBMS Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

RDBMS based cache storage handler
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

from sqlalchemy import text

from .cache_base import CacheStorageHandler
from .cache_types import CacheType


# UTF-8 encoding helpers for JSON operations
def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely encode object to JSON with UTF-8 encoding, handling binary characters"""
    return json.dumps(obj, ensure_ascii=False, **kwargs)


def safe_json_loads(json_str: str) -> Any:
    """Safely decode JSON string with UTF-8 encoding"""
    if isinstance(json_str, bytes):
        json_str = json_str.decode("utf-8", errors="replace")
    return json.loads(json_str)


logger = logging.getLogger(__name__)


class RdbmsCacheStorageHandler(CacheStorageHandler):
    """RDBMS based cache storage handler"""

    def __init__(self, cache_type: CacheType, priority: int = 0):
        """
        Initialize RDBMS cache handler

        Args:
            cache_type: Type of cache
            priority: Priority for lookup
        """
        super().__init__(cache_type, priority)

        # Import here to avoid circular dependencies
        from investigator.infrastructure.database.db import (
            DatabaseManager,
            get_llm_responses_dao,
            get_quarterly_metrics_dao,
            get_sec_companyfacts_dao,
            get_sec_responses_dao,
            get_sec_submissions_dao,
        )

        # Always initialize DatabaseManager for delete operations
        self.db_manager = DatabaseManager()

        # Initialize appropriate DAO based on cache type
        if cache_type == CacheType.SEC_RESPONSE:
            self.dao = get_sec_responses_dao()
        elif cache_type == CacheType.LLM_RESPONSE:
            self.dao = get_llm_responses_dao()
        elif cache_type == CacheType.QUARTERLY_METRICS:
            self.dao = get_quarterly_metrics_dao()
        elif cache_type == CacheType.COMPANY_FACTS:
            self.dao = get_sec_companyfacts_dao()
        elif cache_type == CacheType.SUBMISSION_DATA:
            self.dao = get_sec_submissions_dao()
        elif cache_type == CacheType.MARKET_CONTEXT:
            # MARKET_CONTEXT not stored in RDBMS - better suited for file/parquet cache
            # Market-wide data doesn't benefit from relational storage
            self.dao = None
            logger.debug("MARKET_CONTEXT cache type initialized (RDBMS storage disabled)")
        else:
            raise ValueError(f"Unsupported cache type for RDBMS: {cache_type}")

    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Retrieve data from RDBMS cache"""
        if self.priority < 0:
            return None  # Skip lookup for negative priority

        # MARKET_CONTEXT not stored in RDBMS
        if self.cache_type == CacheType.MARKET_CONTEXT:
            return None

        try:
            key_dict = self._normalize_key(key)

            if self.cache_type == CacheType.SEC_RESPONSE:
                symbol = key_dict.get("symbol")
                form_type = key_dict.get("form_type") or key_dict.get("filing_type") or "10-K"
                category = key_dict.get("category") or "filing"
                period_key = key_dict.get("period")

                if symbol:
                    result = None
                    fiscal_year = None
                    fiscal_period = None

                    if period_key:
                        fiscal_year, fiscal_period = self._parse_period_key(period_key)
                        if fiscal_year is not None and fiscal_period is not None:
                            result = self.dao.get_response(symbol, form_type, fiscal_year, fiscal_period, category)

                    if result is None:
                        result = self.dao.get_latest_response(symbol, form_type, category)

                    if result:
                        payload = result.get("response_data", {}) or {}
                        payload = self._rehydrate_filing_payload(payload)
                        return payload

            elif self.cache_type == CacheType.LLM_RESPONSE:
                # Fetch from llm_response_store
                symbol = key_dict.get("symbol")
                llm_type = key_dict.get("llm_type")
                form_type = key_dict.get("form_type", "N/A")
                period = key_dict.get("period", "N/A")

                if symbol and llm_type:
                    result = self.dao.get_llm_response(symbol, form_type, period, llm_type)
                    if result:
                        logger.debug(f"Cache hit (RDBMS): LLM response for {symbol}")
                        return result

            elif self.cache_type == CacheType.SUBMISSION_DATA:
                # Fetch from sec_submissions table (no materialized view)
                symbol = key_dict.get("symbol")
                cik = key_dict.get("cik")

                # Only proceed if we have both symbol and CIK
                if symbol and cik:
                    from investigator.infrastructure.database.db import get_sec_submissions_dao

                    dao = get_sec_submissions_dao()

                    # Get submission data directly from sec_submissions table
                    result = dao.get_submission(symbol, cik, max_age_days=7)
                    if result:
                        logger.debug(f"Cache hit (RDBMS): Submission data for {symbol}")
                        # Convert datetime to ISO format string for TTL calculation
                        cached_at = result["updated_at"]
                        if hasattr(cached_at, "isoformat"):
                            cached_at = cached_at.isoformat()

                        return {
                            "symbol": symbol,
                            "cik": cik,
                            "company_name": result["company_name"],
                            "submissions_data": result["submissions_data"],
                            "cached_at": cached_at,
                        }

            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Fetch from all_companyfacts_store using DAO
                symbol = key_dict.get("symbol")

                if symbol:
                    from investigator.infrastructure.database.db import get_sec_companyfacts_dao

                    dao = get_sec_companyfacts_dao()
                    result = dao.get_company_facts(symbol)

                    if result:
                        logger.debug(f"Cache hit (RDBMS): Company facts for {symbol}")
                        return result

            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # DEPRECATED: quarterly_metrics table - use sec_companyfacts_processed instead
                # This cache type is no-op and will return None to trigger fallback
                logger.debug(
                    f"DEPRECATED: QUARTERLY_METRICS cache type - returning None to use sec_companyfacts_processed fallback"
                )
                return None
                # # Fetch from quarterly_metrics using composite key
                # symbol = key_dict.get("symbol")
                # fiscal_year = key_dict.get("fiscal_year")
                # fiscal_period = key_dict.get("fiscal_period")
                #
                # if symbol:
                #     result = self.dao.get_metrics(symbol, fiscal_year, fiscal_period)
                #     if result:
                #         logger.debug(f"Cache hit (RDBMS): Quarterly metrics for {symbol} {fiscal_year}-{fiscal_period}")
                #         return result

            logger.debug(f"Cache miss (RDBMS): {key_dict}")
            return None

        except Exception as e:
            logger.error(f"Error reading from RDBMS cache: {e}")
            return None

    def _convert_to_dict(self, value: Any) -> Dict[str, Any]:
        """
        Convert value to dictionary if it's a dataclass or other object.

        Args:
            value: Value to convert (can be dict, dataclass, or other object)

        Returns:
            Dictionary representation of the value
        """
        # Already a dict
        if isinstance(value, dict):
            return value

        # Check if it's a dataclass (like QuarterlyData)
        from dataclasses import asdict, is_dataclass

        if is_dataclass(value):
            return asdict(value)

        # Check for to_dict() method (custom serialization)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()

        # Fallback: try to convert __dict__ attribute
        if hasattr(value, "__dict__"):
            return value.__dict__

        # Last resort: wrap primitive values
        return {"value": value}

    def set(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """Store data in RDBMS cache"""
        # MARKET_CONTEXT not stored in RDBMS (use file/parquet cache instead)
        if self.cache_type == CacheType.MARKET_CONTEXT:
            return False

        try:
            # Convert dataclasses/objects to dicts
            value = self._convert_to_dict(value)

            key_dict = self._normalize_key(key)

            if self.cache_type == CacheType.SEC_RESPONSE:
                symbol = key_dict.get("symbol") or value.get("symbol")
                form_type = (
                    key_dict.get("form_type") or key_dict.get("filing_type") or value.get("filing_type") or "10-K"
                )
                category = key_dict.get("category") or "filing"
                period_key = key_dict.get("period")

                fiscal_year, fiscal_period = self._derive_fiscal_period(form_type, period_key, value)

                if not (symbol and fiscal_year and fiscal_period):
                    logger.debug(
                        f"Skipping RDBMS SEC cache write - insufficient identifiers "
                        f"(symbol={symbol}, fiscal_year={fiscal_year}, fiscal_period={fiscal_period})"
                    )
                    return False

                metadata = {}
                existing_metadata = value.get("metadata")
                if isinstance(existing_metadata, dict):
                    metadata.update(existing_metadata)
                metadata.setdefault("form_type", form_type)
                metadata["cache_period_key"] = period_key
                metadata["cached_at"] = datetime.now(timezone.utc).isoformat()

                return self.dao.save_response(
                    symbol=symbol,
                    form_type=form_type,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    category=category,
                    response_data=value,
                    metadata=metadata,
                )

            elif self.cache_type == CacheType.LLM_RESPONSE:
                # Store in llm_response_store
                llm_type = key_dict.get("llm_type")
                if not llm_type and isinstance(value, dict):
                    llm_type = value.get("llm_type") or value.get("analysis_type")
                if not llm_type:
                    llm_type = "unknown"

                metadata = {}
                if isinstance(value, dict) and isinstance(value.get("metadata"), dict):
                    metadata = dict(value.get("metadata"))
                metadata.setdefault("analysis_type", key_dict.get("analysis_type"))
                metadata.setdefault("llm_type", llm_type)
                metadata.setdefault("cached_at", datetime.now(timezone.utc).isoformat())

                return self.dao.save_llm_response(
                    symbol=key_dict.get("symbol"),
                    form_type=key_dict.get("form_type", "N/A"),
                    period=key_dict.get("period", "N/A"),
                    prompt=value.get("prompt", ""),
                    model_info=value.get("model_info", {}),
                    response=value.get("response", {}),
                    metadata=metadata,
                    llm_type=llm_type,
                )

            elif self.cache_type == CacheType.SUBMISSION_DATA:
                # Store in all_submission_store using DAO
                symbol = key_dict.get("symbol")
                cik = key_dict.get("cik")

                if symbol and cik:
                    from investigator.infrastructure.database.db import get_sec_submissions_dao

                    dao = get_sec_submissions_dao()

                    # Extract latest filing date if available
                    latest_filing_date = None
                    submissions_data = value.get("submissions_data", {})
                    if isinstance(submissions_data, dict):
                        filings = submissions_data.get("filings", {}).get("recent", {})
                        filing_dates = filings.get("filingDate", [])
                        if filing_dates:
                            latest_filing_date = filing_dates[0]  # Assuming sorted desc

                    success = dao.save_submission(
                        symbol=symbol,
                        cik=cik,
                        company_name=value.get("company_name", ""),
                        submissions_data=submissions_data,
                    )

                    if success:
                        logger.debug(f"Stored submission data for {symbol}")
                    return success

            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Store in all_companyfacts_store using DAO
                symbol = key_dict.get("symbol")

                if symbol:
                    from investigator.infrastructure.database.db import get_sec_companyfacts_dao

                    dao = get_sec_companyfacts_dao()

                    # Handle both direct company facts and wrapped data
                    if "companyfacts" in value:
                        companyfacts = value["companyfacts"]
                        metadata = value.get("metadata", {})
                    else:
                        companyfacts = value
                        metadata = {}

                    # Extract CIK with priority order: metadata > companyfacts > lookup
                    cik = None

                    # 1. Try metadata first (most reliable)
                    if metadata and metadata.get("cik"):
                        cik = metadata["cik"]

                    # 2. Try companyfacts data
                    elif companyfacts.get("cik"):
                        cik_val = companyfacts["cik"]
                        if isinstance(cik_val, int):
                            cik = f"{cik_val:010d}"  # Pad to 10 digits
                        else:
                            cik = str(cik_val)

                    # 3. Fall back to ticker-CIK lookup
                    if not cik or cik == "0000000000":
                        from utils.ticker_cik_mapper import TickerCIKMapper

                        mapper = TickerCIKMapper()
                        lookup_cik = mapper.get_cik(symbol)
                        if lookup_cik:
                            cik = f"{int(lookup_cik):010d}"
                        else:
                            logger.warning(f"Could not resolve CIK for {symbol}, skipping RDBMS storage")
                            return False

                    company_name = companyfacts.get("entityName", "")

                    success = dao.store_company_facts(
                        symbol=symbol, cik=cik, company_name=company_name, companyfacts=companyfacts, metadata=metadata
                    )

                    if success:
                        logger.debug(f"Stored company facts for {symbol} with CIK {cik}")
                    return success

            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # DEPRECATED: quarterly_metrics table - use sec_companyfacts_processed instead
                # No-op: Data is saved to sec_companyfacts_processed during SEC processing
                logger.debug(
                    f"DEPRECATED: QUARTERLY_METRICS set operation - skipping (data in sec_companyfacts_processed)"
                )
                return True  # Return success to avoid breaking code
                # # Store in quarterly_metrics using composite key
                # symbol = key_dict.get("symbol")
                # cik = value.get("cik", "")
                #
                # # CRITICAL: If CIK is missing, fetch it from ticker mapper
                # # All SEC data is CIK-based, so we must have a valid CIK
                # if not cik or cik.strip() == "":
                #     from utils.ticker_cik_mapper import TickerCIKMapper
                #
                #     mapper = TickerCIKMapper()
                #     cik = mapper.get_cik(symbol)
                #
                #     if not cik:
                #         logger.error(
                #             f"❌ CRITICAL: Cannot save quarterly metrics for {symbol} "
                #             f"{key_dict.get('fiscal_year')}-{key_dict.get('fiscal_period')} because "
                #             f"CIK could not be found in ticker_cik_mapping. Skipping cache write."
                #         )
                #         return False
                #     else:
                #         logger.debug(f"✓ Fetched missing CIK for {symbol} from ticker mapper: {cik}")
                #
                # return self.dao.save_metrics(
                #     symbol=symbol,
                #     fiscal_year=key_dict.get("fiscal_year"),
                #     fiscal_period=key_dict.get("fiscal_period"),
                #     cik=cik,
                #     form_type=key_dict.get("form_type", "10-K"),
                #     metrics_data=value.get("metrics", {}),
                #     company_name=value.get("company_name", ""),
                # )

            return False

        except Exception as e:
            logger.error(f"Error writing to RDBMS cache: {e}")
            return False

    def _parse_period_key(self, period_key: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
        """Parse canonical period strings like '2024-Q1' or '2023-FY'."""
        if not period_key:
            return None, None
        period_str = str(period_key).strip().upper()
        match = re.match(r"^(\d{4})-(Q[1-4]|FY)$", period_str)
        if match:
            return int(match.group(1)), match.group(2)
        return None, None

    def _derive_fiscal_period(
        self, form_type: Optional[str], period_key: Optional[str], value: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[str]]:
        """Determine fiscal year/period from explicit key or filing metadata."""
        fiscal_year, fiscal_period = self._parse_period_key(period_key)
        if fiscal_year is not None and fiscal_period is not None:
            return fiscal_year, fiscal_period

        candidate = value.get("period_end") or value.get("filing_date")
        dt = self._parse_datetime(candidate)
        if not dt:
            return None, None

        form_upper = (form_type or "").upper()
        if form_upper.startswith("10-Q"):
            quarter = max(1, min(4, ((dt.month - 1) // 3) + 1))
            return dt.year, f"Q{quarter}"

        return dt.year, "FY"

    @staticmethod
    def _parse_datetime(candidate: Any) -> Optional[datetime]:
        """Parse ISO-formatted datetime strings into datetime objects."""
        if isinstance(candidate, datetime):
            return candidate
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized)
            except ValueError:
                try:
                    return datetime.strptime(candidate.strip(), "%Y-%m-%d")
                except ValueError:
                    return None
        return None

    def _rehydrate_filing_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert serialized fields back into richer Python types."""
        if not isinstance(payload, dict):
            return payload

        for field in ("filing_date", "period_end"):
            if field in payload:
                dt = self._parse_datetime(payload[field])
                if dt:
                    payload[field] = dt
        return payload

    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in RDBMS cache"""
        # Use get method to check existence
        return self.get(key) is not None

    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete data from RDBMS cache"""
        try:
            key_dict = self._normalize_key(key)
            symbol = key_dict.get("symbol", "")

            if not symbol:
                logger.warning("Cannot delete from RDBMS cache without symbol")
                return False

            # Use DAO methods for deletion based on cache type
            # DAO methods already log the deletion, so we don't log again here
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                form_type = key_dict.get("form_type")
                period = key_dict.get("period")
                llm_type = key_dict.get("llm_type")

                deleted_count = self.dao.delete_llm_responses(
                    symbol=symbol, form_type=form_type, period=period, llm_type=llm_type
                )
                return deleted_count > 0

            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Delete company facts for this symbol
                if hasattr(self.dao, "delete_companyfacts_by_symbol"):
                    deleted_count = self.dao.delete_companyfacts_by_symbol(symbol)
                    return deleted_count > 0
                else:
                    logger.warning(f"DAO does not support delete_companyfacts_by_symbol")
                    return False

            elif self.cache_type == CacheType.SEC_RESPONSE:
                # Delete SEC responses for this symbol
                if hasattr(self.dao, "delete_responses_by_symbol"):
                    deleted_count = self.dao.delete_responses_by_symbol(symbol)
                    return deleted_count > 0
                else:
                    logger.warning(f"DAO does not support delete_responses_by_symbol")
                    return False

            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # DEPRECATED: quarterly_metrics table - use sec_companyfacts_processed instead
                logger.debug(
                    f"DEPRECATED: QUARTERLY_METRICS delete operation - skipping (use sec_companyfacts_processed cleanup)"
                )
                return False

            else:
                # For other cache types that don't need deletion
                logger.debug(f"Delete operation not needed for cache type: {self.cache_type}")
                return False

        except Exception as e:
            logger.error(f"Error deleting from RDBMS cache: {e}")
            return False

    def delete_by_symbol(self, symbol: str) -> int:
        """
        Optimized symbol-based deletion for RDBMS cache.
        Uses SQL queries with symbol column for efficient deletion.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Number of records deleted
        """
        try:
            deleted_count = 0
            symbol = symbol.upper()  # Normalize to uppercase

            # Delete based on cache type using appropriate DAO methods
            # DAO methods already log the deletion, so we don't log again here
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                # Delete all LLM responses for this symbol across all form types
                deleted_count = self.dao.delete_llm_responses_by_pattern(symbol_pattern=symbol, form_type_pattern="%")

            elif self.cache_type == CacheType.SUBMISSION_DATA:
                # Delete submission data for this symbol
                from investigator.infrastructure.database.db import get_sec_submissions_dao

                dao = get_sec_submissions_dao()
                if hasattr(dao, "delete_submissions_by_symbol"):
                    deleted_count = dao.delete_submissions_by_symbol(symbol)

            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Delete company facts for this symbol
                from investigator.infrastructure.database.db import get_sec_companyfacts_dao

                dao = get_sec_companyfacts_dao()
                if hasattr(dao, "delete_companyfacts_by_symbol"):
                    deleted_count = dao.delete_companyfacts_by_symbol(symbol)

            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # DEPRECATED: quarterly_metrics table - use sec_companyfacts_processed instead
                logger.info(
                    f"Symbol cleanup [RDBMS-QM]: DEPRECATED - skipping (use sec_companyfacts_processed cleanup)"
                )
                # # Delete quarterly metrics for this symbol
                # from investigator.infrastructure.database.db import get_quarterly_metrics_dao
                #
                # dao = get_quarterly_metrics_dao()
                # if hasattr(dao, "delete_metrics_by_symbol"):
                #     deleted_count = dao.delete_metrics_by_symbol(symbol)
                #     logger.info(
                #         f"Symbol cleanup [RDBMS-QM]: Deleted {deleted_count} quarterly metrics for symbol {symbol}"
                #     )

            else:
                logger.debug(f"Symbol deletion not implemented for cache type: {self.cache_type}")
                return 0

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting by symbol '{symbol}' from RDBMS cache: {e}")
            return 0

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all cache entries matching a pattern (legacy method)"""
        try:
            # Use DAO methods for deletion based on cache type
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                # Convert file pattern to SQL LIKE pattern
                sql_pattern = pattern.replace("*", "%").replace("?", "_")

                deleted_count = self.dao.delete_llm_responses_by_pattern(
                    symbol_pattern=sql_pattern, form_type_pattern=sql_pattern
                )
                return deleted_count
            else:
                logger.warning(f"Delete by pattern operation not implemented for cache type: {self.cache_type}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting by pattern from RDBMS cache: {e}")
            return 0

    def clear_all(self) -> bool:
        """Clear all data from RDBMS cache"""
        try:
            # Use DAO methods for deletion based on cache type
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                # Delete all LLM responses using wildcard pattern
                deleted_count = self.dao.delete_llm_responses_by_pattern(symbol_pattern="%", form_type_pattern="%")
                logger.info(f"Cleared all RDBMS cache data ({deleted_count} entries)")
                return True
            else:
                logger.warning(f"Clear all operation not implemented for cache type: {self.cache_type}")
                return False

        except Exception as e:
            logger.error(f"Error clearing RDBMS cache: {e}")
            return False
