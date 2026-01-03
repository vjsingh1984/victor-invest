#!/usr/bin/env python3
"""
DEPRECATED: Import shim for backward compatibility.
This module has been migrated to src/investigator/infrastructure/database/

Canonical import:
    from investigator.infrastructure.database.db import get_db_manager, get_session

DO NOT use this shim in new code. It will be archived once all imports are updated.

Migration Status: COMPLETE (src/ version is canonical)
Active Imports: 7 locations need updating (see docs/UTILS_MIGRATION_STRATEGY.md)
"""
import warnings

# Re-export from new location
from investigator.infrastructure.database.db import (
    # Core classes
    Base,
    DatabaseManager,
    TechnicalIndicators,
    # DAO classes
    TechnicalIndicatorsDAO,
    LLMResponseStoreDAO,
    TickerCIKMappingDAO,
    SECResponseStoreDAO,
    QuarterlyMetricsDAO,
    AllSubmissionStoreDAO,
    AllCompanyFactsStoreDAO,
    # Factory functions
    get_db_manager,
    get_database_engine,
    get_technical_indicators_dao,
    get_llm_responses_dao,
    get_ticker_cik_mapping_dao,
    get_sec_responses_dao,
    get_quarterly_metrics_dao,
    get_sec_submissions_dao,
    get_sec_companyfacts_dao,
    # Utility functions
    safe_json_dumps,
    safe_json_loads,
    is_etf,
)

warnings.warn(
    "utils.db is deprecated. Use 'from investigator.infrastructure.database.db import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Core classes
    "Base",
    "DatabaseManager",
    "TechnicalIndicators",
    # DAO classes
    "TechnicalIndicatorsDAO",
    "LLMResponseStoreDAO",
    "TickerCIKMappingDAO",
    "SECResponseStoreDAO",
    "QuarterlyMetricsDAO",
    "AllSubmissionStoreDAO",
    "AllCompanyFactsStoreDAO",
    # Factory functions
    "get_db_manager",
    "get_database_engine",
    "get_technical_indicators_dao",
    "get_llm_responses_dao",
    "get_ticker_cik_mapping_dao",
    "get_sec_responses_dao",
    "get_quarterly_metrics_dao",
    "get_sec_submissions_dao",
    "get_sec_companyfacts_dao",
    # Utility functions
    "safe_json_dumps",
    "safe_json_loads",
    "is_etf",
]
