# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Market Data Services - Shared module for stock market data operations.

This module provides consistent implementations for:
- Shares outstanding with split detection and normalization
- Historical price lookup with split-adjusted handling
- Data quality validation and anomaly detection
- Symbol metadata caching

All consumers (rl_backtest.py, batch_analysis_runner.py, victor_invest CLI)
should use these services to ensure consistency and avoid code drift.

Example:
    from investigator.domain.services.market_data import (
        SharesService,
        PriceService,
        DataValidationService,
    )

    shares_svc = SharesService()
    price_svc = PriceService()
    validation_svc = DataValidationService()

    # Get split-normalized shares for historical analysis
    shares_df = shares_svc.get_shares_history(symbol, lookback_months=[36, 24, 12, 6, 3])

    # Get historical price
    price = price_svc.get_price(symbol, target_date)

    # Validate data quality
    warnings = validation_svc.validate_shares(symbol, price)
"""

import os


def get_stock_db_url() -> str:
    """
    Build stock database connection URL from environment variables.

    Required environment variables:
    - STOCK_DB_PASSWORD: Password for stock database user

    Optional environment variables:
    - STOCK_DB_USER: Database username (default: stockuser)
    - STOCK_DB_HOST: Database host (default: localhost)
    - STOCK_DB_PORT: Database port (default: 5432)
    - STOCK_DB_NAME: Database name (default: stock)

    Returns:
        PostgreSQL connection URL for stock database

    Raises:
        EnvironmentError: If STOCK_DB_PASSWORD is not set
    """
    password = os.environ.get("STOCK_DB_PASSWORD")
    if not password:
        raise EnvironmentError(
            "STOCK_DB_PASSWORD environment variable not set. "
            "Please source your ~/.investigator/env file or set the variable."
        )
    user = os.environ.get("STOCK_DB_USER", "stockuser")
    host = os.environ.get("STOCK_DB_HOST", "localhost")
    port = os.environ.get("STOCK_DB_PORT", "5432")
    database = os.environ.get("STOCK_DB_NAME", "stock")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def get_sec_db_url() -> str:
    """
    Build SEC database connection URL from environment variables.

    Optional environment variables:
    - SEC_DB_USER: Database username (default: investigator)
    - SEC_DB_PASSWORD or DB_PASSWORD: Password (default: investigator)
    - SEC_DB_HOST or DB_HOST: Database host (default: localhost)
    - SEC_DB_PORT or DB_PORT: Database port (default: 5432)
    - SEC_DB_NAME: Database name (default: sec_database)

    Returns:
        PostgreSQL connection URL for SEC database
    """
    user = os.environ.get("SEC_DB_USER", "investigator")
    password = os.environ.get("SEC_DB_PASSWORD") or os.environ.get("DB_PASSWORD", "investigator")
    host = os.environ.get("SEC_DB_HOST") or os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("SEC_DB_PORT") or os.environ.get("DB_PORT", "5432")
    database = os.environ.get("SEC_DB_NAME", "sec_database")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

from investigator.domain.services.market_data.shares_service import (
    SharesService,
    SharesHistory,
)
from investigator.domain.services.market_data.price_service import (
    PriceService,
)
from investigator.domain.services.market_data.validation_service import (
    DataValidationService,
    DataQualityWarning,
)
from investigator.domain.services.market_data.metadata_service import (
    SymbolMetadataService,
    SymbolMetadata,
)
from investigator.domain.services.market_data.technical_analysis_service import (
    TechnicalAnalysisService,
    TechnicalFeatures,
    get_technical_analysis_service,
)

__all__ = [
    "get_stock_db_url",
    "get_sec_db_url",
    "SharesService",
    "SharesHistory",
    "PriceService",
    "DataValidationService",
    "DataQualityWarning",
    "SymbolMetadataService",
    "SymbolMetadata",
    "TechnicalAnalysisService",
    "TechnicalFeatures",
    "get_technical_analysis_service",
]
