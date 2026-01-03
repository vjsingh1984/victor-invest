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
