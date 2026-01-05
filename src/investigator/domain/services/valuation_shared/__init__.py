# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Shared Valuation Services Module.

Provides centralized, consistent valuation services used across:
- scripts/rl_backtest.py
- scripts/batch_analysis_runner.py
- victor_invest CLI (valuation.py)

This module eliminates code drift and duplication by providing:
- Single source of truth for all valuation config
- Consistent financial data fetching
- Unified TTM calculations
- Standardized ratio calculations
- Config-driven sector multiples

Example:
    from investigator.domain.services.valuation_shared import (
        ValuationConfigService,
        SectorMultiplesService,
        FinancialDataService,
    )

    config_service = ValuationConfigService()
    pe_multiple = config_service.get_sector_pe_multiple("Technology")

    multiples_service = SectorMultiplesService()
    all_multiples = multiples_service.get_multiples("Technology")

    data_service = FinancialDataService()
    quarterly = data_service.get_quarterly_metrics("AAPL")
    ttm = data_service.get_ttm_metrics("AAPL")
"""

from .fair_value_service import FairValueService
from .financial_data_service import FinancialDataService
from .ratio_calculator import RatioCalculator
from .sector_multiples_service import SectorMultiplesService
from .ttm_calculator import TTMCalculator
from .valuation_config_service import ValuationConfigService

__all__ = [
    "ValuationConfigService",
    "SectorMultiplesService",
    "FinancialDataService",
    "TTMCalculator",
    "RatioCalculator",
    "FairValueService",
]
