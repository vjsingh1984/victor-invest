#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.database.ticker_mapper
Migration Date: 2025-11-13 (Original migration: 46cc33fb)
Phase: Phase 3-B-2 (High-Priority Infrastructure Layer Migration)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The ticker-to-CIK mapper infrastructure has been properly located in the
infrastructure/database layer, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.database.ticker_mapper import (
    TickerCIKMapper,
    get_ticker_mapper,
    ticker_to_cik,
    ticker_to_cik_padded,
)

__all__ = [
    'TickerCIKMapper',
    'get_ticker_mapper',
    'ticker_to_cik',
    'ticker_to_cik_padded',
]
