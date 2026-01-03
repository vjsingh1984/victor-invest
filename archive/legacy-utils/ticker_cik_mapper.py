#!/usr/bin/env python3
"""
DEPRECATED: Import shim for backward compatibility.
This module has been migrated to src/investigator/infrastructure/database/

Canonical import:
    from investigator.infrastructure.database.ticker_mapper import TickerCIKMapper, get_ticker_mapper

DO NOT use this shim in new code. It will be archived once all imports are updated.

Migration Status: COMPLETE (src/ version is canonical)
Active Imports: 7 locations need updating (see docs/UTILS_MIGRATION_STRATEGY.md)
"""
import warnings

# Re-export from new location
from investigator.infrastructure.database.ticker_mapper import (
    TickerCIKMapper,
    get_ticker_mapper,
    ticker_to_cik,
    ticker_to_cik_padded,
)

warnings.warn(
    "utils.ticker_cik_mapper is deprecated. Use 'from investigator.infrastructure.database.ticker_mapper import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TickerCIKMapper",
    "get_ticker_mapper",
    "ticker_to_cik",
    "ticker_to_cik_padded",
]
