#!/usr/bin/env python3
"""
DEPRECATED: Import shim for backward compatibility.
This module has been migrated to src/investigator/infrastructure/database/

Canonical import:
    from investigator.infrastructure.database.market_data import DatabaseMarketDataFetcher

DO NOT use this shim in new code. It will be archived once all imports are updated.

Migration Status: COMPLETE (src/ version is canonical with enhanced metadata features)
Active Imports: 7 files (8 total imports) need updating (see archive/legacy-utils/README.md)
"""
import warnings

# Re-export from new location
from investigator.infrastructure.database.market_data import DatabaseMarketDataFetcher

warnings.warn(
    "utils.market_data_fetcher is deprecated. Use 'from investigator.infrastructure.database.market_data import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DatabaseMarketDataFetcher"]
