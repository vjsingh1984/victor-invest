#!/usr/bin/env python3
"""
DEPRECATED: Import shim for backward compatibility.
This module has been migrated to src/investigator/infrastructure/sec/

Canonical import:
    from investigator.infrastructure.sec.data_strategy import SECDataStrategy, get_fiscal_period_strategy

DO NOT use this shim in new code. It will be archived once all imports are updated.

Migration Status: COMPLETE (src/ version is canonical, includes Q1 fiscal year fix)
Active Imports: 2 files (4 total imports) need updating (see docs/UTILS_MIGRATION_STRATEGY.md)
"""
import warnings

# Re-export from new location
from investigator.infrastructure.sec.data_strategy import (
    SECDataStrategy,
    get_fiscal_period_strategy,
    _fiscal_period_to_int,
)

warnings.warn(
    "utils.sec_data_strategy is deprecated. Use 'from investigator.infrastructure.sec.data_strategy import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SECDataStrategy",
    "get_fiscal_period_strategy",
    "_fiscal_period_to_int",
]
