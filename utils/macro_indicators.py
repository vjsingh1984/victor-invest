#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.external.fred.macro_indicators
Migration Date: 2025-11-13
Phase: Phase 3-B-2 (High-Priority Infrastructure Layer Migration)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The FRED macroeconomic indicators infrastructure (559 lines) is now properly located in the
infrastructure layer for external API communication, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.external.fred import (
    MacroIndicatorsFetcher,
    format_indicator_for_display,
    get_stock_db_manager,
)

__all__ = [
    'MacroIndicatorsFetcher',
    'format_indicator_for_display',
    'get_stock_db_manager',
]
