"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.indicators.technical_indicators
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (Technical Indicators to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The technical indicators calculator (322 lines) is now properly located in the
infrastructure layer for technical analysis calculations, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.indicators.technical_indicators import (
    TechnicalIndicatorCalculator,
    get_technical_calculator,
)

__all__ = [
    'TechnicalIndicatorCalculator',
    'get_technical_calculator',
]
