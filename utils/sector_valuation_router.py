"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.domain.services.valuation.sector_valuation_router
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (Sector Valuation Router to Domain Services Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The sector valuation router (294 lines) is now properly located in the
domain services layer for sector-aware valuation routing (Insurance, Banks, REITs),
following clean architecture principles.
"""

# Re-export from canonical location
from investigator.domain.services.valuation.sector_valuation_router import (
    SectorValuationRouter,
    ValuationResult,
)

__all__ = [
    'SectorValuationRouter',
    'ValuationResult',
]
