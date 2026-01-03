#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.domain.services.valuation.ggm
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (Gordon Growth Model to Domain Services)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The Gordon Growth Model valuation logic (565 lines) is now properly located in the
domain services layer as business logic, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.domain.services.valuation.ggm import GordonGrowthModel

__all__ = [
    "GordonGrowthModel",
]
