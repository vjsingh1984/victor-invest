#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.domain.services.valuation.dcf
Migration Date: 2025-11-13
Phase: Phase 2-C-2 (DCF Valuation Migration to Domain Services)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The DCF valuation logic (109KB, ~3,300 lines) is now properly located in the
domain services layer as business logic, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.domain.services.valuation.dcf import DCFValuation

__all__ = [
    "DCFValuation",
]
