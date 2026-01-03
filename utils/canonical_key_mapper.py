#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.sec.canonical_mapper
Migration Date: 2025-11-13
Phase: Phase 2 - Option B (Simple Utility Migrations)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.
"""

# Re-export from canonical location
from investigator.infrastructure.sec.canonical_mapper import (
    CanonicalKeyMapper,
    get_canonical_mapper,
)

__all__ = [
    'CanonicalKeyMapper',
    'get_canonical_mapper',
]
