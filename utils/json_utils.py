#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.utils.json_utils
Migration Date: 2025-11-13
Phase: Phase 2 - Option B (Simple Utility Migrations)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.
"""

# Re-export from canonical location
from investigator.infrastructure.utils.json_utils import (
    safe_json_dumps,
    safe_json_loads,
    extract_json_from_text,
)

__all__ = [
    'safe_json_dumps',
    'safe_json_loads',
    'extract_json_from_text',
]
