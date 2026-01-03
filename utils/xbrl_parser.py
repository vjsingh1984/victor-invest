"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.sec.xbrl_parser
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (XBRL Parser to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The XBRL parser (207 lines) is now properly located in the
infrastructure layer for SEC data parsing, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.sec.xbrl_parser import XBRLParser

__all__ = [
    'XBRLParser',
]
