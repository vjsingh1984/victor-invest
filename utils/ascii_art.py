"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.ui.ascii_art
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (ASCII Art to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The ASCII art module (296 lines) is now properly located in the
infrastructure layer for UI components and terminal formatting,
following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.ui.ascii_art import ASCIIArt

__all__ = [
    'ASCIIArt',
]
