"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.sec.sec_frame_api
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (SEC Frame API to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The SEC Frame API client (341 lines) is now properly located in the
infrastructure layer for SEC EDGAR Frame API access, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.sec.sec_frame_api import (
    FrameAPIRequest,
    SECFrameAPI,
    get_frame_api,
)

__all__ = [
    'FrameAPIRequest',
    'SECFrameAPI',
    'get_frame_api',
]
