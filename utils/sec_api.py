"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.sec.sec_api
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (SEC Infrastructure to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The SEC API client (281 lines) is now properly located in the
infrastructure layer for SEC data access, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.sec.sec_api import SECApiClient

# Backward compatibility alias
SECAPIClient = SECApiClient

__all__ = [
    'SECApiClient',
    'SECAPIClient',  # Legacy name
]
