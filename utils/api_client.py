#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.http.api_client
Migration Date: 2025-11-13
Phase: Phase 3-B-2 (High-Priority Infrastructure Layer Migration)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The HTTP client infrastructure (510 lines) is now properly located in the
infrastructure layer for external API communication, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.http import (
    BaseAPIClient,
    SECAPIClient,
    OllamaAPIClient,
    rate_limit,
    retry_on_failure,
)

__all__ = [
    'BaseAPIClient',
    'SECAPIClient',
    'OllamaAPIClient',
    'rate_limit',
    'retry_on_failure',
]
