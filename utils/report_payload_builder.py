"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.reporting.report_payload_builder
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (Reporting to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The report payload builder module (540 lines) is now properly located in the
infrastructure layer for report data transformation,
following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.reporting.report_payload_builder import (
    ReportPayloadBuilder,
)

__all__ = [
    'ReportPayloadBuilder',
]

# Original description:
# Report Payload Builder
# Transforms raw synthesis agent output into normalized PDF report payloads.
