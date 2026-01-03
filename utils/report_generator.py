#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.reporting.report_generator
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (Reporting to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The report generator module (2,883 lines) is now properly located in the
infrastructure layer for external adapters (PDF generation via ReportLab),
following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.reporting.report_generator import (
    PDFReportGenerator,
    ReportConfig,
)

__all__ = [
    'PDFReportGenerator',
    'ReportConfig',
]

# Original copyright notice:
# InvestiGator - PDF Report Generation Module
# Copyright (c) 2025 Vijaykumar Singh
# Licensed under the Apache License 2.0
