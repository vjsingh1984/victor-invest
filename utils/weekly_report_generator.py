#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.infrastructure.reporting.weekly_report_generator
Migration Date: 2025-11-14
Phase: Clean Architecture Migration (Reporting to Infrastructure Layer)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The weekly report generator module (514 lines) is now properly located in the
infrastructure layer for periodic report generation,
following clean architecture principles.
"""

# Re-export from canonical location
from investigator.infrastructure.reporting.weekly_report_generator import (
    WeeklyReportGenerator,
)

__all__ = [
    'WeeklyReportGenerator',
]

# Original copyright notice:
# InvestiGator - Weekly Report Generation Module
# Copyright (c) 2025 Vijaykumar Singh
# Licensed under the Apache License 2.0
