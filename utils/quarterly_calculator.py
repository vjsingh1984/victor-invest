#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.domain.services.quarterly_processor
Migration Date: 2025-11-17

This file remains as a compatibility shim to avoid breaking existing imports.
All new code should import from the canonical location.

IMPORTANT: Do NOT modify this file. All changes should be made to:
    src/investigator/domain/services/quarterly_processor.py
"""

# Re-export from canonical location
from investigator.domain.services.quarterly_processor import (
    compute_missing_quarter,
    extract_nested_value,
    convert_ytd_to_quarterly,
    get_rolling_ttm_periods,
    analyze_quarterly_patterns,
    validate_computed_quarter,
    _find_consecutive_quarters,
)

__all__ = [
    'compute_missing_quarter',
    'extract_nested_value',
    'convert_ytd_to_quarterly',
    'get_rolling_ttm_periods',
    'analyze_quarterly_patterns',
    'validate_computed_quarter',
    '_find_consecutive_quarters',
]
