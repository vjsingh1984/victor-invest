#!/usr/bin/env python3
"""
InvestiGator - Analysis Synthesis Module (Backward Compatibility Shim)

This module provides backward compatibility by re-exporting from the
Clean Architecture application layer.

DEPRECATED: Import directly from:
    from investigator.application import InvestmentSynthesizer
    from investigator.domain.models import InvestmentRecommendation
"""

import warnings

# Warn about deprecated import path
warnings.warn(
    "Importing from 'synthesizer' module is deprecated. "
    "Use 'from investigator.application import InvestmentSynthesizer' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from Clean Architecture
from investigator.application import InvestmentSynthesizer
from investigator.domain.models import InvestmentRecommendation

__all__ = ["InvestmentSynthesizer", "InvestmentRecommendation"]
