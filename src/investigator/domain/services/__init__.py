"""
Domain Services

Business logic and domain-level utilities.

TD1/TD2/TD3 FIXES: Added FiscalPeriodNormalizer for proper YTD handling.
"""

from investigator.domain.services.data_normalizer import DataNormalizer
from investigator.domain.services.fiscal_period_normalizer import (
    FiscalPeriod,
    FiscalPeriodNormalizer,
    QuarterlyConversionResult,
    compute_q4,
    convert_ytd_to_quarterly,
    get_fiscal_period_normalizer,
)

__all__ = [
    "DataNormalizer",
    # TD1 FIX: FiscalPeriodNormalizer for YTD-to-quarterly conversion
    "FiscalPeriodNormalizer",
    "FiscalPeriod",
    "QuarterlyConversionResult",
    "get_fiscal_period_normalizer",
    "convert_ytd_to_quarterly",
    "compute_q4",
]
