"""
Safe Formatters - Defensive formatting functions for financial data.

All functions handle None, NaN, Inf gracefully with consistent "N/A" output.
Provides centralized, well-tested formatting to prevent crashes from bad data.

Usage:
    from investigator.domain.services.safe_formatters import (
        format_currency,
        format_percentage,
        format_number,
        safe_round,
        round_for_prompt
    )
"""

import math
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Optional


def is_valid_number(value: Any) -> bool:
    """
    Check if value is a valid, finite number.

    Args:
        value: Any value to check

    Returns:
        True if value is a valid finite number, False otherwise

    Examples:
        >>> is_valid_number(123.45)
        True
        >>> is_valid_number(None)
        False
        >>> is_valid_number(float('nan'))
        False
        >>> is_valid_number(float('inf'))
        False
    """
    if value is None:
        return False

    try:
        # Reject booleans (they convert to 0/1 which is misleading)
        if isinstance(value, bool):
            return False

        num = float(value)
        return not (math.isnan(num) or math.isinf(num))
    except (TypeError, ValueError):
        return False


def safe_round(value: Any, decimals: int = 2) -> Optional[float]:
    """
    Round a value safely, returning None for invalid inputs.

    Handles:
    - None values
    - NaN/Inf floats
    - Non-numeric types
    - Boolean values
    - Very large numbers

    Args:
        value: Value to round
        decimals: Number of decimal places

    Returns:
        Rounded float or None if input is invalid

    Examples:
        >>> safe_round(123.456, 2)
        123.46
        >>> safe_round(None, 2)
        None
        >>> safe_round(float('nan'), 2)
        None
    """
    if not is_valid_number(value):
        return None

    try:
        num = Decimal(str(float(value)))

        # Handle very large numbers - reduce precision
        if abs(num) >= 1_000_000:
            decimals = min(decimals, 0)

        quantizer = Decimal(10) ** -decimals
        rounded = num.quantize(quantizer, rounding=ROUND_HALF_UP)
        return float(rounded)

    except (InvalidOperation, ValueError, OverflowError):
        return None


def round_for_prompt(value: Any, decimals: int = 2) -> Optional[float]:
    """
    Safe rounding for LLM prompt values.

    Backward compatible wrapper around safe_round for existing code.

    Args:
        value: Value to round
        decimals: Number of decimal places

    Returns:
        Rounded float or None if input is invalid
    """
    return safe_round(value, decimals)


def format_currency(value: Any, include_cents: bool = True, compact: bool = False, fallback: str = "N/A") -> str:
    """
    Format value as currency with safe handling.

    Args:
        value: Numeric value or None
        include_cents: Include cents for values < 1000
        compact: Use K/M/B suffixes for large numbers
        fallback: String to return for invalid values

    Returns:
        Formatted currency string or fallback

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234567890, compact=True)
        '$1.23B'
        >>> format_currency(None)
        'N/A'
        >>> format_currency(float('nan'))
        'N/A'
        >>> format_currency(-500.50)
        '-$500.50'
    """
    if not is_valid_number(value):
        return fallback

    num = float(value)
    sign = "-" if num < 0 else ""
    abs_val = abs(num)

    if compact:
        if abs_val >= 1e12:
            return f"{sign}${abs_val / 1e12:.2f}T"
        elif abs_val >= 1e9:
            return f"{sign}${abs_val / 1e9:.2f}B"
        elif abs_val >= 1e6:
            return f"{sign}${abs_val / 1e6:.2f}M"
        elif abs_val >= 1e3:
            return f"{sign}${abs_val / 1e3:.2f}K"

    if include_cents and abs_val < 1000:
        return f"{sign}${abs_val:,.2f}"
    else:
        return f"{sign}${abs_val:,.0f}"


def format_percentage(value: Any, decimals: int = 1, fallback: str = "N/A", multiply_by_100: bool = False) -> str:
    """
    Format value as percentage with safe handling.

    Args:
        value: Numeric value or None
        decimals: Decimal places to show
        fallback: String for invalid values
        multiply_by_100: If True, multiply value by 100 first (for decimals like 0.15)

    Returns:
        Formatted percentage string or fallback

    Examples:
        >>> format_percentage(15.6)
        '15.6%'
        >>> format_percentage(0.156, multiply_by_100=True)
        '15.6%'
        >>> format_percentage(None)
        'N/A'
    """
    if not is_valid_number(value):
        return fallback

    num = float(value)

    if multiply_by_100:
        num *= 100

    return f"{num:.{decimals}f}%"


def format_number(value: Any, decimals: int = 2, fallback: str = "N/A", thousands_separator: bool = True) -> str:
    """
    Format number with safe handling.

    Args:
        value: Numeric value or None
        decimals: Decimal places
        fallback: String for invalid values
        thousands_separator: Use comma separators

    Returns:
        Formatted number string or fallback

    Examples:
        >>> format_number(1234567.89)
        '1,234,567.89'
        >>> format_number(None)
        'N/A'
    """
    if not is_valid_number(value):
        return fallback

    num = float(value)

    if thousands_separator:
        return f"{num:,.{decimals}f}"
    else:
        return f"{num:.{decimals}f}"


def format_ratio(value: Any, decimals: int = 2, fallback: str = "N/A") -> str:
    """
    Format ratio (e.g., P/E, D/E) with safe handling.

    Special handling for extreme values:
    - Values > 1000 shown as ">1000"
    - Values < -1000 shown as "<-1000"

    Args:
        value: Numeric value or None
        decimals: Decimal places
        fallback: String for invalid values

    Returns:
        Formatted ratio string or fallback
    """
    if not is_valid_number(value):
        return fallback

    num = float(value)

    if num > 1000:
        return ">1000"
    elif num < -1000:
        return "<-1000"

    return f"{num:.{decimals}f}"


def format_shares(value: Any, fallback: str = "N/A") -> str:
    """
    Format share count with appropriate scaling (K/M/B).

    Args:
        value: Share count or None
        fallback: String for invalid values

    Returns:
        Formatted share count or fallback

    Examples:
        >>> format_shares(1500000000)
        '1.50B'
        >>> format_shares(5000000)
        '5.00M'
    """
    if not is_valid_number(value):
        return fallback

    num = float(value)
    abs_val = abs(num)
    sign = "-" if num < 0 else ""

    if abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.2f}K"

    return f"{sign}{abs_val:.0f}"


def format_int_with_commas(value: Any, fallback: str = "N/A") -> str:
    """
    Format integer with comma separators.

    Args:
        value: Integer value or None
        fallback: String for invalid values

    Returns:
        Formatted integer string or fallback

    Examples:
        >>> format_int_with_commas(1234567)
        '1,234,567'
    """
    if not is_valid_number(value):
        return fallback

    return f"{int(float(value)):,}"


# Convenience aliases for backward compatibility
def _fmt_currency(value: Any) -> str:
    """Backward compatible currency formatter."""
    return format_currency(value, include_cents=True, fallback="N/A")


def _fmt_pct(value: Any) -> str:
    """Backward compatible percentage formatter."""
    return format_percentage(value, decimals=2, fallback="N/A")


def _fmt_float(value: Any, decimals: int = 2) -> str:
    """Backward compatible float formatter."""
    return format_number(value, decimals=decimals, fallback="N/A", thousands_separator=False)


def _fmt_int_comma(value: Any) -> str:
    """Backward compatible integer formatter with commas."""
    return format_int_with_commas(value, fallback="N/A")
