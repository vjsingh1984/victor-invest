"""Prompt-safe formatter helpers for fundamental agent narratives."""

from __future__ import annotations

from typing import Any

from investigator.domain.services.data_normalizer import round_for_prompt


def safe_fmt_pct(value: Any, decimals: int = 1) -> str:
    """Safe percentage formatting with None/NaN handling."""
    if value is None:
        return "N/A"
    rounded = round_for_prompt(value, decimals)
    if rounded is None:
        return "N/A"
    return f"{rounded:.{decimals}f}%"


def safe_fmt_float(value: Any, decimals: int = 2) -> str:
    """Safe float formatting with None/NaN handling."""
    if value is None:
        return "N/A"
    rounded = round_for_prompt(value, decimals)
    if rounded is None:
        return "N/A"
    return f"{rounded:.{decimals}f}"


def safe_fmt_int_comma(value: Any) -> str:
    """Safe integer formatting with comma separators."""
    if value is None:
        return "N/A"
    rounded = round_for_prompt(value, 0)
    if rounded is None:
        return "N/A"
    return f"{rounded:,.0f}"
