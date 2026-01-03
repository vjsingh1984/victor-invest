"""
Unit tests for safe_formatters module.

Tests defensive formatting for None, NaN, inf, and edge cases.
"""

import math
import pytest
from investigator.domain.services.safe_formatters import (
    format_currency,
    format_percentage,
    format_number,
    format_ratio,
    safe_round,
    is_valid_number,
)


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_valid_positive_number(self):
        """Test formatting a valid positive number."""
        # Values >= 1000 are formatted without cents by default
        assert format_currency(1234.56) == "$1,235"

    def test_valid_negative_number(self):
        """Test formatting a valid negative number."""
        result = format_currency(-1234.56)
        assert "$" in result
        assert "1,235" in result

    def test_none_value(self):
        """Test that None returns the fallback."""
        assert format_currency(None) == "N/A"

    def test_nan_value(self):
        """Test that NaN returns the fallback."""
        assert format_currency(float('nan')) == "N/A"

    def test_inf_value(self):
        """Test that inf returns the fallback."""
        assert format_currency(float('inf')) == "N/A"

    def test_negative_inf_value(self):
        """Test that -inf returns the fallback."""
        assert format_currency(float('-inf')) == "N/A"

    def test_custom_fallback(self):
        """Test custom fallback string."""
        assert format_currency(None, fallback="--") == "--"

    def test_zero_value(self):
        """Test formatting zero."""
        assert format_currency(0) == "$0.00"

    def test_large_number(self):
        """Test formatting large number."""
        result = format_currency(1_000_000_000)
        assert "$" in result
        assert "1" in result

    def test_string_number(self):
        """Test that string numbers are converted."""
        # Values >= 1000 are formatted without cents
        assert format_currency("1234.56") == "$1,235"

    def test_invalid_string(self):
        """Test that invalid strings return fallback."""
        assert format_currency("not a number") == "N/A"


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_valid_decimal(self):
        """Test formatting decimal as percentage."""
        # By default, does not multiply by 100 - pass multiply_by_100=True for decimals
        assert format_percentage(0.25, multiply_by_100=True) == "25.0%"

    def test_valid_percentage(self):
        """Test formatting already percentage value."""
        assert format_percentage(25.0) == "25.0%"

    def test_none_value(self):
        """Test that None returns fallback."""
        assert format_percentage(None) == "N/A"

    def test_nan_value(self):
        """Test that NaN returns fallback."""
        assert format_percentage(float('nan')) == "N/A"

    def test_inf_value(self):
        """Test that inf returns fallback."""
        assert format_percentage(float('inf')) == "N/A"

    def test_custom_decimals(self):
        """Test custom decimal places."""
        # Need multiply_by_100=True for decimal inputs
        assert format_percentage(0.2567, decimals=2, multiply_by_100=True) == "25.67%"

    def test_negative_percentage(self):
        """Test negative percentage."""
        # Input is already percentage form (-0.15 = -0.15%)
        result = format_percentage(-15.0)
        assert "15" in result
        assert "%" in result

    def test_zero_percentage(self):
        """Test zero percentage."""
        assert format_percentage(0) == "0.0%"


class TestFormatNumber:
    """Tests for format_number function."""

    def test_valid_number(self):
        """Test formatting valid number."""
        assert format_number(1234.567, decimals=2) == "1,234.57"

    def test_none_value(self):
        """Test that None returns fallback."""
        assert format_number(None) == "N/A"

    def test_nan_value(self):
        """Test that NaN returns fallback."""
        assert format_number(float('nan')) == "N/A"

    def test_custom_decimals(self):
        """Test custom decimal places."""
        assert format_number(3.14159, decimals=3) == "3.142"


class TestFormatRatio:
    """Tests for format_ratio function."""

    def test_valid_ratio(self):
        """Test formatting valid ratio."""
        result = format_ratio(2.5)
        assert "2.5" in result

    def test_none_value(self):
        """Test that None returns fallback."""
        assert format_ratio(None) == "N/A"

    def test_nan_value(self):
        """Test that NaN returns fallback."""
        assert format_ratio(float('nan')) == "N/A"


class TestSafeRound:
    """Tests for safe_round function."""

    def test_valid_number(self):
        """Test rounding valid number."""
        assert safe_round(3.14159, 2) == 3.14

    def test_none_value(self):
        """Test that None returns None."""
        assert safe_round(None) is None

    def test_nan_value(self):
        """Test that NaN returns None."""
        assert safe_round(float('nan')) is None

    def test_inf_value(self):
        """Test that inf returns None."""
        assert safe_round(float('inf')) is None

    def test_negative_number(self):
        """Test rounding negative number."""
        assert safe_round(-3.14159, 2) == -3.14

    def test_zero_decimals(self):
        """Test rounding to zero decimals."""
        assert safe_round(3.7, 0) == 4.0


class TestIsValidNumber:
    """Tests for is_valid_number function."""

    def test_valid_integer(self):
        """Test valid integer."""
        assert is_valid_number(42) is True

    def test_valid_float(self):
        """Test valid float."""
        assert is_valid_number(3.14) is True

    def test_none_value(self):
        """Test None is not valid."""
        assert is_valid_number(None) is False

    def test_nan_value(self):
        """Test NaN is not valid."""
        assert is_valid_number(float('nan')) is False

    def test_inf_value(self):
        """Test inf is not valid."""
        assert is_valid_number(float('inf')) is False

    def test_negative_inf(self):
        """Test -inf is not valid."""
        assert is_valid_number(float('-inf')) is False

    def test_string_number(self):
        """Test string that can be converted is valid."""
        assert is_valid_number("42") is True

    def test_invalid_string(self):
        """Test invalid string is not valid."""
        assert is_valid_number("not a number") is False

    def test_empty_string(self):
        """Test empty string is not valid."""
        assert is_valid_number("") is False

    def test_zero(self):
        """Test zero is valid."""
        assert is_valid_number(0) is True

    def test_negative(self):
        """Test negative number is valid."""
        assert is_valid_number(-42) is True


