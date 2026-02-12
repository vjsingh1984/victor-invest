import math

from investigator.domain.agents.fundamental.formatters import (
    safe_fmt_float,
    safe_fmt_int_comma,
    safe_fmt_pct,
)


def test_safe_fmt_pct_handles_none_and_nan():
    assert safe_fmt_pct(None) == "N/A"
    assert safe_fmt_pct(float("nan")) == "N/A"


def test_safe_fmt_pct_and_float_rounding():
    assert safe_fmt_pct(12.345, 1) == "12.3%"
    assert safe_fmt_float(12.3456, 2) == "12.35"


def test_safe_fmt_int_comma_rounds_and_formats():
    assert safe_fmt_int_comma(1234567.8) == "1,234,568"


def test_safe_fmt_float_handles_non_numeric():
    assert safe_fmt_float("bad") == "N/A"
    assert safe_fmt_float(math.inf) == "N/A"
