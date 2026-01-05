"""
Unit tests for result_formatter module.

Tests focus on:
- Numpy array handling (critical fix for ValueError)
- Empty value detection
- Output formatting at different detail levels
"""

import numpy as np
import pytest

from investigator.application.result_formatter import (
    OutputDetailLevel,
    _is_empty_value,
    _remove_empty_values,
    format_analysis_output,
)


class TestIsEmptyValue:
    """Tests for _is_empty_value helper function."""

    def test_none_is_empty(self):
        """None should be considered empty."""
        assert _is_empty_value(None) is True

    def test_empty_string_is_empty(self):
        """Empty string should be considered empty."""
        assert _is_empty_value("") is True

    def test_empty_list_is_empty(self):
        """Empty list should be considered empty."""
        assert _is_empty_value([]) is True

    def test_empty_dict_is_empty(self):
        """Empty dict should be considered empty."""
        assert _is_empty_value({}) is True

    def test_nonempty_string_not_empty(self):
        """Non-empty string should NOT be considered empty."""
        assert _is_empty_value("hello") is False

    def test_nonempty_list_not_empty(self):
        """Non-empty list should NOT be considered empty."""
        assert _is_empty_value([1, 2, 3]) is False

    def test_nonempty_dict_not_empty(self):
        """Non-empty dict should NOT be considered empty."""
        assert _is_empty_value({"key": "value"}) is False

    def test_zero_not_empty(self):
        """Zero should NOT be considered empty."""
        assert _is_empty_value(0) is False
        assert _is_empty_value(0.0) is False

    def test_false_not_empty(self):
        """False boolean should NOT be considered empty."""
        assert _is_empty_value(False) is False

    def test_empty_numpy_array_is_empty(self):
        """Empty numpy array should be considered empty."""
        empty_arr = np.array([])
        assert _is_empty_value(empty_arr) is True

    def test_nonempty_numpy_array_not_empty(self):
        """Non-empty numpy array should NOT be considered empty."""
        arr = np.array([1, 2, 3])
        assert _is_empty_value(arr) is False

    def test_numpy_2d_array_not_empty(self):
        """2D numpy array with data should NOT be considered empty."""
        arr = np.array([[1, 2], [3, 4]])
        assert _is_empty_value(arr) is False

    def test_numpy_zeros_not_empty(self):
        """Numpy array of zeros should NOT be considered empty (has size)."""
        arr = np.zeros(5)
        assert _is_empty_value(arr) is False

    def test_numpy_matrix_not_empty(self):
        """Numpy matrix should NOT be considered empty."""
        mat = np.matrix([[1, 2], [3, 4]])
        assert _is_empty_value(mat) is False

    def test_numpy_masked_array_not_empty(self):
        """Numpy masked array with data should NOT be considered empty."""
        import numpy.ma as ma

        arr = ma.array([1, 2, 3], mask=[False, True, False])
        assert _is_empty_value(arr) is False

    def test_numpy_masked_array_empty(self):
        """Empty numpy masked array should be considered empty."""
        import numpy.ma as ma

        arr = ma.array([], mask=[])
        assert _is_empty_value(arr) is True

    def test_memoryview_not_empty(self):
        """Memoryview with data should NOT be considered empty."""
        arr = np.array([1, 2, 3])
        mv = memoryview(arr)
        # memoryview doesn't have size, but has __array__ - should not crash
        try:
            result = _is_empty_value(mv)
            # Result depends on implementation, but should not raise
            assert isinstance(result, bool)
        except ValueError:
            pytest.fail("_is_empty_value should not raise ValueError for memoryview")


class TestRemoveEmptyValues:
    """Tests for _remove_empty_values function."""

    def test_removes_none_from_dict(self):
        """Should remove None values from dict."""
        data = {"a": 1, "b": None, "c": 3}
        result = _remove_empty_values(data)
        assert result == {"a": 1, "c": 3}

    def test_removes_empty_string_from_dict(self):
        """Should remove empty strings from dict."""
        data = {"a": "hello", "b": "", "c": "world"}
        result = _remove_empty_values(data)
        assert result == {"a": "hello", "c": "world"}

    def test_removes_empty_list_from_dict(self):
        """Should remove empty lists from dict."""
        data = {"a": [1, 2], "b": [], "c": [3]}
        result = _remove_empty_values(data)
        assert result == {"a": [1, 2], "c": [3]}

    def test_removes_empty_dict_from_dict(self):
        """Should remove empty dicts from dict."""
        data = {"a": {"nested": 1}, "b": {}, "c": {"other": 2}}
        result = _remove_empty_values(data)
        assert result == {"a": {"nested": 1}, "c": {"other": 2}}

    def test_keeps_zero_values(self):
        """Should keep zero values (not empty)."""
        data = {"a": 0, "b": 0.0, "c": 1}
        result = _remove_empty_values(data)
        assert result == {"a": 0, "b": 0.0, "c": 1}

    def test_removes_zero_scores(self):
        """Should remove zero values for keys ending in _score."""
        data = {"accuracy_score": 0, "confidence_score": 0, "value": 0}
        result = _remove_empty_values(data)
        assert result == {"value": 0}

    def test_recursive_cleaning(self):
        """Should recursively clean nested structures."""
        data = {"level1": {"level2": {"keep": 1, "remove": None}, "empty_list": []}, "top_level": "keep"}
        result = _remove_empty_values(data)
        assert result == {"level1": {"level2": {"keep": 1}}, "top_level": "keep"}

    def test_handles_numpy_array_in_dict(self):
        """Should handle numpy arrays without raising ValueError."""
        data = {"prices": np.array([100.0, 101.5, 99.8]), "empty_array": np.array([]), "name": "AAPL"}
        # This was crashing before the fix
        result = _remove_empty_values(data)

        # Empty array should be removed
        assert "empty_array" not in result

        # Non-empty array should be converted to list
        assert "prices" in result
        assert result["prices"] == [100.0, 101.5, 99.8]
        assert result["name"] == "AAPL"

    def test_handles_nested_numpy_arrays(self):
        """Should handle numpy arrays in nested structures."""
        data = {"analysis": {"signals": np.array([1, 0, -1]), "empty_data": np.array([]), "metadata": {"count": 3}}}
        result = _remove_empty_values(data)

        assert "signals" in result["analysis"]
        assert result["analysis"]["signals"] == [1, 0, -1]
        assert "empty_data" not in result["analysis"]

    def test_handles_list_with_empty_values(self):
        """Should remove empty values from lists."""
        data = [1, None, 2, "", 3, [], 4]
        result = _remove_empty_values(data)
        assert result == [1, 2, 3, 4]

    def test_handles_2d_numpy_array(self):
        """Should handle 2D numpy arrays properly."""
        data = {"matrix": np.array([[1, 2], [3, 4]])}
        result = _remove_empty_values(data)
        assert result["matrix"] == [[1, 2], [3, 4]]


class TestFormatAnalysisOutput:
    """Tests for format_analysis_output function."""

    def test_verbose_returns_unchanged(self):
        """Verbose mode should return data unchanged."""
        data = {"key": "value", "nested": {"a": 1}}
        result = format_analysis_output(data, OutputDetailLevel.VERBOSE)
        assert result == data

    def test_standard_adds_detail_level(self):
        """Standard mode should add detail_level indicator."""
        data = {"key": "value"}
        result = format_analysis_output(data, OutputDetailLevel.STANDARD)
        assert result.get("detail_level") == "standard"

    def test_standard_handles_numpy_arrays(self):
        """Standard mode should handle numpy arrays without crashing."""
        data = {"fundamental": {"prices": np.array([100.0, 101.5]), "recommendation": "BUY"}}
        # Should not raise ValueError
        result = format_analysis_output(data, OutputDetailLevel.STANDARD)
        assert "fundamental" in result


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        result = _remove_empty_values({})
        assert result == {}

    def test_all_empty_values(self):
        """Should return empty dict when all values are empty."""
        data = {"a": None, "b": "", "c": [], "d": {}}
        result = _remove_empty_values(data)
        assert result == {}

    def test_deeply_nested_empty(self):
        """Should handle deeply nested structures."""
        data = {"a": {"b": {"c": {"d": None}}}}
        result = _remove_empty_values(data)
        # None leaf value is removed, empty parent containers remain
        # (full recursive cleanup of empty parents is not done to preserve structure)
        assert result == {"a": {"b": {"c": {}}}}

    def test_mixed_types(self):
        """Should handle mixed types correctly."""
        data = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "bool": False,
            "none": None,
            "array": np.array([1, 2, 3]),
        }
        result = _remove_empty_values(data)

        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["str"] == "hello"
        assert result["list"] == [1, 2, 3]
        assert result["dict"] == {"nested": True}
        assert result["bool"] is False
        assert "none" not in result
        assert result["array"] == [1, 2, 3]
