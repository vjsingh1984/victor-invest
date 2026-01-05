"""
Unit tests for bounds_checker module.

Tests input validation, output validation, and bounds enforcement.
"""

import pytest

from investigator.domain.services.valuation.bounds_checker import (
    BoundsChecker,
    BoundsValidationResult,
    ValidationIssue,
    ValidationSeverity,
    get_bounds_checker,
)


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_all_severities_exist(self):
        """Test all severity levels are defined."""
        assert ValidationSeverity.ERROR
        assert ValidationSeverity.WARNING
        assert ValidationSeverity.INFO


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_creation(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            field="growth_rate",
            value=1.5,
            severity=ValidationSeverity.ERROR,
            message="Growth rate exceeds maximum",
            suggested_action="Review assumptions",
        )
        assert issue.field == "growth_rate"
        assert issue.value == 1.5
        assert issue.severity == ValidationSeverity.ERROR
        assert "Growth rate" in issue.message


class TestBoundsValidationResult:
    """Tests for BoundsValidationResult dataclass."""

    def test_creation(self):
        """Test creating a validation result."""
        result = BoundsValidationResult(is_valid=True, issues=[], warnings=[])
        assert result.is_valid
        assert len(result.issues) == 0

    def test_has_errors_true(self):
        """Test has_errors when error exists."""
        result = BoundsValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(field="growth_rate", value=2.0, severity=ValidationSeverity.ERROR, message="Too high")
            ],
        )
        assert result.has_errors

    def test_has_errors_false(self):
        """Test has_errors when no errors."""
        result = BoundsValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue(
                    field="growth_rate", value=0.6, severity=ValidationSeverity.WARNING, message="High but allowed"
                )
            ],
        )
        assert not result.has_errors

    def test_has_warnings(self):
        """Test has_warnings detection."""
        result = BoundsValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue(field="pe_ratio", value=120, severity=ValidationSeverity.WARNING, message="High P/E")
            ],
        )
        assert result.has_warnings

    def test_summary_valid(self):
        """Test summary for valid result."""
        result = BoundsValidationResult(is_valid=True, issues=[])
        summary = result.summary()
        assert "VALID" in summary
        assert "0 errors" in summary

    def test_summary_invalid(self):
        """Test summary for invalid result."""
        result = BoundsValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(field="x", value=0, severity=ValidationSeverity.ERROR, message="Error"),
                ValidationIssue(field="y", value=0, severity=ValidationSeverity.WARNING, message="Warning"),
            ],
        )
        summary = result.summary()
        assert "INVALID" in summary
        assert "1 errors" in summary
        assert "1 warnings" in summary


class TestBoundsChecker:
    """Tests for BoundsChecker class."""

    @pytest.fixture
    def checker(self):
        """Create default checker."""
        return BoundsChecker()

    def test_validate_inputs_valid_dcf(self, checker):
        """Test validating valid DCF inputs."""
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": 0.10,
                "discount_rate": 0.12,
                "terminal_growth": 0.03,
                "fcf": 1_000_000_000,
                "shares_outstanding": 500_000_000,
            },
        )
        assert result.is_valid
        assert len([i for i in result.issues if i.severity == ValidationSeverity.ERROR]) == 0

    def test_validate_inputs_invalid_growth_rate(self, checker):
        """Test detecting invalid growth rate."""
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": 1.50,  # 150% - above 100% max
                "discount_rate": 0.12,
                "shares_outstanding": 100,
            },
        )
        assert not result.is_valid
        assert any(i.field == "growth_rate" for i in result.issues)

    def test_validate_inputs_invalid_discount_rate(self, checker):
        """Test detecting invalid discount rate."""
        result = checker.validate_inputs(
            "dcf",
            {
                "discount_rate": 0.30,  # 30% - above 25% max
                "shares_outstanding": 100,
            },
        )
        assert not result.is_valid
        assert any(i.field == "discount_rate" for i in result.issues)

    def test_validate_inputs_warning_high_growth(self, checker):
        """Test warning for high but valid growth rate."""
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": 0.60,  # 60% - above 50% warning threshold
                "discount_rate": 0.10,
                "shares_outstanding": 100,
            },
        )
        # Valid but has warning
        assert result.is_valid
        assert result.has_warnings

    def test_validate_inputs_pe_model(self, checker):
        """Test validating P/E model inputs."""
        result = checker.validate_inputs(
            "pe",
            {
                "eps": 5.0,
                "pe_ratio": 25.0,
                "growth_rate": 0.15,
            },
        )
        assert result.is_valid

    def test_validate_inputs_pe_extreme_ratio(self, checker):
        """Test detecting extreme P/E ratio."""
        result = checker.validate_inputs(
            "pe",
            {
                "eps": 0.10,
                "pe_ratio": 600,  # Above 500 max
            },
        )
        assert not result.is_valid
        assert any(i.field == "pe_ratio" for i in result.issues)

    def test_validate_inputs_ps_model(self, checker):
        """Test validating P/S model inputs."""
        result = checker.validate_inputs(
            "ps",
            {
                "revenue": 10_000_000_000,
                "ps_ratio": 5.0,
                "growth_rate": 0.25,
            },
        )
        assert result.is_valid

    def test_validate_inputs_ggm_model(self, checker):
        """Test validating GGM inputs."""
        result = checker.validate_inputs(
            "ggm",
            {
                "dividend": 1_000_000_000,
                "growth_rate": 0.05,
                "required_return": 0.10,
            },
        )
        assert result.is_valid

    def test_validate_inputs_missing_required(self, checker):
        """Test detecting missing required fields."""
        result = checker.validate_inputs(
            "dcf",
            {
                # Missing discount_rate and shares_outstanding
                "fcf": 1_000_000_000,
            },
        )
        assert not result.is_valid
        assert any(i.field == "discount_rate" for i in result.issues)
        assert any(i.field == "shares_outstanding" for i in result.issues)

    def test_validate_inputs_non_numeric(self, checker):
        """Test detecting non-numeric values."""
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": "not a number",
                "discount_rate": 0.10,
                "shares_outstanding": 100,
            },
        )
        assert not result.is_valid
        assert any(i.field == "growth_rate" and "not a valid number" in i.message for i in result.issues)

    def test_validate_inputs_none_values(self, checker):
        """Test handling None values."""
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": None,  # Should be skipped
                "discount_rate": 0.10,
                "shares_outstanding": 100,
            },
        )
        # None values are skipped, no error for None growth_rate
        # But discount_rate is OK
        assert result.is_valid

    def test_validate_inputs_unknown_model(self, checker):
        """Test with unknown model type."""
        result = checker.validate_inputs(
            "unknown_model",
            {
                "some_field": 100,
            },
        )
        # Unknown model should pass (no bounds defined)
        assert result.is_valid

    def test_validate_output_valid(self, checker):
        """Test validating valid output."""
        result = checker.validate_output(fair_value=150.0, current_price=100.0, model_type="dcf")
        # 1.5x is within bounds
        assert result.is_valid

    def test_validate_output_too_high(self, checker):
        """Test detecting fair value too high."""
        result = checker.validate_output(
            fair_value=600.0, current_price=100.0, model_type="dcf"  # 6x current price  # DCF max is 5x
        )
        # In non-strict mode, this is a warning
        assert result.has_warnings or result.has_errors

    def test_validate_output_too_low(self, checker):
        """Test detecting fair value too low."""
        result = checker.validate_output(
            fair_value=10.0, current_price=100.0, model_type="dcf"  # 0.1x current price  # DCF min is 0.2x
        )
        assert result.has_warnings or result.has_errors

    def test_validate_output_zero_fair_value(self, checker):
        """Test detecting zero/negative fair value."""
        result = checker.validate_output(fair_value=0.0, current_price=100.0)
        assert not result.is_valid
        assert result.has_errors

    def test_validate_output_negative_fair_value(self, checker):
        """Test detecting negative fair value."""
        result = checker.validate_output(fair_value=-50.0, current_price=100.0)
        assert not result.is_valid
        assert result.has_errors

    def test_validate_output_zero_current_price(self, checker):
        """Test detecting zero/negative current price."""
        result = checker.validate_output(fair_value=100.0, current_price=0.0)
        assert not result.is_valid

    def test_validate_output_none_values(self, checker):
        """Test detecting None values in output."""
        result = checker.validate_output(fair_value=None, current_price=100.0)
        assert not result.is_valid

    def test_clamp_to_bounds_above_max(self, checker):
        """Test clamping value above max."""
        value, was_clamped = checker.clamp_to_bounds("dcf", "growth_rate", 1.50)
        assert was_clamped
        assert value == 1.00  # Max for growth_rate

    def test_clamp_to_bounds_below_min(self, checker):
        """Test clamping value below min."""
        value, was_clamped = checker.clamp_to_bounds("dcf", "growth_rate", -0.60)
        assert was_clamped
        assert value == -0.50  # Min for growth_rate

    def test_clamp_to_bounds_within(self, checker):
        """Test value within bounds not clamped."""
        value, was_clamped = checker.clamp_to_bounds("dcf", "growth_rate", 0.25)
        assert not was_clamped
        assert value == 0.25

    def test_clamp_to_bounds_unknown_field(self, checker):
        """Test clamping unknown field returns unchanged."""
        value, was_clamped = checker.clamp_to_bounds("dcf", "unknown_field", 999)
        assert not was_clamped
        assert value == 999

    def test_get_bounds(self, checker):
        """Test getting bounds for a field."""
        bounds = checker.get_bounds("dcf", "growth_rate")
        assert bounds is not None
        assert bounds == (-0.50, 1.00)

    def test_get_bounds_unknown(self, checker):
        """Test getting bounds for unknown field."""
        bounds = checker.get_bounds("dcf", "unknown_field")
        assert bounds is None

    def test_strict_mode(self):
        """Test strict mode converts warnings to errors."""
        checker = BoundsChecker(strict_mode=True)

        # High growth rate normally triggers warning
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": 0.60,  # Above 50% threshold
                "discount_rate": 0.10,
                "shares_outstanding": 100,
            },
        )
        # In strict mode, warning becomes error
        assert not result.is_valid

    def test_custom_input_bounds(self):
        """Test custom input bounds override defaults."""
        custom_bounds = {"dcf": {"growth_rate": (0.0, 0.30)}}  # Tighter bounds
        checker = BoundsChecker(input_bounds=custom_bounds)

        # 40% growth would be valid by default but invalid with custom
        result = checker.validate_inputs(
            "dcf",
            {
                "growth_rate": 0.40,
                "discount_rate": 0.10,
                "shares_outstanding": 100,
            },
        )
        assert not result.is_valid

    def test_custom_fair_value_bounds(self):
        """Test custom fair value ratio bounds."""
        custom_bounds = {"dcf": (0.5, 2.0)}  # Tighter bounds
        checker = BoundsChecker(fair_value_ratio_bounds=custom_bounds)

        # 3x would be valid by default but triggers warning with custom
        result = checker.validate_output(fair_value=300.0, current_price=100.0, model_type="dcf")
        assert result.has_warnings or result.has_errors


class TestBoundsCheckerRuleOf40:
    """Tests for Rule of 40 model bounds."""

    @pytest.fixture
    def checker(self):
        return BoundsChecker()

    def test_validate_rule_of_40_inputs(self, checker):
        """Test validating Rule of 40 inputs."""
        result = checker.validate_inputs(
            "rule_of_40",
            {
                "revenue_growth": 0.30,
                "fcf_margin": 0.15,
                "score": 45,
            },
        )
        assert result.is_valid

    def test_validate_rule_of_40_extreme_growth(self, checker):
        """Test detecting extreme revenue growth."""
        result = checker.validate_inputs(
            "rule_of_40",
            {
                "revenue_growth": 2.50,  # 250% - above 200% max
                "fcf_margin": 0.15,
            },
        )
        assert not result.is_valid


class TestBoundsCheckerEdgeCases:
    """Edge case tests for BoundsChecker."""

    @pytest.fixture
    def checker(self):
        return BoundsChecker()

    def test_empty_inputs(self, checker):
        """Test with empty inputs dict."""
        result = checker.validate_inputs("dcf", {})
        # Should fail due to missing required fields
        assert not result.is_valid

    def test_very_large_values(self, checker):
        """Test with very large but valid values."""
        result = checker.validate_inputs(
            "dcf",
            {
                "fcf": 9e11,  # $900 billion - within bounds
                "revenue": 1e12,
                "discount_rate": 0.10,
                "shares_outstanding": 1e10,
            },
        )
        assert result.is_valid

    def test_model_type_with_no_bounds(self, checker):
        """Test model type not in bounds dict."""
        result = checker.validate_inputs(
            "nonexistent_model",
            {
                "any_field": 12345,
            },
        )
        # Should pass - no bounds to violate
        assert result.is_valid


class TestSingletonBoundsChecker:
    """Tests for singleton instance."""

    def test_singleton(self):
        """Test get_bounds_checker returns singleton."""
        checker1 = get_bounds_checker()
        checker2 = get_bounds_checker()
        assert checker1 is checker2
