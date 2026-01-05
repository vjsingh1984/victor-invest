"""
Bounds Checker - Input/output bounds validation for valuation models.

Provides:
1. Input validation for valuation model parameters
2. Output bounds checking for fair values
3. Reasonable range enforcement
4. Warning generation for edge cases

Problem being solved:
- Valuation models can produce extreme/unrealistic fair values
- Invalid inputs (negative revenues, extreme growth rates) cause garbage outputs
- No systematic validation of model outputs against reasonable ranges

Solution:
- Pre-validation of all model inputs with reasonable bounds
- Post-validation of model outputs against sanity checks
- Warning system for edge cases that pass but are unusual
- Configurable bounds by model type and sector

Usage:
    from investigator.domain.services.valuation.bounds_checker import BoundsChecker

    checker = BoundsChecker()

    # Validate inputs before running model
    input_result = checker.validate_inputs(
        model_type='dcf',
        inputs={'growth_rate': 0.25, 'discount_rate': 0.10, 'revenue': 1e9}
    )

    # Validate outputs after model runs
    output_result = checker.validate_output(
        fair_value=150.0,
        current_price=100.0,
        model_type='dcf'
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Invalid, cannot proceed
    WARNING = "warning"  # Unusual but allowed
    INFO = "info"  # Notable but normal


@dataclass
class ValidationIssue:
    """A single validation issue."""

    field: str
    value: Any
    severity: ValidationSeverity
    message: str
    suggested_action: Optional[str] = None


@dataclass
class BoundsValidationResult:
    """Result of bounds validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    def summary(self) -> str:
        """Get summary of validation result."""
        error_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        status = "VALID" if self.is_valid else "INVALID"
        return f"{status}: {error_count} errors, {warning_count} warnings"


class BoundsChecker:
    """
    Validates inputs and outputs for valuation models.

    Enforces reasonable bounds on:
    - Growth rates (typically -50% to +100%)
    - Discount rates (typically 5% to 25%)
    - Terminal growth (typically 0% to 5%)
    - Fair value ratios (typically 0.1x to 10x current price)
    - Multiples (P/E 1x-500x, P/S 0.1x-50x, etc.)

    Example:
        checker = BoundsChecker()

        # Check inputs before DCF
        result = checker.validate_inputs('dcf', {
            'growth_rate': 0.25,
            'discount_rate': 0.10,
            'terminal_growth': 0.03,
            'fcf': 1_000_000_000,
        })

        if not result.is_valid:
            for issue in result.issues:
                print(f"{issue.severity.value}: {issue.message}")
    """

    # Default bounds for model inputs
    INPUT_BOUNDS = {
        "dcf": {
            "growth_rate": (-0.50, 1.00),  # -50% to +100%
            "discount_rate": (0.05, 0.25),  # 5% to 25%
            "terminal_growth": (-0.02, 0.05),  # -2% to 5%
            "fcf": (-1e12, 1e12),  # Reasonable FCF range
            "revenue": (0, 1e13),  # Up to $10T
            "shares_outstanding": (1, 1e11),  # At least 1 share
        },
        "pe": {
            "eps": (-1000, 1000),  # EPS range
            "pe_ratio": (1, 500),  # P/E 1x to 500x
            "growth_rate": (-0.50, 1.00),
        },
        "ps": {
            "revenue": (0, 1e13),
            "ps_ratio": (0.1, 50),  # P/S 0.1x to 50x
            "growth_rate": (-0.50, 2.00),  # Higher for growth
        },
        "ggm": {
            "dividend": (0, 1e12),
            "growth_rate": (-0.10, 0.15),  # Conservative for dividends
            "required_return": (0.05, 0.20),
        },
        "ev_ebitda": {
            "ebitda": (-1e12, 1e12),
            "ev_ebitda_multiple": (1, 50),
            "net_debt": (-1e12, 1e12),
        },
        "rule_of_40": {
            "revenue_growth": (-0.50, 2.00),
            "fcf_margin": (-1.00, 0.50),
            "score": (-50, 100),
        },
    }

    # Warning thresholds (unusual but allowed)
    WARNING_THRESHOLDS = {
        "growth_rate": (0.50, "Growth rate > 50% is unusual"),
        "discount_rate": (0.20, "Discount rate > 20% is very high"),
        "pe_ratio": (100, "P/E > 100x is extreme"),
        "ps_ratio": (20, "P/S > 20x is extreme"),
    }

    # Fair value ratio bounds (fair value / current price)
    FAIR_VALUE_RATIO_BOUNDS = {
        "default": (0.10, 10.0),  # 0.1x to 10x current price
        "dcf": (0.20, 5.0),  # DCF typically more conservative
        "pe": (0.25, 4.0),
        "ps": (0.10, 10.0),  # P/S can have wider range
        "ggm": (0.50, 3.0),  # GGM is conservative
    }

    def __init__(
        self,
        input_bounds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        fair_value_ratio_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize bounds checker.

        Args:
            input_bounds: Custom input bounds (overrides defaults)
            fair_value_ratio_bounds: Custom fair value ratio bounds
            strict_mode: If True, warnings become errors
        """
        self.input_bounds = {**self.INPUT_BOUNDS}
        self.fair_value_ratio_bounds = {**self.FAIR_VALUE_RATIO_BOUNDS}
        self.strict_mode = strict_mode

        if input_bounds:
            for model, bounds in input_bounds.items():
                if model in self.input_bounds:
                    self.input_bounds[model].update(bounds)
                else:
                    self.input_bounds[model] = bounds

        if fair_value_ratio_bounds:
            self.fair_value_ratio_bounds.update(fair_value_ratio_bounds)

    def validate_inputs(
        self, model_type: str, inputs: Dict[str, Any], symbol: Optional[str] = None
    ) -> BoundsValidationResult:
        """
        Validate inputs for a valuation model.

        Args:
            model_type: Type of model (dcf, pe, ps, etc.)
            inputs: Dictionary of input parameters
            symbol: Optional symbol for logging

        Returns:
            BoundsValidationResult with validation status and issues
        """
        issues: List[ValidationIssue] = []
        model_bounds = self.input_bounds.get(model_type, {})

        for field, value in inputs.items():
            if value is None:
                continue

            # Check if we have bounds for this field
            if field in model_bounds:
                min_val, max_val = model_bounds[field]

                try:
                    num_value = float(value)

                    # Check bounds
                    if num_value < min_val or num_value > max_val:
                        issues.append(
                            ValidationIssue(
                                field=field,
                                value=num_value,
                                severity=ValidationSeverity.ERROR,
                                message=f"{field}={num_value:.4f} outside bounds [{min_val}, {max_val}]",
                                suggested_action=f"Clamp to bounds or review data source",
                            )
                        )

                    # Check warning thresholds
                    elif field in self.WARNING_THRESHOLDS:
                        threshold, warning_msg = self.WARNING_THRESHOLDS[field]
                        if abs(num_value) > threshold:
                            severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
                            issues.append(
                                ValidationIssue(
                                    field=field,
                                    value=num_value,
                                    severity=severity,
                                    message=warning_msg,
                                    suggested_action="Review assumptions",
                                )
                            )

                except (TypeError, ValueError) as e:
                    issues.append(
                        ValidationIssue(
                            field=field,
                            value=value,
                            severity=ValidationSeverity.ERROR,
                            message=f"{field} is not a valid number: {value}",
                            suggested_action="Provide numeric value",
                        )
                    )

        # Check for required fields that are missing or zero
        self._check_required_fields(model_type, inputs, issues)

        is_valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        if issues and symbol:
            logger.debug(
                f"[{symbol}] {model_type} input validation: "
                f"{sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)} errors, "
                f"{sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)} warnings"
            )

        return BoundsValidationResult(is_valid=is_valid, issues=issues)

    def _check_required_fields(self, model_type: str, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> None:
        """Check for required fields by model type."""
        required_fields = {
            "dcf": ["discount_rate", "shares_outstanding"],
            "pe": ["eps", "pe_ratio"],
            "ps": ["revenue", "ps_ratio"],
            "ggm": ["dividend", "growth_rate", "required_return"],
            "ev_ebitda": ["ebitda", "ev_ebitda_multiple"],
        }

        if model_type in required_fields:
            for field in required_fields[model_type]:
                value = inputs.get(field)
                if value is None:
                    issues.append(
                        ValidationIssue(
                            field=field,
                            value=None,
                            severity=ValidationSeverity.ERROR,
                            message=f"Required field '{field}' is missing for {model_type}",
                            suggested_action="Provide required input",
                        )
                    )

    def validate_output(
        self, fair_value: float, current_price: float, model_type: str = "default", symbol: Optional[str] = None
    ) -> BoundsValidationResult:
        """
        Validate output fair value against reasonable bounds.

        Args:
            fair_value: Calculated fair value
            current_price: Current market price
            model_type: Type of model for model-specific bounds
            symbol: Optional symbol for logging

        Returns:
            BoundsValidationResult with validation status
        """
        issues: List[ValidationIssue] = []

        # Check for invalid values
        if fair_value is None or current_price is None:
            issues.append(
                ValidationIssue(
                    field="fair_value",
                    value=fair_value,
                    severity=ValidationSeverity.ERROR,
                    message="Fair value or current price is None",
                    suggested_action="Check model calculation",
                )
            )
            return BoundsValidationResult(is_valid=False, issues=issues)

        if fair_value <= 0:
            issues.append(
                ValidationIssue(
                    field="fair_value",
                    value=fair_value,
                    severity=ValidationSeverity.ERROR,
                    message=f"Fair value {fair_value:.2f} is non-positive",
                    suggested_action="Review model inputs - may indicate negative FCF or other issues",
                )
            )
            return BoundsValidationResult(is_valid=False, issues=issues)

        if current_price <= 0:
            issues.append(
                ValidationIssue(
                    field="current_price",
                    value=current_price,
                    severity=ValidationSeverity.ERROR,
                    message=f"Current price {current_price:.2f} is non-positive",
                    suggested_action="Verify market data",
                )
            )
            return BoundsValidationResult(is_valid=False, issues=issues)

        # Calculate fair value ratio
        fv_ratio = fair_value / current_price

        # Get bounds for this model type
        bounds = self.fair_value_ratio_bounds.get(model_type, self.fair_value_ratio_bounds["default"])
        min_ratio, max_ratio = bounds

        if fv_ratio < min_ratio:
            severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
            issues.append(
                ValidationIssue(
                    field="fair_value_ratio",
                    value=fv_ratio,
                    severity=severity,
                    message=f"Fair value {fair_value:.2f} is only {fv_ratio:.2f}x current price "
                    f"(below {min_ratio}x threshold)",
                    suggested_action="Review model inputs - fair value may be too conservative",
                )
            )

        elif fv_ratio > max_ratio:
            severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
            issues.append(
                ValidationIssue(
                    field="fair_value_ratio",
                    value=fv_ratio,
                    severity=severity,
                    message=f"Fair value {fair_value:.2f} is {fv_ratio:.2f}x current price "
                    f"(above {max_ratio}x threshold)",
                    suggested_action="Review model inputs - fair value may be too aggressive",
                )
            )

        is_valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        if issues and symbol:
            logger.debug(
                f"[{symbol}] {model_type} output validation: "
                f"fair_value={fair_value:.2f}, ratio={fv_ratio:.2f}x, valid={is_valid}"
            )

        return BoundsValidationResult(is_valid=is_valid, issues=issues)

    def clamp_to_bounds(self, model_type: str, field: str, value: float) -> Tuple[float, bool]:
        """
        Clamp a value to its defined bounds.

        Args:
            model_type: Type of model
            field: Field name
            value: Value to clamp

        Returns:
            Tuple of (clamped_value, was_clamped)
        """
        model_bounds = self.input_bounds.get(model_type, {})

        if field not in model_bounds:
            return (value, False)

        min_val, max_val = model_bounds[field]

        if value < min_val:
            logger.debug(f"Clamping {field}={value:.4f} to min={min_val}")
            return (min_val, True)
        elif value > max_val:
            logger.debug(f"Clamping {field}={value:.4f} to max={max_val}")
            return (max_val, True)

        return (value, False)

    def get_bounds(self, model_type: str, field: str) -> Optional[Tuple[float, float]]:
        """
        Get bounds for a specific model/field combination.

        Args:
            model_type: Type of model
            field: Field name

        Returns:
            Tuple of (min, max) or None if not defined
        """
        model_bounds = self.input_bounds.get(model_type, {})
        return model_bounds.get(field)


# Singleton instance
_bounds_checker: Optional[BoundsChecker] = None


def get_bounds_checker() -> BoundsChecker:
    """Get the singleton BoundsChecker instance."""
    global _bounds_checker
    if _bounds_checker is None:
        _bounds_checker = BoundsChecker()
    return _bounds_checker
