"""
Data Validation Service - Pre-flight checks for financial data.

Provides:
1. Metric completeness scoring (0-100%)
2. Outlier detection with configurable thresholds
3. NaN/None propagation guards
4. Cross-metric consistency checks
5. Model-specific applicability validation

Usage:
    from investigator.domain.services.data_validation import DataValidator

    validator = DataValidator()
    result = validator.validate_quarterly_data(data)

    if result.has_critical_issues():
        logger.error(f"Critical issues: {result.issues}")
        return None
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Potential issue, continue with caution
    ERROR = "error"  # Significant issue, may affect results
    CRITICAL = "critical"  # Severe issue, should halt processing


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the data."""

    field: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None
    value: Optional[Any] = None

    def __str__(self) -> str:
        s = f"[{self.severity.value.upper()}] {self.field}: {self.message}"
        if self.suggestion:
            s += f" (Suggestion: {self.suggestion})"
        return s


@dataclass
class ValidationResult:
    """Result of data validation containing all findings."""

    is_valid: bool
    completeness_score: float  # 0-100
    quality_score: float  # 0-100
    issues: List[ValidationIssue] = field(default_factory=list)
    outlier_flags: Dict[str, str] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    valid_fields: List[str] = field(default_factory=list)

    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found."""
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity."""
        return [i for i in self.issues if i.severity == severity]

    def summary(self) -> str:
        """Get a summary of the validation result."""
        issue_counts = {s: 0 for s in ValidationSeverity}
        for issue in self.issues:
            issue_counts[issue.severity] += 1

        return (
            f"Completeness: {self.completeness_score:.1f}%, "
            f"Quality: {self.quality_score:.1f}%, "
            f"Valid: {self.is_valid}, "
            f"Issues: {len(self.issues)} "
            f"(Critical: {issue_counts[ValidationSeverity.CRITICAL]}, "
            f"Error: {issue_counts[ValidationSeverity.ERROR]}, "
            f"Warning: {issue_counts[ValidationSeverity.WARNING]})"
        )


class DataValidator:
    """
    Validates financial data before calculations.

    Provides pre-flight checks to catch data issues before they cause
    downstream errors or produce invalid valuation results.

    Example usage:
        validator = DataValidator()
        result = validator.validate_quarterly_data(quarterly_data)

        if result.has_critical_issues():
            logger.error(f"Critical issues: {result.issues}")
            return None

        if result.completeness_score < 70:
            logger.warning(f"Low data completeness: {result.completeness_score}%")
    """

    # Required metrics for different valuation models
    MODEL_REQUIREMENTS = {
        "dcf": {
            "required": ["free_cash_flow", "revenue", "operating_cash_flow"],
            "optional": ["net_income", "ebitda", "capital_expenditures"],
            "min_quarters": 4,
        },
        "ggm": {
            "required": ["dividends_paid", "net_income"],
            "optional": ["payout_ratio", "dividend_growth_rate"],
            "min_quarters": 4,
        },
        "pe": {
            "required": ["net_income", "shares_outstanding"],
            "optional": ["eps", "earnings_quality_score"],
            "min_quarters": 4,
        },
        "ps": {
            "required": ["revenue", "shares_outstanding"],
            "optional": ["revenue_growth", "gross_margin"],
            "min_quarters": 4,
        },
        "pb": {
            "required": ["book_value", "shares_outstanding"],
            "optional": ["tangible_book_value", "roe"],
            "min_quarters": 2,
        },
        "ev_ebitda": {
            "required": ["ebitda", "enterprise_value"],
            "optional": ["net_debt", "shares_outstanding"],
            "min_quarters": 4,
        },
    }

    # Outlier thresholds for common metrics
    OUTLIER_THRESHOLDS = {
        # Ratios
        "pe_ratio": {"min": 0, "max": 500, "warn_min": 5, "warn_max": 100},
        "ps_ratio": {"min": 0, "max": 100, "warn_min": 0.5, "warn_max": 30},
        "pb_ratio": {"min": 0, "max": 50, "warn_min": 0.5, "warn_max": 15},
        "ev_ebitda": {"min": 0, "max": 100, "warn_min": 3, "warn_max": 30},
        "peg_ratio": {"min": 0, "max": 10, "warn_min": 0.5, "warn_max": 3},
        # Margins (percentages)
        "gross_margin": {"min": -20, "max": 100, "warn_min": 0, "warn_max": 90},
        "operating_margin": {"min": -100, "max": 80, "warn_min": -20, "warn_max": 50},
        "net_margin": {"min": -200, "max": 80, "warn_min": -50, "warn_max": 40},
        "fcf_margin": {"min": -100, "max": 80, "warn_min": -30, "warn_max": 40},
        # Growth rates (percentages)
        "revenue_growth": {"min": -80, "max": 500, "warn_min": -30, "warn_max": 100},
        "earnings_growth": {"min": -100, "max": 1000, "warn_min": -50, "warn_max": 200},
        # Other ratios
        "debt_to_equity": {"min": 0, "max": 50, "warn_min": 0, "warn_max": 5},
        "current_ratio": {"min": 0, "max": 20, "warn_min": 0.5, "warn_max": 5},
        "payout_ratio": {"min": 0, "max": 200, "warn_min": 0, "warn_max": 100},
    }

    def __init__(self, custom_thresholds: Optional[Dict] = None):
        """
        Initialize the validator with optional custom thresholds.

        Args:
            custom_thresholds: Optional dict to override default outlier thresholds
        """
        self.thresholds = {**self.OUTLIER_THRESHOLDS}
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def is_valid_number(self, value: Any) -> bool:
        """Check if a value is a valid, finite number."""
        if value is None:
            return False
        if isinstance(value, bool):
            return False
        try:
            num = float(value)
            return not (math.isnan(num) or math.isinf(num))
        except (TypeError, ValueError):
            return False

    def validate_quarterly_data(
        self,
        data: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate quarterly financial data.

        Args:
            data: Dictionary of financial metrics
            required_fields: Override default required fields
            context: Additional context (sector, industry, etc.)

        Returns:
            ValidationResult with completeness score, issues, and outlier flags
        """
        issues: List[ValidationIssue] = []
        missing_fields: List[str] = []
        valid_fields: List[str] = []
        outlier_flags: Dict[str, str] = {}

        if not data:
            return ValidationResult(
                is_valid=False,
                completeness_score=0.0,
                quality_score=0.0,
                issues=[
                    ValidationIssue(
                        field="data",
                        severity=ValidationSeverity.CRITICAL,
                        message="Data is empty or None",
                        suggestion="Ensure data is fetched and parsed correctly",
                    )
                ],
                missing_fields=[],
                valid_fields=[],
            )

        # Check for None/NaN values
        for field_name, value in data.items():
            if value is None:
                missing_fields.append(field_name)
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                missing_fields.append(field_name)
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Field contains NaN/Inf value",
                        suggestion="Check upstream data source for calculation errors",
                        value=value,
                    )
                )
            else:
                valid_fields.append(field_name)

        # Calculate completeness
        total_fields = len(data)
        valid_count = len(valid_fields)
        completeness_score = (valid_count / total_fields * 100) if total_fields > 0 else 0

        # Check required fields if specified
        if required_fields:
            for req_field in required_fields:
                if req_field not in valid_fields:
                    issues.append(
                        ValidationIssue(
                            field=req_field,
                            severity=ValidationSeverity.ERROR,
                            message=f"Required field is missing or invalid",
                            suggestion=f"Ensure {req_field} is available in source data",
                        )
                    )

        # Check for outliers
        for field_name, value in data.items():
            if field_name in self.thresholds and self.is_valid_number(value):
                thresholds = self.thresholds[field_name]
                num_value = float(value)

                # Hard limits
                if num_value < thresholds.get("min", float("-inf")):
                    outlier_flags[field_name] = "below_min"
                    issues.append(
                        ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.ERROR,
                            message=f"Value {num_value:.2f} below minimum threshold {thresholds['min']}",
                            suggestion="Verify data accuracy or check for data entry errors",
                            value=num_value,
                        )
                    )
                elif num_value > thresholds.get("max", float("inf")):
                    outlier_flags[field_name] = "above_max"
                    issues.append(
                        ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.ERROR,
                            message=f"Value {num_value:.2f} exceeds maximum threshold {thresholds['max']}",
                            suggestion="Data may be erroneous or exceptional",
                            value=num_value,
                        )
                    )
                # Warning limits
                elif num_value < thresholds.get("warn_min", float("-inf")):
                    outlier_flags[field_name] = "low_warning"
                    issues.append(
                        ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.WARNING,
                            message=f"Value {num_value:.2f} is unusually low",
                            value=num_value,
                        )
                    )
                elif num_value > thresholds.get("warn_max", float("inf")):
                    outlier_flags[field_name] = "high_warning"
                    issues.append(
                        ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.WARNING,
                            message=f"Value {num_value:.2f} is unusually high",
                            value=num_value,
                        )
                    )

        # Calculate quality score
        quality_score = self._calculate_quality_score(completeness_score, len(outlier_flags), len(issues))

        # Determine overall validity
        is_valid = not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues)

        return ValidationResult(
            is_valid=is_valid,
            completeness_score=round(completeness_score, 1),
            quality_score=round(quality_score, 1),
            issues=issues,
            outlier_flags=outlier_flags,
            missing_fields=missing_fields,
            valid_fields=valid_fields,
        )

    def validate_for_model(self, data: Dict[str, Any], model_type: str) -> Tuple[bool, float, List[str]]:
        """
        Validate data for a specific valuation model.

        Args:
            data: Financial data dictionary
            model_type: Model type (dcf, ggm, pe, ps, pb, ev_ebitda)

        Returns:
            Tuple of (is_applicable, confidence_adjustment, missing_required)
            - is_applicable: True if model can run with available data
            - confidence_adjustment: Multiplier for confidence (0.5-1.0)
            - missing_required: List of missing required fields
        """
        model_type = model_type.lower()
        if model_type not in self.MODEL_REQUIREMENTS:
            logger.warning(f"Unknown model type: {model_type}")
            return (True, 1.0, [])

        requirements = self.MODEL_REQUIREMENTS[model_type]
        required_fields = requirements.get("required", [])
        optional_fields = requirements.get("optional", [])

        # Check required fields
        missing_required = []
        for field in required_fields:
            if not self._has_valid_value(data.get(field)):
                missing_required.append(field)

        # Check optional fields for confidence adjustment
        present_optional = sum(1 for field in optional_fields if self._has_valid_value(data.get(field)))

        # Calculate metrics
        required_present = len(required_fields) - len(missing_required)
        required_ratio = required_present / len(required_fields) if required_fields else 1
        optional_ratio = present_optional / len(optional_fields) if optional_fields else 1

        # Model is applicable if at least 50% of required fields present
        is_applicable = required_ratio >= 0.5

        # Confidence adjustment based on data completeness
        # Range: 0.5 (barely applicable) to 1.0 (fully complete)
        confidence_adjustment = 0.7 * required_ratio + 0.3 * optional_ratio
        confidence_adjustment = max(0.5, min(1.0, confidence_adjustment))

        return (is_applicable, confidence_adjustment, missing_required)

    def _has_valid_value(self, value: Any) -> bool:
        """Check if value is valid (not None, NaN, or Inf)."""
        return self.is_valid_number(value)

    def _calculate_quality_score(self, completeness: float, outlier_count: int, issue_count: int) -> float:
        """Calculate aggregate quality score from components."""
        base_score = completeness

        # Deduct for outliers (max 25% penalty)
        outlier_penalty = min(outlier_count * 5, 25)

        # Deduct for issues (max 20% penalty)
        issue_penalty = min(issue_count * 3, 20)

        return max(0, base_score - outlier_penalty - issue_penalty)

    def validate_consistency(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Check cross-metric consistency.

        Validates that related metrics are consistent with each other.
        For example, net_income should roughly equal revenue * net_margin.

        Args:
            data: Financial data dictionary

        Returns:
            List of consistency validation issues
        """
        issues = []

        # Check margin consistency
        if all(self._has_valid_value(data.get(f)) for f in ["revenue", "net_income", "net_margin"]):
            revenue = float(data["revenue"])
            net_income = float(data["net_income"])
            net_margin = float(data["net_margin"])

            if revenue > 0:
                calculated_margin = (net_income / revenue) * 100
                if abs(calculated_margin - net_margin) > 5:  # 5% tolerance
                    issues.append(
                        ValidationIssue(
                            field="net_margin",
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Net margin ({net_margin:.1f}%) inconsistent with "
                                f"calculated ({calculated_margin:.1f}%)"
                            ),
                            suggestion="Check if metrics are from same period",
                        )
                    )

        # Check FCF consistency
        if all(
            self._has_valid_value(data.get(f))
            for f in ["free_cash_flow", "operating_cash_flow", "capital_expenditures"]
        ):
            fcf = float(data["free_cash_flow"])
            ocf = float(data["operating_cash_flow"])
            capex = abs(float(data["capital_expenditures"]))

            calculated_fcf = ocf - capex
            if abs(fcf - calculated_fcf) > abs(fcf) * 0.1:  # 10% tolerance
                issues.append(
                    ValidationIssue(
                        field="free_cash_flow",
                        severity=ValidationSeverity.INFO,
                        message=(f"FCF ({fcf:,.0f}) differs from OCF-CapEx ({calculated_fcf:,.0f})"),
                        suggestion="May include other adjustments",
                    )
                )

        # Check EPS consistency
        if all(self._has_valid_value(data.get(f)) for f in ["net_income", "shares_outstanding", "eps"]):
            net_income = float(data["net_income"])
            shares = float(data["shares_outstanding"])
            eps = float(data["eps"])

            if shares > 0:
                calculated_eps = net_income / shares
                if abs(calculated_eps - eps) > 0.5:  # $0.50 tolerance
                    issues.append(
                        ValidationIssue(
                            field="eps",
                            severity=ValidationSeverity.WARNING,
                            message=(f"EPS ({eps:.2f}) inconsistent with " f"calculated ({calculated_eps:.2f})"),
                            suggestion="Check if using diluted shares",
                        )
                    )

        return issues


# Singleton instance
_validator: Optional[DataValidator] = None


def get_data_validator() -> DataValidator:
    """Get the singleton DataValidator instance."""
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator
