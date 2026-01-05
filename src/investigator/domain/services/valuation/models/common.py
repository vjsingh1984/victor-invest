"""
Shared helpers for multiple-based valuation models.

Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.common
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from investigator.domain.services.valuation.models.base import ModelDiagnostics
from investigator.domain.services.valuation.models.company_profile import CompanyProfile, DataQualityFlag


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Return numerator / denominator with guards for zero/None."""
    try:
        if numerator is None or denominator is None:
            return None
        denominator = float(denominator)
        if abs(denominator) < 1e-9:
            return None
        return float(numerator) / denominator
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def clamp_fair_value(
    fair_value: float,
    current_price: float,
    max_upside_multiple: float = 5.0,
    min_downside_multiple: float = 0.1,
) -> float:
    """
    Cap fair value to reasonable bounds relative to current price.

    This prevents extreme outliers from distorting blended fair values.
    For example, a P/S model producing 10x upside due to high revenue per share.

    Args:
        fair_value: Calculated fair value from a model
        current_price: Current stock price
        max_upside_multiple: Max fair value as multiple of price (default 5x = 400% upside)
        min_downside_multiple: Min fair value as multiple of price (default 0.1x = -90% downside)

    Returns:
        Clamped fair value within reasonable bounds
    """
    if current_price <= 0:
        return fair_value

    max_fv = current_price * max_upside_multiple
    min_fv = current_price * min_downside_multiple

    return clamp(fair_value, min_fv, max_fv)


@dataclass
class MultipleModelContext:
    company_profile: CompanyProfile
    data_quality_score: float = 0.0
    fit_score: float = 0.0
    calibration_score: float = 0.35
    flags: List[str] = field(default_factory=list)

    def to_diagnostics(self) -> ModelDiagnostics:
        return ModelDiagnostics(
            data_quality_score=self.data_quality_score,
            fit_score=self.fit_score,
            calibration_score=self.calibration_score,
            flags=list(self.flags or []),
        )


def baseline_multiple_context(
    profile: CompanyProfile,
    *,
    data_quality_default: float = 0.6,
    fit_default: float = 0.5,
) -> MultipleModelContext:
    dq_score = profile.data_completeness_score if profile.data_completeness_score is not None else data_quality_default
    if dq_score > 1:  # some feeds send 0-100
        dq_score = dq_score / 100.0

    flags = [flag.name for flag in profile.data_quality_flags]
    if profile.has_flag(DataQualityFlag.MISSING_QUARTERS):  # type: ignore[arg-type]
        dq_score = min(dq_score, 0.5)

    return MultipleModelContext(
        company_profile=profile,
        data_quality_score=clamp(float(dq_score), 0.0, 1.0),
        fit_score=fit_default,
        flags=list(flags),
    )


def annotate_reason(diagnostics: ModelDiagnostics, reason: str) -> Dict[str, Any]:
    payload = {"diagnostics": diagnostics, "reason": reason}
    return payload
