"""
Wrapper that adapts the legacy DCF module to the new valuation interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base_valuation_model import (
    BaseValuationModel,
    ModelDiagnostics,
    ModelNotApplicable,
    ValuationModelResult,
)
from ..company_profile import CompanyProfile, DataQualityFlag
from utils.dcf_valuation import DCFValuation

logger = logging.getLogger(__name__)


class DCFValuationModel(BaseValuationModel):
    """Confidence-weighted adapter for the existing DCF implementation."""

    model_name = "dcf"
    methodology = "Discounted Cash Flow"

    def __init__(
        self,
        company_profile: CompanyProfile,
        symbol: str,
        quarterly_metrics: List[Dict[str, Any]],
        multi_year_data: List[Dict[str, Any]],
        db_manager: Any,
        dcf_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(company_profile=company_profile)
        self.symbol = symbol
        self._dcf = DCFValuation(
            symbol=symbol,
            quarterly_metrics=quarterly_metrics,
            multi_year_data=multi_year_data,
            db_manager=db_manager,
        )
        if dcf_config is not None:
            # Give callers a hook to override configuration without exposing
            # internals of the bundled DCF class.
            self._dcf.dcf_config = dcf_config

    def calculate(self, **kwargs: Any) -> ValuationModelResult | ModelNotApplicable:
        try:
            raw_result = self._dcf.calculate_dcf_valuation()
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("DCF valuation failed for %s", self.symbol)
            diagnostics = self._baseline_diagnostics()
            diagnostics.flags.append(f"ERROR:{exc.__class__.__name__}")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason=f"calculation_error:{exc}",
                diagnostics=diagnostics,
            )

        if not raw_result or "fair_value_per_share" not in raw_result:
            diagnostics = self._baseline_diagnostics()
            diagnostics.flags.append("MISSING_RESULT")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_fair_value",
                diagnostics=diagnostics,
            )

        diagnostics = self._build_diagnostics(raw_result)
        confidence = self.estimate_confidence(raw_result)

        return ValuationModelResult(
            model_name=self.model_name,
            fair_value=raw_result.get("fair_value_per_share"),
            confidence_score=confidence,
            methodology=self.methodology,
            assumptions=raw_result.get("assumptions", {}),
            diagnostics=diagnostics,
            metadata={
                "current_price": raw_result.get("current_price"),
                "upside_downside_pct": raw_result.get("upside_downside_pct"),
                "rule_of_40": raw_result.get("rule_of_40"),
                "valuation_breakdown": raw_result.get("valuation_breakdown"),
                "ps_valuation": raw_result.get("ps_valuation"),
                "valuation_assessment": raw_result.get("valuation_assessment"),
            },
        )

    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        """
        Derive a lightweight confidence score until the dedicated scorer lands.

        The heuristics lean on profile completeness, positive free cash flow,
        and whether the legacy module surfaced sensitivity analysis.
        """
        diagnostics = self._build_diagnostics(raw_output)

        # Average the available diagnostics while weighting data quality higher.
        data_quality = diagnostics.data_quality_score
        fit = diagnostics.fit_score
        calibration = diagnostics.calibration_score

        # Weighted average with a small floor to avoid zero confidence when
        # calibration data is not yet available.
        confidence = 0.5 * data_quality + 0.35 * fit + 0.15 * max(calibration, 0.4)

        return max(0.0, min(confidence, 1.0))

    def _baseline_diagnostics(self) -> ModelDiagnostics:
        """Generate a diagnostics object using only profile information."""
        data_quality = self._normalise_score(self.company_profile.data_completeness_score, default=0.5)
        fit_score = 0.8 if self.company_profile.has_positive_fcf else 0.5
        calibration = 0.4  # Placeholder until backtesting is wired

        return ModelDiagnostics(
            data_quality_score=data_quality,
            fit_score=fit_score,
            calibration_score=calibration,
            flags=[flag.name for flag in self.company_profile.data_quality_flags],
        )

    def _build_diagnostics(self, raw_output: Dict[str, Any]) -> ModelDiagnostics:
        diagnostics = self._baseline_diagnostics()

        if not raw_output.get("rule_of_40"):
            diagnostics.flags.append("MISSING_RULE_OF_40")
        if self.company_profile.has_positive_fcf is False:
            diagnostics.flags.append(DataQualityFlag.NEGATIVE_DENOMINATOR.name)

        # If the underlying model produced sensitivity analysis we treat that as
        # a proxy for stable inputs.
        sensitivity = raw_output.get("sensitivity") or {}
        if sensitivity:
            diagnostics.fit_score = min(1.0, diagnostics.fit_score + 0.1)

        return diagnostics

    @staticmethod
    def _normalise_score(value: Optional[float], default: float = 0.0) -> float:
        """Clamp values into [0, 1] while handling missing data."""
        if value is None:
            return default
        if value > 1:
            # Some legacy collectors store percentages as 0-100.
            value = value / 100.0
        return max(0.0, min(float(value), 1.0))
