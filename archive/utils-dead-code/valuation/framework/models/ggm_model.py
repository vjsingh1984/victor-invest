"""
Wrapper that adapts the Gordon Growth Model implementation to the unified API.
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
from utils.gordon_growth_model import GordonGrowthModel

logger = logging.getLogger(__name__)


class GGMValuationModel(BaseValuationModel):
    """Adapter for the dividend discount model."""

    model_name = "ggm"
    methodology = "Gordon Growth Model"

    def __init__(
        self,
        company_profile: CompanyProfile,
        symbol: str,
        quarterly_metrics: List[Dict[str, Any]],
        multi_year_data: List[Dict[str, Any]],
        db_manager: Any,
    ) -> None:
        super().__init__(company_profile=company_profile)
        self.symbol = symbol
        self._ggm = GordonGrowthModel(
            symbol=symbol,
            quarterly_metrics=quarterly_metrics,
            multi_year_data=multi_year_data,
            db_manager=db_manager,
            company_profile=company_profile,
        )

    def calculate(
        self,
        *,
        cost_of_equity: Optional[float] = None,
        **_: Any,
    ) -> ValuationModelResult | ModelNotApplicable:
        if cost_of_equity is None:
            diagnostics = self._baseline_diagnostics()
            diagnostics.flags.append("MISSING_COST_OF_EQUITY")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_cost_of_equity",
                diagnostics=diagnostics,
            )

        try:
            raw_result = self._ggm.calculate_ggm_valuation(cost_of_equity=cost_of_equity)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("GGM valuation failed for %s", self.symbol)
            diagnostics = self._baseline_diagnostics()
            diagnostics.flags.append(f"ERROR:{exc.__class__.__name__}")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason=f"calculation_error:{exc}",
                diagnostics=diagnostics,
            )

        if not raw_result.get("applicable", False):
            reason = raw_result.get("reason", "not_applicable")
            diagnostics = self._baseline_diagnostics()
            if raw_result.get("warnings"):
                diagnostics.flags.extend(str(w) for w in raw_result["warnings"])
            return ModelNotApplicable(
                model_name=self.model_name,
                reason=reason,
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
                "valuation_assessment": raw_result.get("valuation_assessment"),
            },
        )

    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        diagnostics = self._build_diagnostics(raw_output)

        data_quality = diagnostics.data_quality_score
        fit = diagnostics.fit_score
        calibration = diagnostics.calibration_score

        confidence = 0.45 * data_quality + 0.4 * fit + 0.15 * max(calibration, 0.35)

        return max(0.0, min(confidence, 1.0))

    def _baseline_diagnostics(self) -> ModelDiagnostics:
        data_quality = self._normalise_score(self.company_profile.data_completeness_score, default=0.4)
        fit = 0.2  # default low fit until proven otherwise
        calibration = 0.35

        if self.company_profile.pays_dividends:
            fit = 0.6
        if self.company_profile.dividend_growth_rate and self.company_profile.dividend_growth_rate > 0:
            fit += 0.1

        flags = [flag.name for flag in self.company_profile.data_quality_flags]
        if not self.company_profile.pays_dividends:
            flags.append(DataQualityFlag.INCOMPLETE_DIVIDEND_HISTORY.name)

        return ModelDiagnostics(
            data_quality_score=data_quality,
            fit_score=min(fit, 1.0),
            calibration_score=calibration,
            flags=flags,
        )

    def _build_diagnostics(self, raw_output: Dict[str, Any]) -> ModelDiagnostics:
        diagnostics = self._baseline_diagnostics()

        dividend_yield = None
        assumptions = raw_output.get("assumptions") or {}
        try:
            dividend_yield = assumptions.get("dividend_yield")
        except AttributeError:
            dividend_yield = None

        if dividend_yield is not None and dividend_yield > 0:
            diagnostics.fit_score = min(1.0, diagnostics.fit_score + 0.1)

        validation = raw_output.get("validation") or {}
        if validation.get("warnings"):
            diagnostics.flags.extend(str(w) for w in validation["warnings"])

        return diagnostics

    @staticmethod
    def _normalise_score(value: Optional[float], default: float = 0.0) -> float:
        if value is None:
            return default
        if value > 1:
            value = value / 100.0
        return max(0.0, min(float(value), 1.0))
