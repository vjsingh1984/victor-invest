"""
Enterprise Value to EBITDA multiple valuation model.


Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.ev_ebitda
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from investigator.domain.services.valuation.models.base import (
    BaseValuationModel,
    ModelDiagnostics,
    ModelNotApplicable,
    ValuationModelResult,
)
from investigator.domain.services.valuation.models.common import baseline_multiple_context, clamp
from investigator.domain.services.valuation.models.company_profile import CompanyProfile, DataQualityFlag

logger = logging.getLogger(__name__)


class EVEBITDAModel(BaseValuationModel):
    model_name = "ev_ebitda"
    methodology = "EV/EBITDA Multiple"

    def __init__(
        self,
        *,
        company_profile: CompanyProfile,
        ttm_ebitda: Optional[float],
        enterprise_value: Optional[float],
        sector_median_ev_ebitda: Optional[float],
        leverage_adjusted_multiple: Optional[float] = None,
        max_multiple: float = 30.0,
        min_multiple: float = 4.0,
        interest_coverage: Optional[float] = None,
    ) -> None:
        super().__init__(company_profile=company_profile)
        self.ttm_ebitda = ttm_ebitda
        self.enterprise_value = enterprise_value
        self.sector_median_ev_ebitda = sector_median_ev_ebitda
        self.leverage_adjusted_multiple = leverage_adjusted_multiple
        self.max_multiple = max_multiple
        self.min_multiple = min_multiple
        self.interest_coverage = interest_coverage

    def calculate(self, **_: Any) -> ValuationModelResult | ModelNotApplicable:
        if not self._is_applicable():
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append(DataQualityFlag.NEGATIVE_DENOMINATOR.name)
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="negative_or_missing_ebitda",
                diagnostics=diagnostics,
            )

        target_multiple = self._determine_target_multiple()
        if target_multiple is None:
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append("MISSING_REFERENCE_MULTIPLE")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_reference_multiple",
                diagnostics=diagnostics,
            )

        fair_value_ev = float(self.ttm_ebitda) * target_multiple
        equity_value = self._convert_ev_to_equity(fair_value_ev)
        if equity_value is None:
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append("MISSING_NET_DEBT")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_net_debt",
                diagnostics=diagnostics,
            )

        shares_outstanding = self._shares_outstanding()
        if shares_outstanding is None or shares_outstanding <= 0:
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append("MISSING_SHARES")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_shares_outstanding",
                diagnostics=diagnostics,
            )

        fair_value = equity_value / shares_outstanding
        diagnostics = self._build_diagnostics(target_multiple=target_multiple)
        confidence = self.estimate_confidence({"target_multiple": target_multiple})

        current_price = self._current_price()
        metadata: Dict[str, Any] = {}
        if current_price and current_price > 0:
            metadata["current_price"] = current_price
            metadata["upside_downside_pct"] = round(((fair_value / current_price) - 1) * 100, 2)

        assumptions = {
            "ttm_ebitda": self.ttm_ebitda,
            "target_ev_ebitda": target_multiple,
            "sector_median_ev_ebitda": self.sector_median_ev_ebitda,
            "leverage_adjusted_multiple": self.leverage_adjusted_multiple,
            "enterprise_value_fair": fair_value_ev,
        }

        return ValuationModelResult(
            model_name=self.model_name,
            fair_value=fair_value,
            confidence_score=confidence,
            methodology=self.methodology,
            assumptions=assumptions,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        diagnostics = self._build_diagnostics(target_multiple=raw_output.get("target_multiple"))
        return clamp(
            0.5 * diagnostics.data_quality_score + 0.35 * diagnostics.fit_score + 0.15 * diagnostics.calibration_score,
            0.0,
            1.0,
        )

    def _is_applicable(self) -> bool:
        return self.ttm_ebitda is not None and self.ttm_ebitda > 0

    def _determine_target_multiple(self) -> Optional[float]:
        candidates = []
        if self.sector_median_ev_ebitda and self.sector_median_ev_ebitda > 0:
            candidates.append(self.sector_median_ev_ebitda)
        if self.leverage_adjusted_multiple and self.leverage_adjusted_multiple > 0:
            candidates.append(self.leverage_adjusted_multiple)
        if not candidates:
            return None
        multiple = sum(candidates) / len(candidates)
        return clamp(multiple, self.min_multiple, self.max_multiple)

    def _convert_ev_to_equity(self, fair_ev: float) -> Optional[float]:
        net_debt = self._net_debt()
        if net_debt is None:
            return None
        return fair_ev - net_debt

    def _net_debt(self) -> Optional[float]:
        profile = self.company_profile
        total_debt = getattr(profile, "total_debt", None)
        cash = getattr(profile, "cash", None)
        if total_debt is None and profile.net_debt_to_ebitda is not None and self.ttm_ebitda:
            try:
                return float(profile.net_debt_to_ebitda) * float(self.ttm_ebitda)
            except (TypeError, ValueError):
                return None
        if total_debt is None:
            return None
        try:
            return float(total_debt) - float(cash or 0.0)
        except (TypeError, ValueError):
            return None

    def _shares_outstanding(self) -> Optional[float]:
        shares = getattr(self.company_profile, "shares_outstanding", None)
        if shares:
            return float(shares)
        return None

    def _current_price(self) -> Optional[float]:
        return self.company_profile.current_price

    def _build_baseline_diagnostics(self) -> ModelDiagnostics:
        context = baseline_multiple_context(self.company_profile, data_quality_default=0.55, fit_default=0.5)
        if self.interest_coverage is not None and self.interest_coverage < 1.5:
            context.fit_score = clamp(context.fit_score - 0.1, 0.0, 1.0)
            if DataQualityFlag.OUTLIER_DETECTED.name not in context.flags:
                context.flags.append(DataQualityFlag.OUTLIER_DETECTED.name)
        return context.to_diagnostics()

    def _build_diagnostics(self, *, target_multiple: Optional[float]) -> ModelDiagnostics:
        diagnostics = self._build_baseline_diagnostics()
        diagnostics.calibration_score = 0.4

        if target_multiple is None:
            return diagnostics

        if self.company_profile.net_debt_to_ebitda and self.company_profile.net_debt_to_ebitda > 3.5:
            diagnostics.flags.append("HIGH_LEVERAGE")
            diagnostics.fit_score = clamp(diagnostics.fit_score - 0.1, 0.0, 1.0)

        if target_multiple and self.enterprise_value is not None and self.ttm_ebitda not in (None, 0):
            try:
                observed_multiple = float(self.enterprise_value) / float(self.ttm_ebitda)
                delta = abs(observed_multiple - target_multiple) / target_multiple
                if delta < 0.25:
                    diagnostics.fit_score = clamp(diagnostics.fit_score + 0.1, 0.0, 1.0)
                elif delta > 0.75:
                    diagnostics.flags.append("EV_EBITDA_DIVERGENCE")
                    diagnostics.fit_score = clamp(diagnostics.fit_score - 0.1, 0.0, 1.0)
            except (TypeError, ValueError, ZeroDivisionError):
                logger.debug("Unable to compute observed EV/EBITDA multiple for diagnostics")

        return diagnostics
