"""
Price-to-Sales multiple valuation model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base_valuation_model import (
    BaseValuationModel,
    ModelDiagnostics,
    ModelNotApplicable,
    ValuationModelResult,
)
from ..company_profile import CompanyArchetype, CompanyProfile, DataQualityFlag
from .common import baseline_multiple_context, clamp

logger = logging.getLogger(__name__)


class PSMultipleValuationModel(BaseValuationModel):
    model_name = "ps"
    methodology = "P/S Multiple"

    def __init__(
        self,
        *,
        company_profile: CompanyProfile,
        revenue_per_share: Optional[float],
        current_price: Optional[float],
        sector_median_ps: Optional[float],
        max_ps: float = 25.0,
        min_ps: float = 1.0,
        liquidity_floor_usd: float = 5_000_000.0,
    ) -> None:
        super().__init__(company_profile=company_profile)
        self.revenue_per_share = revenue_per_share
        self.current_price = current_price
        self.sector_median_ps = sector_median_ps
        self.max_ps = max_ps
        self.min_ps = min_ps
        self.liquidity_floor_usd = liquidity_floor_usd

    def calculate(self, **_: Any) -> ValuationModelResult | ModelNotApplicable:
        if not self._is_applicable():
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append("INSUFFICIENT_LIQUIDITY")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="insufficient_data_or_liquidity",
                diagnostics=diagnostics,
            )

        target_ps = self._determine_target_ps()
        if target_ps is None:
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append("MISSING_REFERENCE_MULTIPLE")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_reference_multiple",
                diagnostics=diagnostics,
            )

        fair_value = float(self.revenue_per_share) * target_ps
        diagnostics = self._build_diagnostics(target_ps=target_ps)
        confidence = self.estimate_confidence({"target_ps": target_ps})

        metadata: Dict[str, Any] = {}
        if self.current_price is not None and self.current_price > 0:
            metadata["current_price"] = self.current_price
            metadata["upside_downside_pct"] = round(((fair_value / self.current_price) - 1) * 100, 2)

        assumptions = {
            "revenue_per_share": self.revenue_per_share,
            "target_ps": target_ps,
            "sector_median_ps": self.sector_median_ps,
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
        diagnostics = self._build_diagnostics(target_ps=raw_output.get("target_ps"))
        return clamp(
            0.55 * diagnostics.data_quality_score + 0.30 * diagnostics.fit_score + 0.15 * diagnostics.calibration_score,
            0.0,
            1.0,
        )

    def _is_applicable(self) -> bool:
        if self.revenue_per_share is None or self.revenue_per_share <= 0:
            return False
        if self.current_price is None or self.current_price <= 0:
            return False
        liquidity_ok = (
            self.company_profile.daily_liquidity_usd is None
            or self.company_profile.daily_liquidity_usd >= self.liquidity_floor_usd
        )
        return liquidity_ok

    def _determine_target_ps(self) -> Optional[float]:
        candidates = []
        if self.sector_median_ps and self.sector_median_ps > 0:
            candidates.append(self.sector_median_ps)

        if self.company_profile.primary_archetype == CompanyArchetype.HIGH_GROWTH:
            growth_boost = 1.0
            if self.company_profile.revenue_cagr_3y and self.company_profile.revenue_cagr_3y > 0.2:
                growth_boost += min(self.company_profile.revenue_cagr_3y, 0.5)
            if self.company_profile.revenue_growth_yoy and self.company_profile.revenue_growth_yoy > 0.3:
                growth_boost += 0.2
            if candidates:
                candidates.append(candidates[0] * clamp(growth_boost, 1.0, 2.5))

        if not candidates:
            return None

        target = sum(candidates) / len(candidates)
        return clamp(target, self.min_ps, self.max_ps)

    def _build_baseline_diagnostics(self) -> ModelDiagnostics:
        context = baseline_multiple_context(self.company_profile, data_quality_default=0.55, fit_default=0.45)
        if self.company_profile.primary_archetype == CompanyArchetype.HIGH_GROWTH:
            context.fit_score = clamp(context.fit_score + 0.1, 0.0, 1.0)
        return context.to_diagnostics()

    def _build_diagnostics(self, *, target_ps: Optional[float]) -> ModelDiagnostics:
        diagnostics = self._build_baseline_diagnostics()
        diagnostics.calibration_score = 0.35

        if target_ps is None:
            return diagnostics

        if self.current_price and self.revenue_per_share:
            observed_ps = clamp(self.current_price / self.revenue_per_share, 0.0, self.max_ps)
            delta = abs(observed_ps - target_ps) / target_ps if target_ps else 0
            if delta < 0.25:
                diagnostics.fit_score = clamp(diagnostics.fit_score + 0.1, 0.0, 1.0)
            elif delta > 0.75:
                diagnostics.flags.append("PS_DIVERGENCE")
                diagnostics.fit_score = clamp(diagnostics.fit_score - 0.1, 0.0, 1.0)

        if (
            self.company_profile.daily_liquidity_usd
            and self.company_profile.daily_liquidity_usd < self.liquidity_floor_usd
        ):
            diagnostics.flags.append("LOW_LIQUIDITY")
            diagnostics.data_quality_score = clamp(diagnostics.data_quality_score - 0.1, 0.0, 1.0)

        return diagnostics
