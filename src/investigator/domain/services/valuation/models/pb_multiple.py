"""Price-to-Book multiple valuation model.

Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.pb_multiple
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
from investigator.domain.services.valuation.models.company_profile import (
    CompanyArchetype,
    CompanyProfile,
    DataQualityFlag,
)

logger = logging.getLogger(__name__)


class PBMultipleModel(BaseValuationModel):
    model_name = "pb"
    methodology = "P/B Multiple"

    def __init__(
        self,
        *,
        company_profile: CompanyProfile,
        book_value_per_share: Optional[float],
        current_price: Optional[float],
        sector_median_pb: Optional[float],
        tangible_book_value_per_share: Optional[float] = None,
        max_pb: float = 15.0,
        min_pb: float = 0.3,
    ) -> None:
        super().__init__(company_profile=company_profile)
        self.book_value_per_share = book_value_per_share
        self.current_price = current_price
        self.sector_median_pb = sector_median_pb
        self.tangible_book_value_per_share = tangible_book_value_per_share
        self.max_pb = max_pb
        self.min_pb = min_pb

    def calculate(self, **_: Any) -> ValuationModelResult | ModelNotApplicable:
        if not self._is_applicable():
            diagnostics = self._baseline_diagnostics()
            diagnostics.flags.append(DataQualityFlag.NEGATIVE_DENOMINATOR.name)
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_book_value",
                diagnostics=diagnostics,
            )

        target_pb = self._determine_target_multiple()
        if target_pb is None:
            diagnostics = self._baseline_diagnostics()
            diagnostics.flags.append("MISSING_REFERENCE_MULTIPLE")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_reference_multiple",
                diagnostics=diagnostics,
            )

        book_per_share = self.book_value_per_share or self.tangible_book_value_per_share
        fair_value = float(book_per_share) * target_pb
        diagnostics = self._build_diagnostics(target_multiple=target_pb)
        confidence = self.estimate_confidence({"target_pb": target_pb})

        metadata: Dict[str, Any] = {}
        if self.current_price and self.current_price > 0:
            metadata["current_price"] = self.current_price
            metadata["upside_downside_pct"] = round(((fair_value / self.current_price) - 1) * 100, 2)

        assumptions = {
            "book_value_per_share": self.book_value_per_share,
            "tangible_book_value_per_share": self.tangible_book_value_per_share,
            "target_pb": target_pb,
            "sector_median_pb": self.sector_median_pb,
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
        diagnostics = self._build_diagnostics(target_multiple=raw_output.get("target_pb"))
        return clamp(
            0.5 * diagnostics.data_quality_score + 0.35 * diagnostics.fit_score + 0.15 * diagnostics.calibration_score,
            0.0,
            1.0,
        )

    def _is_applicable(self) -> bool:
        book_per_share = self.book_value_per_share or self.tangible_book_value_per_share
        return book_per_share is not None and book_per_share > 0

    def _determine_target_multiple(self) -> Optional[float]:
        candidates = []
        if self.sector_median_pb and self.sector_median_pb > 0:
            candidates.append(self.sector_median_pb)
        if self.company_profile.primary_archetype == CompanyArchetype.FINANCIAL:
            # Financials rely heavily on P/B; allow tangible book to influence if available
            if self.tangible_book_value_per_share and self.book_value_per_share:
                tangible_ratio = self.tangible_book_value_per_share / self.book_value_per_share
                candidates.append(
                    clamp(self.sector_median_pb * tangible_ratio if self.sector_median_pb else tangible_ratio, 0.2, 2.0)
                )
        if not candidates:
            return None
        target = sum(candidates) / len(candidates)
        return clamp(target, self.min_pb, self.max_pb)

    def _baseline_diagnostics(self) -> ModelDiagnostics:
        context = baseline_multiple_context(self.company_profile, data_quality_default=0.6, fit_default=0.5)
        if self.company_profile.primary_archetype == CompanyArchetype.FINANCIAL:
            context.calibration_score = 0.45
        return context.to_diagnostics()

    def _build_diagnostics(self, *, target_multiple: Optional[float]) -> ModelDiagnostics:
        diagnostics = self._baseline_diagnostics()

        if target_multiple is None:
            return diagnostics

        if self.current_price and (self.book_value_per_share or self.tangible_book_value_per_share):
            book_per_share = self.book_value_per_share or self.tangible_book_value_per_share
            observed_pb = clamp(self.current_price / book_per_share, 0.0, self.max_pb)
            delta = abs(observed_pb - target_multiple) / target_multiple if target_multiple else 0
            if delta < 0.25:
                diagnostics.fit_score = clamp(diagnostics.fit_score + 0.1, 0.0, 1.0)
            elif delta > 0.75:
                diagnostics.flags.append("PB_DIVERGENCE")
                diagnostics.fit_score = clamp(diagnostics.fit_score - 0.1, 0.0, 1.0)

        return diagnostics
