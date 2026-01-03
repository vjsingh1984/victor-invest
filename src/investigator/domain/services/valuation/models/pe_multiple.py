"""
Price-to-Earnings multiple valuation model.

Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.pe_multiple
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from investigator.domain.services.valuation.models.base import (
    BaseValuationModel,
    ModelDiagnostics,
    ModelNotApplicable,
    ValuationModelResult,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile, DataQualityFlag
from investigator.domain.services.valuation.models.common import baseline_multiple_context, clamp

logger = logging.getLogger(__name__)


class PEMultipleModel(BaseValuationModel):
    model_name = "pe"
    methodology = "P/E Multiple"

    def __init__(
        self,
        *,
        company_profile: CompanyProfile,
        ttm_eps: Optional[float],
        current_price: Optional[float],
        sector_median_pe: Optional[float],
        growth_adjusted_pe: Optional[float] = None,
        max_pe: float = 80.0,
        min_pe: float = 5.0,
        earnings_quality_score: Optional[float] = None,
    ) -> None:
        super().__init__(company_profile=company_profile)
        self.ttm_eps = ttm_eps
        self.current_price = current_price
        self.sector_median_pe = sector_median_pe
        self.growth_adjusted_pe = growth_adjusted_pe
        self.max_pe = max_pe
        self.min_pe = min_pe
        self.earnings_quality_score = earnings_quality_score

    def calculate(self, **_: Any) -> ValuationModelResult | ModelNotApplicable:
        if not self._is_applicable():
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append(DataQualityFlag.NEGATIVE_DENOMINATOR.name)
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="negative_or_missing_eps",
                diagnostics=diagnostics,
            )

        target_pe = self._determine_target_pe()
        if target_pe is None:
            diagnostics = self._build_baseline_diagnostics()
            diagnostics.flags.append("MISSING_REFERENCE_PE")
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="missing_reference_multiple",
                diagnostics=diagnostics,
            )

        fair_value = float(self.ttm_eps) * target_pe
        confidence = self.estimate_confidence({"target_pe": target_pe})
        diagnostics = self._build_diagnostics(target_pe=target_pe)

        assumptions = {
            "ttm_eps": self.ttm_eps,
            "target_pe": target_pe,
            "sector_median_pe": self.sector_median_pe,
            "growth_adjusted_pe": self.growth_adjusted_pe,
            "max_pe_cap": self.max_pe,
        }

        metadata = {}
        if self.current_price is not None and self.current_price > 0:
            upside = ((fair_value / self.current_price) - 1) * 100
            metadata["current_price"] = self.current_price
            metadata["upside_downside_pct"] = round(upside, 2)

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
        diagnostics = self._build_diagnostics(target_pe=raw_output.get("target_pe"))
        data_quality = diagnostics.data_quality_score
        fit = diagnostics.fit_score
        calibration = diagnostics.calibration_score
        return clamp(0.55 * data_quality + 0.35 * fit + 0.10 * calibration, 0.0, 1.0)

    def _is_applicable(self) -> bool:
        if self.ttm_eps is None or self.ttm_eps <= 0:
            return False
        if self.earnings_quality_score is not None and self.earnings_quality_score < 0.5:
            return False
        return True

    def _determine_target_pe(self) -> Optional[float]:
        # Get symbol for logging (defined once at the beginning)
        symbol = self.company_profile.symbol if hasattr(self.company_profile, 'symbol') else 'UNKNOWN'

        candidates = []
        sources = []

        # Priority 1: Market-derived sector median
        if self.sector_median_pe and self.sector_median_pe > 0:
            candidates.append(self.sector_median_pe)
            sources.append(f"sector_median={self.sector_median_pe:.2f}")

        # Priority 2: Growth-adjusted P/E
        if self.growth_adjusted_pe and self.growth_adjusted_pe > 0:
            candidates.append(self.growth_adjusted_pe)
            sources.append(f"growth_adjusted={self.growth_adjusted_pe:.2f}")

        # Priority 3: Config-based default (NEW FALLBACK)
        if not candidates:
            config_pe = self._get_config_pe_fallback()
            if config_pe:
                candidates.append(config_pe)
                sources.append(f"config_default={config_pe:.2f}")
                logger.info(
                    f"🔧 [PE_FALLBACK] {symbol} - Using config default P/E: {config_pe:.2f} "
                    f"(no market data available)"
                )

        target = None
        if candidates:
            target = sum(candidates) / len(candidates)
            # Log P/E transparency: show which values were used
            logger.info(
                f"🔍 [PE_TRANSPARENCY] {symbol} - Target P/E calculation: "
                f"sources=[{', '.join(sources)}] → average={target:.2f}"
            )

        if target is None:
            logger.warning(
                f"⚠️  [PE_TRANSPARENCY] {symbol} - No valid P/E multiples available "
                f"(sector_median_pe={self.sector_median_pe}, growth_adjusted_pe={self.growth_adjusted_pe}, config=None)"
            )
            return None

        # Apply clamping
        unclamped_target = target
        target = clamp(target, self.min_pe, self.max_pe)

        if unclamped_target != target:
            logger.info(
                f"🔍 [PE_TRANSPARENCY] {symbol} - P/E clamped: "
                f"{unclamped_target:.2f} → {target:.2f} (range: {self.min_pe:.1f}-{self.max_pe:.1f})"
            )

        return target

    def _get_config_pe_fallback(self) -> Optional[float]:
        """
        Get P/E multiple from config.yaml based on sector/industry.

        Priority:
        1. Industry-specific override (most granular)
        2. Sector default (broader)
        3. Global default (universal fallback)

        Returns:
            P/E multiple from config, or None if not found
        """
        try:
            import yaml

            config_path = "config.yaml"
            if not os.path.exists(config_path):
                logger.debug("config.yaml not found, skipping P/E fallback")
                return None

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            pe_config = config.get('pe_multiples', {})
            if not pe_config:
                logger.debug("No pe_multiples section in config.yaml")
                return None

            # Priority 1: Industry override
            if hasattr(self.company_profile, 'industry') and self.company_profile.industry:
                industry_overrides = pe_config.get('industry_overrides', {})
                if self.company_profile.industry in industry_overrides:
                    pe_value = float(industry_overrides[self.company_profile.industry])
                    logger.debug(
                        f"P/E fallback: industry={self.company_profile.industry}, pe={pe_value}"
                    )
                    return pe_value

            # Priority 2: Sector default
            if hasattr(self.company_profile, 'sector') and self.company_profile.sector:
                sector_defaults = pe_config.get('sector_defaults', {})
                if self.company_profile.sector in sector_defaults:
                    pe_value = float(sector_defaults[self.company_profile.sector])
                    logger.debug(
                        f"P/E fallback: sector={self.company_profile.sector}, pe={pe_value}"
                    )
                    return pe_value

            # Priority 3: Global default
            default_pe = pe_config.get('default')
            if default_pe:
                logger.debug(f"P/E fallback: using global default={default_pe}")
                return float(default_pe)

            return None

        except Exception as e:
            logger.warning(f"Error loading P/E config fallback: {e}")
            return None

    def _build_baseline_diagnostics(self) -> ModelDiagnostics:
        context = baseline_multiple_context(self.company_profile, data_quality_default=0.6, fit_default=0.55)
        if self.earnings_quality_score is not None:
            context.fit_score = clamp(self.earnings_quality_score, 0.0, 1.0)
        return context.to_diagnostics()

    def _build_diagnostics(self, *, target_pe: Optional[float]) -> ModelDiagnostics:
        diagnostics = self._build_baseline_diagnostics()

        if target_pe is None:
            return diagnostics

        if self.company_profile.revenue_growth_yoy is not None and self.company_profile.revenue_growth_yoy > 0.15:
            diagnostics.fit_score = clamp(diagnostics.fit_score + 0.1, 0.0, 1.0)

        if self.current_price and self.ttm_eps:
            observed_pe = clamp(self.current_price / self.ttm_eps, 0.0, self.max_pe)
            delta = abs(observed_pe - target_pe) / target_pe if target_pe else 0
            if delta < 0.25:
                diagnostics.fit_score = clamp(diagnostics.fit_score + 0.1, 0.0, 1.0)
            elif delta > 0.75:
                diagnostics.flags.append("PE_DIVERGENCE")
                diagnostics.fit_score = clamp(diagnostics.fit_score - 0.1, 0.0, 1.0)

        return diagnostics
