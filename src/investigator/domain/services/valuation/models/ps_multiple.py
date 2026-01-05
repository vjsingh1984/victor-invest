"""
Price-to-Sales multiple valuation model.


Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.ps_multiple
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


# Industry-specific base P/S multiples (more granular than sector-level)
INDUSTRY_BASE_PS = {
    # Technology - Software & SaaS
    "Technology/Computer Software: Prepackaged Software": 10.0,  # SaaS (Snowflake, Datadog)
    "Technology/Computer Software: Programming, Data Processing": 12.0,  # Dev tools
    "Technology/EDP Services": 8.0,  # Enterprise services
    "Technology/Services-Computer Programming, Data Processing, Etc.": 10.0,  # Cloud services
    # Technology - Hardware & Semiconductors
    "Technology/Semiconductors": 5.0,  # Semiconductor manufacturers
    "Technology/Computer Hardware": 2.0,  # Hardware manufacturers
    "Technology/Electronic Components": 3.0,
    # Technology - Internet & E-commerce
    "Technology/Services-Computer Programming": 9.0,
    "Technology/Retail": 1.5,
    # Healthcare - Biotech & Pharma
    "Health Care/Biotechnology: Biological Products (No Diagnostic Substances)": 8.0,
    "Health Care/Pharmaceutical Preparations": 5.0,
    # Financial Services
    "Financials/Security Brokers, Dealers & Flotation Companies": 3.0,  # Fintech
    "Financials/Banks": 2.0,
    "Financials/Insurance": 1.5,
    # Consumer
    "Consumer Discretionary/Catalog & Mail-Order Houses": 2.0,  # E-commerce
    "Consumer Discretionary/Restaurants": 1.5,
    "Consumer Staples/Food": 1.0,
    # Industrials
    "Industrials/Aerospace": 1.5,
    "Industrials/Machinery": 1.2,
}

# Sector-level fallback P/S multiples (when specific industry not found)
SECTOR_BASE_PS = {
    "Technology": 6.0,
    "Health Care": 4.0,
    "Financials": 2.5,
    "Consumer Discretionary": 1.5,
    "Consumer Staples": 1.0,
    "Industrials": 1.2,
    "Energy": 1.0,
    "Materials": 1.5,
    "Real Estate": 3.0,
    "Utilities": 2.0,
    "Communication Services": 3.0,
}

# Growth tier adjustments (additive, based on YoY revenue growth)
GROWTH_TIER_ADJUSTMENTS = {
    (0.0, 0.10): -1.0,  # 0-10% (slow growth penalty)
    (0.10, 0.15): 0.0,  # 10-15% (baseline)
    (0.15, 0.25): 2.0,  # 15-25% (moderate growth)
    (0.25, 0.35): 4.0,  # 25-35% (high growth) - SNOW: 32%
    (0.35, 0.50): 6.0,  # 35-50% (very high growth)
    (0.50, 1.00): 8.0,  # >50% (exceptional growth)
}

# Stage/profitability adjustments (additive)
STAGE_ADJUSTMENTS = {
    "pre_profitable_high_growth": 2.0,  # High-growth pre-profitable (SNOW)
    "pre_profitable_low_growth": -1.0,  # Low-growth pre-profitable (struggling)
    "profitable_mature": 0.0,  # Baseline
    "early_stage": 1.0,  # Early-stage profitable
}

# Quality premium multipliers (multiplicative, applied AFTER additive adjustments)
QUALITY_PREMIUMS = {
    "rule_of_40_excellent": 1.2,  # Rule of 40 >40% (SNOW: 46.6%)
    "rule_of_40_good": 1.1,  # Rule of 40 30-40%
    "nrr_excellent": 1.15,  # Net Revenue Retention >120%
    "gross_margin_high": 1.1,  # Gross margin >70%
}


class PSMultipleModel(BaseValuationModel):
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
        """
        Calculate target P/S multiple with granular adjustments for:
        - Industry-specific base multiples
        - Growth tier adjustments (based on actual growth rate)
        - Stage/profitability adjustments
        - Quality premiums

        Formula:
        target_ps = (base_ps + growth_adjustment + stage_adjustment) * quality_premium
        """
        # Step 1: Determine base P/S (industry-specific or sector fallback)
        base_ps = None

        # Try industry-specific lookup first
        if self.company_profile.industry:
            base_ps = INDUSTRY_BASE_PS.get(self.company_profile.industry)
            if base_ps:
                logger.debug(f"PS_GRANULAR - Industry base P/S: {base_ps} (industry: {self.company_profile.industry})")

        # Fallback to sector-level lookup
        if base_ps is None and self.company_profile.sector:
            sector_key = (
                self.company_profile.sector.split("/")[0].strip()
                if "/" in self.company_profile.sector
                else self.company_profile.sector.strip()
            )
            base_ps = SECTOR_BASE_PS.get(sector_key)
            if base_ps:
                logger.debug(f"PS_GRANULAR - Sector base P/S: {base_ps} (sector: {sector_key})")

        # Last resort: use sector_median_ps from input
        if base_ps is None and self.sector_median_ps and self.sector_median_ps > 0:
            base_ps = self.sector_median_ps
            logger.debug(f"PS_GRANULAR - Using sector median P/S: {base_ps}")

        if base_ps is None:
            logger.warning(
                f"PS_GRANULAR - No base P/S found (sector: {self.company_profile.sector}, industry: {self.company_profile.industry})"
            )
            return None

        target_ps = float(base_ps)

        # Step 2: Add growth tier adjustment (additive, based on YoY revenue growth)
        growth_adjustment = 0.0
        if self.company_profile.revenue_growth_yoy is not None:
            growth_rate = self.company_profile.revenue_growth_yoy
            for (low, high), adjustment in GROWTH_TIER_ADJUSTMENTS.items():
                if low <= growth_rate < high:
                    growth_adjustment = adjustment
                    logger.info(
                        f"PS_GRANULAR - Growth adjustment: +{adjustment:.1f} (revenue growth: {growth_rate*100:.1f}%)"
                    )
                    break
        else:
            logger.debug("PS_GRANULAR - No revenue_growth_yoy available, skipping growth adjustment")

        target_ps += growth_adjustment

        # Step 3: Add stage/profitability adjustment (additive)
        stage_adjustment = 0.0

        # Determine stage classification
        # Use boolean flags from CompanyProfile (has_positive_earnings, has_positive_ebitda)
        is_pre_profitable = (
            self.company_profile.has_positive_earnings is not None and not self.company_profile.has_positive_earnings
        ) or (self.company_profile.has_positive_ebitda is not None and not self.company_profile.has_positive_ebitda)

        if is_pre_profitable:
            # Check if high growth (>20% YoY)
            is_high_growth = (
                self.company_profile.revenue_growth_yoy is not None and self.company_profile.revenue_growth_yoy > 0.20
            )

            if is_high_growth:
                stage_adjustment = STAGE_ADJUSTMENTS["pre_profitable_high_growth"]
                logger.info(f"PS_GRANULAR - Stage adjustment: +{stage_adjustment:.1f} (pre-profitable high-growth)")
            else:
                stage_adjustment = STAGE_ADJUSTMENTS["pre_profitable_low_growth"]
                logger.info(f"PS_GRANULAR - Stage adjustment: {stage_adjustment:.1f} (pre-profitable low-growth)")
        else:
            # Profitable company - check if early stage or mature
            # Early stage: less than 3 years of positive earnings
            stage_adjustment = STAGE_ADJUSTMENTS["profitable_mature"]
            logger.debug(f"PS_GRANULAR - Stage adjustment: {stage_adjustment:.1f} (profitable mature)")

        target_ps += stage_adjustment

        # Step 4: Apply quality premiums (multiplicative)
        quality_multiplier = 1.0
        applied_premiums = []

        # Rule of 40 premium (revenue growth + FCF margin)
        if self.company_profile.rule_of_40_score is not None:
            if self.company_profile.rule_of_40_score > 0.40:
                quality_multiplier *= QUALITY_PREMIUMS["rule_of_40_excellent"]
                applied_premiums.append(f"Rule of 40 excellent ({self.company_profile.rule_of_40_score*100:.1f}%)")
            elif self.company_profile.rule_of_40_score > 0.30:
                quality_multiplier *= QUALITY_PREMIUMS["rule_of_40_good"]
                applied_premiums.append(f"Rule of 40 good ({self.company_profile.rule_of_40_score*100:.1f}%)")

        # Net Revenue Retention (NRR) premium
        if self.company_profile.net_revenue_retention is not None and self.company_profile.net_revenue_retention > 1.20:
            quality_multiplier *= QUALITY_PREMIUMS["nrr_excellent"]
            applied_premiums.append(f"NRR excellent ({self.company_profile.net_revenue_retention*100:.0f}%)")

        # Gross margin premium
        if self.company_profile.gross_margin is not None and self.company_profile.gross_margin > 0.70:
            quality_multiplier *= QUALITY_PREMIUMS["gross_margin_high"]
            applied_premiums.append(f"Gross margin high ({self.company_profile.gross_margin*100:.1f}%)")

        if applied_premiums:
            logger.info(f"PS_GRANULAR - Quality premiums: {quality_multiplier:.2f}x ({', '.join(applied_premiums)})")

        target_ps *= quality_multiplier

        # Step 5: Clamp to min/max bounds
        final_ps = clamp(target_ps, self.min_ps, self.max_ps)

        logger.info(
            f"PS_GRANULAR - Final P/S: {final_ps:.2f} (base: {base_ps:.1f} + growth: {growth_adjustment:.1f} + stage: {stage_adjustment:.1f}) × quality: {quality_multiplier:.2f})"
        )

        return final_ps

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
