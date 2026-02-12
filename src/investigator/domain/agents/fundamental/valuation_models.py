"""Relative valuation model computation helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from investigator.domain.services.valuation.helpers import normalize_model_output
from investigator.domain.services.valuation.models import (
    EVEBITDAModel,
    PBMultipleModel,
    PEMultipleModel,
    PSMultipleModel,
)
from investigator.domain.services.valuation.models.common import clamp


def calculate_relative_valuation_models(
    *,
    symbol: str,
    company_profile: Any,
    company_data: Dict[str, Any],
    ratios: Dict[str, Any],
    financials: Dict[str, Any],
    market_data: Dict[str, Any],
    config: Any,
    sector_specific_result: Optional[Dict[str, Any]],
    lookup_sector_multiple: Callable[[Optional[str], str], Optional[float]],
    calculate_enterprise_value: Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]],
    logger: Any,
) -> Dict[str, Dict[str, Any]]:
    """Calculate P/E, EV/EBITDA, P/S and P/B model outputs."""
    ttm_eps = ratios.get("eps") or ratios.get("eps_basic") or ratios.get("eps_diluted")
    sector_median_pe = (
        company_data.get("sector_metrics", {}).get("median_pe")
        or company_data.get("sector_data", {}).get("median_pe")
        or lookup_sector_multiple(company_profile.sector, "pe")
    )
    growth_adjusted_pe = None
    peg_ratio = ratios.get("peg_ratio") or ratios.get("peg")
    if peg_ratio and peg_ratio > 0:
        growth_adjusted_pe = sector_median_pe * (1 + min(peg_ratio, 3)) if sector_median_pe else None

    current_price = (
        market_data.get("price")
        or market_data.get("close")
        or market_data.get("current_price")
        or ratios.get("current_price")
    )

    pe_model = PEMultipleModel(
        company_profile=company_profile,
        ttm_eps=ttm_eps,
        current_price=current_price,
        sector_median_pe=sector_median_pe,
        growth_adjusted_pe=growth_adjusted_pe,
        earnings_quality_score=company_profile.earnings_quality_score,
    )
    normalized_pe = normalize_model_output(pe_model.calculate())

    ttm_ebitda = financials.get("ebitda") or ratios.get("ebitda") or financials.get("operating_income")
    enterprise_value = calculate_enterprise_value(market_data, financials)
    sector_ev_ebitda = (
        company_data.get("sector_metrics", {}).get("median_ev_ebitda")
        or company_data.get("sector_data", {}).get("median_ev_ebitda")
        or lookup_sector_multiple(company_profile.sector, "ev_ebitda")
    )

    leverage_adjusted_multiple = None
    if sector_ev_ebitda and company_profile.net_debt_to_ebitda is not None:
        leverage_delta = max(company_profile.net_debt_to_ebitda - 2.0, 0.0)
        leverage_adjusted_multiple = sector_ev_ebitda * clamp(1.0 - 0.06 * leverage_delta, 0.6, 1.1)

    ev_ebitda_model = EVEBITDAModel(
        company_profile=company_profile,
        ttm_ebitda=ttm_ebitda,
        enterprise_value=enterprise_value,
        sector_median_ev_ebitda=sector_ev_ebitda,
        leverage_adjusted_multiple=leverage_adjusted_multiple,
        interest_coverage=ratios.get("interest_coverage") or ratios.get("interest_coverage_ratio"),
    )
    normalized_ev_ebitda = normalize_model_output(ev_ebitda_model.calculate())

    revenue_per_share = None
    if ratios.get("revenue_per_share"):
        revenue_per_share = ratios.get("revenue_per_share")
    elif financials.get("revenues") and company_profile.shares_outstanding:
        try:
            revenue_per_share = float(financials.get("revenues")) / float(company_profile.shares_outstanding)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            revenue_per_share = None
            logger.debug("%s - Failed to calculate revenue_per_share: %s", symbol, exc)

    sector_ps = (
        company_data.get("sector_metrics", {}).get("median_ps")
        or company_data.get("sector_data", {}).get("median_ps")
        or lookup_sector_multiple(company_profile.sector, "ps")
    )

    valuation_settings = getattr(config, "valuation", None)
    liquidity_floor = 5_000_000
    if isinstance(valuation_settings, dict):
        liquidity_floor = valuation_settings.get("liquidity_floor_usd", liquidity_floor)
    elif valuation_settings is not None:
        liquidity_floor = getattr(valuation_settings, "liquidity_floor_usd", liquidity_floor)

    ps_model = PSMultipleModel(
        company_profile=company_profile,
        revenue_per_share=revenue_per_share,
        current_price=current_price,
        sector_median_ps=sector_ps,
        liquidity_floor_usd=liquidity_floor,
    )
    normalized_ps = normalize_model_output(ps_model.calculate())

    sector_pb = (
        company_data.get("sector_metrics", {}).get("median_pb")
        or company_data.get("sector_data", {}).get("median_pb")
        or lookup_sector_multiple(company_profile.sector, "pb")
    )
    pb_model = PBMultipleModel(
        company_profile=company_profile,
        book_value_per_share=company_profile.book_value_per_share,
        tangible_book_value_per_share=ratios.get("tangible_book_value_per_share"),
        current_price=current_price,
        sector_median_pb=sector_pb,
    )
    normalized_pb = normalize_model_output(pb_model.calculate())

    if sector_specific_result and "P/BV" in sector_specific_result.get("method", ""):
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        insurance_confidence = confidence_map.get(sector_specific_result.get("confidence", "medium"), 0.7)
        normalized_pb = {
            "model": "pb",
            "fair_value_per_share": sector_specific_result.get("fair_value"),
            "applicable": True,
            "confidence_score": insurance_confidence,
            "method": sector_specific_result.get("method"),
            "details": sector_specific_result.get("details", {}),
            "warnings": sector_specific_result.get("warnings", []),
            "upside_percent": sector_specific_result.get("upside_percent"),
            "current_price": sector_specific_result.get("current_price"),
        }
        logger.info(
            "🏦 %s - INSURANCE OVERRIDE: Using P/BV insurance valuation for P/B model (FV=$%.2f, confidence=%s)",
            symbol,
            sector_specific_result.get("fair_value", 0),
            sector_specific_result.get("confidence"),
        )

    return {
        "pe": normalized_pe,
        "ev_ebitda": normalized_ev_ebitda,
        "ps": normalized_ps,
        "pb": normalized_pb,
    }
