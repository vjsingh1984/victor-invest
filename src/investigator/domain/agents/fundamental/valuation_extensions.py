"""Helpers for GGM and extension valuation models (Damodaran, Rule of 40, SaaS)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List

from investigator.domain.services.valuation.damodaran_dcf import DamodaranDCFModel
from investigator.domain.services.valuation.models.saas_valuation import SaaSValuationModel
from investigator.domain.services.valuation.rule_of_40_valuation import RuleOf40Valuation


async def calculate_valuation_extensions(
    *,
    symbol: str,
    valuation_results: Dict[str, Any],
    financials: Dict[str, Any],
    ratios: Dict[str, Any],
    market_data: Dict[str, Any],
    company_profile: Any,
    quarterly_data: List[Any],
    calculate_cost_of_equity: Callable[[str], float],
    calculate_ggm: Callable[[str, float, List[Any], Any], Awaitable[Dict[str, Any]]],
    normalize_model_output: Callable[[Any], Dict[str, Any]],
    log_model_result: Callable[[Any, str, str, Dict[str, Any]], None],
    logger: Any,
) -> float:
    """Populate extension valuation models and return payout ratio used for synthesis context."""
    common_divs = abs(financials.get("dividends_paid", 0) or 0)
    preferred_divs = abs(financials.get("preferred_stock_dividends", 0) or 0)
    dividends_paid = common_divs + preferred_divs

    if preferred_divs > 0:
        logger.debug(
            "%s - Preferred stock dividends found: $%s (27%% coverage - rare field)",
            symbol,
            format(preferred_divs, ",.0f"),
        )

    net_income = financials.get("net_income", 0) or 0
    payout_ratio = (dividends_paid / net_income * 100) if net_income > 0 else 0
    is_significant_dividend_stock = dividends_paid > 0 and payout_ratio >= 20.0

    if is_significant_dividend_stock:
        cost_of_equity = calculate_cost_of_equity(symbol)
        logger.info("%s - GGM cost_of_equity passed: %.2f%%", symbol, cost_of_equity * 100)
        ggm_result = await calculate_ggm(symbol, cost_of_equity, quarterly_data, company_profile)
        valuation_results["ggm"] = ggm_result
        logger.info(
            "%s - GGM applicable: payout ratio %.1f%% (≥20%% threshold for meaningful dividend policy)",
            symbol,
            payout_ratio,
        )
        log_model_result(logger, symbol, "GGM", ggm_result)
    else:
        if dividends_paid > 0 and payout_ratio < 20.0:
            reason = (
                f"Low payout ratio ({payout_ratio:.1f}%) - token dividend, not meaningful dividend policy (need ≥20%)"
            )
        elif dividends_paid == 0:
            reason = "No dividends paid - GGM requires dividend-paying stock"
        else:
            reason = "Negative net income - cannot calculate meaningful payout ratio"

        ggm_result = {
            "applicable": False,
            "reason": reason,
            "fair_value_per_share": 0,
            "payout_ratio": payout_ratio,
        }
        valuation_results["ggm"] = ggm_result
        logger.info("%s - GGM not applicable: %s", symbol, reason)
        log_model_result(logger, symbol, "GGM", ggm_result)

    try:
        damodaran_model = DamodaranDCFModel(company_profile)
        damodaran_result = damodaran_model.calculate(
            current_fcf=financials.get("free_cash_flow") or financials.get("fcf"),
            revenue_growth=company_profile.revenue_growth_yoy,
            fcf_margin=ratios.get("fcf_margin") or ratios.get("free_cash_flow_margin"),
            current_revenue=financials.get("revenues") or financials.get("revenue") or financials.get("total_revenue"),
            shares_outstanding=company_profile.shares_outstanding,
        )
        normalized_damodaran = normalize_model_output(damodaran_result)
        valuation_results["damodaran_dcf"] = normalized_damodaran
        log_model_result(logger, symbol, "Damodaran DCF", normalized_damodaran)
    except Exception as exc:
        logger.warning("%s - Damodaran DCF failed: %s", symbol, exc)
        valuation_results["damodaran_dcf"] = {"applicable": False, "reason": str(exc), "model": "damodaran_dcf"}

    is_saas_company = bool(
        company_profile.industry
        and any(kw in company_profile.industry.lower() for kw in ["software", "saas", "cloud", "internet"])
    )
    is_growth_company = bool(company_profile.revenue_growth_yoy and company_profile.revenue_growth_yoy > 0.10)

    if is_saas_company or is_growth_company:
        try:
            rule_of_40_model = RuleOf40Valuation(company_profile)
            rule_of_40_result = rule_of_40_model.calculate(
                revenue_growth=company_profile.revenue_growth_yoy,
                fcf_margin=ratios.get("fcf_margin") or ratios.get("free_cash_flow_margin"),
                current_revenue=financials.get("revenues")
                or financials.get("revenue")
                or financials.get("total_revenue"),
                current_price=market_data.get("price") or market_data.get("close") or market_data.get("current_price"),
                shares_outstanding=company_profile.shares_outstanding,
            )
            normalized_rule_of_40 = normalize_model_output(rule_of_40_result)
            valuation_results["rule_of_40"] = normalized_rule_of_40
            log_model_result(logger, symbol, "Rule of 40", normalized_rule_of_40)
        except Exception as exc:
            logger.warning("%s - Rule of 40 failed: %s", symbol, exc)
            valuation_results["rule_of_40"] = {"applicable": False, "reason": str(exc), "model": "rule_of_40"}
    else:
        valuation_results["rule_of_40"] = {
            "applicable": False,
            "reason": "Not a growth/SaaS company (requires >10% revenue growth or SaaS industry)",
            "model": "rule_of_40",
        }

    if is_saas_company:
        try:
            saas_model = SaaSValuationModel(company_profile)
            saas_result = saas_model.calculate(
                revenue_growth=company_profile.revenue_growth_yoy,
                current_revenue=financials.get("revenues") or financials.get("revenue") or financials.get("total_revenue"),
                current_price=market_data.get("price") or market_data.get("close") or market_data.get("current_price"),
                shares_outstanding=company_profile.shares_outstanding,
                gross_margin=ratios.get("gross_margin") or ratios.get("gross_profit_margin"),
                nrr=ratios.get("net_revenue_retention") or ratios.get("nrr"),
                ltv_cac=ratios.get("ltv_cac") or ratios.get("ltv_cac_ratio"),
                fcf_margin=ratios.get("fcf_margin") or ratios.get("free_cash_flow_margin"),
            )
            normalized_saas = normalize_model_output(saas_result)
            valuation_results["saas"] = normalized_saas
            log_model_result(logger, symbol, "SaaS", normalized_saas)
        except Exception as exc:
            logger.warning("%s - SaaS valuation failed: %s", symbol, exc)
            valuation_results["saas"] = {"applicable": False, "reason": str(exc), "model": "saas"}
    else:
        valuation_results["saas"] = {
            "applicable": False,
            "reason": "Not a SaaS company (requires software/cloud/internet industry)",
            "model": "saas",
        }

    return payout_ratio
