"""Helpers for preparing valuation models for multi-model blending."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def collect_models_for_blending(
    *,
    dcf_professional: Optional[Dict[str, Any]],
    valuation_results: Dict[str, Any],
    normalized_pe: Optional[Dict[str, Any]],
    normalized_ev_ebitda: Optional[Dict[str, Any]],
    normalized_ps: Optional[Dict[str, Any]],
    normalized_pb: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Collect applicable valuation model payloads for orchestrator blending."""
    models_for_blending: List[Dict[str, Any]] = []
    info_messages: List[str] = []

    if isinstance(dcf_professional, dict):
        models_for_blending.append(dcf_professional)
    ggm_entry = valuation_results.get("ggm")
    if isinstance(ggm_entry, dict):
        models_for_blending.append(ggm_entry)
    if isinstance(normalized_pe, dict):
        models_for_blending.append(normalized_pe)
    if isinstance(normalized_ev_ebitda, dict):
        models_for_blending.append(normalized_ev_ebitda)
    if isinstance(normalized_ps, dict):
        models_for_blending.append(normalized_ps)
    if isinstance(normalized_pb, dict):
        models_for_blending.append(normalized_pb)

    sector_spec = valuation_results.get("sector_specific")
    if isinstance(sector_spec, dict) and "P/BV" not in str(sector_spec.get("method", "")):
        models_for_blending.append(sector_spec)
        info_messages.append(
            f"Added sector-specific valuation to blending: {sector_spec.get('method')}"
        )

    damodaran = valuation_results.get("damodaran_dcf")
    if isinstance(damodaran, dict) and damodaran.get("applicable"):
        models_for_blending.append(damodaran)
        info_messages.append("Added Damodaran DCF to blending")

    rule_of_40 = valuation_results.get("rule_of_40")
    if isinstance(rule_of_40, dict) and rule_of_40.get("applicable"):
        models_for_blending.append(rule_of_40)
        info_messages.append("Added Rule of 40 to blending")

    saas = valuation_results.get("saas")
    if isinstance(saas, dict) and saas.get("applicable"):
        models_for_blending.append(saas)
        info_messages.append("Added SaaS valuation to blending")

    return models_for_blending, info_messages


def filter_models_for_company(
    *,
    models_for_blending: Sequence[Dict[str, Any]],
    allowed_models: Optional[Sequence[str]],
    industry: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[List[str]], bool]:
    """
    Apply allowed-model filtering and insurance P/B override rule.

    Returns: filtered models, resolved allowed-model list, insurance-pb-override-flag.
    """
    if allowed_models is None:
        return list(models_for_blending), None, False

    resolved_allowed_models = list(allowed_models)
    is_insurance = bool(industry and "insur" in industry.lower())
    added_pb_for_insurance = False
    if is_insurance and "pb" not in resolved_allowed_models:
        resolved_allowed_models.append("pb")
        added_pb_for_insurance = True

    filtered = [model for model in models_for_blending if model.get("model") in resolved_allowed_models]
    return filtered, resolved_allowed_models, added_pb_for_insurance


def _count_fcf_quarters(quarterly_metrics: Any) -> int:
    count = 0
    if not quarterly_metrics:
        return count

    for quarter in quarterly_metrics:
        if isinstance(quarter, dict):
            cash_flow = quarter.get("cash_flow", {})
            if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                count += 1
            continue

        if hasattr(quarter, "cash_flow"):
            cash_flow = getattr(quarter, "cash_flow", {})
            if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                count += 1

    return count


def hydrate_financials_for_blending(
    *,
    financials: Dict[str, Any],
    company_data: Dict[str, Any],
    company_profile: Any,
    ratios: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Add required fields used by model applicability and dynamic weighting checks."""
    if ratios:
        if "market_cap" in ratios and "market_cap" not in financials:
            financials["market_cap"] = ratios["market_cap"]
        if "shares_outstanding" in ratios and "shares_outstanding" not in financials:
            financials["shares_outstanding"] = ratios["shares_outstanding"]
        if "current_price" in ratios and "current_price" not in financials:
            financials["current_price"] = ratios["current_price"]

    if "revenue" not in financials or financials.get("revenue", 0) == 0:
        ttm_metrics = company_data.get("ttm_metrics", {})
        revenue_value = (
            financials.get("revenues")
            or financials.get("total_revenue")
            or ttm_metrics.get("revenues")
            or ttm_metrics.get("total_revenue")
            or ttm_metrics.get("revenue")
            or 0
        )
        if revenue_value and revenue_value > 0:
            financials["revenue"] = revenue_value

    fcf_quarters_count = _count_fcf_quarters(getattr(company_profile, "quarterly_metrics", None))
    financials["fcf_quarters_count"] = fcf_quarters_count

    profile_fcf = getattr(company_profile, "free_cash_flow", None)
    if profile_fcf:
        financials["free_cash_flow"] = profile_fcf
    elif isinstance(getattr(company_profile, "ttm_metrics", None), dict):
        financials["free_cash_flow"] = company_profile.ttm_metrics.get("free_cash_flow", 0)

    profile_dividends = getattr(company_profile, "dividends_paid", None)
    if profile_dividends:
        financials["dividends_paid"] = profile_dividends
    elif "dividends_paid" not in financials or financials.get("dividends_paid", 0) == 0:
        financials["dividends_paid"] = abs(financials.get("dividends_paid", 0) or 0)

    profile_ebitda = getattr(company_profile, "ebitda", None)
    if profile_ebitda:
        financials["ebitda"] = profile_ebitda
    elif "ebitda" not in financials or financials.get("ebitda", 0) == 0:
        ttm_metrics = company_data.get("ttm_metrics", {})
        ebitda_value = ttm_metrics.get("ebitda") or ttm_metrics.get("operating_income") or financials.get(
            "operating_income"
        )
        if ebitda_value:
            financials["ebitda"] = ebitda_value

    payout_ratio = getattr(company_profile, "dividend_payout_ratio", None)
    if payout_ratio:
        financials["payout_ratio"] = payout_ratio
    elif ratios and ratios.get("payout_ratio"):
        financials["payout_ratio"] = ratios["payout_ratio"]
    elif ratios and ratios.get("dividend_payout_ratio"):
        financials["payout_ratio"] = ratios["dividend_payout_ratio"]

    profile_net_income = getattr(company_profile, "net_income", None)
    if profile_net_income:
        financials["net_income"] = profile_net_income
    elif "net_income" not in financials or financials.get("net_income", 0) == 0:
        net_income_value = (company_data.get("ttm_metrics", {}) or {}).get("net_income") or 0
        if net_income_value:
            financials["net_income"] = net_income_value

    if "stockholders_equity" not in financials and "book_value" not in financials:
        book_value = (
            financials.get("stockholders_equity")
            or financials.get("total_stockholders_equity")
            or (company_data.get("ttm_metrics", {}) or {}).get("stockholders_equity")
            or 0
        )
        if book_value:
            financials["stockholders_equity"] = book_value

    if fcf_quarters_count == 0 and financials.get("free_cash_flow", 0) > 0:
        fcf_quarters_count = 4
        financials["fcf_quarters_count"] = fcf_quarters_count

    return {
        "fcf_quarters_count": fcf_quarters_count,
        "free_cash_flow": financials.get("free_cash_flow", 0),
        "ebitda": financials.get("ebitda", 0),
        "dividends_paid": financials.get("dividends_paid", 0),
        "payout_ratio": financials.get("payout_ratio", 0),
        "net_income": financials.get("net_income", 0),
        "book_value": financials.get("stockholders_equity", financials.get("book_value", 0)),
    }


def apply_weight_lookup(
    *,
    multi_model_summary: Dict[str, Any],
    dcf_professional: Optional[Dict[str, Any]],
    valuation_results: Dict[str, Any],
    normalized_pe: Optional[Dict[str, Any]],
    normalized_ev_ebitda: Optional[Dict[str, Any]],
    normalized_ps: Optional[Dict[str, Any]],
    normalized_pb: Optional[Dict[str, Any]],
) -> None:
    """Propagate orchestrator-assigned weights back into individual model records."""
    weight_lookup = {
        model.get("model"): model.get("weight")
        for model in multi_model_summary.get("models", [])
        if isinstance(model, dict)
    }
    if isinstance(dcf_professional, dict) and "dcf" in weight_lookup:
        dcf_professional["weight"] = weight_lookup["dcf"]
    ggm_entry = valuation_results.get("ggm")
    if isinstance(ggm_entry, dict) and "ggm" in weight_lookup:
        ggm_entry["weight"] = weight_lookup["ggm"]
    if isinstance(normalized_pe, dict) and "pe" in weight_lookup:
        normalized_pe["weight"] = weight_lookup["pe"]
    if isinstance(normalized_ev_ebitda, dict) and "ev_ebitda" in weight_lookup:
        normalized_ev_ebitda["weight"] = weight_lookup["ev_ebitda"]
    if isinstance(normalized_ps, dict) and "ps" in weight_lookup:
        normalized_ps["weight"] = weight_lookup["ps"]
    if isinstance(normalized_pb, dict) and "pb" in weight_lookup:
        normalized_pb["weight"] = weight_lookup["pb"]
