"""Helpers for dynamic/fallback valuation model weighting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _count_fcf_quarters(company_profile: Any) -> int:
    fcf_quarters_count = 0
    quarterly_metrics = getattr(company_profile, "quarterly_metrics", None)
    if not quarterly_metrics:
        return fcf_quarters_count

    for quarter in quarterly_metrics:
        if isinstance(quarter, dict):
            cash_flow = quarter.get("cash_flow", {})
            if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                fcf_quarters_count += 1
            continue
        if hasattr(quarter, "cash_flow"):
            cash_flow = getattr(quarter, "cash_flow", {})
            if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                fcf_quarters_count += 1
    return fcf_quarters_count


def _model_get(model_result: Any, key: str, default: Any = None) -> Any:
    if isinstance(model_result, dict):
        return model_result.get(key, default)
    return getattr(model_result, key, default)


def resolve_fallback_weights(
    *,
    company_profile: Any,
    models_for_blending: List[Dict[str, Any]],
    financials: Optional[Dict[str, Any]],
    ratios: Optional[Dict[str, Any]],
    dynamic_weighting_service: Any,
    config: Any,
    logger: Any,
) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Determine dynamic model weights with static-fallback behavior.

    Returns `(weights_dict_or_none, tier_label)`.
    """
    try:
        symbol = company_profile.symbol if hasattr(company_profile, "symbol") else "UNKNOWN"

        if financials is None:
            logger.debug("%s - No financials provided, reconstructing from company_profile", symbol)
            financials = {
                "net_income": getattr(company_profile, "net_income", 0),
                "revenue": getattr(company_profile, "revenue", 0),
                "dividends_paid": getattr(company_profile, "dividends_paid", 0),
                "ebitda": getattr(company_profile, "ebitda", 0),
                "book_value": getattr(company_profile, "book_value", 0),
                "market_cap": 0,
                "current_price": 0,
            }
        else:
            financials.setdefault("market_cap", 0)
            financials.setdefault("current_price", 0)
            financials.setdefault("book_value", 0)

        fcf_quarters_count = _count_fcf_quarters(company_profile)
        financials["fcf_quarters_count"] = fcf_quarters_count

        if hasattr(company_profile, "free_cash_flow"):
            financials["free_cash_flow"] = getattr(company_profile, "free_cash_flow", 0)
        elif hasattr(company_profile, "ttm_metrics") and isinstance(company_profile.ttm_metrics, dict):
            financials["free_cash_flow"] = company_profile.ttm_metrics.get("free_cash_flow", 0)
        else:
            financials["free_cash_flow"] = 0

        if hasattr(company_profile, "dividends_paid") and company_profile.dividends_paid:
            financials["dividends_paid"] = company_profile.dividends_paid

        if hasattr(company_profile, "ebitda") and company_profile.ebitda:
            financials["ebitda"] = company_profile.ebitda

        if fcf_quarters_count == 0 and financials.get("free_cash_flow", 0) > 0:
            fcf_quarters_count = 4
            financials["fcf_quarters_count"] = fcf_quarters_count
            logger.info("%s - Inferred fcf_quarters_count=4 from TTM FCF value", symbol)

        logger.info(
            "%s - Applicability fields: fcf_quarters=%s, fcf=$%.2fB, ebitda=$%.2fB, dividends_paid=$%.2fB, book_value=$%.2fB",
            symbol,
            fcf_quarters_count,
            financials.get("free_cash_flow", 0) / 1e9,
            financials.get("ebitda", 0) / 1e9,
            financials.get("dividends_paid", 0) / 1e9,
            financials.get("book_value", 0) / 1e9,
        )

        if ratios is None:
            logger.debug("%s - No ratios provided, reconstructing from company_profile", symbol)
            ratios = {
                "payout_ratio": getattr(company_profile, "dividend_payout_ratio", 0),
                "rule_of_40_score": getattr(company_profile, "rule_of_40_score", 0),
                "revenue_growth_pct": getattr(company_profile, "revenue_growth_yoy", 0),
                "fcf_margin_pct": (
                    getattr(company_profile, "fcf_margin", 0) * 100 if getattr(company_profile, "fcf_margin", None) else 0
                ),
                "ttm_eps": 0,
            }

        logger.debug(
            "%s - DEBUG: models_for_blending type: %s, length: %s",
            symbol,
            type(models_for_blending),
            len(models_for_blending) if models_for_blending else 0,
        )

        for idx, model_result in enumerate(models_for_blending):
            if model_result is None:
                continue

            logger.debug("%s - DEBUG: model_result[%s] type: %s", symbol, idx, type(model_result))
            if isinstance(model_result, dict):
                logger.debug("%s - DEBUG: model_result[%s] keys: %s", symbol, idx, list(model_result.keys()))
                if "model_name" in model_result:
                    logger.debug(
                        "%s - DEBUG: model_result[%s] model_name: %s",
                        symbol,
                        idx,
                        model_result.get("model_name"),
                    )
            else:
                logger.debug(
                    "%s - DEBUG: model_result[%s] attributes: %s",
                    symbol,
                    idx,
                    [a for a in dir(model_result) if not a.startswith("_")][:10],
                )
                if hasattr(model_result, "model_name"):
                    logger.debug(
                        "%s - DEBUG: model_result[%s] model_name: %s",
                        symbol,
                        idx,
                        getattr(model_result, "model_name", None),
                    )

            model_name = _model_get(model_result, "model")

            if model_name == "pe":
                assumptions = _model_get(model_result, "assumptions", {}) or {}
                metadata = _model_get(model_result, "metadata", {}) or {}
                current_price = metadata.get("current_price") if isinstance(metadata, dict) else None
                if current_price is not None:
                    financials["current_price"] = current_price
                ttm_eps = assumptions.get("ttm_eps") if isinstance(assumptions, dict) else None
                if ttm_eps is not None:
                    ratios["ttm_eps"] = ttm_eps

            assumptions = _model_get(model_result, "assumptions", {}) or {}
            metadata = _model_get(model_result, "metadata", {}) or {}
            market_cap_from_assumptions = assumptions.get("market_cap", 0) if isinstance(assumptions, dict) else 0
            if market_cap_from_assumptions > 0:
                financials["market_cap"] = market_cap_from_assumptions
            market_cap_from_metadata = metadata.get("market_cap", 0) if isinstance(metadata, dict) else 0
            if market_cap_from_metadata > 0:
                financials["market_cap"] = market_cap_from_metadata

        if financials.get("market_cap", 0) == 0 and financials.get("current_price", 0) > 0:
            for model_result in models_for_blending:
                if model_result is None:
                    continue
                assumptions = _model_get(model_result, "assumptions", {}) or {}
                shares = assumptions.get("shares_outstanding", 0) if isinstance(assumptions, dict) else 0
                if shares > 0:
                    financials["market_cap"] = financials["current_price"] * shares
                    break

        data_quality = getattr(company_profile, "data_quality", None)
        weights, tier_classification, audit_trail = dynamic_weighting_service.determine_weights(
            symbol=symbol,
            financials=financials,
            ratios=ratios,
            data_quality=data_quality,
            market_context=None,
        )

        if audit_trail:
            audit_trail.log_summary()

        logger.info(
            "%s - Dynamic weights determined (tier=%s): %s",
            symbol,
            tier_classification,
            ", ".join([f"{model.upper()}={weight}%" for model, weight in weights.items() if weight > 0]),
        )

        return weights, tier_classification

    except Exception as exc:
        logger.warning("Dynamic weighting failed: %s. Falling back to static weights.", exc)

        valuation_settings = getattr(config, "valuation", None)
        if isinstance(valuation_settings, dict):
            fallback_cfg = valuation_settings.get("model_fallback", {})
        else:
            fallback_cfg = getattr(valuation_settings, "model_fallback", {}) if valuation_settings else {}

        if not isinstance(fallback_cfg, dict) or not fallback_cfg:
            return None, "fallback_error"

        def _normalize_key(key: Optional[str]) -> Optional[str]:
            return key.lower() if key else None

        primary_key = _normalize_key(company_profile.primary_archetype.name if company_profile.primary_archetype else None)
        if primary_key and primary_key in fallback_cfg:
            fallback_node = fallback_cfg[primary_key]
        elif primary_key and primary_key.capitalize() in fallback_cfg:
            fallback_node = fallback_cfg[primary_key.capitalize()]
        else:
            fallback_node = fallback_cfg.get("default")

        fallback_tier = "static_fallback"
        if not fallback_node:
            return None, "no_fallback_node"

        weights = fallback_node.get("weights") if isinstance(fallback_node, dict) else getattr(fallback_node, "weights", None)
        if not isinstance(weights, dict):
            return None, "invalid_fallback_weights"

        available_models = {model.get("model") for model in models_for_blending if model.get("model")}
        resolved = {
            model_key: float(weight)
            for model_key, weight in weights.items()
            if model_key in available_models and weight is not None
        }

        if resolved and all(v <= 1.0 for v in resolved.values()):
            resolved = {k: v * 100 for k, v in resolved.items()}

        return (resolved, fallback_tier) if resolved else (None, "no_resolved_weights")
