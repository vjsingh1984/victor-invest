"""Logging and formatting helpers for the fundamental analysis agent."""

from __future__ import annotations

import datetime
import json
import logging
import math
from typing import Any, Dict, List, Optional

from investigator.domain.services.data_normalizer import round_for_prompt
from investigator.infrastructure.formatters import ValuationTableFormatter

from .models import QuarterlyData


def log_data_quality_issues(logger: logging.Logger, symbol: str, company_data: Dict, ratios: Dict) -> None:
    """Emit standardized data-quality warnings for downstream monitoring."""
    issues: List[str] = []

    if company_data.get("market_cap", 0) == 0:
        issues.append("market_cap_zero")

    if ratios.get("operating_cash_flow", 0) == 0 and ratios.get("free_cash_flow", 0) == 0:
        issues.append("no_cash_flow_data")

    if ratios.get("debt_to_equity", 0) > 0 and ratios.get("debt_to_assets", 0) == 0:
        issues.append("debt_ratio_inconsistency")

    if company_data.get("market_data", {}).get("current_price", 0) == 0:
        issues.append("price_zero")

    if issues:
        logger.warning("📊 DATA QUALITY ISSUES for %s: %s", symbol, ", ".join(issues))


def format_trend_context(company_data: Dict[str, Any]) -> str:
    """Format multi-quarter trend analysis for prompts/logs."""
    trend_analysis = company_data.get("trend_analysis")
    if not trend_analysis:
        return ""

    num_quarters = trend_analysis.get("num_quarters", 0)
    if num_quarters < 4:
        return ""

    revenue = trend_analysis.get("revenue", {})
    margins = trend_analysis.get("margins", {})
    cash_flow = trend_analysis.get("cash_flow", {})
    comparisons = trend_analysis.get("comparisons", {})
    cyclical = trend_analysis.get("cyclical", {})

    quality_score = cash_flow.get("quality_of_earnings", 0)
    quality_rating = (
        "EXCELLENT"
        if quality_score >= 90
        else "GOOD" if quality_score >= 75 else "FAIR" if quality_score >= 60 else "POOR"
    )

    return (
        "\n"
        f"MULTI-QUARTER TREND ANALYSIS ({num_quarters} quarters):\n"
        f"Revenue Trend: {revenue.get('trend', 'unknown')} ("
        f"{revenue.get('early_avg_growth', 0):.1f}% → {revenue.get('late_avg_growth', 0):.1f}%, "
        f"consistency {revenue.get('consistency_score', 0):.0f}/100)\n"
        "Margin Trends:\n"
        f"  - Gross Margin: {margins.get('gross_margin_trend', 'unknown')}\n"
        f"  - Operating Margin: {margins.get('operating_margin_trend', 'unknown')}\n"
        f"  - Net Margin: {margins.get('net_margin_trend', 'unknown')}\n"
        f"Cash Flow Quality: {quality_score:.0f}/100 ({quality_rating})\n"
        f"Quarterly Comparisons: Q/Q {comparisons.get('latest_qoq_growth', 0):.1f}% | "
        f"Y/Y {comparisons.get('avg_yoy_growth', 0):.1f}%\n"
        f"Seasonality: {cyclical.get('seasonal_pattern', 'unknown')}\n"
    )


def format_shared_context(company_data: Dict[str, Any]) -> str:
    """Compose shared data-quality and trend summaries."""
    data_quality = company_data.get("data_quality", {})
    quality_section = (
        "\nDATA QUALITY ASSESSMENT:\n"
        f"- Overall Quality: {data_quality.get('quality_grade', 'Unknown')} "
        f"({data_quality.get('data_quality_score', 0):.1f}%)\n"
        f"- Completeness: {data_quality.get('completeness', 0):.1f}%\n"
        f"- Consistency: {data_quality.get('consistency', 0):.1f}%\n"
    )

    trend_section = format_trend_context(company_data)
    return quality_section + ("\n" + trend_section if trend_section else "")


def log_quarterly_snapshot(
    logger: logging.Logger,
    symbol: str,
    quarterly_data: List[QuarterlyData],
) -> None:
    """Render a compact quarterly snapshot table in the logs."""
    if not quarterly_data:
        return

    limit = min(len(quarterly_data), 8)
    selected = quarterly_data[-limit:]
    rows: List[List[str]] = []

    for quarter in reversed(selected):
        if isinstance(quarter, QuarterlyData):
            label = quarter.period_label
            data = quarter.financial_data or {}
            quality = quarter.data_quality or {}
            if not isinstance(data, dict):
                logger.debug(
                    "Legacy data format for %s: financial_data=%s; coercing to dict",
                    label,
                    type(data).__name__,
                )
                data = {}
            if not isinstance(quality, dict):
                logger.debug(
                    "Legacy data format for %s: data_quality=%s; coercing to dict",
                    label,
                    type(quality).__name__,
                )
                quality = {}
        else:
            logger.warning("Skipping unexpected quarterly data type: %s", type(quarter))
            continue

        rows.append(
            [
                label,
                _format_currency(data.get("revenues")),
                _format_currency(data.get("net_income")),
                _format_currency(data.get("operating_cash_flow")),
                _format_currency(data.get("free_cash_flow")),
                _format_percent(quality.get("completeness")),
                _format_percent(quality.get("consistency")),
            ]
        )

    log_table(
        logger,
        f"{symbol} Quarterly Snapshot (latest {limit} quarters)",
        ["Period", "Revenue", "Net Income", "OCF", "FCF", "Completeness", "Consistency"],
        rows,
    )


def log_table(
    logger: logging.Logger,
    title: str,
    headers: List[str],
    rows: List[List[str]],
    level: str = "info",
) -> None:
    if not rows:
        return

    body = _format_table(headers, rows)
    log_fn = getattr(logger, level, logger.info)
    log_fn("\n%s\n%s", title, body)


def log_individual_model_result(
    logger: logging.Logger,
    symbol: str,
    model_name: str,
    result: Dict[str, Any],
) -> None:
    if not isinstance(result, dict):
        logger.warning("%s - %s result unavailable or invalid", symbol, model_name)
        return

    fair_value = result.get("fair_value_per_share")
    confidence = result.get("confidence_score")
    applicable = result.get("applicable", True)

    if not applicable:
        reason = result.get("reason", "Unknown reason")
        logger.info("📊 %s - %s: NOT APPLICABLE (%s)", symbol, model_name, reason)
        return

    fair_value_str = f"${fair_value:.2f}" if fair_value else "N/A"
    confidence_str = f"{confidence:.1%}" if confidence else "N/A"
    logger.info(
        "📊 %s - %s: Fair Value = %s | Confidence = %s",
        symbol,
        model_name,
        fair_value_str,
        confidence_str,
    )


def log_valuation_snapshot(
    logger: logging.Logger,
    symbol: str,
    valuation_results: Dict[str, Any],
) -> None:
    """Render comprehensive valuation summary using ValuationTableFormatter."""
    logger.debug("METHOD ENTRY: log_valuation_snapshot for %s", symbol)
    logger.debug("valuation_results keys = %s", list(valuation_results.keys()))

    if "multi_model" in valuation_results:
        logger.debug("multi_model keys = %s", list(valuation_results["multi_model"].keys()))
        if "models" in valuation_results["multi_model"]:
            models_preview = valuation_results["multi_model"]["models"]
            logger.debug("multi_model['models'] type = %s", type(models_preview))
            logger.debug("multi_model['models'] length = %s", len(models_preview))
            if isinstance(models_preview, list):
                for i, model in enumerate(models_preview[:3]):
                    logger.debug("models[%s] preview = %s", i, model)

    with open("/tmp/valuation_snapshot_debug.log", "a", encoding="utf-8") as handle:
        handle.write(f"{datetime.datetime.now()} - METHOD ENTRY: log_valuation_snapshot for {symbol}\n")
        handle.write(f"{datetime.datetime.now()} - valuation_results keys: {list(valuation_results.keys())}\n")
        if "multi_model" in valuation_results:
            handle.write(
                f"{datetime.datetime.now()} - multi_model structure: {json.dumps(valuation_results['multi_model'], indent=2, default=str)}\n"
            )
        handle.flush()

    try:
        logger.info("📊 %s - Formatting valuation summary table...", symbol)
        multi_model = valuation_results.get("multi_model") or {}
        multi_model_summary = multi_model  # backwards compatibility alias for downstream references
        logger.debug(
            "%s - multi_model after assignment: %s, keys=%s",
            symbol,
            type(multi_model),
            list(multi_model.keys()),
        )
        logger.info("📊 %s - Multi-model keys: %s", symbol, list(multi_model.keys()))

        blended_fair_value = multi_model.get("blended_fair_value", 0)
        tier_classification = multi_model.get("tier_classification", "N/A")
        logger.info(
            "📊 %s - Blended FV: %s, Tier: %s",
            symbol,
            blended_fair_value,
            tier_classification,
        )

        current_price = valuation_results.get("current_price")
        if not isinstance(current_price, (int, float)):
            current_price = multi_model.get("current_price")
        if not isinstance(current_price, (int, float)):
            for model_entry in multi_model.get("models", []):
                if isinstance(model_entry, dict):
                    metadata = model_entry.get("metadata", {})
                    if isinstance(metadata, dict) and isinstance(
                        metadata.get("current_price"),
                        (int, float),
                    ):
                        current_price = metadata["current_price"]
                        break
        current_price = _to_numeric(current_price)

        all_models_data: List[Dict[str, Any]] = []
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s - Checking if multi_model has 'models' key", symbol)
            logger.debug("%s - multi_model.get('models') type = %s", symbol, type(multi_model.get("models")))
            logger.debug(
                "%s - multi_model.get('models') list check = %s",
                symbol,
                isinstance(multi_model.get("models"), list),
            )

        if isinstance(multi_model.get("models"), list):
            models_list = multi_model.get("models", [])
            logger.debug("%s - models list length = %s", symbol, len(models_list))
            logger.info("📊 %s - Found models list with %s entries", symbol, len(models_list))
            for idx, model in enumerate(models_list):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "%s - Processing model %s: type=%s, is_dict=%s",
                        symbol,
                        idx,
                        type(model),
                        isinstance(model, dict),
                    )
                if isinstance(model, dict):
                    model_name = model.get("model") or model.get("methodology", "unknown")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "%s - Model %s name=%s applicable=%s",
                            symbol,
                            idx,
                            model_name,
                            model.get("applicable"),
                        )
                    _append_model_entry(all_models_data, model, model_name)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("%s - Model %s appended (total=%s)", symbol, idx, len(all_models_data))
        else:
            logger.info("📊 %s - No models list found, using fallback extraction", symbol)
            for key, model_name in [
                ("dcf_professional", "DCF"),
                ("ggm", "GGM"),
                ("pe", "P/E"),
                ("ev_ebitda", "EV/EBITDA"),
                ("ps", "P/S"),
                ("pb", "P/B"),
            ]:
                model_data = valuation_results.get(key)
                if isinstance(model_data, dict):
                    _append_model_entry(all_models_data, model_data, model_name)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s - FINAL all_models_data length = %s", symbol, len(all_models_data))

        try:
            logger.info("📊 %s - Collected %s model entries", symbol, len(all_models_data))
        except Exception as logger_ex:  # pragma: no cover - defensive
            logger.debug("LOGGER EXCEPTION while logging model count: %s", logger_ex)

        if all_models_data:
            logger.info("📊 %s - Calling ValuationTableFormatter...", symbol)
            valuation_table = ValuationTableFormatter.format_valuation_summary_table(
                symbol=symbol,
                all_models=all_models_data,
                dynamic_weights={
                    entry["name"].lower(): (
                        entry.get("weight", 0) / 100.0 if isinstance(entry.get("weight"), (int, float)) else 0
                    )
                    for entry in all_models_data
                },
                blended_fair_value=blended_fair_value or 0,
                current_price=current_price or 0,
                tier=tier_classification,
                notes=(multi_model.get("notes") if isinstance(multi_model, dict) else None),
            )
            logger.info(valuation_table)
            logger.info("📊 %s - ✅ Valuation table logged successfully", symbol)
        else:
            logger.warning("%s - No valuation models available for table formatting", symbol)

    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("%s - ❌ Failed to format valuation summary table: %s", symbol, exc)
        if valuation_results.get("multi_model"):
            logger.info("%s - Multi-model summary: %s", symbol, valuation_results["multi_model"])


def _append_model_entry(
    collector: List[Dict[str, Any]],
    model_dict: Dict[str, Any],
    display_name: Optional[str] = None,
) -> None:
    resolved_name = display_name or model_dict.get("model") or model_dict.get("methodology") or "UNKNOWN"
    resolved_name = str(resolved_name).upper()

    fair_value_raw = model_dict.get("fair_value_per_share")
    fair_value_value = fair_value_raw if isinstance(fair_value_raw, (int, float)) else None
    confidence_pct = _to_percent(model_dict.get("confidence_score"))
    weight_pct = _to_percent(model_dict.get("weight"))
    applicable_flag = bool(model_dict.get("applicable", True)) and fair_value_value is not None and weight_pct > 0

    collector.append(
        {
            "name": resolved_name,
            "fair_value": fair_value_value if fair_value_value is not None else 0,
            "confidence": confidence_pct,
            "weight": weight_pct,
            "applicable": applicable_flag,
        }
    )


def _to_numeric(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _to_percent(value: Any) -> float:
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        return numeric_value * 100.0 if numeric_value <= 1 else numeric_value
    return 0.0


def _format_currency(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(val) or math.isinf(val):
        return "-"
    sign = "-" if val < 0 else ""
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.2f}K"
    return f"{sign}${abs_val:.0f}"


def _format_percent(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(val) or math.isinf(val):
        return "-"
    return f"{val:.1f}%"


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    string_rows = [[str(cell) if cell not in (None, "") else "-" for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    header_line = _format_row(headers)
    divider = "-+-".join("-" * width for width in widths)
    data_lines = [_format_row(row) for row in string_rows]
    return "\n".join([header_line, divider, *data_lines])


def _extract_model_notes(model: Dict[str, Any]) -> str:
    model_id = (model.get("model") or "").lower()
    assumptions = model.get("assumptions") or {}
    metadata = model.get("metadata") or {}
    extras: List[str] = []

    if model_id == "dcf":
        wacc = assumptions.get("wacc")
        if wacc is not None:
            extras.append(f"WACC {_format_percent(wacc)}")
        terminal_growth_rate = assumptions.get("terminal_growth") or assumptions.get("terminal_growth_rate")
        if terminal_growth_rate is not None:
            extras.append(f"Terminal growth {_format_percent(terminal_growth_rate)}")
        rule = metadata.get("rule_of_40")
        if isinstance(rule, dict):
            score = rule.get("score")
            classification = (rule.get("classification") or "").upper()
            if isinstance(score, (int, float)):
                extras.append(f"Rule of 40 {round_for_prompt(score, 1):.1f} ({classification})")
            elif classification:
                extras.append(f"Rule of 40 ({classification})")

    elif model_id == "ggm":
        growth = assumptions.get("growth_rate")
        if growth is not None:
            extras.append(f"Growth {_format_percent(growth)}")
        discount = assumptions.get("discount_rate")
        if discount is not None:
            extras.append(f"Cost of equity {_format_percent(discount)}")

    elif model_id == "pe":
        target = assumptions.get("target_pe")
        if isinstance(target, (int, float)):
            extras.append(f"Target P/E {round_for_prompt(target, 2):.2f}")
        sector_median = assumptions.get("sector_median_pe")
        if isinstance(sector_median, (int, float)):
            extras.append(f"Sector median {round_for_prompt(sector_median, 2):.2f}")

    elif model_id == "ev_ebitda":
        target = assumptions.get("target_ev_ebitda")
        if isinstance(target, (int, float)):
            extras.append(f"Target EV/EBITDA {round_for_prompt(target, 2):.2f}")
        sector_median = assumptions.get("sector_median_ev_ebitda")
        if isinstance(sector_median, (int, float)):
            extras.append(f"Sector median {round_for_prompt(sector_median, 2):.2f}")

    elif model_id == "ps":
        target = assumptions.get("target_ps")
        if isinstance(target, (int, float)):
            extras.append(f"Target P/S {round_for_prompt(target, 2):.2f}")
        sector_median = assumptions.get("sector_median_ps")
        if isinstance(sector_median, (int, float)):
            extras.append(f"Sector median {round_for_prompt(sector_median, 2):.2f}")

    elif model_id == "pb":
        target = assumptions.get("target_pb")
        if isinstance(target, (int, float)):
            extras.append(f"Target P/B {round_for_prompt(target, 2):.2f}")
        sector_median = assumptions.get("sector_median_pb")
        if isinstance(sector_median, (int, float)):
            extras.append(f"Sector median {round_for_prompt(sector_median, 2):.2f}")

    return ", ".join(extras)


__all__ = [
    "format_shared_context",
    "format_trend_context",
    "log_data_quality_issues",
    "log_individual_model_result",
    "log_quarterly_snapshot",
    "log_table",
    "log_valuation_snapshot",
]
