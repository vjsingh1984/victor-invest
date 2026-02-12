"""Formatting helpers for valuation synthesis prompts and summary rendering."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from investigator.domain.services.data_normalizer import round_for_prompt
from investigator.domain.services.safe_formatters import format_number


def _to_row(name: str, result: Dict[str, Any], applicable_default: bool) -> Dict[str, Any]:
    return {
        "name": name,
        "fair_value": result.get("fair_value_per_share", 0),
        "confidence": result.get("confidence_score", 0) * 100,
        "weight": result.get("weight", 0),
        "applicable": result.get("applicable", applicable_default),
    }


def build_valuation_summary_rows(
    *,
    dcf_professional: Optional[Dict[str, Any]],
    ggm_entry: Optional[Dict[str, Any]],
    normalized_pe: Optional[Dict[str, Any]],
    normalized_ev_ebitda: Optional[Dict[str, Any]],
    normalized_ps: Optional[Dict[str, Any]],
    normalized_pb: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build normalized valuation-model rows for table rendering."""
    rows: List[Dict[str, Any]] = []
    if isinstance(dcf_professional, dict):
        rows.append(_to_row("DCF", dcf_professional, True))
    if isinstance(ggm_entry, dict):
        rows.append(_to_row("GGM", ggm_entry, False))
    if isinstance(normalized_pe, dict):
        rows.append(_to_row("P/E", normalized_pe, True))
    if isinstance(normalized_ev_ebitda, dict):
        rows.append(_to_row("EV/EBITDA", normalized_ev_ebitda, True))
    if isinstance(normalized_ps, dict):
        rows.append(_to_row("P/S", normalized_ps, True))
    if isinstance(normalized_pb, dict):
        rows.append(_to_row("P/B", normalized_pb, True))
    return rows


def build_models_detail_lines(
    models: Iterable[Dict[str, Any]],
    *,
    format_currency: Callable[[Any], str],
    format_percentage: Callable[[Any], str],
) -> List[str]:
    """Build human-readable per-model detail lines used in valuation synthesis prompt."""

    def _fmt_float(value: Any, decimals: int = 1) -> str:
        return format_number(value, decimals=decimals, thousands_separator=False)

    lines: List[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_name = model.get("methodology") or model.get("model")
        if model.get("applicable"):
            line = (
                f"- {model_name}: Fair Value {format_currency(model.get('fair_value_per_share'))}, "
                f"Weight {round(model.get('weight', 0.0), 3):.3f}, "
                f"Confidence {round(model.get('confidence_score', 0.0), 3):.3f}"
            )
            assumptions = model.get("assumptions") or {}
            metadata = model.get("metadata") or {}
            extra_bits: List[str] = []
            if model.get("model") == "dcf":
                if "wacc" in assumptions:
                    extra_bits.append(f"WACC {format_percentage(assumptions.get('wacc'))}")
                if metadata.get("rule_of_40"):
                    r40 = metadata["rule_of_40"]
                    score = round_for_prompt(r40.get("score", 0), 1)
                    if score is not None:
                        extra_bits.append(f"Rule of 40 {score:.1f} ({r40.get('classification', '').upper()})")
            if model.get("model") == "ggm":
                if "growth_rate" in assumptions:
                    extra_bits.append(f"Growth {format_percentage(assumptions.get('growth_rate'))}")
                if "discount_rate" in assumptions:
                    extra_bits.append(f"Cost of equity {format_percentage(assumptions.get('discount_rate'))}")
            if model.get("model") == "pe":
                if assumptions.get("target_pe"):
                    extra_bits.append(f"Target P/E {_fmt_float(assumptions.get('target_pe'), 2)}")
                if assumptions.get("sector_median_pe"):
                    extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_pe'), 2)}")
            if model.get("model") == "ev_ebitda":
                if assumptions.get("target_ev_ebitda"):
                    extra_bits.append(f"Target EV/EBITDA {_fmt_float(assumptions.get('target_ev_ebitda'), 2)}")
                if assumptions.get("sector_median_ev_ebitda"):
                    extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_ev_ebitda'), 2)}")
            if model.get("model") == "ps":
                if assumptions.get("target_ps"):
                    extra_bits.append(f"Target P/S {_fmt_float(assumptions.get('target_ps'), 2)}")
                if assumptions.get("sector_median_ps"):
                    extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_ps'), 2)}")
            if model.get("model") == "pb":
                if assumptions.get("target_pb"):
                    extra_bits.append(f"Target P/B {_fmt_float(assumptions.get('target_pb'), 2)}")
                if assumptions.get("sector_median_pb"):
                    extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_pb'), 2)}")
            if extra_bits:
                line += " | " + ", ".join(extra_bits)
        else:
            line = f"- {model_name}: Not applicable ({model.get('reason', 'no reason provided')})"
        lines.append(line)

    if not lines:
        return ["- No valuation models produced outputs."]
    return lines


def build_valuation_synthesis_prompt(
    *,
    data_quality: Dict[str, Any],
    trend_context: str,
    sector: Optional[str],
    industry: Optional[str],
    archetype_labels: str,
    data_quality_flags: Sequence[str],
    current_price: Any,
    market_cap: Any,
    payout_ratio: Any,
    blended_fair_value: Any,
    overall_confidence: Any,
    model_agreement_score: Any,
    divergence_flag: Any,
    applicable_models: Any,
    notes: Sequence[str],
    models_detail_lines: Sequence[str],
    format_currency: Callable[[Any], str],
    format_int_with_commas: Callable[[Any], str],
    format_percentage: Callable[[Any], str],
) -> str:
    """Build valuation synthesis prompt from precomputed model outputs."""

    def _fmt_float(value: Any, decimals: int = 1) -> str:
        return format_number(value, decimals=decimals, thousands_separator=False)

    notes_section = "\n".join(f"- {note}" for note in notes) if notes else "- None"
    models_detail_section = "\n".join(models_detail_lines)
    data_flags = ", ".join(data_quality_flags) or "None"
    consistency_issues = ", ".join(data_quality.get("consistency_issues", [])) or "None detected"

    return f"""
        Synthesize a fair value estimate using the multi-model valuation summary below. Anchor your assessment on the
        blended output and explain any material disagreements across models.

        DATA QUALITY ASSESSMENT:
        - Overall Quality: {data_quality.get('quality_grade', 'Unknown')} ({format_percentage(data_quality.get('data_quality_score', 0))})
        - {data_quality.get('assessment', 'Data quality information not available')}
        - Core Metrics: {data_quality.get('core_metrics_populated', 'N/A')} populated
        - Market Data: {data_quality.get('market_metrics_populated', 'N/A')} populated
        - Consistency Issues: {consistency_issues}
        {trend_context}

        COMPANY PROFILE SNAPSHOT:
        - Sector: {sector}
        - Industry: {industry or 'N/A'}
        - Archetypes: {archetype_labels}
        - Data Flags: {data_flags}

        MARKET CONTEXT:
        - Current Price: {format_currency(current_price)}
        - Market Cap: ${format_int_with_commas(market_cap)}
        - Dividend Payout Ratio: {format_percentage(payout_ratio)}

        === MULTI-MODEL VALUATION SUMMARY ===
        - Blended Fair Value: {format_currency(blended_fair_value)}
        - Overall Confidence: {_fmt_float(overall_confidence or 0, 3)}
        - Model Agreement Score: {_fmt_float(model_agreement_score or 0, 3)} (lower implies divergence)
        - Divergence Flag: {divergence_flag}
        - Applicable Models: {applicable_models}
        - Notes:\n{notes_section}

        === INDIVIDUAL MODEL DETAIL ===
        {models_detail_section}

        VALUATION SYNTHESIS INSTRUCTIONS:
        1. Produce a blended fair value estimate (state the number) and the implied upside/downside vs. current price.
        2. Explain how each model influences the final number, especially when weights differ.
        3. Comment on confidence and whether divergence or data-quality flags warrant caution.
        4. Highlight key drivers/assumptions (growth, margins, discount rates) that matter most.
        5. Outline valuation risks or scenarios that could shift the blend materially.
        6. Recommend a valuation stance (Undervalued / Fairly Valued / Overvalued) and suggest a margin of safety target.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object that captures these points and follows the schema below (values are illustrative):
        {{
          "fair_value_estimate": 150.00,
          "implied_upside_downside": 0.15,
          "model_influence_explanation": "The DCF model has the highest weight (50%) and is the primary driver of the fair value estimate. The P/E and P/S models are used as secondary inputs.",
          "confidence_and_caution": "Confidence is high due to good data quality and model agreement. No divergence flags were raised.",
          "key_drivers_and_assumptions": "The key drivers are the assumed 5% terminal growth rate and the 10% WACC.",
          "valuation_risks": "A significant increase in interest rates or a slowdown in revenue growth could negatively impact the valuation.",
          "valuation_stance": "Undervalued",
          "margin_of_safety_target": 0.20
        }}
        """
