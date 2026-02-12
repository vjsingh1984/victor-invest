"""Unit tests for valuation synthesis rendering helpers."""

from investigator.domain.agents.fundamental.valuation_synthesis import (
    build_models_detail_lines,
    build_valuation_summary_rows,
    build_valuation_synthesis_prompt,
)


def test_build_valuation_summary_rows_includes_supported_models():
    rows = build_valuation_summary_rows(
        dcf_professional={"fair_value_per_share": 120, "confidence_score": 0.8, "weight": 0.4},
        ggm_entry={"fair_value_per_share": 90, "confidence_score": 0.6, "weight": 0.1, "applicable": False},
        normalized_pe={"fair_value_per_share": 110, "confidence_score": 0.7, "weight": 0.2},
        normalized_ev_ebitda={"fair_value_per_share": 115, "confidence_score": 0.65, "weight": 0.15},
        normalized_ps=None,
        normalized_pb={"fair_value_per_share": 100, "confidence_score": 0.55, "weight": 0.15},
    )

    assert [row["name"] for row in rows] == ["DCF", "GGM", "P/E", "EV/EBITDA", "P/B"]
    assert rows[0]["confidence"] == 80.0
    assert rows[1]["applicable"] is False


def test_build_models_detail_lines_formats_assumptions_and_fallbacks():
    lines = build_models_detail_lines(
        [
            {
                "model": "dcf",
                "methodology": "DCF",
                "applicable": True,
                "fair_value_per_share": 120,
                "weight": 0.4,
                "confidence_score": 0.8,
                "assumptions": {"wacc": 0.1},
                "metadata": {"rule_of_40": {"score": 42.3, "classification": "good"}},
            },
            {
                "model": "ggm",
                "methodology": "GGM",
                "applicable": False,
                "reason": "No dividends",
            },
        ],
        format_currency=lambda value: f"${value}",
        format_percentage=lambda value: f"{value:.1%}",
    )

    assert "WACC 10.0%" in lines[0]
    assert "Rule of 40 42.3 (GOOD)" in lines[0]
    assert "Not applicable (No dividends)" in lines[1]


def test_build_valuation_synthesis_prompt_includes_key_sections():
    prompt = build_valuation_synthesis_prompt(
        data_quality={
            "quality_grade": "Good",
            "data_quality_score": 82.0,
            "assessment": "Good enough",
            "core_metrics_populated": "8/10",
            "market_metrics_populated": "2/2",
            "consistency_issues": [],
        },
        trend_context="Trend context block",
        sector="Technology",
        industry="Software",
        archetype_labels="Compounder",
        data_quality_flags=["missing_eps"],
        current_price=100.0,
        market_cap=1_000_000_000,
        payout_ratio=25.0,
        blended_fair_value=120.0,
        overall_confidence=0.78,
        model_agreement_score=0.66,
        divergence_flag=False,
        applicable_models=5,
        notes=["note-a", "note-b"],
        models_detail_lines=["- DCF: Fair Value $120"],
        format_currency=lambda value: f"${value}",
        format_int_with_commas=lambda value: f"{int(value):,}",
        format_percentage=lambda value: f"{value}",
    )

    assert "MULTI-MODEL VALUATION SUMMARY" in prompt
    assert "Sector: Technology" in prompt
    assert "- DCF: Fair Value $120" in prompt
    assert "- note-a" in prompt
