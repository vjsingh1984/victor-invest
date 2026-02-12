from investigator.application.synthesizer_insights import (
    analyze_field_completeness,
    check_fallback_flags,
    detect_markdown_content,
    extract_bullet_points,
    extract_decision_process,
    extract_numerical_insights,
    extract_priority_insights,
    extract_reasoning_themes,
    identify_custom_fields,
    recommend_report_sections,
    suggest_visualizations,
)


def test_extract_reasoning_themes_detects_core_patterns():
    thinking = (
        "Our technical analysis confirms strength. "
        "Fundamental analysis highlights margin durability. "
        "Risk assessment remains manageable."
    )
    themes = extract_reasoning_themes(thinking)
    assert any("technical analysis" in theme for theme in themes)
    assert any("fundamental analysis" in theme for theme in themes)
    assert any("risk assessment" in theme for theme in themes)


def test_extract_decision_process_flags_expected_markers():
    thinking = "First we consider valuation, then assess alternatives; however uncertainty remains."
    decision = extract_decision_process(thinking)
    assert decision["has_structured_approach"] is True
    assert decision["considers_alternatives"] is True
    assert decision["mentions_uncertainty"] is True


def test_markdown_and_bullet_extraction():
    content = "## Highlights\n- Strong free cash flow\n1. Margin expansion"
    assert detect_markdown_content(content) is True
    bullets = extract_bullet_points(content)
    assert "Strong free cash flow" in bullets
    assert "Margin expansion" in bullets


def test_extract_numerical_insights_detects_multiple_types_and_caps_results():
    text = "Revenue up 12.5% to $1,234.56 with leverage at 2:1 and EV/EBITDA near 10x."
    insights = extract_numerical_insights(text)
    types = {item["type"] for item in insights}
    assert {"percentage", "monetary", "ratio", "multiple"}.issubset(types)

    crowded = " ".join([f"{n}%" for n in range(20)])
    crowded_insights = extract_numerical_insights(crowded)
    assert len(crowded_insights) == 10


def test_recommendation_structure_helpers():
    recommendation = {
        "overall_score": 78,
        "investment_thesis": "Quality compounder",
        "recommendation": "BUY",
        "confidence_level": "HIGH",
        "position_size": "5%",
        "time_horizon": "12m",
        "risk_reward_ratio": "3:1",
        "key_catalysts": ["New product", "Margin expansion", "Buybacks"],
        "downside_risks": ["Competition", "Macro", "FX"],
        "processing_metadata": {"tokens": 900},
        "_fallback_created": False,
        "custom_note": "watch valuation",
    }

    completeness = analyze_field_completeness(recommendation)
    assert completeness["completeness_ratio"] == 1.0

    custom_fields = identify_custom_fields(recommendation)
    assert custom_fields == ["custom_note"]

    flags = check_fallback_flags(recommendation)
    assert flags["is_fallback"] is False
    assert flags["has_processing_metadata"] is True

    sections = recommend_report_sections(recommendation, "x" * 250, "y" * 120)
    assert "reasoning_analysis" in sections
    assert "additional_insights" in sections
    assert "catalyst_analysis" in sections
    assert "risk_assessment" in sections
    assert "methodology_notes" in sections

    visuals = suggest_visualizations(recommendation)
    assert "score_gauge_chart" in visuals
    assert "risk_catalyst_matrix" in visuals
    assert "timeline_chart" in visuals
    assert "processing_metrics_chart" in visuals


def test_extract_priority_insights_collects_key_phrases_and_markdown_highlights():
    thinking = "A key driver is pricing power. Another important factor is cash conversion."
    details = "## Upside path\n**Critical catalyst** is operating leverage."
    insights = extract_priority_insights(thinking, details)
    assert any("key driver" in item.lower() for item in insights)
    assert any("important factor" in item.lower() for item in insights)
    assert any("##" in item or "**" in item for item in insights)
