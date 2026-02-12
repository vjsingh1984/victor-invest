from investigator.application.synthesizer_text_insights import (
    extract_comprehensive_insights,
    extract_comprehensive_risks,
    extract_insights_from_text,
)


def test_extract_insights_from_text_structured_sections():
    text = """
    Key Insights:
    - Revenue growth accelerated on enterprise mix
    - Margin expansion driven by automation

    Key Risks:
    - Customer concentration remains elevated
    - Pricing pressure in SMB segment
    """
    insights, risks = extract_insights_from_text(text)
    assert any("Revenue growth accelerated" in item for item in insights)
    assert any("Customer concentration" in item for item in risks)


def test_extract_insights_from_text_fallback_sentence_mode():
    text = (
        "The company has a strength in distribution and positive operating leverage. "
        "A major risk is customer churn pressure in legacy cohorts."
    )
    insights, risks = extract_insights_from_text(text)
    assert insights
    assert risks


def test_extract_comprehensive_risks_deduplicates_and_limits():
    llm_responses = {
        "fundamental": {
            "a": {"content": "Risks: - Supply chain concentration - Demand slowdown"},
            "b": {"content": {"risk_note": "Risk: Supply chain concentration"}},
        }
    }
    ai_recommendation = {"key_risks": ["Supply chain concentration", "Execution drift"]}
    risks = extract_comprehensive_risks(llm_responses, ai_recommendation, ["Demand slowdown"])
    assert "Execution drift" in risks
    assert any("Supply chain concentration" in r for r in risks)


def test_extract_comprehensive_insights_includes_technical_and_catalysts():
    llm_responses = {
        "fundamental": {
            "a": {"content": "Key Insights: - Durable pricing power - Improving retention"}
        },
        "technical": {
            "content": "KEY INSIGHTS: - Breakout above resistance - Volume confirmation\n\nNEXT SECTION"
        },
    }
    ai_recommendation = {"key_catalysts": ["AI demand cycle", "Margin expansion"]}
    insights = extract_comprehensive_insights(llm_responses, ai_recommendation, ["Free cash flow inflection"])
    assert any("Catalyst:" in item for item in insights)
    assert any("Technical:" in item for item in insights)
