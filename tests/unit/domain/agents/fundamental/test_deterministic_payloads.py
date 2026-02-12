from investigator.domain.agents.fundamental.deterministic_payloads import (
    build_deterministic_cache_record,
    build_deterministic_response,
)


def test_build_deterministic_response_contract():
    payload = {"score": 88, "assessment": "Strong"}
    result = build_deterministic_response("agent-x", "financial_health", payload)

    assert result["response"] == payload
    assert result["prompt"] == ""
    assert result["model_info"]["model"] == "deterministic-financial_health"
    assert result["model_info"]["temperature"] == 0.0
    assert result["metadata"]["agent_id"] == "agent-x"
    assert result["metadata"]["cache_type"] == "deterministic_analysis"
    assert result["metadata"]["generated_at"]


def test_build_deterministic_cache_record_with_period():
    key, wrapped = build_deterministic_cache_record(
        symbol="AAPL",
        agent_id="agent-y",
        label="growth_analysis",
        payload={"growth_score": 72},
        period="2025-Q4",
    )

    assert key == {"symbol": "AAPL", "llm_type": "growth_analysis", "period": "2025-Q4"}
    assert wrapped["response"]["growth_score"] == 72
    assert wrapped["metadata"]["agent_id"] == "agent-y"
    assert wrapped["metadata"]["analysis_type"] == "growth_analysis"
    assert wrapped["metadata"]["period"] == "2025-Q4"
    assert wrapped["metadata"]["cached_at"]


def test_build_deterministic_cache_record_without_period():
    key, wrapped = build_deterministic_cache_record(
        symbol="MSFT",
        agent_id="agent-z",
        label="profitability_analysis",
        payload={"profitability_score": 91},
        period=None,
    )

    assert key == {"symbol": "MSFT", "llm_type": "profitability_analysis"}
    assert wrapped["metadata"]["period"] is None
