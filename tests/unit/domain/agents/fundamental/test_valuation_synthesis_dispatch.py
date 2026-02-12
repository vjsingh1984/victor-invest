"""Unit tests for valuation synthesis dispatch helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investigator.domain.agents.fundamental.valuation_orchestrator import dispatch_valuation_synthesis


@pytest.mark.asyncio
@patch("investigator.domain.agents.fundamental.valuation_orchestrator.synthesize_valuation")
async def test_dispatch_valuation_synthesis_uses_deterministic_path(mock_synthesize):
    mock_synthesize.return_value = {"fair_value_estimate": 120.0}
    ollama = MagicMock()
    ollama.generate = AsyncMock()
    build_det = MagicMock(return_value={"wrapped": True})

    result = await dispatch_valuation_synthesis(
        symbol="AAPL",
        prompt="prompt",
        company_data={},
        market_data={"current_price": 100.0},
        valuation_results={"dcf": {}},
        multi_model_summary={"blended_fair_value": 120.0},
        data_quality={},
        company_profile_payload={},
        notes=[],
        use_deterministic=True,
        deterministic_valuation_synthesis=True,
        build_deterministic_response=build_det,
        debug_log_prompt=MagicMock(),
        debug_log_response=MagicMock(),
        ollama_client=ollama,
        valuation_model="model-x",
        cache_llm_response=AsyncMock(),
        wrap_llm_response=MagicMock(),
        logger=MagicMock(),
    )

    assert result == {"wrapped": True}
    mock_synthesize.assert_called_once()
    build_det.assert_called_once()
    ollama.generate.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_valuation_synthesis_uses_llm_path_and_wraps_response():
    ollama = MagicMock()
    ollama.generate = AsyncMock(return_value={"response": '{"fair_value_estimate": 130.0}'})
    cache_llm_response = AsyncMock()
    wrap_llm_response = MagicMock(return_value={"wrapped": "llm"})
    debug_log_prompt = MagicMock()
    debug_log_response = MagicMock()

    result = await dispatch_valuation_synthesis(
        symbol="MSFT",
        prompt="prompt",
        company_data={"fiscal_period": "2025-Q4"},
        market_data={"current_price": 100.0},
        valuation_results={"pe": {}},
        multi_model_summary={"blended_fair_value": 130.0},
        data_quality={},
        company_profile_payload={"sector": "Tech"},
        notes=["n1"],
        use_deterministic=False,
        deterministic_valuation_synthesis=True,
        build_deterministic_response=MagicMock(),
        debug_log_prompt=debug_log_prompt,
        debug_log_response=debug_log_response,
        ollama_client=ollama,
        valuation_model="model-y",
        cache_llm_response=cache_llm_response,
        wrap_llm_response=wrap_llm_response,
        logger=MagicMock(),
    )

    assert result == {"wrapped": "llm"}
    ollama.generate.assert_awaited_once()
    cache_llm_response.assert_awaited_once()
    wrap_llm_response.assert_called_once()
    wrapped_response = wrap_llm_response.call_args.kwargs["response"]
    assert wrapped_response["fair_value_estimate"] == 130.0
    assert "valuation_methods" in wrapped_response
    assert wrapped_response["current_price"] == 100.0
    debug_log_prompt.assert_called_once()
    debug_log_response.assert_called_once()
