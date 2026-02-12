"""Unit tests for valuation weighting helper and agent delegation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from investigator.domain.agents.fundamental.agent import FundamentalAnalysisAgent
from investigator.domain.agents.fundamental.valuation_weighting import resolve_fallback_weights


def _profile():
    return SimpleNamespace(
        symbol="AAPL",
        quarterly_metrics=[],
        free_cash_flow=1000,
        dividends_paid=100,
        ebitda=500,
        data_quality={"data_quality_score": 90},
        primary_archetype=SimpleNamespace(name="growth"),
        dividend_payout_ratio=20,
        rule_of_40_score=35,
        revenue_growth_yoy=0.12,
        fcf_margin=0.2,
    )


def test_resolve_fallback_weights_uses_dynamic_service_when_available():
    dynamic_service = MagicMock()
    dynamic_service.determine_weights.return_value = ({"dcf": 50.0, "pe": 30.0}, "tier_2", None)
    logger = MagicMock()

    weights, tier = resolve_fallback_weights(
        company_profile=_profile(),
        models_for_blending=[{"model": "dcf"}, {"model": "pe"}],
        financials={"book_value": 100},
        ratios={"payout_ratio": 20},
        dynamic_weighting_service=dynamic_service,
        config=SimpleNamespace(valuation={}),
        logger=logger,
    )

    assert weights == {"dcf": 50.0, "pe": 30.0}
    assert tier == "tier_2"
    dynamic_service.determine_weights.assert_called_once()


def test_resolve_fallback_weights_falls_back_to_static_config():
    dynamic_service = MagicMock()
    dynamic_service.determine_weights.side_effect = RuntimeError("boom")
    logger = MagicMock()
    config = SimpleNamespace(valuation={"model_fallback": {"growth": {"weights": {"dcf": 0.6, "pe": 0.4}}}})

    weights, tier = resolve_fallback_weights(
        company_profile=_profile(),
        models_for_blending=[{"model": "dcf"}, {"model": "pe"}],
        financials={"book_value": 100},
        ratios={"payout_ratio": 20},
        dynamic_weighting_service=dynamic_service,
        config=config,
        logger=logger,
    )

    assert tier == "static_fallback"
    assert weights == {"dcf": 60.0, "pe": 40.0}


def test_agent_resolve_fallback_weights_delegates_to_helper():
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent.dynamic_weighting_service = MagicMock()
    agent.config = MagicMock()
    agent.logger = MagicMock()
    agent._resolve_fallback_weights = FundamentalAnalysisAgent._resolve_fallback_weights.__get__(agent)

    expected = ({"dcf": 50.0}, "tier_1")
    with patch(
        "investigator.domain.agents.fundamental.agent.resolve_fallback_weights",
        return_value=expected,
    ) as helper:
        result = agent._resolve_fallback_weights(_profile(), [{"model": "dcf"}], {"book_value": 1}, {"payout_ratio": 1})

    assert result == expected
    helper.assert_called_once()
