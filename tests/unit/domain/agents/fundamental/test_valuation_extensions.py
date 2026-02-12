"""Unit tests for valuation extension helper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investigator.domain.agents.fundamental.valuation_extensions import calculate_valuation_extensions


def _profile(industry="Software", revenue_growth_yoy=0.15):
    return SimpleNamespace(
        industry=industry,
        revenue_growth_yoy=revenue_growth_yoy,
        shares_outstanding=100,
    )


def _model_mock(result):
    model = MagicMock()
    model.calculate.return_value = result
    return model


@pytest.mark.asyncio
@patch("investigator.domain.agents.fundamental.valuation_extensions.SaaSValuationModel")
@patch("investigator.domain.agents.fundamental.valuation_extensions.RuleOf40Valuation")
@patch("investigator.domain.agents.fundamental.valuation_extensions.DamodaranDCFModel")
async def test_calculate_valuation_extensions_populates_models_when_applicable(
    damodaran_cls, rule_cls, saas_cls
):
    damodaran_cls.return_value = _model_mock({"model": "damodaran_dcf", "applicable": True})
    rule_cls.return_value = _model_mock({"model": "rule_of_40", "applicable": True})
    saas_cls.return_value = _model_mock({"model": "saas", "applicable": True})

    valuation_results = {}
    calculate_ggm = AsyncMock(return_value={"model": "ggm", "applicable": True})

    payout_ratio = await calculate_valuation_extensions(
        symbol="AAPL",
        valuation_results=valuation_results,
        financials={"dividends_paid": 200, "net_income": 500, "revenues": 1_000, "free_cash_flow": 100},
        ratios={"fcf_margin": 0.1, "gross_margin": 0.5},
        market_data={"current_price": 100.0},
        company_profile=_profile(),
        quarterly_data=[],
        calculate_cost_of_equity=lambda _symbol: 0.1,
        calculate_ggm=calculate_ggm,
        normalize_model_output=lambda payload: payload,
        log_model_result=lambda *_args, **_kwargs: None,
        logger=MagicMock(),
    )

    assert payout_ratio == 40.0
    assert valuation_results["ggm"]["applicable"] is True
    assert valuation_results["damodaran_dcf"]["model"] == "damodaran_dcf"
    assert valuation_results["rule_of_40"]["model"] == "rule_of_40"
    assert valuation_results["saas"]["model"] == "saas"
    calculate_ggm.assert_awaited_once()


@pytest.mark.asyncio
@patch("investigator.domain.agents.fundamental.valuation_extensions.SaaSValuationModel")
@patch("investigator.domain.agents.fundamental.valuation_extensions.RuleOf40Valuation")
@patch("investigator.domain.agents.fundamental.valuation_extensions.DamodaranDCFModel")
async def test_calculate_valuation_extensions_sets_non_applicable_paths(
    damodaran_cls, _rule_cls, _saas_cls
):
    damodaran_cls.return_value = _model_mock({"model": "damodaran_dcf", "applicable": True})
    valuation_results = {}

    payout_ratio = await calculate_valuation_extensions(
        symbol="KO",
        valuation_results=valuation_results,
        financials={"dividends_paid": 0, "net_income": 500, "revenues": 1_000, "free_cash_flow": 100},
        ratios={},
        market_data={"current_price": 50.0},
        company_profile=_profile(industry="Manufacturing", revenue_growth_yoy=0.05),
        quarterly_data=[],
        calculate_cost_of_equity=lambda _symbol: 0.1,
        calculate_ggm=AsyncMock(return_value={"model": "ggm", "applicable": True}),
        normalize_model_output=lambda payload: payload,
        log_model_result=lambda *_args, **_kwargs: None,
        logger=MagicMock(),
    )

    assert payout_ratio == 0
    assert valuation_results["ggm"]["applicable"] is False
    assert "No dividends paid" in valuation_results["ggm"]["reason"]
    assert valuation_results["rule_of_40"]["applicable"] is False
    assert valuation_results["saas"]["applicable"] is False
