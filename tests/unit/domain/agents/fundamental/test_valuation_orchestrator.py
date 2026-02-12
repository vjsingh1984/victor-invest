"""Unit tests for valuation orchestration helpers."""

from unittest.mock import MagicMock
from types import SimpleNamespace

from investigator.domain.agents.fundamental.valuation_orchestrator import (
    log_multi_model_summary,
    run_multi_model_blending,
)


def test_run_multi_model_blending_updates_summary_and_weights():
    valuation_results = {"ggm": {"model": "ggm", "applicable": False}}
    dcf = {"model": "dcf", "weight": 0.0}
    pe = {"model": "pe", "weight": 0.0}
    orchestrator = MagicMock()
    orchestrator.combine.return_value = {
        "models": [{"model": "dcf", "weight": 0.6}, {"model": "pe", "weight": 0.4}],
        "blended_fair_value": 120.0,
    }

    summary, tier = run_multi_model_blending(
        symbol="AAPL",
        valuation_results=valuation_results,
        company_profile=SimpleNamespace(
            industry="Software",
            quarterly_metrics=[],
            free_cash_flow=100.0,
            ttm_metrics={},
            dividends_paid=0.0,
            ebitda=50.0,
            dividend_payout_ratio=0.0,
            net_income=30.0,
        ),
        company_data={"ttm_metrics": {}},
        ratios={"market_cap": 1_000_000},
        financials={},
        dcf_professional=dcf,
        normalized_pe=pe,
        normalized_ev_ebitda=None,
        normalized_ps=None,
        normalized_pb=None,
        select_models_for_company=lambda _profile: None,
        resolve_fallback_weights=lambda *_args, **_kwargs: ({"dcf": 60.0, "pe": 40.0}, "tier_1"),
        multi_model_orchestrator=orchestrator,
        logger=MagicMock(),
    )

    assert tier == "tier_1"
    assert summary["blended_fair_value"] == 120.0
    assert valuation_results["multi_model"]["blended_fair_value"] == 120.0
    assert dcf["weight"] == 0.6
    assert pe["weight"] == 0.4


def test_log_multi_model_summary_returns_metrics():
    logger = MagicMock()
    valuation_results = {
        "multi_model": {
            "blended_fair_value": 130.0,
            "overall_confidence": 0.8,
            "model_agreement_score": 0.7,
            "divergence_flag": False,
            "applicable_models": 4,
            "notes": ["ok"],
        },
        "ggm": {"model": "ggm", "applicable": False},
    }

    metrics = log_multi_model_summary(
        symbol="MSFT",
        valuation_results=valuation_results,
        company_data={"current_price": 100},
        tier_classification="tier_2",
        dcf_professional={"model": "dcf", "fair_value_per_share": 130, "confidence_score": 0.8, "weight": 0.5},
        normalized_pe={"model": "pe", "fair_value_per_share": 125, "confidence_score": 0.7, "weight": 0.5},
        normalized_ev_ebitda=None,
        normalized_ps=None,
        normalized_pb=None,
        log_valuation_snapshot=lambda *_args, **_kwargs: None,
        format_valuation_summary_table=lambda **_kwargs: "table",
        logger=logger,
    )

    assert metrics["blended_fair_value"] == 130.0
    assert metrics["overall_confidence"] == 0.8
    assert metrics["notes"] == ["ok"]
    logger.info.assert_called()
