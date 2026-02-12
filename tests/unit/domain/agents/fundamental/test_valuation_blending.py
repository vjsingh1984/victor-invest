"""Unit tests for valuation blending preparation helpers."""

from types import SimpleNamespace

from investigator.domain.agents.fundamental.valuation_blending import (
    apply_weight_lookup,
    collect_models_for_blending,
    filter_models_for_company,
    hydrate_financials_for_blending,
)


def test_collect_models_for_blending_adds_applicable_models_and_messages():
    valuation_results = {
        "ggm": {"model": "ggm"},
        "sector_specific": {"model": "sector_specific", "method": "Bank ROE"},
        "damodaran_dcf": {"model": "damodaran_dcf", "applicable": True},
        "rule_of_40": {"model": "rule_of_40", "applicable": True},
        "saas": {"model": "saas", "applicable": False},
    }
    models, messages = collect_models_for_blending(
        dcf_professional={"model": "dcf"},
        valuation_results=valuation_results,
        normalized_pe={"model": "pe"},
        normalized_ev_ebitda={"model": "ev_ebitda"},
        normalized_ps={"model": "ps"},
        normalized_pb={"model": "pb"},
    )

    assert [m["model"] for m in models] == [
        "dcf",
        "ggm",
        "pe",
        "ev_ebitda",
        "ps",
        "pb",
        "sector_specific",
        "damodaran_dcf",
        "rule_of_40",
    ]
    assert "Added Damodaran DCF to blending" in messages


def test_filter_models_for_company_includes_pb_for_insurance():
    filtered, allowed, added = filter_models_for_company(
        models_for_blending=[{"model": "dcf"}, {"model": "pb"}, {"model": "ps"}],
        allowed_models=["dcf", "ps"],
        industry="Property Insurance",
    )

    assert added is True
    assert "pb" in allowed
    assert [m["model"] for m in filtered] == ["dcf", "pb", "ps"]


def test_hydrate_financials_for_blending_populates_required_fields():
    financials = {"revenues": 2_000_000, "operating_income": 200_000}
    company_data = {"ttm_metrics": {"net_income": 150_000}}
    company_profile = SimpleNamespace(
        quarterly_metrics=[],
        free_cash_flow=100_000,
        dividends_paid=50_000,
        ebitda=None,
        dividend_payout_ratio=None,
        net_income=None,
    )
    ratios = {"market_cap": 10_000_000, "current_price": 100.0, "payout_ratio": 22.0}

    summary = hydrate_financials_for_blending(
        financials=financials,
        company_data=company_data,
        company_profile=company_profile,
        ratios=ratios,
    )

    assert financials["market_cap"] == 10_000_000
    assert financials["revenue"] == 2_000_000
    assert financials["fcf_quarters_count"] == 4
    assert financials["ebitda"] == 200_000
    assert financials["payout_ratio"] == 22.0
    assert financials["net_income"] == 150_000
    assert summary["free_cash_flow"] == 100_000


def test_apply_weight_lookup_backfills_model_weights():
    dcf = {"model": "dcf"}
    pe = {"model": "pe"}
    pb = {"model": "pb"}
    valuation_results = {"ggm": {"model": "ggm"}}
    summary = {"models": [{"model": "dcf", "weight": 0.4}, {"model": "pe", "weight": 0.2}, {"model": "ggm", "weight": 0.1}]}

    apply_weight_lookup(
        multi_model_summary=summary,
        dcf_professional=dcf,
        valuation_results=valuation_results,
        normalized_pe=pe,
        normalized_ev_ebitda=None,
        normalized_ps=None,
        normalized_pb=pb,
    )

    assert dcf["weight"] == 0.4
    assert pe["weight"] == 0.2
    assert valuation_results["ggm"]["weight"] == 0.1
    assert "weight" not in pb
