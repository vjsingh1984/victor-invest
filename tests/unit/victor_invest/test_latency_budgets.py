import pytest

from victor_invest.latency_budgets import (
    LATENCY_BUDGET_PROFILES,
    evaluate_latency,
    get_latency_budget,
)


def test_default_latency_budgets_cover_primary_modes():
    production = LATENCY_BUDGET_PROFILES["production"]
    assert set(production.keys()) == {"quick", "standard", "comprehensive"}
    assert production["quick"] < production["standard"]
    assert production["standard"] < production["comprehensive"]


def test_get_latency_budget_validates_mode():
    with pytest.raises(ValueError):
        get_latency_budget("unknown")


def test_get_latency_budget_validates_profile():
    with pytest.raises(ValueError):
        get_latency_budget("quick", profile="unknown")


def test_evaluate_latency_pass_and_fail():
    passed = evaluate_latency("quick", 5.0, profile="production")
    failed = evaluate_latency("quick", 30.0, profile="production")

    assert passed.passed is True
    assert failed.passed is False
    assert failed.delta_seconds > 0
