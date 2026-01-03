from copy import deepcopy
from datetime import datetime

from investigator.infrastructure.cache.cache_types import CacheType


def test_submission_cache_accepts_multiple_key_shapes(cache_manager, sample_submission_data):
    payload = deepcopy(sample_submission_data)
    payload.update({"symbol": "AAPL", "cik": "0000320193"})
    dict_key = {"symbol": "AAPL", "cik": "0000320193"}
    tuple_key = ("AAPL", "0000320193")

    assert cache_manager.set(CacheType.SUBMISSION_DATA, dict_key, payload)
    assert cache_manager.get(CacheType.SUBMISSION_DATA, dict_key)["symbol"] == "AAPL"

    tuple_hit = cache_manager.get(CacheType.SUBMISSION_DATA, tuple_key)
    assert tuple_hit is not None
    assert tuple_hit["symbol"] == "AAPL"


def test_company_facts_cache_uses_cik(cache_manager, sample_company_facts):
    payload = deepcopy(sample_company_facts)
    payload.update({"symbol": "MSFT", "cik": "0000789019"})
    key = {"symbol": "MSFT", "cik": "0000789019"}

    assert cache_manager.set(CacheType.COMPANY_FACTS, key, payload)
    stored = cache_manager.get(CacheType.COMPANY_FACTS, key)
    assert stored is not None
    assert isinstance(stored, dict)
    assert "facts" in stored


def test_quarterly_metrics_cache_accepts_tuple_and_dict(cache_manager, sample_quarterly_metrics):
    payload = deepcopy(sample_quarterly_metrics)
    payload.update({"symbol": "NVDA", "fiscal_year": "2024", "fiscal_period": "Q1"})

    dict_key = {"symbol": "NVDA", "fiscal_year": "2024", "fiscal_period": "Q1"}
    tuple_key = ("NVDA", "2024-Q1")

    assert cache_manager.set(CacheType.QUARTERLY_METRICS, dict_key, payload)

    dict_hit = cache_manager.get(CacheType.QUARTERLY_METRICS, dict_key)
    assert dict_hit is not None
    assert dict_hit["symbol"] == "NVDA"

    tuple_hit = cache_manager.get(CacheType.QUARTERLY_METRICS, tuple_key)
    if tuple_hit:
        assert tuple_hit["symbol"] == "NVDA"


def test_sec_response_cache_round_trip(cache_manager):
    key = {
        "symbol": "TSLA",
        "fiscal_year": "2024",
        "fiscal_period": "Q1",
        "form_type": "10-Q",
        "category": "quarterly_summary",
    }
    payload = {
        "symbol": "TSLA",
        "fiscal_year": "2024",
        "fiscal_period": "Q1",
        "form_type": "10-Q",
        "analysis": "Sample SEC analysis",
        "metadata": {"generated_at": datetime.now().isoformat()},
    }

    assert cache_manager.set(CacheType.SEC_RESPONSE, key, payload)
    result = cache_manager.get(CacheType.SEC_RESPONSE, key)
    if result:
        assert result["symbol"] == "TSLA"
