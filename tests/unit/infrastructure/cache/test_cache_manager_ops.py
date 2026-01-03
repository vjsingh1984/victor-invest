from investigator.infrastructure.cache.cache_types import CacheType


def test_llm_response_cycle(cache_manager, sample_llm_response):
    key = {
        "symbol": "TEST",
        "form_type": "N/A",
        "period": "N/A",
        "llm_type": "synthesis_test",
    }

    assert cache_manager.get(CacheType.LLM_RESPONSE, key) is None
    assert cache_manager.exists(CacheType.LLM_RESPONSE, key) is False

    assert cache_manager.set(CacheType.LLM_RESPONSE, key, sample_llm_response) is True

    result = cache_manager.get(CacheType.LLM_RESPONSE, key)
    assert result is not None
    assert result["response"]["symbol"] == "TEST"

    assert cache_manager.exists(CacheType.LLM_RESPONSE, key) is True
    assert cache_manager.delete(CacheType.LLM_RESPONSE, key) is True
    assert cache_manager.get(CacheType.LLM_RESPONSE, key) is None


def test_technical_data_cycle(cache_manager, sample_technical_data):
    key = ("TEST", "technical_data")

    assert cache_manager.set(CacheType.TECHNICAL_DATA, key, sample_technical_data)
    result = cache_manager.get(CacheType.TECHNICAL_DATA, key)
    assert result["symbol"] == "TEST"
    assert "dataframe" in result
    assert cache_manager.delete(CacheType.TECHNICAL_DATA, key)


def test_company_facts_cycle(cache_manager, sample_company_facts):
    key = {"symbol": "TEST", "cik": "0001234567"}

    assert cache_manager.set(CacheType.COMPANY_FACTS, key, sample_company_facts)
    result = cache_manager.get(CacheType.COMPANY_FACTS, key)
    assert "facts" in result  # handler unwraps to raw SEC structure
    assert cache_manager.delete(CacheType.COMPANY_FACTS, key)


def test_submission_and_quarterly_metrics(cache_manager, sample_submission_data, sample_quarterly_metrics):
    submission_key = ("TEST", "recent_10")
    metrics_key = {"symbol": "TEST", "fiscal_year": "2024", "fiscal_period": "Q1"}

    assert cache_manager.set(CacheType.SUBMISSION_DATA, submission_key, sample_submission_data)
    assert cache_manager.get(CacheType.SUBMISSION_DATA, submission_key)["symbol"] == "TEST"

    assert cache_manager.set(CacheType.QUARTERLY_METRICS, metrics_key, sample_quarterly_metrics)
    assert cache_manager.get(CacheType.QUARTERLY_METRICS, metrics_key)["symbol"] == "TEST"

    cache_manager.delete(CacheType.SUBMISSION_DATA, submission_key)
    cache_manager.delete(CacheType.QUARTERLY_METRICS, metrics_key)


def test_delete_by_pattern_and_clear(cache_manager, sample_llm_response):
    for idx in range(3):
        key = {
            "symbol": "PATTERN",
            "form_type": "N/A",
            "period": "N/A",
            "llm_type": f"pattern_{idx}",
        }
        cache_manager.set(CacheType.LLM_RESPONSE, key, sample_llm_response)

    deleted = cache_manager.delete_by_pattern(CacheType.LLM_RESPONSE, "*PATTERN*")
    assert deleted >= 0

    assert cache_manager.clear_cache_type(CacheType.LLM_RESPONSE)
    assert cache_manager.get(CacheType.LLM_RESPONSE, {"symbol": "PATTERN", "llm_type": "pattern_0"}) is None


def test_performance_stats_and_recent_operations(cache_manager, sample_llm_response):
    key = {
        "symbol": "STATS",
        "form_type": "N/A",
        "period": "N/A",
        "llm_type": "stats",
    }

    cache_manager.get(CacheType.LLM_RESPONSE, key)
    cache_manager.set(CacheType.LLM_RESPONSE, key, sample_llm_response)
    cache_manager.get(CacheType.LLM_RESPONSE, key)

    stats = cache_manager.get_performance_stats()
    assert "llm_response" in stats
    assert stats["llm_response"]["operations"]["writes"] >= 1

    recent = cache_manager.get_recent_operations(CacheType.LLM_RESPONSE, limit=5)
    assert "llm_response" in recent
    assert recent["llm_response"]


def test_delete_by_symbol(cache_manager, sample_llm_response):
    key = {
        "symbol": "SYMTEST",
        "form_type": "N/A",
        "period": "N/A",
        "llm_type": "symbol",
    }

    cache_manager.set(CacheType.LLM_RESPONSE, key, sample_llm_response)
    deletion_result = cache_manager.delete_by_symbol("SYMTEST")
    assert deletion_result
    assert cache_manager.get(CacheType.LLM_RESPONSE, key) is None
