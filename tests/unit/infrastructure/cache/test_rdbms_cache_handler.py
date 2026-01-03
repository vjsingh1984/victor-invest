from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from investigator.infrastructure.cache.cache_types import CacheType
from investigator.infrastructure.cache.rdbms_cache_handler import RdbmsCacheStorageHandler


class FakeLLMDao:
    def __init__(self):
        self.saved_payload = None
        self.deleted = False

    def save_llm_response(self, **kwargs):
        self.saved_payload = {
            "symbol": kwargs["symbol"],
            "form_type": kwargs["form_type"],
            "period": kwargs["period"],
            "response": kwargs["response"],
            "metadata": kwargs["metadata"],
        }
        return True

    def get_llm_response(self, symbol, form_type, period, llm_type):
        if not self.saved_payload:
            return None
        return {
            "symbol": symbol,
            "form_type": form_type,
            "period": period,
            "llm_type": llm_type,
            "response": self.saved_payload["response"],
            "metadata": self.saved_payload["metadata"],
        }

    def delete_llm_responses(self, **kwargs):
        if self.saved_payload:
            self.deleted = True
            self.saved_payload = None
            return 1
        return 0

    def delete_llm_responses_by_pattern(self, **kwargs):
        if self.saved_payload:
            self.saved_payload = None
            return 1
        return 0


class FakeSecDao:
    def __init__(self):
        self.saved = None

    def save_response(self, **kwargs):
        self.saved = kwargs["response_data"]
        return True

    def get_response(self, symbol, form_type, fiscal_year, fiscal_period, category):
        if self.saved:
            return {"response_data": self.saved}
        return None

    def get_latest_response(self, symbol, form_type, category):
        if self.saved:
            return {"response_data": self.saved}
        return None

    def delete_responses_by_symbol(self, symbol):
        if self.saved:
            self.saved = None
            return 1
        return 0


@pytest.fixture
def patch_database_manager(monkeypatch):
    import investigator.infrastructure.cache.rdbms_cache_handler as module

    monkeypatch.setattr("utils.db.DatabaseManager", lambda *args, **kwargs: Mock())
    monkeypatch.setattr("utils.db.get_quarterly_metrics_dao", lambda: None)
    monkeypatch.setattr("utils.db.get_sec_companyfacts_dao", lambda: None)
    monkeypatch.setattr("utils.db.get_sec_responses_dao", lambda: None)
    monkeypatch.setattr("utils.db.get_sec_submissions_dao", lambda: None)
    monkeypatch.setattr("utils.db.get_llm_responses_dao", lambda: None)
    return module


def test_llm_response_round_trip(patch_database_manager, monkeypatch):
    module = patch_database_manager
    dao = FakeLLMDao()
    monkeypatch.setattr("utils.db.get_llm_responses_dao", lambda: dao)

    handler = module.RdbmsCacheStorageHandler(CacheType.LLM_RESPONSE, priority=5)

    key = {
        "symbol": "AAPL",
        "form_type": "10-Q",
        "period": "2024-Q1",
        "llm_type": "synthesis",
    }
    payload = {
        "prompt": "Analyze AAPL",
        "response": {"score": 9.5},
        "metadata": {"analysis_type": "fundamental"},
    }

    assert handler.set(key, payload) is True

    fetched = handler.get(key)
    assert fetched is not None
    assert fetched["response"]["score"] == 9.5

    assert handler.delete(key) is True
    assert dao.deleted is True


def test_sec_response_save_and_fetch(patch_database_manager, monkeypatch):
    module = patch_database_manager
    dao = FakeSecDao()
    monkeypatch.setattr("utils.db.get_sec_responses_dao", lambda: dao)

    handler = module.RdbmsCacheStorageHandler(CacheType.SEC_RESPONSE, priority=5)

    key = {
        "symbol": "MSFT",
        "form_type": "10-K",
        "period": "2024-FY",
        "category": "annual_summary",
    }
    payload = {
        "symbol": "MSFT",
        "filing_date": "2024-02-15",
        "period_end": "2023-12-31",
        "analysis": "Sample SEC summary",
    }

    assert handler.set(key, payload) is True
    result = handler.get(key)
    assert result is not None
    assert result["symbol"] == "MSFT"
    assert handler.delete(key) is True
