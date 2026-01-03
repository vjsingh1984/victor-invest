import pytest

from investigator.infrastructure.cache.cache_base import CacheStorageHandler
from investigator.infrastructure.cache.cache_types import CacheType


def test_cache_storage_handler_is_abstract() -> None:
    with pytest.raises(TypeError):
        CacheStorageHandler(CacheType.LLM_RESPONSE, priority=1)


class DummyHandler(CacheStorageHandler):
    def __init__(self, cache_type: CacheType):
        super().__init__(cache_type, priority=5)
        self._store = {}

    def get(self, key):
        return self._store.get(str(key))

    def set(self, key, value):
        self._store[str(key)] = value
        return True

    def exists(self, key):
        return str(key) in self._store

    def delete(self, key):
        return self._store.pop(str(key), None) is not None

    def delete_by_pattern(self, pattern: str):
        needle = pattern.replace("*", "")
        deleted = 0
        for entry in list(self._store):
            if needle in entry:
                self._store.pop(entry, None)
                deleted += 1
        return deleted

    def clear_all(self):
        self._store.clear()
        return True


def test_dummy_handler_basic_crud_cycle() -> None:
    handler = DummyHandler(CacheType.COMPANY_FACTS)

    assert handler.exists("key") is False
    assert handler.get("key") is None

    handler.set("key", {"value": 1})
    assert handler.exists("key") is True
    assert handler.get("key") == {"value": 1}

    handler.delete("key")
    assert handler.exists("key") is False

    handler.set("foo", 1)
    handler.set("bar", 2)
    assert handler.delete_by_pattern("*o*") == 1
    assert handler.clear_all() is True
    assert handler.exists("bar") is False


@pytest.mark.parametrize(
    ("cache_type", "raw_key", "expected"),
    [
        (
            CacheType.SEC_RESPONSE,
            ("AAPL", "facts", "2024-Q4", "10-K"),
            {"symbol": "AAPL", "category": "facts", "period": "2024-Q4", "form_type": "10-K"},
        ),
        (
            CacheType.LLM_RESPONSE,
            ("MSFT", "synth", "P1", "10-Q", "2024", "Q1"),
            {
                "symbol": "MSFT",
                "llm_type": "synth",
                "period": "P1",
                "form_type": "10-Q",
                "fiscal_year": "2024",
                "fiscal_period": "Q1",
            },
        ),
        (
            CacheType.COMPANY_FACTS,
            ("NVDA", "000032"),
            {"symbol": "NVDA", "cik": "000032"},
        ),
        (
            CacheType.QUARTERLY_METRICS,
            ("TSLA", "2024-Q3"),
            {"symbol": "TSLA", "period": "2024-Q3"},
        ),
    ],
)
def test_normalize_key_handles_known_tuple_formats(cache_type, raw_key, expected) -> None:
    handler = DummyHandler(cache_type)
    normalized = handler._normalize_key(raw_key)  # pylint: disable=protected-access
    assert normalized == expected


def test_normalize_key_accepts_dict_passthrough() -> None:
    handler = DummyHandler(CacheType.LLM_RESPONSE)
    key = {"symbol": "AAPL", "llm_type": "synthesis"}
    normalized = handler._normalize_key(key)  # pylint: disable=protected-access
    assert normalized == {"symbol": "AAPL", "llm_type": "synthesis"}
