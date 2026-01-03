"""Test configuration helpers and fixtures."""

import sys
from pathlib import Path
import json
from typing import Dict, Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (SRC, ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_llm_response() -> Dict[str, Any]:
    return {
        "prompt": "Analyze TEST",
        "response": {
            "symbol": "TEST",
            "overall_score": 8.5,
            "analysis": "Sample analysis",
        },
        "metadata": {"symbol": "TEST"},
        "model_info": {"model": "mock"},
    }


@pytest.fixture
def sample_technical_data() -> Dict[str, Any]:
    return {
        "symbol": "TEST",
        "analysis_date": "2025-01-01",
        "dataframe": {"close": [100, 101, 102]},
        "metadata": {"symbol": "TEST"},
    }


@pytest.fixture
def sample_company_facts() -> Dict[str, Any]:
    return {
        "symbol": "TEST",
        "cik": "0001234567",
        "companyfacts": {"facts": {"us-gaap": {}}},
    }


@pytest.fixture
def sample_submission_data() -> Dict[str, Any]:
    return {
        "symbol": "TEST",
        "cik": "0001234567",
        "submissions": {"recent": {"form": ["10-Q"]}},
    }


@pytest.fixture
def sample_quarterly_metrics() -> Dict[str, Any]:
    return {
        "symbol": "TEST",
        "fiscal_year": "2024",
        "fiscal_period": "Q1",
        "metrics": {"revenue": 1000000000},
    }


@pytest.fixture
def cache_manager(cache_root):
    from investigator.infrastructure.cache.cache_manager import CacheManager, CacheType
    from investigator.infrastructure.cache.file_cache_handler import FileCacheStorageHandler

    manager = CacheManager()
    # Replace handlers with file-based test handlers rooted in tmp dir
    handler_map = {}
    for cache_type in CacheType:
        handler = FileCacheStorageHandler(cache_type=cache_type, base_path=cache_root / cache_type.value, priority=10)
        handler_map[cache_type] = [handler]
    manager.handlers = handler_map
    return manager
