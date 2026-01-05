import pytest

from investigator.infrastructure.cache.cache_key_builder import (
    CacheKeyBuilder,
    build_cache_key,
)
from investigator.infrastructure.cache.cache_types import CacheType


def test_build_llm_response_key() -> None:
    key = CacheKeyBuilder.build_key(
        CacheType.LLM_RESPONSE,
        symbol="aapl",
        fiscal_period="2025-Q2",
        analysis_type="fundamental_analysis",
        context_hash="ctx123",
    )

    assert key["symbol"] == "AAPL"
    assert key["fiscal_period"] == "2025-Q2"
    assert key["analysis_type"] == "fundamental_analysis"
    assert key["context_hash"] == "ctx123"


def test_build_company_facts_key_with_adsh_and_cik() -> None:
    key = CacheKeyBuilder.build_key(
        CacheType.COMPANY_FACTS,
        symbol="MSFT",
        fiscal_year=2024,
        fiscal_period="Q3",
        adsh="0000320193-25-000057",
        cik="0000789019",
    )

    # Implementation normalizes fiscal_period to include year (2024-Q3)
    assert key == {
        "symbol": "MSFT",
        "fiscal_year": 2024,
        "fiscal_period": "2024-Q3",
        "adsh": "0000320193-25-000057",
        "cik": "0000789019",
    }


def test_validate_required_fields() -> None:
    # Missing analysis_type and fiscal_period should raise ValueError
    with pytest.raises(ValueError):
        CacheKeyBuilder.validate_key(CacheType.LLM_RESPONSE, {"symbol": "AAPL"})

    # TD2 fix: fiscal_period is now required for LLM_RESPONSE
    valid_key = {
        "symbol": "AAPL",
        "analysis_type": "fundamental",
        "fiscal_period": "2024-Q3",
    }
    assert CacheKeyBuilder.validate_key(CacheType.LLM_RESPONSE, valid_key)


def test_format_for_filename_truncates_adsh() -> None:
    key = {
        "symbol": "AAPL",
        "fiscal_period": "2025-Q2",
        "analysis_type": "fundamental",
        "adsh": "0000320193-25-000057",
    }
    filename = CacheKeyBuilder.format_for_filename(key)
    assert filename == "AAPL_2025-Q2_fundamental_adsh5-000057"


def test_convenience_wrapper_returns_expected_dict() -> None:
    key = build_cache_key(
        CacheType.TECHNICAL_DATA,
        symbol="TSLA",
        timeframe="short",
    )
    assert key == {"symbol": "TSLA", "timeframe": "short"}
