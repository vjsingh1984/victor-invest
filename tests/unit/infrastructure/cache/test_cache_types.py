from investigator.infrastructure.cache.cache_types import CacheType


def test_expected_cache_types_present() -> None:
    expected = {
        "SEC_RESPONSE": "sec_response",
        "LLM_RESPONSE": "llm_response",
        "TECHNICAL_DATA": "technical_data",
        "SUBMISSION_DATA": "submission_data",
        "COMPANY_FACTS": "company_facts",
        "QUARTERLY_METRICS": "quarterly_metrics",
        "MARKET_CONTEXT": "market_context",
    }

    for name, value in expected.items():
        member = getattr(CacheType, name)
        assert member.value == value


def test_enum_values_unique_and_iterable() -> None:
    members = list(CacheType)
    values = [member.value for member in members]

    assert len(values) == len(set(values))
    assert all(isinstance(member, CacheType) for member in members)


def test_string_representation_matches_enum_contract() -> None:
    member = CacheType.LLM_RESPONSE
    assert str(member) == "CacheType.LLM_RESPONSE"
    assert repr(member) == "<CacheType.LLM_RESPONSE: 'llm_response'>"
