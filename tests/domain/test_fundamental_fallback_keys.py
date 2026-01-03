from investigator.domain.agents.fundamental import FALLBACK_CANONICAL_KEYS


def test_fundamental_fallback_keys_include_debt_metrics():
    """Fallback extraction must request debt metrics for leverage diagnostics."""
    for required_key in ("long_term_debt", "short_term_debt", "total_debt"):
        assert required_key in FALLBACK_CANONICAL_KEYS


def test_fundamental_fallback_keys_include_cash_flow_components():
    """Free cash flow should be derivable even when canonical tag is absent."""
    for required_key in ("operating_cash_flow", "capital_expenditures", "free_cash_flow"):
        assert required_key in FALLBACK_CANONICAL_KEYS
