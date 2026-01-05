"""
Unit tests for fallback_chain module.

Tests fallback model chains and graceful degradation.
"""

import pytest

from investigator.domain.services.valuation.fallback_chain import (
    FallbackChain,
    FallbackChainConfig,
    FallbackReason,
    FallbackResult,
    get_fallback_chain,
)


class TestFallbackReason:
    """Tests for FallbackReason enum."""

    def test_all_reasons_exist(self):
        """Test all fallback reasons are defined."""
        assert FallbackReason.DATA_MISSING
        assert FallbackReason.CALCULATION_ERROR
        assert FallbackReason.INVALID_OUTPUT
        assert FallbackReason.MODEL_NOT_APPLICABLE
        assert FallbackReason.TIMEOUT


class TestFallbackResult:
    """Tests for FallbackResult dataclass."""

    def test_creation(self):
        """Test creating a fallback result."""
        result = FallbackResult(
            original_model="ggm",
            fallback_model="dcf",
            reason=FallbackReason.DATA_MISSING,
            confidence_penalty=0.90,
            fallback_level=1,
            notes=["Used fallback due to missing dividend data"],
        )
        assert result.original_model == "ggm"
        assert result.fallback_model == "dcf"
        assert result.confidence_penalty == 0.90
        assert result.fallback_level == 1

    def test_summary_with_fallback(self):
        """Test summary when fallback was used."""
        result = FallbackResult(
            original_model="ggm",
            fallback_model="dcf",
            reason=FallbackReason.DATA_MISSING,
            confidence_penalty=0.90,
            fallback_level=1,
        )
        summary = result.summary()
        assert "ggm → dcf" in summary
        assert "level 1" in summary
        assert "90%" in summary

    def test_summary_no_fallback(self):
        """Test summary when no fallback available."""
        result = FallbackResult(
            original_model="ggm",
            fallback_model=None,
            reason=FallbackReason.CALCULATION_ERROR,
            confidence_penalty=0.0,
            fallback_level=3,
        )
        summary = result.summary()
        assert "failed" in summary
        assert "no fallback" in summary


class TestFallbackChainConfig:
    """Tests for FallbackChainConfig dataclass."""

    def test_creation(self):
        """Test creating a chain config."""
        config = FallbackChainConfig(model="dcf", fallbacks=["pe", "ps"], penalties=[0.90, 0.80])
        assert config.model == "dcf"
        assert len(config.fallbacks) == 2

    def test_get_fallback_first_level(self):
        """Test getting first fallback."""
        config = FallbackChainConfig(model="dcf", fallbacks=["pe", "ps", "ev_ebitda"], penalties=[0.90, 0.80, 0.70])
        result = config.get_fallback(0)
        assert result == ("pe", 0.90)

    def test_get_fallback_second_level(self):
        """Test getting second fallback."""
        config = FallbackChainConfig(model="dcf", fallbacks=["pe", "ps"], penalties=[0.90, 0.80])
        result = config.get_fallback(1)
        assert result == ("ps", 0.80)

    def test_get_fallback_out_of_range(self):
        """Test getting fallback beyond available."""
        config = FallbackChainConfig(model="dcf", fallbacks=["pe"], penalties=[0.90])
        result = config.get_fallback(5)
        assert result is None


class TestFallbackChain:
    """Tests for FallbackChain class."""

    @pytest.fixture
    def chain(self):
        """Create default fallback chain."""
        return FallbackChain()

    def test_default_chains_exist(self, chain):
        """Test default chains are defined."""
        assert "dcf" in chain.chains
        assert "pe" in chain.chains
        assert "ps" in chain.chains
        assert "ggm" in chain.chains

    def test_get_fallback_dcf(self, chain):
        """Test getting fallback for DCF."""
        result = chain.get_fallback("dcf")
        assert result is not None
        fallback, penalty = result
        assert fallback == "pe"  # First fallback for DCF
        assert penalty == 0.90

    def test_get_fallback_ggm(self, chain):
        """Test getting fallback for GGM."""
        result = chain.get_fallback("ggm")
        assert result is not None
        fallback, penalty = result
        assert fallback == "dcf"  # First fallback for GGM

    def test_get_fallback_skip_failed(self, chain):
        """Test skipping already-failed models."""
        # DCF chain is: dcf -> pe -> ps -> ev_ebitda
        result = chain.get_fallback("dcf", failed_models=["pe"])
        assert result is not None
        fallback, penalty = result
        assert fallback == "ps"  # Skipped pe, got ps
        assert penalty == 0.80  # Second level penalty

    def test_get_fallback_all_failed(self, chain):
        """Test when all fallbacks have failed."""
        result = chain.get_fallback("dcf", failed_models=["pe", "ps", "ev_ebitda"])
        assert result is None

    def test_get_fallback_unknown_model(self, chain):
        """Test fallback for unknown model."""
        result = chain.get_fallback("unknown_model")
        assert result is None

    def test_get_chain_dcf(self, chain):
        """Test getting full chain for DCF."""
        full_chain = chain.get_chain("dcf")
        assert full_chain[0] == "dcf"  # Original model first
        assert "pe" in full_chain
        assert "ps" in full_chain

    def test_get_chain_unknown(self, chain):
        """Test getting chain for unknown model."""
        full_chain = chain.get_chain("unknown")
        assert full_chain == ["unknown"]  # Just the model itself

    def test_get_penalty_first_fallback(self, chain):
        """Test getting penalty for first fallback."""
        penalty = chain.get_penalty("dcf", "pe")
        assert penalty == 0.90

    def test_get_penalty_second_fallback(self, chain):
        """Test getting penalty for second fallback."""
        penalty = chain.get_penalty("dcf", "ps")
        assert penalty == 0.80

    def test_get_penalty_unknown_chain(self, chain):
        """Test getting penalty for unknown model."""
        penalty = chain.get_penalty("unknown", "pe")
        assert penalty == 0.80  # Default penalty

    def test_get_penalty_unknown_fallback(self, chain):
        """Test getting penalty for fallback not in chain."""
        penalty = chain.get_penalty("dcf", "ggm")  # GGM not in DCF chain
        assert penalty == 0.70  # Conservative penalty

    def test_custom_chains(self):
        """Test custom fallback chains."""
        custom = {"custom_model": ["fallback1", "fallback2"]}
        chain = FallbackChain(chains=custom)
        assert "custom_model" in chain.chains
        result = chain.get_fallback("custom_model")
        assert result[0] == "fallback1"

    def test_custom_penalties(self):
        """Test custom penalties."""
        custom_penalties = [0.95, 0.85, 0.75, 0.65]
        chain = FallbackChain(penalties=custom_penalties)
        result = chain.get_fallback("dcf")
        assert result[1] == 0.95  # Custom first level penalty


class TestFallbackChainExecution:
    """Tests for execute_with_fallbacks method."""

    @pytest.fixture
    def chain(self):
        return FallbackChain()

    def test_execute_success_primary(self, chain):
        """Test successful execution with primary model."""

        def executor(**kwargs):
            return {"fair_value": 100.0}

        result, fallback_result = chain.execute_with_fallbacks(model_type="dcf", executor_func=executor)
        assert result is not None
        assert result["fair_value"] == 100.0
        assert fallback_result.fallback_model is None
        assert fallback_result.confidence_penalty == 1.0
        assert fallback_result.fallback_level == 0

    def test_execute_fallback_used(self, chain):
        """Test execution falls back on primary failure."""
        call_count = [0]

        def executor(model_type, **kwargs):
            call_count[0] += 1
            if model_type == "dcf":
                raise ValueError("DCF failed - no FCF data")
            return {"fair_value": 80.0, "model": model_type}

        result, fallback_result = chain.execute_with_fallbacks(model_type="dcf", executor_func=executor)
        assert result is not None
        assert result["model"] == "pe"  # First fallback
        assert fallback_result.fallback_model == "pe"
        assert fallback_result.fallback_level == 1
        assert call_count[0] == 2  # DCF failed, PE succeeded

    def test_execute_multiple_fallbacks(self, chain):
        """Test multiple fallback attempts."""
        attempts = []

        def executor(model_type, **kwargs):
            attempts.append(model_type)
            if model_type in ["dcf", "pe"]:
                raise ValueError(f"{model_type} failed")
            return {"fair_value": 60.0, "model": model_type}

        result, fallback_result = chain.execute_with_fallbacks(model_type="dcf", executor_func=executor)
        assert result is not None
        assert result["model"] == "ps"  # Second fallback
        assert len(attempts) == 3  # dcf, pe, ps
        assert fallback_result.fallback_level == 2

    def test_execute_all_fail(self, chain):
        """Test when all models fail."""

        def executor(**kwargs):
            raise ValueError("All models fail")

        result, fallback_result = chain.execute_with_fallbacks(
            model_type="dcf", executor_func=executor, max_fallbacks=2
        )
        assert result is None
        assert fallback_result.fallback_model is None
        assert fallback_result.confidence_penalty == 0.0
        assert "All models failed" in fallback_result.notes[0]

    def test_execute_max_fallbacks_limit(self, chain):
        """Test max fallbacks limit is respected."""
        attempts = []

        def executor(model_type, **kwargs):
            attempts.append(model_type)
            raise ValueError("Always fail")

        result, fallback_result = chain.execute_with_fallbacks(
            model_type="dcf", executor_func=executor, max_fallbacks=1  # Only try 1 fallback
        )
        assert result is None
        assert len(attempts) == 2  # Primary + 1 fallback


class TestFallbackChainApplicability:
    """Tests for get_applicable_models method."""

    @pytest.fixture
    def chain(self):
        return FallbackChain()

    def test_get_applicable_all_available(self, chain):
        """Test with all data available."""
        available_data = {
            "fcf": True,
            "discount_rate": True,
            "eps": True,
            "pe_ratio": True,
            "revenue": True,
            "ps_ratio": True,
        }
        applicable = chain.get_applicable_models(available_data)
        assert len(applicable) >= 3  # dcf, pe, ps at minimum

    def test_get_applicable_partial_data(self, chain):
        """Test with partial data available."""
        available_data = {
            "revenue": True,
            "ps_ratio": True,
            # Missing FCF, EPS, etc.
        }
        applicable = chain.get_applicable_models(available_data)
        # PS should be applicable
        model_names = [m[0] for m in applicable]
        assert "ps" in model_names

    def test_get_applicable_with_fallbacks(self, chain):
        """Test that fallbacks are considered."""
        available_data = {
            # DCF data missing, but PE data available
            "eps": True,
            "pe_ratio": True,
        }
        applicable = chain.get_applicable_models(available_data, preferred_order=["dcf", "pe"])
        # Should include PE as a fallback option
        model_names = [m[0] for m in applicable]
        assert "pe" in model_names

    def test_get_applicable_preferred_order(self, chain):
        """Test preferred order is respected."""
        available_data = {
            "revenue": True,
            "ps_ratio": True,
            "eps": True,
            "pe_ratio": True,
        }
        applicable = chain.get_applicable_models(available_data, preferred_order=["ps", "pe"])  # PS first
        if len(applicable) >= 2:
            assert applicable[0][0] == "ps"  # PS should be first


class TestFallbackChainEdgeCases:
    """Edge case tests for FallbackChain."""

    def test_empty_failed_models_list(self):
        """Test with empty failed models list."""
        chain = FallbackChain()
        result = chain.get_fallback("dcf", failed_models=[])
        assert result is not None

    def test_none_failed_models(self):
        """Test with None failed models."""
        chain = FallbackChain()
        result = chain.get_fallback("dcf", failed_models=None)
        assert result is not None

    def test_saas_model_chain(self):
        """Test SaaS model fallback chain."""
        chain = FallbackChain()
        full_chain = chain.get_chain("saas")
        assert "ps" in full_chain
        assert "rule_of_40" in full_chain

    def test_rule_of_40_chain(self):
        """Test Rule of 40 fallback chain."""
        chain = FallbackChain()
        full_chain = chain.get_chain("rule_of_40")
        assert "ps" in full_chain


class TestSingletonFallbackChain:
    """Tests for singleton instance."""

    def test_singleton(self):
        """Test get_fallback_chain returns singleton."""
        chain1 = get_fallback_chain()
        chain2 = get_fallback_chain()
        assert chain1 is chain2
