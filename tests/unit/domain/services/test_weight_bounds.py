"""
Unit tests for weight_bounds module.

Tests bounded multiplier application and weight validation.
"""

import pytest
from investigator.domain.services.weight_bounds import (
    BoundedMultiplierApplicator,
    BoundConfig,
    BoundedMultiplierResult,
)


class TestBoundConfig:
    """Tests for BoundConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BoundConfig()
        assert config.cumulative_floor == 0.50
        assert config.cumulative_ceiling == 1.50
        assert config.per_model_minimum == 5.0
        assert config.warning_threshold == 0.70

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BoundConfig(
            cumulative_floor=0.40,
            cumulative_ceiling=1.60,
            per_model_minimum=10.0
        )
        assert config.cumulative_floor == 0.40
        assert config.cumulative_ceiling == 1.60
        assert config.per_model_minimum == 10.0


class TestBoundedMultiplierApplicator:
    """Tests for BoundedMultiplierApplicator class."""

    @pytest.fixture
    def applicator(self):
        """Create default applicator."""
        return BoundedMultiplierApplicator()

    @pytest.fixture
    def base_weights(self):
        """Create standard base weights."""
        return {
            'dcf': 30.0,
            'pe': 25.0,
            'ps': 25.0,
            'ev_ebitda': 20.0,
        }

    def test_no_multipliers(self, applicator, base_weights):
        """Test with no multipliers - weights unchanged."""
        result = applicator.apply_multipliers(
            base_weights=base_weights,
            multiplier_groups={},
            symbol='AAPL'
        )

        assert isinstance(result, BoundedMultiplierResult)
        # Weights should sum to 100
        total = sum(result.adjusted_weights.values())
        assert abs(total - 100.0) < 0.1

    def test_mild_multipliers(self, applicator, base_weights):
        """Test with mild multipliers that don't hit bounds."""
        result = applicator.apply_multipliers(
            base_weights=base_weights,
            multiplier_groups={
                'market_context': {
                    'dcf': 1.1,
                    'pe': 0.95,
                    'ps': 1.0,
                    'ev_ebitda': 1.0,
                }
            },
            symbol='AAPL'
        )

        assert result.bounds_applied is False
        assert len(result.warnings) == 0

    def test_extreme_multipliers_hit_floor(self, applicator, base_weights):
        """Test that extreme multipliers are bounded by floor."""
        result = applicator.apply_multipliers(
            base_weights=base_weights,
            multiplier_groups={
                'quality': {
                    'dcf': 0.1,  # Would reduce to 3.0, below minimum
                    'pe': 0.2,
                    'ps': 0.2,
                    'ev_ebitda': 0.2,
                }
            },
            symbol='AAPL'
        )

        # DCF should be bounded to minimum
        assert result.adjusted_weights['dcf'] >= applicator.config.per_model_minimum
        assert result.bounds_applied is True

    def test_extreme_multipliers_hit_ceiling(self, applicator, base_weights):
        """Test that extreme multipliers are bounded by ceiling."""
        result = applicator.apply_multipliers(
            base_weights=base_weights,
            multiplier_groups={
                'quality': {
                    'dcf': 3.0,  # Would exceed ceiling
                    'pe': 2.5,
                    'ps': 2.0,
                    'ev_ebitda': 2.0,
                }
            },
            symbol='AAPL'
        )

        # Cumulative multiplier should be capped
        for model, weight in result.adjusted_weights.items():
            base = base_weights[model]
            ratio = weight / base if base > 0 else 1.0
            assert ratio <= applicator.config.cumulative_ceiling * 1.1  # Allow small margin

    def test_multiple_multiplier_groups(self, applicator, base_weights):
        """Test with multiple multiplier groups stacked."""
        result = applicator.apply_multipliers(
            base_weights=base_weights,
            multiplier_groups={
                'market_context': {
                    'dcf': 1.1,
                    'pe': 0.9,
                    'ps': 1.0,
                    'ev_ebitda': 1.0,
                },
                'data_quality': {
                    'dcf': 1.0,
                    'pe': 0.8,
                    'ps': 1.2,
                    'ev_ebitda': 0.9,
                }
            },
            symbol='AAPL'
        )

        # Weights should still sum to 100
        total = sum(result.adjusted_weights.values())
        assert abs(total - 100.0) < 0.1

        # Should have audit entries for applied multipliers
        assert len(result.audit_entries) >= 4  # At least one entry per model

    def test_validate_weights_valid(self, applicator):
        """Test weight validation with valid weights."""
        weights = {'dcf': 30.0, 'pe': 30.0, 'ps': 40.0}
        is_valid, issues = applicator.validate_weights(weights)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_weights_negative(self, applicator):
        """Test weight validation with negative weight."""
        weights = {'dcf': -10.0, 'pe': 60.0, 'ps': 50.0}
        is_valid, issues = applicator.validate_weights(weights)
        assert is_valid is False
        # Implementation reports "below minimum" for negative weights
        assert any('below minimum' in issue.lower() for issue in issues)

    def test_validate_weights_zero_total(self, applicator):
        """Test weight validation with zero total."""
        weights = {'dcf': 0.0, 'pe': 0.0, 'ps': 0.0}
        is_valid, issues = applicator.validate_weights(weights)
        assert is_valid is False

    def test_custom_config(self):
        """Test with custom configuration."""
        config = BoundConfig(
            cumulative_floor=0.25,
            cumulative_ceiling=2.0,
            per_model_minimum=2.0
        )
        applicator = BoundedMultiplierApplicator(config=config)

        assert applicator.config.cumulative_floor == 0.25
        assert applicator.config.per_model_minimum == 2.0

    def test_empty_base_weights(self, applicator):
        """Test with empty base weights."""
        result = applicator.apply_multipliers(
            base_weights={},
            multiplier_groups={},
            symbol='AAPL'
        )
        assert result.adjusted_weights == {}

    def test_missing_model_in_multipliers(self, applicator, base_weights):
        """Test when multiplier group doesn't have all models."""
        result = applicator.apply_multipliers(
            base_weights=base_weights,
            multiplier_groups={
                'partial': {
                    'dcf': 1.2,
                    # Missing pe, ps, ev_ebitda - should default to 1.0
                }
            },
            symbol='AAPL'
        )

        # All models should still be present
        assert set(result.adjusted_weights.keys()) == set(base_weights.keys())
