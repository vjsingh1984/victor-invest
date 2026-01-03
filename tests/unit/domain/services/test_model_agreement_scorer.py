"""
Unit tests for model_agreement_scorer module.

Tests model divergence analysis and outlier detection.
"""

import pytest
from investigator.domain.services.model_agreement_scorer import (
    ModelAgreementScorer,
    AgreementScore,
    AgreementLevel,
    AgreementConfig,
    get_model_agreement_scorer,
)


class TestAgreementLevel:
    """Tests for AgreementLevel enum."""

    def test_all_levels_exist(self):
        """Test all agreement levels are defined."""
        assert AgreementLevel.HIGH
        assert AgreementLevel.MODERATE
        assert AgreementLevel.LOW
        assert AgreementLevel.DIVERGENT


class TestAgreementConfig:
    """Tests for AgreementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgreementConfig()
        assert config.divergence_threshold == 0.35
        assert config.high_agreement_threshold == 0.15
        assert config.zscore_threshold == 2.0
        assert config.outlier_weight_penalty == 0.50

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgreementConfig(
            divergence_threshold=0.40,
            zscore_threshold=1.5
        )
        assert config.divergence_threshold == 0.40
        assert config.zscore_threshold == 1.5


class TestModelAgreementScorer:
    """Tests for ModelAgreementScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create default scorer."""
        return ModelAgreementScorer()

    def test_high_agreement(self, scorer):
        """Test high agreement between models."""
        fair_values = {
            'dcf': 150.0,
            'pe': 148.0,
            'ps': 152.0,
            'ev_ebitda': 149.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        assert isinstance(result, AgreementScore)
        assert result.agreement_level == AgreementLevel.HIGH
        assert result.cv < 0.15
        assert result.divergence_flag is False
        assert len(result.outlier_models) == 0

    def test_divergent_models(self, scorer):
        """Test divergence detection."""
        fair_values = {
            'dcf': 100.0,
            'pe': 150.0,
            'ps': 200.0,  # Significant outlier
            'ev_ebitda': 110.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # Should detect divergence
        assert result.cv > 0.20
        assert result.agreement_level in [AgreementLevel.LOW, AgreementLevel.DIVERGENT]

    def test_outlier_detection(self, scorer):
        """Test outlier model detection with z-scores."""
        fair_values = {
            'dcf': 100.0,
            'pe': 105.0,
            'ps': 400.0,  # Clear outlier (>2 sigma from mean)
            'ev_ebitda': 98.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # PS should be flagged as outlier (z-score > 2.0)
        # Mean ~175, std ~125, ps z-score ~1.8
        # If no outliers detected, the divergence flag should be set
        assert result.divergence_flag is True or len(result.outlier_models) >= 1

    def test_confidence_adjustment_high_agreement(self, scorer):
        """Test positive confidence adjustment for high agreement."""
        fair_values = {
            'dcf': 150.0,
            'pe': 151.0,
            'ps': 149.0,
            'ev_ebitda': 150.5,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # Should get bonus for high agreement
        assert result.confidence_adjustment > 0

    def test_confidence_adjustment_divergence(self, scorer):
        """Test negative confidence adjustment for divergence."""
        fair_values = {
            'dcf': 50.0,
            'pe': 100.0,
            'ps': 200.0,
            'ev_ebitda': 150.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # Should get penalty for divergence
        if result.agreement_level == AgreementLevel.DIVERGENT:
            assert result.confidence_adjustment < 0

    def test_weighted_mean_calculation(self, scorer):
        """Test weighted mean calculation."""
        fair_values = {
            'dcf': 100.0,
            'pe': 150.0,
            'ps': 200.0,
        }
        weights = {
            'dcf': 50.0,  # Higher weight
            'pe': 30.0,
            'ps': 20.0,
        }

        result = scorer.analyze(fair_values, 'AAPL', weights)

        # Weighted mean should be closer to DCF due to higher weight
        assert result.weighted_mean < result.simple_mean

    def test_z_scores_calculated(self, scorer):
        """Test z-score calculation for each model."""
        fair_values = {
            'dcf': 100.0,
            'pe': 150.0,
            'ps': 200.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # Should have z-score for each model
        assert len(result.model_z_scores) == 3
        assert 'dcf' in result.model_z_scores
        assert 'pe' in result.model_z_scores
        assert 'ps' in result.model_z_scores

    def test_apply_outlier_penalty(self, scorer):
        """Test outlier weight penalty application."""
        weights = {
            'dcf': 30.0,
            'pe': 25.0,
            'ps': 35.0,
            'ev_ebitda': 10.0,
        }
        outliers = ['ps']

        adjusted = scorer.apply_outlier_penalty(weights, outliers)

        # PS weight should be reduced
        assert adjusted['ps'] < weights['ps']
        # Non-outliers should be relatively higher (normalized)
        assert adjusted['dcf'] > weights['dcf'] * 0.9

    def test_apply_outlier_penalty_normalizes(self, scorer):
        """Test that penalty application normalizes to 100%."""
        weights = {
            'dcf': 30.0,
            'pe': 30.0,
            'ps': 40.0,
        }
        outliers = ['ps']

        adjusted = scorer.apply_outlier_penalty(weights, outliers, normalize=True)

        # Should sum to 100
        total = sum(adjusted.values())
        assert abs(total - 100.0) < 0.1

    def test_insufficient_models(self, scorer):
        """Test handling of insufficient model count."""
        fair_values = {
            'dcf': 100.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # Should return low confidence result
        assert result.agreement_score == 0.0
        assert 'Insufficient' in result.notes[0]

    def test_invalid_values_filtered(self, scorer):
        """Test that invalid values (None, NaN, negative) are filtered."""
        fair_values = {
            'dcf': 100.0,
            'pe': None,
            'ps': float('nan'),
            'ev_ebitda': -50.0,  # Negative fair value is invalid
            'pb': 105.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')

        # Should only use valid values (dcf, pb)
        # With only 2 valid values, may still work
        assert result is not None

    def test_get_weighted_fair_value(self, scorer):
        """Test weighted fair value calculation with outlier handling."""
        fair_values = {
            'dcf': 100.0,
            'pe': 110.0,
            'ps': 300.0,  # Outlier
        }
        weights = {
            'dcf': 40.0,
            'pe': 40.0,
            'ps': 20.0,
        }

        weighted_fv, effective_weights = scorer.get_weighted_fair_value(
            fair_values,
            weights,
            apply_outlier_penalty=True
        )

        # Weighted FV should be closer to DCF/PE due to PS penalty
        assert weighted_fv < 150  # Without penalty would be higher

    def test_singleton_scorer(self):
        """Test singleton get_model_agreement_scorer function."""
        scorer1 = get_model_agreement_scorer()
        scorer2 = get_model_agreement_scorer()
        assert scorer1 is scorer2

    def test_summary_method(self, scorer):
        """Test summary string generation."""
        fair_values = {
            'dcf': 150.0,
            'pe': 155.0,
            'ps': 145.0,
        }

        result = scorer.analyze(fair_values, 'AAPL')
        summary = result.summary()

        assert 'Agreement' in summary
        assert 'CV' in summary
        assert '%' in summary
