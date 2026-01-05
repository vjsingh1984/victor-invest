"""
Unit tests for FeatureNormalizer.
"""

import os
import tempfile
from datetime import date

import numpy as np
import pytest

from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import ValuationContext


class TestFeatureNormalizer:
    """Tests for FeatureNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create a fresh normalizer."""
        return FeatureNormalizer()

    @pytest.fixture
    def sample_contexts(self):
        """Create sample contexts for testing."""
        return [
            ValuationContext(
                symbol="AAPL",
                analysis_date=date(2024, 1, 15),
                sector="Technology",
                industry="Consumer Electronics",
                profitability_score=0.85,
                pe_level=0.6,
                revenue_growth=0.15,
                fcf_margin=0.25,
                rule_of_40_score=45.0,
            ),
            ValuationContext(
                symbol="MSFT",
                analysis_date=date(2024, 1, 15),
                sector="Technology",
                industry="Software - Infrastructure",
                profitability_score=0.90,
                pe_level=0.7,
                revenue_growth=0.20,
                fcf_margin=0.35,
                rule_of_40_score=55.0,
            ),
            ValuationContext(
                symbol="XOM",
                analysis_date=date(2024, 1, 15),
                sector="Energy",
                industry="Oil & Gas Integrated",
                profitability_score=0.60,
                pe_level=0.3,
                revenue_growth=0.05,
                fcf_margin=0.15,
                rule_of_40_score=20.0,
            ),
        ]

    def test_fit_marks_as_fitted(self, normalizer, sample_contexts):
        """Test that fit marks normalizer as fitted."""
        normalizer.fit(sample_contexts)
        assert normalizer.is_fitted is True

    def test_transform_returns_array(self, normalizer, sample_contexts):
        """Test that transform returns numpy array."""
        normalizer.fit(sample_contexts)
        normalized = normalizer.transform(sample_contexts[0])
        assert isinstance(normalized, np.ndarray)

    def test_transform_consistent_length(self, normalizer, sample_contexts):
        """Test that transformed arrays have consistent length."""
        normalizer.fit(sample_contexts)

        normalized1 = normalizer.transform(sample_contexts[0])
        normalized2 = normalizer.transform(sample_contexts[1])
        normalized3 = normalizer.transform(sample_contexts[2])

        assert len(normalized1) == len(normalized2) == len(normalized3)

    def test_save_and_load(self, normalizer, sample_contexts):
        """Test saving and loading normalizer state."""
        normalizer.fit(sample_contexts)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "normalizer.pkl")

            # Save
            normalizer.save(path)
            assert os.path.exists(path)

            # Load into new normalizer
            new_normalizer = FeatureNormalizer()
            loaded = new_normalizer.load(path)

            assert loaded is True
            assert new_normalizer.is_fitted is True

    def test_transform_unfitted_returns_raw(self, normalizer):
        """Test that transform on unfitted normalizer returns raw features."""
        ctx = ValuationContext(
            symbol="TEST",
            analysis_date=date(2024, 1, 15),
            sector="Technology",
            industry="Software",
        )

        # Should not raise, just return features (with warning)
        normalized = normalizer.transform(ctx)
        assert isinstance(normalized, np.ndarray)


class TestNormalizerEdgeCases:
    """Edge case tests for FeatureNormalizer."""

    def test_empty_context_list(self):
        """Test fitting on empty list."""
        normalizer = FeatureNormalizer()
        normalizer.fit([])
        assert normalizer.is_fitted is False

    def test_single_context(self):
        """Test fitting on single context."""
        normalizer = FeatureNormalizer()

        ctx = ValuationContext(
            symbol="SINGLE",
            analysis_date=date(2024, 1, 15),
            sector="Technology",
            industry="Software",
            profitability_score=0.5,
        )

        normalizer.fit([ctx])

        # Should still work
        normalized = normalizer.transform(ctx)
        assert isinstance(normalized, np.ndarray)
        assert not np.any(np.isnan(normalized))

    def test_zero_variance_feature(self):
        """Test handling of features with zero variance."""
        normalizer = FeatureNormalizer()

        # All same values
        contexts = [
            ValuationContext(
                symbol=f"SYM{i}",
                analysis_date=date(2024, 1, 15),
                sector="Technology",
                industry="Software",
                profitability_score=0.5,
            )
            for i in range(5)
        ]

        normalizer.fit(contexts)
        normalized = normalizer.transform(contexts[0])

        # Should handle zero std gracefully
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        normalizer = FeatureNormalizer()
        loaded = normalizer.load("/nonexistent/path.pkl")
        assert loaded is False
