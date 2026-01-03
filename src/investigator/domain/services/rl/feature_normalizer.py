"""
Feature Normalizer for RL Training

Normalizes features for stable RL training using running statistics.
Supports both batch normalization (fit on historical data) and
online normalization (update statistics incrementally).

Important for financial data which is non-stationary (distributions
change over time).

Usage:
    from investigator.domain.services.rl import FeatureNormalizer

    normalizer = FeatureNormalizer()

    # Fit on historical data
    normalizer.fit(contexts)

    # Transform new data
    features = normalizer.transform(context)

    # Save/load for persistence
    normalizer.save("data/rl_models/normalizer.pkl")
    normalizer.load("data/rl_models/normalizer.pkl")
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from investigator.domain.services.rl.models import ValuationContext
from investigator.domain.services.rl.feature_extractor import (
    ValuationContextExtractor,
    GICS_SECTORS,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureStatistics:
    """Running statistics for a single feature."""

    name: str
    count: int = 0
    mean: float = 0.0
    variance: float = 1.0
    min_val: float = float("inf")
    max_val: float = float("-inf")

    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(max(1e-8, self.variance))

    def update(self, value: float) -> None:
        """Update statistics with new value using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.variance = (
            (self.variance * (self.count - 1) + delta * delta2) / self.count if self.count > 1 else self.variance
        )
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def batch_update(self, values: np.ndarray) -> None:
        """Batch update statistics."""
        for v in values:
            self.update(float(v))


@dataclass
class NormalizerState:
    """Complete state of the normalizer for persistence."""

    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    feature_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    normalization_method: str = "z_score"
    include_categorical: bool = True
    clip_range: Tuple[float, float] = (-3.0, 3.0)


class FeatureNormalizer:
    """
    Normalizes features for stable RL training.

    Supports multiple normalization methods:
    - z_score: Subtract mean, divide by std (good for neural networks)
    - min_max: Scale to [0, 1] range
    - robust: Use median and IQR (robust to outliers)

    Uses running statistics that can be updated incrementally,
    important for non-stationary financial data.
    """

    def __init__(
        self,
        normalization_method: str = "z_score",
        include_categorical: bool = True,
        clip_range: Tuple[float, float] = (-3.0, 3.0),
    ):
        """
        Initialize normalizer.

        Args:
            normalization_method: "z_score", "min_max", or "robust"
            include_categorical: Whether to include categorical features.
            clip_range: Range to clip z-scores to (for outlier handling).
        """
        self.normalization_method = normalization_method
        self.include_categorical = include_categorical
        self.clip_range = clip_range
        self.extractor = ValuationContextExtractor()

        # Statistics for each feature
        self._stats: Dict[str, FeatureStatistics] = {}
        self._fitted = False
        self._version = "1.0"
        self._created_at = datetime.now()
        self._updated_at = datetime.now()

        # Get feature names
        self._feature_names = self.extractor.get_feature_names(include_categorical)
        self._initialize_stats()

    def _initialize_stats(self) -> None:
        """Initialize statistics for all features."""
        for name in self._feature_names:
            self._stats[name] = FeatureStatistics(name=name)

    def fit(
        self,
        contexts: List[ValuationContext],
        reset: bool = True,
    ) -> "FeatureNormalizer":
        """
        Fit normalizer on historical data.

        Args:
            contexts: List of ValuationContext objects.
            reset: If True, reset statistics before fitting.

        Returns:
            self for chaining.
        """
        if reset:
            self._initialize_stats()

        if not contexts:
            logger.warning("Empty context list provided to fit()")
            return self

        # Convert contexts to feature matrix
        features_matrix = np.array([self.extractor.to_tensor(ctx, self.include_categorical) for ctx in contexts])

        # Update statistics for each feature
        for i, name in enumerate(self._feature_names):
            if i < features_matrix.shape[1]:
                self._stats[name].batch_update(features_matrix[:, i])

        self._fitted = True
        self._updated_at = datetime.now()
        logger.info(f"Fitted normalizer on {len(contexts)} samples")

        return self

    def partial_fit(
        self,
        context: ValuationContext,
    ) -> "FeatureNormalizer":
        """
        Update normalizer with single new sample (online learning).

        Args:
            context: New ValuationContext to incorporate.

        Returns:
            self for chaining.
        """
        features = self.extractor.to_tensor(context, self.include_categorical)

        for i, name in enumerate(self._feature_names):
            if i < len(features):
                self._stats[name].update(float(features[i]))

        self._fitted = True
        self._updated_at = datetime.now()

        return self

    def transform(
        self,
        context: ValuationContext,
    ) -> np.ndarray:
        """
        Transform context features to normalized values.

        Args:
            context: ValuationContext to transform.

        Returns:
            Normalized feature array.
        """
        raw_features = self.extractor.to_tensor(context, self.include_categorical)

        if not self._fitted:
            logger.warning("Normalizer not fitted, returning raw features")
            return raw_features

        normalized = np.zeros_like(raw_features)

        for i, name in enumerate(self._feature_names):
            if i >= len(raw_features):
                break

            stats = self._stats.get(name)
            if stats is None or stats.count < 2:
                normalized[i] = raw_features[i]
                continue

            if self.normalization_method == "z_score":
                normalized[i] = self._z_score_normalize(raw_features[i], stats.mean, stats.std)
            elif self.normalization_method == "min_max":
                normalized[i] = self._min_max_normalize(raw_features[i], stats.min_val, stats.max_val)
            else:  # robust or unknown
                normalized[i] = self._z_score_normalize(raw_features[i], stats.mean, stats.std)

        return normalized

    def fit_transform(
        self,
        contexts: List[ValuationContext],
    ) -> np.ndarray:
        """
        Fit normalizer and transform data in one step.

        Args:
            contexts: List of ValuationContext objects.

        Returns:
            Array of normalized feature vectors.
        """
        self.fit(contexts)
        return np.array([self.transform(ctx) for ctx in contexts])

    def inverse_transform(
        self,
        normalized: np.ndarray,
    ) -> np.ndarray:
        """
        Convert normalized features back to original scale.

        Args:
            normalized: Normalized feature array.

        Returns:
            Original scale feature array.
        """
        if not self._fitted:
            return normalized

        original = np.zeros_like(normalized)

        for i, name in enumerate(self._feature_names):
            if i >= len(normalized):
                break

            stats = self._stats.get(name)
            if stats is None or stats.count < 2:
                original[i] = normalized[i]
                continue

            if self.normalization_method == "z_score":
                original[i] = normalized[i] * stats.std + stats.mean
            elif self.normalization_method == "min_max":
                range_val = stats.max_val - stats.min_val
                if range_val > 0:
                    original[i] = normalized[i] * range_val + stats.min_val
                else:
                    original[i] = stats.mean
            else:
                original[i] = normalized[i] * stats.std + stats.mean

        return original

    def _z_score_normalize(
        self,
        value: float,
        mean: float,
        std: float,
    ) -> float:
        """Z-score normalization with clipping."""
        if std < 1e-8:
            return 0.0
        z = (value - mean) / std
        return float(np.clip(z, self.clip_range[0], self.clip_range[1]))

    def _min_max_normalize(
        self,
        value: float,
        min_val: float,
        max_val: float,
    ) -> float:
        """Min-max normalization to [0, 1]."""
        range_val = max_val - min_val
        if range_val < 1e-8:
            return 0.5
        return (value - min_val) / range_val

    def save(self, path: str) -> bool:
        """
        Save normalizer state to file.

        Args:
            path: File path to save to.

        Returns:
            True if successful.
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Build state object
            state = NormalizerState(
                version=self._version,
                created_at=self._created_at.isoformat(),
                updated_at=self._updated_at.isoformat(),
                normalization_method=self.normalization_method,
                include_categorical=self.include_categorical,
                clip_range=self.clip_range,
                feature_stats={
                    name: {
                        "count": s.count,
                        "mean": s.mean,
                        "variance": s.variance,
                        "min_val": s.min_val,
                        "max_val": s.max_val,
                    }
                    for name, s in self._stats.items()
                },
            )

            with open(path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved normalizer to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save normalizer: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load normalizer state from file.

        Args:
            path: File path to load from.

        Returns:
            True if successful.
        """
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            if not isinstance(state, NormalizerState):
                logger.error(f"Invalid normalizer state in {path}")
                return False

            self._version = state.version
            self._created_at = datetime.fromisoformat(state.created_at)
            self._updated_at = datetime.fromisoformat(state.updated_at)
            self.normalization_method = state.normalization_method
            self.include_categorical = state.include_categorical
            self.clip_range = state.clip_range

            # Restore feature statistics
            self._stats = {}
            for name, stats_dict in state.feature_stats.items():
                self._stats[name] = FeatureStatistics(
                    name=name,
                    count=stats_dict["count"],
                    mean=stats_dict["mean"],
                    variance=stats_dict["variance"],
                    min_val=stats_dict["min_val"],
                    max_val=stats_dict["max_val"],
                )

            self._fitted = True
            logger.info(f"Loaded normalizer from {path}")
            return True

        except FileNotFoundError:
            logger.warning(f"Normalizer file not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load normalizer: {e}")
            return False

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current feature statistics."""
        return {
            name: {
                "count": s.count,
                "mean": s.mean,
                "std": s.std,
                "min": s.min_val if s.min_val != float("inf") else None,
                "max": s.max_val if s.max_val != float("-inf") else None,
            }
            for name, s in self._stats.items()
        }

    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted."""
        return self._fitted

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self._feature_names)

    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names.copy()


# Factory function
def get_feature_normalizer(
    normalization_method: str = "z_score",
) -> FeatureNormalizer:
    """Get FeatureNormalizer instance."""
    return FeatureNormalizer(normalization_method=normalization_method)
