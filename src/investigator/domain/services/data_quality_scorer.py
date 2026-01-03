"""
Data Quality Scorer - Aggregate quality scoring for valuation model applicability.

Provides:
1. Overall data quality level (EXCELLENT/GOOD/FAIR/POOR/INSUFFICIENT)
2. Per-model applicability multipliers
3. Valuation confidence adjustments
4. Integration with DataValidator for comprehensive quality assessment

Usage:
    from investigator.domain.services.data_quality_scorer import DataQualityScorer

    scorer = DataQualityScorer()
    quality = scorer.score_metrics(financial_data, metadata)

    if quality.level == DataQualityLevel.INSUFFICIENT:
        logger.error("Insufficient data for valuation")
        return None
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from investigator.domain.services.data_validation import DataValidator, ValidationResult, get_data_validator

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality level classifications."""

    EXCELLENT = "excellent"  # 90-100: Full confidence
    GOOD = "good"  # 75-89: Minor gaps, high confidence
    FAIR = "fair"  # 60-74: Some gaps, proceed with caution
    POOR = "poor"  # 40-59: Significant gaps, reduce confidence
    INSUFFICIENT = "insufficient"  # <40: Cannot reliably value


@dataclass
class MetricQuality:
    """Quality assessment for a single metric category."""

    category: str
    completeness: float  # 0-100
    recency_score: float  # 0-100 (how recent is the data)
    consistency_score: float  # 0-100 (cross-metric consistency)
    issues: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Weighted average of quality components."""
        return 0.50 * self.completeness + 0.30 * self.recency_score + 0.20 * self.consistency_score


@dataclass
class AggregateQuality:
    """Aggregate data quality assessment result."""

    overall_score: float  # 0-100
    level: DataQualityLevel
    model_applicability: Dict[str, float]  # Per-model confidence multiplier (0.0-1.0)
    valuation_confidence: float  # Overall valuation confidence (0.0-1.0)
    category_scores: Dict[str, MetricQuality] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_applicable_models(self, min_confidence: float = 0.5) -> List[str]:
        """Get list of models with sufficient data quality."""
        return [model for model, conf in self.model_applicability.items() if conf >= min_confidence]

    def summary(self) -> str:
        """Get a summary of the quality assessment."""
        applicable = self.get_applicable_models()
        return (
            f"Quality: {self.level.value.upper()} ({self.overall_score:.1f}/100), "
            f"Confidence: {self.valuation_confidence:.0%}, "
            f"Applicable models: {len(applicable)}/{len(self.model_applicability)}"
        )


class DataQualityScorer:
    """
    Scores data quality for valuation model applicability.

    Integrates with DataValidator to provide:
    - Aggregate quality levels
    - Per-model confidence multipliers
    - Valuation confidence adjustments

    Example usage:
        scorer = DataQualityScorer()
        quality = scorer.score_metrics(data, metadata)

        if quality.level == DataQualityLevel.INSUFFICIENT:
            logger.error("Cannot value: insufficient data")
            return None

        for model, confidence in quality.model_applicability.items():
            if confidence < 0.5:
                logger.warning(f"Low confidence for {model}: {confidence:.0%}")
    """

    # Metric categories with their constituent fields
    METRIC_CATEGORIES = {
        "income": ["revenue", "gross_profit", "operating_income", "net_income", "ebitda", "ebit", "eps"],
        "cash_flow": [
            "operating_cash_flow",
            "free_cash_flow",
            "capital_expenditures",
            "investing_cash_flow",
            "financing_cash_flow",
        ],
        "balance_sheet": [
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "total_debt",
            "cash_and_equivalents",
            "book_value",
        ],
        "ratios": [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "fcf_margin",
            "roe",
            "roa",
            "roic",
            "debt_to_equity",
            "current_ratio",
        ],
        "growth": ["revenue_growth", "earnings_growth", "fcf_growth", "dividend_growth_rate", "book_value_growth"],
        "valuation": ["pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "peg_ratio", "dividend_yield"],
        "shares": ["shares_outstanding", "market_cap", "enterprise_value"],
    }

    # Quality level thresholds
    QUALITY_THRESHOLDS = {
        DataQualityLevel.EXCELLENT: 90,
        DataQualityLevel.GOOD: 75,
        DataQualityLevel.FAIR: 60,
        DataQualityLevel.POOR: 40,
        DataQualityLevel.INSUFFICIENT: 0,
    }

    # Model to category weights
    MODEL_CATEGORY_WEIGHTS = {
        "dcf": {
            "cash_flow": 0.40,
            "income": 0.25,
            "growth": 0.20,
            "balance_sheet": 0.10,
            "shares": 0.05,
        },
        "ggm": {
            "income": 0.35,
            "growth": 0.30,
            "ratios": 0.20,
            "balance_sheet": 0.10,
            "shares": 0.05,
        },
        "pe": {
            "income": 0.50,
            "valuation": 0.25,
            "growth": 0.15,
            "shares": 0.10,
        },
        "ps": {
            "income": 0.50,
            "valuation": 0.25,
            "growth": 0.15,
            "shares": 0.10,
        },
        "pb": {
            "balance_sheet": 0.50,
            "valuation": 0.25,
            "ratios": 0.15,
            "shares": 0.10,
        },
        "ev_ebitda": {
            "income": 0.40,
            "balance_sheet": 0.30,
            "valuation": 0.20,
            "shares": 0.10,
        },
        "rule_of_40": {
            "income": 0.40,
            "growth": 0.35,
            "cash_flow": 0.25,
        },
        "saas": {
            "income": 0.30,
            "growth": 0.30,
            "ratios": 0.25,
            "cash_flow": 0.15,
        },
    }

    def __init__(self, validator: Optional[DataValidator] = None):
        """
        Initialize the scorer.

        Args:
            validator: Optional DataValidator instance (uses singleton if not provided)
        """
        self.validator = validator or get_data_validator()

    def score_metrics(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> AggregateQuality:
        """
        Calculate aggregate data quality scores.

        Args:
            data: Dictionary of financial metrics
            metadata: Optional metadata (fiscal_period, data_source, etc.)

        Returns:
            AggregateQuality with overall score, level, and model applicability
        """
        if not data:
            return AggregateQuality(
                overall_score=0.0,
                level=DataQualityLevel.INSUFFICIENT,
                model_applicability={m: 0.0 for m in self.MODEL_CATEGORY_WEIGHTS},
                valuation_confidence=0.0,
                issues=["No data provided"],
                recommendations=["Verify data source and refresh data"],
            )

        metadata = metadata or {}

        # Score each category
        category_scores = self._score_categories(data, metadata)

        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)

        # Determine quality level
        level = self._determine_level(overall_score)

        # Calculate per-model applicability
        model_applicability = self._calculate_model_applicability(data, category_scores)

        # Calculate valuation confidence
        valuation_confidence = self._calculate_valuation_confidence(overall_score, model_applicability, level)

        # Generate issues and recommendations
        issues = self._collect_issues(category_scores)
        recommendations = self._generate_recommendations(level, category_scores, model_applicability)

        return AggregateQuality(
            overall_score=round(overall_score, 1),
            level=level,
            model_applicability=model_applicability,
            valuation_confidence=round(valuation_confidence, 2),
            category_scores=category_scores,
            issues=issues,
            recommendations=recommendations,
        )

    def _score_categories(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, MetricQuality]:
        """Score each metric category for completeness and quality."""
        category_scores = {}

        for category, fields in self.METRIC_CATEGORIES.items():
            # Count valid fields
            valid_count = sum(1 for f in fields if self.validator._has_valid_value(data.get(f)))
            total = len(fields)
            completeness = (valid_count / total * 100) if total > 0 else 0

            # Calculate recency score from metadata
            recency_score = self._calculate_recency_score(metadata)

            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(data, category)

            # Collect issues for this category
            issues = []
            missing = [f for f in fields if not self.validator._has_valid_value(data.get(f))]
            if missing and len(missing) <= 3:
                issues.append(f"Missing: {', '.join(missing)}")
            elif len(missing) > 3:
                issues.append(f"Missing {len(missing)} of {total} fields")

            category_scores[category] = MetricQuality(
                category=category,
                completeness=completeness,
                recency_score=recency_score,
                consistency_score=consistency_score,
                issues=issues,
            )

        return category_scores

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on data age."""
        # Default to good recency if no metadata
        if not metadata:
            return 80.0

        # Check for data age indicators
        quarters_old = metadata.get("quarters_old", 0)

        if quarters_old == 0:
            return 100.0  # Current quarter
        elif quarters_old == 1:
            return 90.0  # Last quarter
        elif quarters_old == 2:
            return 75.0  # Two quarters old
        elif quarters_old <= 4:
            return 60.0  # Up to a year old
        else:
            return 40.0  # Over a year old

    def _calculate_consistency_score(self, data: Dict[str, Any], category: str) -> float:
        """Calculate consistency score for a category."""
        # Use DataValidator's consistency checks
        consistency_issues = self.validator.validate_consistency(data)

        # Map issues to categories and deduct points
        category_issues = [
            i for i in consistency_issues if any(f in self.METRIC_CATEGORIES.get(category, []) for f in [i.field])
        ]

        # Start at 100, deduct for each issue
        score = 100.0 - (len(category_issues) * 15)
        return max(0.0, score)

    def _calculate_overall_score(self, category_scores: Dict[str, MetricQuality]) -> float:
        """Calculate weighted overall quality score."""
        if not category_scores:
            return 0.0

        # Weight categories by importance for valuation
        category_weights = {
            "income": 0.25,
            "cash_flow": 0.20,
            "balance_sheet": 0.15,
            "ratios": 0.15,
            "growth": 0.10,
            "valuation": 0.10,
            "shares": 0.05,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for category, quality in category_scores.items():
            weight = category_weights.get(category, 0.1)
            weighted_sum += quality.overall_score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_level(self, score: float) -> DataQualityLevel:
        """Determine quality level from score."""
        for level, threshold in sorted(self.QUALITY_THRESHOLDS.items(), key=lambda x: -x[1]):  # Sort descending
            if score >= threshold:
                return level
        return DataQualityLevel.INSUFFICIENT

    def _calculate_model_applicability(
        self, data: Dict[str, Any], category_scores: Dict[str, MetricQuality]
    ) -> Dict[str, float]:
        """Calculate per-model applicability confidence."""
        model_applicability = {}

        for model, category_weights in self.MODEL_CATEGORY_WEIGHTS.items():
            # Calculate weighted category score for this model
            weighted_score = 0.0
            total_weight = 0.0

            for category, weight in category_weights.items():
                if category in category_scores:
                    weighted_score += category_scores[category].overall_score * weight
                    total_weight += weight

            category_score = weighted_score / total_weight if total_weight > 0 else 0

            # Also check DataValidator's model-specific validation
            is_applicable, confidence_adj, _ = self.validator.validate_for_model(data, model)

            if not is_applicable:
                # Not applicable at all
                model_applicability[model] = 0.0
            else:
                # Combine category score and validator confidence
                combined = (category_score / 100) * confidence_adj
                model_applicability[model] = round(combined, 2)

        return model_applicability

    def _calculate_valuation_confidence(
        self, overall_score: float, model_applicability: Dict[str, float], level: DataQualityLevel
    ) -> float:
        """Calculate overall valuation confidence."""
        # Base confidence from overall score
        base_confidence = overall_score / 100

        # Adjust based on how many models are applicable
        applicable_models = sum(1 for conf in model_applicability.values() if conf >= 0.5)
        total_models = len(model_applicability)
        model_coverage = applicable_models / total_models if total_models > 0 else 0

        # Weight: 70% data quality, 30% model coverage
        confidence = 0.70 * base_confidence + 0.30 * model_coverage

        # Apply level-based caps
        level_caps = {
            DataQualityLevel.EXCELLENT: 1.0,
            DataQualityLevel.GOOD: 0.90,
            DataQualityLevel.FAIR: 0.75,
            DataQualityLevel.POOR: 0.50,
            DataQualityLevel.INSUFFICIENT: 0.25,
        }

        return min(confidence, level_caps.get(level, 1.0))

    def _collect_issues(self, category_scores: Dict[str, MetricQuality]) -> List[str]:
        """Collect all issues from category scores."""
        issues = []
        for category, quality in category_scores.items():
            for issue in quality.issues:
                issues.append(f"[{category}] {issue}")
        return issues

    def _generate_recommendations(
        self, level: DataQualityLevel, category_scores: Dict[str, MetricQuality], model_applicability: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []

        # Level-based recommendations
        if level == DataQualityLevel.INSUFFICIENT:
            recommendations.append(
                "Data quality insufficient for reliable valuation. " "Consider fetching additional data sources."
            )
        elif level == DataQualityLevel.POOR:
            recommendations.append("Data quality is poor. Results should be used with caution.")

        # Category-specific recommendations
        weak_categories = [(cat, q) for cat, q in category_scores.items() if q.completeness < 50]
        if weak_categories:
            cats = ", ".join(c for c, _ in weak_categories[:3])
            recommendations.append(
                f"Low data completeness in: {cats}. " f"Consider supplementing with alternative data sources."
            )

        # Model-specific recommendations
        low_applicability = [model for model, conf in model_applicability.items() if 0 < conf < 0.5]
        if low_applicability:
            models = ", ".join(low_applicability[:3])
            recommendations.append(
                f"Models with limited applicability: {models}. " f"Consider excluding or weighting down these models."
            )

        return recommendations


# Singleton instance
_scorer: Optional[DataQualityScorer] = None


def get_data_quality_scorer() -> DataQualityScorer:
    """Get the singleton DataQualityScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = DataQualityScorer()
    return _scorer
