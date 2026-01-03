"""
Unit tests for InvestmentRecommendation domain model.
"""

import pytest

from investigator.domain.models import InvestmentRecommendation


class TestInvestmentRecommendation:
    """Test InvestmentRecommendation domain model."""

    def test_recommendation_class_exists(self):
        """Test InvestmentRecommendation class can be imported."""
        assert InvestmentRecommendation is not None

    def test_recommendation_is_dataclass(self):
        """Test InvestmentRecommendation is a dataclass."""
        # Check if it has dataclass attributes
        assert hasattr(InvestmentRecommendation, "__dataclass_fields__")

    def test_recommendation_has_required_fields(self):
        """Test InvestmentRecommendation has expected fields."""
        fields = InvestmentRecommendation.__dataclass_fields__

        required_fields = ["symbol", "analysis_timestamp", "overall_score", "fundamental_score", "technical_score"]

        for field in required_fields:
            assert field in fields, f"Missing field: {field}"

    def test_recommendation_has_helper_methods(self):
        """Test InvestmentRecommendation has helper methods."""
        required_methods = ["to_dict", "get_risk_level", "is_buy_candidate", "get_summary"]

        for method in required_methods:
            assert hasattr(InvestmentRecommendation, method), f"Missing method: {method}"
