"""
Tests for the extracted RecommendationBuilder module.

Verifies the extracted module has identical behavior to the original
InvestmentSynthesizer methods.

Author: InvestiGator Team
Date: 2025-01-05
"""

import pytest

from investigator.application.recommendation_builder import (
    RecommendationBuilder,
    get_recommendation_builder,
)


class TestDetermineFinalRecommendation:
    """Tests for determine_final_recommendation method."""

    @pytest.fixture
    def builder(self):
        """Create a fresh builder for each test."""
        return RecommendationBuilder()

    def test_returns_investment_recommendation_format(self, builder):
        """Should extract from investment_recommendation structure."""
        ai_rec = {"investment_recommendation": {"recommendation": "BUY", "confidence_level": "HIGH"}}
        result = builder.determine_final_recommendation(7.0, ai_rec, 0.8)

        assert result["recommendation"] == "BUY"
        assert result["confidence"] == "HIGH"

    def test_handles_direct_recommendation(self, builder):
        """Should handle direct recommendation string."""
        ai_rec = {"recommendation": "SELL", "confidence": "MEDIUM"}
        result = builder.determine_final_recommendation(4.0, ai_rec, 0.8)

        assert result["recommendation"] == "SELL"
        assert result["confidence"] == "MEDIUM"

    def test_handles_dict_recommendation(self, builder):
        """Should handle recommendation as dict."""
        ai_rec = {"recommendation": {"rating": "HOLD", "confidence": "LOW"}}
        result = builder.determine_final_recommendation(5.0, ai_rec, 0.8)

        assert result["recommendation"] == "HOLD"
        assert result["confidence"] == "LOW"

    def test_downgrades_strong_rec_on_low_quality(self, builder):
        """Should downgrade STRONG recommendations on low data quality."""
        ai_rec = {"investment_recommendation": {"recommendation": "STRONG BUY", "confidence_level": "HIGH"}}
        result = builder.determine_final_recommendation(8.0, ai_rec, 0.3)

        assert result["recommendation"] == "BUY"
        assert result["confidence"] == "LOW"

    def test_adjusts_to_buy_on_high_score(self, builder):
        """Should adjust to BUY when score >= 8.0."""
        ai_rec = {"recommendation": "HOLD", "confidence": "MEDIUM"}
        result = builder.determine_final_recommendation(8.5, ai_rec, 0.8)

        assert result["recommendation"] == "BUY"

    def test_adjusts_to_sell_on_low_score(self, builder):
        """Should adjust to SELL when score <= 3.0."""
        ai_rec = {"recommendation": "HOLD", "confidence": "MEDIUM"}
        result = builder.determine_final_recommendation(2.5, ai_rec, 0.8)

        assert result["recommendation"] == "SELL"

    def test_adjusts_to_hold_on_neutral_score(self, builder):
        """Should adjust STRONG to HOLD when score is 4-6."""
        ai_rec = {"investment_recommendation": {"recommendation": "STRONG BUY", "confidence_level": "HIGH"}}
        result = builder.determine_final_recommendation(5.0, ai_rec, 0.8)

        assert result["recommendation"] == "HOLD"

    def test_defaults_to_hold_when_missing(self, builder):
        """Should default to HOLD when no recommendation."""
        result = builder.determine_final_recommendation(5.0, {}, 0.8)

        assert result["recommendation"] == "HOLD"


class TestCalculatePriceTarget:
    """Tests for calculate_price_target method."""

    @pytest.fixture
    def builder(self):
        """Create a fresh builder for each test."""
        return RecommendationBuilder()

    def test_returns_structured_target_price(self, builder):
        """Should return target price from structured recommendation."""
        ai_rec = {"investment_recommendation": {"target_price": {"12_month_target": 150.0}}}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 130.0)

        assert result == 150.0

    def test_returns_legacy_format_target(self, builder):
        """Should return target from legacy price_targets format."""
        ai_rec = {"price_targets": {"12_month": 120.0}}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 100.0)

        assert result == 120.0

    def test_calculates_from_score_high(self, builder):
        """Should calculate 15% upside for high score."""
        ai_rec = {"overall_score": 8.5}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 100.0)

        assert result == 115.0

    def test_calculates_from_score_medium_high(self, builder):
        """Should calculate 10% upside for medium-high score."""
        ai_rec = {"overall_score": 7.0}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 100.0)

        assert result == 110.0

    def test_calculates_from_score_medium(self, builder):
        """Should calculate 5% upside for medium score."""
        ai_rec = {"overall_score": 5.5}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 100.0)

        assert result == 105.0

    def test_calculates_from_score_low(self, builder):
        """Should calculate -5% for low score."""
        ai_rec = {"overall_score": 4.0}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 100.0)

        assert result == 95.0

    def test_handles_zero_price_with_fallback(self, builder):
        """Should use fallback when price is 0."""
        ai_rec = {"overall_score": 5.0}
        result = builder.calculate_price_target("AAPL", {}, ai_rec, 0.0)

        # Fallback to 100, then 100 * 1.05 = 105.0
        assert result == 105.0


class TestExtractPositionSize:
    """Tests for extract_position_size method."""

    @pytest.fixture
    def builder(self):
        """Create a fresh builder for each test."""
        return RecommendationBuilder()

    def test_returns_large_for_high_weight(self, builder):
        """Should return LARGE for weight >= 5%."""
        ai_rec = {"investment_recommendation": {"position_sizing": {"recommended_weight": 0.06}}}
        result = builder.extract_position_size(ai_rec)

        assert result == "LARGE"

    def test_returns_moderate_for_medium_weight(self, builder):
        """Should return MODERATE for 3-5% weight."""
        ai_rec = {"investment_recommendation": {"position_sizing": {"recommended_weight": 0.04}}}
        result = builder.extract_position_size(ai_rec)

        assert result == "MODERATE"

    def test_returns_small_for_low_weight(self, builder):
        """Should return SMALL for weight < 3%."""
        ai_rec = {"investment_recommendation": {"position_sizing": {"recommended_weight": 0.02}}}
        result = builder.extract_position_size(ai_rec)

        assert result == "SMALL"

    def test_falls_back_to_direct_position_size(self, builder):
        """Should use direct position_size if no investment_recommendation."""
        ai_rec = {"position_size": "SMALL"}
        result = builder.extract_position_size(ai_rec)

        assert result == "SMALL"

    def test_defaults_to_moderate(self, builder):
        """Should default to MODERATE when no data."""
        result = builder.extract_position_size({})

        assert result == "MODERATE"


class TestExtractCatalysts:
    """Tests for extract_catalysts method."""

    @pytest.fixture
    def builder(self):
        """Create a fresh builder for each test."""
        return RecommendationBuilder()

    def test_extracts_from_list_of_dicts(self, builder):
        """Should extract catalysts from list of dict format."""
        ai_rec = {
            "key_catalysts": [
                {"catalyst": "New product launch"},
                {"catalyst": "Market expansion"},
                {"catalyst": "Cost reduction"},
            ]
        }
        result = builder.extract_catalysts(ai_rec)

        assert len(result) == 3
        assert "New product launch" in result

    def test_extracts_from_list_of_strings(self, builder):
        """Should extract catalysts from list of strings."""
        ai_rec = {
            "key_catalysts": [
                "Earnings growth",
                "Dividend increase",
            ]
        }
        result = builder.extract_catalysts(ai_rec)

        assert len(result) == 2
        assert "Earnings growth" in result

    def test_limits_to_three_catalysts(self, builder):
        """Should limit to 3 catalysts maximum."""
        ai_rec = {
            "key_catalysts": [
                {"catalyst": "A"},
                {"catalyst": "B"},
                {"catalyst": "C"},
                {"catalyst": "D"},
                {"catalyst": "E"},
            ]
        }
        result = builder.extract_catalysts(ai_rec)

        assert len(result) == 3

    def test_falls_back_to_catalysts_list(self, builder):
        """Should fall back to simple catalysts list."""
        ai_rec = {"catalysts": ["Growth", "Value"]}
        result = builder.extract_catalysts(ai_rec)

        assert result == ["Growth", "Value"]

    def test_returns_empty_list_when_none(self, builder):
        """Should return empty list when no catalysts."""
        result = builder.extract_catalysts({})

        assert result == []


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_builder_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        b1 = get_recommendation_builder()
        b2 = get_recommendation_builder()
        assert b1 is b2
