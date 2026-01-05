"""
Regression tests for InvestmentSynthesizer recommendation methods.

These tests capture the CURRENT BEHAVIOR of the methods BEFORE extraction.
They serve as regression tests to ensure the extracted modules maintain
identical behavior.

Methods tested:
- _determine_final_recommendation
- _calculate_price_target
- _extract_position_size
- _extract_catalysts

Author: InvestiGator Team
Date: 2025-01-05
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_synthesizer():
    """Create a mock synthesizer with real methods bound."""
    from investigator.application.synthesizer import InvestmentSynthesizer

    mock = MagicMock(spec=InvestmentSynthesizer)

    # Bind the real methods
    mock._determine_final_recommendation = InvestmentSynthesizer._determine_final_recommendation.__get__(mock)
    mock._calculate_price_target = InvestmentSynthesizer._calculate_price_target.__get__(mock)
    mock._extract_position_size = InvestmentSynthesizer._extract_position_size.__get__(mock)
    mock._extract_catalysts = InvestmentSynthesizer._extract_catalysts.__get__(mock)

    # Mock logger
    mock.main_logger = MagicMock()

    return mock


class TestDetermineFinalRecommendation:
    """Tests for _determine_final_recommendation method."""

    def test_returns_investment_recommendation_format(self, mock_synthesizer):
        """Should extract from investment_recommendation structure."""
        ai_rec = {"investment_recommendation": {"recommendation": "BUY", "confidence_level": "HIGH"}}
        result = mock_synthesizer._determine_final_recommendation(7.0, ai_rec, 0.8)

        assert result["recommendation"] == "BUY"
        assert result["confidence"] == "HIGH"

    def test_handles_direct_recommendation(self, mock_synthesizer):
        """Should handle direct recommendation string."""
        ai_rec = {"recommendation": "SELL", "confidence": "MEDIUM"}
        result = mock_synthesizer._determine_final_recommendation(4.0, ai_rec, 0.8)

        assert result["recommendation"] == "SELL"
        assert result["confidence"] == "MEDIUM"

    def test_handles_dict_recommendation(self, mock_synthesizer):
        """Should handle recommendation as dict."""
        ai_rec = {"recommendation": {"rating": "HOLD", "confidence": "LOW"}}
        result = mock_synthesizer._determine_final_recommendation(5.0, ai_rec, 0.8)

        assert result["recommendation"] == "HOLD"
        assert result["confidence"] == "LOW"

    def test_downgrades_strong_rec_on_low_data_quality(self, mock_synthesizer):
        """Should downgrade STRONG recommendations on low data quality."""
        ai_rec = {"investment_recommendation": {"recommendation": "STRONG BUY", "confidence_level": "HIGH"}}
        result = mock_synthesizer._determine_final_recommendation(8.0, ai_rec, 0.3)

        # Low data quality (<0.5) should downgrade STRONG BUY to BUY
        assert result["recommendation"] == "BUY"
        assert result["confidence"] == "LOW"

    def test_adjusts_to_buy_on_high_score(self, mock_synthesizer):
        """Should adjust to BUY when score >= 8.0."""
        ai_rec = {"recommendation": "HOLD", "confidence": "MEDIUM"}
        result = mock_synthesizer._determine_final_recommendation(8.5, ai_rec, 0.8)

        assert result["recommendation"] == "BUY"

    def test_adjusts_to_sell_on_low_score(self, mock_synthesizer):
        """Should adjust to SELL when score <= 3.0."""
        ai_rec = {"recommendation": "HOLD", "confidence": "MEDIUM"}
        result = mock_synthesizer._determine_final_recommendation(2.5, ai_rec, 0.8)

        assert result["recommendation"] == "SELL"

    def test_adjusts_to_hold_on_neutral_score(self, mock_synthesizer):
        """Should adjust STRONG to HOLD when score is 4-6."""
        ai_rec = {"investment_recommendation": {"recommendation": "STRONG BUY", "confidence_level": "HIGH"}}
        result = mock_synthesizer._determine_final_recommendation(5.0, ai_rec, 0.8)

        assert result["recommendation"] == "HOLD"

    def test_defaults_to_hold_when_missing(self, mock_synthesizer):
        """Should default to HOLD when no recommendation present."""
        result = mock_synthesizer._determine_final_recommendation(5.0, {}, 0.8)

        assert result["recommendation"] == "HOLD"


class TestCalculatePriceTarget:
    """Tests for _calculate_price_target method."""

    def test_returns_structured_target_price(self, mock_synthesizer):
        """Should return target price from structured recommendation."""
        ai_rec = {"investment_recommendation": {"target_price": {"12_month_target": 150.0}}}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 130.0)

        assert result == 150.0

    def test_returns_legacy_format_target(self, mock_synthesizer):
        """Should return target from legacy price_targets format."""
        ai_rec = {"price_targets": {"12_month": 120.0}}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 100.0)

        assert result == 120.0

    def test_calculates_from_score_high(self, mock_synthesizer):
        """Should calculate 15% upside for high score."""
        ai_rec = {"overall_score": 8.5}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 100.0)

        # 100 * 1.15 = 115.0
        assert result == 115.0

    def test_calculates_from_score_medium_high(self, mock_synthesizer):
        """Should calculate 10% upside for medium-high score."""
        ai_rec = {"overall_score": 7.0}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 100.0)

        # 100 * 1.10 = 110.0
        assert result == 110.0

    def test_calculates_from_score_medium(self, mock_synthesizer):
        """Should calculate 5% upside for medium score."""
        ai_rec = {"overall_score": 5.5}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 100.0)

        # 100 * 1.05 = 105.0
        assert result == 105.0

    def test_calculates_from_score_low(self, mock_synthesizer):
        """Should calculate -5% for low score."""
        ai_rec = {"overall_score": 4.0}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 100.0)

        # 100 * 0.95 = 95.0
        assert result == 95.0

    def test_handles_zero_price_with_fallback(self, mock_synthesizer):
        """Should use fallback when price is 0."""
        ai_rec = {"overall_score": 5.0}
        result = mock_synthesizer._calculate_price_target("AAPL", {}, ai_rec, 0.0)

        # Fallback to 100, then 100 * 1.05 = 105.0
        assert result == 105.0


class TestExtractPositionSize:
    """Tests for _extract_position_size method."""

    def test_returns_large_for_high_weight(self, mock_synthesizer):
        """Should return LARGE for weight >= 5%."""
        ai_rec = {"investment_recommendation": {"position_sizing": {"recommended_weight": 0.06}}}
        result = mock_synthesizer._extract_position_size(ai_rec)

        assert result == "LARGE"

    def test_returns_moderate_for_medium_weight(self, mock_synthesizer):
        """Should return MODERATE for 3-5% weight."""
        ai_rec = {"investment_recommendation": {"position_sizing": {"recommended_weight": 0.04}}}
        result = mock_synthesizer._extract_position_size(ai_rec)

        assert result == "MODERATE"

    def test_returns_small_for_low_weight(self, mock_synthesizer):
        """Should return SMALL for weight < 3%."""
        ai_rec = {"investment_recommendation": {"position_sizing": {"recommended_weight": 0.02}}}
        result = mock_synthesizer._extract_position_size(ai_rec)

        assert result == "SMALL"

    def test_falls_back_to_direct_position_size(self, mock_synthesizer):
        """Should use direct position_size if no investment_recommendation."""
        ai_rec = {"position_size": "SMALL"}
        result = mock_synthesizer._extract_position_size(ai_rec)

        assert result == "SMALL"

    def test_defaults_to_moderate(self, mock_synthesizer):
        """Should default to MODERATE when no data."""
        result = mock_synthesizer._extract_position_size({})

        assert result == "MODERATE"


class TestExtractCatalysts:
    """Tests for _extract_catalysts method."""

    def test_extracts_from_list_of_dicts(self, mock_synthesizer):
        """Should extract catalysts from list of dict format."""
        ai_rec = {
            "key_catalysts": [
                {"catalyst": "New product launch"},
                {"catalyst": "Market expansion"},
                {"catalyst": "Cost reduction"},
            ]
        }
        result = mock_synthesizer._extract_catalysts(ai_rec)

        assert len(result) == 3
        assert "New product launch" in result

    def test_extracts_from_list_of_strings(self, mock_synthesizer):
        """Should extract catalysts from list of strings."""
        ai_rec = {
            "key_catalysts": [
                "Earnings growth",
                "Dividend increase",
            ]
        }
        result = mock_synthesizer._extract_catalysts(ai_rec)

        assert len(result) == 2
        assert "Earnings growth" in result

    def test_limits_to_three_catalysts(self, mock_synthesizer):
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
        result = mock_synthesizer._extract_catalysts(ai_rec)

        assert len(result) == 3

    def test_falls_back_to_catalysts_list(self, mock_synthesizer):
        """Should fall back to simple catalysts list."""
        ai_rec = {"catalysts": ["Growth", "Value"]}
        result = mock_synthesizer._extract_catalysts(ai_rec)

        assert result == ["Growth", "Value"]

    def test_returns_empty_list_when_none(self, mock_synthesizer):
        """Should return empty list when no catalysts."""
        result = mock_synthesizer._extract_catalysts({})

        assert result == []
