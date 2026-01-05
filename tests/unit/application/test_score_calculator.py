"""
Tests for the extracted ScoreCalculator module.

Verifies the extracted module has identical behavior to the original
InvestmentSynthesizer methods.

Author: InvestiGator Team
Date: 2025-01-05
"""

import json

import pytest

from investigator.application.score_calculator import (
    ScoreCalculator,
    ScoreWeights,
    get_score_calculator,
)


class TestScoreCalculatorFundamentalScore:
    """Tests for calculate_fundamental_score method."""

    @pytest.fixture
    def calculator(self):
        """Create a fresh calculator for each test."""
        return ScoreCalculator()

    def test_empty_responses_returns_zero(self, calculator):
        """Empty llm_responses should return 0.0."""
        result = calculator.calculate_fundamental_score({})
        assert result == 0.0

    def test_no_fundamental_key_returns_zero(self, calculator):
        """Missing 'fundamental' key should return 0.0."""
        result = calculator.calculate_fundamental_score({"technical": {}})
        assert result == 0.0

    def test_comprehensive_dict_financial_health_score(self, calculator):
        """Should extract financial_health_score from comprehensive dict response."""
        llm_responses = {
            "fundamental": {"comprehensive": {"content": {"financial_health_score": 7.5, "other_data": "value"}}}
        }
        result = calculator.calculate_fundamental_score(llm_responses)
        assert result == 7.5

    def test_comprehensive_dict_overall_score(self, calculator):
        """Should fallback to overall_score if financial_health_score missing."""
        llm_responses = {"fundamental": {"comprehensive": {"content": {"overall_score": 8.0, "other_data": "value"}}}}
        result = calculator.calculate_fundamental_score(llm_responses)
        assert result == 8.0

    def test_comprehensive_string_json_response(self, calculator):
        """Should parse JSON string response."""
        llm_responses = {"fundamental": {"comprehensive": {"content": json.dumps({"financial_health_score": 6.5})}}}
        result = calculator.calculate_fundamental_score(llm_responses)
        assert result == 6.5

    def test_average_quarterly_scores(self, calculator):
        """Should average multiple quarterly scores if no comprehensive."""
        llm_responses = {
            "fundamental": {
                "Q1_2024": {"content": {"financial_health_score": 6.0}},
                "Q2_2024": {"content": {"financial_health_score": 7.0}},
                "Q3_2024": {"content": {"financial_health_score": 8.0}},
            }
        }
        result = calculator.calculate_fundamental_score(llm_responses)
        assert result == 7.0  # Average of 6, 7, 8


class TestScoreCalculatorTechnicalScore:
    """Tests for calculate_technical_score method."""

    @pytest.fixture
    def calculator(self):
        """Create a fresh calculator for each test."""
        return ScoreCalculator()

    def test_empty_responses_returns_zero(self, calculator):
        """Empty llm_responses should return 0.0."""
        result = calculator.calculate_technical_score({})
        assert result == 0.0

    def test_dict_content_with_score_dict(self, calculator):
        """Should extract score from nested dict format."""
        llm_responses = {"technical": {"content": {"technical_score": {"score": 7.5, "confidence": "high"}}}}
        result = calculator.calculate_technical_score(llm_responses)
        assert result == 7.5

    def test_dict_content_with_score_float(self, calculator):
        """Should handle direct float technical_score."""
        llm_responses = {"technical": {"content": {"technical_score": 8.0}}}
        result = calculator.calculate_technical_score(llm_responses)
        assert result == 8.0

    def test_string_json_content(self, calculator):
        """Should parse JSON string content."""
        llm_responses = {"technical": {"content": json.dumps({"technical_score": {"score": 6.5}})}}
        result = calculator.calculate_technical_score(llm_responses)
        assert result == 6.5


class TestScoreCalculatorWeightedScore:
    """Tests for calculate_weighted_score method."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with default weights."""
        return ScoreCalculator(ScoreWeights(fundamental_weight=0.6, technical_weight=0.4))

    def test_equal_scores(self, calculator):
        """Equal scores should return that score."""
        result = calculator.calculate_weighted_score(7.0, 7.0)
        assert result == 7.0

    def test_fundamental_higher(self, calculator):
        """Weighted average with fundamental higher."""
        result = calculator.calculate_weighted_score(8.0, 6.0)
        expected = 8.0 * 0.6 + 6.0 * 0.4  # 7.2
        assert abs(result - expected) < 0.2

    def test_zero_scores(self, calculator):
        """Both zero should return 0."""
        result = calculator.calculate_weighted_score(0.0, 0.0)
        assert result == 0.0


class TestScoreCalculatorMomentumSignals:
    """Tests for extract_momentum_signals method."""

    @pytest.fixture
    def calculator(self):
        """Create a fresh calculator for each test."""
        return ScoreCalculator()

    def test_empty_content_returns_empty_list(self, calculator):
        """Empty content should return empty list."""
        result = calculator.extract_momentum_signals({})
        assert result == []

    def test_extracts_rsi_signal(self, calculator):
        """Should extract RSI from momentum_analysis."""
        content = {
            "momentum_analysis": {
                "rsi_14": 70.5,
                "rsi_assessment": "OVERBOUGHT",
            }
        }
        result = calculator.extract_momentum_signals(content)
        assert any("RSI" in signal.upper() for signal in result)

    def test_extracts_volume_signal(self, calculator):
        """Should extract volume trend signal."""
        content = {"volume_analysis": {"volume_trend": "INCREASING"}}
        result = calculator.extract_momentum_signals(content)
        assert any("volume" in signal.lower() for signal in result)


class TestScoreCalculatorStopLoss:
    """Tests for calculate_stop_loss method."""

    @pytest.fixture
    def calculator(self):
        """Create a fresh calculator for each test."""
        return ScoreCalculator()

    def test_basic_stop_loss(self, calculator):
        """Should calculate stop loss below current price."""
        result = calculator.calculate_stop_loss(100.0, {"recommendation": "HOLD"}, 7.0)
        assert result == 92.0  # 100 * (1 - 0.08)

    def test_buy_recommendation_wider_stop(self, calculator):
        """BUY recommendation should have wider stop (10%)."""
        result = calculator.calculate_stop_loss(100.0, {"recommendation": "BUY"}, 7.0)
        assert result == 90.0  # 100 * (1 - 0.10)

    def test_low_score_tighter_stop(self, calculator):
        """Low score (<4.0) should result in tighter stop loss."""
        result = calculator.calculate_stop_loss(100.0, {"recommendation": "HOLD"}, 3.0)
        assert result == 96.0  # 100 * (1 - 0.04)

    def test_zero_price_returns_zero(self, calculator):
        """Zero price should return zero stop loss."""
        result = calculator.calculate_stop_loss(0.0, {}, 7.0)
        assert result == 0.0


class TestScoreCalculatorSingleton:
    """Tests for singleton pattern."""

    def test_get_score_calculator_returns_same_instance(self):
        """Should return the same instance on multiple calls."""
        # Note: This test may be affected by other tests due to global state
        calc1 = get_score_calculator()
        calc2 = get_score_calculator()
        assert calc1 is calc2

    def test_custom_weights(self):
        """Should respect custom weights."""
        weights = ScoreWeights(fundamental_weight=0.7, technical_weight=0.3)
        calc = ScoreCalculator(weights)
        assert calc.weights.fundamental_weight == 0.7
        assert calc.weights.technical_weight == 0.3


class TestScoreCalculatorTechnicalIndicators:
    """Tests for extract_technical_indicators method."""

    @pytest.fixture
    def calculator(self):
        """Create a fresh calculator for each test."""
        return ScoreCalculator()

    def test_empty_responses_returns_empty_dict(self, calculator):
        """Empty responses should return empty dict."""
        result = calculator.extract_technical_indicators({})
        assert result == {}

    def test_structured_dict_response(self, calculator):
        """Should extract indicators from structured dict response."""
        llm_responses = {
            "technical": {
                "content": {
                    "technical_score": {"score": 7.5},
                    "trend_analysis": {
                        "primary_trend": "BULLISH",
                        "trend_strength": "STRONG",
                    },
                    "support_resistance": {
                        "immediate_support": 150.0,
                        "major_support": 145.0,
                        "immediate_resistance": 160.0,
                        "major_resistance": 165.0,
                    },
                    "recommendation": {
                        "technical_rating": "BUY",
                        "confidence": "HIGH",
                    },
                }
            }
        }
        result = calculator.extract_technical_indicators(llm_responses)

        assert result["technical_score"] == 7.5
        assert result["trend_direction"] == "BULLISH"
        assert result["trend_strength"] == "STRONG"
        assert result["support_levels"] == [150.0, 145.0]
        assert result["resistance_levels"] == [160.0, 165.0]
        assert result["recommendation"] == "BUY"
        assert result["confidence"] == "HIGH"
