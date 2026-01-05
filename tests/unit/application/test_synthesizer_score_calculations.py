"""
Regression tests for InvestmentSynthesizer score calculation methods.

These tests capture the current behavior before refactoring to ensure
no regressions occur when breaking up the monolithic synthesizer.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestCalculateFundamentalScore:
    """Tests for _calculate_fundamental_score method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with just the scoring method."""
        # Import the method directly to test in isolation
        from investigator.application.synthesizer import InvestmentSynthesizer

        # Create a minimal mock that has the method we want to test
        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._calculate_fundamental_score = InvestmentSynthesizer._calculate_fundamental_score.__get__(
            mock, InvestmentSynthesizer
        )
        return mock

    def test_empty_responses_returns_zero(self, mock_synthesizer):
        """Empty llm_responses should return 0.0."""
        result = mock_synthesizer._calculate_fundamental_score({})
        assert result == 0.0

    def test_no_fundamental_key_returns_zero(self, mock_synthesizer):
        """Missing 'fundamental' key should return 0.0."""
        result = mock_synthesizer._calculate_fundamental_score({"technical": {}})
        assert result == 0.0

    def test_comprehensive_dict_financial_health_score(self, mock_synthesizer):
        """Should extract financial_health_score from comprehensive dict response."""
        llm_responses = {
            "fundamental": {"comprehensive": {"content": {"financial_health_score": 7.5, "other_data": "value"}}}
        }
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 7.5

    def test_comprehensive_dict_overall_score(self, mock_synthesizer):
        """Should fallback to overall_score if financial_health_score missing."""
        llm_responses = {"fundamental": {"comprehensive": {"content": {"overall_score": 8.0, "other_data": "value"}}}}
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 8.0

    def test_comprehensive_string_json_response(self, mock_synthesizer):
        """Should parse JSON string response."""
        llm_responses = {"fundamental": {"comprehensive": {"content": json.dumps({"financial_health_score": 6.5})}}}
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 6.5

    def test_comprehensive_string_regex_fallback(self, mock_synthesizer):
        """Should extract score via regex from unstructured text."""
        llm_responses = {
            "fundamental": {"comprehensive": {"content": "The Financial Health Score: 7.2/10 based on analysis..."}}
        }
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 7.2

    def test_average_quarterly_scores(self, mock_synthesizer):
        """Should average multiple quarterly scores if no comprehensive."""
        llm_responses = {
            "fundamental": {
                "Q1_2024": {"content": {"financial_health_score": 6.0}},
                "Q2_2024": {"content": {"financial_health_score": 7.0}},
                "Q3_2024": {"content": {"financial_health_score": 8.0}},
            }
        }
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 7.0  # Average of 6, 7, 8

    def test_quarterly_string_responses(self, mock_synthesizer):
        """Should extract scores from quarterly string responses."""
        llm_responses = {
            "fundamental": {
                "Q1_2024": {"content": "Overall Score: 6.0/10"},
                "Q2_2024": {"content": "Financial Health: 8.0/10"},
            }
        }
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 7.0  # Average of 6 and 8

    def test_comprehensive_takes_priority_over_quarterly(self, mock_synthesizer):
        """Comprehensive score should be used even if quarterly exists."""
        llm_responses = {
            "fundamental": {
                "comprehensive": {"content": {"financial_health_score": 9.0}},
                "Q1_2024": {"content": {"financial_health_score": 5.0}},
            }
        }
        result = mock_synthesizer._calculate_fundamental_score(llm_responses)
        assert result == 9.0  # Comprehensive, not quarterly average


class TestCalculateTechnicalScore:
    """Tests for _calculate_technical_score method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with just the scoring method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._calculate_technical_score = InvestmentSynthesizer._calculate_technical_score.__get__(
            mock, InvestmentSynthesizer
        )
        return mock

    def test_empty_responses_returns_zero(self, mock_synthesizer):
        """Empty llm_responses should return 0.0."""
        result = mock_synthesizer._calculate_technical_score({})
        assert result == 0.0

    def test_no_technical_key_returns_zero(self, mock_synthesizer):
        """Missing 'technical' key should return 0.0."""
        result = mock_synthesizer._calculate_technical_score({"fundamental": {}})
        assert result == 0.0

    def test_dict_content_with_score_dict(self, mock_synthesizer):
        """Should extract score from nested dict format."""
        llm_responses = {"technical": {"content": {"technical_score": {"score": 7.5, "confidence": "high"}}}}
        result = mock_synthesizer._calculate_technical_score(llm_responses)
        assert result == 7.5

    def test_dict_content_with_score_float(self, mock_synthesizer):
        """Should handle direct float technical_score."""
        llm_responses = {"technical": {"content": {"technical_score": 8.0}}}
        result = mock_synthesizer._calculate_technical_score(llm_responses)
        assert result == 8.0

    def test_string_json_content(self, mock_synthesizer):
        """Should parse JSON string content."""
        llm_responses = {"technical": {"content": json.dumps({"technical_score": {"score": 6.5}})}}
        result = mock_synthesizer._calculate_technical_score(llm_responses)
        assert result == 6.5

    def test_string_with_ai_response_header(self, mock_synthesizer):
        """Should handle file format with === AI RESPONSE === header."""
        content = """=== METADATA ===
timestamp: 2024-01-01
=== AI RESPONSE ===
{"technical_score": {"score": 7.0}}"""
        llm_responses = {"technical": {"content": content}}
        result = mock_synthesizer._calculate_technical_score(llm_responses)
        assert result == 7.0

    def test_regex_fallback_for_legacy_format(self, mock_synthesizer):
        """Should extract score via regex for legacy format."""
        llm_responses = {"technical": {"content": "Analysis complete. TECHNICAL_SCORE: 6.8 out of 10"}}
        result = mock_synthesizer._calculate_technical_score(llm_responses)
        assert result == 6.8

    def test_invalid_json_returns_zero(self, mock_synthesizer):
        """Should return 0.0 for unparseable content without score pattern."""
        llm_responses = {"technical": {"content": "Some random text without scores"}}
        result = mock_synthesizer._calculate_technical_score(llm_responses)
        assert result == 0.0


class TestCalculateWeightedScore:
    """Tests for _calculate_weighted_score method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with the weighted score method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._calculate_weighted_score = InvestmentSynthesizer._calculate_weighted_score.__get__(
            mock, InvestmentSynthesizer
        )
        # Mock config with analysis weights
        mock.config = MagicMock()
        mock.config.analysis = MagicMock()
        mock.config.analysis.fundamental_weight = 0.6
        mock.config.analysis.technical_weight = 0.4
        return mock

    def test_equal_scores(self, mock_synthesizer):
        """Equal scores should return that score (weights sum to 1)."""
        result = mock_synthesizer._calculate_weighted_score(7.0, 7.0)
        assert result == 7.0

    def test_fundamental_higher(self, mock_synthesizer):
        """Weighted average with fundamental higher."""
        result = mock_synthesizer._calculate_weighted_score(8.0, 6.0)
        # Weights: fundamental 0.6, technical 0.4
        expected = 8.0 * 0.6 + 6.0 * 0.4  # 7.2
        assert abs(result - expected) < 0.2

    def test_technical_higher(self, mock_synthesizer):
        """Weighted average with technical higher."""
        result = mock_synthesizer._calculate_weighted_score(6.0, 8.0)
        expected = 6.0 * 0.6 + 8.0 * 0.4  # 6.8
        assert abs(result - expected) < 0.2

    def test_zero_scores(self, mock_synthesizer):
        """Both zero should return 0."""
        result = mock_synthesizer._calculate_weighted_score(0.0, 0.0)
        assert result == 0.0

    def test_boundary_scores(self, mock_synthesizer):
        """Test with boundary values (0 and 10)."""
        result = mock_synthesizer._calculate_weighted_score(10.0, 0.0)
        # Note: extreme scores get weight adjustment
        # 10.0 >= 8.5 so fund_weight *= 1.2 -> 0.72
        # total_weight = 0.72 + 0.4 = 1.12
        # norm_fund = 0.72/1.12 = 0.643
        # result = 10 * 0.643 = 6.43
        assert result > 5.5  # Should be above midpoint


class TestExtractTechnicalIndicators:
    """Tests for _extract_technical_indicators method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with the method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._extract_technical_indicators = InvestmentSynthesizer._extract_technical_indicators.__get__(
            mock, InvestmentSynthesizer
        )
        mock._extract_momentum_signals = InvestmentSynthesizer._extract_momentum_signals.__get__(
            mock, InvestmentSynthesizer
        )
        mock.main_logger = MagicMock()
        return mock

    def test_empty_responses_returns_empty_dict(self, mock_synthesizer):
        """Empty responses should return empty dict."""
        result = mock_synthesizer._extract_technical_indicators({})
        assert result == {}

    def test_no_technical_key_returns_empty(self, mock_synthesizer):
        """Missing technical key should return empty dict."""
        result = mock_synthesizer._extract_technical_indicators({"fundamental": {}})
        assert result == {}

    def test_structured_dict_response(self, mock_synthesizer):
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
                        "time_horizon": "SHORT",
                    },
                }
            }
        }
        result = mock_synthesizer._extract_technical_indicators(llm_responses)

        assert result["technical_score"] == 7.5
        assert result["trend_direction"] == "BULLISH"
        assert result["trend_strength"] == "STRONG"
        assert result["support_levels"] == [150.0, 145.0]
        assert result["resistance_levels"] == [160.0, 165.0]
        assert result["recommendation"] == "BUY"
        assert result["confidence"] == "HIGH"

    def test_json_string_response(self, mock_synthesizer):
        """Should parse JSON string response."""
        content = json.dumps(
            {
                "technical_score": 6.5,
                "trend_direction": "BEARISH",
                "trend_strength": "MODERATE",
                "support_levels": [100.0, 95.0],
                "resistance_levels": [110.0, 115.0],
                "recommendation": "SELL",
                "confidence": "MEDIUM",
            }
        )
        llm_responses = {"technical": {"content": content}}
        result = mock_synthesizer._extract_technical_indicators(llm_responses)

        assert result["technical_score"] == 6.5
        assert result["trend_direction"] == "BEARISH"
        assert result["recommendation"] == "SELL"

    def test_handles_missing_fields_gracefully(self, mock_synthesizer):
        """Should provide defaults for missing fields."""
        llm_responses = {"technical": {"content": {"technical_score": {"score": 5.0}}}}
        result = mock_synthesizer._extract_technical_indicators(llm_responses)

        assert result["technical_score"] == 5.0
        assert result["trend_direction"] == "NEUTRAL"  # Default
        assert result["trend_strength"] == "WEAK"  # Default


class TestExtractMomentumSignals:
    """Tests for _extract_momentum_signals method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with the method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._extract_momentum_signals = InvestmentSynthesizer._extract_momentum_signals.__get__(
            mock, InvestmentSynthesizer
        )
        return mock

    def test_empty_content_returns_empty_list(self, mock_synthesizer):
        """Empty content should return empty list."""
        result = mock_synthesizer._extract_momentum_signals({})
        assert result == []

    def test_extracts_rsi_signal(self, mock_synthesizer):
        """Should extract RSI from momentum_analysis."""
        # Actual structure uses momentum_analysis, not momentum_indicators
        content = {
            "momentum_analysis": {
                "rsi_14": 70.5,
                "rsi_assessment": "OVERBOUGHT",
            }
        }
        result = mock_synthesizer._extract_momentum_signals(content)
        assert any("RSI" in signal.upper() for signal in result)

    def test_extracts_macd_signal(self, mock_synthesizer):
        """Should extract MACD signal."""
        content = {"momentum_analysis": {"macd": {"signal": "BULLISH_CROSS"}}}
        result = mock_synthesizer._extract_momentum_signals(content)
        assert any("MACD" in signal.upper() for signal in result)

    def test_extracts_volume_signal(self, mock_synthesizer):
        """Should extract volume trend signal."""
        content = {"volume_analysis": {"volume_trend": "INCREASING"}}
        result = mock_synthesizer._extract_momentum_signals(content)
        assert any("volume" in signal.lower() for signal in result)


class TestStopLossCalculation:
    """Tests for _calculate_stop_loss method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with the method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._calculate_stop_loss = InvestmentSynthesizer._calculate_stop_loss.__get__(mock, InvestmentSynthesizer)
        return mock

    def test_basic_stop_loss(self, mock_synthesizer):
        """Should calculate stop loss below current price."""
        current_price = 100.0
        recommendation = {"recommendation": "HOLD"}
        overall_score = 7.0

        result = mock_synthesizer._calculate_stop_loss(current_price, recommendation, overall_score)

        # Stop loss should be below current price (HOLD = 8%)
        assert result < current_price
        assert result > 0
        assert result == 92.0  # 100 * (1 - 0.08)

    def test_buy_recommendation_wider_stop(self, mock_synthesizer):
        """BUY recommendation should have wider stop (10%)."""
        current_price = 100.0
        recommendation = {"recommendation": "BUY"}
        overall_score = 7.0

        result = mock_synthesizer._calculate_stop_loss(current_price, recommendation, overall_score)
        assert result == 90.0  # 100 * (1 - 0.10)

    def test_low_score_tighter_stop(self, mock_synthesizer):
        """Low score (<4.0) should result in tighter stop loss (halved %)."""
        current_price = 100.0
        recommendation = {"recommendation": "HOLD"}

        # Low score (< 4.0) triggers 0.5x multiplier
        low_score_stop = mock_synthesizer._calculate_stop_loss(current_price, recommendation, 3.0)
        # HOLD = 8%, but low score halves it to 4%
        assert low_score_stop == 96.0  # 100 * (1 - 0.04)

    def test_zero_price_returns_zero(self, mock_synthesizer):
        """Zero price should return zero stop loss."""
        result = mock_synthesizer._calculate_stop_loss(0.0, {}, 7.0)
        assert result == 0.0

    def test_sell_recommendation_tightest_stop(self, mock_synthesizer):
        """SELL recommendation should have tightest stop (5%)."""
        current_price = 100.0
        recommendation = {"recommendation": "SELL"}
        overall_score = 7.0

        result = mock_synthesizer._calculate_stop_loss(current_price, recommendation, overall_score)
        assert result == 95.0  # 100 * (1 - 0.05)


class TestDataQualityAssessment:
    """Tests for _assess_data_quality method."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with the method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._assess_data_quality = InvestmentSynthesizer._assess_data_quality.__get__(mock, InvestmentSynthesizer)
        return mock

    def test_empty_data_low_quality(self, mock_synthesizer):
        """Empty data should result in low quality score."""
        result = mock_synthesizer._assess_data_quality({}, {})
        assert result <= 0.5  # Low quality

    def test_complete_data_high_quality(self, mock_synthesizer):
        """Complete data should result in higher quality score."""
        llm_responses = {
            "fundamental": {"comprehensive": {"content": {"score": 7.0}}},
            "technical": {"content": {"score": 7.0}},
        }
        latest_data = {
            "current_price": 150.0,
            "volume": 1000000,
            "market_cap": 1e12,
        }
        result = mock_synthesizer._assess_data_quality(llm_responses, latest_data)
        assert result >= 0.5  # Reasonable quality


class TestParseSynthesisResponse:
    """Tests for _parse_synthesis_response method.

    Note: This method parses structured TEXT responses using regex,
    not JSON. The response format uses headers like:
    FINAL RECOMMENDATION: [BUY]
    CONFIDENCE LEVEL: [HIGH]
    """

    @pytest.fixture
    def mock_synthesizer(self):
        """Create a mock synthesizer with the method."""
        from investigator.application.synthesizer import InvestmentSynthesizer

        mock = MagicMock(spec=InvestmentSynthesizer)
        mock._parse_synthesis_response = InvestmentSynthesizer._parse_synthesis_response.__get__(
            mock, InvestmentSynthesizer
        )
        mock.main_logger = MagicMock()
        return mock

    def test_structured_text_response(self, mock_synthesizer):
        """Should parse structured text response with headers."""
        response = """
**FINAL RECOMMENDATION: [STRONG BUY]**

**CONFIDENCE LEVEL: [HIGH]**

**INVESTMENT THESIS:**
Strong fundamentals with growth potential.

**KEY CATALYSTS:**
- New product launch
- Market expansion

**12-month Target: $180.00**
"""
        result = mock_synthesizer._parse_synthesis_response(response)

        assert result["recommendation"] == "STRONG BUY"
        assert result["confidence"] == "HIGH"
        assert "12_month" in result["price_targets"]
        assert result["price_targets"]["12_month"] == 180.0

    def test_default_values_for_missing_fields(self, mock_synthesizer):
        """Should return default values for missing fields."""
        response = "Some random text without expected patterns"
        result = mock_synthesizer._parse_synthesis_response(response)

        assert isinstance(result, dict)
        assert result["recommendation"] == "HOLD"  # Default
        assert result["confidence"] == "MEDIUM"  # Default
        assert result["position_size"] == "MODERATE"  # Default

    def test_buy_recommendation_parsing(self, mock_synthesizer):
        """Should correctly identify BUY recommendation."""
        response = "FINAL RECOMMENDATION: BUY\nCONFIDENCE LEVEL: MEDIUM"
        result = mock_synthesizer._parse_synthesis_response(response)
        assert result["recommendation"] == "BUY"

    def test_sell_recommendation_parsing(self, mock_synthesizer):
        """Should correctly identify SELL recommendation."""
        response = "FINAL RECOMMENDATION: SELL\nCONFIDENCE LEVEL: HIGH"
        result = mock_synthesizer._parse_synthesis_response(response)
        assert result["recommendation"] == "SELL"

    def test_position_sizing_extraction(self, mock_synthesizer):
        """Should extract position sizing."""
        response = """
FINAL RECOMMENDATION: BUY
POSITION SIZING: LARGE
"""
        result = mock_synthesizer._parse_synthesis_response(response)
        assert result["position_size"] == "LARGE"
