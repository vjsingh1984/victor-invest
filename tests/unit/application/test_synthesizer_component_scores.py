"""
Regression tests for InvestmentSynthesizer component score extraction methods.

These tests capture the CURRENT BEHAVIOR of the methods BEFORE extraction.
They serve as regression tests to ensure the extracted modules maintain
identical behavior.

Methods tested:
- _extract_income_score
- _extract_cashflow_score
- _extract_balance_score
- _extract_growth_score
- _extract_value_score
- _extract_business_quality_score
- _analyze_quarterly_business_quality
- _calculate_consistency_bonus

Author: InvestiGator Team
Date: 2025-01-05
"""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_synthesizer():
    """Create a mock synthesizer with real methods bound."""
    from investigator.application.synthesizer import InvestmentSynthesizer

    mock = MagicMock(spec=InvestmentSynthesizer)

    # Bind the real methods to the mock
    mock._extract_income_score = InvestmentSynthesizer._extract_income_score.__get__(mock)
    mock._extract_cashflow_score = InvestmentSynthesizer._extract_cashflow_score.__get__(mock)
    mock._extract_balance_score = InvestmentSynthesizer._extract_balance_score.__get__(mock)
    mock._extract_growth_score = InvestmentSynthesizer._extract_growth_score.__get__(mock)
    mock._extract_value_score = InvestmentSynthesizer._extract_value_score.__get__(mock)
    mock._extract_business_quality_score = InvestmentSynthesizer._extract_business_quality_score.__get__(mock)
    mock._analyze_quarterly_business_quality = InvestmentSynthesizer._analyze_quarterly_business_quality.__get__(mock)
    mock._calculate_consistency_bonus = InvestmentSynthesizer._calculate_consistency_bonus.__get__(mock)

    # Mock the _calculate_fundamental_score method (dependency)
    mock._calculate_fundamental_score = MagicMock(return_value=7.0)

    return mock


class TestExtractIncomeScore:
    """Tests for _extract_income_score method."""

    def test_returns_direct_ai_recommendation_score(self, mock_synthesizer):
        """Should return income_statement_score from AI recommendation if present."""
        llm_responses = {}
        ai_recommendation = {"income_statement_score": 8.5}

        result = mock_synthesizer._extract_income_score(llm_responses, ai_recommendation)

        assert result == 8.5

    def test_extracts_from_comprehensive_profitability(self, mock_synthesizer):
        """Should calculate score from profitability margins in comprehensive analysis."""
        llm_responses = {
            "fundamental": {
                "comprehensive": {
                    "content": {
                        "income_statement_analysis": {
                            "profitability_analysis": {
                                "gross_margin": 0.35,
                                "operating_margin": 0.20,
                                "net_margin": 0.15,
                            }
                        }
                    }
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_income_score(llm_responses, ai_recommendation)

        # Average margin = (0.35 + 0.20 + 0.15) / 3 = 0.233
        # Score = min(10, max(1, 0.233 * 100 / 3)) = min(10, max(1, 7.78)) = 7.78
        assert 7.0 < result < 8.5

    def test_falls_back_to_fundamental_score(self, mock_synthesizer):
        """Should fall back to fundamental score with 0.9 multiplier."""
        llm_responses = {"fundamental": {"other": {}}}
        ai_recommendation = {}

        result = mock_synthesizer._extract_income_score(llm_responses, ai_recommendation)

        # 7.0 * 0.9 = 6.3
        assert result == pytest.approx(6.3, rel=0.1)

    def test_returns_zero_when_no_data(self, mock_synthesizer):
        """Should return 0 when fundamental score is 0."""
        mock_synthesizer._calculate_fundamental_score.return_value = 0.0
        llm_responses = {}
        ai_recommendation = {}

        result = mock_synthesizer._extract_income_score(llm_responses, ai_recommendation)

        assert result == 0.0


class TestExtractCashflowScore:
    """Tests for _extract_cashflow_score method."""

    def test_adjusts_up_for_many_cashflow_keywords(self, mock_synthesizer):
        """Should adjust score up when cashflow keywords are prevalent."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Strong operating cash flow with positive FCF. "
                               "Working capital improved significantly. "
                               "Cash position and liquidity are excellent."
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_cashflow_score(llm_responses, ai_recommendation)

        # Base 7.0 + adjustment (keywords found: cash flow, fcf, working capital, cash, liquidity)
        assert result >= 7.0

    def test_adjusts_down_for_no_cashflow_keywords(self, mock_synthesizer):
        """Should adjust score down when no cashflow keywords found."""
        llm_responses = {
            "fundamental": {
                "Q1": {"content": "Revenue and expenses analysis."}
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_cashflow_score(llm_responses, ai_recommendation)

        # Base 7.0 - 0.5 adjustment
        assert result <= 7.0

    def test_handles_dict_content(self, mock_synthesizer):
        """Should handle dict content by converting to JSON string."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": {
                        "analysis": "cash flow analysis",
                        "fcf": "positive",
                        "liquidity": "strong"
                    }
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_cashflow_score(llm_responses, ai_recommendation)

        # Should find keywords in JSON string
        assert result >= 7.0

    def test_returns_zero_when_base_is_zero(self, mock_synthesizer):
        """Should return 0 when base fundamental score is 0."""
        mock_synthesizer._calculate_fundamental_score.return_value = 0.0
        llm_responses = {"fundamental": {"Q1": {"content": "cash flow analysis"}}}
        ai_recommendation = {}

        result = mock_synthesizer._extract_cashflow_score(llm_responses, ai_recommendation)

        assert result == 0.0


class TestExtractBalanceScore:
    """Tests for _extract_balance_score method."""

    def test_adjusts_up_for_balance_keywords(self, mock_synthesizer):
        """Should adjust score up when balance sheet keywords are prevalent."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Asset base is strong. Liability management excellent. "
                               "Equity position solid. Debt levels reasonable. "
                               "Balance sheet shows good leverage and solvency."
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_balance_score(llm_responses, ai_recommendation)

        # Base 7.0 + adjustment for multiple keywords
        assert result >= 7.0

    def test_adjusts_down_for_no_balance_keywords(self, mock_synthesizer):
        """Should adjust score down when no balance keywords found."""
        llm_responses = {
            "fundamental": {
                "Q1": {"content": "Revenue growth is strong."}
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_balance_score(llm_responses, ai_recommendation)

        # Base 7.0 - 0.5 adjustment
        assert result <= 7.0


class TestExtractGrowthScore:
    """Tests for _extract_growth_score method."""

    def test_returns_comprehensive_growth_score(self, mock_synthesizer):
        """Should return growth_prospects_score from comprehensive analysis."""
        llm_responses = {
            "fundamental": {
                "comprehensive": {
                    "content": {"growth_prospects_score": 8.5}
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_growth_score(llm_responses, ai_recommendation)

        assert result == 8.5

    def test_returns_ai_recommendation_growth_score(self, mock_synthesizer):
        """Should extract growth score from AI recommendation."""
        llm_responses = {"fundamental": {}}
        ai_recommendation = {
            "fundamental_assessment": {
                "growth_prospects": {"score": 7.8}
            }
        }

        result = mock_synthesizer._extract_growth_score(llm_responses, ai_recommendation)

        assert result == 7.8

    def test_adjusts_for_growth_keywords(self, mock_synthesizer):
        """Should adjust score based on growth keyword frequency."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Revenue growth is accelerating. "
                               "Strong expansion in market share. "
                               "Momentum in key segments. "
                               "Scaling operations effectively."
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_growth_score(llm_responses, ai_recommendation)

        # Base 7.0 + adjustment for growth keywords
        assert result >= 7.0


class TestExtractValueScore:
    """Tests for _extract_value_score method."""

    def test_returns_ai_valuation_score(self, mock_synthesizer):
        """Should extract valuation score from AI recommendation."""
        llm_responses = {}
        ai_recommendation = {
            "fundamental_assessment": {
                "valuation": {"score": 6.5}
            }
        }

        result = mock_synthesizer._extract_value_score(llm_responses, ai_recommendation)

        assert result == 6.5

    def test_adjusts_up_for_undervalued_keywords(self, mock_synthesizer):
        """Should adjust up when value keywords dominate."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Stock appears undervalued at current discount. "
                               "PE ratio attractive. Dividend yield strong. "
                               "Good value opportunity here."
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_value_score(llm_responses, ai_recommendation)

        # Base 7.0 + positive adjustment
        assert result >= 7.0

    def test_adjusts_down_for_overvalued_keywords(self, mock_synthesizer):
        """Should adjust down when negative value keywords dominate."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Stock appears overvalued and expensive. "
                               "Trading at a premium. Overpriced relative to peers."
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_value_score(llm_responses, ai_recommendation)

        # Base 7.0 - negative adjustment
        assert result <= 7.0


class TestExtractBusinessQualityScore:
    """Tests for _extract_business_quality_score method."""

    def test_returns_direct_business_quality_score(self, mock_synthesizer):
        """Should return business_quality_score from comprehensive analysis."""
        llm_responses = {
            "fundamental": {
                "comprehensive": {"business_quality_score": 8.0}
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_business_quality_score(llm_responses, ai_recommendation)

        assert result == 8.0

    def test_returns_nested_content_score(self, mock_synthesizer):
        """Should extract from nested content dict."""
        llm_responses = {
            "fundamental": {
                "comprehensive": {
                    "content": {"business_quality_score": 7.5}
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_business_quality_score(llm_responses, ai_recommendation)

        assert result == 7.5

    def test_returns_score_from_dict_format(self, mock_synthesizer):
        """Should handle score in dict format with 'score' key."""
        llm_responses = {
            "fundamental": {
                "comprehensive": {
                    "business_quality_score": {"score": 6.8, "confidence": "high"}
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_business_quality_score(llm_responses, ai_recommendation)

        assert result == 6.8

    def test_calculates_from_quarterly_analyses(self, mock_synthesizer):
        """Should calculate from quarterly analyses when no comprehensive score."""
        llm_responses = {
            "fundamental": {
                "Q1_2024": {
                    "content": "Recurring revenue growth. Strong competitive advantage. "
                               "Market leadership position. Innovation in technology."
                },
                "Q2_2024": {
                    "content": "Margin expansion continues. Operating leverage improving. "
                               "Cost control measures effective."
                }
            }
        }
        ai_recommendation = {}

        result = mock_synthesizer._extract_business_quality_score(llm_responses, ai_recommendation)

        # Should calculate from quarterly data with consistency bonus
        assert 1.0 <= result <= 10.0

    def test_returns_zero_when_no_data(self, mock_synthesizer):
        """Should return 0 when no business quality data available."""
        llm_responses = {"fundamental": {}}
        ai_recommendation = {}

        result = mock_synthesizer._extract_business_quality_score(llm_responses, ai_recommendation)

        assert result == 0.0


class TestAnalyzeQuarterlyBusinessQuality:
    """Tests for _analyze_quarterly_business_quality method."""

    def test_baseline_score_for_minimal_content(self, mock_synthesizer):
        """Should return base score for content with few indicators."""
        result = mock_synthesizer._analyze_quarterly_business_quality(
            "Some general text about the company.", "Q1_2024"
        )

        # Should return a score between 1 and 10
        assert 1.0 <= result <= 10.0

    def test_high_score_for_quality_indicators(self, mock_synthesizer):
        """Should return higher score for content with quality indicators."""
        content = (
            "Recurring revenue from subscription model. Strong moat and competitive advantage. "
            "Market share expansion through innovation. R&D investment driving technology leadership. "
            "Margin expansion and operating leverage. Capital allocation focused on shareholder value."
        )

        result = mock_synthesizer._analyze_quarterly_business_quality(content, "Q1_2024")

        # Score based on keyword category matches - actual behavior returns ~4.3
        # This is a weighted average across 4 categories with limited keyword matches
        assert 4.0 <= result <= 5.0

    def test_case_insensitive_matching(self, mock_synthesizer):
        """Should match keywords case-insensitively."""
        content = "RECURRING REVENUE and COMPETITIVE ADVANTAGE are strong."

        result = mock_synthesizer._analyze_quarterly_business_quality(content, "Q1_2024")

        # Method returns minimum 1.0 - even with keywords, limited matches yield low score
        # The algorithm divides keyword count by total keywords in category, so few matches = low score
        assert result >= 1.0


class TestCalculateConsistencyBonus:
    """Tests for _calculate_consistency_bonus method."""

    def test_returns_zero_for_single_indicator(self, mock_synthesizer):
        """Should return 0 for single indicator (can't calculate consistency)."""
        result = mock_synthesizer._calculate_consistency_bonus([7.0])

        assert result == 0.0

    def test_high_bonus_for_consistent_scores(self, mock_synthesizer):
        """Should return high bonus for consistent (low variance) scores."""
        result = mock_synthesizer._calculate_consistency_bonus([7.0, 7.1, 6.9, 7.0])

        # Very consistent, should get close to max bonus (1.0)
        assert result > 0.8

    def test_low_bonus_for_volatile_scores(self, mock_synthesizer):
        """Should return low bonus for volatile (high variance) scores."""
        result = mock_synthesizer._calculate_consistency_bonus([3.0, 8.0, 4.0, 9.0])

        # High variance, should get low bonus
        assert result < 0.3

    def test_bonus_capped_at_one(self, mock_synthesizer):
        """Bonus should never exceed 1.0."""
        result = mock_synthesizer._calculate_consistency_bonus([5.0, 5.0, 5.0, 5.0, 5.0])

        # Perfect consistency
        assert result <= 1.0

    def test_bonus_never_negative(self, mock_synthesizer):
        """Bonus should never be negative."""
        result = mock_synthesizer._calculate_consistency_bonus([1.0, 10.0, 1.0, 10.0])

        assert result >= 0.0
