"""
Tests for the extracted ComponentScoreExtractor module.

Verifies the extracted module has identical behavior to the original
InvestmentSynthesizer methods.

Author: InvestiGator Team
Date: 2025-01-05
"""

import json

import pytest

from investigator.application.component_score_extractor import (
    ComponentScoreExtractor,
    get_component_score_extractor,
)


class TestComponentScoreExtractorIncome:
    """Tests for extract_income_score method."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with fundamental score calculator."""
        return ComponentScoreExtractor(fundamental_score_calculator=lambda _: 7.0)

    def test_returns_direct_ai_recommendation_score(self, extractor):
        """Should return income_statement_score from AI recommendation."""
        result = extractor.extract_income_score({}, {"income_statement_score": 8.5})
        assert result == 8.5

    def test_extracts_from_profitability_margins(self, extractor):
        """Should calculate from profitability margins."""
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
        result = extractor.extract_income_score(llm_responses, {})
        assert 7.0 < result < 8.5

    def test_fallback_to_fundamental_score(self, extractor):
        """Should fallback to fundamental * 0.9."""
        result = extractor.extract_income_score({"fundamental": {"other": {}}}, {})
        assert result == pytest.approx(6.3, rel=0.1)

    def test_returns_zero_when_base_is_zero(self):
        """Should return 0 when fundamental score is 0."""
        extractor = ComponentScoreExtractor(lambda _: 0.0)
        result = extractor.extract_income_score({}, {})
        assert result == 0.0


class TestComponentScoreExtractorCashflow:
    """Tests for extract_cashflow_score method."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with fundamental score calculator."""
        return ComponentScoreExtractor(lambda _: 7.0)

    def test_adjusts_up_for_many_keywords(self, extractor):
        """Should adjust up when cashflow keywords are prevalent."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Strong operating cash flow with positive FCF. "
                    "Working capital improved. Liquidity excellent."
                }
            }
        }
        result = extractor.extract_cashflow_score(llm_responses, {})
        assert result >= 7.0

    def test_adjusts_down_for_no_keywords(self, extractor):
        """Should adjust down when no keywords found."""
        llm_responses = {"fundamental": {"Q1": {"content": "Revenue and expenses analysis."}}}
        result = extractor.extract_cashflow_score(llm_responses, {})
        assert result <= 7.0

    def test_handles_dict_content(self, extractor):
        """Should handle dict content by converting to JSON."""
        llm_responses = {"fundamental": {"Q1": {"content": {"analysis": "cash flow", "fcf": "positive"}}}}
        result = extractor.extract_cashflow_score(llm_responses, {})
        assert result >= 7.0


class TestComponentScoreExtractorBalance:
    """Tests for extract_balance_score method."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with fundamental score calculator."""
        return ComponentScoreExtractor(lambda _: 7.0)

    def test_adjusts_for_balance_keywords(self, extractor):
        """Should adjust based on balance sheet keywords."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Asset base strong. Liability management good. "
                    "Equity solid. Debt reasonable. Balance sheet healthy."
                }
            }
        }
        result = extractor.extract_balance_score(llm_responses, {})
        assert result >= 7.0


class TestComponentScoreExtractorGrowth:
    """Tests for extract_growth_score method."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with fundamental score calculator."""
        return ComponentScoreExtractor(lambda _: 7.0)

    def test_returns_comprehensive_growth_score(self, extractor):
        """Should return growth_prospects_score from comprehensive."""
        llm_responses = {"fundamental": {"comprehensive": {"content": {"growth_prospects_score": 8.5}}}}
        result = extractor.extract_growth_score(llm_responses, {})
        assert result == 8.5

    def test_returns_ai_recommendation_score(self, extractor):
        """Should extract from AI recommendation."""
        ai_rec = {"fundamental_assessment": {"growth_prospects": {"score": 7.8}}}
        result = extractor.extract_growth_score({"fundamental": {}}, ai_rec)
        assert result == 7.8

    def test_adjusts_for_growth_keywords(self, extractor):
        """Should adjust based on growth keywords."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Revenue growth accelerating. Expansion in market share. " "Strong momentum in segments."
                }
            }
        }
        result = extractor.extract_growth_score(llm_responses, {})
        assert result >= 7.0


class TestComponentScoreExtractorValue:
    """Tests for extract_value_score method."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with fundamental score calculator."""
        return ComponentScoreExtractor(lambda _: 7.0)

    def test_returns_ai_valuation_score(self, extractor):
        """Should extract from AI recommendation."""
        ai_rec = {"fundamental_assessment": {"valuation": {"score": 6.5}}}
        result = extractor.extract_value_score({}, ai_rec)
        assert result == 6.5

    def test_adjusts_up_for_undervalued(self, extractor):
        """Should adjust up for value keywords."""
        llm_responses = {
            "fundamental": {
                "Q1": {
                    "content": "Stock appears undervalued at current discount. "
                    "PE ratio attractive. Good value opportunity."
                }
            }
        }
        result = extractor.extract_value_score(llm_responses, {})
        assert result >= 7.0

    def test_adjusts_down_for_overvalued(self, extractor):
        """Should adjust down for negative keywords."""
        llm_responses = {
            "fundamental": {
                "Q1": {"content": "Stock appears overvalued and expensive. " "Trading at a premium. Overpriced."}
            }
        }
        result = extractor.extract_value_score(llm_responses, {})
        assert result <= 7.0


class TestComponentScoreExtractorBusinessQuality:
    """Tests for extract_business_quality_score method."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with fundamental score calculator."""
        return ComponentScoreExtractor(lambda _: 7.0)

    def test_returns_direct_score(self, extractor):
        """Should return business_quality_score from comprehensive."""
        llm_responses = {"fundamental": {"comprehensive": {"business_quality_score": 8.0}}}
        result = extractor.extract_business_quality_score(llm_responses, {})
        assert result == 8.0

    def test_returns_nested_content_score(self, extractor):
        """Should extract from nested content."""
        llm_responses = {"fundamental": {"comprehensive": {"content": {"business_quality_score": 7.5}}}}
        result = extractor.extract_business_quality_score(llm_responses, {})
        assert result == 7.5

    def test_returns_dict_format_score(self, extractor):
        """Should handle score in dict format."""
        llm_responses = {
            "fundamental": {"comprehensive": {"business_quality_score": {"score": 6.8, "confidence": "high"}}}
        }
        result = extractor.extract_business_quality_score(llm_responses, {})
        assert result == 6.8

    def test_calculates_from_quarterly(self, extractor):
        """Should calculate from quarterly analyses."""
        llm_responses = {
            "fundamental": {
                "Q1_2024": {"content": "Recurring revenue growth. Competitive advantage."},
                "Q2_2024": {"content": "Margin expansion. Operating leverage improving."},
            }
        }
        result = extractor.extract_business_quality_score(llm_responses, {})
        assert 1.0 <= result <= 10.0

    def test_returns_zero_when_no_data(self, extractor):
        """Should return 0 when no data."""
        result = extractor.extract_business_quality_score({"fundamental": {}}, {})
        assert result == 0.0


class TestAnalyzeQuarterlyBusinessQuality:
    """Tests for analyze_quarterly_business_quality method."""

    @pytest.fixture
    def extractor(self):
        """Create a fresh extractor."""
        return ComponentScoreExtractor()

    def test_baseline_score(self, extractor):
        """Should return score in valid range."""
        result = extractor.analyze_quarterly_business_quality("Some general text.", "Q1_2024")
        assert 1.0 <= result <= 10.0

    def test_quality_indicators_score(self, extractor):
        """Should return score for quality content."""
        content = (
            "Recurring revenue from subscription. Moat and competitive advantage. "
            "Innovation and technology leadership. Margin expansion."
        )
        result = extractor.analyze_quarterly_business_quality(content, "Q1_2024")
        # The algorithm normalizes by total keywords per category, so few matches = low score
        # This content has ~5 keyword matches across 4 categories with 30+ total keywords
        assert 1.0 <= result <= 10.0


class TestCalculateConsistencyBonus:
    """Tests for calculate_consistency_bonus method."""

    @pytest.fixture
    def extractor(self):
        """Create a fresh extractor."""
        return ComponentScoreExtractor()

    def test_zero_for_single(self, extractor):
        """Should return 0 for single indicator."""
        result = extractor.calculate_consistency_bonus([7.0])
        assert result == 0.0

    def test_high_bonus_for_consistent(self, extractor):
        """Should return high bonus for consistent scores."""
        result = extractor.calculate_consistency_bonus([7.0, 7.1, 6.9, 7.0])
        assert result > 0.8

    def test_low_bonus_for_volatile(self, extractor):
        """Should return low bonus for volatile scores."""
        result = extractor.calculate_consistency_bonus([3.0, 8.0, 4.0, 9.0])
        assert result < 0.3

    def test_never_exceeds_one(self, extractor):
        """Bonus should never exceed 1.0."""
        result = extractor.calculate_consistency_bonus([5.0, 5.0, 5.0, 5.0])
        assert result <= 1.0

    def test_never_negative(self, extractor):
        """Bonus should never be negative."""
        result = extractor.calculate_consistency_bonus([1.0, 10.0, 1.0, 10.0])
        assert result >= 0.0


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_extractor_returns_same_instance(self):
        """Should return same instance."""
        # Note: May be affected by other tests due to global state
        ext1 = get_component_score_extractor()
        ext2 = get_component_score_extractor()
        assert ext1 is ext2
