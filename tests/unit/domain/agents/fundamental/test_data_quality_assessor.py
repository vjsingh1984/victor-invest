"""
Tests for the extracted DataQualityAssessor module.

Verifies the extracted module has identical behavior to the original
FundamentalAnalysisAgent methods.

Author: InvestiGator Team
Date: 2025-01-05
"""

import pytest

from investigator.domain.agents.fundamental.data_quality_assessor import (
    DataQualityAssessor,
    get_data_quality_assessor,
)


class TestAssessDataQuality:
    """Tests for assess_data_quality method."""

    @pytest.fixture
    def assessor(self):
        """Create a fresh assessor for each test."""
        return DataQualityAssessor()

    def test_empty_data_returns_very_poor_quality(self, assessor):
        """Empty data should result in very poor quality grade."""
        result = assessor.assess_data_quality({}, {})

        assert result["quality_grade"] == "Very Poor"
        assert result["data_quality_score"] <= 40

    def test_returns_all_expected_keys(self, assessor):
        """Should return all expected keys in result."""
        result = assessor.assess_data_quality({"symbol": "TEST"}, {})

        expected_keys = [
            "data_quality_score",
            "quality_grade",
            "completeness_score",
            "consistency_score",
            "core_metrics_populated",
            "market_metrics_populated",
            "ratio_metrics_populated",
            "consistency_issues",
            "assessment",
            "extraction_quality",
            "quality_improvement",
            "improvement_sources",
            "enhancement_summary",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_market_data_increases_quality(self, assessor):
        """Adding market data should improve quality score."""
        company_data_base = {"symbol": "TEST", "financials": {"revenues": 1000000}}

        result_no_market = assessor.assess_data_quality(company_data_base, {})

        company_data_with_market = {
            **company_data_base,
            "market_data": {"current_price": 100, "market_cap": 5000000},
        }

        result_with_market = assessor.assess_data_quality(company_data_with_market, {})

        assert result_with_market["data_quality_score"] > result_no_market["data_quality_score"]
        assert result_with_market["market_metrics_populated"] == "2/2"

    def test_ratios_increase_quality(self, assessor):
        """Adding calculated ratios should improve quality score."""
        company_data = {"symbol": "TEST", "financials": {"revenues": 1000000}}

        result_no_ratios = assessor.assess_data_quality(company_data, {})

        ratios = {"pe_ratio": 15.0, "roe": 0.12, "current_ratio": 2.0}
        result_with_ratios = assessor.assess_data_quality(company_data, ratios)

        assert result_with_ratios["data_quality_score"] > result_no_ratios["data_quality_score"]

    def test_consistency_issues_detected(self, assessor):
        """Should detect and report consistency issues."""
        # Net loss exceeds revenue
        company_data = {
            "symbol": "TEST",
            "financials": {"revenues": 1000000, "net_income": -2000000},
        }

        result = assessor.assess_data_quality(company_data, {})

        assert len(result["consistency_issues"]) > 0
        assert result["consistency_score"] < 100


class TestCheckConsistency:
    """Tests for _check_consistency method."""

    @pytest.fixture
    def assessor(self):
        """Create a fresh assessor for each test."""
        return DataQualityAssessor()

    def test_no_issues_for_clean_data(self, assessor):
        """Clean data should have no consistency issues."""
        financials = {"revenues": 1000000, "net_income": 100000}
        ratios = {"current_ratio": 2.0}

        issues = assessor._check_consistency(financials, ratios)

        assert issues == []

    def test_detects_net_loss_exceeds_revenue(self, assessor):
        """Should detect when net loss exceeds revenue."""
        financials = {"revenues": 1000000, "net_income": -2000000}

        issues = assessor._check_consistency(financials, {})

        assert any("Net loss exceeds revenue" in issue for issue in issues)

    def test_detects_liabilities_exceed_assets(self, assessor):
        """Should detect current liabilities exceeding total assets."""
        financials = {
            "current_liabilities": 500000,
            "total_assets": 100000,
        }

        issues = assessor._check_consistency(financials, {})

        assert any("Current liabilities exceed" in issue for issue in issues)

    def test_detects_unrealistic_current_ratio(self, assessor):
        """Should detect impossibly high current ratio."""
        financials = {}
        ratios = {"current_ratio": 150}

        issues = assessor._check_consistency(financials, ratios)

        assert any("Unrealistic current ratio" in issue for issue in issues)


class TestScoreToGrade:
    """Tests for _score_to_grade method."""

    @pytest.fixture
    def assessor(self):
        """Create a fresh assessor for each test."""
        return DataQualityAssessor()

    def test_excellent_grade(self, assessor):
        """Score >= 90 should be Excellent."""
        assert assessor._score_to_grade(90) == "Excellent"
        assert assessor._score_to_grade(95) == "Excellent"
        assert assessor._score_to_grade(100) == "Excellent"

    def test_good_grade(self, assessor):
        """Score 75-89 should be Good."""
        assert assessor._score_to_grade(75) == "Good"
        assert assessor._score_to_grade(80) == "Good"
        assert assessor._score_to_grade(89) == "Good"

    def test_fair_grade(self, assessor):
        """Score 60-74 should be Fair."""
        assert assessor._score_to_grade(60) == "Fair"
        assert assessor._score_to_grade(65) == "Fair"
        assert assessor._score_to_grade(74) == "Fair"

    def test_poor_grade(self, assessor):
        """Score 40-59 should be Poor."""
        assert assessor._score_to_grade(40) == "Poor"
        assert assessor._score_to_grade(50) == "Poor"
        assert assessor._score_to_grade(59) == "Poor"

    def test_very_poor_grade(self, assessor):
        """Score < 40 should be Very Poor."""
        assert assessor._score_to_grade(39) == "Very Poor"
        assert assessor._score_to_grade(20) == "Very Poor"
        assert assessor._score_to_grade(0) == "Very Poor"


class TestCalculateConfidenceLevel:
    """Tests for calculate_confidence_level method."""

    @pytest.fixture
    def assessor(self):
        """Create a fresh assessor for each test."""
        return DataQualityAssessor()

    def test_very_high_confidence(self, assessor):
        """Excellent quality should yield very high confidence."""
        data_quality = {"data_quality_score": 95, "quality_grade": "Excellent", "consistency_issues": []}

        result = assessor.calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "VERY HIGH"
        assert result["confidence_score"] == 95

    def test_high_confidence(self, assessor):
        """Good quality should yield high confidence."""
        data_quality = {"data_quality_score": 80, "quality_grade": "Good", "consistency_issues": []}

        result = assessor.calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "HIGH"
        assert result["confidence_score"] == 85

    def test_moderate_confidence(self, assessor):
        """Fair quality should yield moderate confidence."""
        data_quality = {"data_quality_score": 65, "quality_grade": "Fair", "consistency_issues": []}

        result = assessor.calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "MODERATE"
        assert result["confidence_score"] == 70

    def test_low_confidence(self, assessor):
        """Poor quality should yield low confidence."""
        data_quality = {"data_quality_score": 45, "quality_grade": "Poor", "consistency_issues": []}

        result = assessor.calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "LOW"
        assert result["confidence_score"] == 50

    def test_very_low_confidence(self, assessor):
        """Very poor quality should yield very low confidence."""
        data_quality = {"data_quality_score": 30, "quality_grade": "Very Poor", "consistency_issues": []}

        result = assessor.calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "VERY LOW"
        assert result["confidence_score"] == 30

    def test_consistency_issues_reduce_confidence(self, assessor):
        """Consistency issues should reduce confidence by 10."""
        data_quality = {"data_quality_score": 80, "quality_grade": "Good", "consistency_issues": ["Issue 1"]}

        result = assessor.calculate_confidence_level(data_quality)

        assert result["confidence_score"] == 75  # 85 - 10
        assert "adjusted down" in result["rationale"]

    def test_returns_all_expected_keys(self, assessor):
        """Should return all expected keys in result."""
        data_quality = {"data_quality_score": 70, "quality_grade": "Fair", "consistency_issues": []}

        result = assessor.calculate_confidence_level(data_quality)

        expected_keys = ["confidence_level", "confidence_score", "rationale", "based_on_data_quality", "quality_grade"]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestAssessQuarterQuality:
    """Tests for assess_quarter_quality method."""

    @pytest.fixture
    def assessor(self):
        """Create a fresh assessor for each test."""
        return DataQualityAssessor()

    def test_empty_data_zero_completeness(self, assessor):
        """Empty financial data should have zero completeness."""
        result = assessor.assess_quarter_quality({})

        assert result["completeness"] == 0.0

    def test_complete_data_full_completeness(self, assessor):
        """Complete data should have 100% completeness."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 200000,
            "stockholders_equity": 300000,
            "operating_cash_flow": 120000,
            "capital_expenditures": 50000,
        }

        result = assessor.assess_quarter_quality(financial_data)

        assert result["completeness"] == 100.0

    def test_partial_completeness(self, assessor):
        """Partial data should have proportional completeness."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
        }

        result = assessor.assess_quarter_quality(financial_data)

        # 3 of 7 fields = ~42.86%
        assert 42 <= result["completeness"] <= 43

    def test_balance_sheet_mismatch_detected(self, assessor):
        """Should detect balance sheet mismatch."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 100000,  # Mismatch
            "stockholders_equity": 300000,
        }

        result = assessor.assess_quarter_quality(financial_data)

        assert result["consistency"] < 100
        assert any("Balance sheet mismatch" in issue for issue in result["issues"])

    def test_zero_revenue_detected(self, assessor):
        """Should detect zero or negative revenue."""
        financial_data = {"revenues": 0}

        result = assessor.assess_quarter_quality(financial_data)

        assert any("Zero or negative revenue" in issue for issue in result["issues"])

    def test_unusual_cash_conversion_detected(self, assessor):
        """Should detect unusual cash conversion ratio."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "operating_cash_flow": 10000,  # Low OCF/NI
        }

        result = assessor.assess_quarter_quality(financial_data)

        assert any("Unusual cash conversion" in issue for issue in result["issues"])

    def test_consistency_non_negative(self, assessor):
        """Consistency score should never go below zero."""
        financial_data = {
            "revenues": 0,
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 100000,  # Mismatch
            "stockholders_equity": 300000,
            "operating_cash_flow": 10000,  # Low OCF/NI
        }

        result = assessor.assess_quarter_quality(financial_data)

        assert result["consistency"] >= 0

    def test_returns_all_expected_keys(self, assessor):
        """Should return all expected keys in result."""
        result = assessor.assess_quarter_quality({})

        expected_keys = ["completeness", "consistency", "issues"]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_assessor_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        a1 = get_data_quality_assessor()
        a2 = get_data_quality_assessor()
        assert a1 is a2
