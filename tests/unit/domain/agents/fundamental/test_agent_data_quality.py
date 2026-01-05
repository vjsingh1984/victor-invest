"""
Regression tests for data quality methods in FundamentalAnalysisAgent.

These tests capture the EXACT behavior of the original methods BEFORE extraction.
Run these tests before AND after extracting DataQualityAssessor to ensure no regressions.

Methods tested:
- _assess_data_quality: Comprehensive data quality assessment
- _calculate_confidence_level: Maps quality to confidence
- _assess_quarter_quality: Single quarter quality assessment

Author: InvestiGator Team
Date: 2025-01-05
"""

import pytest
from unittest.mock import MagicMock, patch


class TestAssessDataQuality:
    """Regression tests for _assess_data_quality method."""

    @pytest.fixture
    def agent(self):
        """Create agent instance with mocked dependencies."""
        with patch(
            "investigator.domain.agents.fundamental.agent.FundamentalAnalysisAgent.__init__",
            return_value=None,
        ):
            from investigator.domain.agents.fundamental.agent import (
                FundamentalAnalysisAgent,
            )

            agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
            agent.logger = MagicMock()
            return agent

    def test_empty_data_returns_very_poor_quality(self, agent):
        """Empty data should result in very poor quality grade."""
        result = agent._assess_data_quality({}, {})

        assert result["quality_grade"] == "Very Poor"
        assert result["data_quality_score"] <= 40

    def test_complete_financials_good_quality(self, agent):
        """Complete financial data should yield fair or better quality."""
        company_data = {
            "symbol": "TEST",
            "financials": {
                "revenues": 1000000,
                "net_income": 100000,
                "total_assets": 500000,
                "total_liabilities": 200000,
                "stockholders_equity": 300000,
                "operating_cash_flow": 150000,
                "capital_expenditures": 50000,
                "current_assets": 200000,
                "current_liabilities": 100000,
                "long_term_debt": 100000,
                "short_term_debt": 50000,
            },
            "market_data": {
                "current_price": 150.0,
                "market_cap": 10000000,
            },
        }
        ratios = {
            "pe_ratio": 15.0,
            "price_to_book": 2.0,
            "current_ratio": 2.0,
            "debt_to_equity": 0.5,
            "roe": 0.15,
            "roa": 0.10,
            "gross_margin": 0.40,
            "operating_margin": 0.20,
        }

        result = agent._assess_data_quality(company_data, ratios)

        # Note: Quality grade is "Fair" due to DataNormalizer's core metrics check
        # which requires specific fields not all provided here
        assert result["quality_grade"] in ["Fair", "Good", "Excellent"]
        assert result["data_quality_score"] >= 60
        assert result["consistency_score"] == 100

    def test_detects_net_loss_exceeds_revenue(self, agent):
        """Should detect when net loss exceeds revenue."""
        company_data = {
            "symbol": "TEST",
            "financials": {
                "revenues": 1000000,
                "net_income": -2000000,  # Net loss > revenue
            },
        }

        result = agent._assess_data_quality(company_data, {})

        assert "Net loss exceeds revenue" in str(result["consistency_issues"])
        assert result["consistency_score"] < 100

    def test_detects_current_liabilities_exceed_assets(self, agent):
        """Should detect current liabilities exceeding total assets."""
        company_data = {
            "symbol": "TEST",
            "financials": {
                "current_liabilities": 500000,
                "total_assets": 100000,  # Liabilities > Assets
            },
        }

        result = agent._assess_data_quality(company_data, {})

        assert any("Current liabilities exceed" in issue for issue in result["consistency_issues"])

    def test_detects_unrealistic_current_ratio(self, agent):
        """Should detect impossibly high current ratio."""
        company_data = {"symbol": "TEST", "financials": {}}
        ratios = {"current_ratio": 150}  # Impossibly high

        result = agent._assess_data_quality(company_data, ratios)

        assert any("Unrealistic current ratio" in issue for issue in result["consistency_issues"])

    def test_calculates_enhancement_summary(self, agent):
        """Should calculate enhancement summary for enriched data."""
        company_data = {
            "symbol": "TEST",
            "financials": {"revenues": 1000000},
            "market_data": {"current_price": 100, "market_cap": 5000000},
        }
        ratios = {"pe_ratio": 15.0, "roe": 0.12}

        result = agent._assess_data_quality(company_data, ratios)

        assert "enhancement_summary" in result
        assert result["quality_improvement"] >= 0
        assert "improvement_sources" in result

    def test_returns_all_expected_keys(self, agent):
        """Should return all expected keys in result."""
        result = agent._assess_data_quality({"symbol": "TEST"}, {})

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

    def test_quality_grades_mapping(self, agent):
        """Test quality grade mappings at boundaries."""
        # Test Poor grade (40-60)
        company_data = {
            "symbol": "TEST",
            "financials": {"revenues": 1000000, "net_income": 50000},
        }

        result = agent._assess_data_quality(company_data, {})

        # Should be in the lower quality grades due to missing data
        assert result["quality_grade"] in ["Very Poor", "Poor", "Fair"]

    def test_market_data_completeness(self, agent):
        """Should correctly calculate market data completeness."""
        # Only price
        company_data = {
            "symbol": "TEST",
            "market_data": {"current_price": 100},
        }

        result = agent._assess_data_quality(company_data, {})

        assert result["market_metrics_populated"] == "1/2"

        # Both price and market cap
        company_data["market_data"]["market_cap"] = 1000000

        result = agent._assess_data_quality(company_data, {})

        assert result["market_metrics_populated"] == "2/2"


class TestCalculateConfidenceLevel:
    """Regression tests for _calculate_confidence_level method."""

    @pytest.fixture
    def agent(self):
        """Create agent instance with mocked dependencies."""
        with patch(
            "investigator.domain.agents.fundamental.agent.FundamentalAnalysisAgent.__init__",
            return_value=None,
        ):
            from investigator.domain.agents.fundamental.agent import (
                FundamentalAnalysisAgent,
            )

            agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
            agent.logger = MagicMock()
            return agent

    def test_very_high_confidence_for_excellent_quality(self, agent):
        """Excellent quality should yield very high confidence."""
        data_quality = {
            "data_quality_score": 95,
            "quality_grade": "Excellent",
            "consistency_issues": [],
        }

        result = agent._calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "VERY HIGH"
        assert result["confidence_score"] == 95

    def test_high_confidence_for_good_quality(self, agent):
        """Good quality should yield high confidence."""
        data_quality = {
            "data_quality_score": 80,
            "quality_grade": "Good",
            "consistency_issues": [],
        }

        result = agent._calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "HIGH"
        assert result["confidence_score"] == 85

    def test_moderate_confidence_for_fair_quality(self, agent):
        """Fair quality should yield moderate confidence."""
        data_quality = {
            "data_quality_score": 65,
            "quality_grade": "Fair",
            "consistency_issues": [],
        }

        result = agent._calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "MODERATE"
        assert result["confidence_score"] == 70

    def test_low_confidence_for_poor_quality(self, agent):
        """Poor quality should yield low confidence."""
        data_quality = {
            "data_quality_score": 45,
            "quality_grade": "Poor",
            "consistency_issues": [],
        }

        result = agent._calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "LOW"
        assert result["confidence_score"] == 50

    def test_very_low_confidence_for_very_poor_quality(self, agent):
        """Very poor quality should yield very low confidence."""
        data_quality = {
            "data_quality_score": 30,
            "quality_grade": "Very Poor",
            "consistency_issues": [],
        }

        result = agent._calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "VERY LOW"
        assert result["confidence_score"] == 30

    def test_confidence_adjusted_down_for_consistency_issues(self, agent):
        """Confidence should be reduced when there are consistency issues."""
        data_quality = {
            "data_quality_score": 80,
            "quality_grade": "Good",
            "consistency_issues": ["Balance sheet mismatch"],
        }

        result = agent._calculate_confidence_level(data_quality)

        assert result["confidence_level"] == "HIGH"
        assert result["confidence_score"] == 75  # 85 - 10
        assert "adjusted down" in result["rationale"]

    def test_multiple_consistency_issues_adjustment(self, agent):
        """Multiple consistency issues should reduce confidence once (by 10)."""
        data_quality = {
            "data_quality_score": 80,
            "quality_grade": "Good",
            "consistency_issues": ["Issue 1", "Issue 2", "Issue 3"],
        }

        result = agent._calculate_confidence_level(data_quality)

        # Adjustment is -10 regardless of number of issues
        assert result["confidence_score"] == 75
        assert "3 data consistency issue" in result["rationale"]

    def test_returns_all_expected_keys(self, agent):
        """Should return all expected keys in result."""
        data_quality = {
            "data_quality_score": 70,
            "quality_grade": "Fair",
            "consistency_issues": [],
        }

        result = agent._calculate_confidence_level(data_quality)

        expected_keys = [
            "confidence_level",
            "confidence_score",
            "rationale",
            "based_on_data_quality",
            "quality_grade",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_boundary_scores(self, agent):
        """Test confidence levels at exact boundaries."""
        # Exactly 90
        result = agent._calculate_confidence_level(
            {"data_quality_score": 90, "quality_grade": "Excellent", "consistency_issues": []}
        )
        assert result["confidence_level"] == "VERY HIGH"

        # Exactly 75
        result = agent._calculate_confidence_level(
            {"data_quality_score": 75, "quality_grade": "Good", "consistency_issues": []}
        )
        assert result["confidence_level"] == "HIGH"

        # Exactly 60
        result = agent._calculate_confidence_level(
            {"data_quality_score": 60, "quality_grade": "Fair", "consistency_issues": []}
        )
        assert result["confidence_level"] == "MODERATE"

        # Exactly 40
        result = agent._calculate_confidence_level(
            {"data_quality_score": 40, "quality_grade": "Poor", "consistency_issues": []}
        )
        assert result["confidence_level"] == "LOW"


class TestAssessQuarterQuality:
    """Regression tests for _assess_quarter_quality method."""

    @pytest.fixture
    def agent(self):
        """Create agent instance with mocked dependencies."""
        with patch(
            "investigator.domain.agents.fundamental.agent.FundamentalAnalysisAgent.__init__",
            return_value=None,
        ):
            from investigator.domain.agents.fundamental.agent import (
                FundamentalAnalysisAgent,
            )

            agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
            agent.logger = MagicMock()
            return agent

    def test_empty_data_zero_completeness(self, agent):
        """Empty financial data should have zero completeness."""
        result = agent._assess_quarter_quality({})

        assert result["completeness"] == 0.0
        assert result["consistency"] < 100  # Should have issues

    def test_complete_data_full_completeness(self, agent):
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

        result = agent._assess_quarter_quality(financial_data)

        assert result["completeness"] == 100.0

    def test_partial_data_partial_completeness(self, agent):
        """Partial data should have proportional completeness."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
            # Missing: total_liabilities, stockholders_equity, operating_cash_flow, capital_expenditures
        }

        result = agent._assess_quarter_quality(financial_data)

        # 3 of 7 fields = ~42.86%
        assert 42 <= result["completeness"] <= 43

    def test_balance_sheet_mismatch_detected(self, agent):
        """Should detect balance sheet mismatch (assets != liabilities + equity)."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 100000,  # Should be 200000
            "stockholders_equity": 300000,
            "operating_cash_flow": 120000,
            "capital_expenditures": 50000,
        }

        result = agent._assess_quarter_quality(financial_data)

        assert result["consistency"] < 100
        assert any("Balance sheet mismatch" in issue for issue in result["issues"])

    def test_zero_revenue_detected(self, agent):
        """Should detect zero or negative revenue."""
        financial_data = {
            "revenues": 0,
            "net_income": 100000,
        }

        result = agent._assess_quarter_quality(financial_data)

        assert result["consistency"] < 100
        assert any("Zero or negative revenue" in issue for issue in result["issues"])

    def test_unusual_cash_conversion_detected(self, agent):
        """Should detect unusual cash conversion ratio."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "operating_cash_flow": 10000,  # OCF/NI = 0.1, below 0.3 threshold
        }

        result = agent._assess_quarter_quality(financial_data)

        assert result["consistency"] < 100
        assert any("Unusual cash conversion" in issue for issue in result["issues"])

    def test_high_cash_conversion_detected(self, agent):
        """Should detect unusually high cash conversion ratio."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "operating_cash_flow": 600000,  # OCF/NI = 6.0, above 5.0 threshold
        }

        result = agent._assess_quarter_quality(financial_data)

        assert result["consistency"] < 100
        assert any("Unusual cash conversion" in issue for issue in result["issues"])

    def test_consistency_score_non_negative(self, agent):
        """Consistency score should never go below zero."""
        financial_data = {
            "revenues": 0,  # -30
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 100000,  # Balance mismatch: -20
            "stockholders_equity": 300000,
            "operating_cash_flow": 10000,  # Unusual conversion: -15
        }

        result = agent._assess_quarter_quality(financial_data)

        assert result["consistency"] >= 0

    def test_returns_all_expected_keys(self, agent):
        """Should return all expected keys in result."""
        result = agent._assess_quarter_quality({})

        expected_keys = ["completeness", "consistency", "issues"]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_balanced_balance_sheet_no_issues(self, agent):
        """Balanced balance sheet should not create issues."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 200000,
            "stockholders_equity": 300000,  # 200000 + 300000 = 500000
            "operating_cash_flow": 120000,  # OCF/NI = 1.2, normal
            "capital_expenditures": 50000,
        }

        result = agent._assess_quarter_quality(financial_data)

        assert result["consistency"] == 100
        assert result["issues"] == []

    def test_within_tolerance_balance_sheet(self, agent):
        """Balance sheet within 5% tolerance should be acceptable."""
        financial_data = {
            "revenues": 1000000,
            "net_income": 100000,
            "total_assets": 500000,
            "total_liabilities": 198000,  # 2% off
            "stockholders_equity": 300000,  # Total: 498000 (0.4% error)
            "operating_cash_flow": 120000,
            "capital_expenditures": 50000,
        }

        result = agent._assess_quarter_quality(financial_data)

        # Should NOT trigger balance sheet mismatch (within 5%)
        assert not any("Balance sheet mismatch" in issue for issue in result["issues"])


class TestDataQualityIntegration:
    """Integration tests for data quality flow."""

    @pytest.fixture
    def agent(self):
        """Create agent instance with mocked dependencies."""
        with patch(
            "investigator.domain.agents.fundamental.agent.FundamentalAnalysisAgent.__init__",
            return_value=None,
        ):
            from investigator.domain.agents.fundamental.agent import (
                FundamentalAnalysisAgent,
            )

            agent = FundamentalAnalysisAgent.__new__(FundamentalAnalysisAgent)
            agent.logger = MagicMock()
            return agent

    def test_data_quality_to_confidence_flow(self, agent):
        """Test full flow from data quality to confidence level."""
        company_data = {
            "symbol": "TEST",
            "financials": {
                "revenues": 1000000,
                "net_income": 100000,
                "total_assets": 500000,
                "total_liabilities": 200000,
                "stockholders_equity": 300000,
            },
            "market_data": {
                "current_price": 100,
                "market_cap": 5000000,
            },
        }
        ratios = {
            "pe_ratio": 15.0,
            "price_to_book": 2.0,
            "current_ratio": 2.0,
            "debt_to_equity": 0.5,
        }

        # Get data quality
        data_quality = agent._assess_data_quality(company_data, ratios)

        # Get confidence level
        confidence = agent._calculate_confidence_level(data_quality)

        # Verify the flow
        assert confidence["based_on_data_quality"] == data_quality["data_quality_score"]
        assert confidence["quality_grade"] == data_quality["quality_grade"]

    def test_poor_quality_leads_to_low_confidence(self, agent):
        """Poor data quality should result in low confidence."""
        company_data = {"symbol": "TEST", "financials": {}}

        data_quality = agent._assess_data_quality(company_data, {})
        confidence = agent._calculate_confidence_level(data_quality)

        assert data_quality["quality_grade"] in ["Very Poor", "Poor"]
        assert confidence["confidence_level"] in ["VERY LOW", "LOW"]
