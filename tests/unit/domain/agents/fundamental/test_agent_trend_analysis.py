"""
Regression tests for FundamentalAnalysisAgent trend analysis methods.

These tests capture the CURRENT BEHAVIOR of the methods BEFORE extraction.
They serve as regression tests to ensure the extracted modules maintain
identical behavior.

Methods tested:
- _analyze_revenue_trend
- _analyze_margin_trend
- _analyze_cash_flow_trend
- _calculate_quarterly_comparisons
- _detect_cyclical_patterns

Author: InvestiGator Team
Date: 2025-01-05
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest


@dataclass
class MockQuarterlyData:
    """Mock QuarterlyData for testing."""

    fiscal_year: int
    fiscal_period: str
    financial_data: Dict[str, Any]
    ratios: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None
    data_quality: Optional[Dict[str, Any]] = None
    filing_date: Optional[str] = None
    period_end_date: Optional[str] = None
    is_ytd_cashflow: bool = False
    is_ytd_income: bool = False
    value_type: str = "quarterly"


@pytest.fixture
def mock_agent():
    """Create a mock agent with real methods bound."""
    from investigator.domain.agents.fundamental.agent import FundamentalAnalysisAgent

    mock = MagicMock(spec=FundamentalAnalysisAgent)

    # Bind the real methods
    mock._analyze_revenue_trend = FundamentalAnalysisAgent._analyze_revenue_trend.__get__(mock)
    mock._analyze_margin_trend = FundamentalAnalysisAgent._analyze_margin_trend.__get__(mock)
    mock._analyze_cash_flow_trend = FundamentalAnalysisAgent._analyze_cash_flow_trend.__get__(mock)
    mock._calculate_quarterly_comparisons = FundamentalAnalysisAgent._calculate_quarterly_comparisons.__get__(mock)
    mock._detect_cyclical_patterns = FundamentalAnalysisAgent._detect_cyclical_patterns.__get__(mock)

    return mock


class TestAnalyzeRevenueTrend:
    """Tests for _analyze_revenue_trend method."""

    def test_insufficient_data_returns_default(self, mock_agent):
        """Should return insufficient_data for less than 2 quarters."""
        result = mock_agent._analyze_revenue_trend([])

        assert result["trend"] == "insufficient_data"
        assert result["q_over_q_growth"] == []
        assert result["average_growth"] == 0.0

    def test_calculates_qoq_growth(self, mock_agent):
        """Should calculate quarter-over-quarter growth rates."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 110}),
            MockQuarterlyData(2024, "Q3", {"revenues": 121}),
        ]

        result = mock_agent._analyze_revenue_trend(quarterly_data)

        # Q1→Q2: (110-100)/100 = 10%, Q2→Q3: (121-110)/110 = 10%
        assert len(result["q_over_q_growth"]) == 2
        assert result["q_over_q_growth"][0] == 10.0
        assert result["q_over_q_growth"][1] == 10.0

    def test_calculates_yoy_growth(self, mock_agent):
        """Should calculate year-over-year growth rates."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100}),
            MockQuarterlyData(2023, "Q4", {"revenues": 100}),
            MockQuarterlyData(2024, "Q1", {"revenues": 120}),
        ]

        result = mock_agent._analyze_revenue_trend(quarterly_data)

        # Q1 2024 vs Q1 2023: (120-100)/100 = 20%
        assert len(result["y_over_y_growth"]) == 1
        assert result["y_over_y_growth"][0] == 20.0

    def test_stable_trend_with_consistent_growth(self, mock_agent):
        """Should detect stable trend when growth is consistent."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 105}),
            MockQuarterlyData(2023, "Q3", {"revenues": 110}),
            MockQuarterlyData(2023, "Q4", {"revenues": 115}),
            MockQuarterlyData(2024, "Q1", {"revenues": 121}),
            MockQuarterlyData(2024, "Q2", {"revenues": 127}),
            MockQuarterlyData(2024, "Q3", {"revenues": 133}),
        ]

        result = mock_agent._analyze_revenue_trend(quarterly_data)

        assert result["trend"] == "stable"

    def test_accelerating_trend(self, mock_agent):
        """Should detect accelerating trend when late growth exceeds early."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 102}),
            MockQuarterlyData(2023, "Q3", {"revenues": 104}),
            MockQuarterlyData(2023, "Q4", {"revenues": 106}),
            MockQuarterlyData(2024, "Q1", {"revenues": 115}),
            MockQuarterlyData(2024, "Q2", {"revenues": 125}),
            MockQuarterlyData(2024, "Q3", {"revenues": 140}),
        ]

        result = mock_agent._analyze_revenue_trend(quarterly_data)

        assert result["trend"] == "accelerating"

    def test_decelerating_trend(self, mock_agent):
        """Should detect decelerating trend when late growth is slower."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 115}),
            MockQuarterlyData(2023, "Q3", {"revenues": 132}),
            MockQuarterlyData(2023, "Q4", {"revenues": 152}),
            MockQuarterlyData(2024, "Q1", {"revenues": 155}),
            MockQuarterlyData(2024, "Q2", {"revenues": 158}),
            MockQuarterlyData(2024, "Q3", {"revenues": 160}),
        ]

        result = mock_agent._analyze_revenue_trend(quarterly_data)

        assert result["trend"] == "decelerating"

    def test_calculates_volatility(self, mock_agent):
        """Should calculate volatility (standard deviation)."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 110}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100}),
            MockQuarterlyData(2024, "Q4", {"revenues": 110}),
        ]

        result = mock_agent._analyze_revenue_trend(quarterly_data)

        assert result["volatility"] > 0

    def test_consistency_score_calculation(self, mock_agent):
        """Should calculate consistency score (0-100)."""
        # Low volatility → high consistency
        stable_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100}),
        ]

        result = mock_agent._analyze_revenue_trend(stable_data)

        assert result["consistency_score"] == 100.0


class TestAnalyzeMarginTrend:
    """Tests for _analyze_margin_trend method."""

    def test_insufficient_data_returns_default(self, mock_agent):
        """Should return insufficient_data for less than 2 quarters."""
        result = mock_agent._analyze_margin_trend([])

        assert result["gross_margin_trend"] == "insufficient_data"
        assert result["net_margins"] == []

    def test_calculates_margins_from_revenue_and_income(self, mock_agent):
        """Should calculate margins from revenue and net income."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100, "net_income": 12}),
        ]

        result = mock_agent._analyze_margin_trend(quarterly_data)

        # 10/100 = 10%, 12/100 = 12%
        assert result["net_margins"][0] == 10.0
        assert result["net_margins"][1] == 12.0

    def test_stable_margins(self, mock_agent):
        """Should detect stable margin trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 110, "net_income": 11}),
            MockQuarterlyData(2024, "Q3", {"revenues": 120, "net_income": 12}),
            MockQuarterlyData(2024, "Q4", {"revenues": 130, "net_income": 13}),
        ]

        result = mock_agent._analyze_margin_trend(quarterly_data)

        # All quarters have ~10% margin
        assert result["net_margin_trend"] == "stable"

    def test_expanding_margins(self, mock_agent):
        """Should detect expanding margin trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 5}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100, "net_income": 6}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q4", {"revenues": 100, "net_income": 12}),
        ]

        result = mock_agent._analyze_margin_trend(quarterly_data)

        # Early avg: 5.5%, Late avg: 11%
        assert result["net_margin_trend"] == "expanding"

    def test_contracting_margins(self, mock_agent):
        """Should detect contracting margin trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 15}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100, "net_income": 14}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100, "net_income": 8}),
            MockQuarterlyData(2024, "Q4", {"revenues": 100, "net_income": 6}),
        ]

        result = mock_agent._analyze_margin_trend(quarterly_data)

        # Early avg: 14.5%, Late avg: 7%
        assert result["net_margin_trend"] == "contracting"


class TestAnalyzeCashFlowTrend:
    """Tests for _analyze_cash_flow_trend method."""

    def test_insufficient_data_returns_default(self, mock_agent):
        """Should return insufficient_data for less than 2 quarters."""
        result = mock_agent._analyze_cash_flow_trend([])

        assert result["trend"] == "insufficient_data"
        assert result["quality_of_earnings"] == 0.0

    def test_calculates_free_cash_flow(self, mock_agent):
        """Should calculate FCF from OCF minus CapEx."""
        quarterly_data = [
            MockQuarterlyData(
                2024,
                "Q1",
                {
                    "operating_cash_flow": 100,
                    "capital_expenditures": 20,
                    "net_income": 80,
                },
            ),
            MockQuarterlyData(
                2024,
                "Q2",
                {
                    "operating_cash_flow": 120,
                    "capital_expenditures": 25,
                    "net_income": 90,
                },
            ),
        ]

        result = mock_agent._analyze_cash_flow_trend(quarterly_data)

        # FCF = OCF - CapEx
        assert result["free_cash_flow"][0] == 80
        assert result["free_cash_flow"][1] == 95

    def test_cash_conversion_ratio(self, mock_agent):
        """Should calculate cash conversion ratio (OCF/Net Income)."""
        quarterly_data = [
            MockQuarterlyData(
                2024,
                "Q1",
                {
                    "operating_cash_flow": 120,
                    "capital_expenditures": 20,
                    "net_income": 100,
                },
            ),
            MockQuarterlyData(
                2024,
                "Q2",
                {
                    "operating_cash_flow": 110,
                    "capital_expenditures": 20,
                    "net_income": 100,
                },
            ),
        ]

        result = mock_agent._analyze_cash_flow_trend(quarterly_data)

        # 120/100 = 120%, 110/100 = 110%
        assert result["cash_conversion_ratio"][0] == 120.0
        assert result["cash_conversion_ratio"][1] == 110.0

    def test_quality_of_earnings_high(self, mock_agent):
        """Should calculate high quality of earnings for good conversion."""
        quarterly_data = [
            MockQuarterlyData(
                2024,
                "Q1",
                {
                    "operating_cash_flow": 150,
                    "capital_expenditures": 20,
                    "net_income": 100,
                },
            ),
            MockQuarterlyData(
                2024,
                "Q2",
                {
                    "operating_cash_flow": 140,
                    "capital_expenditures": 20,
                    "net_income": 100,
                },
            ),
        ]

        result = mock_agent._analyze_cash_flow_trend(quarterly_data)

        # Avg conversion = 145%, should be > 95
        assert result["quality_of_earnings"] >= 95

    def test_improving_trend(self, mock_agent):
        """Should detect improving cash flow trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q2", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q3", {"operating_cash_flow": 120, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q4", {"operating_cash_flow": 120, "capital_expenditures": 0, "net_income": 80}),
        ]

        result = mock_agent._analyze_cash_flow_trend(quarterly_data)

        # Late avg (120) > Early avg (100) * 1.1
        assert result["trend"] == "improving"

    def test_deteriorating_trend(self, mock_agent):
        """Should detect deteriorating cash flow trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"operating_cash_flow": 150, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q2", {"operating_cash_flow": 150, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q3", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q4", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
        ]

        result = mock_agent._analyze_cash_flow_trend(quarterly_data)

        # Late avg (100) < Early avg (150) * 0.9
        assert result["trend"] == "deteriorating"


class TestCalculateQuarterlyComparisons:
    """Tests for _calculate_quarterly_comparisons method."""

    def test_insufficient_data_returns_empty(self, mock_agent):
        """Should return empty lists for less than 2 quarters."""
        result = mock_agent._calculate_quarterly_comparisons([])

        assert result["q_over_q"]["revenue"] == []
        assert result["y_over_y"]["revenue"] == []

    def test_calculates_qoq_revenue(self, mock_agent):
        """Should calculate Q/Q revenue growth."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 120, "net_income": 15}),
        ]

        result = mock_agent._calculate_quarterly_comparisons(quarterly_data)

        # (120-100)/100 = 20%
        assert result["q_over_q"]["revenue"][0] == 20.0

    def test_calculates_qoq_net_income(self, mock_agent):
        """Should calculate Q/Q net income growth."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 120, "net_income": 15}),
        ]

        result = mock_agent._calculate_quarterly_comparisons(quarterly_data)

        # (15-10)/10 = 50%
        assert result["q_over_q"]["net_income"][0] == 50.0

    def test_calculates_yoy_comparisons(self, mock_agent):
        """Should calculate Y/Y comparisons with 4 quarter lag."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2023, "Q4", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q1", {"revenues": 130, "net_income": 15}),
        ]

        result = mock_agent._calculate_quarterly_comparisons(quarterly_data)

        # (130-100)/100 = 30% revenue, (15-10)/10 = 50% net income
        assert result["y_over_y"]["revenue"][0] == 30.0
        assert result["y_over_y"]["net_income"][0] == 50.0


class TestDetectCyclicalPatterns:
    """Tests for _detect_cyclical_patterns method."""

    def test_insufficient_data_returns_default(self, mock_agent):
        """Should return non-cyclical for less than 8 quarters."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
        ]

        result = mock_agent._detect_cyclical_patterns(quarterly_data)

        assert result["is_cyclical"] is False
        assert result["seasonal_pattern"] == "insufficient_data"
        assert result["pattern_confidence"] == 0.0

    def test_non_cyclical_even_distribution(self, mock_agent):
        """Should detect non-cyclical for even revenue distribution."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100}),
            MockQuarterlyData(2023, "Q4", {"revenues": 100}),
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100}),
            MockQuarterlyData(2024, "Q4", {"revenues": 100}),
        ]

        result = mock_agent._detect_cyclical_patterns(quarterly_data)

        assert result["is_cyclical"] is False
        assert result["seasonal_pattern"] == "non_cyclical"

    def test_cyclical_q4_strong(self, mock_agent):
        """Should detect cyclical with Q4 strength (like retail)."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 80}),
            MockQuarterlyData(2023, "Q2", {"revenues": 90}),
            MockQuarterlyData(2023, "Q3", {"revenues": 90}),
            MockQuarterlyData(2023, "Q4", {"revenues": 140}),
            MockQuarterlyData(2024, "Q1", {"revenues": 80}),
            MockQuarterlyData(2024, "Q2", {"revenues": 90}),
            MockQuarterlyData(2024, "Q3", {"revenues": 90}),
            MockQuarterlyData(2024, "Q4", {"revenues": 140}),
        ]

        result = mock_agent._detect_cyclical_patterns(quarterly_data)

        assert result["is_cyclical"] is True
        assert "Q4" in result["seasonal_pattern"]
        assert result["quarterly_strength"]["Q4"] > 0

    def test_quarterly_strength_calculation(self, mock_agent):
        """Should calculate quarterly strength relative to average."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 80}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100}),
            MockQuarterlyData(2023, "Q4", {"revenues": 120}),
            MockQuarterlyData(2024, "Q1", {"revenues": 80}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100}),
            MockQuarterlyData(2024, "Q4", {"revenues": 120}),
        ]

        result = mock_agent._detect_cyclical_patterns(quarterly_data)

        # Average = 100, Q1 = 80 (-20%), Q4 = 120 (+20%)
        assert result["quarterly_strength"]["Q1"] < 0
        assert result["quarterly_strength"]["Q4"] > 0

    def test_pattern_confidence_increases_with_deviation(self, mock_agent):
        """Should have higher confidence with larger deviations."""
        mild_cyclical = [
            MockQuarterlyData(2023, "Q1", {"revenues": 95}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100}),
            MockQuarterlyData(2023, "Q4", {"revenues": 105}),
            MockQuarterlyData(2024, "Q1", {"revenues": 95}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100}),
            MockQuarterlyData(2024, "Q4", {"revenues": 105}),
        ]

        strong_cyclical = [
            MockQuarterlyData(2023, "Q1", {"revenues": 70}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100}),
            MockQuarterlyData(2023, "Q4", {"revenues": 130}),
            MockQuarterlyData(2024, "Q1", {"revenues": 70}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100}),
            MockQuarterlyData(2024, "Q4", {"revenues": 130}),
        ]

        mild_result = mock_agent._detect_cyclical_patterns(mild_cyclical)
        strong_result = mock_agent._detect_cyclical_patterns(strong_cyclical)

        assert strong_result["pattern_confidence"] > mild_result["pattern_confidence"]
