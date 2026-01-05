"""
Tests for the extracted TrendAnalyzer module.

Verifies the extracted module has identical behavior to the original
FundamentalAnalysisAgent methods.

Author: InvestiGator Team
Date: 2025-01-05
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from investigator.domain.agents.fundamental.trend_analyzer import (
    TrendAnalyzer,
    get_trend_analyzer,
)


@dataclass
class MockQuarterlyData:
    """Mock QuarterlyData for testing."""

    fiscal_year: int
    fiscal_period: str
    financial_data: Dict[str, Any]
    ratios: Optional[Dict[str, Any]] = None


class TestAnalyzeRevenueTrend:
    """Tests for analyze_revenue_trend method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return TrendAnalyzer()

    def test_insufficient_data_returns_default(self, analyzer):
        """Should return insufficient_data for less than 2 quarters."""
        result = analyzer.analyze_revenue_trend([])

        assert result["trend"] == "insufficient_data"
        assert result["q_over_q_growth"] == []

    def test_calculates_qoq_growth(self, analyzer):
        """Should calculate quarter-over-quarter growth rates."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 110}),
            MockQuarterlyData(2024, "Q3", {"revenues": 121}),
        ]

        result = analyzer.analyze_revenue_trend(quarterly_data)

        assert result["q_over_q_growth"][0] == 10.0
        assert result["q_over_q_growth"][1] == 10.0

    def test_calculates_yoy_growth(self, analyzer):
        """Should calculate year-over-year growth rates."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100}),
            MockQuarterlyData(2023, "Q4", {"revenues": 100}),
            MockQuarterlyData(2024, "Q1", {"revenues": 120}),
        ]

        result = analyzer.analyze_revenue_trend(quarterly_data)

        assert result["y_over_y_growth"][0] == 20.0

    def test_stable_trend(self, analyzer):
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

        result = analyzer.analyze_revenue_trend(quarterly_data)

        assert result["trend"] == "stable"

    def test_accelerating_trend(self, analyzer):
        """Should detect accelerating trend."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 102}),
            MockQuarterlyData(2023, "Q3", {"revenues": 104}),
            MockQuarterlyData(2023, "Q4", {"revenues": 106}),
            MockQuarterlyData(2024, "Q1", {"revenues": 115}),
            MockQuarterlyData(2024, "Q2", {"revenues": 125}),
            MockQuarterlyData(2024, "Q3", {"revenues": 140}),
        ]

        result = analyzer.analyze_revenue_trend(quarterly_data)

        assert result["trend"] == "accelerating"

    def test_decelerating_trend(self, analyzer):
        """Should detect decelerating trend."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100}),
            MockQuarterlyData(2023, "Q2", {"revenues": 115}),
            MockQuarterlyData(2023, "Q3", {"revenues": 132}),
            MockQuarterlyData(2023, "Q4", {"revenues": 152}),
            MockQuarterlyData(2024, "Q1", {"revenues": 155}),
            MockQuarterlyData(2024, "Q2", {"revenues": 158}),
            MockQuarterlyData(2024, "Q3", {"revenues": 160}),
        ]

        result = analyzer.analyze_revenue_trend(quarterly_data)

        assert result["trend"] == "decelerating"


class TestAnalyzeMarginTrend:
    """Tests for analyze_margin_trend method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return TrendAnalyzer()

    def test_insufficient_data_returns_default(self, analyzer):
        """Should return insufficient_data for less than 2 quarters."""
        result = analyzer.analyze_margin_trend([])

        assert result["gross_margin_trend"] == "insufficient_data"

    def test_calculates_margins(self, analyzer):
        """Should calculate margins from revenue and net income."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100, "net_income": 12}),
        ]

        result = analyzer.analyze_margin_trend(quarterly_data)

        assert result["net_margins"][0] == 10.0
        assert result["net_margins"][1] == 12.0

    def test_stable_margins(self, analyzer):
        """Should detect stable margin trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 110, "net_income": 11}),
            MockQuarterlyData(2024, "Q3", {"revenues": 120, "net_income": 12}),
            MockQuarterlyData(2024, "Q4", {"revenues": 130, "net_income": 13}),
        ]

        result = analyzer.analyze_margin_trend(quarterly_data)

        assert result["net_margin_trend"] == "stable"

    def test_expanding_margins(self, analyzer):
        """Should detect expanding margin trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 5}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100, "net_income": 6}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q4", {"revenues": 100, "net_income": 12}),
        ]

        result = analyzer.analyze_margin_trend(quarterly_data)

        assert result["net_margin_trend"] == "expanding"

    def test_contracting_margins(self, analyzer):
        """Should detect contracting margin trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 15}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100, "net_income": 14}),
            MockQuarterlyData(2024, "Q3", {"revenues": 100, "net_income": 8}),
            MockQuarterlyData(2024, "Q4", {"revenues": 100, "net_income": 6}),
        ]

        result = analyzer.analyze_margin_trend(quarterly_data)

        assert result["net_margin_trend"] == "contracting"


class TestAnalyzeCashFlowTrend:
    """Tests for analyze_cash_flow_trend method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return TrendAnalyzer()

    def test_insufficient_data_returns_default(self, analyzer):
        """Should return insufficient_data for less than 2 quarters."""
        result = analyzer.analyze_cash_flow_trend([])

        assert result["trend"] == "insufficient_data"
        assert result["quality_of_earnings"] == 0.0

    def test_calculates_free_cash_flow(self, analyzer):
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

        result = analyzer.analyze_cash_flow_trend(quarterly_data)

        assert result["free_cash_flow"][0] == 80
        assert result["free_cash_flow"][1] == 95

    def test_improving_trend(self, analyzer):
        """Should detect improving cash flow trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q2", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q3", {"operating_cash_flow": 120, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q4", {"operating_cash_flow": 120, "capital_expenditures": 0, "net_income": 80}),
        ]

        result = analyzer.analyze_cash_flow_trend(quarterly_data)

        assert result["trend"] == "improving"

    def test_deteriorating_trend(self, analyzer):
        """Should detect deteriorating cash flow trend."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"operating_cash_flow": 150, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q2", {"operating_cash_flow": 150, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q3", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
            MockQuarterlyData(2024, "Q4", {"operating_cash_flow": 100, "capital_expenditures": 0, "net_income": 80}),
        ]

        result = analyzer.analyze_cash_flow_trend(quarterly_data)

        assert result["trend"] == "deteriorating"


class TestCalculateQuarterlyComparisons:
    """Tests for calculate_quarterly_comparisons method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return TrendAnalyzer()

    def test_insufficient_data_returns_empty(self, analyzer):
        """Should return empty lists for less than 2 quarters."""
        result = analyzer.calculate_quarterly_comparisons([])

        assert result["q_over_q"]["revenue"] == []
        assert result["y_over_y"]["revenue"] == []

    def test_calculates_qoq_revenue(self, analyzer):
        """Should calculate Q/Q revenue growth."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q2", {"revenues": 120, "net_income": 15}),
        ]

        result = analyzer.calculate_quarterly_comparisons(quarterly_data)

        assert result["q_over_q"]["revenue"][0] == 20.0

    def test_calculates_yoy_comparisons(self, analyzer):
        """Should calculate Y/Y comparisons with 4 quarter lag."""
        quarterly_data = [
            MockQuarterlyData(2023, "Q1", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2023, "Q2", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2023, "Q3", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2023, "Q4", {"revenues": 100, "net_income": 10}),
            MockQuarterlyData(2024, "Q1", {"revenues": 130, "net_income": 15}),
        ]

        result = analyzer.calculate_quarterly_comparisons(quarterly_data)

        assert result["y_over_y"]["revenue"][0] == 30.0
        assert result["y_over_y"]["net_income"][0] == 50.0


class TestDetectCyclicalPatterns:
    """Tests for detect_cyclical_patterns method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return TrendAnalyzer()

    def test_insufficient_data_returns_default(self, analyzer):
        """Should return non-cyclical for less than 8 quarters."""
        quarterly_data = [
            MockQuarterlyData(2024, "Q1", {"revenues": 100}),
            MockQuarterlyData(2024, "Q2", {"revenues": 100}),
        ]

        result = analyzer.detect_cyclical_patterns(quarterly_data)

        assert result["is_cyclical"] is False
        assert result["seasonal_pattern"] == "insufficient_data"

    def test_non_cyclical_even_distribution(self, analyzer):
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

        result = analyzer.detect_cyclical_patterns(quarterly_data)

        assert result["is_cyclical"] is False
        assert result["seasonal_pattern"] == "non_cyclical"

    def test_cyclical_q4_strong(self, analyzer):
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

        result = analyzer.detect_cyclical_patterns(quarterly_data)

        assert result["is_cyclical"] is True
        assert "Q4" in result["seasonal_pattern"]


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_analyzer_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        a1 = get_trend_analyzer()
        a2 = get_trend_analyzer()
        assert a1 is a2
