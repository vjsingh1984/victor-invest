"""
Regression tests for deterministic analysis methods in FundamentalAnalysisAgent.

These tests capture the EXACT behavior of the original methods BEFORE extraction.
Run these tests before AND after extracting DeterministicAnalyzer to ensure no regressions.

Methods tested:
- _analyze_financial_health: Liquidity, solvency, capital structure assessment
- _analyze_growth: Revenue and earnings growth analysis
- _analyze_profitability: Margin and returns analysis

Author: InvestiGator Team
Date: 2025-01-05
"""

from unittest.mock import MagicMock, patch

import pytest


class TestAnalyzeFinancialHealth:
    """Regression tests for _analyze_financial_health method."""

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
            agent.agent_id = "test-agent"
            return agent

    @pytest.mark.asyncio
    async def test_strong_liquidity_high_current_ratio(self, agent):
        """Strong liquidity when current ratio >= 2.0."""
        company_data = {
            "financials": {
                "current_assets": 200000,
                "current_liabilities": 80000,
            }
        }
        ratios = {"current_ratio": 2.5, "quick_ratio": 2.0}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["liquidity_position"]["assessment"] == "Strong"
        assert payload["overall_health_score"] >= 70

    @pytest.mark.asyncio
    async def test_weak_liquidity_low_current_ratio(self, agent):
        """Weak liquidity when current ratio < 1.2."""
        company_data = {
            "financials": {
                "current_assets": 100000,
                "current_liabilities": 100000,
            }
        }
        ratios = {"current_ratio": 1.0, "quick_ratio": 0.8}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["liquidity_position"]["assessment"] == "Weak"
        assert any(rf["risk"] == "Tight liquidity" for rf in payload["risk_factors"])

    @pytest.mark.asyncio
    async def test_debt_free_company(self, agent):
        """Debt-free company should have optimal solvency."""
        company_data = {
            "financials": {
                "total_debt": 0,
                "stockholders_equity": 500000,
            }
        }
        ratios = {"current_ratio": 2.0, "debt_to_equity": 0}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["solvency"]["assessment"] == "Debt Free"

    @pytest.mark.asyncio
    async def test_leveraged_company(self, agent):
        """Highly leveraged company detection."""
        company_data = {
            "financials": {
                "total_debt": 600000,
                "stockholders_equity": 200000,
            }
        }
        ratios = {"current_ratio": 1.5, "debt_to_equity": 3.0, "debt_to_assets": 0.6}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["solvency"]["assessment"] == "Leveraged"
        assert any(rf["risk"] == "High leverage" for rf in payload["risk_factors"])

    @pytest.mark.asyncio
    async def test_net_cash_position(self, agent):
        """Net cash position when cash exceeds debt."""
        company_data = {
            "financials": {
                "cash": 500000,
                "total_debt": 200000,
                "stockholders_equity": 300000,
            }
        }
        ratios = {"current_ratio": 2.0, "debt_to_equity": 0.67}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["capital_structure_quality"]["assessment"] == "Net Cash"

    @pytest.mark.asyncio
    async def test_efficient_working_capital(self, agent):
        """Efficient working capital with high OCF margin."""
        company_data = {
            "financials": {
                "operating_cash_flow": 250000,
                "revenues": 1000000,
            }
        }
        ratios = {"current_ratio": 2.0}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["working_capital_management"]["assessment"] == "Efficient"

    @pytest.mark.asyncio
    async def test_returns_all_expected_keys(self, agent):
        """Should return all expected payload keys."""
        company_data = {"financials": {"revenues": 1000000}}
        ratios = {"current_ratio": 1.5}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        payload = result["response"]
        expected_keys = [
            "liquidity_position",
            "solvency",
            "capital_structure_quality",
            "working_capital_management",
            "debt_serviceability",
            "financial_flexibility",
            "risk_factors",
            "overall_health_score",
        ]

        for key in expected_keys:
            assert key in payload, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_deterministic_response_format(self, agent):
        """Should return proper deterministic response format."""
        company_data = {"financials": {"revenues": 1000000}}
        ratios = {"current_ratio": 1.5}

        result = await agent._analyze_financial_health(company_data, ratios, "TEST")

        assert "response" in result
        assert "model_info" in result
        assert result["model_info"]["model"] == "deterministic-financial_health"


class TestAnalyzeGrowth:
    """Regression tests for _analyze_growth method."""

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
            agent.agent_id = "test-agent"
            return agent

    @pytest.mark.asyncio
    async def test_high_growth_classification(self, agent):
        """High growth when YoY >= 8%."""
        company_data = {
            "trend_analysis": {
                "revenue": {
                    "y_over_y_growth": [10.0, 12.0, 9.0, 8.0],
                    "q_over_q_growth": [3.0, 2.5, 3.0, 2.0],
                    "consistency_score": 85,
                    "trend": "accelerating",
                },
                "margins": {"net_margin_trend": "expanding"},
            }
        }

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        assert payload["revenue_growth_sustainability"]["assessment"] == "High"
        assert payload["growth_score"] >= 75

    @pytest.mark.asyncio
    async def test_contracting_growth(self, agent):
        """Contracting growth when YoY < 0."""
        company_data = {
            "trend_analysis": {
                "revenue": {
                    "y_over_y_growth": [-5.0, -3.0, -4.0],
                    "q_over_q_growth": [-1.0, -2.0, -1.5],
                    "consistency_score": 60,
                    "trend": "decelerating",
                },
                "margins": {"net_margin_trend": "contracting"},
            }
        }

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        assert payload["revenue_growth_sustainability"]["assessment"] == "Contracting"
        assert "Negative revenue comp" in payload["growth_risks_and_headwinds"]

    @pytest.mark.asyncio
    async def test_stable_growth(self, agent):
        """Stable growth when YoY between 0-3%."""
        company_data = {
            "trend_analysis": {
                "revenue": {
                    "y_over_y_growth": [1.0, 2.0, 1.5, 2.5],
                    "q_over_q_growth": [0.5, 0.8, 0.3],
                    "trend": "stable",
                },
                "margins": {"net_margin_trend": "stable"},
            }
        }

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        assert payload["revenue_growth_sustainability"]["assessment"] == "Stable"

    @pytest.mark.asyncio
    async def test_cyclical_business_risk(self, agent):
        """Should detect cyclical business risk."""
        company_data = {
            "trend_analysis": {
                "revenue": {
                    "y_over_y_growth": [5.0, 6.0, 5.5],
                    "trend": "stable",
                },
                "margins": {"net_margin_trend": "stable"},
                "cyclical": {"is_cyclical": True},
            }
        }

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        assert "Seasonality swings" in payload["growth_risks_and_headwinds"]

    @pytest.mark.asyncio
    async def test_growth_drivers_populated(self, agent):
        """Growth drivers should be populated."""
        company_data = {
            "trend_analysis": {
                "revenue": {
                    "y_over_y_growth": [7.0, 8.0, 6.0],
                    "trend": "accelerating",
                },
                "margins": {"net_margin_trend": "expanding"},
            }
        }

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        assert len(payload["growth_drivers_and_catalysts"]) > 0
        assert "Operational leverage" in payload["growth_drivers_and_catalysts"]

    @pytest.mark.asyncio
    async def test_empty_trend_data(self, agent):
        """Should handle empty trend data gracefully."""
        company_data = {"trend_analysis": {}}

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        assert payload["revenue_growth_sustainability"]["assessment"] == "Unknown"

    @pytest.mark.asyncio
    async def test_returns_all_expected_keys(self, agent):
        """Should return all expected payload keys."""
        company_data = {"trend_analysis": {}}

        result = await agent._analyze_growth(company_data, "TEST")

        payload = result["response"]
        expected_keys = [
            "revenue_growth_sustainability",
            "earnings_growth_quality",
            "growth_consistency_and_volatility",
            "market_share_trends",
            "growth_drivers_and_catalysts",
            "future_growth_potential",
            "growth_risks_and_headwinds",
            "growth_score",
        ]

        for key in expected_keys:
            assert key in payload, f"Missing key: {key}"


class TestAnalyzeProfitability:
    """Regression tests for _analyze_profitability method."""

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
            agent.agent_id = "test-agent"
            return agent

    @pytest.mark.asyncio
    async def test_wide_margin_classification(self, agent):
        """Wide margins when >= 25%."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {
            "gross_margin": 0.35,
            "operating_margin": 0.25,
            "net_margin": 0.18,
            "roe": 0.20,
            "roa": 0.12,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["competitive_advantages_moat"]["assessment"] == "Wide"

    @pytest.mark.asyncio
    async def test_thin_margin_classification(self, agent):
        """Thin margins when between 5-15%."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {
            "gross_margin": 0.08,
            "operating_margin": 0.05,
            "net_margin": 0.03,
            "roe": 0.08,
            "roa": 0.04,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["competitive_advantages_moat"]["assessment"] in ["Thin", "Negative"]

    @pytest.mark.asyncio
    async def test_high_roe_classification(self, agent):
        """High ROE when >= 18%."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {
            "gross_margin": 0.30,
            "operating_margin": 0.20,
            "net_margin": 0.15,
            "roe": 0.22,
            "roa": 0.15,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["return_on_capital_efficiency"]["assessment"] == "High"

    @pytest.mark.asyncio
    async def test_margin_trend_expanding(self, agent):
        """Should detect expanding margin trend."""
        company_data = {
            "trend_analysis": {
                "margins": {
                    "net_margin_trend": "expanding",
                    "gross_margins": [20.0, 22.0, 24.0, 25.0],
                    "net_margins": [10.0, 12.0, 14.0, 16.0],
                }
            }
        }
        ratios = {
            "gross_margin": 0.25,
            "operating_margin": 0.18,
            "net_margin": 0.16,
            "roe": 0.15,
            "roa": 0.10,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["margin_trends_and_sustainability"]["assessment"] == "Expanding"

    @pytest.mark.asyncio
    async def test_lean_cost_structure(self, agent):
        """Lean cost structure when spread <= 10%."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {
            "gross_margin": 0.30,
            "operating_margin": 0.22,  # Spread = 8%
            "net_margin": 0.15,
            "roe": 0.15,
            "roa": 0.10,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["cost_structure_analysis"]["assessment"] == "Lean"

    @pytest.mark.asyncio
    async def test_heavy_cost_structure(self, agent):
        """Heavy cost structure when spread > 20%."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {
            "gross_margin": 0.40,
            "operating_margin": 0.15,  # Spread = 25%
            "net_margin": 0.10,
            "roe": 0.12,
            "roa": 0.08,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["cost_structure_analysis"]["assessment"] == "Heavy"

    @pytest.mark.asyncio
    async def test_profitability_drivers_populated(self, agent):
        """Profitability drivers should be populated."""
        company_data = {
            "trend_analysis": {
                "margins": {
                    "gross_margins": [30.0, 32.0, 31.0],
                    "net_margins": [18.0, 19.0, 20.0],
                }
            }
        }
        ratios = {
            "gross_margin": 0.32,
            "operating_margin": 0.22,
            "net_margin": 0.20,
            "roe": 0.18,
            "roa": 0.12,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert len(payload["profitability_drivers"]) > 0

    @pytest.mark.asyncio
    async def test_missing_ratios_handled(self, agent):
        """Should handle missing ratios gracefully."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {}

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert payload["competitive_advantages_moat"]["assessment"] == "Unknown"
        assert payload["return_on_capital_efficiency"]["assessment"] == "Unknown"

    @pytest.mark.asyncio
    async def test_returns_all_expected_keys(self, agent):
        """Should return all expected payload keys."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {}

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        expected_keys = [
            "margin_trends_and_sustainability",
            "return_on_capital_efficiency",
            "competitive_advantages_moat",
            "pricing_power_indicators",
            "cost_structure_analysis",
            "operating_leverage",
            "profitability_drivers",
            "profitability_score",
        ]

        for key in expected_keys:
            assert key in payload, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_profitability_score_calculation(self, agent):
        """Should calculate profitability score as average of component scores."""
        company_data = {"trend_analysis": {"margins": {}}}
        ratios = {
            "gross_margin": 0.30,
            "operating_margin": 0.20,
            "net_margin": 0.15,
            "roe": 0.15,
            "roa": 0.10,
        }

        result = await agent._analyze_profitability(company_data, ratios, "TEST")

        payload = result["response"]
        assert 0 <= payload["profitability_score"] <= 100


class TestDeterministicAnalysisIntegration:
    """Integration tests for deterministic analysis flow."""

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
            agent.agent_id = "test-agent"
            return agent

    @pytest.mark.asyncio
    async def test_all_analyses_consistent_format(self, agent):
        """All analysis methods should return consistent format."""
        company_data = {
            "financials": {"revenues": 1000000},
            "trend_analysis": {},
        }
        ratios = {"current_ratio": 1.5}

        health = await agent._analyze_financial_health(company_data, ratios, "TEST")
        growth = await agent._analyze_growth(company_data, "TEST")
        profit = await agent._analyze_profitability(company_data, ratios, "TEST")

        # All should have same structure
        for result in [health, growth, profit]:
            assert "response" in result
            assert "model_info" in result
            assert "metadata" in result
            assert result["model_info"]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, agent):
        """All scores should be in valid 0-100 range."""
        company_data = {
            "financials": {
                "revenues": 1000000,
                "total_debt": 200000,
                "stockholders_equity": 500000,
                "operating_cash_flow": 150000,
            },
            "trend_analysis": {
                "revenue": {"y_over_y_growth": [5.0, 6.0, 5.5]},
                "margins": {"net_margin_trend": "stable"},
            },
        }
        ratios = {
            "current_ratio": 2.0,
            "debt_to_equity": 0.4,
            "gross_margin": 0.30,
            "operating_margin": 0.20,
            "net_margin": 0.15,
            "roe": 0.15,
            "roa": 0.10,
        }

        health = await agent._analyze_financial_health(company_data, ratios, "TEST")
        growth = await agent._analyze_growth(company_data, "TEST")
        profit = await agent._analyze_profitability(company_data, ratios, "TEST")

        assert 0 <= health["response"]["overall_health_score"] <= 100
        assert 0 <= growth["response"]["growth_score"] <= 100
        assert 0 <= profit["response"]["profitability_score"] <= 100
