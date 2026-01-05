"""
Tests for the extracted DeterministicAnalyzer module.

Verifies the extracted module has identical behavior to the original
FundamentalAnalysisAgent methods.

Author: InvestiGator Team
Date: 2025-01-05
"""

import pytest

from investigator.domain.agents.fundamental.deterministic_analyzer import (
    DeterministicAnalyzer,
    get_deterministic_analyzer,
)


class TestAnalyzeFinancialHealth:
    """Tests for analyze_financial_health method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return DeterministicAnalyzer(agent_id="test-agent")

    @pytest.mark.asyncio
    async def test_strong_liquidity(self, analyzer):
        """Strong liquidity when current ratio >= 2.0."""
        company_data = {"financials": {}}
        ratios = {"current_ratio": 2.5, "quick_ratio": 2.0}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["liquidity_position"]["assessment"] == "Strong"

    @pytest.mark.asyncio
    async def test_adequate_liquidity(self, analyzer):
        """Adequate liquidity when 1.2 <= current ratio < 2.0."""
        company_data = {"financials": {}}
        ratios = {"current_ratio": 1.5}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["liquidity_position"]["assessment"] == "Adequate"

    @pytest.mark.asyncio
    async def test_weak_liquidity(self, analyzer):
        """Weak liquidity when current ratio < 1.2."""
        company_data = {"financials": {}}
        ratios = {"current_ratio": 0.8}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["liquidity_position"]["assessment"] == "Weak"

    @pytest.mark.asyncio
    async def test_debt_free_solvency(self, analyzer):
        """Debt-free company detection."""
        company_data = {"financials": {"total_debt": 0}}
        ratios = {"current_ratio": 2.0}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["solvency"]["assessment"] == "Debt Free"

    @pytest.mark.asyncio
    async def test_leveraged_solvency(self, analyzer):
        """Leveraged when debt_to_equity > 2.0."""
        company_data = {"financials": {"total_debt": 300000}}
        ratios = {"current_ratio": 1.5, "debt_to_equity": 2.5}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["solvency"]["assessment"] == "Leveraged"

    @pytest.mark.asyncio
    async def test_net_cash_position(self, analyzer):
        """Net cash when cash exceeds debt."""
        company_data = {"financials": {"cash": 500000, "total_debt": 200000}}
        ratios = {"current_ratio": 2.0, "debt_to_equity": 0.5}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["capital_structure_quality"]["assessment"] == "Net Cash"

    @pytest.mark.asyncio
    async def test_efficient_working_capital(self, analyzer):
        """Efficient when OCF margin >= 20%."""
        company_data = {
            "financials": {
                "operating_cash_flow": 250000,
                "revenues": 1000000,
            }
        }
        ratios = {"current_ratio": 2.0}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["response"]["working_capital_management"]["assessment"] == "Efficient"

    @pytest.mark.asyncio
    async def test_risk_factors_populated(self, analyzer):
        """Risk factors should be populated for weak metrics."""
        company_data = {"financials": {"total_debt": 500000}}
        ratios = {"current_ratio": 0.8, "debt_to_equity": 3.0}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        risk_factors = result["response"]["risk_factors"]
        assert len(risk_factors) >= 2
        risk_names = [rf["risk"] for rf in risk_factors]
        assert "Tight liquidity" in risk_names
        assert "High leverage" in risk_names


class TestAnalyzeGrowth:
    """Tests for analyze_growth method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return DeterministicAnalyzer(agent_id="test-agent")

    @pytest.mark.asyncio
    async def test_high_growth(self, analyzer):
        """High growth when YoY >= 8%."""
        company_data = {
            "trend_analysis": {
                "revenue": {"y_over_y_growth": [10.0, 12.0, 9.0]},
            }
        }

        result = await analyzer.analyze_growth(company_data, "TEST")

        assert result["response"]["revenue_growth_sustainability"]["assessment"] == "High"

    @pytest.mark.asyncio
    async def test_moderate_growth(self, analyzer):
        """Moderate growth when 3% <= YoY < 8%."""
        company_data = {
            "trend_analysis": {
                "revenue": {"y_over_y_growth": [5.0, 4.0, 6.0]},
            }
        }

        result = await analyzer.analyze_growth(company_data, "TEST")

        assert result["response"]["revenue_growth_sustainability"]["assessment"] == "Moderate"

    @pytest.mark.asyncio
    async def test_contracting_growth(self, analyzer):
        """Contracting growth when YoY < 0%."""
        company_data = {
            "trend_analysis": {
                "revenue": {"y_over_y_growth": [-5.0, -3.0, -4.0]},
            }
        }

        result = await analyzer.analyze_growth(company_data, "TEST")

        assert result["response"]["revenue_growth_sustainability"]["assessment"] == "Contracting"

    @pytest.mark.asyncio
    async def test_growth_drivers_momentum(self, analyzer):
        """Product demand momentum when YoY >= 5%."""
        company_data = {
            "trend_analysis": {
                "revenue": {"y_over_y_growth": [6.0, 7.0, 5.0]},
            }
        }

        result = await analyzer.analyze_growth(company_data, "TEST")

        assert "Product demand momentum" in result["response"]["growth_drivers_and_catalysts"]

    @pytest.mark.asyncio
    async def test_cyclical_risk(self, analyzer):
        """Seasonality swings risk for cyclical businesses."""
        company_data = {
            "trend_analysis": {
                "cyclical": {"is_cyclical": True},
                "revenue": {"y_over_y_growth": [5.0]},
            }
        }

        result = await analyzer.analyze_growth(company_data, "TEST")

        assert "Seasonality swings" in result["response"]["growth_risks_and_headwinds"]

    @pytest.mark.asyncio
    async def test_empty_trend_data(self, analyzer):
        """Should handle empty trend data."""
        company_data = {}

        result = await analyzer.analyze_growth(company_data, "TEST")

        assert result["response"]["revenue_growth_sustainability"]["assessment"] == "Unknown"


class TestAnalyzeProfitability:
    """Tests for analyze_profitability method."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return DeterministicAnalyzer(agent_id="test-agent")

    @pytest.mark.asyncio
    async def test_wide_margin(self, analyzer):
        """Wide margin when >= 25%."""
        company_data = {"trend_analysis": {}}
        ratios = {"gross_margin": 0.30}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["competitive_advantages_moat"]["assessment"] == "Wide"

    @pytest.mark.asyncio
    async def test_healthy_margin(self, analyzer):
        """Healthy margin when 15% <= margin < 25%."""
        company_data = {"trend_analysis": {}}
        ratios = {"gross_margin": 0.18}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["competitive_advantages_moat"]["assessment"] == "Healthy"

    @pytest.mark.asyncio
    async def test_thin_margin(self, analyzer):
        """Thin margin when 5% <= margin < 15%."""
        company_data = {"trend_analysis": {}}
        ratios = {"gross_margin": 0.08}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["competitive_advantages_moat"]["assessment"] == "Thin"

    @pytest.mark.asyncio
    async def test_high_roe(self, analyzer):
        """High ROE when >= 18%."""
        company_data = {"trend_analysis": {}}
        ratios = {"roe": 0.22}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["return_on_capital_efficiency"]["assessment"] == "High"

    @pytest.mark.asyncio
    async def test_lean_cost_structure(self, analyzer):
        """Lean cost structure when spread <= 10%."""
        company_data = {"trend_analysis": {}}
        ratios = {"gross_margin": 0.30, "operating_margin": 0.22}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["cost_structure_analysis"]["assessment"] == "Lean"

    @pytest.mark.asyncio
    async def test_heavy_cost_structure(self, analyzer):
        """Heavy cost structure when spread > 20%."""
        company_data = {"trend_analysis": {}}
        ratios = {"gross_margin": 0.40, "operating_margin": 0.15}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["cost_structure_analysis"]["assessment"] == "Heavy"

    @pytest.mark.asyncio
    async def test_margin_trend_expanding(self, analyzer):
        """Should detect expanding margin trend."""
        company_data = {"trend_analysis": {"margins": {"net_margin_trend": "expanding"}}}
        ratios = {}

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        assert result["response"]["margin_trends_and_sustainability"]["assessment"] == "Expanding"

    @pytest.mark.asyncio
    async def test_profitability_score_range(self, analyzer):
        """Profitability score should be in 0-100 range."""
        company_data = {"trend_analysis": {}}
        ratios = {
            "gross_margin": 0.30,
            "operating_margin": 0.20,
            "net_margin": 0.15,
            "roe": 0.18,
            "roa": 0.12,
        }

        result = await analyzer.analyze_profitability(company_data, ratios, "TEST")

        score = result["response"]["profitability_score"]
        assert 0 <= score <= 100


class TestDeterministicResponse:
    """Tests for deterministic response format."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return DeterministicAnalyzer(agent_id="test-agent")

    @pytest.mark.asyncio
    async def test_response_structure(self, analyzer):
        """Should return proper response structure."""
        company_data = {"financials": {}}
        ratios = {"current_ratio": 1.5}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert "response" in result
        assert "prompt" in result
        assert "model_info" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_model_info_deterministic(self, analyzer):
        """Model info should indicate deterministic analysis."""
        company_data = {"financials": {}}
        ratios = {}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["model_info"]["model"].startswith("deterministic-")
        assert result["model_info"]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_metadata_contains_agent_id(self, analyzer):
        """Metadata should contain agent_id."""
        company_data = {"financials": {}}
        ratios = {}

        result = await analyzer.analyze_financial_health(company_data, ratios, "TEST")

        assert result["metadata"]["agent_id"] == "test-agent"


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_analyzer_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        a1 = get_deterministic_analyzer()
        a2 = get_deterministic_analyzer()
        assert a1 is a2
