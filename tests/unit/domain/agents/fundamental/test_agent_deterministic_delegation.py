"""Contract tests for FundamentalAnalysisAgent deterministic-analysis delegation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investigator.domain.agents.fundamental.agent import FundamentalAnalysisAgent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_method", "analyzer_method", "args"),
    [
        (
            "_analyze_financial_health",
            "analyze_financial_health",
            ({"financials": {}}, {"current_ratio": 1.5}, "AAPL"),
        ),
        (
            "_analyze_growth",
            "analyze_growth",
            ({"trend_analysis": {}}, "AAPL"),
        ),
        (
            "_analyze_profitability",
            "analyze_profitability",
            ({"trend_analysis": {}}, {"gross_margin": 0.2}, "AAPL"),
        ),
    ],
)
async def test_deterministic_methods_delegate_to_deterministic_analyzer(agent_method, analyzer_method, args):
    """Each deterministic agent method should forward work to DeterministicAnalyzer."""
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent._get_deterministic_analyzer = MagicMock()
    setattr(
        agent,
        agent_method,
        getattr(FundamentalAnalysisAgent, agent_method).__get__(agent),
    )

    expected = {"response": {"source": analyzer_method}}
    analyzer = MagicMock()
    setattr(analyzer, analyzer_method, AsyncMock(return_value=expected))
    agent._get_deterministic_analyzer.return_value = analyzer

    result = await getattr(agent, agent_method)(*args)

    assert result == expected
    agent._get_deterministic_analyzer.assert_called_once_with()
    getattr(analyzer, analyzer_method).assert_awaited_once_with(*args)


def test_get_deterministic_analyzer_caches_per_agent_instance():
    """Analyzer should be created once and cached on the agent instance."""
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent.agent_id = "agent-test"
    agent.logger = MagicMock()
    agent._get_deterministic_analyzer = FundamentalAnalysisAgent._get_deterministic_analyzer.__get__(agent)

    analyzer = MagicMock()
    with patch("investigator.domain.agents.fundamental.agent.DeterministicAnalyzer", return_value=analyzer) as cls:
        first = agent._get_deterministic_analyzer()
        second = agent._get_deterministic_analyzer()

    assert first is analyzer
    assert second is analyzer
    cls.assert_called_once_with(agent_id="agent-test", logger=agent.logger)
