"""Contract tests for FundamentalAnalysisAgent trend delegation."""

from unittest.mock import MagicMock, patch

import pytest

from investigator.domain.agents.fundamental.agent import FundamentalAnalysisAgent


@pytest.mark.parametrize(
    ("agent_method", "analyzer_method"),
    [
        ("_analyze_revenue_trend", "analyze_revenue_trend"),
        ("_analyze_margin_trend", "analyze_margin_trend"),
        ("_analyze_cash_flow_trend", "analyze_cash_flow_trend"),
        ("_calculate_quarterly_comparisons", "calculate_quarterly_comparisons"),
        ("_detect_cyclical_patterns", "detect_cyclical_patterns"),
    ],
)
def test_trend_methods_delegate_to_trend_analyzer(agent_method, analyzer_method):
    """Each trend-related agent method should forward work to TrendAnalyzer."""
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent._get_trend_analyzer = MagicMock()
    setattr(
        agent,
        agent_method,
        getattr(FundamentalAnalysisAgent, agent_method).__get__(agent),
    )

    quarterly_data = [object()]
    expected = {"source": analyzer_method}

    analyzer = MagicMock()
    setattr(analyzer, analyzer_method, MagicMock(return_value=expected))
    agent._get_trend_analyzer.return_value = analyzer

    result = getattr(agent, agent_method)(quarterly_data)

    assert result == expected
    agent._get_trend_analyzer.assert_called_once_with()
    getattr(analyzer, analyzer_method).assert_called_once_with(quarterly_data)


def test_get_trend_analyzer_caches_singleton_resolution():
    """Resolver should be invoked once per agent instance and then cached."""
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent._get_trend_analyzer = FundamentalAnalysisAgent._get_trend_analyzer.__get__(agent)

    analyzer = MagicMock()
    with patch("investigator.domain.agents.fundamental.agent.get_trend_analyzer", return_value=analyzer) as resolver:
        first = agent._get_trend_analyzer()
        second = agent._get_trend_analyzer()

    assert first is analyzer
    assert second is analyzer
    resolver.assert_called_once()
