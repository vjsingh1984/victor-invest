"""Contract tests for FundamentalAnalysisAgent data-quality delegation."""

from unittest.mock import MagicMock, patch

import pytest

from investigator.domain.agents.fundamental.agent import FundamentalAnalysisAgent


@pytest.mark.parametrize(
    ("agent_method", "assessor_method", "payload"),
    [
        ("_assess_quarter_quality", "assess_quarter_quality", {"revenues": 1}),
        ("_assess_data_quality", "assess_data_quality", ({"symbol": "AAPL"}, {"pe_ratio": 10})),
        (
            "_calculate_confidence_level",
            "calculate_confidence_level",
            {"data_quality_score": 85, "quality_grade": "Good", "consistency_issues": []},
        ),
    ],
)
def test_data_quality_methods_delegate_to_assessor(agent_method, assessor_method, payload):
    """Each data-quality agent method should forward work to DataQualityAssessor."""
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent._get_data_quality_assessor = MagicMock()
    setattr(
        agent,
        agent_method,
        getattr(FundamentalAnalysisAgent, agent_method).__get__(agent),
    )

    expected = {"source": assessor_method}
    assessor = MagicMock()
    setattr(assessor, assessor_method, MagicMock(return_value=expected))
    agent._get_data_quality_assessor.return_value = assessor

    if isinstance(payload, tuple):
        result = getattr(agent, agent_method)(*payload)
        getattr(assessor, assessor_method).assert_called_once_with(*payload)
    else:
        result = getattr(agent, agent_method)(payload)
        getattr(assessor, assessor_method).assert_called_once_with(payload)

    assert result == expected
    agent._get_data_quality_assessor.assert_called_once_with()


def test_get_data_quality_assessor_caches_singleton_resolution():
    """Resolver should be invoked once per agent instance and then cached."""
    agent = MagicMock(spec=FundamentalAnalysisAgent)
    agent._get_data_quality_assessor = FundamentalAnalysisAgent._get_data_quality_assessor.__get__(agent)

    assessor = MagicMock()
    with patch(
        "investigator.domain.agents.fundamental.agent.get_data_quality_assessor",
        return_value=assessor,
    ) as resolver:
        first = agent._get_data_quality_assessor()
        second = agent._get_data_quality_assessor()

    assert first is assessor
    assert second is assessor
    resolver.assert_called_once()
