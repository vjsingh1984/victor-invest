from unittest.mock import Mock

import pytest

from investigator.domain.agents.base import (
    InvestmentAgent,
    get_cache_type_for_analysis,
)
from investigator.domain.models.analysis import (
    AgentCapability,
    AgentResult,
    AgentTask,
    AnalysisType,
    Priority,
    TaskStatus,
)
from investigator.infrastructure.cache.cache_types import CacheType


class DummyAgent(InvestmentAgent):
    def register_capabilities(self):
        return [
            AgentCapability(
                analysis_type=AnalysisType.SEC_FUNDAMENTAL,
                min_data_required={"symbol": str, "filing_type": str},
                max_processing_time=60,
                required_models=["mock-model"],
                cache_ttl=3600,
            )
        ]

    async def process(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={"ok": True},
            processing_time=0.1,
        )


def test_cache_type_mapping() -> None:
    assert get_cache_type_for_analysis(AnalysisType.SEC_FUNDAMENTAL) == CacheType.SEC_RESPONSE
    assert get_cache_type_for_analysis(AnalysisType.TECHNICAL_ANALYSIS) == CacheType.TECHNICAL_DATA
    assert get_cache_type_for_analysis(AnalysisType.INVESTMENT_SYNTHESIS) == CacheType.LLM_RESPONSE


def test_agent_can_handle_task_with_valid_context():
    agent = DummyAgent("dummy", ollama_client=Mock(), event_bus=Mock(), cache_manager=Mock())
    task = AgentTask(
        task_id="task-1",
        symbol="AAPL",
        analysis_type=AnalysisType.SEC_FUNDAMENTAL,
        priority=Priority.HIGH,
        context={"symbol": "AAPL", "filing_type": "10-Q"},
    )

    import asyncio

    assert asyncio.run(agent.can_handle_task(task)) is True


def test_agent_rejects_task_when_context_incomplete():
    agent = DummyAgent("dummy", ollama_client=Mock(), event_bus=Mock(), cache_manager=Mock())
    task = AgentTask(
        task_id="task-2",
        symbol="AAPL",
        analysis_type=AnalysisType.SEC_FUNDAMENTAL,
        context={"symbol": "AAPL"},
    )

    import asyncio

    assert asyncio.run(agent.can_handle_task(task)) is False
