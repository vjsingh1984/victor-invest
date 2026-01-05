"""Unit tests for the clean-architecture orchestrator."""

from unittest.mock import Mock

import networkx as nx
import pytest

from investigator.application import AgentOrchestrator, AnalysisMode, OrchestrationTask, Priority
from investigator.application import orchestrator as orchestrator_module


@pytest.fixture
def orchestrator_instance():
    return AgentOrchestrator(cache_manager=Mock(), metrics_collector=Mock())


class TestAgentOrchestrator:
    """Tests covering the public surface of AgentOrchestrator."""

    def test_analysis_mode_enum(self):
        assert {mode.name for mode in AnalysisMode} >= {"QUICK", "STANDARD", "COMPREHENSIVE"}

    def test_priority_enum(self):
        assert {member.name for member in Priority} >= {"CRITICAL", "HIGH", "NORMAL", "LOW"}

    def test_orchestrator_exposes_async_api(self, orchestrator_instance):
        required_methods = ["start", "stop", "analyze", "analyze_batch", "get_status", "get_results"]
        for method in required_methods:
            assert hasattr(orchestrator_instance, method)


class TestOrchestrationWorkflow:
    """Dependency graph and task scheduling behaviour."""

    def test_dependency_graph_structure(self, orchestrator_instance):
        graph = orchestrator_instance.dependency_graph
        expected_nodes = {"sec", "technical", "fundamental", "market_context", "symbol_update", "synthesis"}
        assert set(graph.nodes) == expected_nodes
        assert nx.is_directed_acyclic_graph(graph)
        assert set(graph.predecessors("synthesis")) == {
            "sec",
            "technical",
            "fundamental",
            "market_context",
            "symbol_update",
        }

    def test_task_priority_ordering(self):
        high = OrchestrationTask(id="1", symbol="AAPL", mode=AnalysisMode.QUICK, agents=["sec"], priority=Priority.HIGH)
        low = OrchestrationTask(id="2", symbol="MSFT", mode=AnalysisMode.QUICK, agents=["sec"], priority=Priority.LOW)
        assert high < low

    def test_get_agents_for_modes(self, orchestrator_instance):
        assert orchestrator_instance._get_agents_for_mode("TEST", AnalysisMode.QUICK, []) == [
            "technical",
            "market_context",
        ]
        standard_agents = orchestrator_instance._get_agents_for_mode("TEST", AnalysisMode.STANDARD, [])
        assert standard_agents == ["sec", "technical", "fundamental", "symbol_update", "market_context", "synthesis"]
        custom_agents = orchestrator_instance._get_agents_for_mode("TEST", AnalysisMode.CUSTOM, ["sec"])
        assert custom_agents == ["sec"]

    def test_analyze_enqueues_task_with_priority(self, orchestrator_instance):
        async def runner():
            task_id = await orchestrator_instance.analyze("NVDA", AnalysisMode.QUICK, Priority.CRITICAL)
            priority_value, task = await orchestrator_instance.task_queue.get()

            assert task_id.startswith("NVDA_")
            assert priority_value == Priority.CRITICAL.value
            assert task.symbol == "NVDA"
            assert task.mode == AnalysisMode.QUICK
            assert task.agents == ["technical", "market_context"]

        import asyncio

        asyncio.run(runner())


class TestOrchestratorResilience:
    """Lifecycle and resilience behaviours."""

    def test_logger_available_when_market_data_init_fails(self, monkeypatch):
        def raising_fetcher(*args, **kwargs):
            raise RuntimeError("db unavailable")

        # Patch the actual function used by orchestrator
        monkeypatch.setattr(orchestrator_module, "get_market_data_fetcher", raising_fetcher)
        orch = AgentOrchestrator(cache_manager=Mock(), metrics_collector=Mock())
        assert orch.logger is not None
        assert orch.market_data_fetcher is None

    @pytest.mark.asyncio
    async def test_start_stop_cancels_background_tasks(self, monkeypatch):
        class DummyPool:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize_servers(self):
                return None

            async def get_pool_status(self):
                return {"available_servers": 1, "total_servers": 1, "total_capacity_gb": 1}

        class DummyCache:
            async def start_cleanup_service(self, interval_seconds=3600):
                return None

            async def stop_cleanup_service(self):
                return None

        monkeypatch.setattr(orchestrator_module, "create_resource_aware_pool", lambda cfg: DummyPool())
        monkeypatch.setattr(AgentOrchestrator, "_initialize_agents", lambda self: {})

        orch = AgentOrchestrator(
            cache_manager=DummyCache(), metrics_collector=Mock(), max_concurrent_analyses=0, max_concurrent_agents=0
        )

        await orch.start()
        assert len(orch._background_tasks) == 2

        await orch.stop()

        assert all(task.cancelled() or task.done() for task in orch._background_tasks)

    @pytest.mark.asyncio
    async def test_metrics_loop_handles_exceptions_and_exit(self, monkeypatch):
        call_count = 0

        async def fast_sleep(_seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                orch.running = False

        class MetricsCollector:
            def record_orchestrator_stats(self, _stats):
                raise ValueError("metrics sink unavailable")

        orch = AgentOrchestrator(cache_manager=Mock(), metrics_collector=MetricsCollector())
        orch.performance_stats["successful_analyses"] = 1
        orch.performance_stats["total_analyses"] = 1
        orch.running = True

        monkeypatch.setattr(orchestrator_module.asyncio, "sleep", fast_sleep)

        task = orchestrator_module.asyncio.create_task(orch._report_metrics())
        await task
        assert call_count >= 2
