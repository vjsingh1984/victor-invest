import asyncio

from victor.workflows.executor import (
    ExecutorNodeStatus,
    NodeResult,
    WorkflowExecutor,
    get_compute_handler,
    register_compute_handler,
)
from victor.tools.registry import ToolRegistry

from victor_invest.workflows import InvestmentWorkflowProvider, ensure_handlers_registered


class _MinimalOrchestrator:
    pass


def _stub_handler(output):
    async def _handler(node, context, tool_registry):
        output_key = node.output_key or node.id
        context.set(output_key, output)
        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=output,
            duration_seconds=0.0,
            tool_calls_used=0,
        )

    return _handler


def test_quick_workflow_executes_with_stub_handlers():
    ensure_handlers_registered()
    provider = InvestmentWorkflowProvider()
    workflow = provider.get_workflow("quick")
    assert workflow is not None

    handlers = {
        "fetch_market_data": _stub_handler({"status": "success", "data": {}}),
        "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": {}}}),
        "run_synthesis": _stub_handler({"status": "success", "recommendation": "HOLD"}),
    }

    original = {name: get_compute_handler(name) for name in handlers}

    try:
        for name, handler in handlers.items():
            register_compute_handler(name, handler)

        executor = WorkflowExecutor(_MinimalOrchestrator(), tool_registry=ToolRegistry())
        result = asyncio.run(
            executor.execute(workflow, initial_context={"symbol": "AAPL"}, timeout=60.0)
        )

        assert result.success
        assert result.context.get("synthesis") is not None
    finally:
        for name, handler in original.items():
            if handler is not None:
                register_compute_handler(name, handler)


def test_standard_workflow_executes_with_stub_handlers():
    ensure_handlers_registered()
    provider = InvestmentWorkflowProvider()
    workflow = provider.get_workflow("standard")
    assert workflow is not None

    handlers = {
        "fetch_sec_data": _stub_handler({"status": "success", "data": {}}),
        "fetch_market_data": _stub_handler({"status": "success", "data": {}}),
        "run_fundamental_analysis": _stub_handler({"status": "success", "data": {"score": 70}}),
        "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": {}}}),
        "run_synthesis": _stub_handler({"status": "success", "recommendation": "HOLD"}),
    }

    original = {name: get_compute_handler(name) for name in handlers}

    try:
        for name, handler in handlers.items():
            register_compute_handler(name, handler)

        executor = WorkflowExecutor(_MinimalOrchestrator(), tool_registry=ToolRegistry())
        result = asyncio.run(
            executor.execute(workflow, initial_context={"symbol": "AAPL"}, timeout=60.0)
        )

        assert result.success
        assert result.context.get("fundamental_analysis") is not None
        assert result.context.get("technical_analysis") is not None
        assert result.context.get("synthesis") is not None
    finally:
        for name, handler in original.items():
            if handler is not None:
                register_compute_handler(name, handler)


def test_comprehensive_workflow_executes_with_stub_handlers():
    ensure_handlers_registered()
    provider = InvestmentWorkflowProvider()
    workflow = provider.get_workflow("comprehensive")
    assert workflow is not None

    handlers = {
        "fetch_sec_data": _stub_handler({"status": "success", "data": {}}),
        "fetch_market_data": _stub_handler({"status": "success", "data": {}}),
        "fetch_macro_data": _stub_handler({"status": "success", "data": {}}),
        "run_fundamental_analysis": _stub_handler({"status": "success", "data": {"score": 70}}),
        "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": {}}}),
        "run_market_context_analysis": _stub_handler({"status": "success", "market_regime": "neutral"}),
        "identify_peers": _stub_handler({"peers": [], "peer_metrics": {}}),
        "run_synthesis": _stub_handler({"status": "success", "recommendation": "HOLD"}),
        "generate_report": _stub_handler({"status": "success", "path": "report.pdf"}),
    }

    original = {name: get_compute_handler(name) for name in handlers}

    try:
        for name, handler in handlers.items():
            register_compute_handler(name, handler)

        executor = WorkflowExecutor(_MinimalOrchestrator(), tool_registry=ToolRegistry())
        result = asyncio.run(
            executor.execute(workflow, initial_context={"symbol": "AAPL"}, timeout=60.0)
        )

        assert result.success
        assert result.context.get("fundamental_analysis") is not None
        assert result.context.get("technical_analysis") is not None
        assert result.context.get("synthesis") is not None
        assert result.context.get("report") is not None
    finally:
        for name, handler in original.items():
            if handler is not None:
                register_compute_handler(name, handler)
