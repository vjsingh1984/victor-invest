import asyncio
import json
from pathlib import Path

from victor.tools.registry import ToolRegistry
from victor.workflows.executor import (
    ExecutorNodeStatus,
    NodeResult,
    WorkflowExecutor,
    get_compute_handler,
    register_compute_handler,
)

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


def _load_expected(name: str):
    fixture_path = Path("tests/fixtures/victor_invest/golden") / name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _extract_golden_payload(result, symbol: str):
    return {
        "symbol": symbol,
        "fundamental_status": (result.context.get("fundamental_analysis") or {}).get("status"),
        "technical_status": (result.context.get("technical_analysis") or {}).get("status"),
        "synthesis_status": (result.context.get("synthesis") or {}).get("status"),
        "recommendation": (result.context.get("synthesis") or {}).get("recommendation"),
        "confidence": (result.context.get("synthesis") or {}).get("confidence"),
        "price_target": (result.context.get("synthesis") or {}).get("price_target"),
    }


def _run_standard_workflow(symbol: str, synthesis_output: dict):
    ensure_handlers_registered()
    provider = InvestmentWorkflowProvider()
    workflow = provider.get_workflow("standard")
    assert workflow is not None

    handlers = {
        "fetch_sec_data": _stub_handler({"status": "success", "data": {"source": "sec"}}),
        "fetch_market_data": _stub_handler({"status": "success", "data": {"source": "market"}}),
        "run_fundamental_analysis": _stub_handler({"status": "success", "data": {"score": 72}}),
        "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": "bullish"}}),
        "run_synthesis": _stub_handler(synthesis_output),
    }

    original = {name: get_compute_handler(name) for name in handlers}

    try:
        for name, handler in handlers.items():
            register_compute_handler(name, handler)

        executor = WorkflowExecutor(_MinimalOrchestrator(), tool_registry=ToolRegistry())
        return asyncio.run(
            executor.execute(workflow, initial_context={"symbol": symbol}, timeout=60.0)
        )
    finally:
        for name, handler in original.items():
            if handler is not None:
                register_compute_handler(name, handler)


def test_standard_workflow_golden_output_aapl():
    result = _run_standard_workflow(
        "AAPL",
        {
            "status": "success",
            "recommendation": "BUY",
            "confidence": "HIGH",
            "price_target": 220.0,
        },
    )
    assert result.success

    expected = _load_expected("standard_aapl.json")
    assert _extract_golden_payload(result, "AAPL") == expected


def test_standard_workflow_golden_output_msft():
    result = _run_standard_workflow(
        "MSFT",
        {
            "status": "success",
            "recommendation": "HOLD",
            "confidence": "MEDIUM",
            "price_target": 415.0,
        },
    )
    assert result.success

    expected = _load_expected("standard_msft.json")
    assert _extract_golden_payload(result, "MSFT") == expected
