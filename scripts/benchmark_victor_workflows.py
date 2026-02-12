#!/usr/bin/env python3
"""Benchmark Victor workflow modes against latency budgets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure local package imports work when executed as a script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from victor_invest.latency_budgets import evaluate_latency


def _run_analysis_mode_cli(symbol: str, mode: str, force_refresh: bool) -> Dict[str, object]:
    cmd = [
        sys.executable,
        "-m",
        "victor_invest.cli",
        "analyze",
        symbol,
        "--mode",
        mode,
    ]
    if force_refresh:
        cmd.append("--force-refresh")

    started_at = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "elapsed_seconds": elapsed_seconds,
        "exit_code": proc.returncode,
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }


def _stub_handler(output):
    from victor.workflows.executor import ExecutorNodeStatus, NodeResult

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


def _build_stub_handlers_for_mode(mode: str) -> Dict[str, object]:
    mode_lower = mode.lower().strip()

    if mode_lower == "quick":
        return {
            "fetch_market_data": _stub_handler({"status": "success", "data": {"source": "market"}}),
            "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": "bullish"}}),
            "run_synthesis": _stub_handler(
                {
                    "status": "success",
                    "recommendation": "HOLD",
                    "confidence": "MEDIUM",
                    "price_target": 200.0,
                }
            ),
        }

    if mode_lower == "standard":
        return {
            "fetch_sec_data": _stub_handler({"status": "success", "data": {"source": "sec"}}),
            "fetch_market_data": _stub_handler({"status": "success", "data": {"source": "market"}}),
            "run_fundamental_analysis": _stub_handler({"status": "success", "data": {"score": 72}}),
            "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": "bullish"}}),
            "run_synthesis": _stub_handler(
                {
                    "status": "success",
                    "recommendation": "BUY",
                    "confidence": "HIGH",
                    "price_target": 220.0,
                }
            ),
        }

    if mode_lower == "comprehensive":
        return {
            "fetch_sec_data": _stub_handler({"status": "success", "data": {"source": "sec"}}),
            "fetch_market_data": _stub_handler({"status": "success", "data": {"source": "market"}}),
            "fetch_macro_data": _stub_handler({"status": "success", "data": {"regime": "neutral"}}),
            "run_fundamental_analysis": _stub_handler({"status": "success", "data": {"score": 72}}),
            "run_technical_analysis": _stub_handler({"status": "success", "data": {"trend": "bullish"}}),
            "run_market_context_analysis": _stub_handler({"status": "success", "market_regime": "neutral"}),
            "identify_peers": _stub_handler({"peers": ["MSFT"], "peer_metrics": {"MSFT": {}}}),
            "run_synthesis": _stub_handler(
                {
                    "status": "success",
                    "recommendation": "BUY",
                    "confidence": "HIGH",
                    "price_target": 220.0,
                }
            ),
            "generate_report": _stub_handler({"status": "success", "path": "report.pdf"}),
        }

    raise ValueError(f"Unsupported mode for stub benchmarking: {mode}")


def _run_analysis_mode_stub(symbol: str, mode: str) -> Dict[str, object]:
    try:
        from victor.tools.registry import ToolRegistry
        from victor.workflows.executor import WorkflowExecutor, get_compute_handler, register_compute_handler
        from victor_invest.workflows import InvestmentWorkflowProvider, ensure_handlers_registered
    except ModuleNotFoundError as exc:
        return {
            "elapsed_seconds": 0.0,
            "exit_code": 1,
            "stderr_tail": f"Stub runner dependency missing: {exc}",
        }

    class _MinimalOrchestrator:
        pass

    ensure_handlers_registered()
    provider = InvestmentWorkflowProvider()
    workflow = provider.get_workflow(mode)
    if workflow is None:
        return {
            "elapsed_seconds": 0.0,
            "exit_code": 1,
            "stderr_tail": f"Unknown workflow: {mode}",
        }

    handlers = _build_stub_handlers_for_mode(mode)
    original = {name: get_compute_handler(name) for name in handlers}

    try:
        for name, handler in handlers.items():
            register_compute_handler(name, handler)

        executor = WorkflowExecutor(_MinimalOrchestrator(), tool_registry=ToolRegistry())
        started_at = time.perf_counter()
        result = executor.execute(workflow, initial_context={"symbol": symbol}, timeout=60.0)

        # Keep asyncio import local so --help works without importing event loop modules.
        import asyncio

        workflow_result = asyncio.run(result)
        elapsed_seconds = time.perf_counter() - started_at

        return {
            "elapsed_seconds": elapsed_seconds,
            "exit_code": 0 if workflow_result.success else 1,
            "stderr_tail": "" if workflow_result.success else "Workflow execution returned failure",
        }
    finally:
        for name, handler in original.items():
            if handler is not None:
                register_compute_handler(name, handler)


def _run_analysis_mode(symbol: str, mode: str, force_refresh: bool, runner: str) -> Dict[str, object]:
    if runner == "cli":
        return _run_analysis_mode_cli(symbol, mode, force_refresh)
    if runner == "stub":
        return _run_analysis_mode_stub(symbol, mode)
    raise ValueError(f"Unsupported runner: {runner}")


def _benchmark_mode(
    symbol: str,
    mode: str,
    force_refresh: bool,
    runner: str,
    budget_profile: str,
) -> Dict[str, object]:
    run_result = _run_analysis_mode(symbol=symbol, mode=mode, force_refresh=force_refresh, runner=runner)
    evaluation = evaluate_latency(mode, run_result["elapsed_seconds"], profile=budget_profile)

    return {
        "mode": evaluation.mode,
        "runner": runner,
        "budget_profile": evaluation.profile,
        "symbol": symbol,
        "elapsed_seconds": round(evaluation.elapsed_seconds, 3),
        "budget_seconds": evaluation.budget_seconds,
        "delta_seconds": round(evaluation.delta_seconds, 3),
        "passed": evaluation.passed and run_result["exit_code"] == 0,
        "exit_code": run_result["exit_code"],
        "stderr_tail": run_result["stderr_tail"],
    }


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL", help="Symbol to benchmark (default: AAPL)")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["quick", "standard", "comprehensive"],
        help="Workflow modes to benchmark",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass caches for worst-case latency measurements",
    )
    parser.add_argument(
        "--runner",
        choices=["stub", "cli"],
        default="stub",
        help="Execution runner: 'stub' for deterministic CI, 'cli' for end-to-end CLI benchmarking",
    )
    parser.add_argument(
        "--budget-profile",
        choices=["production", "ci_stub"],
        help="Latency budget profile. Defaults to ci_stub for stub runner and production for cli runner.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write full benchmark report to a JSON file",
    )
    parser.add_argument(
        "--fail-on-budget-breach",
        action="store_true",
        help="Exit non-zero when any mode exceeds budget or command fails",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    budget_profile = args.budget_profile or ("ci_stub" if args.runner == "stub" else "production")

    results = [
        _benchmark_mode(
            symbol=args.symbol,
            mode=mode,
            force_refresh=args.force_refresh,
            runner=args.runner,
            budget_profile=budget_profile,
        )
        for mode in args.modes
    ]
    failed = [item for item in results if not item["passed"]]

    summary = {
        "symbol": args.symbol,
        "modes": args.modes,
        "runner": args.runner,
        "budget_profile": budget_profile,
        "failed_count": len(failed),
        "results": results,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.fail_on_budget_breach and failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
