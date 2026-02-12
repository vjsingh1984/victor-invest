# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Investment Analysis Workflows.

This package provides workflow definitions for investment analysis tasks:
- Quick analysis (technical only)
- Standard analysis (technical + fundamental)
- Comprehensive analysis (full institutional-grade)
- RL backtest (historical backtesting)
- Peer comparison (relative valuation)

Uses Victor's YAML-first architecture with Python escape hatches for complex
conditions and transforms that cannot be expressed in YAML.

Example:
    from victor_invest.workflows import InvestmentWorkflowProvider

    provider = InvestmentWorkflowProvider()

    # Agentic workflow execution (with LLM support)
    result = await provider.run_agentic_workflow(
        "comprehensive",
        context={"symbol": "AAPL"},
        provider="ollama",
        model="gpt-oss:20b",
    )
    if result.success:
        synthesis = result.context.get("synthesis")
        print(f"Recommendation: {synthesis.get('recommendation')}")

    # Compute-only workflow execution (no orchestrator needed)
    result = await provider.run_workflow_with_handlers(
        "comprehensive",
        context={"symbol": "AAPL"},
    )

Available workflows (all YAML-defined):
- quick: Technical analysis only (~5 seconds)
- standard: Technical + Fundamental (~30 seconds)
- comprehensive: Full institutional-grade analysis (~60 seconds)
- rl_backtest: Historical backtesting for RL training
- peer_comparison: Peer group relative analysis

Architecture Decision: Direct Tool Invocation + Context Stuffing
================================================================
This package follows Victor's architecture:
- Phase 1-2: Direct tool/handler calls (deterministic, no LLM)
- Phase 3: Single LLM inference with all data (context stuffing)

Handlers are defined in victor_invest.handlers and registered with Victor's
workflow handler registry. YAML workflows reference handlers by path.

Note on Execution Models:
- run_agentic_workflow(): Uses WorkflowExecutor with orchestrator for agent nodes
- run_workflow_with_handlers(): Uses WorkflowExecutor for compute handlers
- run_compiled_workflow(): Uses UnifiedWorkflowCompiler (LangGraph) for transforms
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider

if TYPE_CHECKING:
    from victor.workflows.executor import WorkflowResult

from victor_invest.workflows.graphs import (
    build_comprehensive_graph,
    build_graph_for_mode,
    build_quick_graph,
    build_standard_graph,
    run_analysis,
    run_yaml_analysis,
)
from victor_invest.workflows.rl_backtest import (
    RLBacktestWorkflowState,
    build_rl_backtest_graph,
    generate_lookback_list,
    run_rl_backtest,
    run_rl_backtest_batch,
)
from victor_invest.workflows.state import AnalysisMode, AnalysisWorkflowState

logger = logging.getLogger(__name__)


class InvestmentWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides investment-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading and caching
    - Escape hatches registration from victor_invest.escape_hatches
    - Unified workflow compilation via UnifiedWorkflowCompiler

    Example:
        provider = InvestmentWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Agentic execution (with LLM synthesis via orchestrator)
        result = await provider.run_agentic_workflow(
            "comprehensive",
            context={"symbol": "AAPL"},
            provider="ollama",
        )

        # Compute-only execution (uses registered handlers)
        result = await provider.run_workflow_with_handlers(
            "comprehensive",
            context={"symbol": "AAPL"},
        )

    Execution Models:
        - run_agentic_workflow(): Full orchestrator support for agent nodes
        - run_workflow_with_handlers(): WorkflowExecutor for compute handlers
        - run_compiled_workflow(): UnifiedWorkflowCompiler for LangGraph transforms
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for investment escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor_invest.escape_hatches"

    def _get_workflows_directory(self) -> Path:
        """Return the directory containing YAML workflow files.

        Returns:
            Path to victor_invest/workflows/ directory
        """
        return Path(__file__).parent

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        Returns:
            List of (regex_pattern, workflow_name) tuples for auto-triggering
        """
        return [
            (r"quick\s+analysis", "quick"),
            (r"analyze\s+\w+\s+quickly", "quick"),
            (r"standard\s+analysis", "standard"),
            (r"analyze\s+stock", "standard"),
            (r"comprehensive\s+analysis", "comprehensive"),
            (r"full\s+analysis", "comprehensive"),
            (r"institutional.*analysis", "comprehensive"),
            (r"deep\s+dive", "comprehensive"),
            (r"backtest", "rl_backtest"),
            (r"rl\s+training", "rl_backtest"),
            (r"peer\s+comparison", "peer_comparison"),
            (r"compare.*peers", "peer_comparison"),
            (r"relative\s+valuation", "peer_comparison"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get appropriate workflow for task type.

        Args:
            task_type: Type of task (e.g., "analysis", "backtest")

        Returns:
            Workflow name string or None if no mapping exists
        """
        mapping = {
            "quick": "quick",
            "standard": "standard",
            "comprehensive": "comprehensive",
            "analysis": "standard",
            "research": "comprehensive",
            "backtest": "rl_backtest",
            "rl": "rl_backtest",
            "peer": "peer_comparison",
            "comparison": "peer_comparison",
        }
        return mapping.get(task_type.lower())

    async def run_agentic_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> "WorkflowResult":
        """Execute a YAML workflow with full agent node support.

        Uses Victor's public Agent.create() API for proper orchestrator creation,
        enabling agent nodes for LLM reasoning. This approach:
        - Leverages Victor's unified provider abstraction
        - Applies vertical-specific configuration automatically
        - Follows the framework's golden path for agent creation

        For compute-only workflows, use the simpler `run_workflow()` method
        inherited from BaseYAMLWorkflowProvider.

        Args:
            workflow_name: Name of the YAML workflow (e.g., "comprehensive")
            context: Initial context data (e.g., {"symbol": "AAPL"})
            provider: LLM provider ("ollama", "anthropic", "openai")
            model: Model identifier. If None, uses provider default.
            timeout: Optional overall timeout in seconds (default: 300)

        Returns:
            WorkflowResult with execution outcome and outputs

        Raises:
            ValueError: If workflow_name is not found

        Example:
            provider = InvestmentWorkflowProvider()
            result = await provider.run_agentic_workflow(
                "comprehensive",
                {"symbol": "AAPL"},
                provider="ollama",
                model="gpt-oss:20b",
            )
            if result.success:
                synthesis = result.context.get("synthesis")
                print(f"Recommendation: {synthesis.get('recommendation')}")
        """
        from victor.workflows.executor import WorkflowExecutor

        from victor_invest.framework_bootstrap import create_investment_orchestrator

        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        orchestrator = await create_investment_orchestrator(
            provider=provider,
            model=model,
            ensure_handlers=ensure_handlers_registered,
            warning_callback=logger.warning,
        )

        # Create executor with proper orchestrator
        executor = WorkflowExecutor(
            orchestrator,
            max_parallel=4,
            default_timeout=timeout or 300.0,
        )

        # Execute workflow with initial context
        return await executor.execute(
            workflow,
            initial_context=context or {},
            timeout=timeout,
        )

    async def run_workflow_with_handlers(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> "WorkflowResult":
        """Execute a YAML workflow using registered compute handlers.

        This method executes workflows using WorkflowExecutor with the handlers
        registered via register_compute_handler(). This is the recommended method
        for running investment workflows that use the context-stuffing pattern.

        Note: This method uses WorkflowExecutor (handler-based execution) rather
        than UnifiedWorkflowCompiler (LangGraph-based execution). The handlers
        in victor_invest.handlers are designed for WorkflowExecutor.

        For workflows with full agent node support (LLM reasoning), use
        run_agentic_workflow() instead.

        Args:
            workflow_name: Name of the YAML workflow (e.g., "comprehensive")
            context: Initial context data (e.g., {"symbol": "AAPL"})
            timeout: Optional overall timeout in seconds (default: 300)

        Returns:
            WorkflowResult with execution outcome and outputs

        Raises:
            ValueError: If workflow_name is not found

        Example:
            provider = InvestmentWorkflowProvider()
            result = await provider.run_workflow_with_handlers(
                "comprehensive",
                context={"symbol": "AAPL"},
            )
            if result.success:
                synthesis = result.context.get("synthesis")
                print(f"Recommendation: {synthesis.get('recommendation')}")
        """
        import warnings

        from victor.tools.registry import ToolRegistry
        from victor.workflows.executor import WorkflowExecutor

        # Ensure handlers are registered
        ensure_handlers_registered()

        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Create minimal mock orchestrator for compute-only workflows
        # Agent nodes would fail, but compute handlers work fine
        class _MinimalOrchestrator:
            pass

        orchestrator = _MinimalOrchestrator()

        # Create tool registry (handlers may need it)
        tool_registry = ToolRegistry()

        # Register investment tools for compute node tool access
        try:
            from victor_invest.tools import register_investment_tools

            stats = register_investment_tools(tool_registry)
            if stats.get("errors"):
                logger.warning("Some investment tools failed to register: %s", stats["errors"])
        except Exception as exc:
            logger.warning("Investment tool registration failed: %s", exc)

        # Create executor with handler support
        executor = WorkflowExecutor(
            orchestrator,
            tool_registry=tool_registry,
            max_parallel=4,
            default_timeout=timeout or 300.0,
        )

        # Execute workflow with initial context
        return await executor.execute(
            workflow,
            initial_context=context or {},
            timeout=timeout,
        )

    # Inherited from BaseYAMLWorkflowProvider:
    # - run_compiled_workflow(): Uses UnifiedWorkflowCompiler (LangGraph)
    # - stream_compiled_workflow(): Streams via UnifiedWorkflowCompiler
    # - compile_workflow(): Returns CachedCompiledGraph for manual execution


# Lazy handler registration to prevent circular imports
_handlers_registered = False


def ensure_handlers_registered() -> None:
    """Register Investment domain handlers lazily on first use.

    This lazy registration pattern prevents circular imports that can occur
    when handlers.py imports from workflows or related modules during module
    initialization. Handlers are registered once on first workflow execution.
    """
    global _handlers_registered
    if _handlers_registered:
        return
    from victor_invest.handlers import register_handlers
    from victor.framework.handler_registry import sync_handlers_with_executor

    register_handlers()
    sync_handlers_with_executor(direction="to_executor")
    _handlers_registered = True


__all__ = [
    # YAML-first workflow provider
    "InvestmentWorkflowProvider",
    # Lazy handler registration
    "ensure_handlers_registered",
    # Analysis state definitions
    "AnalysisMode",
    "AnalysisWorkflowState",
    # Analysis graph builders (Python-based, for backwards compatibility)
    "build_graph_for_mode",
    "build_quick_graph",
    "build_standard_graph",
    "build_comprehensive_graph",
    # Analysis convenience
    "run_analysis",
    "run_yaml_analysis",
    # RL Backtest state
    "RLBacktestWorkflowState",
    # RL Backtest graph builders
    "build_rl_backtest_graph",
    # RL Backtest convenience
    "run_rl_backtest",
    "run_rl_backtest_batch",
    "generate_lookback_list",
]
