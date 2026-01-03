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

"""Investment workflows for victor-invest.

This package provides workflow definitions for investment analysis:
- Single stock analysis (investigator_v2.sh)
- Batch analysis (batch_analysis_runner.py)
- RL backtesting (rl_backtest.py)

Supports hybrid loading:
- Python workflows (inline @workflow definitions)
- YAML workflows (external files in workflows/*.yaml)

YAML workflows override Python workflows when names collide.

Example:
    from investigator.workflows import InvestmentWorkflowProvider

    provider = InvestmentWorkflowProvider()
    workflows = provider.get_workflows()

    # Execute single stock analysis
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(
        provider.get_workflow("single_stock_analysis"),
        {"symbol": "AAPL"}
    )
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.workflows.executor import WorkflowExecutor
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor


try:
    from victor.core.verticals.protocols import WorkflowProviderProtocol
    from victor.workflows.definition import WorkflowDefinition

    class InvestmentWorkflowProvider(WorkflowProviderProtocol):
        """Provides investment-specific workflows.

        Implements WorkflowProviderProtocol with hybrid Python/YAML loading.
        YAML workflows override Python workflows when names collide.

        Example:
            provider = InvestmentWorkflowProvider()
            workflow = provider.get_workflow("single_stock_analysis")
            executor = provider.create_executor(orchestrator)
            result = await executor.execute(workflow, {"symbol": "AAPL"})
        """

        def __init__(self) -> None:
            """Initialize the provider."""
            self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

        def _load_python_workflows(self) -> Dict[str, WorkflowDefinition]:
            """Load Python-defined workflows.

            Returns:
                Dict mapping workflow names to definitions
            """
            # No Python workflows defined yet - all in YAML
            return {}

        def _load_yaml_workflows(self) -> Dict[str, WorkflowDefinition]:
            """Load YAML-defined workflows from workflows/*.yaml.

            Returns:
                Dict mapping workflow names to definitions
            """
            try:
                from victor.workflows.yaml_loader import load_workflows_from_directory

                # Load from the workflows directory (same as this file)
                workflows_dir = Path(__file__).parent
                yaml_workflows = load_workflows_from_directory(workflows_dir)
                logger.debug(f"Loaded {len(yaml_workflows)} YAML workflows from {workflows_dir}")
                return yaml_workflows
            except Exception as e:
                logger.warning(f"Failed to load YAML workflows: {e}")
                return {}

        def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
            """Lazy load all workflows with hybrid Python/YAML support.

            YAML workflows override Python workflows when names collide.

            Returns:
                Dict mapping workflow names to definitions
            """
            if self._workflows is None:
                # Start with Python workflows as base
                python_workflows = self._load_python_workflows()

                # Override with YAML workflows
                yaml_workflows = self._load_yaml_workflows()

                # Merge: YAML takes precedence
                self._workflows = {**python_workflows, **yaml_workflows}

                logger.debug(
                    f"Loaded {len(python_workflows)} Python + {len(yaml_workflows)} YAML workflows "
                    f"= {len(self._workflows)} total"
                )
            return self._workflows

        def get_workflows(self) -> Dict[str, WorkflowDefinition]:
            """Get workflow definitions for this vertical."""
            return self._load_workflows()

        def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
            """Get a specific workflow by name."""
            return self._load_workflows().get(name)

        def get_workflow_names(self) -> List[str]:
            """Get all available workflow names."""
            return list(self._load_workflows().keys())

        def get_auto_workflows(self) -> List[Tuple[str, str]]:
            """Get automatic workflow triggers based on query patterns."""
            return [
                # Single stock patterns
                (r"analyze\s+\w+", "single_stock_analysis"),
                (r"valuation\s+\w+", "single_stock_analysis"),
                (r"quick\s+analysis", "quick_stock_analysis"),
                # Batch patterns
                (r"batch\s+analysis", "batch_analysis"),
                (r"screen\s+stocks", "quick_screening"),
                (r"sector\s+analysis", "sector_batch_analysis"),
                # Backtest patterns
                (r"backtest", "rl_backtest"),
                (r"historical\s+test", "rl_backtest"),
                (r"training\s+data", "rl_training_data"),
            ]

        def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
            """Get appropriate workflow for task type."""
            mapping = {
                # Single stock
                "analysis": "single_stock_analysis",
                "valuation": "single_stock_analysis",
                "quick": "quick_stock_analysis",
                # Batch
                "batch": "batch_analysis",
                "screening": "quick_screening",
                "sector": "sector_batch_analysis",
                # Backtest
                "backtest": "rl_backtest",
                "training": "rl_training_data",
            }
            return mapping.get(task_type.lower())

        def create_executor(
            self,
            orchestrator: "AgentOrchestrator",
        ) -> "WorkflowExecutor":
            """Create a standard workflow executor."""
            from victor.workflows.executor import WorkflowExecutor

            return WorkflowExecutor(orchestrator)

        def create_streaming_executor(
            self,
            orchestrator: "AgentOrchestrator",
        ) -> "StreamingWorkflowExecutor":
            """Create a streaming workflow executor."""
            from victor.workflows.streaming_executor import StreamingWorkflowExecutor

            return StreamingWorkflowExecutor(orchestrator)

        async def astream(
            self,
            workflow_name: str,
            orchestrator: "AgentOrchestrator",
            context: Optional[Dict[str, Any]] = None,
        ) -> AsyncIterator["WorkflowStreamChunk"]:
            """Stream workflow execution with real-time events."""
            workflow = self.get_workflow(workflow_name)
            if not workflow:
                raise ValueError(f"Unknown workflow: {workflow_name}")

            executor = self.create_streaming_executor(orchestrator)
            async for chunk in executor.astream(workflow, context or {}):
                yield chunk

        def __repr__(self) -> str:
            return f"InvestmentWorkflowProvider(workflows={len(self._load_workflows())})"

except ImportError:
    # Victor framework not available - provide stub
    class InvestmentWorkflowProvider:  # type: ignore
        """Stub provider when victor framework is not available."""

        def __init__(self) -> None:
            logger.warning("Victor workflows not available, using stub provider")

        def get_workflows(self) -> Dict[str, Any]:
            return {}

        def get_workflow(self, name: str) -> None:
            return None

        def get_workflow_names(self) -> List[str]:
            return []


# Register investment domain handlers when this module is loaded
from investigator.domain.handlers import register_handlers as _register_handlers

_register_handlers()


__all__ = [
    "InvestmentWorkflowProvider",
]
