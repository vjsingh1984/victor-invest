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

"""Base classes for Victor Invest tools.

This module provides the foundational classes for all investment tools:
- ToolResult: Standardized result container for tool execution
- BaseTool: Abstract base class defining the tool interface

All tools inherit from BaseTool and return ToolResult instances.

ARCHITECTURE DECISION: Dual-Mode Tool Design
============================================

Tools in Victor-Invest are designed to work in TWO modes:

1. DIRECT INVOCATION (Context Stuffing Pattern)
   - Called by StateGraph workflow nodes
   - Deterministic data collection
   - Results included in LLM prompts
   - No LLM involvement in fetching

   Example:
       # In workflow node
       sec_tool = SECFilingTool()
       result = await sec_tool.execute(symbol="AAPL")
       state.sec_data = result.data  # Added to synthesis prompt

2. LLM TOOL CALLING (On-Demand Pattern)
   - Registered with Victor Agent
   - LLM decides when to invoke
   - Used for exploratory/adaptive analysis
   - LLM determines what data is needed

   Example:
       # Agent with tools
       agent = Agent(tools=[SECFilingTool, ValuationTool])
       # LLM reasoning: "I need peer comparison data..."
       # → Agent invokes valuation tool

DESIGN PRINCIPLE: Tools are mode-agnostic. The orchestration layer
(StateGraph vs Agent) determines the invocation pattern. This allows:
- Predictable batch processing (direct invocation)
- Flexible exploration (tool calling)
- Same tool implementation serves both patterns

See: docs/ARCHITECTURE_DECISION_DATA_ACCESS.md for full rationale.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardized result container for tool execution.

    All tools return a ToolResult to provide consistent handling of
    success/failure states and structured data.

    Attributes:
        success: Whether the tool execution succeeded
        data: The result data if successful (can be any dict structure)
        error: Error message if execution failed
        warnings: Optional list of non-fatal warnings encountered
        metadata: Optional metadata about the execution (timing, source, etc.)

    Example:
        # Successful result
        result = ToolResult(
            success=True,
            data={"fair_value": 150.25, "upside": 12.5},
            metadata={"model": "DCF", "execution_time_ms": 234}
        )

        # Failed result
        result = ToolResult(
            success=False,
            error="Symbol INVALID not found",
            warnings=["Attempted cache lookup"]
        )
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure warnings and metadata are always lists/dicts."""
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    @classmethod
    def success_result(
        cls,
        data: Dict[str, Any],
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult":
        """Factory method for successful results."""
        return cls(
            success=True,
            data=data,
            warnings=warnings or [],
            metadata=metadata or {}
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult":
        """Factory method for error results."""
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
            metadata=metadata or {}
        )


class BaseTool(ABC):
    """Abstract base class for all investment tools.

    Tools wrap existing investigator infrastructure to provide a clean,
    async interface for the Victor agent framework. Each tool:

    1. Has a unique name and description for tool discovery
    2. Provides an async execute() method
    3. Returns structured ToolResult instances
    4. Handles errors gracefully without raising exceptions

    Subclasses must implement:
        - name: Tool identifier (lowercase, underscores)
        - description: Human-readable description for tool selection
        - execute(**kwargs): Async method performing the tool's work

    Example:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something useful"

            async def execute(self, symbol: str, **kwargs) -> ToolResult:
                try:
                    # Do work
                    return ToolResult.success_result({"result": "data"})
                except Exception as e:
                    return ToolResult.error_result(str(e))
    """

    name: str = ""
    description: str = ""

    def __init__(self, config: Optional[Any] = None):
        """Initialize tool with optional configuration.

        Args:
            config: Optional configuration object (typically from investigator.config)
        """
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Async initialization hook for tools that need setup.

        Override this method if your tool needs async setup (e.g., establishing
        database connections, loading models). Called automatically before
        first execute() if not already initialized.
        """
        self._initialized = True

    async def ensure_initialized(self) -> None:
        """Ensure the tool is initialized before use."""
        if not self._initialized:
            await self.initialize()

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with provided parameters.

        Args:
            **kwargs: Tool-specific parameters (documented in subclasses)

        Returns:
            ToolResult with success/failure status and data/error

        Note:
            Implementations should never raise exceptions to callers.
            Instead, catch all exceptions and return ToolResult.error_result().
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for this tool's parameters.

        Override in subclasses to provide parameter validation schema.
        Used by agent frameworks for tool discovery and validation.

        Returns:
            JSON Schema dict describing tool parameters
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
