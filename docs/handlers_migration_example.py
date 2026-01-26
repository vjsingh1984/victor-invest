"""
Example: Migrated FetchSECDataHandler using @handler_decorator + BaseHandler

This file demonstrates the migration from the old pattern to the new Victor framework pattern.
"""

# =============================================================================
# BEFORE: Old Pattern (victor_invest/handlers.py, lines 57-112)
# =============================================================================

"""
from dataclasses import dataclass
import time
from victor.workflows.executor import NodeResult, NodeStatus

@dataclass
class FetchSECDataHandler:
    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        symbol = context.get("symbol", "")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No symbol provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            from victor_invest.tools.sec_filing import SECFilingTool

            sec_tool = SECFilingTool()
            result = await sec_tool.execute(
                {},
                symbol=symbol,
                action="get_company_facts",
            )

            output = {
                "status": "success" if result.success else "error",
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }

            output_key = node.output_key or "sec_data"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"SEC data fetch error for {symbol}: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

# Stats: 56 lines, ~87% boilerplate
"""

# =============================================================================
# AFTER: New Pattern (Recommended)
# =============================================================================

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

from victor.framework.handler_registry import handler_decorator
from victor.framework.workflows.base_handler import BaseHandler

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import WorkflowContext


@handler_decorator("fetch_sec_data", vertical="investment", description="Fetch SEC filing data for analysis")
@dataclass
class FetchSECDataHandler(BaseHandler):
    """Fetch SEC filing data for analysis.

    This handler retrieves SEC filing data including financial statements,
    company facts, and other regulatory filings for investment analysis.

    Example YAML usage:
        - id: fetch_sec
          type: compute
          handler: fetch_sec_data
          output: sec_data
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute SEC data fetch.

        Args:
            node: Compute node with handler configuration
            context: Workflow context containing symbol
            tool_registry: Tool registry (not used for this handler)

        Returns:
            Tuple of (output_dict, tool_calls_count)

        Raises:
            ValueError: If symbol not provided in context
        """
        # Extract inputs from context
        symbol = context.get("symbol", "")

        # Validation
        if not symbol:
            return {
                "status": "error",
                "error": "No symbol provided",
                "data": None,
            }, 0

        # Invoke tool (direct tool invocation pattern)
        from victor_invest.tools.sec_filing import SECFilingTool

        sec_tool = SECFilingTool()
        result = await sec_tool.execute(
            {},  # _exec_ctx (not used by investment tools)
            symbol=symbol,
            action="get_company_facts",
        )

        # Return (output, tool_calls_count)
        # BaseHandler handles timing, error handling, NodeResult construction, context.set()
        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0  # No LLM tool calls made


# Stats: 83 lines (with docstrings), ~50% business logic, ~0% boilerplate
# Reduction: 56 lines → 36 lines of actual code (-36%)


# =============================================================================
# Comparison Summary
# =============================================================================

"""
PATTERN COMPARISON:

| Aspect                  | Old Pattern                     | New Pattern                           |
|-------------------------|---------------------------------|---------------------------------------|
| Base Class              | Plain dataclass                 | BaseHandler                           |
| Decorator               | None                            | @handler_decorator                    |
| Method signature        | __call__ -> NodeResult          | execute() -> Tuple[Any, int]         |
| Timing                  | Manual (time.time())            | Automatic (BaseHandler)               |
| Error handling          | Manual try/except               | Automatic (BaseHandler)               |
| NodeResult construction | Manual                           | Automatic (BaseHandler)               |
| Context storage         | Manual (context.set())          | Automatic (BaseHandler)               |
| Registration            | Manual (register_handlers())    | Auto (@handler_decorator)             |
| Boilerplate percentage  | ~87%                            | ~0%                                   |
| Lines of code           | 56 lines                        | 36 lines (-36%)                       |

MIGRATION STEPS:

1. Add imports:
   - from victor.framework.handler_registry import handler_decorator
   - from victor.framework.workflows.base_handler import BaseHandler

2. Add decorator to class:
   - @handler_decorator("handler_name", vertical="investment", description="...")

3. Change base class:
   - Old: @dataclass class MyHandler:
   - New: @handler_decorator(...) @dataclass class MyHandler(BaseHandler):

4. Rename method:
   - Old: async def __call__(self, ...)
   - New: async def execute(self, ...)

5. Change return type:
   - Old: -> NodeResult
   - New: -> Tuple[Any, int]

6. Update return statement:
   - Old: return NodeResult(node_id=..., status=..., output=..., duration_seconds=...)
   - New: return output_dict, tool_calls_count

7. Remove boilerplate:
   - Remove start_time = time.time()
   - Remove try/except blocks
   - Remove NodeResult construction
   - Remove context.set() calls
   - Remove manual duration calculations

8. Update error handling:
   - Old: return NodeResult(..., status=NodeStatus.FAILED, error=...)
   - New: return {"status": "error", "error": "..."}, 0
   - OR: raise ValueError("...")

9. Remove manual registration:
   - Delete from HANDLERS dict
   - register_handlers() becomes no-op
"""

# =============================================================================
# Complex Handler Example: RunSynthesisHandler (with LLM integration)
# =============================================================================

"""
For complex handlers like RunSynthesisHandler that have:
- LLM client initialization
- Config access
- Cleanup in finally block
- Multiple helper methods

The migration pattern is the same, but you need to handle special cases:

1. Config access: Pass via context or use dependency injection
2. Cleanup: Implement cleanup() method if BaseHandler supports it
3. Helper methods: Keep as class methods, no changes needed
4. LLM calls: No change, just return (output, tool_calls_count)

Example structure:

@handler_decorator("run_synthesis", vertical="investment", description="Run synthesis analysis")
@dataclass
class RunSynthesisHandler(BaseHandler):
    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        # Get inputs
        symbol = context.get("symbol", "")
        fundamental = context.get("fundamental_analysis", {})
        technical = context.get("technical_analysis", {})

        # Main logic (unchanged from old implementation)
        llm_result = await self._llm_synthesis(symbol, technical, fundamental)
        if llm_result:
            output = self._build_output(llm_result)
        else:
            output = self._rule_based_synthesis(fundamental, technical)

        # Return (output, tool_calls_count)
        return output, 1  # 1 LLM tool call

    def _llm_synthesis(self, symbol, technical, fundamental):
        # Helper method (unchanged)
        pass

    def _rule_based_synthesis(self, fundamental, technical):
        # Helper method (unchanged)
        pass

    def _build_output(self, llm_result):
        # Helper method (unchanged)
        pass
"""

# =============================================================================
# Testing the Migrated Handler
# =============================================================================

"""
# Test 1: Direct handler invocation
import asyncio
from victor_invest.handlers_v2 import FetchSECDataHandler
from victor.workflows.executor import WorkflowContext
from victor.workflows.definition import ComputeNode

async def test_handler():
    handler = FetchSECDataHandler()
    context = WorkflowContext({"symbol": "AAPL"})
    node = ComputeNode(id="test", handler="fetch_sec_data", output_key="sec_data")

    output, tool_calls = await handler.execute(node, context, None)
    assert output["status"] in ["success", "error"]
    assert tool_calls == 0

asyncio.run(test_handler())

# Test 2: Via workflow
from victor_invest.workflows import InvestmentWorkflowProvider

async def test_workflow():
    provider = InvestmentWorkflowProvider()
    result = await provider.run_workflow_with_handlers(
        "quick",
        context={"symbol": "AAPL"},
    )
    assert result.success

asyncio.run(test_workflow())

# Test 3: Verify auto-registration
from victor.framework.handler_registry import HandlerRegistry

registry = HandlerRegistry()
investment_handlers = registry.get_handlers("investment")
assert "fetch_sec_data" in investment_handlers
"""

# =============================================================================
# Migration Checklist
# =============================================================================

"""
For each handler, verify:

[ ] Imports updated (handler_decorator, BaseHandler)
[ ] @handler_decorator added with correct parameters
[ ] Base class changed to BaseHandler
[ ] Method renamed from __call__ to execute
[ ] Return type changed to Tuple[Any, int]
[ ] All timing code removed (start_time, duration_seconds)
[ ] All NodeResult constructions removed
[ ] All try/except blocks removed (or kept for specific error handling)
[ ] context.set() calls removed
[ ] Return statement updated to (output, tool_calls_count)
[ ] Error handling updated (return error dict or raise exception)
[ ] Handler tested (unit test or manual)
[ ] Handler auto-registers correctly
[ ] YAML workflow still works with handler
"""

__all__ = ["FetchSECDataHandler"]
