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

"""RL Backtest Workflow - StateGraph-based RL backtesting orchestration.

This module provides workflow orchestration for RL backtesting using
victor-core's StateGraph pattern. It aligns with the existing analysis
workflows while focusing on historical valuation and reward calculation.

Architecture:
    Uses Direct Tool Invocation (same as analysis workflows) for:
    - Historical data collection (prices, shares, metadata)
    - Multi-period reward calculation
    - Prediction recording to database

Data Flow:
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 1: Historical Data Collection (Direct Tool Invocation)│
    │   fetch_historical_data() → Get prices/shares at past dates │
    │   (Parallel per lookback period, deterministic, no LLM)     │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 2: Valuation Analysis (Direct Tool Invocation)        │
    │   run_historical_valuation() → ValuationTool at past dates  │
    │   (Computation, no LLM reasoning needed)                    │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 3: Reward Calculation & Recording                     │
    │   calculate_rewards() → Multi-period rewards (1m-36m)       │
    │   record_predictions() → Store in valuation_outcomes        │
    └─────────────────────────────────────────────────────────────┘

Example:
    from victor_invest.workflows.rl_backtest import run_rl_backtest

    # Run backtest for AAPL with quarterly lookbacks over 5 years
    result = await run_rl_backtest(
        symbol="AAPL",
        lookback_months_list=[12, 24, 36, 48, 60],
    )
    print(f"Processed {len(result.predictions)} lookback periods")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional

from victor.framework.graph import END, StateGraph

from victor_invest.tools import RLBacktestTool, ValuationTool

logger = logging.getLogger(__name__)

# Holding periods for multi-period rewards (aligned with RLBacktestTool)
HOLDING_PERIODS = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "12m": 365,
    "18m": 540,
    "24m": 730,
    "36m": 1095,
}


@dataclass
class RLBacktestWorkflowState:
    """State container for RL backtest workflow.

    Attributes:
        symbol: Stock ticker symbol being backtested.
        lookback_months_list: List of lookback periods in months.
        interval: Interval type (quarterly or monthly).
        predictions: List of prediction results per lookback period.
        errors: List of error messages encountered.
        completed_steps: Set of completed workflow steps.
        metadata: Additional metadata about the backtest run.
    """

    symbol: str
    lookback_months_list: List[int] = field(default_factory=list)
    interval: str = "quarterly"
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Intermediate state
    historical_data: Dict[str, Any] = field(default_factory=dict)
    valuation_results: Dict[str, Any] = field(default_factory=dict)
    reward_data: Dict[str, Any] = field(default_factory=dict)

    def mark_step_completed(self, step: str) -> None:
        """Mark a workflow step as completed."""
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_prediction(self, prediction: Dict[str, Any]) -> None:
        """Add a prediction result."""
        self.predictions.append(prediction)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "symbol": self.symbol,
            "lookback_months_list": self.lookback_months_list,
            "interval": self.interval,
            "predictions": self.predictions,
            "errors": self.errors,
            "completed_steps": self.completed_steps,
            "metadata": self.metadata,
            "historical_data": self.historical_data,
            "valuation_results": self.valuation_results,
            "reward_data": self.reward_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLBacktestWorkflowState":
        """Create state from dictionary."""
        return cls(
            symbol=data.get("symbol", ""),
            lookback_months_list=data.get("lookback_months_list", []),
            interval=data.get("interval", "quarterly"),
            predictions=data.get("predictions", []),
            errors=data.get("errors", []),
            completed_steps=data.get("completed_steps", []),
            metadata=data.get("metadata", {}),
            historical_data=data.get("historical_data", {}),
            valuation_results=data.get("valuation_results", {}),
            reward_data=data.get("reward_data", {}),
        )


def _ensure_state(state_input) -> RLBacktestWorkflowState:
    """Convert dict to RLBacktestWorkflowState if needed."""
    if isinstance(state_input, dict):
        return RLBacktestWorkflowState.from_dict(state_input)
    return state_input


def _state_to_dict(state: RLBacktestWorkflowState) -> dict:
    """Convert RLBacktestWorkflowState to dict."""
    return state.to_dict()


# Tool instances (lazy-initialized)
_rl_backtest_tool: Optional[RLBacktestTool] = None
_valuation_tool: Optional[ValuationTool] = None


async def _get_rl_backtest_tool() -> RLBacktestTool:
    """Get or create RL backtest tool instance."""
    global _rl_backtest_tool
    if _rl_backtest_tool is None:
        _rl_backtest_tool = RLBacktestTool()
        await _rl_backtest_tool.initialize()
    return _rl_backtest_tool


async def _get_valuation_tool() -> ValuationTool:
    """Get or create valuation tool instance."""
    global _valuation_tool
    if _valuation_tool is None:
        _valuation_tool = ValuationTool()
    return _valuation_tool


# =============================================================================
# Node Functions
# =============================================================================


async def fetch_historical_data(state_input) -> dict:
    """Fetch historical price and shares data for all lookback periods.

    Uses RLBacktestTool to retrieve historical prices and shares data
    at each lookback date for the symbol.

    Args:
        state_input: Current workflow state (dict or RLBacktestWorkflowState).

    Returns:
        Updated state dict with historical_data populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Fetching historical data for {state.symbol}")
        rl_tool = await _get_rl_backtest_tool()
        today = date.today()

        historical_data = {}
        for months_back in state.lookback_months_list:
            analysis_date = today - relativedelta(months=months_back)
            result = await rl_tool.execute(
                action="get_historical_data",
                symbol=state.symbol,
                analysis_date=analysis_date,
            )
            if result.success:
                historical_data[months_back] = {
                    "analysis_date": analysis_date.isoformat(),
                    "data": result.data,
                }
            else:
                historical_data[months_back] = {
                    "analysis_date": analysis_date.isoformat(),
                    "error": result.error,
                }

        state.historical_data = historical_data
        state.mark_step_completed("fetch_historical_data")

    except Exception as e:
        error_msg = f"Historical data fetch failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)

    return _state_to_dict(state)


async def run_historical_valuation(state_input) -> dict:
    """Run valuation analysis at historical dates.

    Executes valuation models using only data available at each
    historical analysis date.

    Args:
        state_input: Current workflow state (dict or RLBacktestWorkflowState).

    Returns:
        Updated state dict with valuation_results populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Running historical valuations for {state.symbol}")
        valuation_tool = await _get_valuation_tool()

        valuation_results = {}
        for months_back, hist_data in state.historical_data.items():
            if "error" in hist_data:
                valuation_results[months_back] = {"error": hist_data["error"]}
                continue

            data = hist_data.get("data", {})
            price = data.get("price")

            if price and price > 0:
                # Run valuation with historical price
                result = await valuation_tool.execute(
                    symbol=state.symbol,
                    model="all",
                    current_price=price,
                )
                if result.success:
                    valuation_results[months_back] = {
                        "analysis_date": hist_data["analysis_date"],
                        "price": price,
                        "valuation": result.data,
                    }
                else:
                    valuation_results[months_back] = {
                        "analysis_date": hist_data["analysis_date"],
                        "price": price,
                        "error": result.error,
                    }
            else:
                valuation_results[months_back] = {
                    "analysis_date": hist_data["analysis_date"],
                    "error": "No price data available",
                }

        state.valuation_results = valuation_results
        state.mark_step_completed("run_historical_valuation")

    except Exception as e:
        error_msg = f"Historical valuation failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)

    return _state_to_dict(state)


async def calculate_rewards(state_input) -> dict:
    """Calculate multi-period rewards for all historical predictions.

    Computes rewards for holding periods 1m, 3m, 6m, 12m, 18m, 24m, 36m
    using future prices relative to each historical analysis date.

    Args:
        state_input: Current workflow state (dict or RLBacktestWorkflowState).

    Returns:
        Updated state dict with reward_data populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Calculating multi-period rewards for {state.symbol}")
        rl_tool = await _get_rl_backtest_tool()

        reward_data = {}
        for months_back, val_result in state.valuation_results.items():
            if "error" in val_result:
                reward_data[months_back] = {"error": val_result["error"]}
                continue

            analysis_date = date.fromisoformat(val_result["analysis_date"])
            price = val_result.get("price")

            if price and price > 0:
                result = await rl_tool.execute(
                    action="calculate_rewards",
                    symbol=state.symbol,
                    analysis_date=analysis_date,
                    current_price=price,
                )
                if result.success:
                    reward_data[months_back] = {
                        "analysis_date": val_result["analysis_date"],
                        "price": price,
                        "multi_period": result.data.get("multi_period", {}),
                    }
                else:
                    reward_data[months_back] = {
                        "analysis_date": val_result["analysis_date"],
                        "price": price,
                        "error": result.error,
                    }
            else:
                reward_data[months_back] = {"error": "No price for reward calculation"}

        state.reward_data = reward_data
        state.mark_step_completed("calculate_rewards")

    except Exception as e:
        error_msg = f"Reward calculation failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)

    return _state_to_dict(state)


async def record_predictions(state_input) -> dict:
    """Record all predictions to the database.

    Stores predictions in valuation_outcomes table with JSONB multi-period
    data for both LONG and SHORT positions.

    Args:
        state_input: Current workflow state (dict or RLBacktestWorkflowState).

    Returns:
        Updated state dict with predictions populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Recording predictions for {state.symbol}")
        rl_tool = await _get_rl_backtest_tool()

        for months_back, reward_info in state.reward_data.items():
            if "error" in reward_info:
                state.add_prediction({
                    "lookback_months": months_back,
                    "status": "skipped",
                    "error": reward_info["error"],
                })
                continue

            val_result = state.valuation_results.get(months_back, {})
            valuation = val_result.get("valuation", {})

            # Extract fair values and weights from valuation
            fair_values = {}
            weights = {}
            if isinstance(valuation, dict):
                fair_values = valuation.get("fair_values", {})
                weights = valuation.get("weights", {})
                blended_fair_value = valuation.get("blended_fair_value", 0)
                tier = valuation.get("tier_classification", "")
            else:
                blended_fair_value = 0
                tier = ""

            analysis_date = date.fromisoformat(reward_info["analysis_date"])
            price = reward_info.get("price", 0)

            result = await rl_tool.execute(
                action="record_prediction",
                symbol=state.symbol,
                analysis_date=analysis_date,
                current_price=price,
                fair_value=blended_fair_value,
                fair_values=fair_values,
                weights=weights,
                tier_classification=tier,
                context_features={
                    "lookback_months": months_back,
                    "interval": state.interval,
                },
            )

            if result.success:
                state.add_prediction({
                    "lookback_months": months_back,
                    "analysis_date": reward_info["analysis_date"],
                    "price": price,
                    "fair_value": blended_fair_value,
                    "record_ids": result.data.get("record_ids", []),
                    "status": "recorded",
                })
            else:
                state.add_prediction({
                    "lookback_months": months_back,
                    "analysis_date": reward_info["analysis_date"],
                    "status": "failed",
                    "error": result.error,
                })

        state.mark_step_completed("record_predictions")

    except Exception as e:
        error_msg = f"Prediction recording failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)

    return _state_to_dict(state)


async def finalize_backtest(state_input) -> dict:
    """Finalize backtest and compile summary.

    Args:
        state_input: Current workflow state (dict or RLBacktestWorkflowState).

    Returns:
        Updated state dict with metadata summary.
    """
    state = _ensure_state(state_input)

    successful = sum(1 for p in state.predictions if p.get("status") == "recorded")
    failed = sum(1 for p in state.predictions if p.get("status") == "failed")
    skipped = sum(1 for p in state.predictions if p.get("status") == "skipped")

    state.metadata["summary"] = {
        "symbol": state.symbol,
        "total_lookback_periods": len(state.lookback_months_list),
        "successful_predictions": successful,
        "failed_predictions": failed,
        "skipped_predictions": skipped,
        "errors": state.errors,
        "completed_steps": state.completed_steps,
    }

    state.mark_step_completed("finalize_backtest")
    logger.info(
        f"Backtest complete for {state.symbol}: "
        f"{successful} recorded, {failed} failed, {skipped} skipped"
    )

    return _state_to_dict(state)


# =============================================================================
# Graph Builders
# =============================================================================


def build_rl_backtest_graph() -> StateGraph:
    """Build RL backtest workflow graph.

    Workflow:
        START -> fetch_historical_data -> run_historical_valuation
        -> calculate_rewards -> record_predictions -> finalize -> END

    Returns:
        StateGraph configured for RL backtesting.
    """
    graph = StateGraph()

    # Add nodes
    graph.add_node("fetch_historical_data", fetch_historical_data)
    graph.add_node("run_historical_valuation", run_historical_valuation)
    graph.add_node("calculate_rewards", calculate_rewards)
    graph.add_node("record_predictions", record_predictions)
    graph.add_node("finalize_backtest", finalize_backtest)

    # Add edges: linear flow for backtesting
    graph.add_edge("fetch_historical_data", "run_historical_valuation")
    graph.add_edge("run_historical_valuation", "calculate_rewards")
    graph.add_edge("calculate_rewards", "record_predictions")
    graph.add_edge("record_predictions", "finalize_backtest")
    graph.add_edge("finalize_backtest", END)

    # Set entry point
    graph.set_entry_point("fetch_historical_data")

    logger.debug("Built RL backtest graph")
    return graph


# =============================================================================
# Execution Helpers
# =============================================================================


def generate_lookback_list(
    max_months: int,
    interval: str = "quarterly",
) -> List[int]:
    """Generate list of lookback periods.

    Args:
        max_months: Maximum lookback in months (e.g., 120 for 10 years).
        interval: "quarterly" (every 3 months) or "monthly".

    Returns:
        List of lookback months from 3 to max_months.
    """
    step = 3 if interval == "quarterly" else 1
    return list(range(3, max_months + 1, step))


async def run_rl_backtest(
    symbol: str,
    lookback_months_list: Optional[List[int]] = None,
    max_lookback_months: int = 120,
    interval: str = "quarterly",
    use_yaml_workflow: bool = True,
) -> RLBacktestWorkflowState:
    """Convenience function to run a complete RL backtest workflow.

    Uses InvestmentWorkflowProvider (BaseYAMLWorkflowProvider pattern) for
    YAML-based workflow execution with shared handlers, or falls back to
    Python StateGraph for backwards compatibility.

    Args:
        symbol: Stock ticker symbol to backtest.
        lookback_months_list: Explicit list of lookback periods.
        max_lookback_months: Max lookback (used if lookback_months_list not provided).
        interval: Interval type ("quarterly" or "monthly").
        use_yaml_workflow: If True, use YAML workflow via InvestmentWorkflowProvider.

    Returns:
        Final RLBacktestWorkflowState with results.

    Example:
        # Run 10-year quarterly backtest via YAML workflow
        result = await run_rl_backtest("AAPL", max_lookback_months=120)

        # Run with explicit lookback periods
        result = await run_rl_backtest("AAPL", lookback_months_list=[12, 24, 36])

        # Use Python StateGraph fallback
        result = await run_rl_backtest("AAPL", use_yaml_workflow=False)
    """
    # Generate lookback list if not provided
    if lookback_months_list is None:
        lookback_months_list = generate_lookback_list(max_lookback_months, interval)

    if use_yaml_workflow:
        # Use InvestmentWorkflowProvider (BaseYAMLWorkflowProvider pattern)
        try:
            from victor.workflows.executor import WorkflowExecutor, WorkflowContext

            # Import here to avoid circular imports
            from victor_invest.workflows import InvestmentWorkflowProvider

            provider = InvestmentWorkflowProvider()
            workflow = provider.get_workflow("rl_backtest")

            if workflow:
                # Create execution context
                context = WorkflowContext({
                    "symbol": symbol,
                    "max_lookback_months": max_lookback_months,
                    "interval": interval,
                    "lookback_dates": lookback_months_list,
                })

                # Execute via YAML workflow with shared handlers
                executor = WorkflowExecutor(orchestrator=None)
                workflow_result = await executor.execute(workflow, context)

                # Convert to RLBacktestWorkflowState
                return _convert_yaml_result_to_state(symbol, lookback_months_list, interval, workflow_result)

        except Exception as e:
            logger.warning(f"YAML workflow execution failed, falling back to Python: {e}")

    # Fallback: Python StateGraph execution
    state = RLBacktestWorkflowState(
        symbol=symbol,
        lookback_months_list=lookback_months_list,
        interval=interval,
    )

    # Build and compile graph
    graph = build_rl_backtest_graph()
    compiled = graph.compile()

    # Execute workflow
    result = await compiled.invoke(state.to_dict())

    # Convert result back to state
    if hasattr(result, 'state'):
        result_data = result.state
    elif isinstance(result, dict):
        result_data = result
    else:
        result_data = result

    if isinstance(result_data, dict):
        return RLBacktestWorkflowState.from_dict(result_data)
    elif isinstance(result_data, RLBacktestWorkflowState):
        return result_data
    else:
        return state


def _convert_yaml_result_to_state(
    symbol: str,
    lookback_months_list: List[int],
    interval: str,
    workflow_result: Any,
) -> RLBacktestWorkflowState:
    """Convert YAML workflow result to RLBacktestWorkflowState."""
    state = RLBacktestWorkflowState(
        symbol=symbol,
        lookback_months_list=lookback_months_list,
        interval=interval,
    )

    # Extract from workflow result context
    if hasattr(workflow_result, 'context'):
        ctx = workflow_result.context
        if hasattr(ctx, 'get'):
            state.predictions = ctx.get("predictions", [])
            state.metadata = ctx.get("metadata", {})
            if ctx.get("backtest_results"):
                backtest = ctx.get("backtest_results", {})
                state.predictions = backtest.get("predictions", state.predictions)
                state.metadata = backtest.get("metadata", state.metadata)
    elif isinstance(workflow_result, dict):
        state.predictions = workflow_result.get("predictions", [])
        state.metadata = workflow_result.get("metadata", {})

    state.mark_step_completed("yaml_workflow_complete")
    return state


async def run_rl_backtest_batch(
    symbols: List[str],
    max_lookback_months: int = 120,
    interval: str = "quarterly",
    parallel_limit: int = 5,
) -> List[RLBacktestWorkflowState]:
    """Run RL backtest for multiple symbols with parallelism control.

    Args:
        symbols: List of stock symbols to backtest.
        max_lookback_months: Max lookback in months.
        interval: Interval type.
        parallel_limit: Max concurrent backtests.

    Returns:
        List of RLBacktestWorkflowState results.
    """
    semaphore = asyncio.Semaphore(parallel_limit)

    async def limited_backtest(symbol: str) -> RLBacktestWorkflowState:
        async with semaphore:
            return await run_rl_backtest(
                symbol=symbol,
                max_lookback_months=max_lookback_months,
                interval=interval,
            )

    tasks = [limited_backtest(s) for s in symbols]
    return await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    # State
    "RLBacktestWorkflowState",
    # Graph builders
    "build_rl_backtest_graph",
    # Node functions
    "fetch_historical_data",
    "run_historical_valuation",
    "calculate_rewards",
    "record_predictions",
    "finalize_backtest",
    # Execution helpers
    "run_rl_backtest",
    "run_rl_backtest_batch",
    "generate_lookback_list",
    # Constants
    "HOLDING_PERIODS",
]
