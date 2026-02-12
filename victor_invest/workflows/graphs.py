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

"""StateGraph-based workflow builders for investment analysis.

This module provides workflow graph builders using victor-core's StateGraph
pattern. Three workflow types are supported:

1. Quick Graph: Technical analysis only (fastest)
2. Standard Graph: Technical + Fundamental (balanced)
3. Comprehensive Graph: All agents (institutional-grade)

ARCHITECTURE DECISION: Direct Tool Invocation (Context Stuffing Pattern)
========================================================================

This module uses DIRECT TOOL INVOCATION rather than LLM tool calling because:

1. DETERMINISTIC DATA COLLECTION
   - Financial data must be exact (revenue, earnings, etc.)
   - Same query should always return same results
   - No LLM hallucination risk for data fetching

2. LATENCY OPTIMIZATION
   - Direct DB: ~50-200ms per query
   - LLM tool calling: +2-10s per inference round
   - Parallel data fetching requires predictable timing

3. COST EFFICIENCY
   - Direct DB: $0 (local database only)
   - LLM tool calling: Token usage per tool decision
   - Batch processing would be expensive with LLM

4. BOUNDED SCOPE
   - Analysis workflow has fixed data requirements
   - No exploration/discovery needed for data collection
   - LLM reasoning happens in synthesis, not data fetching

WHEN TOOL CALLING WOULD BE APPROPRIATE:
- Exploratory analysis ("why is AAPL underperforming?")
- Peer discovery ("find similar companies")
- Interactive Q&A with unknown follow-ups
- Error recovery with alternative data sources

DATA FLOW:
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 1: Data Collection (Direct Tool Invocation)          │
    │   fetch_sec_data() → SECFilingTool.execute()               │
    │   fetch_market_data() → MarketDataTool.execute()           │
    │   (Parallel, deterministic, no LLM)                        │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 2: Analysis (Direct Tool Invocation)                 │
    │   run_fundamental_analysis() → ValuationTool.execute()     │
    │   run_technical_analysis() → TechnicalTool.execute()       │
    │   (Computation, no LLM reasoning needed)                   │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 3: Synthesis (Context Stuffing → LLM)                │
    │   All data included in prompt → Single LLM inference       │
    │   (Bounded scope, structured output, no tool calling)      │
    └─────────────────────────────────────────────────────────────┘

See: docs/ARCHITECTURE_DECISION_DATA_ACCESS.md for full rationale.

Example:
    from victor_invest.workflows import build_graph_for_mode, AnalysisMode

    # Build a standard analysis graph
    graph = build_graph_for_mode(AnalysisMode.STANDARD)
    compiled = graph.compile()

    # Execute
    result = await compiled.invoke(initial_state)
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional
from weakref import WeakKeyDictionary

from victor.framework.graph import END, StateGraph

from victor_invest.agents import (
    FUNDAMENTAL_AGENT_SPEC,
    MARKET_AGENT_SPEC,
    SEC_AGENT_SPEC,
    SYNTHESIS_AGENT_SPEC,
    TECHNICAL_AGENT_SPEC,
)
from victor_invest.tools import (
    MarketDataTool,
    SECFilingTool,
    TechnicalIndicatorsTool,
    ValuationTool,
)
from victor_invest.workflows.state import AnalysisMode, AnalysisWorkflowState

logger = logging.getLogger(__name__)


def _ensure_state(state_input) -> AnalysisWorkflowState:
    """Convert dict to AnalysisWorkflowState if needed."""
    if isinstance(state_input, dict):
        return AnalysisWorkflowState.from_dict(state_input)
    return state_input


def _state_to_dict(state: AnalysisWorkflowState) -> dict:
    """Convert AnalysisWorkflowState to dict."""
    return state.to_dict()


# Task-scoped tool instances (lazy-initialized)
# Avoids cross-run shared mutable state while preserving reuse within a run.
_task_tool_cache: WeakKeyDictionary = WeakKeyDictionary()


def _get_task_tool_cache() -> dict[str, Any]:
    """Get or create tool cache for the current asyncio task."""
    task = asyncio.current_task()
    if task is None:
        # Fallback path should be rare; caller will create non-cached instances.
        return {}

    cache = _task_tool_cache.get(task)
    if cache is None:
        cache = {}
        _task_tool_cache[task] = cache
    return cache


async def _get_sec_tool() -> SECFilingTool:
    """Get or create SEC filing tool instance."""
    cache = _get_task_tool_cache()
    tool = cache.get("sec_tool")
    if tool is None:
        tool = SECFilingTool()
        if asyncio.current_task() is not None:
            cache["sec_tool"] = tool
    return tool


async def _get_valuation_tool() -> ValuationTool:
    """Get or create valuation tool instance."""
    cache = _get_task_tool_cache()
    tool = cache.get("valuation_tool")
    if tool is None:
        tool = ValuationTool()
        if asyncio.current_task() is not None:
            cache["valuation_tool"] = tool
    return tool


async def _get_technical_tool() -> TechnicalIndicatorsTool:
    """Get or create technical indicators tool instance."""
    cache = _get_task_tool_cache()
    tool = cache.get("technical_tool")
    if tool is None:
        tool = TechnicalIndicatorsTool()
        if asyncio.current_task() is not None:
            cache["technical_tool"] = tool
    return tool


async def _get_market_tool() -> MarketDataTool:
    """Get or create market data tool instance."""
    cache = _get_task_tool_cache()
    tool = cache.get("market_tool")
    if tool is None:
        tool = MarketDataTool()
        if asyncio.current_task() is not None:
            cache["market_tool"] = tool
    return tool


# =============================================================================
# Node Functions
# =============================================================================


async def fetch_sec_data(state_input) -> dict:
    """Fetch SEC filings and fundamental data.

    Uses SEC_AGENT_SPEC capabilities to determine data extraction scope.
    Executes SECFilingTool to retrieve company facts and financial metrics.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with sec_data populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Fetching SEC data for {state.symbol}")
        sec_tool = await _get_sec_tool()

        # Get company facts (structured financial data from SEC CompanyFacts API)
        facts_result = await sec_tool.execute(symbol=state.symbol, action="get_company_facts")

        # Extract financial metrics
        metrics_result = await sec_tool.execute(symbol=state.symbol, action="extract_metrics")

        if facts_result.success or metrics_result.success:
            state.sec_data = {
                "symbol": state.symbol,
                "company_facts": facts_result.output if facts_result.success else None,
                "financial_metrics": metrics_result.output if metrics_result.success else None,
                "agent_spec": SEC_AGENT_SPEC.name,
                "status": "success",
            }
        else:
            state.sec_data = {
                "symbol": state.symbol,
                "status": "partial_failure",
                "facts_error": facts_result.error if not facts_result.success else None,
                "metrics_error": metrics_result.error if not metrics_result.success else None,
            }

        state.mark_step_completed("fetch_sec_data")
    except Exception as e:
        error_msg = f"SEC data fetch failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)
        state.sec_data = {"symbol": state.symbol, "status": "error", "error": str(e)}
    return _state_to_dict(state)


async def fetch_market_data(state_input) -> dict:
    """Fetch market data including price and volume.

    Uses MarketDataTool to retrieve current quotes, historical data,
    and company information for technical analysis.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with market_data populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Fetching market data for {state.symbol}")
        market_tool = await _get_market_tool()

        # Get current quote
        quote_result = await market_tool.execute(symbol=state.symbol, action="get_quote")

        # Get historical data (1 year for technical analysis)
        history_result = await market_tool.execute(symbol=state.symbol, action="get_history", days=365)

        # Get company info
        info_result = await market_tool.execute(symbol=state.symbol, action="get_info")

        if quote_result.success or history_result.success:
            state.market_data = {
                "symbol": state.symbol,
                "quote": quote_result.output if quote_result.success else None,
                "history": history_result.output if history_result.success else None,
                "company_info": info_result.output if info_result.success else None,
                "status": "success",
            }
        else:
            state.market_data = {
                "symbol": state.symbol,
                "status": "partial_failure",
                "quote_error": quote_result.error if not quote_result.success else None,
                "history_error": history_result.error if not history_result.success else None,
            }

        state.mark_step_completed("fetch_market_data")
    except Exception as e:
        error_msg = f"Market data fetch failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)
        state.market_data = {"symbol": state.symbol, "status": "error", "error": str(e)}
    return _state_to_dict(state)


async def run_fundamental_analysis(state_input) -> dict:
    """Run fundamental analysis on SEC data.

    Uses FUNDAMENTAL_AGENT_SPEC configuration and ValuationTool to perform
    multi-model valuation analysis (DCF, P/E, P/S, P/B, EV/EBITDA, GGM).

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with fundamental_analysis populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Running fundamental analysis for {state.symbol}")
        if state.sec_data is None:
            raise ValueError("SEC data not available for fundamental analysis")

        valuation_tool = await _get_valuation_tool()
        market_tool = await _get_market_tool()

        # Get current price for valuation models
        quote_result = await market_tool.execute(symbol=state.symbol, action="get_quote")
        current_price = None
        if quote_result.success and quote_result.output:
            current_price = quote_result.output.get("current_price")

        # Extract quarterly metrics from SEC data for valuation models
        quarterly_metrics = []
        if state.sec_data:
            financial_metrics = state.sec_data.get("financial_metrics", {})
            if financial_metrics:
                # Convert SEC metrics to quarterly format for valuation
                quarterly_metrics = [financial_metrics]  # Single period for now

        # Run all valuation models with pre-fetched SEC data
        valuation_result = await valuation_tool.execute(
            symbol=state.symbol,
            model="all",
            current_price=current_price,
            quarterly_metrics=quarterly_metrics,
        )

        # Get archetype detection (identifies company type for model weighting)
        archetype_result = await valuation_tool.execute(symbol=state.symbol, model="detect_archetype")

        if valuation_result.success:
            state.fundamental_analysis = {
                "symbol": state.symbol,
                "valuation_models": valuation_result.output,
                "archetype": archetype_result.output if archetype_result.success else None,
                "current_price": current_price,
                "agent_spec": FUNDAMENTAL_AGENT_SPEC.name,
                "model_weights": SYNTHESIS_AGENT_SPEC.metadata.get("weight_distribution", {}),
                "status": "success",
            }
        else:
            state.fundamental_analysis = {
                "symbol": state.symbol,
                "status": "partial_failure",
                "error": valuation_result.error,
            }

        state.mark_step_completed("fundamental_analysis")
    except Exception as e:
        error_msg = f"Fundamental analysis failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)
        state.fundamental_analysis = {"symbol": state.symbol, "status": "error", "error": str(e)}
    return _state_to_dict(state)


async def run_technical_analysis(state_input) -> dict:
    """Run technical analysis on market data.

    Uses TECHNICAL_AGENT_SPEC configuration and TechnicalIndicatorsTool
    to compute 80+ technical indicators including trend, momentum, volatility,
    and volume indicators.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with technical_analysis populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Running technical analysis for {state.symbol}")
        if state.market_data is None:
            raise ValueError("Market data not available for technical analysis")

        technical_tool = await _get_technical_tool()

        # Run comprehensive technical analysis
        analysis_result = await technical_tool.execute(symbol=state.symbol, action="calculate_all")

        # Get technical summary with signals
        trend_result = await technical_tool.execute(symbol=state.symbol, action="get_summary")

        # Get support/resistance levels
        levels_result = await technical_tool.execute(symbol=state.symbol, action="get_support_resistance")

        if analysis_result.success:
            state.technical_analysis = {
                "symbol": state.symbol,
                "indicators": analysis_result.output,
                "trend": trend_result.output if trend_result.success else None,
                "support_resistance": levels_result.output if levels_result.success else None,
                "agent_spec": TECHNICAL_AGENT_SPEC.name,
                "status": "success",
            }
        else:
            state.technical_analysis = {
                "symbol": state.symbol,
                "status": "partial_failure",
                "error": analysis_result.error,
            }

        state.mark_step_completed("technical_analysis")
    except Exception as e:
        error_msg = f"Technical analysis failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)
        state.technical_analysis = {"symbol": state.symbol, "status": "error", "error": str(e)}
    return _state_to_dict(state)


async def run_market_context_analysis(state_input) -> dict:
    """Analyze market regime and sector context.

    Uses MARKET_AGENT_SPEC configuration and MarketDataTool to analyze
    sector performance, market regime, and peer comparisons.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with market_context populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Running market context analysis for {state.symbol}")
        if state.market_data is None:
            raise ValueError("Market data not available for context analysis")

        market_tool = await _get_market_tool()

        # Get company info for sector/industry context
        info = state.market_data.get("company_info", {})
        sector = info.get("sector") if info else None

        # Get price change for multiple periods
        periods = ["1d", "5d", "1m", "3m", "1y"]
        price_changes = {}

        for period in periods:
            result = await market_tool.execute(symbol=state.symbol, action="get_price_change", period=period)
            if result.success:
                price_changes[period] = result.output

        # Calculate relative performance vs market (SPY)
        market_result = await market_tool.execute(symbol="SPY", action="get_price_change", period="1y")

        relative_performance = None
        if market_result.success and "1y" in price_changes:
            stock_return = price_changes["1y"].get("percent_change", 0)
            market_return = market_result.output.get("percent_change", 0)
            relative_performance = stock_return - market_return

        state.market_context = {
            "symbol": state.symbol,
            "sector": sector,
            "industry": info.get("industry") if info else None,
            "price_changes": price_changes,
            "relative_performance_vs_spy": relative_performance,
            "market_benchmark": market_result.output if market_result.success else None,
            "beta": info.get("beta") if info else None,
            "agent_spec": MARKET_AGENT_SPEC.name,
            "status": "success",
        }

        state.mark_step_completed("market_context_analysis")
    except Exception as e:
        error_msg = f"Market context analysis failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)
        state.market_context = {"symbol": state.symbol, "status": "error", "error": str(e)}
    return _state_to_dict(state)


async def _run_llm_synthesis(
    symbol: str,
    technical: dict,
    fundamental: dict,
    market_context: dict,
    composite_score: float,
    rule_based_recommendation: str,
) -> Optional[dict]:
    """Use LLM to generate intelligent investment synthesis.

    Args:
        symbol: Stock ticker symbol
        technical: Technical analysis data
        fundamental: Fundamental analysis data
        market_context: Market context data
        composite_score: Rule-based composite score
        rule_based_recommendation: Rule-based recommendation

    Returns:
        LLM synthesis dict with executive_summary, catalysts, risks, etc.
        Returns None if LLM fails.
    """
    import json

    try:
        from investigator.config import get_config
        from investigator.infrastructure.llm import OllamaClient

        config = get_config()
        client = OllamaClient(config)

        # Extract key technical data
        trend = technical.get("trend", {})
        current_price = trend.get("current_price", "N/A")
        overall_signal = trend.get("overall_signal", "neutral")
        signal_pcts = trend.get("signal_percentages", {})

        sr = technical.get("support_resistance", {})
        week_52 = sr.get("52_week", {})
        support = sr.get("support_levels", {}).get("support_1", "N/A")
        resistance = sr.get("resistance_levels", {}).get("resistance_1", "N/A")

        # Extract market context
        sector = market_context.get("sector", "Unknown")
        rel_perf = market_context.get("relative_performance_vs_spy", 0)
        beta = market_context.get("beta", 1.0)

        # Format fundamental data
        fund_summary = "Fundamental data not available."
        if fundamental and fundamental.get("status") == "success":
            valuation = fundamental.get("valuation_models", {})
            if valuation.get("composite_fair_value"):
                fund_summary = f"Composite Fair Value: ${valuation['composite_fair_value']:.2f}"
            if valuation.get("composite_upside_percent"):
                fund_summary += f", Upside: {valuation['composite_upside_percent']:.1f}%"

        # Build prompt
        prompt = f"""You are an expert investment analyst. Synthesize the following analysis data for {symbol} into a coherent investment recommendation.

## Technical Analysis
- Current Price: ${current_price}
- Overall Signal: {overall_signal}
- Bullish Signals: {signal_pcts.get('bullish_pct', 0):.0f}%
- Bearish Signals: {signal_pcts.get('bearish_pct', 0):.0f}%
- Support Level: ${support}
- Resistance Level: ${resistance}
- 52-Week Range: ${week_52.get('low', 'N/A')} - ${week_52.get('high', 'N/A')}

## Market Context
- Sector: {sector}
- Relative Performance vs SPY: {rel_perf:.1f}%
- Beta: {beta:.2f}

## Fundamental Analysis
{fund_summary}

## Rule-Based Analysis
- Composite Score: {composite_score:.1f}/100
- Initial Recommendation: {rule_based_recommendation}

Based on this analysis, provide a JSON response with:
{{
    "executive_summary": "2-3 sentence investment thesis explaining the recommendation",
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "HIGH/MEDIUM/LOW",
    "key_catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],
    "key_risks": ["risk 1", "risk 2", "risk 3"],
    "reasoning": "Brief explanation of the recommendation"
}}

Respond ONLY with the JSON object, no other text."""

        model = config.ollama.models.get("synthesis", "gpt-oss:20b")

        response = await client.generate(
            prompt=prompt,
            model=model,
            options={"temperature": 0.3, "num_predict": 1024},
        )

        response_text = response.get("response", "")

        # Parse JSON response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            return json.loads(json_str)

        return None

    except Exception as e:
        logger.warning(f"LLM synthesis error: {e}")
        return None


async def run_synthesis(state_input) -> dict:
    """Synthesize all analysis results into final output.

    Uses SYNTHESIS_AGENT_SPEC configuration with weight distribution to
    combine fundamental, technical, and market context analyses into
    an actionable investment recommendation.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with synthesis and recommendation populated.
    """
    state = _ensure_state(state_input)
    try:
        logger.info(f"Running synthesis for {state.symbol}")

        # Collect available analyses
        available_analyses = []
        scores = {}

        # Get weight distribution from SYNTHESIS_AGENT_SPEC
        weights = SYNTHESIS_AGENT_SPEC.metadata.get(
            "weight_distribution",
            {
                "fundamental": 0.35,
                "technical": 0.20,
                "market_context": 0.15,
                "sentiment": 0.15,
                "sec_quality": 0.15,
            },
        )

        # Get decision thresholds
        thresholds = SYNTHESIS_AGENT_SPEC.metadata.get(
            "decision_thresholds",
            {
                "strong_buy": 80,
                "buy": 65,
                "hold_upper": 65,
                "hold_lower": 35,
                "sell": 35,
                "strong_sell": 20,
            },
        )

        # Process fundamental analysis
        if state.fundamental_analysis and state.fundamental_analysis.get("status") == "success":
            available_analyses.append("fundamental")
            valuation_data = state.fundamental_analysis.get("valuation_models", {})
            # Extract composite score if available, otherwise estimate from upside
            if isinstance(valuation_data, dict):
                composite_upside = valuation_data.get("composite_upside_percent")
                if composite_upside is not None:
                    # Convert upside % to score (0-100 scale)
                    # +30% upside = 80, 0% = 50, -30% = 20
                    scores["fundamental"] = min(100, max(0, 50 + (composite_upside * 1.0)))

        # Process technical analysis
        if state.technical_analysis and state.technical_analysis.get("status") == "success":
            available_analyses.append("technical")
            indicators = state.technical_analysis.get("indicators", {})
            trend = state.technical_analysis.get("trend", {})
            # Use trend signal for scoring
            if isinstance(trend, dict):
                trend_signal = trend.get("overall_signal", "neutral")
                trend_scores = {"bullish": 75, "neutral": 50, "bearish": 25}
                scores["technical"] = trend_scores.get(trend_signal, 50)

        # Process market context
        if state.market_context and state.market_context.get("status") == "success":
            available_analyses.append("market_context")
            rel_perf = state.market_context.get("relative_performance_vs_spy")
            if rel_perf is not None:
                # +20% outperformance = 80, 0% = 50, -20% = 20
                scores["market_context"] = min(100, max(0, 50 + (rel_perf * 1.5)))

        if not available_analyses:
            raise ValueError("No analysis results available for synthesis")

        # Calculate weighted composite score
        weighted_sum = 0.0
        total_weight = 0.0

        for analysis_type, score in scores.items():
            weight = weights.get(analysis_type, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        composite_score = weighted_sum / total_weight if total_weight > 0 else 50

        # Determine recommendation based on thresholds
        if composite_score >= thresholds["strong_buy"]:
            recommendation_action = "STRONG BUY"
            confidence = "high"
        elif composite_score >= thresholds["buy"]:
            recommendation_action = "BUY"
            confidence = "medium-high"
        elif composite_score >= thresholds["hold_lower"]:
            recommendation_action = "HOLD"
            confidence = "medium"
        elif composite_score >= thresholds["strong_sell"]:
            recommendation_action = "SELL"
            confidence = "medium-high"
        else:
            recommendation_action = "STRONG SELL"
            confidence = "high"

        # Try LLM synthesis for enhanced narrative
        llm_synthesis = None
        try:
            llm_synthesis = await _run_llm_synthesis(
                state.symbol,
                state.technical_analysis or {},
                state.fundamental_analysis or {},
                state.market_context or {},
                composite_score,
                recommendation_action,
            )
        except Exception as e:
            logger.warning(f"LLM synthesis failed, using rule-based: {e}")

        state.synthesis = {
            "symbol": state.symbol,
            "analyses_included": available_analyses,
            "individual_scores": scores,
            "weights_applied": {k: v for k, v in weights.items() if k in scores},
            "composite_score": round(composite_score, 2),
            "agent_spec": SYNTHESIS_AGENT_SPEC.name,
            "mode": state.mode.value,
            "status": "success",
            # LLM-generated content (if available)
            "synthesis_method": "llm" if llm_synthesis else "rule_based",
            "executive_summary": llm_synthesis.get("executive_summary", "") if llm_synthesis else "",
            "key_catalysts": llm_synthesis.get("key_catalysts", []) if llm_synthesis else [],
            "key_risks": llm_synthesis.get("key_risks", []) if llm_synthesis else [],
            "reasoning": llm_synthesis.get("reasoning", "") if llm_synthesis else "",
        }

        state.recommendation = {
            "symbol": state.symbol,
            "action": (
                llm_synthesis.get("recommendation", recommendation_action) if llm_synthesis else recommendation_action
            ),
            "composite_score": round(composite_score, 2),
            "confidence": llm_synthesis.get("confidence", confidence) if llm_synthesis else confidence,
            "analyses_included": available_analyses,
            "thresholds_used": thresholds,
            "errors_during_analysis": state.errors,
            "executive_summary": llm_synthesis.get("executive_summary", "") if llm_synthesis else "",
            "key_catalysts": llm_synthesis.get("key_catalysts", []) if llm_synthesis else [],
            "key_risks": llm_synthesis.get("key_risks", []) if llm_synthesis else [],
        }

        state.mark_step_completed("synthesis")
    except Exception as e:
        error_msg = f"Synthesis failed: {e}"
        logger.error(error_msg)
        state.add_error(error_msg)
        state.synthesis = {"symbol": state.symbol, "status": "error", "error": str(e)}
        state.recommendation = {"symbol": state.symbol, "action": "UNABLE TO RECOMMEND", "error": str(e)}
    return _state_to_dict(state)


# =============================================================================
# Parallel Execution Helpers
# =============================================================================


async def fetch_data_parallel(state_input) -> dict:
    """Fetch SEC and market data in parallel.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with both sec_data and market_data populated.
    """
    state = _ensure_state(state_input)
    logger.info(f"Fetching data in parallel for {state.symbol}")

    # Run both fetches concurrently - they return dicts
    sec_task = asyncio.create_task(fetch_sec_data(state.to_dict()))
    market_task = asyncio.create_task(fetch_market_data(state.to_dict()))

    # Wait for both to complete
    results = await asyncio.gather(sec_task, market_task, return_exceptions=True)

    # Merge results back into state
    for result in results:
        if isinstance(result, dict):
            if "sec_data" in result and result["sec_data"]:
                state.sec_data = result["sec_data"]
            if "market_data" in result and result["market_data"]:
                state.market_data = result["market_data"]
            if result.get("errors"):
                state.errors.extend(result["errors"])

    state.mark_step_completed("parallel_data_fetch")
    return _state_to_dict(state)


async def run_analyses_parallel_standard(state_input) -> dict:
    """Run fundamental and technical analyses in parallel.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with both analyses populated.
    """
    state = _ensure_state(state_input)
    logger.info(f"Running standard analyses in parallel for {state.symbol}")

    # Run both analyses concurrently
    fundamental_task = asyncio.create_task(run_fundamental_analysis(state.to_dict()))
    technical_task = asyncio.create_task(run_technical_analysis(state.to_dict()))

    results = await asyncio.gather(fundamental_task, technical_task, return_exceptions=True)

    # Merge results back into state
    for result in results:
        if isinstance(result, dict):
            if "fundamental_analysis" in result and result["fundamental_analysis"]:
                state.fundamental_analysis = result["fundamental_analysis"]
            if "technical_analysis" in result and result["technical_analysis"]:
                state.technical_analysis = result["technical_analysis"]
            if result.get("errors"):
                state.errors.extend(result["errors"])

    state.mark_step_completed("parallel_standard_analysis")
    return _state_to_dict(state)


async def run_analyses_parallel_comprehensive(state_input) -> dict:
    """Run all analyses in parallel.

    Args:
        state_input: Current workflow state (dict or AnalysisWorkflowState).

    Returns:
        Updated state dict with all analyses populated.
    """
    state = _ensure_state(state_input)
    logger.info(f"Running comprehensive analyses in parallel for {state.symbol}")

    # Run all three analyses concurrently
    fundamental_task = asyncio.create_task(run_fundamental_analysis(state.to_dict()))
    technical_task = asyncio.create_task(run_technical_analysis(state.to_dict()))
    context_task = asyncio.create_task(run_market_context_analysis(state.to_dict()))

    results = await asyncio.gather(fundamental_task, technical_task, context_task, return_exceptions=True)

    # Merge results back into state
    for result in results:
        if isinstance(result, dict):
            if "fundamental_analysis" in result and result["fundamental_analysis"]:
                state.fundamental_analysis = result["fundamental_analysis"]
            if "technical_analysis" in result and result["technical_analysis"]:
                state.technical_analysis = result["technical_analysis"]
            if "market_context" in result and result["market_context"]:
                state.market_context = result["market_context"]
            if result.get("errors"):
                state.errors.extend(result["errors"])

    state.mark_step_completed("parallel_comprehensive_analysis")
    return _state_to_dict(state)


# =============================================================================
# Graph Builders
# =============================================================================


def build_quick_graph() -> StateGraph:
    """Build Quick analysis workflow graph.

    Quick workflow: START -> market_data -> technical -> END

    This is the fastest workflow, focusing only on technical analysis
    using market data.

    Returns:
        StateGraph configured for quick analysis.
    """
    graph = StateGraph()

    # Add nodes
    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("technical_analysis", run_technical_analysis)
    graph.add_node("synthesis", run_synthesis)

    # Add edges: linear flow
    graph.add_edge("fetch_market_data", "technical_analysis")
    graph.add_edge("technical_analysis", "synthesis")
    graph.add_edge("synthesis", END)

    # Set entry point
    graph.set_entry_point("fetch_market_data")

    logger.debug("Built quick analysis graph")
    return graph


def build_standard_graph() -> StateGraph:
    """Build Standard analysis workflow graph.

    Standard workflow:
        START -> parallel_data_fetch (SEC + market)
        -> parallel_analysis (fundamental + technical)
        -> synthesis -> END

    This provides balanced analysis combining fundamental and technical
    perspectives.

    Returns:
        StateGraph configured for standard analysis.
    """
    graph = StateGraph()

    # Add nodes
    graph.add_node("parallel_data_fetch", fetch_data_parallel)
    graph.add_node("parallel_analysis", run_analyses_parallel_standard)
    graph.add_node("synthesis", run_synthesis)

    # Add edges
    graph.add_edge("parallel_data_fetch", "parallel_analysis")
    graph.add_edge("parallel_analysis", "synthesis")
    graph.add_edge("synthesis", END)

    # Set entry point
    graph.set_entry_point("parallel_data_fetch")

    logger.debug("Built standard analysis graph")
    return graph


def build_comprehensive_graph() -> StateGraph:
    """Build Comprehensive analysis workflow graph.

    Comprehensive workflow:
        START -> parallel_data_fetch (SEC + market)
        -> parallel_analysis (fundamental + technical + market_context)
        -> synthesis -> END

    This is the full institutional-grade analysis using all available agents.

    Returns:
        StateGraph configured for comprehensive analysis.
    """
    graph = StateGraph()

    # Add nodes
    graph.add_node("parallel_data_fetch", fetch_data_parallel)
    graph.add_node("parallel_analysis", run_analyses_parallel_comprehensive)
    graph.add_node("synthesis", run_synthesis)

    # Add edges
    graph.add_edge("parallel_data_fetch", "parallel_analysis")
    graph.add_edge("parallel_analysis", "synthesis")
    graph.add_edge("synthesis", END)

    # Set entry point
    graph.set_entry_point("parallel_data_fetch")

    logger.debug("Built comprehensive analysis graph")
    return graph


def build_graph_for_mode(mode: AnalysisMode) -> StateGraph:
    """Factory function to build appropriate graph for analysis mode.

    Args:
        mode: The analysis mode determining workflow structure.

    Returns:
        StateGraph configured for the specified mode.

    Raises:
        ValueError: If mode is not supported.

    Example:
        graph = build_graph_for_mode(AnalysisMode.STANDARD)
        compiled = graph.compile()
        result = await compiled.invoke(state)
    """
    builders: Dict[AnalysisMode, Callable[[], StateGraph]] = {
        AnalysisMode.QUICK: build_quick_graph,
        AnalysisMode.STANDARD: build_standard_graph,
        AnalysisMode.COMPREHENSIVE: build_comprehensive_graph,
    }

    if mode == AnalysisMode.CUSTOM:
        # Custom mode defaults to comprehensive
        logger.info("Custom mode requested, using comprehensive workflow")
        return build_comprehensive_graph()

    builder = builders.get(mode)
    if builder is None:
        raise ValueError(f"Unsupported analysis mode: {mode}")

    return builder()


# =============================================================================
# Execution Helpers
# =============================================================================


async def run_analysis(
    symbol: str,
    mode: AnalysisMode = AnalysisMode.STANDARD,
) -> AnalysisWorkflowState:
    """Convenience function to run a complete analysis workflow.

    Args:
        symbol: Stock ticker symbol to analyze.
        mode: Analysis mode (default: STANDARD).

    Returns:
        Final AnalysisWorkflowState with results.

    Example:
        result = await run_analysis("AAPL", AnalysisMode.COMPREHENSIVE)
        print(result.synthesis)
    """
    # Create initial state
    state = AnalysisWorkflowState(symbol=symbol, mode=mode)

    # Build and compile graph
    graph = build_graph_for_mode(mode)
    compiled = graph.compile()

    # Execute workflow
    result = await compiled.invoke(state.to_dict())

    # Convert result back to state - handle various result formats
    if hasattr(result, "state"):
        # Victor returns a result object with .state attribute
        result_data = result.state
    elif isinstance(result, dict):
        # Direct dict return
        result_data = result
    else:
        # Unknown format, try to use as-is
        result_data = result

    if isinstance(result_data, dict):
        return AnalysisWorkflowState.from_dict(result_data)
    elif isinstance(result_data, AnalysisWorkflowState):
        return result_data
    else:
        # Fallback - return the original state with any updates
        return state


async def run_yaml_analysis(
    symbol: str,
    mode: AnalysisMode = AnalysisMode.STANDARD,
) -> AnalysisWorkflowState:
    """Run analysis using YAML workflow with BaseYAMLProvider and handlers.

    This execution path uses the YAML workflow definitions with registered
    compute handlers from handlers.py. The handlers implement the actual
    business logic including LLM synthesis via OllamaClient.

    This approach provides:
    - YAML-defined workflow DAG for flexibility
    - Handlers.py for actual implementation
    - LLM synthesis via RunSynthesisHandler with OllamaClient
    - Escape hatches for complex conditions

    Args:
        symbol: Stock ticker symbol to analyze.
        mode: Analysis mode (default: STANDARD).

    Returns:
        Final AnalysisWorkflowState with results including LLM synthesis.

    Example:
        result = await run_yaml_analysis("AAPL", AnalysisMode.COMPREHENSIVE)
        print(result.synthesis)  # LLM-synthesized investment narrative
    """
    from victor_invest.handlers import HANDLERS
    from victor_invest.workflows import InvestmentWorkflowProvider

    # Map mode to workflow name
    workflow_map = {
        AnalysisMode.QUICK: "quick",
        AnalysisMode.STANDARD: "standard",
        AnalysisMode.COMPREHENSIVE: "comprehensive",
    }
    workflow_name = workflow_map.get(mode, "standard")

    logger.info(f"Running YAML workflow '{workflow_name}' for {symbol} using handlers")

    # Initialize workflow provider
    provider = InvestmentWorkflowProvider()

    # Get the workflow definition
    workflow = provider.get_workflow(workflow_name)
    if not workflow:
        raise ValueError(f"Workflow '{workflow_name}' not found")

    # Create workflow context
    class WorkflowContext:
        """Simple context for workflow execution."""

        def __init__(self, initial_data: dict):
            self._data = initial_data.copy()

        def get(self, key: str, default=None):
            return self._data.get(key, default)

        def set(self, key: str, value):
            self._data[key] = value

        def all(self) -> dict:
            return self._data.copy()

    # Create mock node for handlers
    class MockNode:
        def __init__(self, node_id: str, output_key: str = None):
            self.id = node_id
            self.output_key = output_key

    ctx = WorkflowContext(
        {
            "symbol": symbol.upper(),
            "mode": mode.value,
        }
    )

    # Phase 1: Fetch data in parallel
    logger.info(f"Fetching data in parallel for {symbol}")

    async def execute_handler(handler_name: str, node_id: str, output_key: str):
        handler = HANDLERS.get(handler_name)
        if handler:
            try:
                result = await handler(MockNode(node_id, output_key), ctx, None)
                return result.output if hasattr(result, "output") else {}
            except Exception as e:
                logger.warning(f"Handler {handler_name} failed: {e}")
                return {"status": "error", "error": str(e)}
        return {"status": "error", "error": f"Handler {handler_name} not found"}

    # Execute data fetches in parallel
    sec_result, market_result = await asyncio.gather(
        execute_handler("fetch_sec_data", "fetch_sec_data", "sec_data"),
        execute_handler("fetch_market_data", "fetch_market_data", "market_data"),
    )
    ctx.set("sec_data", sec_result)
    ctx.set("market_data", market_result)

    # Phase 2: Run analyses in parallel
    logger.info(f"Running analyses in parallel for {symbol}")

    fundamental_result, technical_result, market_context_result = await asyncio.gather(
        execute_handler("run_fundamental_analysis", "fundamental_analysis", "fundamental_analysis"),
        execute_handler("run_technical_analysis", "technical_analysis", "technical_analysis"),
        execute_handler("run_market_context_analysis", "market_context_analysis", "market_context"),
    )
    ctx.set("fundamental_analysis", fundamental_result)
    ctx.set("technical_analysis", technical_result)
    ctx.set("market_context", market_context_result)

    # Phase 3: LLM Synthesis using RunSynthesisHandler
    logger.info(f"Running LLM synthesis for {symbol}")
    synthesis_result = await execute_handler("run_synthesis", "synthesize", "synthesis")
    ctx.set("synthesis", synthesis_result)

    # Build recommendation from synthesis
    synthesis = ctx.get("synthesis", {})
    recommendation = {
        "action": synthesis.get("recommendation", "HOLD"),
        "confidence": synthesis.get("confidence", "MEDIUM"),
        "key_catalysts": synthesis.get("key_catalysts", []),
        "key_risks": synthesis.get("key_risks", []),
        "executive_summary": synthesis.get("executive_summary", ""),
        "reasoning": synthesis.get("reasoning", ""),
    }
    ctx.set("recommendation", recommendation)

    # Convert to AnalysisWorkflowState
    return AnalysisWorkflowState(
        symbol=symbol.upper(),
        mode=mode,
        fundamental_analysis=ctx.get("fundamental_analysis"),
        technical_analysis=ctx.get("technical_analysis"),
        market_context=ctx.get("market_context"),
        synthesis=ctx.get("synthesis"),
        recommendation=ctx.get("recommendation"),
        errors=[],
    )


__all__ = [
    # Graph builders
    "build_quick_graph",
    "build_standard_graph",
    "build_comprehensive_graph",
    "build_graph_for_mode",
    # Execution helpers
    "run_analysis",
    "run_yaml_analysis",
    # Node functions (for testing/extension)
    "fetch_sec_data",
    "fetch_market_data",
    "run_fundamental_analysis",
    "run_technical_analysis",
    "run_market_context_analysis",
    "run_synthesis",
    # Convenience
    "run_analysis",
]
