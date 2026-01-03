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

"""Domain handlers for Investment vertical workflows.

Registers compute node handlers for investment analysis workflows.
These handlers are invoked by YAML workflow nodes of type "compute".

Example YAML usage:
    - id: fetch_sec_data
      type: compute
      handler: fetch_sec_data
      output: sec_data

Usage:
    from victor_invest import handlers
    handlers.register_handlers()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import NodeResult, WorkflowContext

logger = logging.getLogger(__name__)

# Track registration state
_handlers_registered = False


# =============================================================================
# Data Collection Handlers
# =============================================================================


@dataclass
class FetchSECDataHandler:
    """Fetch SEC filing data for analysis."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()
        symbol = context.get("symbol", "")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No symbol provided",
                duration=time.time() - start_time,
            )

        try:
            # Use the SEC filing tool
            result = await tool_registry.execute(
                "sec_filing",
                action="get_company_facts",
                symbol=symbol,
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
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"SEC data fetch error for {symbol}: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class FetchMarketDataHandler:
    """Fetch market/price data for analysis."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()
        symbol = context.get("symbol", "")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No symbol provided",
                duration=time.time() - start_time,
            )

        try:
            result = await tool_registry.execute(
                "market_data",
                action="get_price_history",
                symbol=symbol,
                period="1y",
            )

            output = {
                "status": "success" if result.success else "error",
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }

            output_key = node.output_key or "market_data"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Market data fetch error for {symbol}: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class FetchMacroDataHandler:
    """Fetch macroeconomic data for context."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        try:
            from investigator.domain.services.data_sources import DataSourceManager
            from datetime import date

            symbol = context.get("symbol", "SPY")
            manager = DataSourceManager()
            consolidated = manager.get_data(symbol=symbol, as_of_date=date.today())

            macro_data = {
                "treasury": {},
                "volatility": {},
                "fed_indicators": {},
                "status": "success",
            }

            if consolidated.treasury_yield and consolidated.treasury_yield.data:
                treasury = consolidated.treasury_yield.data
                macro_data["treasury"] = {
                    "yield_10y": treasury.get("DGS10"),
                    "yield_2y": treasury.get("DGS2"),
                    "yield_curve_slope": treasury.get("curve_slope"),
                }

            if consolidated.cboe_volatility and consolidated.cboe_volatility.data:
                vol = consolidated.cboe_volatility.data
                macro_data["volatility"] = {
                    "vix": vol.get("vix"),
                    "skew": vol.get("skew"),
                }

            output_key = node.output_key or "macro_data"
            context.set(output_key, macro_data)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=macro_data,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Macro data fetch error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


# =============================================================================
# Analysis Handlers
# =============================================================================


@dataclass
class RunFundamentalAnalysisHandler:
    """Run fundamental analysis on SEC data."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()
        sec_data = context.get("sec_data", {})

        if sec_data.get("status") != "success":
            output = {"status": "skipped", "reason": "No SEC data"}
            context.set(node.output_key or "fundamental_analysis", output)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        try:
            symbol = context.get("symbol", "")
            result = await tool_registry.execute(
                "valuation",
                action="full_valuation",
                symbol=symbol,
            )

            output = {
                "status": "success" if result.success else "error",
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }

            output_key = node.output_key or "fundamental_analysis"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Fundamental analysis error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class RunTechnicalAnalysisHandler:
    """Run technical analysis on market data."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()
        market_data = context.get("market_data", {})

        if market_data.get("status") != "success":
            output = {"status": "skipped", "reason": "No market data"}
            context.set(node.output_key or "technical_analysis", output)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        try:
            symbol = context.get("symbol", "")
            result = await tool_registry.execute(
                "technical_indicators",
                action="full_analysis",
                symbol=symbol,
            )

            output = {
                "status": "success" if result.success else "error",
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }

            output_key = node.output_key or "technical_analysis"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class RunMarketContextHandler:
    """Analyze market context and sector dynamics."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        try:
            market_data = context.get("market_data", {})
            macro_data = context.get("macro_data", {})

            output = {
                "status": "success",
                "market_regime": "unknown",
                "sector_momentum": "neutral",
                "macro_environment": {},
            }

            if macro_data.get("status") == "success":
                vix = macro_data.get("volatility", {}).get("vix")
                if vix:
                    if vix < 15:
                        output["market_regime"] = "low_volatility"
                    elif vix < 25:
                        output["market_regime"] = "normal"
                    elif vix < 35:
                        output["market_regime"] = "elevated"
                    else:
                        output["market_regime"] = "high_volatility"

                output["macro_environment"] = {
                    "treasury": macro_data.get("treasury", {}),
                    "vix": vix,
                }

            output_key = node.output_key or "market_context"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Market context analysis error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class RunSynthesisHandler:
    """Synthesize all analyses into recommendation."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        try:
            fundamental = context.get("fundamental_analysis", {})
            technical = context.get("technical_analysis", {})
            market_context = context.get("market_context", {})

            fundamental_score = fundamental.get("data", {}).get("overall_score", 50)
            technical_score = technical.get("data", {}).get("overall_score", 50)

            composite_score = fundamental_score * 0.6 + technical_score * 0.4

            if composite_score >= 70:
                recommendation = "buy"
                confidence = "high"
            elif composite_score >= 55:
                recommendation = "buy"
                confidence = "medium"
            elif composite_score >= 45:
                recommendation = "hold"
                confidence = "medium"
            elif composite_score >= 30:
                recommendation = "sell"
                confidence = "medium"
            else:
                recommendation = "sell"
                confidence = "high"

            output = {
                "status": "success",
                "composite_score": composite_score,
                "individual_scores": {
                    "fundamental": fundamental_score,
                    "technical": technical_score,
                },
                "recommendation": {
                    "action": recommendation,
                    "confidence": confidence,
                },
                "market_regime": market_context.get("market_regime", "unknown"),
            }

            output_key = node.output_key or "synthesis"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class GenerateReportHandler:
    """Generate PDF report from analysis."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        try:
            from investigator.infrastructure.reporting.report_generator import ReportGenerator
            from datetime import datetime

            synthesis = context.get("synthesis", {})
            symbol = context.get("symbol", "UNKNOWN")

            report_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "synthesis": synthesis,
                "fundamental": context.get("fundamental_analysis", {}),
                "technical": context.get("technical_analysis", {}),
                "market_context": context.get("market_context", {}),
            }

            generator = ReportGenerator()
            report_path = generator.generate(report_data, report_type="investment")

            output = {"path": str(report_path), "status": "success"}
            output_key = node.output_key or "report"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class IdentifyPeersHandler:
    """Identify peer companies for comparison."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        try:
            from investigator.infrastructure.database.db import get_database_engine
            from sqlalchemy import text

            symbol = context.get("symbol", "")
            market_context = context.get("market_context", {})
            sector = market_context.get("sector")

            peers = []

            if sector:
                engine = get_database_engine()
                with engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT DISTINCT symbol, name, market_cap
                            FROM symbols
                            WHERE sector = :sector
                            AND symbol != :target
                            AND is_active = true
                            ORDER BY market_cap DESC NULLS LAST
                            LIMIT 5
                        """),
                        {"sector": sector, "target": symbol}
                    )
                    for row in result:
                        peers.append({
                            "symbol": row[0],
                            "name": row[1],
                            "market_cap": float(row[2]) if row[2] else None
                        })

            output_key = node.output_key or "peer_list"
            context.set(output_key, peers)
            context.set("has_peers", len(peers) > 0)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=peers,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Peer identification error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class AnalyzePeersHandler:
    """Analyze peer companies."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()
        peers = context.get("peer_list", [])

        if not peers:
            output_key = node.output_key or "peer_analyses"
            context.set(output_key, [])
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=[],
                duration=time.time() - start_time,
            )

        try:
            import asyncio
            from victor_invest.workflows import run_analysis, AnalysisMode

            async def analyze_one(peer):
                symbol = peer.get("symbol") if isinstance(peer, dict) else peer
                try:
                    result = await run_analysis(symbol, AnalysisMode.QUICK)
                    return {
                        "symbol": symbol,
                        "composite_score": result.synthesis.get("composite_score", 50) if result.synthesis else 50,
                        "status": "success",
                    }
                except Exception as e:
                    return {"symbol": symbol, "status": "error", "error": str(e)}

            tasks = [analyze_one(p) for p in peers[:5]]
            peer_analyses = await asyncio.gather(*tasks)

            output_key = node.output_key or "peer_analyses"
            context.set(output_key, peer_analyses)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=peer_analyses,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Peer analysis error: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


# =============================================================================
# RL Backtest Handlers
# =============================================================================


@dataclass
class GenerateLookbackDatesHandler:
    """Generate lookback dates for RL backtesting."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        try:
            from victor_invest.workflows.rl_backtest import generate_lookback_list

            max_months = context.get("max_lookback_months", 120)
            interval = context.get("interval", "quarterly")

            lookback_dates = generate_lookback_list(max_months, interval)

            output_key = node.output_key or "lookback_dates"
            context.set(output_key, lookback_dates)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=lookback_dates,
                duration=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"GenerateLookbackDatesHandler failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class ProcessBacktestBatchHandler:
    """Process a batch of backtest dates for RL training."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        try:
            from victor_invest.workflows.rl_backtest import run_rl_backtest

            symbol = context.get("symbol")
            lookback_dates = context.get("lookback_dates", [])
            interval = context.get("interval", "quarterly")

            if not symbol:
                return NodeResult(
                    node_id=node.id,
                    status=NodeStatus.FAILED,
                    error="No symbol provided",
                    duration=time.time() - start_time,
                )

            result = await run_rl_backtest(
                symbol=symbol,
                lookback_months_list=lookback_dates,
                interval=interval,
            )

            output_key = node.output_key or "backtest_results"
            context.set(output_key, result.to_dict())

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=result.to_dict(),
                duration=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"ProcessBacktestBatchHandler failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


@dataclass
class SaveRLPredictionsHandler:
    """Save RL predictions to database."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        try:
            backtest_results = context.get("backtest_results", {})

            # The predictions are already saved during run_rl_backtest
            # This handler just returns the summary
            predictions = backtest_results.get("predictions", [])
            metadata = backtest_results.get("metadata", {})

            output = {
                "predictions_count": len(predictions),
                "summary": metadata.get("summary", {}),
            }

            output_key = node.output_key or "saved_predictions"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"SaveRLPredictionsHandler failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )


# =============================================================================
# Handler Registry
# =============================================================================


HANDLERS = {
    "fetch_sec_data": FetchSECDataHandler(),
    "fetch_market_data": FetchMarketDataHandler(),
    "fetch_macro_data": FetchMacroDataHandler(),
    "run_fundamental_analysis": RunFundamentalAnalysisHandler(),
    "run_technical_analysis": RunTechnicalAnalysisHandler(),
    "run_market_context_analysis": RunMarketContextHandler(),
    "run_synthesis": RunSynthesisHandler(),
    "generate_report": GenerateReportHandler(),
    "identify_peers": IdentifyPeersHandler(),
    "analyze_peers": AnalyzePeersHandler(),
    # RL Backtest handlers
    "generate_lookback_dates": GenerateLookbackDatesHandler(),
    "process_backtest_batch": ProcessBacktestBatchHandler(),
    "save_rl_predictions": SaveRLPredictionsHandler(),
}


def register_handlers() -> None:
    """Register Investment handlers with the workflow executor.

    This function should be called once when the workflows module is loaded.
    Subsequent calls are no-ops to prevent duplicate registration.
    """
    global _handlers_registered
    if _handlers_registered:
        return

    from victor.workflows.executor import register_compute_handler

    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered Investment handler: {name}")

    _handlers_registered = True
    logger.info("Investment domain handlers registered successfully")


__all__ = [
    "FetchSECDataHandler",
    "FetchMarketDataHandler",
    "FetchMacroDataHandler",
    "RunFundamentalAnalysisHandler",
    "RunTechnicalAnalysisHandler",
    "RunMarketContextHandler",
    "RunSynthesisHandler",
    "GenerateReportHandler",
    "IdentifyPeersHandler",
    "AnalyzePeersHandler",
    # RL Backtest handlers
    "GenerateLookbackDatesHandler",
    "ProcessBacktestBatchHandler",
    "SaveRLPredictionsHandler",
    "HANDLERS",
    "register_handlers",
]
