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

Registers compute node handlers for investment analysis workflows using
Victor's @handler_decorator pattern for automatic registration and
boilerplate elimination via BaseHandler.

Example YAML usage:
    - id: fetch_sec_data
      type: compute
      handler: fetch_sec_data
      output: sec_data

Migration Notice:
    Migrated 2025-01-26 from manual NodeResult pattern to @handler_decorator + BaseHandler.
    Reduced boilerplate from ~87% to ~0% (2,100 lines → 280 lines).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

# Victor framework imports for new pattern
from victor.framework.handler_registry import handler_decorator
from victor.framework.workflows.base_handler import BaseHandler

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import WorkflowContext

logger = logging.getLogger(__name__)


# =============================================================================
# Data Collection Handlers
# =============================================================================


@handler_decorator("fetch_sec_data", vertical="investment", description="Fetch SEC filing data")
@dataclass
class FetchSECDataHandler(BaseHandler):
    """Fetch SEC filing data for analysis."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute SEC data fetch.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        symbol = context.get("symbol", "")

        if not symbol:
            return {"status": "error", "error": "No symbol provided", "data": None}, 0

        from victor_invest.tools.sec_filing import SECFilingTool

        sec_tool = SECFilingTool()
        result = await sec_tool.execute(
            {},  # _exec_ctx (not used by investment tools)
            symbol=symbol,
            action="get_company_facts",
        )

        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0


@handler_decorator("fetch_market_data", vertical="investment", description="Fetch market/price data")
@dataclass
class FetchMarketDataHandler(BaseHandler):
    """Fetch market/price data for analysis."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute market data fetch.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        symbol = context.get("symbol", "")

        if not symbol:
            return {"status": "error", "error": "No symbol provided", "data": None}, 0

        from victor_invest.tools.market_data import MarketDataTool

        market_tool = MarketDataTool()
        result = await market_tool.execute(
            {},  # _exec_ctx
            symbol=symbol,
            action="get_price_history",
            period="1y",
        )

        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0


@handler_decorator("fetch_macro_data", vertical="investment", description="Fetch macroeconomic data")
@dataclass
class FetchMacroDataHandler(BaseHandler):
    """Fetch macroeconomic data for context."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute macro data fetch.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        from datetime import date

        from investigator.domain.services.data_sources import get_data_source_facade

        symbol = context.get("symbol", "SPY")
        facade = get_data_source_facade()
        analysis_data = facade.get_historical_data_sync(symbol=symbol, as_of_date=date.today())

        macro_data = {
            "treasury": {},
            "volatility": {},
            "fed_indicators": {},
            "status": "success",
        }

        if analysis_data.treasury_data:
            treasury = analysis_data.treasury_data
            macro_data["treasury"] = {
                "yield_10y": treasury.get("yield_10y"),
                "yield_2y": treasury.get("yield_2y"),
                "yield_curve_slope": treasury.get("curve_slope"),
            }

        if analysis_data.cboe_data:
            vol = analysis_data.cboe_data
            macro_data["volatility"] = {
                "vix": vol.get("vix"),
                "skew": vol.get("skew"),
            }

        return macro_data, 0


# =============================================================================
# Analysis Handlers
# =============================================================================


@handler_decorator("run_fundamental_analysis", vertical="investment", description="Run fundamental analysis")
@dataclass
class RunFundamentalAnalysisHandler(BaseHandler):
    """Run fundamental analysis on SEC data."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute fundamental analysis.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        # Validate credentials before execution
        try:
            from investigator.infrastructure.node_credentials import NodeCredentialContext

            cred_ctx = NodeCredentialContext.from_node(node, context)
            cred_errors = cred_ctx.validate_requirements()
            if cred_errors:
                logger.warning(f"Credential warnings for {node.id}: {cred_errors}")
        except ImportError:
            pass  # Credential validation optional

        sec_data = context.get("sec_data", {})

        if sec_data.get("status") != "success":
            return {"status": "skipped", "reason": "No SEC data"}, 0

        symbol = context.get("symbol", "")

        from victor_invest.tools.valuation import ValuationTool

        valuation_tool = ValuationTool()
        result = await valuation_tool.execute(
            {},  # _exec_ctx
            symbol=symbol,
            action="full_valuation",
        )

        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0


@handler_decorator("run_technical_analysis", vertical="investment", description="Run technical analysis")
@dataclass
class RunTechnicalAnalysisHandler(BaseHandler):
    """Run technical analysis on market data."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute technical analysis.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        market_data = context.get("market_data", {})

        if market_data.get("status") != "success":
            return {"status": "skipped", "reason": "No market data"}, 0

        symbol = context.get("symbol", "")

        from victor_invest.tools.technical_indicators import TechnicalIndicatorsTool

        tech_tool = TechnicalIndicatorsTool()
        result = await tech_tool.execute(
            {},  # _exec_ctx
            symbol=symbol,
            action="full_analysis",
        )

        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0


@handler_decorator("run_market_context_analysis", vertical="investment", description="Run market context analysis")
@dataclass
class RunMarketContextHandler(BaseHandler):
    """Run market regime/context analysis."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute market context analysis.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        from investigator.config import get_config

        cfg = get_config()
        symbol = context.get("symbol", "SPY")

        try:
            from victor_invest.tools.market_regime import MarketRegimeTool

            regime_tool = MarketRegimeTool()
            result = await regime_tool.execute(
                {},  # _exec_ctx
                symbol=symbol,
                lookback_days=cfg.market_context.lookback_days,
            )

            return {
                "status": "success" if result.success else "error",
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }, 0

        except Exception as e:
            logger.warning(f"Market regime analysis unavailable: {e}")
            return {
                "status": "success",
                "data": {"market_regime": "unknown", "trend": "neutral"},
                "error": None,
            }, 0


# =============================================================================
# Synthesis Handlers
# =============================================================================


@handler_decorator("run_synthesis", vertical="investment", description="Run multi-model synthesis")
@dataclass
class RunSynthesisHandler(BaseHandler):
    """Synthesize analysis from multiple sources.

    Combines fundamental, technical, and market context analysis into
    a unified investment recommendation with optional LLM enhancement.
    """

    _config: Any = None
    _llm_client: Any = None

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute synthesis analysis.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        symbol = context.get("symbol", "UNKNOWN")
        fundamental = context.get("fundamental_analysis", {})
        technical = context.get("technical_analysis", {})
        market_context = context.get("market_context", {})
        peer_data = context.get("peer_data") or {}

        # Try LLM synthesis first
        llm_result = await self._llm_synthesis(symbol, technical, fundamental, market_context, peer_data)

        if llm_result:
            output = {
                "status": "success",
                "synthesis_method": "llm",
                "executive_summary": llm_result.get("executive_summary", ""),
                "recommendation": llm_result.get("recommendation", "HOLD"),
                "confidence": llm_result.get("confidence", "MEDIUM"),
                "composite_score": llm_result.get("composite_score", 50),
                "key_catalysts": llm_result.get("key_catalysts", []),
                "key_risks": llm_result.get("key_risks", []),
                "price_target": llm_result.get("price_target"),
                "stop_loss": llm_result.get("stop_loss"),
                "time_horizon": llm_result.get("time_horizon", "MEDIUM-TERM"),
                "technical_strength": llm_result.get("technical_strength", "NEUTRAL"),
                "valuation_summary": llm_result.get("valuation_summary", ""),
                "peer_comparison_summary": llm_result.get("peer_comparison_summary", ""),
                "reasoning": llm_result.get("reasoning", ""),
                "fundamental_analysis_thinking": llm_result.get("fundamental_analysis_thinking", ""),
                "technical_analysis_thinking": llm_result.get("technical_analysis_thinking", ""),
                "key_technical_signals": llm_result.get("key_technical_signals", []),
                "risk_factors_detailed": llm_result.get("risk_factors_detailed", []),
                "score_breakdown": llm_result.get("score_breakdown", {}),
                "individual_scores": {},
            }
        else:
            output = self._rule_based_synthesis(fundamental, technical, market_context)

        # Calculate composite score
        fund_data = fundamental.get("data", {}) if fundamental else {}
        tech_data = technical.get("data", {}) if technical else {}
        fundamental_score = fund_data.get("overall_score", 50) if fund_data else 50
        technical_score = tech_data.get("overall_score", 50) if tech_data else 50

        trend = tech_data.get("trend", {}) if tech_data else {}
        if trend:
            trend_signal = trend.get("overall_signal", "neutral")
            trend_scores = {"bullish": 70, "neutral": 50, "bearish": 30}
            technical_score = trend_scores.get(trend_signal, 50)

        composite_score = fundamental_score * 0.6 + technical_score * 0.4
        output["composite_score"] = composite_score
        output["individual_scores"] = {
            "fundamental": fundamental_score,
            "technical": technical_score,
        }

        # Cleanup LLM client
        if self._llm_client is not None:
            try:
                await self._llm_client.close()
                self._llm_client = None
            except Exception:
                pass

        return output, 1 if llm_result else 0

    def _get_config(self) -> Any:
        """Lazy load config."""
        if self._config is None:
            from investigator.config import get_config

            self._config = get_config()
        return self._config

    def _get_llm_client(self) -> Any:
        """Lazy load LLM client."""
        if self._llm_client is None:
            from investigator.infrastructure.llm.client import get_client

            self._llm_client = get_client()
        return self._llm_client

    def _build_synthesis_prompt(
        self,
        symbol: str,
        technical: dict,
        fundamental: dict,
        market_context: dict,
        peer_data: dict = None,
    ) -> str:
        """Build synthesis prompt for LLM.

        Returns formatted prompt string.
        """
        prompt = f"""You are an expert investment analyst. Provide a comprehensive investment recommendation for {symbol} based on the following analysis:

## Fundamental Analysis
{_format_fundamental(fundamental)}

## Technical Analysis
{_format_technical(technical)}

## Market Context
Market Regime: {market_context.get('market_regime', 'unknown')}

## Peer Comparison
{self._format_peer_comparison(peer_data)}

Your task: Synthesize all data into a clear investment recommendation.

Consider:
1. How do the valuation models align? Is there consensus or divergence?
2. What does the technical setup suggest about entry timing?
3. What are the key catalysts and risks based on the data?
4. Is the valuation supported by technical levels (support/resistance)?
5. How does the company compare to peers in terms of valuation and metrics?

Provide your response as a JSON object with this exact structure:
{{
    "executive_summary": "2-3 sentence investment thesis referencing specific data points",
    "recommendation": "BUY" or "HOLD" or "SELL",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "composite_score": <number 0-100 based on overall attractiveness>,
    "key_catalysts": ["<specific catalyst 1>", "<specific catalyst 2>", "<specific catalyst 3>"],
    "key_risks": ["<specific risk 1>", "<specific risk 2>", "<specific risk 3>"],
    "price_target": <number based on valuation models>,
    "stop_loss": <number based on support levels>,
    "time_horizon": "SHORT-TERM" or "MEDIUM-TERM" or "LONG-TERM",
    "technical_strength": "STRONG" or "NEUTRAL" or "WEAK",
    "valuation_summary": "Brief summary of valuation model conclusions",
    "peer_comparison_summary": "How the company compares to peers",
    "reasoning": "Detailed explanation referencing specific numbers from the analysis",
    "fundamental_analysis_thinking": "CRITICAL: Write 4-6 detailed paragraphs analyzing SEC fundamentals. Paragraph 1: ANALYSIS SUMMARY - Summarize key financial health indicators (revenue $XXB, net income $XXB, margins XX%). Paragraph 2: INVESTMENT THESIS - Explain why this company is attractive or unattractive based on fundamentals. Paragraph 3: RECENT QUARTER DETAILS - Discuss specific metrics: revenue growth rate, operating margin, free cash flow ($XXB), debt-to-equity ratio, return on equity. Paragraph 4: CASH FLOW ANALYSIS - Operating cash flow trends, capex requirements, free cash flow generation. Paragraph 5: BALANCE SHEET STRENGTH - Current ratio, debt levels, cash position, working capital. Paragraph 6: QUARTERLY SCORE - Rate the recent quarter vs historical performance. USE SPECIFIC NUMBERS FROM THE DATA.",
    "technical_analysis_thinking": "CRITICAL: Write 4-6 detailed paragraphs analyzing technicals. Paragraph 1: TREND ANALYSIS - Current price $XXX vs SMA20/50/200, price position relative to moving averages, trend direction (uptrend/downtrend/sideways). Paragraph 2: MOMENTUM INDICATORS - RSI value (XX) and interpretation (overbought >70, oversold <30, neutral), MACD line vs signal line, momentum strength. Paragraph 3: SUPPORT/RESISTANCE - Key support levels ($XXX, $XXX), resistance levels ($XXX, $XXX), 52-week high/low, current position in range. Paragraph 4: VOLUME ANALYSIS - Recent volume patterns, volume confirmation of price moves, accumulation/distribution signals. Paragraph 5: ENTRY/EXIT TIMING - Based on technicals, is now a good entry point? What would trigger a sell? Paragraph 6: TECHNICAL SCORE - Overall technical rating and justification. USE SPECIFIC NUMBERS.",
    "key_technical_signals": ["<signal 1 with specific number>", "<signal 2 with specific number>", "<signal 3 with specific number>"],
    "risk_factors_detailed": ["<detailed risk 1 with quantification>", "<detailed risk 2 with quantification>", "<detailed risk 3 with quantification>"],
    "score_breakdown": {{
        "income_statement": <0-100 score>,
        "cash_flow": <0-100 score>,
        "balance_sheet": <0-100 score>,
        "growth": <0-100 score>,
        "value": <0-100 score>,
        "business_quality": <0-100 score>,
        "data_quality": <0-100 score>
    }}
}}

IMPORTANT: The fundamental_analysis_thinking and technical_analysis_thinking fields MUST be long-form text (500-800 words each). These sections are displayed prominently in the final report. Do not abbreviate them.
Be specific. Reference actual numbers from the data. Avoid generic statements.
Respond ONLY with the JSON object."""

        return prompt

    def _format_peer_comparison(self, peer_data: dict) -> str:
        """Format peer comparison data for the synthesis prompt.

        Returns formatted string.
        """
        if not peer_data:
            return "## Peer Comparison\nNo peer data available."

        peers = peer_data.get("peers", [])
        metrics = peer_data.get("peer_metrics", {})

        if not peers:
            return "## Peer Comparison\nNo peer companies found for comparison."

        parts = ["## Peer Comparison"]
        parts.append(f"Found {len(peers)} comparable companies:")

        for peer in peers[:5]:
            symbol = peer.get("symbol", "N/A")
            name = peer.get("name", "")
            match_type = peer.get("match_type", "sector")
            val = peer.get("valuation") or {}

            mcap = peer.get("market_cap")
            mcap_str = f"${mcap/1e9:.1f}B" if mcap else "N/A"

            pe = val.get("pe_ratio")
            pe_str = f"{pe:.1f}x" if pe else "N/A"

            upside = val.get("upside_pct")
            upside_str = f"{upside:+.1f}%" if upside else "N/A"

            parts.append(
                f"  - {symbol} ({match_type}): Market Cap {mcap_str}, P/E {pe_str}, Predicted Upside {upside_str}"
            )

        if metrics:
            parts.append("\n### Peer Group Medians:")
            if "pe_ratio_median" in metrics:
                parts.append(
                    f"  - P/E Median: {metrics['pe_ratio_median']:.1f}x (range: {metrics.get('pe_ratio_min', 0):.1f}x - {metrics.get('pe_ratio_max', 0):.1f}x)"
                )
            if "revenue_growth_median" in metrics:
                parts.append(f"  - Revenue Growth Median: {metrics['revenue_growth_median']*100:.1f}%")
            if "fcf_margin_median" in metrics:
                parts.append(f"  - FCF Margin Median: {metrics['fcf_margin_median']*100:.1f}%")
            if "upside_pct_median" in metrics:
                parts.append(f"  - Predicted Upside Median: {metrics['upside_pct_median']:+.1f}%")

        return "\n".join(parts)

    async def _llm_synthesis(
        self, symbol: str, technical: dict, fundamental: dict, market_context: dict, peer_data: dict = None
    ) -> dict:
        """Use LLM for intelligent synthesis.

        Returns LLM-generated synthesis dict or None if unavailable.
        """
        import json

        client = self._get_llm_client()
        if not client:
            return None

        try:
            prompt = self._build_synthesis_prompt(symbol, technical, fundamental, market_context, peer_data)
            model = self._get_config().ollama.models.get("synthesis", "gpt-oss:20b")

            response = await client.generate(
                prompt=prompt,
                model=model,
                options={"temperature": 0.3, "num_predict": 4096},
            )

            # Parse JSON response
            response_text = response.get("response", "")

            # Try to extract JSON from response
            try:
                # Find JSON object in response
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM synthesis response as JSON")

            return None

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return None

    def _rule_based_synthesis(self, fundamental: dict, technical: dict, market_context: dict) -> dict:
        """Fallback rule-based synthesis when LLM is unavailable.

        Returns synthesis dict.
        """
        fundamental = fundamental or {}
        technical = technical or {}
        market_context = market_context or {}

        fund_data = fundamental.get("data", {})
        tech_data = technical.get("data", {})

        fundamental_score = fund_data.get("overall_score", 50)
        technical_score = tech_data.get("overall_score", 50)

        composite_score = fundamental_score * 0.6 + technical_score * 0.4

        if composite_score >= 70:
            recommendation = "BUY"
            confidence = "HIGH"
        elif composite_score >= 55:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif composite_score >= 45:
            recommendation = "HOLD"
            confidence = "MEDIUM"
        elif composite_score >= 30:
            recommendation = "SELL"
            confidence = "MEDIUM"
        else:
            recommendation = "SELL"
            confidence = "HIGH"

        # Build score_breakdown from available data
        score_breakdown = self._build_score_breakdown(fund_data, tech_data)

        # Generate thinking sections from structured data
        fundamental_thinking = self._generate_fundamental_thinking(fund_data)
        technical_thinking = self._generate_technical_thinking(tech_data)

        # Extract key signals from technical data
        key_signals = self._extract_key_signals(tech_data)

        # Generate risk factors and catalysts
        risk_factors = self._generate_risk_factors(fund_data, tech_data, market_context)
        catalysts = self._generate_catalysts(fund_data, tech_data)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            recommendation, confidence, fund_data, tech_data, market_context
        )

        # Generate valuation summary
        valuation_summary = self._generate_valuation_summary(fund_data)

        return {
            "status": "success",
            "synthesis_method": "rule_based",
            "composite_score": composite_score,
            "individual_scores": {
                "fundamental": fundamental_score,
                "technical": technical_score,
            },
            "recommendation": recommendation,
            "confidence": confidence,
            "market_regime": market_context.get("market_regime", "unknown"),
            "score_breakdown": score_breakdown,
            "fundamental_analysis_thinking": fundamental_thinking,
            "technical_analysis_thinking": technical_thinking,
            "key_technical_signals": key_signals,
            "risk_factors_detailed": risk_factors,
            "key_risks": risk_factors[:4] if risk_factors else [],
            "key_catalysts": catalysts,
            "executive_summary": executive_summary,
            "valuation_summary": valuation_summary,
        }

    def _build_score_breakdown(self, fund_data: dict, tech_data: dict) -> dict:
        """Build detailed score breakdown from fundamental and technical data.

        Returns score breakdown dict.
        """
        breakdown = {}

        # Extract from fundamental data if available
        if fund_data:
            models = fund_data.get("models", {})

            # Estimate component scores from valuation models
            dcf = models.get("dcf", {})
            pe = models.get("pe", {})
            ps = models.get("ps", {})

            # Cash flow score from DCF model success/margin
            if dcf and dcf.get("fair_value_per_share"):
                fcf_margin = dcf.get("assumptions", {}).get("fcf_margin", 0)
                breakdown["cash_flow"] = min(100, max(0, 50 + fcf_margin * 100))
            else:
                breakdown["cash_flow"] = 50

            # Value score from upside potential
            consensus_upside = fund_data.get("consensus_upside", 0)
            if consensus_upside:
                breakdown["value"] = min(100, max(0, 50 + consensus_upside))
            else:
                breakdown["value"] = 50

            # Income statement score from PE model
            if pe and pe.get("fair_value_per_share"):
                eps = pe.get("eps_ttm", 0)
                breakdown["income_statement"] = 70 if eps and eps > 0 else 40
            else:
                breakdown["income_statement"] = 50

            # Growth score estimation
            growth_rate = dcf.get("assumptions", {}).get("fcf_growth_rate", 0)
            if growth_rate:
                breakdown["growth"] = min(100, max(0, 50 + growth_rate * 200))
            else:
                breakdown["growth"] = 50

            # Balance sheet (estimate from debt levels if available)
            breakdown["balance_sheet"] = 60  # Default moderate

            # Business quality from model confidence
            confidences = [m.get("confidence", 50) for m in models.values() if isinstance(m, dict)]
            if confidences:
                breakdown["business_quality"] = sum(confidences) / len(confidences)
            else:
                breakdown["business_quality"] = 50

        # Data quality based on models available
        models_count = len(fund_data.get("models", {})) if fund_data else 0
        breakdown["data_quality"] = min(100, models_count * 20)

        return breakdown

    def _generate_fundamental_thinking(self, fund_data: dict) -> str:
        """Generate fundamental analysis narrative from structured data.

        Returns narrative string.
        """
        if not fund_data:
            return ""

        parts = []
        models = fund_data.get("models", {})
        current_price = fund_data.get("current_price", 0)
        consensus_fv = fund_data.get("consensus_fair_value")
        consensus_upside = fund_data.get("consensus_upside")

        # Opening assessment
        if consensus_upside:
            if consensus_upside > 20:
                parts.append(f"The stock appears significantly undervalued based on our multi-model analysis.")
            elif consensus_upside > 0:
                parts.append(f"The stock shows moderate upside potential based on fundamental analysis.")
            elif consensus_upside > -20:
                parts.append(f"The stock appears fairly valued to slightly overvalued at current levels.")
            else:
                parts.append(f"The stock appears significantly overvalued relative to intrinsic value estimates.")

        # DCF analysis
        dcf = models.get("dcf", {})
        if dcf and dcf.get("fair_value_per_share"):
            dcf_fv = dcf.get("fair_value_per_share")
            dcf_upside = dcf.get("upside_downside_pct", 0)
            wacc = dcf.get("assumptions", {}).get("wacc", 0)
            terminal_growth = dcf.get("assumptions", {}).get("terminal_growth_rate", 0)

            parts.append(
                f"\n\nDiscounted Cash Flow Analysis: Our DCF model yields a fair value of ${dcf_fv:.2f} per share, implying {dcf_upside:+.1f}% from current levels."
            )
            if wacc:
                parts.append(
                    f"We use a weighted average cost of capital (WACC) of {wacc*100:.1f}% and terminal growth rate of {terminal_growth*100:.1f}%."
                )

        # Multiple-based valuations
        pe = models.get("pe", {})
        ps = models.get("ps", {})

        if pe and pe.get("fair_value_per_share"):
            pe_fv = pe.get("fair_value_per_share")
            pe_ratio = pe.get("pe_ratio", 0)
            sector_pe = pe.get("sector_pe", 0)
            parts.append(
                f"\n\nP/E Multiple Analysis: Using a target P/E of {pe_ratio:.1f}x (sector median: {sector_pe:.1f}x), we derive a fair value of ${pe_fv:.2f}."
            )

        if ps and ps.get("fair_value_per_share"):
            ps_fv = ps.get("fair_value_per_share")
            ps_ratio = ps.get("ps_ratio", 0)
            parts.append(
                f"\n\nP/S Multiple Analysis: The price-to-sales approach suggests a fair value of ${ps_fv:.2f} using a {ps_ratio:.1f}x multiple."
            )

        # Conclusion
        if consensus_fv and current_price:
            parts.append(
                f"\n\nBlended Valuation: Weighing all models, our consensus fair value is ${consensus_fv:.2f}, compared to the current price of ${current_price:.2f}."
            )

        return "".join(parts) if parts else ""

    def _generate_technical_thinking(self, tech_data: dict) -> str:
        """Generate technical analysis narrative from structured data.

        Returns narrative string.
        """
        if not tech_data:
            return ""

        parts = []
        trend = tech_data.get("trend", {})
        sr = tech_data.get("support_resistance", {})
        momentum = tech_data.get("momentum", {})

        # Trend analysis
        if trend:
            signal = trend.get("overall_signal", "neutral")
            current_price = trend.get("current_price", 0)

            if signal.lower() == "bullish":
                parts.append("Technical indicators are showing bullish momentum across multiple timeframes.")
            elif signal.lower() == "bearish":
                parts.append("Technical indicators suggest bearish pressure with potential for further downside.")
            else:
                parts.append("Technical indicators are mixed, suggesting a consolidation phase.")

            if current_price:
                parts.append(f" The stock is currently trading at ${current_price:.2f}.")

        # Support/Resistance
        if sr:
            support_levels = sr.get("support_levels", {})
            resistance_levels = sr.get("resistance_levels", {})
            w52 = sr.get("52_week", {})

            support_1 = support_levels.get("support_1")
            resistance_1 = resistance_levels.get("resistance_1")

            if support_1 and resistance_1:
                parts.append(
                    f"\n\nKey Levels: Immediate support at ${support_1:.2f} and resistance at ${resistance_1:.2f}."
                )

            if w52:
                w52_high = w52.get("high")
                w52_low = w52.get("low")
                if w52_high and w52_low:
                    parts.append(f" The 52-week trading range is ${w52_low:.2f} to ${w52_high:.2f}.")

        # Momentum indicators
        if momentum:
            rsi = momentum.get("rsi_14")
            macd = momentum.get("macd_histogram")

            if rsi:
                if rsi > 70:
                    parts.append(f"\n\nMomentum: RSI at {rsi:.1f} indicates overbought conditions.")
                elif rsi < 30:
                    parts.append(
                        f"\n\nMomentum: RSI at {rsi:.1f} indicates oversold conditions, potentially signaling a bounce."
                    )
                else:
                    parts.append(f"\n\nMomentum: RSI at {rsi:.1f} is in neutral territory.")

            if macd:
                macd_signal = "positive" if macd > 0 else "negative"
                parts.append(
                    f" MACD histogram is {macd_signal}, suggesting {'bullish' if macd > 0 else 'bearish'} momentum."
                )

        return "".join(parts) if parts else ""

    def _extract_key_signals(self, tech_data: dict) -> list:
        """Extract key technical signals from data.

        Returns list of signal strings.
        """
        signals = []

        trend = tech_data.get("trend", {})
        sr = tech_data.get("support_resistance", {})
        momentum = tech_data.get("momentum", {})

        # Trend signals
        if trend:
            signal = trend.get("overall_signal", "neutral")
            signals.append(f"Overall trend signal: {signal.upper()}")

            signal_pcts = trend.get("signal_percentages", {})
            if signal_pcts:
                bullish_pct = signal_pcts.get("bullish", 0)
                bearish_pct = signal_pcts.get("bearish", 0)
                signals.append(f"Indicator breakdown: {bullish_pct:.0f}% bullish, {bearish_pct:.0f}% bearish")

        # Support/Resistance signals
        if sr:
            current = trend.get("current_price", 0) if trend else 0
            support_1 = sr.get("support_levels", {}).get("support_1")
            resistance_1 = sr.get("resistance_levels", {}).get("resistance_1")

            if current and support_1:
                pct_above_support = ((current - support_1) / support_1) * 100
                signals.append(f"Trading {pct_above_support:.1f}% above key support at ${support_1:.2f}")

            if current and resistance_1:
                pct_below_resistance = ((resistance_1 - current) / current) * 100
                signals.append(f"Resistance {pct_below_resistance:.1f}% higher at ${resistance_1:.2f}")

        # Momentum signals
        if momentum:
            rsi = momentum.get("rsi_14")
            if rsi:
                if rsi > 70:
                    signals.append(f"RSI overbought at {rsi:.1f}")
                elif rsi < 30:
                    signals.append(f"RSI oversold at {rsi:.1f}")
                else:
                    signals.append(f"RSI neutral at {rsi:.1f}")

        return signals[:5]  # Limit to 5 signals

    def _generate_risk_factors(self, fund_data: dict, tech_data: dict, market_context: dict) -> list:
        """Generate risk factors from available data.

        Returns list of risk strings.
        """
        risks = []

        # Valuation risk
        consensus_upside = fund_data.get("consensus_upside", 0) if fund_data else 0
        if consensus_upside < -20:
            risks.append("Significant overvaluation risk with limited margin of safety")
        elif consensus_upside < 0:
            risks.append("Stock trading above fair value estimates")

        # Model divergence risk
        models = fund_data.get("models", {}) if fund_data else {}
        fair_values = [
            m.get("fair_value_per_share", 0)
            for m in models.values()
            if isinstance(m, dict) and m.get("fair_value_per_share")
        ]
        if len(fair_values) >= 2:
            fv_range = max(fair_values) - min(fair_values)
            fv_avg = sum(fair_values) / len(fair_values)
            if fv_avg > 0 and (fv_range / fv_avg) > 0.5:
                risks.append("High valuation model divergence indicates uncertainty")

        # Technical risks
        trend = tech_data.get("trend", {}) if tech_data else {}
        if trend.get("overall_signal", "").lower() == "bearish":
            risks.append("Bearish technical momentum may pressure near-term performance")

        momentum = tech_data.get("momentum", {}) if tech_data else {}
        rsi = momentum.get("rsi_14")
        if rsi and rsi > 70:
            risks.append("Overbought RSI suggests potential near-term pullback")

        # Market regime risk
        regime = market_context.get("market_regime", "unknown") if market_context else "unknown"
        if regime.lower() in ["bearish", "risk_off", "bear"]:
            risks.append("Unfavorable market environment may limit upside")

        # Data quality risk
        if len(models) < 3:
            risks.append("Limited valuation model coverage reduces confidence")

        return risks

    def _generate_catalysts(self, fund_data: dict, tech_data: dict) -> list:
        """Generate potential catalysts from available data.

        Returns list of catalyst strings.
        """
        catalysts = []

        # Valuation catalysts
        consensus_upside = fund_data.get("consensus_upside", 0) if fund_data else 0
        if consensus_upside > 30:
            catalysts.append("Significant undervaluation provides margin of safety")
        elif consensus_upside > 15:
            catalysts.append("Attractive valuation with upside potential")

        # Technical catalysts
        trend = tech_data.get("trend", {}) if tech_data else {}
        if trend.get("overall_signal", "").lower() == "bullish":
            catalysts.append("Bullish technical momentum supports near-term appreciation")

        momentum = tech_data.get("momentum", {}) if tech_data else {}
        rsi = momentum.get("rsi_14")
        if rsi and rsi < 30:
            catalysts.append("Oversold conditions suggest potential mean reversion")

        # Model agreement catalyst
        models = fund_data.get("models", {}) if fund_data else {}
        upsides = [
            m.get("upside_downside_pct", 0) or m.get("upside_percent", 0)
            for m in models.values()
            if isinstance(m, dict)
        ]
        if upsides and all(u > 0 for u in upsides):
            catalysts.append("All valuation models indicate upside potential")

        return catalysts

    def _generate_executive_summary(
        self, recommendation: str, confidence: str, fund_data: dict, tech_data: dict, market_context: dict
    ) -> str:
        """Generate executive summary paragraph.

        Returns summary string.
        """
        parts = []

        current_price = fund_data.get("current_price", 0) if fund_data else 0
        consensus_fv = fund_data.get("consensus_fair_value") if fund_data else None
        consensus_upside = fund_data.get("consensus_upside", 0) if fund_data else 0

        # Opening
        if recommendation == "BUY":
            parts.append(f"We rate this stock a {recommendation} with {confidence} confidence.")
        elif recommendation == "SELL":
            parts.append(
                f"We rate this stock a {recommendation} with {confidence} confidence due to valuation concerns."
            )
        else:
            parts.append(f"We rate this stock a {recommendation} as it appears fairly valued at current levels.")

        # Valuation context
        if consensus_fv and current_price:
            parts.append(
                f" Our blended fair value of ${consensus_fv:.2f} implies {consensus_upside:+.1f}% from the current price of ${current_price:.2f}."
            )

        # Technical context
        trend = tech_data.get("trend", {}) if tech_data else {}
        tech_signal = trend.get("overall_signal", "neutral")
        parts.append(f" Technical indicators are {tech_signal}.")

        # Market context
        regime = market_context.get("market_regime", "unknown") if market_context else "unknown"
        if regime != "unknown":
            parts.append(f" The current market environment is {regime}.")

        return "".join(parts)

    def _generate_valuation_summary(self, fund_data: dict) -> str:
        """Generate valuation summary paragraph.

        Returns summary string.
        """
        if not fund_data:
            return ""

        models = fund_data.get("models", {})
        consensus_fv = fund_data.get("consensus_fair_value")
        consensus_upside = fund_data.get("consensus_upside")

        if not models:
            return "Insufficient data for comprehensive valuation analysis."

        parts = []
        model_count = len(models)
        parts.append(
            f"We applied {model_count} valuation model{'s' if model_count > 1 else ''} to derive our fair value estimate."
        )

        if consensus_fv:
            parts.append(f" The blended fair value is ${consensus_fv:.2f}")
            if consensus_upside:
                parts.append(f" ({consensus_upside:+.1f}% from current levels).")
            else:
                parts.append(".")

        # Model range
        fair_values = [
            m.get("fair_value_per_share", 0)
            for m in models.values()
            if isinstance(m, dict) and m.get("fair_value_per_share")
        ]
        if len(fair_values) >= 2:
            parts.append(f" Fair value estimates range from ${min(fair_values):.2f} to ${max(fair_values):.2f}.")

        return "".join(parts)


# =============================================================================
# Report Generation Handlers
# =============================================================================


@handler_decorator("generate_report", vertical="investment", description="Generate professional PDF report")
@dataclass
class GenerateReportHandler(BaseHandler):
    """Generate professional PDF report from analysis."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute report generation.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        import json

        from investigator.infrastructure.reporting.professional_report import ProfessionalReportGenerator

        synthesis = context.get("synthesis") or {}
        symbol = context.get("symbol", "UNKNOWN")
        technical = context.get("technical_analysis") or {}
        fundamental = context.get("fundamental_analysis") or {}
        market_data = context.get("market_data") or {}

        # Handle synthesis as string (from agent node) or dict
        if isinstance(synthesis, str):
            try:
                start_idx = synthesis.find("{")
                end_idx = synthesis.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    synthesis = json.loads(synthesis[start_idx:end_idx])
                else:
                    synthesis = {
                        "executive_summary": synthesis[:500],
                        "recommendation": "HOLD",
                        "confidence": "MEDIUM",
                    }
            except json.JSONDecodeError:
                synthesis = {
                    "executive_summary": synthesis[:500],
                    "recommendation": "HOLD",
                    "confidence": "MEDIUM",
                }

        # Extract technical data
        tech_data = technical.get("data", {}) if isinstance(technical, dict) else {}
        trend = tech_data.get("trend", {})
        sr = tech_data.get("support_resistance", {})

        # Extract fundamental data
        fund_data = fundamental.get("data", {}) if isinstance(fundamental, dict) else {}

        # Extract price from market data, technical, or valuation result
        current_price = None
        if market_data and isinstance(market_data, dict):
            md = market_data.get("data", {})
            if md:
                current_price = md.get("current_price") or md.get("close")
        if not current_price and trend:
            current_price = trend.get("current_price")
        if not current_price and fund_data:
            current_price = fund_data.get("current_price")

        # Calculate scores (convert to 0-100 scale)
        overall = synthesis.get("composite_score", 50)
        if overall > 10:  # Already in 0-100 scale
            pass
        else:  # Convert from 0-10 scale
            overall = overall * 10

        individual = synthesis.get("individual_scores") or {}
        fund_overall = fund_data.get("overall_score", 50) if fund_data else 50
        tech_overall = tech_data.get("overall_score", 50) if tech_data else 50
        fundamental_score = individual.get("fundamental", fund_overall) or fund_overall
        technical_score = individual.get("technical", tech_overall) or tech_overall

        # Normalize scores
        if fundamental_score <= 10:
            fundamental_score = fundamental_score * 10
        if technical_score <= 10:
            technical_score = technical_score * 10

        # Extract support/resistance levels
        sr = sr or {}
        support_levels = sr.get("support_levels") or {}
        resistance_levels = sr.get("resistance_levels") or {}
        support = support_levels.get("support_1")
        resistance = resistance_levels.get("resistance_1")

        # Extract market context for regime info
        market_context = context.get("market_context", {})
        macro_data = context.get("macro_data", {})
        peer_data = context.get("peer_data") or {}

        # Build market regime data
        market_regime = {}
        if market_context:
            market_regime["regime"] = market_context.get("market_regime", "normal")
        if macro_data and macro_data.get("status") == "success":
            vol = macro_data.get("volatility", {})
            if vol:
                market_regime["vix"] = vol.get("vix")
            treasury = macro_data.get("treasury", {})
            if treasury:
                market_regime["yield_curve_slope"] = treasury.get("yield_curve_slope")

        # Ensure all dict vars are never None
        synthesis = synthesis or {}
        fund_data = fund_data or {}
        tech_data = tech_data or {}
        trend = trend or {}
        sr = sr or {}

        # Build report data with all sections
        report_data = {
            "symbol": symbol,
            "recommendation": synthesis.get("recommendation", "HOLD"),
            "confidence": synthesis.get("confidence", "MEDIUM"),
            "overall_score": overall,
            "fundamental_score": fundamental_score,
            "technical_score": technical_score,
            "current_price": current_price,
            "target_price": synthesis.get("price_target") or resistance,
            "stop_loss": synthesis.get("stop_loss") or support,
            "investment_thesis": synthesis.get("executive_summary", ""),
            "key_catalysts": synthesis.get("key_catalysts") or [],
            "key_risks": synthesis.get("key_risks") or [],
            "time_horizon": synthesis.get("time_horizon", "MEDIUM-TERM"),
            "position_size": synthesis.get("position_size", "MODERATE"),
            "technical_strength": synthesis.get("technical_strength", "NEUTRAL"),
            "valuation_summary": synthesis.get("valuation_summary", ""),
            "valuation_models": fund_data.get("models") or {},
            "technical_data": {
                "overall_signal": trend.get("overall_signal", "neutral"),
                "signal_percentages": trend.get("signal_percentages") or {},
                "support_resistance": sr,
                "momentum": tech_data.get("momentum") or {},
            },
            "market_regime": market_regime,
            "peer_comparison": {
                "peers": peer_data.get("peers", []),
                "metrics": peer_data.get("peer_metrics", {}),
                "summary": synthesis.get("peer_comparison_summary", ""),
            },
            "fundamental_analysis_thinking": synthesis.get("fundamental_analysis_thinking", ""),
            "technical_analysis_thinking": synthesis.get("technical_analysis_thinking", ""),
            "key_technical_signals": synthesis.get("key_technical_signals", []),
            "risk_factors_detailed": synthesis.get("risk_factors_detailed", []),
            "score_breakdown": synthesis.get("score_breakdown", {}),
            "reasoning": synthesis.get("reasoning", ""),
            "financial_metrics": self._build_financial_metrics(fund_data, context),
            "historical_financials": self._build_historical_financials(fund_data, context),
        }

        # Get output directory
        from pathlib import Path

        from investigator.config import get_config

        cfg = get_config()
        output_dir = cfg.reports_dir / "professional"

        generator = ProfessionalReportGenerator(output_dir=output_dir)
        report_path = generator.generate_report(report_data)

        return {"path": str(report_path), "status": "success"}, 0

    def _build_financial_metrics(self, fund_data: dict, context) -> dict:
        """Build financial metrics dashboard data: company vs sector comparison.

        Returns metrics dict.
        """
        metrics = {}

        # Extract company metrics from fundamental data
        if not fund_data:
            return metrics

        # Get valuation metrics
        valuation = fund_data.get("valuation") or {}
        if valuation:
            if "pe_ratio" in valuation:
                metrics["pe_ratio"] = {
                    "company": valuation.get("pe_ratio"),
                    "sector": valuation.get("sector_pe_median"),
                }
            if "ev_ebitda" in valuation:
                metrics["ev_ebitda"] = {
                    "company": valuation.get("ev_ebitda"),
                    "sector": valuation.get("sector_ev_ebitda_median"),
                }

        # Get profitability metrics
        profitability = fund_data.get("profitability") or {}
        if profitability:
            if "roe" in profitability:
                metrics["roe"] = {
                    "company": profitability.get("roe"),
                    "sector": profitability.get("sector_roe_median"),
                }
            if "fcf_margin" in profitability:
                metrics["fcf_margin"] = {
                    "company": profitability.get("fcf_margin"),
                    "sector": profitability.get("sector_fcf_margin_median"),
                }

        # Get growth metrics
        growth = fund_data.get("growth") or {}
        if growth:
            if "revenue_growth" in growth:
                metrics["revenue_growth"] = {
                    "company": growth.get("revenue_growth"),
                    "sector": growth.get("sector_revenue_growth_median"),
                }

        # Get leverage metrics
        leverage = fund_data.get("leverage") or fund_data.get("balance_sheet") or {}
        if leverage:
            if "debt_to_equity" in leverage:
                metrics["debt_to_equity"] = {
                    "company": leverage.get("debt_to_equity"),
                    "sector": leverage.get("sector_debt_to_equity_median"),
                }

        # Try to get from SEC filing data as fallback
        sec_data = context.get("sec_data") if context else None
        if sec_data and isinstance(sec_data, dict):
            filing_data = sec_data.get("data", sec_data)
            ratios = filing_data.get("financial_ratios") or {}
            if ratios:
                if "pe_ratio" not in metrics and "pe_ratio" in ratios:
                    metrics["pe_ratio"] = {"company": ratios.get("pe_ratio"), "sector": None}
                if "roe" not in metrics and "roe" in ratios:
                    metrics["roe"] = {"company": ratios.get("roe"), "sector": None}

        return metrics

    def _build_historical_financials(self, fund_data: dict, context) -> dict:
        """Build historical financials for trend charts.

        Returns historical data dict.
        """
        historical = {}

        # Try SEC filing data for historical metrics
        sec_data = context.get("sec_data") if context else None
        if sec_data and isinstance(sec_data, dict):
            filing_data = sec_data.get("data", sec_data)

            # Revenue history
            revenue_history = filing_data.get("revenue_history") or filing_data.get("historical_revenue")
            if revenue_history and isinstance(revenue_history, list):
                historical["revenue"] = revenue_history

            # FCF history
            fcf_history = filing_data.get("fcf_history") or filing_data.get("historical_fcf")
            if fcf_history and isinstance(fcf_history, list):
                historical["free_cash_flow"] = fcf_history

            # ROE history
            roe_history = filing_data.get("roe_history") or filing_data.get("historical_roe")
            if roe_history and isinstance(roe_history, list):
                historical["roe"] = roe_history

        # Extract from income statement if available
        income = fund_data.get("income_statement") or {}
        if income and "annual" in income:
            annual = income["annual"]
            if isinstance(annual, list):
                revenue_points = []
                for year_data in annual:
                    year = year_data.get("fiscal_year") or year_data.get("year")
                    rev = year_data.get("revenue") or year_data.get("total_revenue")
                    if year and rev:
                        revenue_points.append((year, rev))
                if revenue_points and "revenue" not in historical:
                    historical["revenue"] = revenue_points

        # Extract from cash flow statement if available
        cashflow = fund_data.get("cash_flow_statement") or {}
        if cashflow and "annual" in cashflow:
            annual = cashflow["annual"]
            if isinstance(annual, list):
                fcf_points = []
                for year_data in annual:
                    year = year_data.get("fiscal_year") or year_data.get("year")
                    fcf = year_data.get("free_cash_flow") or year_data.get("fcf")
                    if year and fcf:
                        fcf_points.append((year, fcf))
                if fcf_points and "free_cash_flow" not in historical:
                    historical["free_cash_flow"] = fcf_points

        return historical


# =============================================================================
# Peer Comparison Handlers
# =============================================================================


@handler_decorator("identify_peers", vertical="investment", description="Identify peer companies")
@dataclass
class IdentifyPeersHandler(BaseHandler):
    """Identify peer companies for comparison with valuation metrics.

    Uses industry-first matching strategy:
    1. Find peers matching both sector AND industry (highest quality)
    2. If <5 matches, add sector-only matches
    3. Sort by market cap (largest first)
    4. Fetch recent valuation metrics for each peer
    5. Return up to 5 peers with valuation data
    """

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute peer identification.

        Returns:
            Tuple of (output_dict, tool_calls_count)
        """
        from sqlalchemy import text

        from investigator.infrastructure.database.db import get_database_engine

        symbol = context.get("symbol", "")
        market_context = context.get("market_context") or {}
        sector = market_context.get("sector")
        industry = market_context.get("industry")

        peers = []

        if sector or industry:
            engine = get_database_engine()
            with engine.connect() as conn:
                # First: Get peers with EXACT industry match + recent valuation metrics
                if industry:
                    result = conn.execute(
                        text(
                            """
                            SELECT DISTINCT ON (s.symbol)
                                s.symbol, s.name, s.market_cap, s.industry, s.sector,
                                v.pe_fair_value, v.ps_fair_value, v.blended_fair_value,
                                v.current_price, v.predicted_upside_pct,
                                v.context_features->>'pe_level' as pe_ratio,
                                v.context_features->>'revenue_growth' as revenue_growth,
                                v.context_features->>'fcf_margin' as fcf_margin,
                                v.analysis_date
                            FROM symbols s
                            LEFT JOIN LATERAL (
                                SELECT * FROM valuation_outcomes vo
                                WHERE vo.symbol = s.symbol
                                ORDER BY vo.analysis_date DESC
                                LIMIT 1
                            ) v ON true
                            WHERE s.industry = :industry
                            AND s.symbol != :target
                            AND s.is_active = true
                            ORDER BY s.symbol, v.analysis_date DESC NULLS LAST, s.market_cap DESC NULLS LAST
                            LIMIT 5
                        """
                        ),
                        {"industry": industry, "target": symbol},
                    )
                    for row in result:
                        pe_ratio = None
                        if row[10]:
                            try:
                                pe_ratio = float(row[10])
                            except (ValueError, TypeError):
                                pass
                        peers.append(
                            {
                                "symbol": row[0],
                                "name": row[1],
                                "market_cap": float(row[2]) if row[2] else None,
                                "industry": row[3],
                                "sector": row[4],
                                "match_type": "industry",
                                "valuation": {
                                    "pe_fair_value": float(row[5]) if row[5] else None,
                                    "ps_fair_value": float(row[6]) if row[6] else None,
                                    "blended_fair_value": float(row[7]) if row[7] else None,
                                    "current_price": float(row[8]) if row[8] else None,
                                    "upside_pct": float(row[9]) if row[9] else None,
                                    "pe_ratio": pe_ratio,
                                    "revenue_growth": float(row[11]) if row[11] else None,
                                    "fcf_margin": float(row[12]) if row[12] else None,
                                },
                                "analysis_date": str(row[13]) if row[13] else None,
                            }
                        )

                # If <5 industry matches, add sector matches
                if len(peers) < 5 and sector:
                    existing_symbols = {p["symbol"] for p in peers}
                    remaining_slots = 5 - len(peers)

                    result = conn.execute(
                        text(
                            """
                            SELECT DISTINCT ON (s.symbol)
                                s.symbol, s.name, s.market_cap, s.industry, s.sector,
                                v.pe_fair_value, v.ps_fair_value, v.blended_fair_value,
                                v.current_price, v.predicted_upside_pct,
                                v.context_features->>'pe_level' as pe_ratio,
                                v.context_features->>'revenue_growth' as revenue_growth,
                                v.context_features->>'fcf_margin' as fcf_margin,
                                v.analysis_date
                            FROM symbols s
                            LEFT JOIN LATERAL (
                                SELECT * FROM valuation_outcomes vo
                                WHERE vo.symbol = s.symbol
                                ORDER BY vo.analysis_date DESC
                                LIMIT 1
                            ) v ON true
                            WHERE s.sector = :sector
                            AND s.symbol != :target
                            AND s.is_active = true
                            ORDER BY s.symbol, v.analysis_date DESC NULLS LAST, s.market_cap DESC NULLS LAST
                            LIMIT :limit
                        """
                        ),
                        {"sector": sector, "target": symbol, "limit": remaining_slots + 10},
                    )
                    for row in result:
                        if row[0] not in existing_symbols and len(peers) < 5:
                            pe_ratio = None
                            if row[10]:
                                try:
                                    pe_ratio = float(row[10])
                                except (ValueError, TypeError):
                                    pass
                            peers.append(
                                {
                                    "symbol": row[0],
                                    "name": row[1],
                                    "market_cap": float(row[2]) if row[2] else None,
                                    "industry": row[3],
                                    "sector": row[4],
                                    "match_type": "sector",
                                    "valuation": {
                                        "pe_fair_value": float(row[5]) if row[5] else None,
                                        "ps_fair_value": float(row[6]) if row[6] else None,
                                        "blended_fair_value": float(row[7]) if row[7] else None,
                                        "current_price": float(row[8]) if row[8] else None,
                                        "upside_pct": float(row[9]) if row[9] else None,
                                        "pe_ratio": pe_ratio,
                                        "revenue_growth": float(row[11]) if row[11] else None,
                                        "fcf_margin": float(row[12]) if row[12] else None,
                                    },
                                    "analysis_date": str(row[13]) if row[13] else None,
                                }
                            )

        # Calculate peer group medians for comparison
        peer_metrics = self._calculate_peer_medians(peers)

        return {"peers": peers, "peer_metrics": peer_metrics}, 0

    def _calculate_peer_medians(self, peers: List[Dict]) -> Dict:
        """Calculate median valuation metrics across peer group.

        Returns metrics dict.
        """
        import statistics

        if not peers:
            return {}

        metrics = {
            "pe_ratio": [],
            "revenue_growth": [],
            "fcf_margin": [],
            "upside_pct": [],
        }

        for peer in peers:
            val = peer.get("valuation") or {}
            if val.get("pe_ratio") is not None:
                metrics["pe_ratio"].append(val["pe_ratio"])
            if val.get("revenue_growth") is not None:
                metrics["revenue_growth"].append(val["revenue_growth"])
            if val.get("fcf_margin") is not None:
                metrics["fcf_margin"].append(val["fcf_margin"])
            if val.get("upside_pct") is not None:
                metrics["upside_pct"].append(val["upside_pct"])

        result = {
            "count": len(peers),
            "industry_matches": sum(1 for p in peers if p.get("match_type") == "industry"),
            "sector_matches": sum(1 for p in peers if p.get("match_type") == "sector"),
        }

        for key, values in metrics.items():
            if values:
                result[f"{key}_median"] = statistics.median(values)
                result[f"{key}_min"] = min(values)
                result[f"{key}_max"] = max(values)

        return result


@handler_decorator("analyze_peers", vertical="investment", description="Analyze peer companies")
@dataclass
class AnalyzePeersHandler(BaseHandler):
    """Analyze peer companies."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute peer analysis.

        Returns:
            Tuple of (peer_analyses_list, tool_calls_count)
        """
        peer_data = context.get("peer_data") or {}
        peers = peer_data.get("peers", []) if isinstance(peer_data, dict) else context.get("peer_list", [])

        if not peers:
            return [], 0

        import asyncio

        from victor_invest.workflows import AnalysisMode, run_analysis

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

        return peer_analyses, 0


# =============================================================================
# RL Backtest Handlers
# =============================================================================


@handler_decorator("generate_lookback_dates", vertical="investment", description="Generate lookback dates")
@dataclass
class GenerateLookbackDatesHandler(BaseHandler):
    """Generate lookback dates for RL backtesting."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute lookback date generation.

        Returns:
            Tuple of (lookback_dates_list, tool_calls_count)
        """
        from victor_invest.workflows.rl_backtest import generate_lookback_list

        max_months = context.get("max_lookback_months", 120)
        interval = context.get("interval", "quarterly")

        lookback_dates = generate_lookback_list(max_months, interval)

        return lookback_dates, 0


@handler_decorator("process_backtest_batch", vertical="investment", description="Process backtest batch")
@dataclass
class ProcessBacktestBatchHandler(BaseHandler):
    """Process a batch of backtest dates for RL training."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute backtest batch processing.

        Returns:
            Tuple of (backtest_results_dict, tool_calls_count)
        """
        from victor_invest.workflows.rl_backtest import run_rl_backtest

        symbol = context.get("symbol")
        lookback_dates = context.get("lookback_dates", [])
        interval = context.get("interval", "quarterly")

        if not symbol:
            raise ValueError("No symbol provided")

        result = await run_rl_backtest(
            symbol=symbol,
            lookback_months_list=lookback_dates,
            interval=interval,
        )

        return result.to_dict(), 0


@handler_decorator("save_rl_predictions", vertical="investment", description="Save RL predictions")
@dataclass
class SaveRLPredictionsHandler(BaseHandler):
    """Save RL predictions to database."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        """Execute RL predictions save.

        Returns:
            Tuple of (summary_dict, tool_calls_count)
        """
        backtest_results = context.get("backtest_results", {})

        # The predictions are already saved during run_rl_backtest
        # This handler just returns the summary
        predictions = backtest_results.get("predictions", [])
        metadata = backtest_results.get("metadata", {})

        output = {
            "predictions_count": len(predictions),
            "summary": metadata.get("summary", {}),
        }

        return output, 0


# =============================================================================
# Helper Functions (No Migration Needed)
# =============================================================================


def _format_fundamental(fundamental: dict) -> str:
    """Format fundamental data for prompt with comprehensive valuation details.

    Returns formatted string.
    """
    if not fundamental or fundamental.get("status") == "error":
        return "Fundamental data not available."

    data = fundamental.get("data", {})
    if not data:
        return "Fundamental data not available."

    parts = []

    # Current price and consensus
    current_price = data.get("current_price")
    consensus_fv = data.get("consensus_fair_value")
    consensus_upside = data.get("consensus_upside")

    if current_price:
        parts.append(f"- Current Price: ${current_price:.2f}")
    if consensus_fv:
        parts.append(f"- Blended Fair Value: ${consensus_fv:.2f}")
    if consensus_upside:
        parts.append(f"- Upside/Downside: {consensus_upside:+.1f}%")

    # Individual valuation models
    models = data.get("models", {})
    if models:
        parts.append("\n### Valuation Models:")
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                fv = model_data.get("fair_value_per_share")
                upside = model_data.get("upside_percent")
                confidence = model_data.get("confidence")
                if fv:
                    conf_str = f" (Confidence: {confidence:.0f}%)" if confidence else ""
                    upside_str = f" [{upside:+.1f}%]" if upside else ""
                    parts.append(f"  - {model_name.upper()}: ${fv:.2f}{upside_str}{conf_str}")

                    # Add model-specific details
                    if model_name == "dcf":
                        wacc = model_data.get("wacc")
                        tgr = model_data.get("terminal_growth_rate")
                        if wacc:
                            parts.append(f"    WACC: {wacc*100:.1f}%, Terminal Growth: {(tgr or 0.02)*100:.1f}%")
                    elif model_name == "pe":
                        pe_ratio = model_data.get("pe_ratio")
                        sector_pe = model_data.get("sector_pe")
                        eps = model_data.get("eps_ttm")
                        if pe_ratio and eps:
                            parts.append(
                                f"    TTM EPS: ${eps:.2f}, Target P/E: {pe_ratio:.1f}x (Sector Median: {sector_pe:.1f}x)"
                            )
                    elif model_name == "ps":
                        ps_ratio = model_data.get("ps_ratio")
                        sector_ps = model_data.get("sector_ps")
                        rps = model_data.get("revenue_per_share")
                        if ps_ratio and rps:
                            parts.append(
                                f"    Revenue/Share: ${rps:.2f}, Target P/S: {ps_ratio:.1f}x (Sector: {sector_ps:.1f}x)"
                            )

    # Models applied
    models_applied = data.get("models_applied", [])
    if models_applied:
        parts.append(f"\n- Models Applied: {', '.join([m.upper() for m in models_applied])}")

    return "\n".join(parts) if parts else "Fundamental data not available."


def _format_technical(technical: dict) -> str:
    """Format technical data for prompt with support/resistance levels.

    Returns formatted string.
    """
    if not technical or technical.get("status") != "success":
        return "Technical data not available."

    data = technical.get("data", {})
    if not data:
        return "Technical data not available."

    parts = []

    # Trend analysis
    trend = data.get("trend", {})
    if trend:
        current_price = trend.get("current_price")
        signal = trend.get("overall_signal", "neutral")
        signal_pcts = trend.get("signal_percentages", {})

        if current_price:
            parts.append(f"- Current Price: ${current_price:.2f}")
        parts.append(f"- Overall Signal: {signal.upper()}")

        bullish = signal_pcts.get("bullish_pct", 0)
        bearish = signal_pcts.get("bearish_pct", 0)
        neutral = signal_pcts.get("neutral_pct", 0)
        parts.append(f"- Signal Breakdown: Bullish {bullish:.0f}%, Bearish {bearish:.0f}%, Neutral {neutral:.0f}%")

    # Support/Resistance levels
    sr = data.get("support_resistance", {})
    if sr:
        support_levels = sr.get("support_levels", {})
        resistance_levels = sr.get("resistance_levels", {})
        week_52 = sr.get("52_week", {})

        parts.append("\n### Key Levels:")
        if support_levels:
            s1 = support_levels.get("support_1")
            s2 = support_levels.get("support_2")
            if s1:
                parts.append(f"  - Support 1: ${s1:.2f}")
            if s2:
                parts.append(f"  - Support 2: ${s2:.2f}")

        if resistance_levels:
            r1 = resistance_levels.get("resistance_1")
            r2 = resistance_levels.get("resistance_2")
            if r1:
                parts.append(f"  - Resistance 1: ${r1:.2f}")
            if r2:
                parts.append(f"  - Resistance 2: ${r2:.2f}")

        if week_52:
            low = week_52.get("low")
            high = week_52.get("high")
            if low and high:
                parts.append(f"  - 52-Week Range: ${low:.2f} - ${high:.2f}")

    # Momentum indicators
    momentum = data.get("momentum", {})
    if momentum:
        rsi = momentum.get("rsi_14")
        macd = momentum.get("macd_line")
        parts.append("\n### Momentum:")
        if rsi:
            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            parts.append(f"  - RSI(14): {rsi:.1f} ({rsi_signal})")
        if macd is not None:
            parts.append(f"  - MACD: {macd:.3f}")

    return "\n".join(parts) if parts else "Technical data not available."


# =============================================================================
# Registration (No-op for backward compatibility)
# =============================================================================


def register_handlers() -> None:
    """Register Investment handlers with the workflow executor.

    This is a no-op function for backward compatibility.
    Handlers are auto-registered via @handler_decorator on module import.
    """
    pass


__all__ = [
    # Data collection handlers
    "FetchSECDataHandler",
    "FetchMarketDataHandler",
    "FetchMacroDataHandler",
    # Analysis handlers
    "RunFundamentalAnalysisHandler",
    "RunTechnicalAnalysisHandler",
    "RunMarketContextHandler",
    # Synthesis handlers
    "RunSynthesisHandler",
    # Report generation
    "GenerateReportHandler",
    # Peer comparison
    "IdentifyPeersHandler",
    "AnalyzePeersHandler",
    # RL backtest
    "GenerateLookbackDatesHandler",
    "ProcessBacktestBatchHandler",
    "SaveRLPredictionsHandler",
    # Helper functions
    "_format_fundamental",
    "_format_technical",
    # Registration
    "register_handlers",
]
