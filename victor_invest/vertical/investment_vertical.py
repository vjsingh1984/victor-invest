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

"""Investment Vertical for Victor framework.

Provides comprehensive investment analysis capabilities including:
- SEC filings analysis
- Fundamental and valuation analysis
- Technical analysis
- Market context analysis
- Investment thesis synthesis

ARCHITECTURE DECISION: Tool Registration for Dual-Mode Operation
================================================================

Tools are registered by NAME (not instance) to enable both:

1. DIRECT INVOCATION (StateGraph workflow)
   - Workflow nodes instantiate tools directly
   - Used for deterministic data collection
   - No LLM involvement in data fetching

   Example:
       # In workflow node
       from victor_invest.tools import get_tool
       sec_tool = get_tool("sec_filing")
       result = await sec_tool.execute(symbol="AAPL")

2. LLM TOOL CALLING (Agent exploration)
   - Victor Agent loads tools from registry
   - LLM decides when to invoke based on context
   - Used for interactive/exploratory analysis

   Example:
       agent = await InvestmentVertical.create_agent()
       # Agent can invoke tools via LLM reasoning

TOOL LIST RATIONALE:
- sec_filing: SEC EDGAR data (10-K, 10-Q, company facts)
- valuation: Multi-model valuation (DCF, P/E, P/S, P/B, GGM, EV/EBITDA)
- technical_indicators: 80+ technical indicators (RSI, MACD, etc.)
- market_data: Price/volume data, sector context
- cache: Multi-tier caching for performance

DATABASE ACCESS PATTERN:
- Tools access PostgreSQL databases directly (not via LLM)
- Credentials via environment variables (DB_PASSWORD, etc.)
- .env files are gitignored for security

See: docs/ARCHITECTURE_DECISION_DATA_ACCESS.md for full rationale.
"""

from typing import Any, Dict, List, Optional

from victor.core.verticals import StageDefinition, VerticalBase, VerticalConfig

from victor_invest.prompts.investment_prompts import INVESTMENT_SYSTEM_PROMPT


class InvestmentVertical(VerticalBase):
    """Investment research and analysis vertical.

    Implements institutional-grade equity analysis with multi-model
    valuation, technical analysis, and SEC filings integration.

    Example:
        config = InvestmentVertical.get_config()
        agent = await InvestmentVertical.create_agent()
    """

    name = "investment"
    description = "Institutional-grade investment research and equity analysis"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tool names for investment analysis.

        Returns:
            List of tool names to enable.
        """
        return [
            "sec_filing",
            "valuation",
            "technical_indicators",
            "market_data",
            "cache",
            "entry_exit_signals",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for investment analysis.

        Returns:
            System prompt text with investment domain expertise.
        """
        return INVESTMENT_SYSTEM_PROMPT

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get investment-specific stage definitions.

        Returns:
            Dictionary mapping stage names to definitions.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the investment research request",
                keywords=["analyze", "research", "evaluate", "what", "how"],
                next_stages={"DATA_GATHERING", "PLANNING"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning the analysis approach",
                keywords=["plan", "approach", "strategy", "methodology"],
                next_stages={"DATA_GATHERING"},
            ),
            "DATA_GATHERING": StageDefinition(
                name="DATA_GATHERING",
                description="Collecting SEC filings and market data",
                keywords=["sec", "filing", "data", "fetch", "gather"],
                tools={"sec_filing", "market_data", "cache"},
                next_stages={"FUNDAMENTAL_ANALYSIS", "TECHNICAL_ANALYSIS"},
            ),
            "FUNDAMENTAL_ANALYSIS": StageDefinition(
                name="FUNDAMENTAL_ANALYSIS",
                description="Performing valuation and financial analysis",
                keywords=["valuation", "dcf", "pe", "fundamental", "financial"],
                tools={"valuation", "cache"},
                next_stages={"TECHNICAL_ANALYSIS", "SYNTHESIS"},
            ),
            "TECHNICAL_ANALYSIS": StageDefinition(
                name="TECHNICAL_ANALYSIS",
                description="Analyzing price action, technical indicators, and entry/exit signals",
                keywords=["technical", "chart", "indicator", "rsi", "macd", "entry", "exit", "signal"],
                tools={"technical_indicators", "market_data", "entry_exit_signals", "cache"},
                next_stages={"MARKET_CONTEXT", "SYNTHESIS"},
            ),
            "MARKET_CONTEXT": StageDefinition(
                name="MARKET_CONTEXT",
                description="Evaluating market conditions and sector dynamics",
                keywords=["market", "sector", "macro", "context"],
                tools={"market_data", "cache"},
                next_stages={"SYNTHESIS"},
            ),
            "SYNTHESIS": StageDefinition(
                name="SYNTHESIS",
                description="Synthesizing analysis into investment recommendation",
                keywords=["recommend", "conclusion", "thesis", "summary"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Finalizing investment analysis",
                keywords=["done", "finished", "complete", "final"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Get hints for provider selection.

        Investment analysis benefits from strong reasoning capabilities.

        Returns:
            Dictionary with provider preferences.
        """
        return {
            "preferred_providers": ["anthropic", "openai"],
            "min_context_window": 100000,
            "requires_tool_calling": True,
            "preferred_capabilities": ["reasoning", "structured_output"],
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Get criteria for evaluating investment analysis quality.

        Returns:
            List of evaluation criteria descriptions.
        """
        return [
            "Data accuracy and completeness",
            "Multi-model valuation consistency",
            "Technical analysis precision",
            "Market context relevance",
            "Recommendation clarity and confidence",
            "Risk factor identification",
        ]

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Get investment-specific task type hints.

        Returns:
            Dictionary mapping task types to TaskTypeHint-like dicts.
        """
        return {
            "research": {
                "task_type": "research",
                "hint": "[RESEARCH MODE] Gather comprehensive data from SEC filings and market sources.",
                "tool_budget": 30,
                "priority_tools": ["sec_filing", "market_data", "cache"],
            },
            "valuation": {
                "task_type": "valuation",
                "hint": "[VALUATION MODE] Apply multiple valuation models and cross-validate results.",
                "tool_budget": 25,
                "priority_tools": ["valuation", "sec_filing", "cache"],
            },
            "technical": {
                "task_type": "technical",
                "hint": "[TECHNICAL MODE] Analyze price action, trends, and technical indicators.",
                "tool_budget": 20,
                "priority_tools": ["technical_indicators", "market_data"],
            },
            "screening": {
                "task_type": "screening",
                "hint": "[SCREENING MODE] Filter stocks based on quantitative criteria.",
                "tool_budget": 15,
                "priority_tools": ["market_data", "valuation"],
            },
            "synthesis": {
                "task_type": "synthesis",
                "hint": "[SYNTHESIS MODE] Combine analysis streams into actionable recommendations.",
                "tool_budget": 35,
                "priority_tools": ["sec_filing", "valuation", "technical_indicators", "market_data"],
            },
        }

    @classmethod
    def get_mode_config(cls) -> Dict[str, Any]:
        """Get investment-specific operational modes.

        Returns:
            Dictionary mapping mode names to ModeConfig-like dicts.
        """
        return {
            "quick": {
                "name": "quick",
                "tool_budget": 10,
                "max_iterations": 15,
                "temperature": 0.5,
                "description": "Quick overview with key metrics",
            },
            "standard": {
                "name": "standard",
                "tool_budget": 30,
                "max_iterations": 40,
                "temperature": 0.7,
                "description": "Standard analysis with multiple models",
            },
            "deep_dive": {
                "name": "deep_dive",
                "tool_budget": 60,
                "max_iterations": 80,
                "temperature": 0.7,
                "description": "Comprehensive institutional-grade analysis",
            },
            "screening": {
                "name": "screening",
                "tool_budget": 20,
                "max_iterations": 25,
                "temperature": 0.5,
                "description": "Quantitative screening mode",
            },
        }

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for investment analysis.

        Provides access to YAML-defined investment workflows including:
        - quick: Technical analysis only
        - standard: Technical + Fundamental
        - comprehensive: Full institutional-grade analysis
        - rl_backtest: Historical backtesting
        - peer_comparison: Peer group analysis

        Returns:
            InvestmentWorkflowProvider instance (extends BaseYAMLWorkflowProvider).
        """

        def _create():
            from victor_invest.workflows import InvestmentWorkflowProvider

            return InvestmentWorkflowProvider()

        return cls._get_cached_extension("workflow", _create)

    @classmethod
    async def create_orchestrator(
        cls,
        provider: str = "ollama",
        model: Optional[str] = None,
    ) -> Any:
        """Create an AgentOrchestrator for YAML workflow execution.

        This creates a Victor AgentOrchestrator configured with the Investment
        vertical's tools and prompts. The orchestrator can be used with
        WorkflowExecutor for executing YAML workflows with agent nodes.

        Args:
            provider: LLM provider name (ollama, anthropic, openai).
            model: Model identifier. If None, uses config default.

        Returns:
            Configured AgentOrchestrator instance.

        Example:
            # For most use cases, prefer run_agentic_workflow():
            provider = InvestmentWorkflowProvider()
            result = await provider.run_agentic_workflow(
                "comprehensive",
                context={"symbol": "AAPL"},
                provider="ollama",
            )

            # For custom orchestrator usage:
            orchestrator = await InvestmentVertical.create_orchestrator(
                provider="ollama",
                model="gpt-oss:20b"
            )
            # Use with WorkflowExecutor directly
            executor = WorkflowExecutor(orchestrator)
            workflow = provider.get_workflow("comprehensive")
            result = await executor.execute(workflow, {"symbol": "AAPL"})
        """
        from victor.framework import Agent

        # Get default model from investigator config if not specified
        if model is None and provider == "ollama":
            try:
                from investigator.config import get_config

                config = get_config()
                model = config.ollama.models.get("synthesis", "gpt-oss:20b")
            except Exception:
                model = "gpt-oss:20b"

        # Create Agent with Investment vertical
        agent = await Agent.create(
            provider=provider,
            model=model,
            tools=cls.get_tools(),
            vertical=cls,
            temperature=0.3,
        )

        # Return the underlying orchestrator
        return agent.get_orchestrator()

    @classmethod
    async def run_analysis(
        cls,
        symbol: str,
        mode: str = "standard",
    ) -> Dict[str, Any]:
        """Run investment analysis using the workflow system.

        This is the primary entry point for running investment analysis
        through Victor's workflow framework. Uses WorkflowExecutor with
        registered compute handlers for the context-stuffing pattern.

        Args:
            symbol: Stock ticker symbol to analyze.
            mode: Analysis mode (quick, standard, comprehensive).

        Returns:
            Analysis results dictionary.

        Example:
            results = await InvestmentVertical.run_analysis("AAPL", mode="comprehensive")
            print(results["recommendation"]["action"])
        """
        workflow_provider = cls.get_workflow_provider()
        if workflow_provider:
            # Use run_workflow_with_handlers() for handler-based execution
            # This avoids deprecated run_workflow() while maintaining handler support
            result = await workflow_provider.run_workflow_with_handlers(
                mode,
                context={"symbol": symbol},
            )
            # Convert WorkflowResult to dict
            if hasattr(result, "context") and result.context:
                return result.context.to_dict() if hasattr(result.context, "to_dict") else dict(result.context)
            return {"success": result.success, "error": getattr(result, "error", None)}

        # Fallback to direct workflow call
        from victor_invest.workflows import AnalysisMode
        from victor_invest.workflows import run_analysis as direct_run

        mode_map = {
            "quick": AnalysisMode.QUICK,
            "standard": AnalysisMode.STANDARD,
            "comprehensive": AnalysisMode.COMPREHENSIVE,
        }
        analysis_mode = mode_map.get(mode, AnalysisMode.STANDARD)
        result = await direct_run(symbol, analysis_mode)
        return result.to_dict()
