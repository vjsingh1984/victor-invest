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

"""Victor Invest - AI-powered investment research and analysis vertical.

A Victor vertical for institutional-grade equity research combining:
- SEC fundamentals analysis with XBRL processing
- Multi-model valuation (DCF, P/E, P/S, P/B, GGM, EV/EBITDA)
- Technical indicators and market context
- Dynamic model weighting with company archetype detection

Uses the Victor-Core framework for:
- Declarative AgentSpec definitions
- StateGraph workflow orchestration
- Multi-provider LLM support (Ollama, Anthropic, OpenAI, etc.)

Example:
    from victor_invest import InvestmentVertical, run_analysis, AnalysisMode

    # Run analysis using StateGraph workflow
    result = await run_analysis("AAPL", AnalysisMode.COMPREHENSIVE)
    print(result.recommendation)

    # Or use the vertical with Victor Agent directly
    from victor.framework import Agent
    config = InvestmentVertical.get_config()
    agent = await Agent.create(vertical=InvestmentVertical)
"""

__version__ = "0.1.0"
__author__ = "Vijaykumar Singh"
__email__ = "singhvjd@gmail.com"
__license__ = "Apache-2.0"

# Use lazy imports to avoid import errors when victor package has issues


def __getattr__(name):
    """Lazy import handler to support optional victor dependencies."""
    # Vertical (Victor integration) - requires victor package
    if name in ("InvestmentVertical", "InvestmentAssistant"):
        from victor_invest.vertical import InvestmentVertical
        if name == "InvestmentAssistant":
            return InvestmentVertical  # Backward compatibility alias
        return InvestmentVertical

    # Workflows (StateGraph-based)
    if name in (
        "AnalysisMode", "AnalysisWorkflowState", "build_graph_for_mode",
        "build_quick_graph", "build_standard_graph", "build_comprehensive_graph",
        "run_analysis"
    ):
        from victor_invest import workflows
        return getattr(workflows, name)

    # Agent specifications
    if name in (
        "SEC_AGENT_SPEC", "FUNDAMENTAL_AGENT_SPEC", "TECHNICAL_AGENT_SPEC",
        "MARKET_AGENT_SPEC", "SYNTHESIS_AGENT_SPEC"
    ):
        from victor_invest import agents
        return getattr(agents, name)

    # Tools - these work without victor
    if name in (
        "BaseTool", "ToolResult", "SECFilingTool", "ValuationTool",
        "TechnicalIndicatorsTool", "MarketDataTool", "CacheTool",
        "get_tool", "get_all_tools", "get_tool_names"
    ):
        from victor_invest import tools
        return getattr(tools, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version info
    "__version__",
    # Vertical
    "InvestmentVertical",
    "InvestmentAssistant",  # Backward compatibility
    # Workflows
    "AnalysisMode",
    "AnalysisWorkflowState",
    "build_graph_for_mode",
    "build_quick_graph",
    "build_standard_graph",
    "build_comprehensive_graph",
    "run_analysis",
    # Agent specifications
    "SEC_AGENT_SPEC",
    "FUNDAMENTAL_AGENT_SPEC",
    "TECHNICAL_AGENT_SPEC",
    "MARKET_AGENT_SPEC",
    "SYNTHESIS_AGENT_SPEC",
    # Tools
    "BaseTool",
    "ToolResult",
    "SECFilingTool",
    "ValuationTool",
    "TechnicalIndicatorsTool",
    "MarketDataTool",
    "CacheTool",
    "get_tool",
    "get_all_tools",
    "get_tool_names",
]
