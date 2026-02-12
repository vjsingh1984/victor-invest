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

"""Victor Invest Tools Module.

This module provides investment analysis tools that wrap the existing
investigator infrastructure for use with the Victor agent framework.

Tools available:
- SECFilingTool: SEC EDGAR filing retrieval and XBRL parsing
- ValuationTool: Multi-model valuation (DCF, GGM, P/E, P/S, P/B, EV/EBITDA)
- TechnicalIndicatorsTool: Technical analysis indicators
- MarketDataTool: Market data and stock information
- CacheTool: Cache management operations
- MacroDataTool: FRED macroeconomic data and indicators
- CreditRiskTool: Credit risk models (Altman Z, Beneish M, Piotroski F)
- InsiderTradingTool: SEC Form 4 insider trading sentiment analysis
- TreasuryDataTool: Treasury yield curve and market regime analysis
- InstitutionalHoldingsTool: SEC Form 13F institutional holdings analysis
- ShortInterestTool: FINRA short interest and squeeze risk analysis
- MarketRegimeTool: Comprehensive market regime detection and investment recommendations
- ValuationSignalsTool: Integrated valuation signal analysis (credit risk, insider, short interest, regime)

Base classes:
- BaseTool: Abstract base class for all tools
- ToolResult: Standardized result container

Example usage:
    from victor_invest.tools import (
        SECFilingTool,
        ValuationTool,
        TechnicalIndicatorsTool,
        MarketDataTool,
        CacheTool,
        ToolResult,
    )

    # Create and use SEC filing tool
    sec_tool = SECFilingTool()
    result = await sec_tool.execute(
        symbol="AAPL",
        action="get_company_facts"
    )

    if result.success:
        facts = result.output
    else:
        print(f"Error: {result.error}")

    # Create and use valuation tool
    val_tool = ValuationTool()
    result = await val_tool.execute(
        symbol="AAPL",
        model="all"
    )

    # Get all available tools
    tools = get_all_tools()
"""

from victor.tools.base import BaseTool as VictorBaseTool

from victor_invest.tools.base import BaseTool, ToolResult
from victor_invest.tools.cache import CacheTool
from victor_invest.tools.credit_risk import CreditRiskTool
from victor_invest.tools.entry_exit_signals import EntryExitSignalTool
from victor_invest.tools.insider_trading import InsiderTradingTool
from victor_invest.tools.institutional_holdings import InstitutionalHoldingsTool
from victor_invest.tools.macro_data import MacroDataTool
from victor_invest.tools.market_data import MarketDataTool
from victor_invest.tools.market_regime import MarketRegimeTool
from victor_invest.tools.rl_backtest import RLBacktestTool
from victor_invest.tools.sec_filing import SECFilingTool
from victor_invest.tools.short_interest import ShortInterestTool
from victor_invest.tools.technical_indicators import TechnicalIndicatorsTool
from victor_invest.tools.treasury_data import TreasuryDataTool
from victor_invest.tools.valuation import ValuationTool
from victor_invest.tools.valuation_signals import ValuationSignalsTool

# All tool classes
TOOL_CLASSES = [
    SECFilingTool,
    ValuationTool,
    TechnicalIndicatorsTool,
    MarketDataTool,
    CacheTool,
    EntryExitSignalTool,
    RLBacktestTool,
    MacroDataTool,
    CreditRiskTool,
    InsiderTradingTool,
    TreasuryDataTool,
    InstitutionalHoldingsTool,
    ShortInterestTool,
    MarketRegimeTool,
    ValuationSignalsTool,
]

# Tool registry mapping names to classes
TOOL_REGISTRY = {tool_cls.name: tool_cls for tool_cls in TOOL_CLASSES if hasattr(tool_cls, "name")}


def get_tool(name: str, config=None) -> BaseTool:
    """Get a tool instance by name.

    Args:
        name: Tool name (e.g., "sec_filing", "valuation")
        config: Optional configuration object

    Returns:
        Tool instance

    Raises:
        ValueError: If tool name is not found

    Example:
        tool = get_tool("sec_filing")
        result = await tool.execute(symbol="AAPL")
    """
    if name not in TOOL_REGISTRY:
        available = list(TOOL_REGISTRY.keys())
        raise ValueError(f"Unknown tool: {name}. Available: {available}")

    return TOOL_REGISTRY[name](config=config)


def get_all_tools(config=None) -> list:
    """Get instances of all available tools.

    Args:
        config: Optional configuration object (shared by all tools)

    Returns:
        List of tool instances

    Example:
        tools = get_all_tools()
        for tool in tools:
            print(f"{tool.name}: {tool.description[:50]}...")
    """
    return [tool_cls(config=config) for tool_cls in TOOL_CLASSES]


def get_tool_names() -> list:
    """Get list of available tool names.

    Returns:
        List of tool name strings
    """
    return list(TOOL_REGISTRY.keys())


def get_tool_descriptions() -> dict:
    """Get descriptions of all available tools.

    Returns:
        Dict mapping tool names to descriptions

    Example:
        descriptions = get_tool_descriptions()
        for name, desc in descriptions.items():
            print(f"- {name}: {desc[:100]}...")
    """
    return {name: cls.description for name, cls in TOOL_REGISTRY.items()}


def register_investment_tools(
    tool_registry,
    config=None,
    *,
    enabled: bool = True,
    strict: bool = False,
) -> dict:
    """Register all Victor-Invest tools into a Victor ToolRegistry.

    This enables LLM tool calling for the investment vertical by ensuring
    the Victor orchestrator has the investment toolset available.

    Args:
        tool_registry: Victor ToolRegistry instance to register tools with.
        config: Optional config object passed into tool constructors.
        enabled: Whether tools are enabled by default in the registry.
        strict: If True, raise on the first registration error.

    Returns:
        Dict with registration stats: {registered, errors}.
    """
    errors = []
    registered = 0

    for tool_cls in TOOL_CLASSES:
        try:
            tool_instance = tool_cls(config=config)
            if not isinstance(tool_instance, VictorBaseTool):
                raise TypeError(f"Expected Victor BaseTool instance, got {type(tool_instance).__name__}")
            try:
                tool_registry.register(tool_instance, enabled=enabled)
            except TypeError:
                # Fallback for older ToolRegistry variants without `enabled` arg.
                tool_registry.register(tool_instance)
            registered += 1
        except Exception as exc:
            error_msg = f"{getattr(tool_cls, '__name__', 'UnknownTool')}: {exc}"
            errors.append(error_msg)
            if strict:
                raise

    return {"registered": registered, "errors": errors}


__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    # Tool implementations
    "SECFilingTool",
    "ValuationTool",
    "TechnicalIndicatorsTool",
    "MarketDataTool",
    "CacheTool",
    "EntryExitSignalTool",
    "RLBacktestTool",
    "MacroDataTool",
    "CreditRiskTool",
    "InsiderTradingTool",
    "TreasuryDataTool",
    "InstitutionalHoldingsTool",
    "ShortInterestTool",
    "MarketRegimeTool",
    "ValuationSignalsTool",
    # Utility functions
    "get_tool",
    "get_all_tools",
    "get_tool_names",
    "get_tool_descriptions",
    "register_investment_tools",
    # Registry
    "TOOL_REGISTRY",
    "TOOL_CLASSES",
]
