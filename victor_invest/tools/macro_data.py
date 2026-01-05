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

"""Macro Data Tool for Victor Invest.

This tool wraps the existing FRED macro indicators infrastructure to expose
macroeconomic data via CLI and agent interfaces.

Infrastructure wrapped:
- investigator.infrastructure.external.fred.macro_indicators.MacroIndicatorsFetcher

Data Categories:
- growth: GDP, Real GDP, GDP Growth Rate, GDPNow Forecast
- employment: Unemployment Rate, Nonfarm Payrolls, Job Openings
- inflation: CPI, PCE, Sticky CPI, Breakeven Inflation
- rates: Fed Funds, 10Y Treasury, 10Y-2Y Spread, Mortgage Rates
- credit: High Yield Credit Spread
- debt: Federal Debt/GDP, Household Debt/GDP, Corporate Debt
- market: S&P 500, VIX Volatility
- sentiment: Consumer Sentiment
- housing: Housing Starts, Case-Shiller Index
- monetary: M2 Money Stock, Fed Assets, Savings Rate
- trade: Dollar Index, Trade Balance, USD/EUR

Example:
    tool = MacroDataTool()

    # Get all key indicators
    result = await tool.execute(action="get_summary")

    # Get specific category
    result = await tool.execute(
        action="get_category",
        category="rates"
    )

    # Get specific indicators
    result = await tool.execute(
        action="get_indicators",
        indicators=["DGS10", "FEDFUNDS", "VIXCLS"]
    )

    # Get Buffett Indicator
    result = await tool.execute(action="buffett_indicator")

    # Get time series for an indicator
    result = await tool.execute(
        action="get_time_series",
        indicator_id="DGS10",
        limit=365
    )
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# Category mappings for organized access
INDICATOR_CATEGORIES = {
    "growth": ["GDP", "GDPC1", "A939RX0Q048SBEA", "GDPNOW", "NYGDPPCAPKDUSA"],
    "employment": ["UNRATE", "PAYEMS", "JTSJOL"],
    "inflation": ["CPIAUCSL", "PCEPI", "CORESTICKM159SFRBATL", "T10YIE"],
    "rates": ["FEDFUNDS", "DFF", "DGS10", "T10Y2Y", "MORTGAGE30US"],
    "credit": ["BAMLH0A0HYM2"],
    "debt": [
        "GFDEGDQ188S", "GFDGDPA188S", "HDTGPDUSQ163N", "CMDEBT",
        "NCBDBIQ027S", "TBSDODNS", "TDSP", "FODSP"
    ],
    "market": ["SP500", "VIXCLS"],
    "sentiment": ["UMCSENT"],
    "housing": ["HOUST", "CSUSHPISA"],
    "monetary": ["M2SL", "WALCL", "PSAVERT"],
    "trade": ["DTWEXBGS", "BOPGSTB", "DEXUSEU"],
}

# Flatten all indicators for quick lookup
ALL_INDICATORS = {
    ind: cat for cat, inds in INDICATOR_CATEGORIES.items() for ind in inds
}


class MacroDataTool(BaseTool):
    """Tool for accessing FRED macroeconomic data.

    Provides CLI and agent access to Federal Reserve Economic Data (FRED)
    stored in the PostgreSQL database. Supports category-based queries,
    specific indicator lookups, time series retrieval, and derived
    indicators like the Buffett Indicator.

    Supported actions:
    - get_summary: Get comprehensive macro summary with alerts
    - get_category: Get indicators for a specific category
    - get_indicators: Get specific indicators by ID
    - get_time_series: Get historical data for an indicator
    - buffett_indicator: Calculate Stock Market to GDP ratio
    - list_categories: List available categories and their indicators

    Attributes:
        name: "macro_data"
        description: Tool description for agent discovery
    """

    name = "macro_data"
    description = """Access FRED macroeconomic data for investment analysis.

Actions:
- get_summary: Get comprehensive macro summary with categorized indicators and alerts
- get_category: Get all indicators for a category (growth, inflation, rates, etc.)
- get_indicators: Get specific indicators by FRED series ID
- get_time_series: Get historical time series for an indicator
- buffett_indicator: Calculate Buffett Indicator (Market Cap / GDP)
- list_categories: List available categories and their indicators

Parameters:
- action: One of the actions above (required)
- category: Category name for get_category (growth, employment, inflation, rates, credit, debt, market, sentiment, housing, monetary, trade)
- indicators: List of FRED series IDs for get_indicators
- indicator_id: Single FRED series ID for get_time_series
- lookback_days: Days of historical data (default: 1095 = 3 years)
- limit: Max data points for time series (default: 1000)

Example indicators: DGS10 (10Y Treasury), FEDFUNDS (Fed Funds Rate), VIXCLS (VIX), GDP, UNRATE (Unemployment), CPIAUCSL (CPI)
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Macro Data Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._fetcher = None

    async def initialize(self) -> None:
        """Initialize FRED infrastructure components."""
        try:
            from investigator.infrastructure.external.fred.macro_indicators import (
                MacroIndicatorsFetcher
            )

            self._fetcher = MacroIndicatorsFetcher()
            self._initialized = True
            logger.info("MacroDataTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MacroDataTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Dict[str, Any],
        action: str = "get_summary",
        category: Optional[str] = None,
        indicators: Optional[List[str]] = None,
        indicator_id: Optional[str] = None,
        lookback_days: int = 1095,
        limit: int = 1000,
        **kwargs
    ) -> ToolResult:
        """Execute macro data operation.

        Args:
            action: Operation to perform:
                - "get_summary": Comprehensive macro summary
                - "get_category": Category-specific indicators
                - "get_indicators": Specific indicators by ID
                - "get_time_series": Historical time series
                - "buffett_indicator": Market/GDP ratio
                - "list_categories": Available categories
            category: Category name (growth, inflation, rates, etc.)
            indicators: List of FRED series IDs
            indicator_id: Single indicator for time series
            lookback_days: Historical lookback period (default: 1095)
            limit: Max data points for time series
            **kwargs: Additional parameters

        Returns:
            ToolResult with macro data or error message
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "get_summary":
                return await self._get_summary()
            elif action == "get_category":
                return await self._get_category(category, lookback_days)
            elif action == "get_indicators":
                return await self._get_indicators(indicators, lookback_days)
            elif action == "get_time_series":
                return await self._get_time_series(indicator_id, limit)
            elif action == "buffett_indicator":
                return await self._get_buffett_indicator()
            elif action == "list_categories":
                return await self._list_categories()
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "get_summary, get_category, get_indicators, get_time_series, "
                    "buffett_indicator, list_categories"
                )

        except Exception as e:
            logger.error(f"MacroDataTool execute error: {e}")
            return ToolResult.error_result(
                f"Macro data operation failed: {str(e)}",
                metadata={"action": action}
            )

    async def _get_summary(self) -> ToolResult:
        """Get comprehensive macro summary with alerts.

        Returns:
            ToolResult with categorized indicators and risk assessment
        """
        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                None,
                self._fetcher.get_macro_summary
            )

            if not summary:
                return ToolResult.error_result("Failed to retrieve macro summary")

            # Format for cleaner output
            formatted = {
                "timestamp": summary.get("timestamp"),
                "overall_assessment": summary.get("overall_assessment"),
                "alerts": summary.get("alerts", []),
                "categories": {},
            }

            # Format each category
            for cat_name, cat_data in summary.get("categories", {}).items():
                formatted["categories"][cat_name] = self._format_category_data(cat_data)

            # Add Buffett Indicator if available
            if "buffett_indicator" in summary:
                bi = summary["buffett_indicator"]
                formatted["buffett_indicator"] = {
                    "ratio": round(bi["ratio"], 2) if bi.get("ratio") else None,
                    "interpretation": bi.get("interpretation"),
                    "signal": bi.get("signal"),
                    "components": {
                        "vti_price": bi.get("vti_price"),
                        "vti_date": str(bi.get("vti_date")) if bi.get("vti_date") else None,
                        "gdp_billions": bi.get("gdp"),
                        "gdp_date": str(bi.get("gdp_date")) if bi.get("gdp_date") else None,
                        "estimated_market_cap_billions": bi.get("estimated_market_cap"),
                    }
                }

            return ToolResult.success_result(
                data=formatted,
                metadata={
                    "source": "fred",
                    "indicator_count": len(summary.get("indicators", {})),
                    "alert_count": len(summary.get("alerts", [])),
                }
            )

        except Exception as e:
            logger.error(f"Error getting macro summary: {e}")
            return ToolResult.error_result(f"Failed to get macro summary: {str(e)}")

    async def _get_category(
        self,
        category: Optional[str],
        lookback_days: int
    ) -> ToolResult:
        """Get indicators for a specific category.

        Args:
            category: Category name
            lookback_days: Historical lookback period

        Returns:
            ToolResult with category indicators
        """
        if not category:
            return ToolResult.error_result(
                "Category is required. Use 'list_categories' to see available categories."
            )

        category = category.lower().strip()
        if category not in INDICATOR_CATEGORIES:
            available = list(INDICATOR_CATEGORIES.keys())
            return ToolResult.error_result(
                f"Unknown category: {category}. Available: {available}"
            )

        try:
            indicator_ids = INDICATOR_CATEGORIES[category]

            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self._fetcher.get_latest_values(
                    indicator_ids=indicator_ids,
                    lookback_days=lookback_days
                )
            )

            if not data:
                return ToolResult.error_result(
                    f"No data found for category: {category}"
                )

            return ToolResult.success_result(
                data={
                    "category": category,
                    "indicators": self._format_category_data(data),
                    "indicator_count": len(data),
                },
                metadata={
                    "source": "fred",
                    "lookback_days": lookback_days,
                }
            )

        except Exception as e:
            logger.error(f"Error getting category {category}: {e}")
            return ToolResult.error_result(f"Failed to get category data: {str(e)}")

    async def _get_indicators(
        self,
        indicators: Optional[List[str]],
        lookback_days: int
    ) -> ToolResult:
        """Get specific indicators by ID.

        Args:
            indicators: List of FRED series IDs
            lookback_days: Historical lookback period

        Returns:
            ToolResult with indicator data
        """
        if not indicators:
            return ToolResult.error_result(
                "Indicators list is required. Example: ['DGS10', 'FEDFUNDS', 'VIXCLS']"
            )

        try:
            # Normalize indicator IDs
            indicator_ids = [ind.upper().strip() for ind in indicators]

            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self._fetcher.get_latest_values(
                    indicator_ids=indicator_ids,
                    lookback_days=lookback_days
                )
            )

            if not data:
                return ToolResult.error_result(
                    f"No data found for indicators: {indicator_ids}"
                )

            # Note missing indicators
            found = set(data.keys())
            requested = set(indicator_ids)
            missing = requested - found

            warnings = []
            if missing:
                warnings.append(f"Missing indicators: {list(missing)}")

            return ToolResult.success_result(
                data={
                    "indicators": self._format_category_data(data),
                    "requested": indicator_ids,
                    "found": list(found),
                },
                warnings=warnings,
                metadata={
                    "source": "fred",
                    "lookback_days": lookback_days,
                }
            )

        except Exception as e:
            logger.error(f"Error getting indicators: {e}")
            return ToolResult.error_result(f"Failed to get indicators: {str(e)}")

    async def _get_time_series(
        self,
        indicator_id: Optional[str],
        limit: int
    ) -> ToolResult:
        """Get historical time series for an indicator.

        Args:
            indicator_id: FRED series ID
            limit: Maximum data points

        Returns:
            ToolResult with time series data
        """
        if not indicator_id:
            return ToolResult.error_result(
                "indicator_id is required. Example: 'DGS10' for 10-Year Treasury"
            )

        try:
            indicator_id = indicator_id.upper().strip()

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self._fetcher.get_time_series(
                    indicator_id=indicator_id,
                    limit=limit
                )
            )

            if df.empty:
                return ToolResult.error_result(
                    f"No time series data found for: {indicator_id}"
                )

            # Convert to list of dicts for JSON serialization
            time_series = []
            for _, row in df.iterrows():
                time_series.append({
                    "date": str(row["date"]),
                    "value": row["value"],
                })

            # Calculate summary stats
            values = df["value"].values
            stats = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()) if len(values) > 1 else 0,
                "latest": float(values[-1]) if len(values) > 0 else None,
                "earliest": float(values[0]) if len(values) > 0 else None,
            }

            return ToolResult.success_result(
                data={
                    "indicator_id": indicator_id,
                    "category": ALL_INDICATORS.get(indicator_id, "unknown"),
                    "data_points": len(time_series),
                    "date_range": {
                        "start": time_series[0]["date"] if time_series else None,
                        "end": time_series[-1]["date"] if time_series else None,
                    },
                    "statistics": stats,
                    "time_series": time_series,
                },
                metadata={
                    "source": "fred",
                    "limit": limit,
                }
            )

        except Exception as e:
            logger.error(f"Error getting time series for {indicator_id}: {e}")
            return ToolResult.error_result(f"Failed to get time series: {str(e)}")

    async def _get_buffett_indicator(self) -> ToolResult:
        """Calculate Buffett Indicator (Total Market Cap / GDP).

        Returns:
            ToolResult with Buffett Indicator calculation
        """
        try:
            loop = asyncio.get_event_loop()
            buffett = await loop.run_in_executor(
                None,
                self._fetcher.calculate_buffett_indicator
            )

            if not buffett:
                return ToolResult.error_result(
                    "Failed to calculate Buffett Indicator. "
                    "Missing VTI price or GDP data."
                )

            # Interpretation guide
            interpretation_guide = {
                "< 75%": "Significantly Undervalued - Strong buy signal",
                "75-90%": "Moderately Undervalued - Buy signal",
                "90-115%": "Fair Value - Neutral",
                "115-140%": "Moderately Overvalued - Caution",
                "> 140%": "Significantly Overvalued - Warning",
            }

            return ToolResult.success_result(
                data={
                    "ratio_percent": round(buffett["ratio"], 2),
                    "interpretation": buffett["interpretation"],
                    "signal": buffett["signal"],
                    "components": {
                        "vti_price": buffett["vti_price"],
                        "vti_date": str(buffett["vti_date"]),
                        "gdp_billions": buffett["gdp"],
                        "gdp_date": str(buffett["gdp_date"]),
                        "estimated_wilshire5000_index": round(buffett["estimated_w5k_index"], 2),
                        "estimated_market_cap_billions": round(buffett["estimated_market_cap"], 2),
                    },
                    "calculation": {
                        "formula": "(Estimated Total Market Cap / GDP) × 100",
                        "note": buffett["note"],
                    },
                    "interpretation_guide": interpretation_guide,
                },
                metadata={
                    "source": "fred+tickerdata",
                    "calculation_method": "VTI proxy for Wilshire 5000",
                }
            )

        except Exception as e:
            logger.error(f"Error calculating Buffett Indicator: {e}")
            return ToolResult.error_result(f"Failed to calculate Buffett Indicator: {str(e)}")

    async def _list_categories(self) -> ToolResult:
        """List available categories and their indicators.

        Returns:
            ToolResult with category listing
        """
        categories_info = {}
        for cat_name, indicator_ids in INDICATOR_CATEGORIES.items():
            categories_info[cat_name] = {
                "indicator_count": len(indicator_ids),
                "indicators": indicator_ids,
            }

        return ToolResult.success_result(
            data={
                "categories": categories_info,
                "total_categories": len(INDICATOR_CATEGORIES),
                "total_indicators": len(ALL_INDICATORS),
            },
            metadata={"source": "static_mapping"}
        )

    def _format_category_data(self, data: Dict) -> Dict:
        """Format indicator data for clean output.

        Args:
            data: Raw indicator data from fetcher

        Returns:
            Formatted indicator dict
        """
        formatted = {}
        for ind_id, ind_data in data.items():
            if ind_data is None:
                continue

            formatted[ind_id] = {
                "name": ind_data.get("name", ind_id),
                "value": ind_data.get("value"),
                "date": str(ind_data.get("date")) if ind_data.get("date") else None,
                "units": ind_data.get("units"),
                "frequency": ind_data.get("frequency"),
                "change": {
                    "absolute": ind_data.get("change_abs"),
                    "percent": ind_data.get("change_pct"),
                    "previous_value": ind_data.get("prev_value"),
                    "previous_date": str(ind_data.get("prev_date")) if ind_data.get("prev_date") else None,
                } if ind_data.get("change_abs") is not None else None,
            }

        return formatted

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Macro Data Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_summary", "get_category", "get_indicators",
                        "get_time_series", "buffett_indicator", "list_categories"
                    ],
                    "description": "Action to perform",
                    "default": "get_summary"
                },
                "category": {
                    "type": "string",
                    "enum": list(INDICATOR_CATEGORIES.keys()),
                    "description": "Category for get_category action"
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of FRED series IDs for get_indicators"
                },
                "indicator_id": {
                    "type": "string",
                    "description": "Single FRED series ID for get_time_series"
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Days of historical data",
                    "default": 1095,
                    "minimum": 30,
                    "maximum": 7300
                },
                "limit": {
                    "type": "integer",
                    "description": "Max data points for time series",
                    "default": 1000,
                    "minimum": 10,
                    "maximum": 10000
                }
            },
            "required": ["action"]
        }
