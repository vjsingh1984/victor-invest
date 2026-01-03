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

"""Market Data Tool for Victor Invest.

This tool wraps the existing market data infrastructure to provide
stock price data, company information, and market metrics.

Infrastructure wrapped:
- investigator.infrastructure.database.market_data.DatabaseMarketDataFetcher

Example:
    tool = MarketDataTool()

    # Get current quote
    result = await tool.execute(
        symbol="AAPL",
        action="get_quote"
    )

    # Get historical data
    result = await tool.execute(
        symbol="AAPL",
        action="get_history",
        days=365
    )

    # Get company info
    result = await tool.execute(
        symbol="AAPL",
        action="get_info"
    )
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MarketDataTool(BaseTool):
    """Tool for retrieving market data and stock information.

    Provides access to:
    - Current stock quotes and prices
    - Historical OHLCV data
    - Company information (sector, industry, market cap, beta)
    - 52-week highs/lows
    - Available symbols list

    Attributes:
        name: "market_data"
        description: Tool description for agent discovery
    """

    name = "market_data"
    description = """Retrieve market data and stock information.

Actions:
- get_quote: Get current price and basic quote data
- get_history: Get historical OHLCV data
- get_info: Get company information (sector, beta, market cap)
- get_price_change: Calculate price change over a period
- check_available: Check if symbol is available in database
- list_symbols: List available symbols in database

Parameters:
- symbol: Stock ticker symbol (required for most actions)
- action: One of the actions above (default: "get_quote")
- days: Number of days for historical data (default: 365)
- period: Period for price change calculation ("1d", "5d", "1m", "3m", "1y")

Returns current market data, historical prices, and company metadata.
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Market Data Tool.

        Args:
            config: Optional investigator config object
        """
        super().__init__(config)
        self._fetcher = None

    async def initialize(self) -> None:
        """Initialize market data infrastructure."""
        try:
            from investigator.infrastructure.database.market_data import (
                get_market_data_fetcher
            )

            if self.config is None:
                from investigator.config import get_config
                self.config = get_config()

            self._fetcher = get_market_data_fetcher(self.config)

            self._initialized = True
            logger.info("MarketDataTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MarketDataTool: {e}")
            raise

    async def execute(
        self,
        symbol: Optional[str] = None,
        action: str = "get_quote",
        days: int = 365,
        period: str = "1y",
        **kwargs
    ) -> ToolResult:
        """Execute market data retrieval operation.

        Args:
            symbol: Stock ticker symbol (required for most actions)
            action: Operation to perform:
                - "get_quote": Current price and quote data
                - "get_history": Historical OHLCV data
                - "get_info": Company information
                - "get_price_change": Price change calculation
                - "check_available": Check symbol availability
                - "list_symbols": List available symbols
            days: Days of historical data for get_history
            period: Period for price change ("1d", "5d", "1m", "3m", "1y")
            **kwargs: Additional parameters

        Returns:
            ToolResult with market data or error
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            # Some actions don't require a symbol
            if action == "list_symbols":
                return await self._list_symbols()

            if action == "check_available" and symbol:
                return await self._check_available(symbol.upper().strip())

            # All other actions require a symbol
            if not symbol:
                return ToolResult.error_result("Symbol is required for this action")

            symbol = symbol.upper().strip()

            if action == "get_quote":
                return await self._get_quote(symbol)
            elif action == "get_history":
                return await self._get_history(symbol, days)
            elif action == "get_info":
                return await self._get_info(symbol)
            elif action == "get_price_change":
                return await self._get_price_change(symbol, period)
            elif action == "check_available":
                return await self._check_available(symbol)
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "get_quote, get_history, get_info, get_price_change, "
                    "check_available, list_symbols"
                )

        except Exception as e:
            logger.error(f"MarketDataTool execute error: {e}")
            return ToolResult.error_result(
                f"Market data operation failed: {str(e)}",
                metadata={"symbol": symbol, "action": action}
            )

    async def _get_quote(self, symbol: str) -> ToolResult:
        """Get current quote data for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            ToolResult with quote data
        """
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                self._fetcher.get_stock_info,
                symbol
            )

            if not info:
                return ToolResult.error_result(
                    f"No quote data available for {symbol}",
                    metadata={"symbol": symbol}
                )

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "current_price": info.get("current_price"),
                    "volume": info.get("current_volume"),
                    "avg_volume": info.get("avg_volume"),
                    "market_cap": info.get("market_cap"),
                    "52_week_high": info.get("52_week_high"),
                    "52_week_low": info.get("52_week_low"),
                    "beta": info.get("beta"),
                },
                metadata={"source": "market_data_database"}
            )

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return ToolResult.error_result(f"Failed to get quote: {str(e)}")

    async def _get_history(
        self,
        symbol: str,
        days: int
    ) -> ToolResult:
        """Get historical OHLCV data.

        Args:
            symbol: Stock ticker
            days: Number of days

        Returns:
            ToolResult with historical data
        """
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self._fetcher.get_stock_data,
                symbol,
                days
            )

            if df is None or df.empty:
                return ToolResult.error_result(
                    f"No historical data available for {symbol}",
                    metadata={"symbol": symbol, "days": days}
                )

            # Convert to list of records
            records = []
            for idx, row in df.iterrows():
                record = {
                    "date": str(idx),
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "adj_close": row.get("Adj Close"),
                    "volume": int(row.get("Volume", 0)),
                }
                records.append(record)

            # Calculate summary statistics
            summary = {
                "high": float(df["High"].max()),
                "low": float(df["Low"].min()),
                "avg_close": float(df["Close"].mean()),
                "avg_volume": float(df["Volume"].mean()),
                "start_price": float(df["Close"].iloc[0]),
                "end_price": float(df["Close"].iloc[-1]),
                "return_pct": float(
                    ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
                ),
            }

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "days_returned": len(records),
                    "date_range": {
                        "start": str(df.index[0]),
                        "end": str(df.index[-1]),
                    },
                    "summary": summary,
                    "data": records  # Full OHLCV data
                },
                metadata={
                    "source": "market_data_database",
                    "requested_days": days
                }
            )

        except Exception as e:
            logger.error(f"Error getting history for {symbol}: {e}")
            return ToolResult.error_result(f"Failed to get history: {str(e)}")

    async def _get_info(self, symbol: str) -> ToolResult:
        """Get detailed company information.

        Args:
            symbol: Stock ticker

        Returns:
            ToolResult with company info
        """
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                self._fetcher.get_stock_info,
                symbol
            )

            if not info:
                return ToolResult.error_result(
                    f"No company info available for {symbol}",
                    metadata={"symbol": symbol}
                )

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("market_cap"),
                    "shares_outstanding": info.get("shares_outstanding"),
                    "float_shares": info.get("float_shares"),
                    "beta": info.get("beta"),
                    "pe_ratio": info.get("pe_ratio"),
                    "forward_pe": info.get("forward_pe"),
                    "dividend_yield": info.get("dividend_yield"),
                    "52_week_high": info.get("52_week_high"),
                    "52_week_low": info.get("52_week_low"),
                    "is_etf": info.get("is_etf", False),
                    "asset_type": info.get("asset_type"),
                    "cik": info.get("cik"),
                    "sic_code": info.get("sic_code"),
                },
                metadata={"source": "market_data_database"}
            )

        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return ToolResult.error_result(f"Failed to get info: {str(e)}")

    async def _get_price_change(
        self,
        symbol: str,
        period: str
    ) -> ToolResult:
        """Calculate price change over a period.

        Args:
            symbol: Stock ticker
            period: Period string ("1d", "5d", "1m", "3m", "1y")

        Returns:
            ToolResult with price change data
        """
        try:
            # Map period to days
            period_map = {
                "1d": 1,
                "5d": 5,
                "1w": 7,
                "1m": 30,
                "3m": 90,
                "6m": 180,
                "1y": 365,
                "ytd": 365,  # Simplified
            }

            days = period_map.get(period.lower(), 365)

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self._fetcher.get_stock_data,
                symbol,
                days + 5  # Buffer for market holidays
            )

            if df is None or len(df) < 2:
                return ToolResult.error_result(
                    f"Insufficient data for price change calculation",
                    metadata={"symbol": symbol, "period": period}
                )

            # Get prices
            current_price = float(df["Close"].iloc[-1])
            start_price = float(df["Close"].iloc[0])

            # Calculate changes
            price_change = current_price - start_price
            percent_change = ((current_price / start_price) - 1) * 100

            # Get high/low for period
            period_high = float(df["High"].max())
            period_low = float(df["Low"].min())

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "period": period,
                    "current_price": current_price,
                    "start_price": start_price,
                    "price_change": price_change,
                    "percent_change": round(percent_change, 2),
                    "period_high": period_high,
                    "period_low": period_low,
                    "date_range": {
                        "start": str(df.index[0]),
                        "end": str(df.index[-1]),
                    },
                    "trading_days": len(df)
                },
                metadata={"source": "market_data_database"}
            )

        except Exception as e:
            logger.error(f"Error calculating price change for {symbol}: {e}")
            return ToolResult.error_result(f"Failed to calculate price change: {str(e)}")

    async def _check_available(self, symbol: str) -> ToolResult:
        """Check if symbol is available in database.

        Args:
            symbol: Stock ticker

        Returns:
            ToolResult with availability status
        """
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self._fetcher.get_stock_data,
                symbol,
                5  # Just check if any data exists
            )

            is_available = df is not None and not df.empty
            data_points = len(df) if is_available else 0

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "available": is_available,
                    "data_points": data_points,
                    "latest_date": str(df.index[-1]) if is_available else None,
                },
                metadata={"source": "market_data_database"}
            )

        except Exception as e:
            logger.error(f"Error checking availability for {symbol}: {e}")
            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "available": False,
                    "data_points": 0,
                    "error": str(e)
                }
            )

    async def _list_symbols(self) -> ToolResult:
        """List all available symbols in database.

        Returns:
            ToolResult with list of symbols
        """
        try:
            loop = asyncio.get_event_loop()
            symbols = await loop.run_in_executor(
                None,
                self._fetcher.get_available_symbols
            )

            return ToolResult.success_result(
                data={
                    "count": len(symbols),
                    "symbols": symbols
                },
                metadata={"source": "market_data_database"}
            )

        except Exception as e:
            logger.error(f"Error listing symbols: {e}")
            return ToolResult.error_result(f"Failed to list symbols: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Market Data Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "get_quote", "get_history", "get_info",
                        "get_price_change", "check_available", "list_symbols"
                    ],
                    "description": "Action to perform",
                    "default": "get_quote"
                },
                "days": {
                    "type": "integer",
                    "description": "Days of historical data",
                    "default": 365,
                    "minimum": 1,
                    "maximum": 1825
                },
                "period": {
                    "type": "string",
                    "enum": ["1d", "5d", "1w", "1m", "3m", "6m", "1y", "ytd"],
                    "description": "Period for price change calculation",
                    "default": "1y"
                }
            },
            "required": []  # symbol required for most actions but not list_symbols
        }
