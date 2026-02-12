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

"""Insider Trading Tool for Victor Invest.

This tool provides access to SEC Form 4 insider trading data and sentiment
analysis via CLI and agent interfaces.

Available Actions:
- sentiment: Get insider sentiment analysis (buy/sell ratio, key insiders)
- recent: Get recent insider transactions
- clusters: Detect coordinated insider activity
- key_insiders: Get key insider (C-suite, directors) summary
- fetch: Fetch latest Form 4 filings from SEC EDGAR

Example:
    tool = InsiderTradingTool()

    # Get sentiment analysis
    result = await tool.execute(symbol="AAPL", action="sentiment", days=90)

    # Get recent transactions
    result = await tool.execute(symbol="AAPL", action="recent", days=30)

    # Detect cluster activity
    result = await tool.execute(symbol="AAPL", action="clusters")
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class InsiderTradingTool(BaseTool):
    """Tool for insider trading analysis.

    Provides CLI and agent access to SEC Form 4 data including
    sentiment analysis, cluster detection, and key insider tracking.

    Supported actions:
    - sentiment: Insider sentiment score and classification
    - recent: Recent insider transactions
    - clusters: Detect coordinated buying/selling
    - key_insiders: C-suite and director activity summary
    - fetch: Fetch fresh data from SEC EDGAR

    Attributes:
        name: "insider_trading"
        description: Tool description for agent discovery
    """

    name = "insider_trading"
    description = """Analyze SEC Form 4 insider trading data for investment signals.

Actions:
- sentiment: Get insider sentiment score (-1 to +1) with buy/sell analysis
- recent: List recent insider transactions for a symbol
- clusters: Detect coordinated insider buying or selling
- key_insiders: Summarize C-suite and director activity
- fetch: Fetch latest Form 4 filings from SEC EDGAR

Parameters:
- symbol: Stock ticker symbol (required)
- action: One of the actions above (default: "sentiment")
- days: Analysis period in days (default: 90)
- significant_only: For 'recent' action, filter to significant transactions

Returns sentiment classification (bullish/bearish/neutral), transaction counts,
values, and cluster detection flags.
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Insider Trading Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._sentiment_service = None
        self._dao = None
        self._fetcher = None
        self._data_source_manager = None

    async def initialize(self) -> None:
        """Initialize insider trading services."""
        try:
            from dao.insider_trading_dao import get_insider_trading_dao
            from investigator.domain.services.data_sources.manager import DataSourceManager
            from investigator.domain.services.sentiment import get_insider_activity_service

            self._sentiment_service = get_insider_activity_service()
            self._dao = get_insider_trading_dao()
            self._data_source_manager = DataSourceManager()

            self._initialized = True
            logger.info("InsiderTradingTool initialized with DataSourceManager")

        except Exception as e:
            logger.error(f"Failed to initialize InsiderTradingTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Optional[Dict[str, Any]] = None,
        symbol: str = "",
        action: str = "sentiment",
        days: int = 90,
        significant_only: bool = False,
        **kwargs,
    ) -> ToolResult:
        """Execute insider trading analysis.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
            action: Analysis type:
                - "sentiment": Insider sentiment analysis
                - "recent": Recent transactions
                - "clusters": Cluster detection
                - "key_insiders": Key insider summary
                - "fetch": Fetch fresh SEC data
            days: Analysis period in days
            significant_only: For 'recent', filter to significant only
            **kwargs: Additional parameters

        Returns:
            ToolResult with insider trading analysis
        """
        try:
            await self.ensure_initialized()

            symbol = symbol.upper().strip()
            if not symbol:
                return ToolResult.create_failure("Symbol is required")

            action = action.lower().strip()

            if action == "sentiment":
                return await self._get_sentiment(symbol, days)
            elif action == "recent":
                return await self._get_recent(symbol, days, significant_only)
            elif action == "clusters":
                return await self._detect_clusters(symbol, days)
            elif action == "key_insiders":
                return await self._get_key_insiders(symbol, days)
            elif action == "fetch":
                return await self._fetch_filings(symbol, days)
            else:
                return ToolResult.create_failure(
                    f"Unknown action: {action}. Valid actions: " "sentiment, recent, clusters, key_insiders, fetch"
                )

        except Exception as e:
            logger.error(f"InsiderTradingTool execute error for {symbol}: {e}")
            return ToolResult.create_failure(
                f"Insider trading analysis failed: {str(e)}", metadata={"symbol": symbol, "action": action}
            )

    async def _get_sentiment(self, symbol: str, days: int) -> ToolResult:
        """Get insider sentiment analysis.

        Tries DataSourceManager first for consolidated data, then falls back
        to the specialized sentiment service for detailed analysis.
        """
        # Try DataSourceManager first for consolidated insider data
        if self._data_source_manager:
            try:
                consolidated = self._data_source_manager.get_data(symbol)
                if consolidated.insider:
                    insider_data = consolidated.insider
                    summary = insider_data.get("summary", {})

                    # Build sentiment response from consolidated data
                    buys = summary.get("buys", 0)
                    sells = summary.get("sells", 0)
                    total = buys + sells

                    # Calculate sentiment score (-1 to +1)
                    if total > 0:
                        sentiment_score = (buys - sells) / total
                    else:
                        sentiment_score = 0.0

                    # Classify sentiment
                    if sentiment_score > 0.3:
                        classification = "bullish"
                    elif sentiment_score < -0.3:
                        classification = "bearish"
                    else:
                        classification = "neutral"

                    return ToolResult.create_success(output={
                            "symbol": symbol,
                            "period_days": days,
                            "sentiment_score": round(sentiment_score, 3),
                            "classification": classification,
                            "buy_count": buys,
                            "sell_count": sells,
                            "total_transactions": total,
                            "buy_value": summary.get("buy_value", 0),
                            "sell_value": summary.get("sell_value", 0),
                            "net_value": summary.get("net_value", 0),
                        },
                        metadata={
                            "source": "data_source_manager",
                            "is_signal": abs(sentiment_score) > 0.3,
                            "signal_strength": abs(sentiment_score),
                        },
                    )
            except Exception as e:
                logger.debug(f"DataSourceManager fallback for {symbol}: {e}")

        # Fallback to specialized sentiment service for detailed analysis
        sentiment = await self._sentiment_service.analyze_sentiment(symbol, days)

        return ToolResult.create_success(output=sentiment.to_dict(),
            metadata={
                "source": "insider_activity_service",
                "is_signal": sentiment.is_signal,
                "signal_strength": sentiment.signal_strength,
                "warnings": sentiment.warnings,
            },
        )

    async def _get_recent(self, symbol: str, days: int, significant_only: bool) -> ToolResult:
        """Get recent insider transactions.

        Tries DataSourceManager first for consolidated data, then falls back
        to the DAO for detailed transaction lists.
        """
        # Try DataSourceManager first for consolidated insider data
        if self._data_source_manager and not significant_only:
            try:
                consolidated = self._data_source_manager.get_data(symbol)
                if consolidated.insider:
                    insider_data = consolidated.insider
                    transactions = insider_data.get("transactions", [])

                    if transactions:
                        # Format transactions from consolidated data
                        formatted_filings = []
                        for t in transactions:
                            formatted_filings.append(
                                {
                                    "date": t.get("filing_date") or t.get("date"),
                                    "insider": t.get("owner_name") or t.get("insider"),
                                    "title": t.get("owner_title") or t.get("title"),
                                    "type": t.get("transaction_type") or t.get("type"),
                                    "code": t.get("transaction_code") or t.get("code"),
                                    "shares": t.get("shares"),
                                    "price": t.get("price_per_share") or t.get("price"),
                                    "value": t.get("total_value") or t.get("value"),
                                    "is_director": t.get("is_director"),
                                    "is_officer": t.get("is_officer"),
                                    "is_significant": t.get("is_significant"),
                                }
                            )

                        # Calculate summary from transactions
                        total_purchases = sum(
                            t.get("total_value", 0) or t.get("value", 0)
                            for t in transactions
                            if (t.get("transaction_code") or t.get("code")) == "P"
                        )
                        total_sales = sum(
                            abs(t.get("total_value", 0) or t.get("value", 0))
                            for t in transactions
                            if (t.get("transaction_code") or t.get("code")) == "S"
                        )

                        return ToolResult.create_success(output={
                                "symbol": symbol,
                                "period_days": days,
                                "transactions": formatted_filings,
                                "summary": {
                                    "total_count": len(formatted_filings),
                                    "purchase_value": total_purchases,
                                    "sale_value": total_sales,
                                    "net_value": total_purchases - total_sales,
                                },
                            },
                            metadata={
                                "source": "data_source_manager",
                                "significant_only": significant_only,
                                "transaction_count": len(formatted_filings),
                            },
                        )
            except Exception as e:
                logger.debug(f"DataSourceManager fallback for recent {symbol}: {e}")

        # Fallback to DAO for detailed transaction data or significant_only filter
        loop = asyncio.get_event_loop()
        filings = await loop.run_in_executor(None, self._dao.get_recent_activity, symbol, days, significant_only)

        # Format for output
        formatted_filings = []
        for f in filings:
            formatted_filings.append(
                {
                    "date": f.get("filing_date"),
                    "insider": f.get("owner_name"),
                    "title": f.get("owner_title"),
                    "type": f.get("transaction_type"),
                    "code": f.get("transaction_code"),
                    "shares": f.get("shares"),
                    "price": f.get("price_per_share"),
                    "value": f.get("total_value"),
                    "is_director": f.get("is_director"),
                    "is_officer": f.get("is_officer"),
                    "is_significant": f.get("is_significant"),
                }
            )

        # Calculate summary stats
        total_purchases = sum(f.get("total_value", 0) for f in filings if f.get("transaction_code") == "P")
        total_sales = sum(abs(f.get("total_value", 0)) for f in filings if f.get("transaction_code") == "S")

        return ToolResult.create_success(output={
                "symbol": symbol,
                "period_days": days,
                "transactions": formatted_filings,
                "summary": {
                    "total_count": len(filings),
                    "purchase_value": total_purchases,
                    "sale_value": total_sales,
                    "net_value": total_purchases - total_sales,
                },
            },
            metadata={
                "source": "insider_trading_dao",
                "significant_only": significant_only,
                "transaction_count": len(filings),
            },
        )

    async def _detect_clusters(self, symbol: str, days: int) -> ToolResult:
        """Detect cluster activity."""
        clusters = await self._sentiment_service.detect_cluster_activity(symbol, days)

        cluster_data = [c.to_dict() for c in clusters]

        # Determine overall cluster signal
        has_buying_cluster = any(c.cluster_type.value == "buying_cluster" and c.is_significant for c in clusters)
        has_selling_cluster = any(c.cluster_type.value == "selling_cluster" and c.is_significant for c in clusters)

        if has_buying_cluster and not has_selling_cluster:
            signal = "bullish_cluster"
        elif has_selling_cluster and not has_buying_cluster:
            signal = "bearish_cluster"
        elif has_buying_cluster and has_selling_cluster:
            signal = "mixed_clusters"
        else:
            signal = "no_significant_clusters"

        return ToolResult.create_success(output={
                "symbol": symbol,
                "period_days": days,
                "clusters": cluster_data,
                "cluster_signal": signal,
                "significant_clusters": sum(1 for c in clusters if c.is_significant),
            },
            metadata={
                "total_clusters": len(clusters),
                "has_signal": signal != "no_significant_clusters",
            },
        )

    async def _get_key_insiders(self, symbol: str, days: int) -> ToolResult:
        """Get key insider summary."""
        summary = await self._sentiment_service.get_key_insider_summary(symbol, days)

        return ToolResult.create_success(output=summary,
            metadata={
                "key_insider_count": len(summary.get("key_insiders", [])),
            },
        )

    async def _fetch_filings(self, symbol: str, days: int) -> ToolResult:
        """Fetch fresh Form 4 filings from SEC EDGAR."""
        try:
            # Lazy load fetcher
            if self._fetcher is None:
                from investigator.infrastructure.external.sec.insider_transactions import InsiderTransactionFetcher

                self._fetcher = InsiderTransactionFetcher()

            # Fetch filings
            filings = await self._fetcher.fetch_recent_filings(symbol, days)

            if not filings:
                return ToolResult.create_success(output={
                        "symbol": symbol,
                        "filings_fetched": 0,
                        "filings_saved": 0,
                        "message": "No Form 4 filings found",
                    }
                )

            # Save to database
            loop = asyncio.get_event_loop()
            saved_count = await loop.run_in_executor(None, self._dao.save_filings_batch, filings)

            return ToolResult.create_success(output={
                    "symbol": symbol,
                    "filings_fetched": len(filings),
                    "filings_saved": saved_count,
                    "message": f"Fetched {len(filings)} filings, saved {saved_count}",
                },
                metadata={
                    "source": "SEC EDGAR",
                    "fetch_period_days": days,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching filings for {symbol}: {e}")
            return ToolResult.create_failure(f"Failed to fetch filings: {str(e)}", metadata={"symbol": symbol})

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Insider Trading Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"},
                "action": {
                    "type": "string",
                    "enum": ["sentiment", "recent", "clusters", "key_insiders", "fetch"],
                    "description": "Analysis type to perform",
                    "default": "sentiment",
                },
                "days": {"type": "integer", "description": "Analysis period in days", "default": 90},
                "significant_only": {
                    "type": "boolean",
                    "description": "For 'recent' action: filter to significant transactions only",
                    "default": False,
                },
            },
            "required": ["symbol"],
        }
