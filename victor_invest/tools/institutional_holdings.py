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

"""Institutional Holdings Tool for Victor Invest.

This tool provides access to SEC Form 13F institutional holdings data,
allowing analysis of institutional ownership patterns and changes.

Available Actions:
- holdings: Get institutional holdings for a symbol
- top_holders: Get top institutional holders
- changes: Get ownership changes over time
- institution: Get holdings for a specific institution
- search: Search for institutions by name

Example:
    tool = InstitutionalHoldingsTool()

    # Get holdings for AAPL
    result = await tool.execute(symbol="AAPL", action="holdings")

    # Get top holders
    result = await tool.execute(symbol="AAPL", action="top_holders", limit=20)

    # Get ownership changes
    result = await tool.execute(symbol="AAPL", action="changes", quarters=4)
"""

import logging
from typing import Any, Dict, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class InstitutionalHoldingsTool(BaseTool):
    """Tool for SEC Form 13F institutional holdings analysis.

    Provides CLI and agent access to institutional ownership data,
    including top holders, ownership changes, and institution searches.

    Supported actions:
    - holdings: Current institutional ownership snapshot
    - top_holders: List of largest institutional holders
    - changes: Ownership changes over multiple quarters
    - institution: Holdings for a specific institution
    - search: Search for institutions by name

    Attributes:
        name: "institutional_holdings"
        description: Tool description for agent discovery
    """

    name = "institutional_holdings"
    description = """Access SEC Form 13F institutional holdings data.

Actions:
- holdings: Get institutional ownership for a symbol (total shares, value, # institutions)
- top_holders: Get top institutional holders by value
- changes: Get quarter-over-quarter ownership changes
- institution: Get all holdings for a specific institution (by CIK)
- search: Search for institutions by name

Parameters:
- symbol: Stock ticker symbol (required for holdings, top_holders, changes)
- action: One of the actions above (default: "holdings")
- limit: Number of results to return (default: 20)
- quarters: Number of quarters for changes (default: 4)
- cik: Institution CIK for institution action
- query: Search query for search action

Investment Signals:
- Increasing institutional ownership is typically bullish
- Cluster buying by multiple institutions signals strong conviction
- Large new positions by top funds often precede price appreciation
- Sudden exits by major holders warrant caution
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Institutional Holdings Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._fetcher = None
        self._data_source_manager = None

    async def initialize(self) -> None:
        """Initialize institutional holdings fetcher and DataSourceManager."""
        try:
            from investigator.infrastructure.external.sec.institutional_holdings import (
                get_institutional_holdings_fetcher,
            )

            self._fetcher = get_institutional_holdings_fetcher()

            # Initialize DataSourceManager for consolidated data access
            try:
                from investigator.domain.services.data_sources.manager import DataSourceManager

                self._data_source_manager = DataSourceManager()
                logger.info("DataSourceManager initialized for institutional holdings")
            except ImportError as e:
                logger.warning(f"DataSourceManager not available, using fetcher only: {e}")
                self._data_source_manager = None

            self._initialized = True
            logger.info("InstitutionalHoldingsTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize InstitutionalHoldingsTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Optional[Dict[str, Any]] = None,
        action: str = "holdings",
        symbol: Optional[str] = None,
        limit: int = 20,
        quarters: int = 4,
        cik: Optional[str] = None,
        query: Optional[str] = None,
        quarter: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute institutional holdings query.

        Args:
            action: Query type:
                - "holdings": Ownership snapshot for symbol
                - "top_holders": Top institutional holders
                - "changes": Ownership changes over quarters
                - "institution": Holdings for a specific institution
                - "search": Search for institutions
            symbol: Stock ticker symbol
            limit: Maximum results to return
            quarters: Number of quarters for change analysis
            cik: Institution CIK for institution action
            query: Search query for search action
            quarter: Specific quarter (e.g., "2024-Q4")
            **kwargs: Additional parameters

        Returns:
            ToolResult with institutional holdings data
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "holdings":
                if not symbol:
                    return ToolResult.create_failure("Symbol required for holdings action")
                return await self._get_holdings(symbol, quarter)

            elif action == "top_holders":
                if not symbol:
                    return ToolResult.create_failure("Symbol required for top_holders action")
                return await self._get_top_holders(symbol, limit)

            elif action == "changes":
                if not symbol:
                    return ToolResult.create_failure("Symbol required for changes action")
                return await self._get_ownership_changes(symbol, quarters)

            elif action == "institution":
                if not cik:
                    return ToolResult.create_failure("CIK required for institution action")
                return await self._get_institution_holdings(cik, quarter)

            elif action == "search":
                if not query:
                    return ToolResult.create_failure("Query required for search action")
                return await self._search_institutions(query, limit)

            else:
                return ToolResult.create_failure(
                    f"Unknown action: {action}. Valid actions: " "holdings, top_holders, changes, institution, search"
                )

        except Exception as e:
            logger.error(f"InstitutionalHoldingsTool execute error: {e}")
            return ToolResult.create_failure(
                f"Institutional holdings query failed: {str(e)}", metadata={"action": action, "symbol": symbol}
            )

    async def _get_holdings(self, symbol: str, quarter: Optional[str] = None) -> ToolResult:
        """Get institutional holdings for a symbol.

        Uses DataSourceManager for consolidated data access when available,
        falling back to direct fetcher for specialized queries or when
        DataSourceManager is not available.
        """
        # Try DataSourceManager first for standard holdings queries (no specific quarter)
        if self._data_source_manager and quarter is None:
            try:
                consolidated = self._data_source_manager.get_data(symbol)
                if consolidated.institutional:
                    inst_data = consolidated.institutional
                    # Build response from DataSourceManager data
                    summary = inst_data.get("summary", {})
                    top_holders = inst_data.get("top_holders", [])

                    # Create ownership-like structure for signal calculation
                    ownership_proxy = type(
                        "OwnershipProxy",
                        (),
                        {
                            "num_institutions": summary.get("total_holders", 0),
                            "total_shares": summary.get("total_shares", 0),
                            "total_value": summary.get("total_value", 0),
                            "qoq_change_pct": None,  # Not available from DataSourceManager
                            "to_dict": lambda self: {
                                "symbol": symbol,
                                "num_institutions": self.num_institutions,
                                "total_shares": self.total_shares,
                                "total_value": self.total_value,
                                "top_holders": top_holders,
                                "report_date": inst_data.get("report_date"),
                            },
                        },
                    )()

                    signal = self._calculate_ownership_signal(ownership_proxy)

                    return ToolResult.create_success(output={
                            **ownership_proxy.to_dict(),
                            "investment_signal": signal,
                        },
                        metadata={
                            "source": "sec_form_13f",
                            "data_source": "data_source_manager",
                            "signal": signal["level"],
                        },
                    )
            except Exception as e:
                logger.debug(f"DataSourceManager fallback to fetcher: {e}")
                # Fall through to use fetcher

        # Fallback to direct fetcher (required for quarter-specific queries
        # or when DataSourceManager is unavailable/failed)
        ownership = await self._fetcher.get_holdings_by_symbol(symbol, quarter)

        if ownership.num_institutions == 0:
            return ToolResult.create_success(output={
                    "symbol": symbol,
                    "message": "No institutional holdings data found",
                    "num_institutions": 0,
                },
                metadata={"warnings": ["No 13F data available for this symbol"]},
            )

        # Calculate signal based on ownership
        signal = self._calculate_ownership_signal(ownership)

        return ToolResult.create_success(output={
                **ownership.to_dict(),
                "investment_signal": signal,
            },
            metadata={
                "source": "sec_form_13f",
                "data_source": "fetcher",
                "signal": signal["level"],
            },
        )

    async def _get_top_holders(self, symbol: str, limit: int = 20) -> ToolResult:
        """Get top institutional holders for a symbol.

        Uses DataSourceManager when available for consolidated access,
        falling back to direct fetcher for custom limit queries.
        """
        # Try DataSourceManager first (default limit of 20 matches DataSourceManager)
        if self._data_source_manager and limit == 20:
            try:
                consolidated = self._data_source_manager.get_data(symbol)
                if consolidated.institutional:
                    inst_data = consolidated.institutional
                    top_holders = inst_data.get("top_holders", [])

                    if top_holders:
                        # Calculate total value from top holders
                        # DataSourceManager returns value in dollars, not thousands
                        total_value_dollars = sum(h.get("value", 0) or 0 for h in top_holders)

                        return ToolResult.create_success(output={
                                "symbol": symbol,
                                "num_holders": len(top_holders),
                                "total_value_thousands": total_value_dollars / 1000,
                                "total_value_dollars": total_value_dollars,
                                "top_holders": top_holders,
                            },
                            metadata={
                                "source": "sec_form_13f",
                                "data_source": "data_source_manager",
                            },
                        )
            except Exception as e:
                logger.debug(f"DataSourceManager fallback to fetcher for top_holders: {e}")
                # Fall through to use fetcher

        # Fallback to direct fetcher (for custom limits or when DataSourceManager unavailable)
        holders = await self._fetcher.get_top_holders(symbol, limit)

        if not holders:
            return ToolResult.create_success(output={
                    "symbol": symbol,
                    "top_holders": [],
                    "message": "No institutional holders found",
                },
                metadata={"warnings": ["No 13F data available for this symbol"]},
            )

        # Calculate total value from top holders
        total_value = sum(h.get("value_thousands", 0) for h in holders)

        return ToolResult.create_success(output={
                "symbol": symbol,
                "num_holders": len(holders),
                "total_value_thousands": total_value,
                "total_value_dollars": total_value * 1000,
                "top_holders": holders,
            },
            metadata={
                "source": "sec_form_13f",
                "data_source": "fetcher",
            },
        )

    async def _get_ownership_changes(self, symbol: str, quarters: int = 4) -> ToolResult:
        """Get ownership changes over multiple quarters."""
        changes = await self._fetcher.get_ownership_changes(symbol, quarters)

        if not changes:
            return ToolResult.create_success(output={
                    "symbol": symbol,
                    "changes": [],
                    "message": "No historical ownership data found",
                },
                metadata={"warnings": ["Insufficient 13F history for this symbol"]},
            )

        # Analyze trend
        trend = self._analyze_ownership_trend(changes)

        return ToolResult.create_success(output={
                "symbol": symbol,
                "quarters_analyzed": len(changes),
                "changes": changes,
                "trend_analysis": trend,
            },
            metadata={
                "source": "sec_form_13f",
                "trend": trend["direction"],
            },
        )

    async def _get_institution_holdings(self, cik: str, quarter: Optional[str] = None) -> ToolResult:
        """Get holdings for a specific institution."""
        holdings = await self._fetcher.get_institution_holdings(cik, quarter)

        if not holdings:
            return ToolResult.create_success(output={
                    "cik": cik,
                    "holdings": [],
                    "message": "No holdings found for this institution",
                }
            )

        # Summarize holdings
        total_value = sum(h.value for h in holdings)
        total_positions = len(holdings)

        holdings_data = [
            {
                "cusip": h.cusip,
                "symbol": h.symbol,
                "issuer_name": h.issuer_name,
                "shares": h.shares,
                "value_thousands": h.value,
                "value_dollars": h.value_dollars,
                "investment_discretion": h.investment_discretion,
            }
            for h in holdings[:50]  # Limit to top 50
        ]

        return ToolResult.create_success(output={
                "cik": cik,
                "total_positions": total_positions,
                "total_value_thousands": total_value,
                "total_value_dollars": total_value * 1000,
                "top_holdings": holdings_data,
            },
            metadata={
                "source": "sec_form_13f",
                "positions_shown": len(holdings_data),
            },
        )

    async def _search_institutions(self, query: str, limit: int = 20) -> ToolResult:
        """Search for institutions by name."""
        institutions = await self._fetcher.search_institutions(query, limit)

        if not institutions:
            return ToolResult.create_success(output={
                    "query": query,
                    "institutions": [],
                    "message": f"No institutions found matching '{query}'",
                }
            )

        institution_data = [
            {
                "cik": inst.cik,
                "name": inst.name,
                "latest_filing": str(inst.filing_date) if inst.filing_date else None,
            }
            for inst in institutions
        ]

        return ToolResult.create_success(output={
                "query": query,
                "num_results": len(institution_data),
                "institutions": institution_data,
            },
            metadata={
                "source": "sec_form_13f",
            },
        )

    def _calculate_ownership_signal(self, ownership) -> Dict[str, Any]:
        """Calculate investment signal from ownership data.

        Args:
            ownership: InstitutionalOwnership object

        Returns:
            Signal dict with level and interpretation
        """
        signal = {
            "level": "neutral",
            "interpretation": "",
            "factors": [],
        }

        # Check QoQ change
        if ownership.qoq_change_pct is not None:
            if ownership.qoq_change_pct > 10:
                signal["level"] = "bullish"
                signal["factors"].append(f"Strong institutional buying (+{ownership.qoq_change_pct:.1f}% QoQ)")
            elif ownership.qoq_change_pct > 5:
                signal["level"] = "moderately_bullish"
                signal["factors"].append(f"Moderate institutional buying (+{ownership.qoq_change_pct:.1f}% QoQ)")
            elif ownership.qoq_change_pct < -10:
                signal["level"] = "bearish"
                signal["factors"].append(f"Strong institutional selling ({ownership.qoq_change_pct:.1f}% QoQ)")
            elif ownership.qoq_change_pct < -5:
                signal["level"] = "moderately_bearish"
                signal["factors"].append(f"Moderate institutional selling ({ownership.qoq_change_pct:.1f}% QoQ)")

        # Check number of institutions
        if ownership.num_institutions > 100:
            signal["factors"].append(f"High institutional interest ({ownership.num_institutions} holders)")
        elif ownership.num_institutions < 10:
            signal["factors"].append(f"Low institutional interest ({ownership.num_institutions} holders)")

        # Set interpretation
        if signal["level"] == "bullish":
            signal["interpretation"] = "Strong institutional accumulation suggests positive outlook"
        elif signal["level"] == "bearish":
            signal["interpretation"] = "Significant institutional distribution warrants caution"
        elif signal["level"] == "moderately_bullish":
            signal["interpretation"] = "Moderate institutional buying indicates growing interest"
        elif signal["level"] == "moderately_bearish":
            signal["interpretation"] = "Moderate institutional selling may indicate reduced conviction"
        else:
            signal["interpretation"] = "Institutional ownership relatively stable"

        return signal

    def _analyze_ownership_trend(self, changes: list) -> Dict[str, Any]:
        """Analyze ownership trend from quarterly changes.

        Args:
            changes: List of quarterly change data

        Returns:
            Trend analysis dict
        """
        if len(changes) < 2:
            return {
                "direction": "insufficient_data",
                "interpretation": "Need at least 2 quarters for trend analysis",
            }

        # Calculate average change
        changes_with_pct = [c for c in changes if c.get("qoq_change_pct") is not None]
        if not changes_with_pct:
            return {
                "direction": "stable",
                "interpretation": "No significant ownership changes detected",
            }

        avg_change = sum(c["qoq_change_pct"] for c in changes_with_pct) / len(changes_with_pct)

        # Count up vs down quarters
        up_quarters = sum(1 for c in changes_with_pct if c["qoq_change_pct"] > 0)
        down_quarters = len(changes_with_pct) - up_quarters

        # Determine trend
        if avg_change > 5 and up_quarters > down_quarters:
            direction = "accumulating"
            interpretation = (
                f"Consistent institutional accumulation "
                f"(avg +{avg_change:.1f}% per quarter, {up_quarters}/{len(changes_with_pct)} up quarters)"
            )
        elif avg_change < -5 and down_quarters > up_quarters:
            direction = "distributing"
            interpretation = (
                f"Consistent institutional distribution "
                f"(avg {avg_change:.1f}% per quarter, {down_quarters}/{len(changes_with_pct)} down quarters)"
            )
        elif abs(avg_change) <= 5:
            direction = "stable"
            interpretation = f"Stable institutional ownership " f"(avg {avg_change:+.1f}% per quarter)"
        else:
            direction = "mixed"
            interpretation = f"Mixed institutional activity " f"({up_quarters} up, {down_quarters} down quarters)"

        # Total change over period
        first_shares = changes[0].get("total_shares", 0)
        last_shares = changes[-1].get("total_shares", 0)
        total_change_pct = ((last_shares - first_shares) / first_shares * 100) if first_shares else 0

        return {
            "direction": direction,
            "interpretation": interpretation,
            "avg_quarterly_change_pct": round(avg_change, 2),
            "total_change_pct": round(total_change_pct, 2),
            "up_quarters": up_quarters,
            "down_quarters": down_quarters,
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Institutional Holdings Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["holdings", "top_holders", "changes", "institution", "search"],
                    "description": "Type of institutional holdings query",
                    "default": "holdings",
                },
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "limit": {"type": "integer", "description": "Maximum results to return", "default": 20},
                "quarters": {"type": "integer", "description": "Number of quarters for change analysis", "default": 4},
                "cik": {"type": "string", "description": "Institution CIK for institution action"},
                "query": {"type": "string", "description": "Search query for search action"},
                "quarter": {"type": "string", "description": "Specific quarter (e.g., '2024-Q4')"},
            },
            "required": [],
        }
