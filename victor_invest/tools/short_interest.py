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

"""Short Interest Tool for Victor Invest.

This tool provides access to FINRA short interest data, including
current short interest, historical trends, and squeeze risk assessment.

Available Actions:
- current: Get current short interest for a symbol
- history: Get historical short interest data
- volume: Get daily short volume data
- squeeze: Calculate short squeeze risk
- most_shorted: Get list of most shorted stocks

Example:
    tool = ShortInterestTool()

    # Get current short interest
    result = await tool.execute(symbol="GME", action="current")

    # Get short interest history
    result = await tool.execute(symbol="AMC", action="history", periods=12)

    # Calculate squeeze risk
    result = await tool.execute(symbol="TSLA", action="squeeze")
"""

import logging
from typing import Any, Dict, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ShortInterestTool(BaseTool):
    """Tool for FINRA short interest analysis.

    Provides CLI and agent access to short interest data, including
    current levels, historical trends, and squeeze risk assessment.

    Supported actions:
    - current: Current short interest snapshot
    - history: Historical short interest over multiple periods
    - volume: Daily short volume data
    - squeeze: Short squeeze risk assessment
    - most_shorted: List of most shorted stocks

    Attributes:
        name: "short_interest"
        description: Tool description for agent discovery
    """

    name = "short_interest"
    description = """Access FINRA short interest and short volume data.

Actions:
- current: Get current short interest for a symbol (shares short, days to cover, % of float)
- history: Get historical short interest over multiple periods
- volume: Get daily short volume data
- squeeze: Calculate short squeeze risk score and assessment
- most_shorted: Get list of most shorted stocks

Parameters:
- symbol: Stock ticker symbol (required for current, history, volume, squeeze)
- action: One of the actions above (default: "current")
- periods: Number of bi-monthly periods for history (default: 12)
- days: Number of trading days for volume (default: 30)
- limit: Number of stocks for most_shorted (default: 20)

Investment Signals:
- High short interest (>10% float): Potential squeeze or strong bearish sentiment
- Days to cover >5: Extended squeeze potential if covering begins
- Rising short interest: Increasing bearish conviction
- Falling short interest: Short covering, potentially bullish catalyst
- Short ratio spike: Contrarian buy signal if fundamentals strong
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Short Interest Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._fetcher = None
        self._data_source_manager = None

    async def initialize(self) -> None:
        """Initialize short interest fetcher and DataSourceManager."""
        try:
            from investigator.infrastructure.external.finra.short_interest import (
                get_short_interest_fetcher,
            )

            self._fetcher = get_short_interest_fetcher()

            # Initialize DataSourceManager for unified data access
            try:
                from investigator.domain.services.data_sources.manager import DataSourceManager
                self._data_source_manager = DataSourceManager()
                logger.debug("DataSourceManager initialized for short interest")
            except ImportError as e:
                logger.warning(f"DataSourceManager not available, using fetcher only: {e}")
                self._data_source_manager = None

            self._initialized = True
            logger.info("ShortInterestTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ShortInterestTool: {e}")
            raise

    async def execute(
        self,
        action: str = "current",
        symbol: Optional[str] = None,
        periods: int = 12,
        days: int = 30,
        limit: int = 20,
        **kwargs
    ) -> ToolResult:
        """Execute short interest query.

        Args:
            action: Query type:
                - "current": Current short interest
                - "history": Historical short interest
                - "volume": Daily short volume
                - "squeeze": Squeeze risk assessment
                - "most_shorted": Most shorted stocks
            symbol: Stock ticker symbol
            periods: Number of bi-monthly periods for history
            days: Number of trading days for volume
            limit: Number of stocks for most_shorted
            **kwargs: Additional parameters

        Returns:
            ToolResult with short interest data
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "current":
                if not symbol:
                    return ToolResult.error_result("Symbol required for current action")
                return await self._get_current(symbol)

            elif action == "history":
                if not symbol:
                    return ToolResult.error_result("Symbol required for history action")
                return await self._get_history(symbol, periods)

            elif action == "volume":
                if not symbol:
                    return ToolResult.error_result("Symbol required for volume action")
                return await self._get_volume(symbol, days)

            elif action == "squeeze":
                if not symbol:
                    return ToolResult.error_result("Symbol required for squeeze action")
                return await self._get_squeeze_risk(symbol)

            elif action == "most_shorted":
                return await self._get_most_shorted(limit)

            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "current, history, volume, squeeze, most_shorted"
                )

        except Exception as e:
            logger.error(f"ShortInterestTool execute error: {e}")
            return ToolResult.error_result(
                f"Short interest query failed: {str(e)}",
                metadata={"action": action, "symbol": symbol}
            )

    async def _get_current(self, symbol: str) -> ToolResult:
        """Get current short interest for a symbol.

        Uses DataSourceManager as the primary source for consolidated data.
        Falls back to direct fetcher if DataSourceManager is unavailable.
        """
        data = None
        source = "finra"

        # Try DataSourceManager first for unified data access
        if self._data_source_manager:
            try:
                consolidated = self._data_source_manager.get_data(symbol)
                if consolidated.short_interest:
                    current = consolidated.short_interest.get("current", {})
                    if current:
                        # Convert DataSourceManager format to tool format
                        data = self._convert_consolidated_to_data(symbol, current)
                        source = "data_source_manager"
                        logger.debug(f"Using DataSourceManager data for {symbol}")
            except Exception as e:
                logger.warning(f"DataSourceManager failed for {symbol}: {e}, falling back to fetcher")

        # Fall back to direct fetcher if DataSourceManager didn't provide data
        if not data:
            data = await self._fetcher.get_short_interest(symbol)
            source = "finra"

        if not data:
            return ToolResult.success_result(
                data={
                    "symbol": symbol.upper(),
                    "message": "No short interest data found",
                },
                warnings=["No FINRA short interest data available for this symbol"]
            )

        # Calculate signal
        signal = self._calculate_signal(data)

        return ToolResult.success_result(
            data={
                **data.to_dict(),
                "signal": signal,
            },
            metadata={
                "source": source,
                "signal_level": signal["level"],
            }
        )

    async def _get_history(self, symbol: str, periods: int) -> ToolResult:
        """Get historical short interest data."""
        history = await self._fetcher.get_short_interest_history(symbol, periods)

        if not history:
            return ToolResult.success_result(
                data={
                    "symbol": symbol.upper(),
                    "history": [],
                    "message": "No historical short interest data found",
                },
                warnings=["Insufficient historical data for this symbol"]
            )

        # Calculate trend
        trend = self._analyze_trend(history)

        return ToolResult.success_result(
            data={
                "symbol": symbol.upper(),
                "periods": len(history),
                "history": [h.to_dict() for h in history],
                "trend_analysis": trend,
            },
            metadata={
                "source": "finra",
                "trend": trend["direction"],
            }
        )

    async def _get_volume(self, symbol: str, days: int) -> ToolResult:
        """Get daily short volume data."""
        volume = await self._fetcher.get_short_volume(symbol, days)

        if not volume:
            return ToolResult.success_result(
                data={
                    "symbol": symbol.upper(),
                    "volume": [],
                    "message": "No short volume data found",
                },
                warnings=["No daily short volume data available for this symbol"]
            )

        # Calculate average short volume ratio
        avg_short_pct = sum(v.short_percent for v in volume) / len(volume) if volume else 0

        return ToolResult.success_result(
            data={
                "symbol": symbol.upper(),
                "days": len(volume),
                "avg_short_percent": round(avg_short_pct, 2),
                "volume": [v.to_dict() for v in volume[:20]],  # Limit to 20 most recent
            },
            metadata={
                "source": "finra",
                "total_days": len(volume),
            }
        )

    async def _get_squeeze_risk(self, symbol: str) -> ToolResult:
        """Calculate short squeeze risk assessment."""
        risk = await self._fetcher.calculate_squeeze_risk(symbol)

        return ToolResult.success_result(
            data=risk.to_dict(),
            metadata={
                "source": "finra",
                "risk_level": risk.risk_level,
            }
        )

    async def _get_most_shorted(self, limit: int) -> ToolResult:
        """Get list of most shorted stocks."""
        stocks = await self._fetcher.get_most_shorted(limit)

        if not stocks:
            return ToolResult.success_result(
                data={
                    "stocks": [],
                    "message": "No most shorted data available",
                }
            )

        return ToolResult.success_result(
            data={
                "count": len(stocks),
                "stocks": stocks,
            },
            metadata={
                "source": "finra",
            }
        )

    def _convert_consolidated_to_data(self, symbol: str, current: Dict[str, Any]):
        """Convert DataSourceManager format to ShortInterestData.

        Args:
            symbol: Stock ticker symbol
            current: Current short interest dict from DataSourceManager

        Returns:
            ShortInterestData object compatible with existing signal calculation
        """
        from investigator.infrastructure.external.finra.short_interest import (
            ShortInterestData,
        )
        from datetime import date as dt_date

        # Parse settlement date
        settlement_date = dt_date.today()
        if current.get("date"):
            try:
                if isinstance(current["date"], str):
                    settlement_date = dt_date.fromisoformat(current["date"])
                else:
                    settlement_date = current["date"]
            except (ValueError, TypeError):
                pass

        return ShortInterestData(
            symbol=symbol.upper(),
            settlement_date=settlement_date,
            short_interest=int(current.get("short_interest", 0) or 0),
            avg_daily_volume=int(current.get("avg_volume", 0) or 0),
            days_to_cover=float(current.get("days_to_cover", 0.0) or 0.0),
            short_percent_float=float(current.get("short_pct_float") or 0.0) if current.get("short_pct_float") else None,
            short_percent_outstanding=None,  # Not available from DataSourceManager
            previous_short_interest=None,  # Not available from DataSourceManager
            change_from_previous=None,
            change_percent=None,
        )

    def _calculate_signal(self, data) -> Dict[str, Any]:
        """Calculate investment signal from short interest data.

        Args:
            data: ShortInterestData object

        Returns:
            Signal dict with level and interpretation
        """
        signal = {
            "level": "neutral",
            "interpretation": "",
            "factors": [],
        }

        # Check short percent of float
        if data.short_percent_float:
            spf = data.short_percent_float
            if spf >= 25:
                signal["level"] = "very_high_short"
                signal["factors"].append(f"Very high short interest: {spf:.1f}% of float")
            elif spf >= 15:
                signal["level"] = "high_short"
                signal["factors"].append(f"High short interest: {spf:.1f}% of float")
            elif spf >= 10:
                signal["level"] = "elevated_short"
                signal["factors"].append(f"Elevated short interest: {spf:.1f}% of float")
            elif spf <= 3:
                signal["factors"].append(f"Low short interest: {spf:.1f}% of float")

        # Check days to cover
        dtc = data.days_to_cover
        if dtc >= 7:
            signal["factors"].append(f"High days to cover: {dtc:.1f} days")
            if signal["level"] == "neutral":
                signal["level"] = "elevated_short"
        elif dtc >= 5:
            signal["factors"].append(f"Elevated days to cover: {dtc:.1f} days")

        # Check trend
        if data.change_percent:
            if data.change_percent > 20:
                signal["factors"].append(f"Rapidly increasing: +{data.change_percent:.1f}%")
                if "high" not in signal["level"]:
                    signal["level"] = "increasing_short"
            elif data.change_percent > 10:
                signal["factors"].append(f"Increasing: +{data.change_percent:.1f}%")
            elif data.change_percent < -15:
                signal["factors"].append(f"Short covering: {data.change_percent:.1f}%")
                signal["level"] = "covering"
            elif data.change_percent < -5:
                signal["factors"].append(f"Slight covering: {data.change_percent:.1f}%")

        # Set interpretation
        if "very_high" in signal["level"]:
            signal["interpretation"] = (
                "Very high short interest indicates strong bearish positioning. "
                "Potential squeeze candidate if positive catalyst emerges."
            )
        elif "high_short" in signal["level"]:
            signal["interpretation"] = (
                "High short interest suggests significant bearish sentiment. "
                "Watch for squeeze potential or continued weakness."
            )
        elif "elevated" in signal["level"]:
            signal["interpretation"] = (
                "Elevated short interest warrants monitoring. "
                "Could indicate informed bearish view or squeeze buildup."
            )
        elif signal["level"] == "covering":
            signal["interpretation"] = (
                "Active short covering in progress. "
                "Could support near-term price appreciation."
            )
        elif signal["level"] == "increasing_short":
            signal["interpretation"] = (
                "Short interest increasing rapidly. "
                "Bears are building positions - watch for fundamental concerns."
            )
        else:
            signal["interpretation"] = (
                "Normal short interest levels. "
                "No significant short-driven dynamics expected."
            )

        return signal

    def _analyze_trend(self, history: list) -> Dict[str, Any]:
        """Analyze short interest trend from history.

        Args:
            history: List of ShortInterestData objects

        Returns:
            Trend analysis dict
        """
        if len(history) < 2:
            return {
                "direction": "insufficient_data",
                "interpretation": "Need at least 2 periods for trend analysis",
            }

        # Calculate changes
        changes = [h.change_percent for h in history if h.change_percent is not None]
        if not changes:
            return {
                "direction": "stable",
                "interpretation": "No significant changes detected",
            }

        avg_change = sum(changes) / len(changes)

        # Calculate total change
        first_si = history[0].short_interest
        last_si = history[-1].short_interest
        total_change_pct = ((last_si - first_si) / first_si * 100) if first_si else 0

        # Count increases vs decreases
        increases = sum(1 for c in changes if c > 0)
        decreases = len(changes) - increases

        # Determine trend
        if avg_change > 10 and increases > decreases:
            direction = "rapidly_increasing"
            interpretation = (
                f"Short interest rapidly increasing ({avg_change:+.1f}% avg per period). "
                f"Bears are aggressively building positions."
            )
        elif avg_change > 5:
            direction = "increasing"
            interpretation = (
                f"Short interest steadily increasing ({avg_change:+.1f}% avg per period). "
                f"Growing bearish sentiment."
            )
        elif avg_change < -10 and decreases > increases:
            direction = "rapidly_decreasing"
            interpretation = (
                f"Short interest rapidly decreasing ({avg_change:+.1f}% avg per period). "
                f"Significant short covering underway."
            )
        elif avg_change < -5:
            direction = "decreasing"
            interpretation = (
                f"Short interest steadily decreasing ({avg_change:+.1f}% avg per period). "
                f"Bears reducing positions."
            )
        else:
            direction = "stable"
            interpretation = (
                f"Short interest relatively stable ({avg_change:+.1f}% avg per period). "
                f"No significant directional bias from shorts."
            )

        return {
            "direction": direction,
            "interpretation": interpretation,
            "avg_change_per_period": round(avg_change, 2),
            "total_change_pct": round(total_change_pct, 2),
            "periods_increasing": increases,
            "periods_decreasing": decreases,
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Short Interest Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["current", "history", "volume", "squeeze", "most_shorted"],
                    "description": "Type of short interest query",
                    "default": "current"
                },
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of bi-monthly periods for history",
                    "default": 12
                },
                "days": {
                    "type": "integer",
                    "description": "Number of trading days for volume",
                    "default": 30
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of stocks for most_shorted",
                    "default": 20
                }
            },
            "required": []
        }
