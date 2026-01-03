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

"""Treasury Data Tool for Victor Invest.

This tool provides access to Treasury yield curve data, market regime analysis,
and recession indicators via CLI and agent interfaces.

Available Actions:
- curve: Get current yield curve with all maturities
- spread: Get yield spread analysis (10Y-2Y, 10Y-3M)
- regime: Get market regime assessment
- recession: Get recession probability and economic phase
- history: Get historical yield or spread data

Example:
    tool = TreasuryDataTool()

    # Get yield curve
    result = await tool.execute(action="curve")

    # Get market regime
    result = await tool.execute(action="regime")

    # Get recession assessment
    result = await tool.execute(action="recession")
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TreasuryDataTool(BaseTool):
    """Tool for treasury yield curve and market regime analysis.

    Provides CLI and agent access to treasury yields, yield curve shape
    analysis, and recession probability indicators.

    Supported actions:
    - curve: Current yield curve snapshot
    - spread: Yield spread analysis
    - regime: Market regime assessment
    - recession: Recession probability
    - history: Historical data
    - summary: Complete market regime summary

    Attributes:
        name: "treasury_data"
        description: Tool description for agent discovery
    """

    name = "treasury_data"
    description = """Access Treasury yield curve data and market regime indicators.

Actions:
- curve: Get current yield curve (1m to 30y maturities)
- spread: Get yield spread analysis (10Y-2Y, 10Y-3M inversion detection)
- regime: Get yield curve shape and investment signal
- recession: Get recession probability and economic phase assessment
- history: Get historical yields or spreads
- summary: Get comprehensive market regime summary

Parameters:
- action: One of the actions above (default: "curve")
- days: For history action, number of days (default: 365)
- maturity: For history action, specific maturity (default: "10y")

Returns yield curve data, spread analysis, recession probability,
and investment recommendations based on current market regime.
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Treasury Data Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._treasury_client = None
        self._nyfed_client = None
        self._yield_analyzer = None
        self._recession_indicator = None

    async def initialize(self) -> None:
        """Initialize treasury and market regime services."""
        try:
            from investigator.infrastructure.external.treasury import get_treasury_client
            from investigator.infrastructure.external.nyfed import get_nyfed_client
            from investigator.domain.services.market_regime import (
                get_yield_curve_analyzer,
                get_recession_indicator,
            )

            self._treasury_client = get_treasury_client()
            self._nyfed_client = get_nyfed_client()
            self._yield_analyzer = get_yield_curve_analyzer()
            self._recession_indicator = get_recession_indicator()

            self._initialized = True
            logger.info("TreasuryDataTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TreasuryDataTool: {e}")
            raise

    async def execute(
        self,
        action: str = "curve",
        days: int = 365,
        maturity: str = "10y",
        **kwargs
    ) -> ToolResult:
        """Execute treasury data query.

        Args:
            action: Query type:
                - "curve": Current yield curve
                - "spread": Yield spread analysis
                - "regime": Market regime from yield curve
                - "recession": Recession probability
                - "history": Historical data
                - "summary": Complete market summary
            days: For history, number of days
            maturity: For history, specific maturity
            **kwargs: Additional parameters

        Returns:
            ToolResult with treasury/market regime data
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "curve":
                return await self._get_yield_curve()
            elif action == "spread":
                return await self._get_spread_analysis()
            elif action == "regime":
                return await self._get_market_regime()
            elif action == "recession":
                return await self._get_recession_assessment()
            elif action == "history":
                return await self._get_history(days, maturity)
            elif action == "summary":
                return await self._get_summary()
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "curve, spread, regime, recession, history, summary"
                )

        except Exception as e:
            logger.error(f"TreasuryDataTool execute error: {e}")
            return ToolResult.error_result(
                f"Treasury data query failed: {str(e)}",
                metadata={"action": action}
            )

    async def _get_yield_curve(self) -> ToolResult:
        """Get current yield curve."""
        curve = await self._treasury_client.get_yield_curve()

        if curve is None:
            return ToolResult.error_result("Could not retrieve yield curve data")

        return ToolResult.success_result(
            data=curve.to_dict(),
            metadata={
                "source": "treasury.gov",
                "curve_shape": curve.curve_shape,
            }
        )

    async def _get_spread_analysis(self) -> ToolResult:
        """Get yield spread analysis."""
        curve = await self._treasury_client.get_yield_curve()

        if curve is None:
            return ToolResult.error_result("Could not retrieve yield curve data")

        # Get yield curve analysis for shape
        analysis = await self._yield_analyzer.analyze()

        return ToolResult.success_result(
            data={
                "date": str(curve.date),
                "spreads": {
                    "10y_2y_bps": curve.spread_10y_2y,
                    "10y_3m_bps": curve.spread_10y_3m,
                },
                "inversion": {
                    "is_inverted": curve.is_inverted,
                    "is_deeply_inverted": curve.is_deeply_inverted,
                    "days_inverted": analysis.days_inverted if analysis else 0,
                },
                "curve_shape": curve.curve_shape,
                "investment_signal": analysis.investment_signal.value if analysis else "unknown",
            },
            metadata={
                "historical_avg_spread_bps": 90,
            }
        )

    async def _get_market_regime(self) -> ToolResult:
        """Get market regime from yield curve analysis."""
        analysis = await self._yield_analyzer.analyze()

        return ToolResult.success_result(
            data=analysis.to_dict(),
            warnings=analysis.warnings,
            metadata={
                "source": "yield_curve_analyzer",
                "curve_shape": analysis.shape.value,
            }
        )

    async def _get_recession_assessment(self) -> ToolResult:
        """Get recession probability and economic phase."""
        assessment = await self._recession_indicator.assess()

        return ToolResult.success_result(
            data=assessment.to_dict(),
            warnings=assessment.warnings,
            metadata={
                "economic_phase": assessment.phase.value,
                "investment_posture": assessment.investment_posture.value,
                "confidence": assessment.confidence,
            }
        )

    async def _get_history(self, days: int, maturity: str) -> ToolResult:
        """Get historical yield data."""
        history = await self._treasury_client.get_yield_history(days, maturity)

        if not history:
            return ToolResult.error_result(
                f"Could not retrieve historical data for {maturity}"
            )

        # Calculate summary statistics
        yields = [h.get('yield') for h in history if h.get('yield') is not None]
        if yields:
            avg_yield = sum(yields) / len(yields)
            min_yield = min(yields)
            max_yield = max(yields)
            current_yield = yields[0] if yields else None
        else:
            avg_yield = min_yield = max_yield = current_yield = None

        return ToolResult.success_result(
            data={
                "maturity": maturity,
                "period_days": days,
                "data_points": len(history),
                "history": history[:30],  # Return last 30 for display
                "summary": {
                    "current": current_yield,
                    "average": round(avg_yield, 4) if avg_yield else None,
                    "min": min_yield,
                    "max": max_yield,
                },
            },
            metadata={
                "full_data_points": len(history),
            }
        )

    async def _get_summary(self) -> ToolResult:
        """Get comprehensive market regime summary."""
        summary = await self._recession_indicator.get_market_regime_summary()

        return ToolResult.success_result(
            data=summary,
            metadata={
                "source": "market_regime_services",
                "includes": ["yield_curve", "recession", "sector_recommendations"],
            }
        )

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Treasury Data Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["curve", "spread", "regime", "recession", "history", "summary"],
                    "description": "Type of treasury data query",
                    "default": "curve"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days for historical data",
                    "default": 365
                },
                "maturity": {
                    "type": "string",
                    "description": "Maturity for historical data (e.g., '10y', '2y')",
                    "default": "10y"
                }
            },
            "required": []
        }
