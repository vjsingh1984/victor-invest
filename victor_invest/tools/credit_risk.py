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

"""Credit Risk Tool for Victor Invest.

This tool wraps the credit risk service to provide credit risk assessment
via CLI and agent interfaces.

Available Models:
- Altman Z-Score: Bankruptcy prediction (Z > 2.99 = Safe, Z < 1.81 = Distress)
- Beneish M-Score: Earnings manipulation detection (M > -1.78 = Likely manipulation)
- Piotroski F-Score: Financial strength (8-9 = Strong, 0-4 = Weak)
- Composite: Combined assessment with distress tier and valuation discount

Example:
    tool = CreditRiskTool()

    # Get all credit risk scores
    result = await tool.execute(symbol="AAPL", action="all")

    # Get specific score
    result = await tool.execute(symbol="AAPL", action="altman")

    # Get composite assessment
    result = await tool.execute(symbol="AAPL", action="composite")
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class CreditRiskTool(BaseTool):
    """Tool for credit risk assessment.

    Provides CLI and agent access to credit risk models including
    Altman Z-Score, Beneish M-Score, Piotroski F-Score, and composite
    distress assessment.

    Supported actions:
    - all: Calculate all credit risk scores
    - altman: Altman Z-Score bankruptcy prediction
    - beneish: Beneish M-Score earnings manipulation
    - piotroski: Piotroski F-Score financial strength
    - composite: Composite distress tier with valuation discount

    Attributes:
        name: "credit_risk"
        description: Tool description for agent discovery
    """

    name = "credit_risk"
    description = """Assess credit risk using multiple financial models.

Actions:
- all: Get comprehensive credit risk assessment with all scores
- altman: Altman Z-Score for bankruptcy prediction
- beneish: Beneish M-Score for earnings manipulation detection
- piotroski: Piotroski F-Score for financial strength
- composite: Unified distress tier with recommended valuation discount

Parameters:
- symbol: Stock ticker symbol (required)
- action: One of the actions above (default: "composite")

Returns distress tier (1-5), valuation discount (0-50%), and detailed score breakdown.
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Credit Risk Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._service = None
        self._sec_tool = None

    async def initialize(self) -> None:
        """Initialize credit risk service and dependencies."""
        try:
            from investigator.domain.services.credit_risk import (
                get_credit_risk_service
            )
            from victor_invest.tools.sec_filing import SECFilingTool

            self._service = get_credit_risk_service()
            self._sec_tool = SECFilingTool(config=self.config)
            await self._sec_tool.initialize()

            self._initialized = True
            logger.info("CreditRiskTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CreditRiskTool: {e}")
            raise

    async def execute(
        self,
        symbol: str,
        action: str = "composite",
        **kwargs
    ) -> ToolResult:
        """Execute credit risk assessment.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
            action: Assessment type:
                - "all": All credit risk scores
                - "altman": Altman Z-Score only
                - "beneish": Beneish M-Score only
                - "piotroski": Piotroski F-Score only
                - "composite": Composite distress assessment
            **kwargs: Additional parameters

        Returns:
            ToolResult with credit risk assessment
        """
        try:
            await self.ensure_initialized()

            symbol = symbol.upper().strip()
            if not symbol:
                return ToolResult.error_result("Symbol is required")

            action = action.lower().strip()

            # Get SEC financial data
            sec_result = await self._sec_tool.execute(
                symbol=symbol,
                action="extract_metrics"
            )

            if not sec_result.success:
                return ToolResult.error_result(
                    f"Failed to get financial data for {symbol}: {sec_result.error}",
                    metadata={"symbol": symbol}
                )

            # Transform to FinancialData format
            from investigator.domain.services.credit_risk.protocols import FinancialData
            fin_data = self._transform_sec_data(symbol, sec_result.data)

            # Execute requested action
            if action == "all":
                return await self._calculate_all(fin_data)
            elif action == "altman":
                return await self._calculate_altman(fin_data)
            elif action == "beneish":
                return await self._calculate_beneish(fin_data)
            elif action == "piotroski":
                return await self._calculate_piotroski(fin_data)
            elif action == "composite":
                return await self._calculate_composite(fin_data)
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "all, altman, beneish, piotroski, composite"
                )

        except Exception as e:
            logger.error(f"CreditRiskTool execute error for {symbol}: {e}")
            return ToolResult.error_result(
                f"Credit risk assessment failed: {str(e)}",
                metadata={"symbol": symbol, "action": action}
            )

    def _transform_sec_data(self, symbol: str, sec_data: Dict[str, Any]):
        """Transform SEC tool data to FinancialData format."""
        from investigator.domain.services.credit_risk.protocols import FinancialData

        bs = sec_data.get("balance_sheet", {})
        is_ = sec_data.get("income_statement", {})
        cf = sec_data.get("cash_flow", {})
        ratios = sec_data.get("ratios", {})

        return FinancialData(
            symbol=symbol,
            fiscal_year=sec_data.get("fiscal_year"),
            fiscal_period=sec_data.get("fiscal_period"),
            # Balance Sheet - Assets
            total_assets=bs.get("total_assets"),
            current_assets=bs.get("current_assets"),
            cash_and_equivalents=bs.get("cash_and_equivalents"),
            accounts_receivable=bs.get("accounts_receivable"),
            inventory=bs.get("inventory"),
            property_plant_equipment=bs.get("property_plant_equipment"),
            # Balance Sheet - Liabilities & Equity
            total_liabilities=bs.get("total_liabilities"),
            current_liabilities=bs.get("current_liabilities"),
            total_debt=bs.get("total_debt"),
            long_term_debt=bs.get("long_term_debt"),
            short_term_debt=bs.get("short_term_debt"),
            stockholders_equity=bs.get("stockholders_equity"),
            retained_earnings=bs.get("retained_earnings"),
            # Income Statement
            revenue=is_.get("revenue") or is_.get("revenues") or is_.get("total_revenue"),
            gross_profit=is_.get("gross_profit"),
            operating_income=is_.get("operating_income"),
            net_income=is_.get("net_income") or is_.get("net_income_loss"),
            cost_of_revenue=is_.get("cost_of_revenue"),
            sga_expense=is_.get("sga_expense"),
            depreciation_amortization=is_.get("depreciation_amortization"),
            interest_expense=is_.get("interest_expense"),
            # Cash Flow
            operating_cash_flow=cf.get("operating_cash_flow"),
            capital_expenditures=cf.get("capital_expenditures") or cf.get("capex"),
            # Market Data
            shares_outstanding=bs.get("shares_outstanding") or ratios.get("shares_outstanding"),
        )

    async def _calculate_all(self, fin_data) -> ToolResult:
        """Calculate all credit risk scores."""
        loop = asyncio.get_event_loop()
        assessment = await loop.run_in_executor(
            None,
            self._service.calculate_all,
            fin_data
        )

        return ToolResult.success_result(
            data=assessment.to_dict(),
            metadata={
                "source": "credit_risk_service",
                "data_quality": assessment.data_quality,
            }
        )

    async def _calculate_altman(self, fin_data) -> ToolResult:
        """Calculate Altman Z-Score."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._service.calculate_altman,
            fin_data
        )

        return ToolResult.success_result(
            data=result.to_dict(),
            warnings=result.warnings,
            metadata={
                "score_name": "Altman Z-Score",
                "model": result.model_used.value if result.model_used else None,
            }
        )

    async def _calculate_beneish(self, fin_data) -> ToolResult:
        """Calculate Beneish M-Score."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._service.calculate_beneish,
            fin_data
        )

        return ToolResult.success_result(
            data=result.to_dict(),
            warnings=result.warnings,
            metadata={
                "score_name": "Beneish M-Score",
                "requires_prior_period": True,
            }
        )

    async def _calculate_piotroski(self, fin_data) -> ToolResult:
        """Calculate Piotroski F-Score."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._service.calculate_piotroski,
            fin_data
        )

        return ToolResult.success_result(
            data=result.to_dict(),
            warnings=result.warnings,
            metadata={
                "score_name": "Piotroski F-Score",
                "max_score": 9,
                "requires_prior_period": True,
            }
        )

    async def _calculate_composite(self, fin_data) -> ToolResult:
        """Calculate composite distress assessment."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._service.calculate_composite,
            fin_data
        )

        return ToolResult.success_result(
            data=result.to_dict(),
            warnings=result.warnings,
            metadata={
                "score_name": "Composite Credit Risk",
                "includes": ["altman_zscore", "beneish_mscore", "piotroski_fscore"],
            }
        )

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Credit Risk Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                },
                "action": {
                    "type": "string",
                    "enum": ["all", "altman", "beneish", "piotroski", "composite"],
                    "description": "Assessment type to perform",
                    "default": "composite"
                }
            },
            "required": ["symbol"]
        }
