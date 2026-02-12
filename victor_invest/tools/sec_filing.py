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

"""SEC Filing Tool for Victor Invest.

This tool wraps the existing SEC infrastructure to provide SEC EDGAR
filing retrieval, XBRL parsing, and company facts extraction.

Infrastructure wrapped:
- investigator.infrastructure.sec.sec_api.SECApiClient
- investigator.infrastructure.sec.xbrl_parser.XBRLParser
- investigator.infrastructure.sec.companyfacts_extractor.SECCompanyFactsExtractor

Example:
    tool = SECFilingTool()

    # Get latest 10-K filing
    result = await tool.execute(
        symbol="AAPL",
        form_type="10-K",
        action="get_filing"
    )

    # Get company facts (financial metrics)
    result = await tool.execute(
        symbol="AAPL",
        action="get_company_facts"
    )

    # Search for filings
    result = await tool.execute(
        symbol="AAPL",
        action="search_filings",
        form_type="10-Q",
        limit=5
    )
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SECFilingTool(BaseTool):
    """Tool for retrieving and parsing SEC EDGAR filings.

    Provides access to SEC filings (10-K, 10-Q, 8-K), company facts via
    the CompanyFacts API, and XBRL financial data extraction.

    Supported actions:
    - get_filing: Retrieve a specific filing's content
    - get_company_facts: Get structured financial data
    - search_filings: Search for filings by type
    - extract_metrics: Extract financial metrics from company facts
    - parse_xbrl: Parse XBRL content from a filing

    Attributes:
        name: "sec_filing"
        description: Tool description for agent discovery
    """

    name = "sec_filing"
    description = """Retrieve and analyze SEC EDGAR filings for US public companies.

Actions:
- get_filing: Get full filing content (10-K, 10-Q, 8-K)
- get_company_facts: Get structured financial data from SEC CompanyFacts API
- search_filings: Search for recent filings by form type
- extract_metrics: Extract key financial metrics (revenues, assets, liabilities, etc.)
- parse_xbrl: Parse XBRL content for detailed financial data

Parameters:
- symbol: Stock ticker symbol (required)
- action: One of the actions above (required)
- form_type: Filing form type (default: "10-K")
- period: Filing period ("latest" or specific quarter like "2024-Q3")
- limit: Number of filings to return for search (default: 5)
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize SEC Filing Tool.

        Args:
            config: Optional investigator config object. If not provided,
                   will use get_config() on first use.
        """
        super().__init__(config)
        self._sec_client = None
        self._xbrl_parser = None
        self._facts_extractor = None

    async def initialize(self) -> None:
        """Initialize SEC infrastructure components."""
        try:
            # Lazy import to avoid circular dependencies
            from investigator.infrastructure.sec.companyfacts_extractor import get_sec_companyfacts_extractor
            from investigator.infrastructure.sec.sec_api import SECApiClient
            from investigator.infrastructure.sec.xbrl_parser import XBRLParser

            if self.config is None:
                from investigator.config import get_config

                self.config = get_config()

            self._sec_client = SECApiClient(config=self.config)
            self._xbrl_parser = XBRLParser()
            self._facts_extractor = get_sec_companyfacts_extractor()

            self._initialized = True
            logger.info("SECFilingTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SECFilingTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Optional[Dict[str, Any]] = None,
        symbol: str = "",
        action: str = "get_company_facts",
        form_type: str = "10-K",
        period: str = "latest",
        limit: int = 5,
        **kwargs,
    ) -> ToolResult:
        """Execute SEC filing operation.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
            action: Operation to perform:
                - "get_filing": Get full filing content
                - "get_company_facts": Get structured financial data
                - "search_filings": Search for filings
                - "extract_metrics": Extract financial metrics
                - "parse_xbrl": Parse XBRL content
            form_type: SEC form type ("10-K", "10-Q", "8-K")
            period: Filing period ("latest" or specific period)
            limit: Max filings to return for search action
            **kwargs: Additional action-specific parameters

        Returns:
            ToolResult with filing data or error message
        """
        try:
            await self.ensure_initialized()

            symbol = symbol.upper().strip()
            if not symbol:
                return ToolResult.create_failure("Symbol is required")

            action = action.lower().strip()

            if action == "get_filing":
                return await self._get_filing(symbol, form_type, period)
            elif action == "get_company_facts":
                return await self._get_company_facts(symbol)
            elif action == "search_filings":
                return await self._search_filings(symbol, form_type, limit)
            elif action == "extract_metrics":
                return await self._extract_metrics(symbol)
            elif action == "parse_xbrl":
                xbrl_content = kwargs.get("xbrl_content", "")
                return await self._parse_xbrl(xbrl_content)
            else:
                return ToolResult.create_failure(
                    f"Unknown action: {action}. Valid actions: "
                    "get_filing, get_company_facts, search_filings, extract_metrics, parse_xbrl"
                )

        except Exception as e:
            logger.error(f"SECFilingTool execute error for {symbol}: {e}")
            return ToolResult.create_failure(
                f"SEC filing operation failed: {str(e)}", metadata={"symbol": symbol, "action": action}
            )

    async def _get_filing(self, symbol: str, form_type: str, period: str) -> ToolResult:
        """Get full filing content for a symbol.

        Args:
            symbol: Stock ticker
            form_type: Form type (10-K, 10-Q, 8-K)
            period: "latest" or specific period

        Returns:
            ToolResult with filing content and metadata
        """
        try:
            filing_data = await self._sec_client.get_filing_by_symbol(symbol=symbol, form_type=form_type, period=period)

            if not filing_data:
                return ToolResult.create_failure(
                    f"No {form_type} filing found for {symbol}", metadata={"symbol": symbol, "form_type": form_type}
                )

            return ToolResult.create_success(output={
                    "symbol": symbol,
                    "form_type": filing_data.get("filing_type", form_type),
                    "filing_date": filing_data.get("filing_date"),
                    "period_end": filing_data.get("period_end"),
                    "form_url": filing_data.get("form_url"),
                    "xbrl_url": filing_data.get("xbrl_url"),
                    "cik": filing_data.get("cik"),
                    "text": filing_data.get("text", "")[:50000],  # Truncate large filings
                },
                metadata={
                    "source": "sec_edgar",
                    "form_type": form_type,
                    "period": period,
                    "truncated": len(filing_data.get("text", "")) > 50000,
                },
            )

        except Exception as e:
            logger.error(f"Error getting filing for {symbol}: {e}")
            return ToolResult.create_failure(f"Failed to get filing: {str(e)}")

    async def _get_company_facts(self, symbol: str) -> ToolResult:
        """Get company facts from SEC CompanyFacts API.

        Args:
            symbol: Stock ticker

        Returns:
            ToolResult with company facts data
        """
        try:
            # Run synchronous method in thread pool
            loop = asyncio.get_event_loop()
            facts_data = await loop.run_in_executor(None, self._facts_extractor.get_company_facts, symbol)

            if not facts_data:
                return ToolResult.create_failure(
                    f"No company facts found for {symbol}. "
                    "The symbol may not be in the cache. Consider triggering SEC data fetch.",
                    metadata={"symbol": symbol},
                )

            return ToolResult.create_success(output={
                    "symbol": facts_data.get("symbol", symbol),
                    "cik": facts_data.get("cik"),
                    "entity_name": facts_data.get("entityName"),
                    "facts": facts_data.get("facts", {}),
                    "source": facts_data.get("source", "unknown"),
                    "fetched_at": facts_data.get("fetched_at"),
                },
                metadata={
                    "source": facts_data.get("source", "unknown"),
                    "has_us_gaap": "us-gaap" in facts_data.get("facts", {}),
                },
            )

        except Exception as e:
            logger.error(f"Error getting company facts for {symbol}: {e}")
            return ToolResult.create_failure(f"Failed to get company facts: {str(e)}")

    async def _search_filings(self, symbol: str, form_type: str, limit: int) -> ToolResult:
        """Search for recent filings.

        Args:
            symbol: Stock ticker
            form_type: Form type filter
            limit: Maximum filings to return

        Returns:
            ToolResult with list of filing metadata
        """
        try:
            filings = await self._sec_client.search_filings(symbol=symbol, form_type=form_type, limit=limit)

            if not filings:
                return ToolResult.create_failure(
                    f"No {form_type} filings found for {symbol}", metadata={"symbol": symbol, "form_type": form_type}
                )

            return ToolResult.create_success(output={"symbol": symbol, "form_type": form_type, "count": len(filings), "filings": filings},
                metadata={"source": "sec_edgar", "limit": limit},
            )

        except Exception as e:
            logger.error(f"Error searching filings for {symbol}: {e}")
            return ToolResult.create_failure(f"Failed to search filings: {str(e)}")

    async def _extract_metrics(self, symbol: str) -> ToolResult:
        """Extract financial metrics from company facts.

        Args:
            symbol: Stock ticker

        Returns:
            ToolResult with extracted financial metrics
        """
        try:
            # Run synchronous method in thread pool
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(None, self._facts_extractor.extract_financial_metrics, symbol)

            if not metrics or all(
                v is None
                for k, v in metrics.items()
                if k not in ["symbol", "cik", "company_name", "data_date", "source", "fiscal_year", "fiscal_period"]
            ):
                return ToolResult.create_failure(
                    f"No financial metrics available for {symbol}", metadata={"symbol": symbol}
                )

            # Calculate financial ratios
            ratios = await loop.run_in_executor(
                None, self._facts_extractor.calculate_financial_ratios, symbol, None  # current_price
            )

            # Get shares outstanding from multiple possible fields
            shares_outstanding = (
                metrics.get("shares_outstanding")
                or metrics.get("common_stock_shares_outstanding")
                or metrics.get("weighted_average_shares_diluted")
                or metrics.get("weighted_average_diluted_shares_outstanding")
            )

            return ToolResult.create_success(output={
                    "symbol": symbol,
                    "fiscal_year": metrics.get("fiscal_year"),
                    "fiscal_period": metrics.get("fiscal_period"),
                    "shares_outstanding": shares_outstanding,  # Top-level for easy access
                    "balance_sheet": {
                        "total_assets": metrics.get("assets"),
                        "current_assets": metrics.get("assets_current"),
                        "total_liabilities": metrics.get("liabilities"),
                        "current_liabilities": metrics.get("liabilities_current"),
                        "stockholders_equity": metrics.get("equity"),
                        "total_debt": metrics.get("total_debt"),
                        "long_term_debt": metrics.get("long_term_debt"),
                        "short_term_debt": metrics.get("debt_current"),
                        "cash_and_equivalents": metrics.get("cash_and_equivalents"),
                        "inventory": metrics.get("inventory"),
                        "accounts_receivable": metrics.get("accounts_receivable"),
                        "property_plant_equipment": metrics.get("property_plant_equipment"),
                        "shares_outstanding": shares_outstanding,
                    },
                    "income_statement": {
                        "total_revenue": metrics.get("revenues"),  # Alias for DCF
                        "revenue": metrics.get("revenues"),  # Alias
                        "revenues": metrics.get("revenues"),  # Original
                        "net_income": metrics.get("net_income"),
                        "net_income_loss": metrics.get("net_income"),  # Alias
                        "gross_profit": metrics.get("gross_profit"),
                        "operating_income": metrics.get("operating_income"),
                        "cost_of_revenue": metrics.get("cost_of_revenue"),
                        "ebitda": metrics.get("ebitda"),
                        "depreciation_amortization": metrics.get("depreciation_amortization"),
                    },
                    "cash_flow": {
                        "operating_cash_flow": metrics.get("operating_cash_flow"),
                        "capital_expenditures": metrics.get("capital_expenditures"),
                        "capex": metrics.get("capital_expenditures"),  # Alias
                        "free_cash_flow": metrics.get("free_cash_flow"),
                        "dividends_paid": metrics.get("dividends_paid"),
                    },
                    "ratios": {
                        "current_ratio": ratios.get("current_ratio"),
                        "quick_ratio": ratios.get("quick_ratio"),
                        "debt_to_equity": ratios.get("debt_to_equity"),
                        "debt_to_assets": ratios.get("debt_to_assets"),
                        "roe": ratios.get("roe"),
                        "roa": ratios.get("roa"),
                        "gross_margin": ratios.get("gross_margin"),
                        "operating_margin": ratios.get("operating_margin"),
                        "net_margin": ratios.get("net_margin"),
                        "shares_outstanding": shares_outstanding,
                    },
                    # Sector-specific metrics
                    "insurance_metrics": {
                        "premiums_earned": metrics.get("premiums_earned"),
                        "claims_incurred": metrics.get("claims_incurred"),
                        "policy_acquisition_costs": metrics.get("policy_acquisition_costs"),
                        "combined_ratio": self._calculate_combined_ratio(metrics),
                    },
                    "defense_metrics": {
                        "order_backlog": metrics.get("order_backlog"),
                        "contract_liability": metrics.get("contract_liability"),
                        "unbilled_contracts_receivable": metrics.get("unbilled_contracts_receivable"),
                        "backlog_to_revenue": self._calculate_backlog_ratio(metrics),
                    },
                },
                metadata={"source": metrics.get("source", "unknown"), "data_date": metrics.get("data_date")},
            )

        except Exception as e:
            logger.error(f"Error extracting metrics for {symbol}: {e}")
            return ToolResult.create_failure(f"Failed to extract metrics: {str(e)}")

    def _calculate_combined_ratio(self, metrics: Dict) -> Optional[float]:
        """Calculate insurance combined ratio from XBRL data.

        Combined Ratio = (Claims + Expenses) / Premiums
        A ratio < 100% indicates underwriting profit.

        Args:
            metrics: Extracted financial metrics

        Returns:
            Combined ratio as percentage or None if insufficient data
        """
        premiums = metrics.get("premiums_earned")
        claims = metrics.get("claims_incurred")

        if not premiums or premiums <= 0:
            return None

        # Get expenses (policy acquisition costs or operating expenses)
        expenses = metrics.get("policy_acquisition_costs", 0) or 0

        if claims is None:
            return None

        combined_ratio = ((claims + expenses) / premiums) * 100
        return round(combined_ratio, 2)

    def _calculate_backlog_ratio(self, metrics: Dict) -> Optional[float]:
        """Calculate backlog-to-revenue ratio for defense contractors.

        Backlog/Revenue > 2.0x typically indicates strong revenue visibility.

        Args:
            metrics: Extracted financial metrics

        Returns:
            Backlog-to-revenue ratio or None if insufficient data
        """
        backlog = metrics.get("order_backlog")
        revenue = metrics.get("revenues")

        if not backlog or not revenue or revenue <= 0:
            return None

        return round(backlog / revenue, 2)

    async def _parse_xbrl(self, xbrl_content: str) -> ToolResult:
        """Parse XBRL content from a filing.

        Args:
            xbrl_content: Raw XBRL XML content

        Returns:
            ToolResult with parsed XBRL data
        """
        try:
            if not xbrl_content:
                return ToolResult.create_failure("No XBRL content provided")

            parsed_data = await self._xbrl_parser.parse_filing(xbrl_content)

            if not parsed_data:
                return ToolResult.create_failure("Failed to parse XBRL content")

            # Extract metrics from parsed data
            metrics = await self._xbrl_parser.extract_metrics(parsed_data)

            return ToolResult.create_success(output={
                    "document_info": parsed_data.get("document_info", {}),
                    "financial_data": parsed_data.get("financial_data", {}),
                    "metrics": metrics,
                },
                metadata={"source": "xbrl_parser"},
            )

        except Exception as e:
            logger.error(f"Error parsing XBRL: {e}")
            return ToolResult.create_failure(f"Failed to parse XBRL: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for SEC Filing Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"},
                "action": {
                    "type": "string",
                    "enum": ["get_filing", "get_company_facts", "search_filings", "extract_metrics", "parse_xbrl"],
                    "description": "Action to perform",
                    "default": "get_company_facts",
                },
                "form_type": {
                    "type": "string",
                    "enum": ["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"],
                    "description": "SEC form type",
                    "default": "10-K",
                },
                "period": {
                    "type": "string",
                    "description": "Filing period (e.g., 'latest', '2024-Q3')",
                    "default": "latest",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum filings to return for search",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["symbol"],
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._sec_client:
            self._sec_client.close()
