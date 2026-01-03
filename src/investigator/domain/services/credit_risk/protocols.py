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

"""Credit Risk Model Protocols and Data Types.

This module defines the interfaces and common data structures for all
credit risk scoring models. Using protocols enables:
- Type safety with static type checkers
- Dependency injection for testing
- Plugin architecture for new models

SOLID: Interface Segregation & Dependency Inversion
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable


@dataclass
class FinancialData:
    """Standardized financial data container for credit risk calculations.

    This data class provides a consistent interface for all financial metrics
    required by credit risk models. Data typically comes from SEC XBRL filings.

    Attributes:
        symbol: Stock ticker symbol
        fiscal_year: Fiscal year of the data
        fiscal_period: Fiscal period (Q1, Q2, Q3, Q4, FY)
        data_date: Date of the financial data

        # Balance Sheet - Assets
        total_assets: Total assets
        current_assets: Current assets
        cash_and_equivalents: Cash and cash equivalents
        accounts_receivable: Net accounts receivable
        inventory: Total inventory
        property_plant_equipment: Net PP&E

        # Balance Sheet - Liabilities & Equity
        total_liabilities: Total liabilities
        current_liabilities: Current liabilities
        total_debt: Total debt (short + long term)
        long_term_debt: Long-term debt
        short_term_debt: Short-term debt / current portion
        stockholders_equity: Total stockholders' equity
        retained_earnings: Retained earnings

        # Income Statement
        revenue: Total revenue / net sales
        gross_profit: Gross profit
        operating_income: Operating income / EBIT
        net_income: Net income
        cost_of_revenue: Cost of goods sold / cost of revenue
        sga_expense: Selling, general & administrative expense
        depreciation_amortization: Depreciation and amortization
        interest_expense: Interest expense

        # Cash Flow
        operating_cash_flow: Cash from operations
        capital_expenditures: Capital expenditures

        # Market Data
        market_cap: Market capitalization
        shares_outstanding: Shares outstanding

        # Prior Period Data (for year-over-year comparisons)
        prior_period: Optional prior period financial data
    """
    symbol: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    data_date: Optional[date] = None

    # Balance Sheet - Assets
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    property_plant_equipment: Optional[float] = None

    # Balance Sheet - Liabilities & Equity
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    stockholders_equity: Optional[float] = None
    retained_earnings: Optional[float] = None

    # Income Statement
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    sga_expense: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    interest_expense: Optional[float] = None

    # Cash Flow
    operating_cash_flow: Optional[float] = None
    capital_expenditures: Optional[float] = None

    # Market Data
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None

    # Prior Period (for YoY comparisons)
    prior_period: Optional["FinancialData"] = None

    @property
    def working_capital(self) -> Optional[float]:
        """Calculate working capital (Current Assets - Current Liabilities)."""
        if self.current_assets is not None and self.current_liabilities is not None:
            return self.current_assets - self.current_liabilities
        return None

    @property
    def ebit(self) -> Optional[float]:
        """EBIT (Earnings Before Interest and Taxes) = Operating Income."""
        return self.operating_income

    @property
    def free_cash_flow(self) -> Optional[float]:
        """Free Cash Flow = Operating Cash Flow - CapEx."""
        if self.operating_cash_flow is not None and self.capital_expenditures is not None:
            return self.operating_cash_flow - abs(self.capital_expenditures)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "symbol": self.symbol,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "data_date": str(self.data_date) if self.data_date else None,
            "total_assets": self.total_assets,
            "current_assets": self.current_assets,
            "cash_and_equivalents": self.cash_and_equivalents,
            "accounts_receivable": self.accounts_receivable,
            "inventory": self.inventory,
            "property_plant_equipment": self.property_plant_equipment,
            "total_liabilities": self.total_liabilities,
            "current_liabilities": self.current_liabilities,
            "total_debt": self.total_debt,
            "long_term_debt": self.long_term_debt,
            "short_term_debt": self.short_term_debt,
            "stockholders_equity": self.stockholders_equity,
            "retained_earnings": self.retained_earnings,
            "revenue": self.revenue,
            "gross_profit": self.gross_profit,
            "operating_income": self.operating_income,
            "net_income": self.net_income,
            "cost_of_revenue": self.cost_of_revenue,
            "sga_expense": self.sga_expense,
            "depreciation_amortization": self.depreciation_amortization,
            "interest_expense": self.interest_expense,
            "operating_cash_flow": self.operating_cash_flow,
            "capital_expenditures": self.capital_expenditures,
            "market_cap": self.market_cap,
            "shares_outstanding": self.shares_outstanding,
            "working_capital": self.working_capital,
            "ebit": self.ebit,
            "free_cash_flow": self.free_cash_flow,
        }
        if self.prior_period:
            result["prior_period"] = self.prior_period.to_dict()
        return result


@dataclass
class CreditScoreResult:
    """Base class for credit score results.

    All credit score calculators return results that extend this base class.
    This provides a consistent interface for accessing score values and metadata.

    Attributes:
        symbol: Stock ticker symbol
        score: The calculated score value
        score_name: Name of the score (e.g., "Altman Z-Score")
        interpretation: Human-readable interpretation
        calculation_date: When the score was calculated
        data_date: Date of the financial data used
        components: Individual components of the score calculation
        warnings: Any warnings or data quality issues
        metadata: Additional contextual information
    """
    symbol: str
    score: Optional[float] = None
    score_name: str = ""
    interpretation: str = ""
    calculation_date: Optional[date] = None
    data_date: Optional[date] = None
    components: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the score was successfully calculated."""
        return self.score is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "score": self.score,
            "score_name": self.score_name,
            "interpretation": self.interpretation,
            "calculation_date": str(self.calculation_date) if self.calculation_date else None,
            "data_date": str(self.data_date) if self.data_date else None,
            "components": self.components,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
        }


# Type variable for generic credit score result types
T = TypeVar("T", bound=CreditScoreResult)


@runtime_checkable
class CreditScoreCalculator(Protocol[T]):
    """Protocol defining the interface for credit score calculators.

    All credit risk models must implement this protocol. This enables:
    - Dependency injection for testing
    - Plugin architecture for new models
    - Type-safe usage with generics

    SOLID: Dependency Inversion - depend on abstractions, not concretions
    """

    @property
    def name(self) -> str:
        """Return the name of this score calculator."""
        ...

    @property
    def description(self) -> str:
        """Return a description of what this score measures."""
        ...

    def calculate(self, data: FinancialData) -> T:
        """Calculate the credit score from financial data.

        Args:
            data: Standardized financial data

        Returns:
            Score result with value, interpretation, and components
        """
        ...

    def validate_data(self, data: FinancialData) -> List[str]:
        """Validate that required data fields are present.

        Args:
            data: Financial data to validate

        Returns:
            List of missing or invalid field names
        """
        ...
