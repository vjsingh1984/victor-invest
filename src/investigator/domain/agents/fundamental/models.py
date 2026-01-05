"""Data models shared by the fundamental analysis agent modules.

TD3 FIX: Added value_type field to distinguish YTD vs quarterly values.
This resolves the schema mismatch affecting 80% of S&P 100 companies where
YTD and quarterly values were mixed, causing incorrect Q4 computations.

Author: InvestiGator Team
Updated: 2025-12-29 (TD3 value_type fix)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValueType(Enum):
    """
    TD3 FIX: Enumeration of financial value types.

    This resolves the schema mismatch where YTD and quarterly values
    were stored without proper distinction, causing:
    - Incorrect Q4 computations (negative values for 80% of companies)
    - Cache collisions between YTD and quarterly cached data
    - Inconsistent financial analysis results

    Values:
        QUARTERLY: Individual quarter value (Q1, Q2, Q3, Q4 standalone)
        YTD: Year-to-date cumulative value (Q1=Q1, Q2=Q1+Q2, Q3=Q1+Q2+Q3)
        ANNUAL: Full fiscal year value (FY/10-K)
        TTM: Trailing twelve months (rolling 4-quarter sum)
    """

    QUARTERLY = "quarterly"
    YTD = "ytd"
    ANNUAL = "annual"
    TTM = "ttm"

    @classmethod
    def from_string(cls, value: str) -> "ValueType":
        """Convert string to ValueType enum, with fallback to QUARTERLY."""
        if not value:
            return cls.QUARTERLY

        value_lower = value.lower().strip()

        if value_lower in ("quarterly", "qtr", "quarter", "q"):
            return cls.QUARTERLY
        elif value_lower in ("ytd", "year-to-date", "year_to_date", "cumulative"):
            return cls.YTD
        elif value_lower in ("annual", "fy", "full_year", "yearly"):
            return cls.ANNUAL
        elif value_lower in ("ttm", "trailing", "trailing_twelve_months", "ltm"):
            return cls.TTM
        else:
            return cls.QUARTERLY  # Default fallback


@dataclass
class QuarterlyData:
    """
    Type-safe container for quarterly financial data.

    TD3 FIX: Added value_type field to properly distinguish between:
    - QUARTERLY: Individual quarter values (for direct comparison)
    - YTD: Year-to-date cumulative values (need conversion for Q4)
    - ANNUAL: Full year values (used for Q4 = FY - YTD_Q3 computation)
    - TTM: Trailing twelve month values (for growth calculations)

    This fixes the schema mismatch affecting 80% of S&P 100 companies.
    """

    fiscal_year: int
    fiscal_period: str
    financial_data: Dict[str, Any]
    ratios: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None
    data_quality: Optional[Dict[str, Any]] = None
    filing_date: Optional[str] = None
    period_end_date: Optional[str] = None

    # Legacy YTD flags (kept for backward compatibility)
    is_ytd_cashflow: bool = False
    is_ytd_income: bool = False

    # TD3 FIX: New unified value_type field
    # This replaces the fragmented is_ytd_cashflow/is_ytd_income with a single,
    # consistent field that applies to all financial data in this record.
    value_type: str = "quarterly"  # "quarterly", "ytd", "annual", "ttm"

    # TD3 FIX: Additional metadata for value type tracking per statement
    value_types_by_statement: Optional[Dict[str, str]] = None
    # Example: {"income_statement": "ytd", "cash_flow": "ytd", "balance_sheet": "quarterly"}

    def __post_init__(self) -> None:
        valid_periods = ["Q1", "Q2", "Q3", "Q4", "FY"]
        if self.fiscal_period not in valid_periods:
            raise ValueError(f"Invalid fiscal_period: {self.fiscal_period}. Must be one of {valid_periods}")

        # TD3 FIX: Validate value_type
        valid_value_types = ["quarterly", "ytd", "annual", "ttm"]
        if self.value_type not in valid_value_types:
            raise ValueError(f"Invalid value_type: {self.value_type}. Must be one of {valid_value_types}")

        # TD3 FIX: Sync legacy flags with new value_type for backward compatibility
        if self.value_type == "ytd":
            # If value_type is YTD, ensure legacy flags are set appropriately
            # (unless overridden by value_types_by_statement)
            if self.value_types_by_statement is None:
                self.is_ytd_cashflow = True
                self.is_ytd_income = True
        elif self.is_ytd_cashflow or self.is_ytd_income:
            # Legacy flags are set - infer value_type if not explicitly set
            if self.value_type == "quarterly":  # Default value, might need updating
                # Check if both are YTD or just one
                if self.is_ytd_cashflow and self.is_ytd_income:
                    self.value_type = "ytd"
                # If only one is YTD, keep as quarterly but set value_types_by_statement
                elif self.value_types_by_statement is None:
                    self.value_types_by_statement = {
                        "income_statement": "ytd" if self.is_ytd_income else "quarterly",
                        "cash_flow": "ytd" if self.is_ytd_cashflow else "quarterly",
                        "balance_sheet": "quarterly",  # Balance sheet is never YTD
                    }

    @property
    def is_ytd(self) -> bool:
        """TD3 FIX: Check if this data represents YTD values."""
        return self.value_type == "ytd"

    @property
    def is_annual(self) -> bool:
        """TD3 FIX: Check if this data represents full year (annual) values."""
        return self.value_type == "annual" or self.fiscal_period == "FY"

    @property
    def is_ttm(self) -> bool:
        """TD3 FIX: Check if this data represents trailing twelve month values."""
        return self.value_type == "ttm"

    @property
    def value_type_enum(self) -> ValueType:
        """TD3 FIX: Get value_type as enum for type-safe comparisons."""
        return ValueType.from_string(self.value_type)

    @property
    def period_label(self) -> str:
        return f"{self.fiscal_year}-{self.fiscal_period}"

    @property
    def calendar_year(self) -> int:
        return self.fiscal_year

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert QuarterlyData to dictionary format.

        TD3 FIX: Now includes value_type and value_types_by_statement
        fields for proper YTD/quarterly distinction.
        """
        fd = self.financial_data or {}

        # TD3 FIX: Determine value_type for each statement section
        statement_types = self.value_types_by_statement or {}
        income_value_type = statement_types.get("income_statement", self.value_type)
        cashflow_value_type = statement_types.get("cash_flow", self.value_type)
        balance_value_type = statement_types.get("balance_sheet", "quarterly")  # Always point-in-time

        return {
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            # TD3 FIX: Include unified value_type at top level
            "value_type": self.value_type,
            "value_types_by_statement": self.value_types_by_statement,
            "shares_outstanding": (
                fd.get("weighted_average_diluted_shares_outstanding")
                or fd.get("shares_outstanding")
                or fd.get("market_metrics", {}).get("shares_outstanding")
                or 0
            ),
            "cash_flow": {
                "operating_cash_flow": fd.get("operating_cash_flow", 0),
                "capital_expenditures": fd.get("capital_expenditures", 0),
                "free_cash_flow": fd.get("free_cash_flow", 0),
                "dividends_paid": fd.get("dividends_paid", 0),
                "is_ytd": self.is_ytd_cashflow,  # Legacy flag
                "value_type": cashflow_value_type,  # TD3 FIX: New field
            },
            "income_statement": {
                "total_revenue": fd.get("revenues", 0) or fd.get("total_revenue", 0),
                "net_income": fd.get("net_income", 0),
                "gross_profit": fd.get("gross_profit", 0),
                "operating_income": fd.get("operating_income", 0),
                "interest_expense": fd.get("interest_expense", 0),
                "income_tax_expense": fd.get("income_tax_expense", 0),
                "cost_of_revenue": fd.get("cost_of_revenue", 0),
                "is_ytd": self.is_ytd_income,  # Legacy flag
                "value_type": income_value_type,  # TD3 FIX: New field
            },
            "balance_sheet": {
                "total_assets": fd.get("total_assets", 0),
                "total_liabilities": fd.get("total_liabilities", 0),
                "stockholders_equity": fd.get("stockholders_equity", 0),
                "current_assets": fd.get("current_assets", 0),
                "current_liabilities": fd.get("current_liabilities", 0),
                "accounts_receivable": fd.get("accounts_receivable", 0),
                "inventory": fd.get("inventory", 0),
                "cash_and_equivalents": fd.get("cash_and_equivalents", 0),
                "property_plant_equipment_net": fd.get("property_plant_equipment_net", 0),
                "total_debt": fd.get("total_debt", 0),
                "long_term_debt": fd.get("long_term_debt", 0),
                "short_term_debt": fd.get("short_term_debt", 0) or fd.get("debt_current", 0),
                "value_type": balance_value_type,  # TD3 FIX: Always "quarterly" (point-in-time)
            },
            "ratios": self.ratios or {},
            "data_quality": self.data_quality,
            "filing_date": self.filing_date,
            "period_end_date": self.period_end_date,
            "period_label": self.period_label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuarterlyData":
        """
        Create QuarterlyData from dictionary.

        TD3 FIX: Now handles value_type and value_types_by_statement fields
        for proper YTD/quarterly distinction.
        """
        if "cash_flow" in data or "income_statement" in data:
            cash_flow = data.get("cash_flow", {})
            income = data.get("income_statement", {})
            balance = data.get("balance_sheet", {})
            if not isinstance(cash_flow, dict):
                cash_flow = {}
            if not isinstance(income, dict):
                income = {}
            if not isinstance(balance, dict):
                balance = {}

            # TD3 FIX: Prefer new value_type field, fall back to legacy is_ytd
            is_ytd_cashflow = cash_flow.get("value_type") == "ytd" or cash_flow.get("is_ytd", False)
            is_ytd_income = income.get("value_type") == "ytd" or income.get("is_ytd", False)

            financial_data = {
                "operating_cash_flow": cash_flow.get("operating_cash_flow", 0),
                "capital_expenditures": cash_flow.get("capital_expenditures", 0),
                "free_cash_flow": cash_flow.get("free_cash_flow", 0),
                "dividends_paid": cash_flow.get("dividends_paid", 0),
                "revenues": income.get("total_revenue", 0),
                "total_revenue": income.get("total_revenue", 0),
                "net_income": income.get("net_income", 0),
                "gross_profit": income.get("gross_profit", 0),
                "operating_income": income.get("operating_income", 0),
                "interest_expense": income.get("interest_expense", 0),
                "income_tax_expense": income.get("income_tax_expense", 0),
                "cost_of_revenue": income.get("cost_of_revenue", 0),
                "total_assets": balance.get("total_assets", 0),
                "total_liabilities": balance.get("total_liabilities", 0),
                "stockholders_equity": balance.get("stockholders_equity", 0),
                "current_assets": balance.get("current_assets", 0),
                "current_liabilities": balance.get("current_liabilities", 0),
                "accounts_receivable": balance.get("accounts_receivable", 0),
                "inventory": balance.get("inventory", 0),
                "cash_and_equivalents": balance.get("cash_and_equivalents", 0),
                "property_plant_equipment_net": balance.get("property_plant_equipment_net", 0),
                "total_debt": balance.get("total_debt", 0),
                "long_term_debt": balance.get("long_term_debt", 0),
                "short_term_debt": balance.get("short_term_debt", 0) or balance.get("debt_current", 0),
            }
        else:
            financial_data = data.get("financial_data", {})
            is_ytd_cashflow = False
            is_ytd_income = False

        data_quality_raw = data.get("data_quality")
        if isinstance(data_quality_raw, dict):
            data_quality = data_quality_raw
        elif isinstance(data_quality_raw, (int, float)):
            data_quality = {
                "score": float(data_quality_raw),
                "quality_level": (
                    "Excellent" if data_quality_raw >= 90 else "Good" if data_quality_raw >= 70 else "Fair"
                ),
                "missing_core": [],
            }
        else:
            data_quality = None

        # TD3 FIX: Extract value_type from data, with intelligent defaults
        value_type = data.get("value_type", "quarterly")
        value_types_by_statement = data.get("value_types_by_statement")

        # TD3 FIX: If value_type not explicitly set but legacy flags indicate YTD
        if value_type == "quarterly" and (is_ytd_cashflow or is_ytd_income):
            if is_ytd_cashflow and is_ytd_income:
                value_type = "ytd"
            elif value_types_by_statement is None:
                # Build value_types_by_statement from legacy flags
                value_types_by_statement = {
                    "income_statement": "ytd" if is_ytd_income else "quarterly",
                    "cash_flow": "ytd" if is_ytd_cashflow else "quarterly",
                    "balance_sheet": "quarterly",
                }

        return cls(
            fiscal_year=data.get("fiscal_year"),
            fiscal_period=data.get("fiscal_period"),
            financial_data=financial_data,
            ratios=data.get("ratios"),
            market_data=data.get("market_data"),
            data_quality=data_quality,
            filing_date=data.get("filing_date"),
            period_end_date=data.get("period_end_date"),
            is_ytd_cashflow=is_ytd_cashflow,
            is_ytd_income=is_ytd_income,
            value_type=value_type,
            value_types_by_statement=value_types_by_statement,
        )

    def get_statement_value_type(self, statement: str) -> str:
        """
        TD3 FIX: Get the value_type for a specific financial statement.

        Args:
            statement: One of "income_statement", "cash_flow", "balance_sheet"

        Returns:
            Value type string: "quarterly", "ytd", "annual", or "ttm"
        """
        if self.value_types_by_statement and statement in self.value_types_by_statement:
            return self.value_types_by_statement[statement]
        return self.value_type

    def requires_ytd_conversion(self, statement: Optional[str] = None) -> bool:
        """
        TD3 FIX: Check if data requires YTD-to-quarterly conversion.

        Args:
            statement: Optional statement type to check. If None, checks overall.

        Returns:
            True if the data (or specific statement) is YTD and needs conversion.
        """
        if statement:
            return self.get_statement_value_type(statement) == "ytd"
        return self.value_type == "ytd"


__all__ = ["QuarterlyData", "ValueType"]
