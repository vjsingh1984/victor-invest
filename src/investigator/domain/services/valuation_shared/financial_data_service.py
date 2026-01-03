# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Financial Data Service - Unified financial data fetching for all valuation consumers.

This service provides a single source of truth for fetching financial data from
the SEC database. Eliminates duplication between:
- scripts/rl_backtest.py (get_quarterly_metrics_structured, get_multi_year_data)
- victor_invest/tools/sec_filing.py (_extract_metrics)
- batch_analysis_runner.py

Key features:
- Consistent data format (nested income_statement/cash_flow/balance_sheet)
- Point-in-time queries (only data available as of specified date)
- Format detection and normalization
- TTM calculation support

Example:
    from investigator.domain.services.valuation_shared import FinancialDataService

    service = FinancialDataService()

    # Get 12 quarters of data as of a specific date
    quarterly = service.get_quarterly_metrics("AAPL", as_of_date=date(2024, 3, 15))

    # Get 5 years of annual data
    annual = service.get_annual_metrics("AAPL", as_of_date=date(2024, 3, 15))

    # Get TTM (trailing twelve months) metrics
    ttm = service.get_ttm_metrics("AAPL", as_of_date=date(2024, 3, 15))
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class FinancialDataService:
    """
    Unified service for fetching financial data from SEC database.

    Provides consistent data format and point-in-time queries for
    all valuation consumers.
    """

    def __init__(
        self,
        db_url: str = "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database",
    ):
        """
        Initialize FinancialDataService.

        Args:
            db_url: Database connection string for SEC database.
        """
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_quarterly_metrics(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
        num_quarters: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Get quarterly metrics in the structured format expected by valuation models.

        Returns data in nested structure:
        {
            'fiscal_year': 2024,
            'fiscal_period': 'Q3',
            'filed_date': date(...),
            'income_statement': {...},
            'cash_flow': {...},
            'balance_sheet': {...},
            'shares_outstanding': float,
            'roe': float,
        }

        Args:
            symbol: Stock ticker symbol
            as_of_date: Only include data filed on or before this date (default: today)
            num_quarters: Number of quarters to retrieve (default: 12)

        Returns:
            List of quarterly metrics dicts, most recent first
        """
        if as_of_date is None:
            as_of_date = date.today()

        symbol = symbol.upper()

        with self.Session() as session:
            query = """
                SELECT
                    symbol, fiscal_year, fiscal_period, filed_date,
                    total_revenue, net_income, gross_profit, operating_income,
                    operating_cash_flow, free_cash_flow, capital_expenditures,
                    dividends_paid,
                    total_assets, total_liabilities, stockholders_equity,
                    cash_and_equivalents, long_term_debt, short_term_debt,
                    current_assets, current_liabilities,
                    shares_outstanding, interest_expense, income_tax_expense,
                    depreciation_amortization, roe
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND filed_date <= :as_of_date
                  AND fiscal_period != 'FY'
                ORDER BY fiscal_year DESC,
                         CASE fiscal_period
                             WHEN 'Q4' THEN 4
                             WHEN 'Q3' THEN 3
                             WHEN 'Q2' THEN 2
                             WHEN 'Q1' THEN 1
                         END DESC
                LIMIT :num_quarters
            """
            results = session.execute(
                text(query),
                {"symbol": symbol, "as_of_date": as_of_date, "num_quarters": num_quarters}
            ).fetchall()

            quarterly_metrics = []
            for row in results:
                operating_income = self._to_float(row[7])
                depreciation = self._to_float(row[23])
                ebitda = operating_income + depreciation if operating_income else None

                quarterly_metrics.append({
                    "fiscal_year": row[1],
                    "fiscal_period": row[2],
                    "filed_date": row[3],
                    "income_statement": {
                        "total_revenue": self._to_float(row[4]),
                        "net_income": self._to_float(row[5]),
                        "gross_profit": self._to_float(row[6]),
                        "operating_income": operating_income,
                        "interest_expense": self._to_float(row[21]),
                        "income_tax_expense": self._to_float(row[22]),
                        "ebitda": ebitda,
                    },
                    "cash_flow": {
                        "operating_cash_flow": self._to_float(row[8]),
                        "free_cash_flow": self._to_float(row[9]),
                        "capital_expenditures": self._to_float(row[10]),
                        "dividends_paid": self._to_float(row[11]),
                    },
                    "balance_sheet": {
                        "total_assets": self._to_float(row[12]),
                        "total_liabilities": self._to_float(row[13]),
                        "stockholders_equity": self._to_float(row[14]),
                        "cash_and_equivalents": self._to_float(row[15]),
                        "long_term_debt": self._to_float(row[16]),
                        "short_term_debt": self._to_float(row[17]),
                        "current_assets": self._to_float(row[18]),
                        "current_liabilities": self._to_float(row[19]),
                    },
                    "shares_outstanding": self._to_float(row[20]),
                    "roe": self._to_float(row[24]),
                })

            return quarterly_metrics

    def get_annual_metrics(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
        num_years: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get multi-year annual (FY) data for growth calculations.

        Args:
            symbol: Stock ticker symbol
            as_of_date: Only include data filed on or before this date (default: today)
            num_years: Number of years to retrieve (default: 5)

        Returns:
            List of annual metrics dicts in chronological order (oldest first)
        """
        if as_of_date is None:
            as_of_date = date.today()

        symbol = symbol.upper()

        with self.Session() as session:
            query = """
                SELECT
                    symbol, fiscal_year, fiscal_period,
                    total_revenue, net_income, gross_profit, operating_income,
                    operating_cash_flow, free_cash_flow, capital_expenditures,
                    dividends_paid,
                    total_assets, total_liabilities, stockholders_equity,
                    shares_outstanding
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND filed_date <= :as_of_date
                  AND fiscal_period = 'FY'
                ORDER BY fiscal_year DESC
                LIMIT :num_years
            """
            results = session.execute(
                text(query),
                {"symbol": symbol, "as_of_date": as_of_date, "num_years": num_years}
            ).fetchall()

            multi_year_data = []
            for row in results:
                multi_year_data.append({
                    "fiscal_year": row[1],
                    "fiscal_period": row[2],
                    "total_revenue": self._to_float(row[3]),
                    "net_income": self._to_float(row[4]),
                    "gross_profit": self._to_float(row[5]),
                    "operating_income": self._to_float(row[6]),
                    "operating_cash_flow": self._to_float(row[7]),
                    "free_cash_flow": self._to_float(row[8]),
                    "capital_expenditures": self._to_float(row[9]),
                    "dividends_paid": self._to_float(row[10]),
                    "total_assets": self._to_float(row[11]),
                    "total_liabilities": self._to_float(row[12]),
                    "stockholders_equity": self._to_float(row[13]),
                    "shares_outstanding": self._to_float(row[14]),
                })

            # Reverse to chronological order (oldest first)
            return list(reversed(multi_year_data))

    def get_ttm_metrics(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Get trailing twelve months (TTM) metrics.

        Sums the last 4 quarters of data for flow metrics (revenue, income, cash flow).
        Uses the most recent quarter's data for stock metrics (balance sheet).

        Args:
            symbol: Stock ticker symbol
            as_of_date: Only include data filed on or before this date (default: today)

        Returns:
            Dict with TTM metrics in nested structure, or empty dict if insufficient data
        """
        quarterly = self.get_quarterly_metrics(symbol, as_of_date, num_quarters=4)

        if len(quarterly) < 4:
            logger.warning(f"Insufficient quarterly data for TTM: {symbol} has {len(quarterly)} quarters")
            return {}

        # Sum flow metrics (last 4 quarters)
        ttm = {
            "income_statement": {
                "total_revenue": sum(q["income_statement"].get("total_revenue", 0) or 0 for q in quarterly),
                "net_income": sum(q["income_statement"].get("net_income", 0) or 0 for q in quarterly),
                "gross_profit": sum(q["income_statement"].get("gross_profit", 0) or 0 for q in quarterly),
                "operating_income": sum(q["income_statement"].get("operating_income", 0) or 0 for q in quarterly),
                "interest_expense": sum(q["income_statement"].get("interest_expense", 0) or 0 for q in quarterly),
                "income_tax_expense": sum(q["income_statement"].get("income_tax_expense", 0) or 0 for q in quarterly),
                "ebitda": sum(q["income_statement"].get("ebitda", 0) or 0 for q in quarterly),
            },
            "cash_flow": {
                "operating_cash_flow": sum(q["cash_flow"].get("operating_cash_flow", 0) or 0 for q in quarterly),
                "free_cash_flow": sum(q["cash_flow"].get("free_cash_flow", 0) or 0 for q in quarterly),
                "capital_expenditures": sum(q["cash_flow"].get("capital_expenditures", 0) or 0 for q in quarterly),
                "dividends_paid": sum(q["cash_flow"].get("dividends_paid", 0) or 0 for q in quarterly),
            },
            # Balance sheet uses most recent quarter (stock metric, not flow)
            "balance_sheet": quarterly[0]["balance_sheet"].copy(),
            # Shares from most recent quarter
            "shares_outstanding": quarterly[0].get("shares_outstanding"),
            # Metadata
            "quarters_included": len(quarterly),
            "most_recent_quarter": {
                "fiscal_year": quarterly[0]["fiscal_year"],
                "fiscal_period": quarterly[0]["fiscal_period"],
            },
        }

        return ttm

    def get_latest_metrics(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent quarterly metrics.

        Args:
            symbol: Stock ticker symbol
            as_of_date: Only include data filed on or before this date (default: today)

        Returns:
            Most recent quarter's metrics or None if not found
        """
        quarterly = self.get_quarterly_metrics(symbol, as_of_date, num_quarters=1)
        return quarterly[0] if quarterly else None

    def normalize_to_nested_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize flat data to the nested structure expected by valuation models.

        Handles different input formats:
        - Already nested (income_statement/cash_flow/balance_sheet keys)
        - Flat dict with metric keys directly

        Args:
            data: Input data dict (flat or nested)

        Returns:
            Normalized dict with nested structure
        """
        # Check if already nested
        if "income_statement" in data and isinstance(data["income_statement"], dict):
            return data

        # Convert flat to nested
        return {
            "income_statement": {
                "total_revenue": data.get("total_revenue") or data.get("revenue"),
                "net_income": data.get("net_income"),
                "gross_profit": data.get("gross_profit"),
                "operating_income": data.get("operating_income"),
                "interest_expense": data.get("interest_expense"),
                "income_tax_expense": data.get("income_tax_expense"),
                "ebitda": data.get("ebitda"),
            },
            "cash_flow": {
                "operating_cash_flow": data.get("operating_cash_flow"),
                "free_cash_flow": data.get("free_cash_flow"),
                "capital_expenditures": data.get("capital_expenditures") or data.get("capex"),
                "dividends_paid": data.get("dividends_paid") or data.get("dividends"),
            },
            "balance_sheet": {
                "total_assets": data.get("total_assets"),
                "total_liabilities": data.get("total_liabilities"),
                "stockholders_equity": data.get("stockholders_equity") or data.get("equity"),
                "cash_and_equivalents": data.get("cash_and_equivalents") or data.get("cash"),
                "long_term_debt": data.get("long_term_debt"),
                "short_term_debt": data.get("short_term_debt"),
                "current_assets": data.get("current_assets"),
                "current_liabilities": data.get("current_liabilities"),
            },
            "shares_outstanding": data.get("shares_outstanding") or data.get("shares"),
            "fiscal_year": data.get("fiscal_year"),
            "fiscal_period": data.get("fiscal_period"),
        }

    def detect_data_format(self, data: Any) -> str:
        """
        Detect the format of input data.

        Args:
            data: Input data (single dict or list)

        Returns:
            One of: "sec_fy", "quarterly_list", "single_quarter", "flat", "unknown"
        """
        if isinstance(data, list):
            if not data:
                return "empty"
            first = data[0]
            if isinstance(first, dict):
                if first.get("fiscal_period") == "FY":
                    return "sec_fy"
                elif first.get("fiscal_period") in ("Q1", "Q2", "Q3", "Q4"):
                    return "quarterly_list"
        elif isinstance(data, dict):
            if data.get("fiscal_period") == "FY":
                return "sec_fy"
            elif "income_statement" in data:
                return "single_quarter"
            elif "total_revenue" in data or "revenue" in data:
                return "flat"

        return "unknown"

    def _to_float(self, value: Any) -> Optional[float]:
        """Convert value to float, handling None and Decimal."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
