"""Summary and extraction helpers for fundamental agent payload shaping."""

from __future__ import annotations

from typing import Any, Dict, List


def get_historical_trend(financials: Dict[str, Any]) -> Dict[str, Any]:
    """Get historical financial trends."""
    _ = financials  # Placeholder until multi-year extraction is implemented.
    return {
        "revenue_trend": [100, 110, 121, 133],
        "earnings_trend": [10, 12, 15, 18],
        "years": [2021, 2022, 2023, 2024],
    }


def summarize_company_data(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary of company data for report."""
    financials = company_data.get("financials") or {}
    market_data = company_data.get("market_data", {})

    market_cap = company_data.get("market_cap", 0) or market_data.get("market_cap", 0) or 0

    return {
        "symbol": company_data["symbol"],
        "market_cap": market_cap,
        "price": market_data.get("price", 0) or company_data.get("current_price", 0),
        "revenue": financials.get("revenues") or 0,
        "net_income": financials.get("net_income") or 0,
        "total_assets": financials.get("total_assets") or 0,
        "total_equity": financials.get("stockholders_equity") or 0,
    }


def extract_latest_financials(quarterly_data: List[Any]) -> Dict[str, Any]:
    """Extract latest financial statement from quarterly data.

    Supports both legacy dict payloads and object payloads with `financial_data`.
    """
    if not quarterly_data or len(quarterly_data) == 0:
        return {}

    latest_quarter = quarterly_data[0] if isinstance(quarterly_data, list) else quarterly_data

    if hasattr(latest_quarter, "financial_data"):
        financial_data = latest_quarter.financial_data

        def safe_get(statement_dict, key, default=0):
            if statement_dict and isinstance(statement_dict, dict):
                return statement_dict.get(key, default)
            return default

        income_stmt = financial_data.income_statement if hasattr(financial_data, "income_statement") else {}
        balance_sheet = financial_data.balance_sheet if hasattr(financial_data, "balance_sheet") else {}
        cash_flow = financial_data.cash_flow_statement if hasattr(financial_data, "cash_flow_statement") else {}
        quarterly = financial_data.quarterly_data if hasattr(financial_data, "quarterly_data") else {}

        operating_income = safe_get(income_stmt, "operating_income") or safe_get(quarterly, "operating_income")
        depreciation_amortization = safe_get(cash_flow, "depreciation_amortization") or safe_get(
            quarterly, "depreciation_amortization"
        )

        ebitda = 0
        if operating_income and depreciation_amortization:
            ebitda = operating_income + depreciation_amortization
        elif operating_income:
            ebitda = operating_income

        return {
            "revenues": safe_get(income_stmt, "revenue")
            or safe_get(income_stmt, "revenues")
            or safe_get(quarterly, "revenue"),
            "net_income": safe_get(income_stmt, "net_income")
            or safe_get(income_stmt, "earnings")
            or safe_get(quarterly, "net_income"),
            "total_assets": safe_get(balance_sheet, "total_assets") or safe_get(quarterly, "total_assets"),
            "total_liabilities": safe_get(balance_sheet, "total_liabilities")
            or safe_get(quarterly, "total_liabilities"),
            "stockholders_equity": safe_get(balance_sheet, "stockholders_equity")
            or safe_get(balance_sheet, "shareholderEquity")
            or safe_get(quarterly, "stockholders_equity"),
            "total_debt": safe_get(balance_sheet, "total_debt")
            or safe_get(balance_sheet, "long_term_debt")
            or safe_get(quarterly, "total_debt"),
            "cash": safe_get(balance_sheet, "cash")
            or safe_get(balance_sheet, "cash_and_equivalents")
            or safe_get(quarterly, "cash"),
            "current_assets": safe_get(balance_sheet, "current_assets") or safe_get(quarterly, "current_assets"),
            "current_liabilities": safe_get(balance_sheet, "current_liabilities")
            or safe_get(quarterly, "current_liabilities"),
            "gross_profit": safe_get(income_stmt, "gross_profit") or safe_get(quarterly, "gross_profit"),
            "operating_income": operating_income,
            "depreciation_amortization": depreciation_amortization,
            "ebitda": ebitda,
            "operating_cash_flow": safe_get(cash_flow, "operating_cash_flow")
            or safe_get(quarterly, "operating_cash_flow"),
            "capital_expenditures": safe_get(cash_flow, "capital_expenditures")
            or safe_get(quarterly, "capital_expenditures"),
            "free_cash_flow": safe_get(cash_flow, "free_cash_flow") or safe_get(quarterly, "free_cash_flow"),
            "inventory": safe_get(balance_sheet, "inventory") or safe_get(quarterly, "inventory"),
            "cost_of_revenue": safe_get(income_stmt, "cost_of_revenue") or safe_get(quarterly, "cost_of_revenue"),
            "dividends": safe_get(cash_flow, "dividends") or safe_get(quarterly, "dividends"),
            "shares_outstanding": safe_get(balance_sheet, "shares_outstanding")
            or safe_get(quarterly, "shares_outstanding"),
        }

    if isinstance(latest_quarter, dict):
        income_stmt = latest_quarter.get("income_statement", {})
        cash_flow = latest_quarter.get("cash_flow", {})
        balance_sheet = latest_quarter.get("balance_sheet", {})

        operating_income = latest_quarter.get("operating_income", 0) or income_stmt.get("operating_income", 0)
        depreciation_amortization = latest_quarter.get("depreciation_amortization", 0) or cash_flow.get(
            "depreciation_amortization", 0
        )

        ebitda = 0
        if operating_income and depreciation_amortization:
            ebitda = operating_income + depreciation_amortization
        elif operating_income:
            ebitda = operating_income

        return {
            "revenues": latest_quarter.get("revenue", 0)
            or latest_quarter.get("revenues", 0)
            or income_stmt.get("total_revenue", 0),
            "net_income": latest_quarter.get("net_income", 0)
            or latest_quarter.get("earnings", 0)
            or income_stmt.get("net_income", 0),
            "total_assets": latest_quarter.get("total_assets", 0) or balance_sheet.get("total_assets", 0),
            "total_liabilities": latest_quarter.get("total_liabilities", 0) or balance_sheet.get("total_liabilities", 0),
            "stockholders_equity": latest_quarter.get("stockholders_equity", 0)
            or latest_quarter.get("shareholderEquity", 0)
            or balance_sheet.get("stockholders_equity", 0),
            "total_debt": latest_quarter.get("total_debt", 0)
            or latest_quarter.get("long_term_debt", 0)
            or balance_sheet.get("total_debt", 0),
            "cash": latest_quarter.get("cash", 0)
            or latest_quarter.get("cash_and_equivalents", 0)
            or balance_sheet.get("cash_and_equivalents", 0),
            "current_assets": latest_quarter.get("current_assets", 0) or balance_sheet.get("current_assets", 0),
            "current_liabilities": latest_quarter.get("current_liabilities", 0)
            or balance_sheet.get("current_liabilities", 0),
            "gross_profit": latest_quarter.get("gross_profit", 0) or income_stmt.get("gross_profit", 0),
            "operating_income": operating_income,
            "depreciation_amortization": depreciation_amortization,
            "ebitda": ebitda,
            "operating_cash_flow": latest_quarter.get("operating_cash_flow", 0)
            or cash_flow.get("operating_cash_flow", 0),
            "capital_expenditures": latest_quarter.get("capital_expenditures", 0)
            or cash_flow.get("capital_expenditures", 0),
            "free_cash_flow": latest_quarter.get("free_cash_flow", 0) or cash_flow.get("free_cash_flow", 0),
            "inventory": latest_quarter.get("inventory", 0) or balance_sheet.get("inventory", 0),
            "cost_of_revenue": latest_quarter.get("cost_of_revenue", 0) or income_stmt.get("cost_of_revenue", 0),
            "dividends": latest_quarter.get("dividends", 0) or cash_flow.get("dividends_paid", 0),
            "shares_outstanding": latest_quarter.get("shares_outstanding", 0)
            or balance_sheet.get("shares_outstanding", 0),
        }

    return {}
