# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
TTM Calculator - Consistent trailing twelve months calculation regardless of input format.

This service handles the key drift point between data sources:
- SEC Filing Tool returns single FY snapshot (already represents ~TTM)
- Database returns quarterly data (needs 4 quarters summed for TTM)

The calculator detects the input format and applies the appropriate logic.

Example:
    from investigator.domain.services.valuation_shared import TTMCalculator

    calc = TTMCalculator()

    # From SEC FY data (single snapshot = TTM)
    sec_data = [{"fiscal_period": "FY", "income_statement": {"total_revenue": 400000}, ...}]
    ttm = calc.calculate_ttm(sec_data)

    # From quarterly data (sum 4 quarters)
    quarterly_data = [{"fiscal_period": "Q1", ...}, {"fiscal_period": "Q2", ...}, ...]
    ttm = calc.calculate_ttm(quarterly_data)
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TTMCalculator:
    """
    Calculator for trailing twelve months metrics.

    Handles format detection and consistent TTM calculation regardless
    of whether input is SEC FY data or quarterly data.
    """

    # Flow metrics that should be summed for TTM
    FLOW_METRICS = {
        "income_statement": [
            "total_revenue",
            "net_income",
            "gross_profit",
            "operating_income",
            "interest_expense",
            "income_tax_expense",
            "ebitda",
        ],
        "cash_flow": [
            "operating_cash_flow",
            "free_cash_flow",
            "capital_expenditures",
            "dividends_paid",
        ],
    }

    # Stock metrics that use most recent value (not summed)
    STOCK_METRICS = {
        "balance_sheet": [
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "cash_and_equivalents",
            "long_term_debt",
            "short_term_debt",
            "current_assets",
            "current_liabilities",
        ],
    }

    def detect_data_format(self, data: Union[Dict, List]) -> str:
        """
        Detect the format of input financial data.

        Args:
            data: Input data (single dict or list of dicts)

        Returns:
            One of:
            - "sec_fy": SEC annual filing (FY period) - treat as TTM
            - "quarterly": Quarterly data - sum 4 quarters for TTM
            - "ttm": Already calculated TTM (has quarters_included key)
            - "nested_single": Single period with nested structure
            - "flat": Flat structure with direct metric keys
            - "empty": Empty or None input
            - "unknown": Unrecognized format
        """
        if data is None:
            return "empty"

        if isinstance(data, list):
            if not data:
                return "empty"

            first = data[0] if data else {}
            if not isinstance(first, dict):
                return "unknown"

            fiscal_period = first.get("fiscal_period", "")
            if fiscal_period == "FY":
                return "sec_fy"
            elif fiscal_period in ("Q1", "Q2", "Q3", "Q4"):
                return "quarterly"
            elif "income_statement" in first:
                return "nested_single"
            else:
                return "flat"

        elif isinstance(data, dict):
            # Check if already TTM
            if "quarters_included" in data:
                return "ttm"

            fiscal_period = data.get("fiscal_period", "")
            if fiscal_period == "FY":
                return "sec_fy"
            elif fiscal_period in ("Q1", "Q2", "Q3", "Q4"):
                return "quarterly"
            elif "income_statement" in data:
                return "nested_single"
            elif "total_revenue" in data or "revenue" in data:
                return "flat"

        return "unknown"

    def calculate_ttm(
        self,
        data: Union[Dict, List[Dict]],
        require_full_year: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate TTM metrics from input data.

        Handles different formats:
        - SEC FY: Returns the FY data as TTM (already represents ~12 months)
        - Quarterly list: Sums the last 4 quarters
        - Single quarter: Returns as-is with warning
        - Already TTM: Returns as-is

        Args:
            data: Input financial data (dict or list of dicts)
            require_full_year: If True, returns empty dict if < 4 quarters available

        Returns:
            Dict with TTM metrics in nested structure:
            {
                "income_statement": {...},
                "cash_flow": {...},
                "balance_sheet": {...},
                "shares_outstanding": float,
                "data_format": str,
                "quarters_included": int,
            }
        """
        data_format = self.detect_data_format(data)

        if data_format == "empty":
            return {}

        if data_format == "ttm":
            # Already TTM, return as-is
            return data

        if data_format == "sec_fy":
            # SEC FY data - treat as TTM
            return self._fy_to_ttm(data)

        if data_format == "quarterly":
            # Sum 4 quarters
            return self._quarterly_to_ttm(data, require_full_year)

        if data_format == "nested_single":
            # Single nested record - return with warning
            logger.warning("Single quarter data provided, returning as-is")
            if isinstance(data, list):
                data = data[0]
            data["quarters_included"] = 1
            data["data_format"] = "single_quarter"
            return data

        if data_format == "flat":
            # Flat structure - normalize and return
            return self._flat_to_ttm(data)

        logger.warning(f"Unknown data format: {data_format}")
        return {}

    def _fy_to_ttm(self, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Convert SEC FY data to TTM format.

        FY data already represents ~12 months, so we just normalize the structure.
        """
        # Handle list with single FY record
        if isinstance(data, list):
            if not data:
                return {}
            data = data[0]

        # If already nested structure
        if "income_statement" in data and isinstance(data["income_statement"], dict):
            result = {
                "income_statement": data.get("income_statement", {}).copy(),
                "cash_flow": data.get("cash_flow", {}).copy(),
                "balance_sheet": data.get("balance_sheet", {}).copy(),
                "shares_outstanding": data.get("shares_outstanding"),
                "fiscal_year": data.get("fiscal_year"),
                "fiscal_period": "TTM",
                "data_format": "sec_fy",
                "quarters_included": 4,  # FY = 4 quarters
            }
        else:
            # Flat structure from FY record
            result = {
                "income_statement": {
                    "total_revenue": data.get("total_revenue"),
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
                    "capital_expenditures": data.get("capital_expenditures"),
                    "dividends_paid": data.get("dividends_paid"),
                },
                "balance_sheet": {
                    "total_assets": data.get("total_assets"),
                    "total_liabilities": data.get("total_liabilities"),
                    "stockholders_equity": data.get("stockholders_equity"),
                    "cash_and_equivalents": data.get("cash_and_equivalents"),
                    "long_term_debt": data.get("long_term_debt"),
                    "short_term_debt": data.get("short_term_debt"),
                    "current_assets": data.get("current_assets"),
                    "current_liabilities": data.get("current_liabilities"),
                },
                "shares_outstanding": data.get("shares_outstanding"),
                "fiscal_year": data.get("fiscal_year"),
                "fiscal_period": "TTM",
                "data_format": "sec_fy",
                "quarters_included": 4,
            }

        return result

    def _quarterly_to_ttm(
        self,
        quarters: List[Dict],
        require_full_year: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate TTM by summing 4 quarters of data.

        Flow metrics are summed, stock metrics use most recent value.
        """
        if not quarters:
            return {}

        # Take up to 4 most recent quarters
        recent_quarters = quarters[:4] if len(quarters) >= 4 else quarters

        if require_full_year and len(recent_quarters) < 4:
            logger.warning(f"Insufficient quarters for TTM: {len(recent_quarters)} < 4")
            return {}

        # Sum flow metrics
        ttm_income = {}
        for metric in self.FLOW_METRICS["income_statement"]:
            values = []
            for q in recent_quarters:
                if "income_statement" in q:
                    val = q["income_statement"].get(metric)
                else:
                    val = q.get(metric)
                if val is not None:
                    values.append(float(val))
            ttm_income[metric] = sum(values) if values else None

        ttm_cash_flow = {}
        for metric in self.FLOW_METRICS["cash_flow"]:
            values = []
            for q in recent_quarters:
                if "cash_flow" in q:
                    val = q["cash_flow"].get(metric)
                else:
                    val = q.get(metric)
                if val is not None:
                    values.append(float(val))
            ttm_cash_flow[metric] = sum(values) if values else None

        # Stock metrics use most recent quarter
        most_recent = recent_quarters[0]
        if "balance_sheet" in most_recent:
            ttm_balance_sheet = most_recent["balance_sheet"].copy()
        else:
            ttm_balance_sheet = {metric: most_recent.get(metric) for metric in self.STOCK_METRICS["balance_sheet"]}

        return {
            "income_statement": ttm_income,
            "cash_flow": ttm_cash_flow,
            "balance_sheet": ttm_balance_sheet,
            "shares_outstanding": most_recent.get("shares_outstanding"),
            "fiscal_year": most_recent.get("fiscal_year"),
            "fiscal_period": "TTM",
            "data_format": "quarterly",
            "quarters_included": len(recent_quarters),
            "most_recent_quarter": {
                "fiscal_year": most_recent.get("fiscal_year"),
                "fiscal_period": most_recent.get("fiscal_period"),
            },
        }

    def _flat_to_ttm(self, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Convert flat structure to TTM format.

        Assumes flat data represents TTM values already.
        """
        if isinstance(data, list):
            if not data:
                return {}
            data = data[0]

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
            "fiscal_period": "TTM",
            "data_format": "flat",
            "quarters_included": 4,  # Assume flat data represents TTM
        }

    def get_metric(
        self,
        ttm_data: Dict[str, Any],
        metric_name: str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """
        Extract a specific metric from TTM data.

        Searches in order: income_statement, cash_flow, balance_sheet, top-level.

        Args:
            ttm_data: TTM data dict
            metric_name: Metric to extract
            default: Default value if not found

        Returns:
            Metric value or default
        """
        # Check nested sections first
        for section in ["income_statement", "cash_flow", "balance_sheet"]:
            if section in ttm_data and isinstance(ttm_data[section], dict):
                if metric_name in ttm_data[section]:
                    return ttm_data[section][metric_name]

        # Check top-level
        if metric_name in ttm_data:
            return ttm_data[metric_name]

        return default
