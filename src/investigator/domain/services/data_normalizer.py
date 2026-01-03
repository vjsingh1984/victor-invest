"""
Data Quality Normalization Module

Provides centralized schema normalization and judicious rounding for financial data.
This module ensures consistent field naming (camelCase/snake_case harmonization)
and token-efficient numerical precision across all agents.

Author: InvestiGator Team
Date: 2025-11-02
"""

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Central schema normalizer for harmonizing financial data formats.

    Features:
    - camelCase ↔ snake_case field mapping
    - Judicious rounding (2-4 decimal places) for token efficiency
    - Missing data tracking with explicit warnings
    - Comprehensive field validation
    """

    # Field mapping: snake_case → camelCase (SEC CompanyFacts uses camelCase)
    FIELD_MAPPINGS = {
        # Income Statement
        "total_revenue": "totalRevenue",
        "revenues": "totalRevenue",  # Alias
        "net_income": "netIncome",
        "operating_income": "operatingIncome",
        "gross_profit": "grossProfit",
        "cost_of_revenue": "costOfRevenue",
        "operating_expenses": "operatingExpenses",
        "ebitda": "EBITDA",
        # Balance Sheet
        "total_assets": "totalAssets",
        "assets": "totalAssets",  # Alias
        "current_assets": "currentAssets",
        "assets_current": "currentAssets",  # Alias
        "total_liabilities": "totalLiabilities",
        "liabilities": "totalLiabilities",  # Alias
        "current_liabilities": "currentLiabilities",
        "liabilities_current": "currentLiabilities",  # Alias
        "long_term_debt": "longTermDebt",
        "total_debt": "totalDebt",
        "short_term_debt": "shortTermDebt",
        "debt_current": "shortTermDebt",  # FIXED: Map debt_current to shortTermDebt
        "stockholders_equity": "stockholdersEquity",
        "equity": "stockholdersEquity",  # Alias
        "total_equity": "stockholdersEquity",  # Alias
        "retained_earnings": "retainedEarnings",
        "cash_and_equivalents": "cashAndEquivalents",
        "total_cash": "totalCash",
        "inventory": "inventory",
        "accounts_receivable": "accountsReceivable",
        "accounts_payable": "accountsPayable",
        # Cash Flow
        "operating_cash_flow": "operatingCashFlow",
        "investing_cash_flow": "investingCashFlow",
        "financing_cash_flow": "financingCashFlow",
        "free_cash_flow": "freeCashFlow",
        "capital_expenditures": "capitalExpenditures",
        "dividends_paid": "dividendsPaid",  # For GGM valuation
        "payout_ratio": "payoutRatio",  # Derived: dividends / net_income
        # Per-Share Metrics
        "earnings_per_share": "earningsPerShare",
        "book_value_per_share": "bookValuePerShare",
        "revenue_per_share": "revenuePerShare",
        # Market Data
        "market_cap": "marketCap",
        "shares_outstanding": "sharesOutstanding",
        "enterprise_value": "enterpriseValue",
        "current_price": "currentPrice",
        "price_target": "priceTarget",
        # Valuation Ratios
        "pe_ratio": "peRatio",
        "pb_ratio": "pbRatio",
        "p_e_ratio": "peRatio",  # Alias
        "p_b_ratio": "pbRatio",  # Alias
        "debt_to_equity": "debtToEquity",
        "debt_to_assets": "debtToAssets",
        # Technical Indicators
        "fifty_day_ma": "fiftyDayMA",
        "two_hundred_day_ma": "twoHundredDayMA",
        "sma_50": "sma50",
        "sma_200": "sma200",
        # Growth Metrics
        "revenue_growth": "revenueGrowth",
        "earnings_growth": "earningsGrowth",
        # Margins
        "operating_margin": "operatingMargin",
        "profit_margin": "profitMargin",
        "gross_margin": "grossMargin",
        # Other
        "fiscal_period": "fiscalPeriod",
        "fiscal_year": "fiscalYear",
    }

    # Reverse mapping: camelCase → snake_case
    REVERSE_MAPPINGS = {v: k for k, v in FIELD_MAPPINGS.items()}

    # Core metrics that MUST be present for quality assessment
    CORE_METRICS = [
        "totalRevenue",
        "netIncome",
        "totalAssets",
        "totalLiabilities",
        "currentAssets",
        "currentLiabilities",
        "stockholdersEquity",
    ]

    # Debt metrics for enhanced completeness scoring (snake_case - canonical format)
    DEBT_METRICS = ["total_debt", "long_term_debt", "short_term_debt", "current_liabilities"]

    # Ratios that should warn if zeroed due to missing data
    CRITICAL_RATIOS = [
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "debt_to_assets",
        "roe",
        "roa",
        "operating_cash_flow",
        "free_cash_flow",
    ]

    @classmethod
    def normalize_field_names(cls, data: Dict[str, Any], to_camel_case: bool = True) -> Dict[str, Any]:
        """
        Normalize field names between snake_case and camelCase.

        Args:
            data: Dictionary with financial data
            to_camel_case: If True, convert to camelCase; if False, convert to snake_case

        Returns:
            Dictionary with normalized field names
        """
        if not data:
            return {}

        mapping = cls.FIELD_MAPPINGS if to_camel_case else cls.REVERSE_MAPPINGS
        normalized = {}

        for key, value in data.items():
            # Map known fields
            new_key = mapping.get(key, key)

            # Recursively normalize nested dictionaries
            if isinstance(value, dict):
                normalized[new_key] = cls.normalize_field_names(value, to_camel_case)
            else:
                normalized[new_key] = value

        return normalized

    @classmethod
    def round_number(cls, value: Any, decimal_places: int = 2) -> Optional[float]:
        """
        Apply judicious rounding to a numerical value.

        Rounding guidelines:
        - Prices/monetary values: 2 decimal places
        - Ratios/percentages: 2-4 decimal places
        - Large numbers (>1M): Round to whole numbers

        Args:
            value: Numerical value to round
            decimal_places: Number of decimal places (default: 2)

        Returns:
            Rounded float or None if value is None/invalid
        """
        if value is None:
            return None

        # Handle boolean values (convert to int: True→1, False→0)
        if isinstance(value, bool):
            return float(int(value))

        # Handle NaN values (return None for missing/invalid data)
        try:
            import math

            if isinstance(value, float) and math.isnan(value):
                return None
        except (TypeError, ValueError):
            pass  # Not a float or can't check NaN, continue with normal processing

        # Handle pandas NaN
        try:
            import pandas as pd

            if pd.isna(value):
                return None
        except (ImportError, TypeError, ValueError):
            pass  # pandas not available or can't check, continue

        try:
            # Convert to Decimal for precise rounding
            num = Decimal(str(value))

            # Special handling for very large numbers
            if abs(num) >= 1_000_000:
                # Round to whole numbers for millions/billions
                return float(num.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

            # Determine appropriate decimal places
            if decimal_places is None:
                # Auto-detect based on magnitude
                if abs(num) >= 100:
                    decimal_places = 2
                elif abs(num) >= 1:
                    decimal_places = 2
                else:
                    decimal_places = 4  # For ratios < 1

            # Create quantizer (e.g., '0.01' for 2 decimal places)
            quantizer = Decimal(10) ** -decimal_places
            rounded = num.quantize(quantizer, rounding=ROUND_HALF_UP)

            return float(rounded)

        except (ValueError, TypeError, ArithmeticError) as e:
            logger.warning(f"Failed to round value {value}: {e}")
            return None

    @classmethod
    def round_financial_data(cls, data: Dict[str, Any], config: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Apply judicious rounding to all numerical values in financial data.

        Args:
            data: Dictionary with financial data
            config: Optional custom rounding config {field_pattern: decimal_places}

        Returns:
            Dictionary with rounded values
        """
        if not data:
            return {}

        # Default rounding configuration
        default_config = {
            "price": 2,
            "ratio": 2,
            "margin": 4,
            "percentage": 2,
            "eps": 2,
            "market_cap": 0,
            "revenue": 0,
            "cash": 0,
            "debt": 0,
            "assets": 0,
        }

        rounding_config = {**default_config, **(config or {})}
        rounded = {}

        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively round nested dictionaries
                rounded[key] = cls.round_financial_data(value, config)
            elif isinstance(value, (int, float, Decimal)):
                # Determine decimal places based on key name
                decimal_places = 2  # default

                key_lower = key.lower()
                if any(pattern in key_lower for pattern in ["price", "eps", "book_value"]):
                    decimal_places = 2
                elif any(pattern in key_lower for pattern in ["margin", "yield", "turnover"]):
                    decimal_places = 4
                elif any(pattern in key_lower for pattern in ["ratio", "roe", "roa"]):
                    decimal_places = 2
                elif any(
                    pattern in key_lower
                    for pattern in ["market_cap", "revenue", "cash", "debt", "assets", "liabilities", "equity"]
                ):
                    decimal_places = 0

                rounded[key] = cls.round_number(value, decimal_places)
            else:
                rounded[key] = value

        return rounded

    @classmethod
    def assess_completeness(cls, data: Dict[str, Any], include_debt_metrics: bool = True) -> Dict[str, Any]:
        """
        Assess data completeness with enhanced debt metrics tracking.

        Args:
            data: Financial data dictionary (canonical snake_case format)
            include_debt_metrics: Whether to include debt metrics in scoring

        Returns:
            Dictionary with completeness assessment:
            {
                'score': float (0-100),
                'core_metrics_count': int,
                'core_metrics_total': int,
                'debt_metrics_count': int,
                'debt_metrics_total': int,
                'missing_core': List[str],
                'missing_debt': List[str],
                'quality_level': str
            }
        """
        # CRITICAL FIX: Check snake_case keys (canonical format) first, then fall back to camelCase
        # This ensures we prioritize canonical data format as defined in FIELD_NAME_MAP

        # Helper function to check for a key in both snake_case and camelCase formats
        def get_value_flexible(data_dict: Dict[str, Any], snake_case_key: str) -> Any:
            """Try snake_case first (canonical), then camelCase (for backwards compatibility)"""
            # First try snake_case (canonical format)
            if snake_case_key in data_dict:
                return data_dict.get(snake_case_key)

            # Fall back to camelCase if defined in FIELD_MAPPINGS
            camel_case_key = cls.FIELD_MAPPINGS.get(snake_case_key)
            if camel_case_key and camel_case_key in data_dict:
                return data_dict.get(camel_case_key)

            return None

        # Check core metrics
        missing_core = []
        present_core = 0

        for metric in cls.CORE_METRICS:
            value = get_value_flexible(data, metric) if metric.islower() or "_" in metric else data.get(metric)
            if value is not None and value != 0:
                present_core += 1
            else:
                missing_core.append(metric)

        core_completeness = (present_core / len(cls.CORE_METRICS)) * 100 if cls.CORE_METRICS else 100

        # Check debt metrics if requested
        debt_completeness = 100  # Default
        missing_debt = []
        present_debt = 0

        if include_debt_metrics:
            for metric in cls.DEBT_METRICS:
                value = get_value_flexible(data, metric)
                if value is not None and value != 0:
                    present_debt += 1
                else:
                    missing_debt.append(metric)

            debt_completeness = (present_debt / len(cls.DEBT_METRICS)) * 100 if cls.DEBT_METRICS else 100

        # Calculate overall score (weighted: 70% core, 30% debt)
        overall_score = (core_completeness * 0.7) + (debt_completeness * 0.3)

        # Determine quality level
        if overall_score >= 90:
            quality_level = "Excellent"
        elif overall_score >= 75:
            quality_level = "Good"
        elif overall_score >= 50:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        return {
            "score": round(overall_score, 1),
            "core_metrics_count": present_core,
            "core_metrics_total": len(cls.CORE_METRICS),
            "debt_metrics_count": present_debt,
            "debt_metrics_total": len(cls.DEBT_METRICS),
            "missing_core": missing_core,
            "missing_debt": missing_debt,
            "quality_level": quality_level,
        }

    @classmethod
    def validate_and_warn(
        cls, ratios: Dict[str, Any], symbol: str, logger_instance: Optional[logging.Logger] = None
    ) -> None:
        """
        Validate critical ratios and log explicit warnings when they're zeroed due to missing data.

        Args:
            ratios: Dictionary of calculated ratios
            symbol: Stock symbol for logging context
            logger_instance: Optional logger instance (uses module logger if None)
        """
        log = logger_instance or logger

        for ratio_name in cls.CRITICAL_RATIOS:
            value = ratios.get(ratio_name)

            # Check if ratio is zero or None (indicating missing upstream data)
            if value is None or value == 0:
                # Convert to human-readable name
                readable_name = ratio_name.replace("_", " ").title()

                # Log explicit warning
                log.warning(
                    f"⚠️  UPSTREAM DATA GAP for {symbol}: {readable_name} is {value} "
                    f"(likely due to missing financial data). This may affect analysis quality."
                )

    @classmethod
    def normalize_and_round(cls, data: Dict[str, Any], to_camel_case: bool = True) -> Dict[str, Any]:
        """
        Convenience method: normalize field names AND apply judicious rounding.

        Args:
            data: Financial data dictionary
            to_camel_case: If True, convert to camelCase; if False, convert to snake_case

        Returns:
            Dictionary with normalized and rounded data
        """
        # Step 1: Normalize field names
        normalized = cls.normalize_field_names(data, to_camel_case)

        # Step 2: Apply judicious rounding
        rounded = cls.round_financial_data(normalized)

        return rounded


# Convenience functions for common operations
def normalize_financials(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize financial data to camelCase with rounding."""
    return DataNormalizer.normalize_and_round(data, to_camel_case=True)


def round_for_prompt(value: Any, decimal_places: int = 2) -> Optional[float]:
    """Round a single value for use in LLM prompts (token efficiency)."""
    return DataNormalizer.round_number(value, decimal_places)


def assess_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess financial data completeness including debt metrics."""
    return DataNormalizer.assess_completeness(data, include_debt_metrics=True)
