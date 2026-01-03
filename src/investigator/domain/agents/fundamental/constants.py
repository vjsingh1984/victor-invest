"""Constants used across the fundamental analysis agent modules."""

FALLBACK_CANONICAL_KEYS = [
    "total_revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "short_term_debt",
    "total_debt",
    "operating_cash_flow",
    "capital_expenditures",
    "dividends_paid",
    "free_cash_flow",
    "weighted_average_diluted_shares_outstanding",
]

PROCESSED_ADDITIONAL_FINANCIAL_KEYS = [
    "research_and_development_expense",
    "selling_general_administrative_expense",
    "operating_expenses",
    "interest_expense",
    "income_tax_expense",
    "depreciation_amortization",
    "stock_based_compensation",
    "retained_earnings",
    "accounts_payable",
    "accrued_liabilities",
    "property_plant_equipment",
    "accumulated_depreciation",
    "property_plant_equipment_net",
    "goodwill",
    "intangible_assets",
    "deferred_revenue",
    "treasury_stock",
    "other_comprehensive_income",
    "book_value",
    "book_value_per_share",
    "working_capital",
    "net_debt",
    "dividends_paid",  # CRITICAL FIX: Added for GGM model applicability
    "preferred_stock_dividends",
    "common_stock_dividends",
    "investing_cash_flow",
    "financing_cash_flow",
    "shares_outstanding",
    "earnings_per_share",
    "earnings_per_share_diluted",
    "market_cap",
    "enterprise_value",
    "ebitda",  # CRITICAL FIX: Added for EV/EBITDA model applicability
]

PROCESSED_RATIO_KEYS = [
    "dividend_payout_ratio",
    "dividend_yield",
    "effective_tax_rate",
    "interest_coverage",
    "return_on_assets",
    "return_on_equity",
    "asset_turnover",
]

__all__ = [
    "FALLBACK_CANONICAL_KEYS",
    "PROCESSED_ADDITIONAL_FINANCIAL_KEYS",
    "PROCESSED_RATIO_KEYS",
]
