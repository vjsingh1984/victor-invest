#!/usr/bin/env python3
"""
Script to convert all camelCase financial metric keys to snake_case in fundamental_agent.py
This ensures consistency across the entire data pipeline.
"""

import re
from pathlib import Path

# Mapping of camelCase to snake_case for all financial metrics
CAMEL_TO_SNAKE = {
    # Income Statement
    "totalRevenue": "revenues",
    "netIncome": "net_income",
    "grossProfit": "gross_profit",
    "operatingIncome": "operating_income",
    "costOfRevenue": "cost_of_revenue",
    "incomeTaxExpense": "income_tax_expense",
    "pretaxIncome": "pretax_income",
    "comprehensiveIncome": "comprehensive_income",
    "otherIncomeExpense": "other_income_expense",
    # Balance Sheet
    "totalAssets": "total_assets",
    "currentAssets": "current_assets",
    "noncurrentAssets": "noncurrent_assets",
    "totalLiabilities": "total_liabilities",
    "currentLiabilities": "current_liabilities",
    "noncurrentLiabilities": "noncurrent_liabilities",
    "stockholdersEquity": "stockholders_equity",
    "cashAndEquivalents": "cash_and_equivalents",
    "accountsReceivable": "accounts_receivable",
    "inventory": "inventory",
    "propertyPlantEquipment": "property_plant_equipment",
    "goodwill": "goodwill",
    "intangibleAssets": "intangible_assets",
    "accountsPayable": "accounts_payable",
    "accruedLiabilities": "accrued_liabilities",
    "deferredRevenue": "deferred_revenue",
    "longTermDebt": "long_term_debt",
    "shortTermDebt": "short_term_debt",
    "totalDebt": "total_debt",
    # Cash Flow Statement
    "operatingCashFlow": "operating_cash_flow",
    "investingCashFlow": "investing_cash_flow",
    "financingCashFlow": "financing_cash_flow",
    "capitalExpenditures": "capital_expenditures",
    "freeCashFlow": "free_cash_flow",
    # Ratios
    "currentRatio": "current_ratio",
    "quickRatio": "quick_ratio",
    "debtToEquity": "debt_to_equity",
    "debtToAssets": "debt_to_assets",
    "interestCoverage": "interest_coverage",
    "returnOnAssets": "roa",
    "returnOnEquity": "roe",
    "grossMargin": "gross_margin",
    "operatingMargin": "operating_margin",
    "netMargin": "net_margin",
    "assetTurnover": "asset_turnover",
    "inventoryTurnover": "inventory_turnover",
    "receivablesTurnover": "receivables_turnover",
    # Market Data
    "marketCap": "market_cap",
    "currentPrice": "current_price",
    "peRatio": "pe_ratio",
    "pbRatio": "pb_ratio",
    "psRatio": "ps_ratio",
    "priceToBook": "price_to_book",
    "priceToSales": "price_to_sales",
    "earningsPerShare": "earnings_per_share",
    "epsBasic": "eps_basic",
    "epsDiluted": "eps_diluted",
    "dividendYield": "dividend_yield",
    "sharesOutstanding": "shares_outstanding",
    # Growth Metrics
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "assetGrowth": "asset_growth",
    # Other
    "fiscalYear": "fiscal_year",
    "fiscalPeriod": "fiscal_period",
    "dataDate": "data_date",
    "companyName": "company_name",
}


def convert_file(file_path: Path):
    """Convert all camelCase keys to snake_case in the given file"""

    # Read file content
    content = file_path.read_text()
    original_content = content

    # Track replacements
    replacements = {}

    # Convert each camelCase pattern to snake_case
    for camel, snake in CAMEL_TO_SNAKE.items():
        # Pattern 1: Dictionary key access with quotes - 'key' or "key"
        pattern1 = rf"(['\"]){camel}\1"
        matches1 = re.findall(pattern1, content)
        if matches1:
            content = re.sub(pattern1, rf"\1{snake}\1", content)
            replacements[f"'{camel}'"] = len(matches1)

        # Pattern 2: .get('key', ...) or .get("key", ...)
        pattern2 = rf"\.get\((['\"]){camel}\1"
        matches2 = re.findall(pattern2, content)
        if matches2:
            content = re.sub(pattern2, rf".get(\1{snake}\1", content)
            if f"'{camel}'" in replacements:
                replacements[f"'{camel}'"] += len(matches2)
            else:
                replacements[f"'{camel}'"] = len(matches2)

    # Write back if changed
    if content != original_content:
        file_path.write_text(content)
        print(f"✅ Updated {file_path.name}")
        print(f"   Replacements made:")
        for key, count in sorted(replacements.items(), key=lambda x: -x[1])[:10]:
            print(f"     {key}: {count} occurrences")
        print(f"   Total patterns replaced: {sum(replacements.values())}")
        return True
    else:
        print(f"ℹ️  No changes needed in {file_path.name}")
        return False


def main():
    """Convert fundamental_agent.py to snake_case"""

    agent_file = Path("/Users/vijaysingh/code/InvestiGator/agents/fundamental_agent.py")

    print("=" * 80)
    print("Converting camelCase financial keys to snake_case")
    print("=" * 80)
    print()

    if not agent_file.exists():
        print(f"❌ File not found: {agent_file}")
        return 1

    print(f"Processing: {agent_file}")
    print()

    if convert_file(agent_file):
        print()
        print("✅ Conversion completed successfully!")
        print()
        print("Next steps:")
        print("  1. Review the changes: git diff agents/fundamental_agent.py")
        print("  2. Run tests to verify nothing broke")
        print("  3. Test with NEE: python3 cli_orchestrator.py analyze NEE -m standard --force-refresh")
    else:
        print()
        print("ℹ️  File already uses snake_case consistently")

    return 0


if __name__ == "__main__":
    exit(main())
