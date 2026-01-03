#!/usr/bin/env python3
"""
Analyze investigator_v2.sh logs for sector-specific coverage.

Checks for:
1. SEC data extraction success
2. Tag coverage for sector-specific concepts
3. LLM analysis completion
4. Any errors or warnings specific to the sector
"""

import re
import sys
from pathlib import Path

SECTORS = {
    "AAPL": "Technology - Consumer Electronics",
    "NEE": "Utilities - Electric Utility",
    "AMT": "Real Estate - REIT (Telecom Towers)",
    "JPM": "Financials - Banking",
    "XOM": "Energy - Oil & Gas",
}


def analyze_log(symbol, sector):
    """Analyze a single log file for a sector."""
    log_file = Path(f"run_{symbol}.log")

    if not log_file.exists():
        print(f"\n{'='*80}")
        print(f"{symbol} ({sector})")
        print(f"{'='*80}")
        print(f"❌ Log file not found: {log_file}")
        return

    content = log_file.read_text()

    print(f"\n{'='*80}")
    print(f"{symbol} ({sector})")
    print(f"{'='*80}")

    # Check SEC data extraction
    sec_extraction = re.search(r"Extracting SEC company facts.*?(?:✅|❌|ERROR)", content, re.DOTALL)
    if sec_extraction:
        print(f"✅ SEC extraction found")
    else:
        print(f"⏳ SEC extraction in progress or not started")

    # Check for revenue extraction (sector-specific)
    if sector.startswith("Utilities"):
        if "RegulatedAndUnregulatedOperatingRevenue" in content:
            print(f"✅ Utility-specific revenue tag found: RegulatedAndUnregulatedOperatingRevenue")
        elif "total_revenue" in content and "None" not in content:
            print(f"✅ Revenue extracted")
        else:
            print(f"⚠️  Revenue extraction unclear")
    elif sector.startswith("Financials"):
        if "InterestAndDividendIncomeOperating" in content or "InterestIncomeOperating" in content:
            print(f"✅ Financial institution revenue tags found")
        elif "total_revenue" in content:
            print(f"✅ Revenue extracted")
    else:
        if "total_revenue" in content:
            print(f"✅ Revenue extraction attempted")

    # Check for warnings/errors
    error_count = len(re.findall(r"ERROR", content))
    warning_count = len(re.findall(r"WARNING", content))

    if error_count > 0:
        print(f"⚠️  Errors found: {error_count}")
        # Show first few errors
        errors = re.findall(r".*ERROR.*", content)[:3]
        for err in errors:
            print(f"    - {err[:120]}...")

    if warning_count > 10:  # Only report if significant warnings
        print(f"⚠️  Warnings found: {warning_count}")
        # Show unique warning types
        warnings = set(re.findall(r"WARNING - (.*)", content))
        for warn in list(warnings)[:5]:
            print(f"    - {warn[:100]}...")

    # Check LLM analysis completion
    llm_patterns = {
        "fundamental_growth": r"fundamental_growth_analysis",
        "profitability": r"profitability_analysis",
        "balance_sheet": r"balance_sheet_analysis",
        "forecast": r"fundamental_forecast",
    }

    print(f"\nLLM Analysis Components:")
    for name, pattern in llm_patterns.items():
        if re.search(pattern, content):
            # Check if there are JSON extraction failures for this type
            context = re.findall(rf"{pattern}.*?(?:Failed to extract JSON|✅)", content, re.DOTALL)
            if context and "Failed to extract JSON" in str(context):
                print(f"  ⚠️  {name}: JSON extraction issues")
            else:
                print(f"  ✅ {name}: Found")
        else:
            print(f"  ⏳ {name}: Not found")

    # Check for data quality metrics
    quality_match = re.search(r"Overall Quality: (\w+) \(([0-9.]+)%\)", content)
    if quality_match:
        quality_level, quality_pct = quality_match.groups()
        print(f"\n📊 Data Quality: {quality_level} ({quality_pct}%)")

    # Check for critical metrics
    critical_metrics = ["total_revenue", "net_income", "total_assets", "total_liabilities"]
    metrics_found = []
    for metric in critical_metrics:
        if re.search(rf"'{metric}':\s*[0-9]", content) or re.search(rf'"{metric}":\s*[0-9]', content):
            metrics_found.append(metric)

    if metrics_found:
        print(f"✅ Critical metrics found: {len(metrics_found)}/4 ({', '.join(metrics_found)})")

    # Sector-specific checks
    print(f"\nSector-Specific Checks:")
    if sector.startswith("Utilities"):
        if "RegulatoryAssets" in content or "RegulatoryLiability" in content:
            print(f"  ✅ Utility regulatory items detected")
        if "TemporaryEquity" in content:
            print(f"  ✅ Complex equity structure detected (expected for utilities)")
    elif sector.startswith("Real Estate"):
        if "REIT" in content or "NoncontrollingInterest" in content:
            print(f"  ✅ REIT-specific items detected")
        if "TemporaryEquity" in content or "RedeemableNoncontrollingInterest" in content:
            print(f"  ✅ Complex REIT equity structure detected")
    elif sector.startswith("Financials"):
        if "InterestIncome" in content or "InterestExpense" in content:
            print(f"  ✅ Banking-specific interest income/expense detected")
    elif sector.startswith("Energy"):
        if "PropertyPlantAndEquipment" in content:
            print(f"  ✅ Capital-intensive assets detected (expected for energy)")

    # Check file size
    file_size_kb = log_file.stat().st_size / 1024
    print(f"\n📁 Log file size: {file_size_kb:.1f} KB")

    # Estimate completion
    if "✅ Analysis complete" in content or "SYNTHESIS COMPLETE" in content:
        print(f"✅ Analysis COMPLETE")
    elif file_size_kb > 100:
        print(f"⏳ Analysis in progress (substantial data)")
    elif file_size_kb < 10:
        print(f"⚠️  Analysis may have failed early")


def main():
    print("=" * 80)
    print("SECTOR-SPECIFIC LOG ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing investigator_v2.sh runs for each sector...")

    for symbol, sector in SECTORS.items():
        analyze_log(symbol, sector)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
