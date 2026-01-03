#!/usr/bin/env python3
"""
Test S&P 100 Data Extraction

Extracts 2024-FY data for 20 S&P 100 companies to:
1. Validate extraction works across diverse companies
2. Identify new XBRL tag variations not in FAANG
3. Generate test data for integration tests
"""

from dao.sec_bulk_dao import SECBulkDAO
import json
from decimal import Decimal

# S&P 100 sample companies (first 20 with 2024-FY data)
SP100_SAMPLE = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BMY",
    "C",
    "CAT",
    "CHTR",
]


def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


dao = SECBulkDAO()

results = {}
missing_tags = {}  # Track tags we couldn't resolve
coverage_stats = []

print("=" * 80)
print("S&P 100 EXTRACTION TEST - 2024 Annual Filings")
print("=" * 80)
print()

for symbol in SP100_SAMPLE:
    print(f"Extracting {symbol}...", end=" ", flush=True)

    try:
        metrics = dao.fetch_financial_metrics(symbol, 2024, "FY")

        # Check for errors
        if "error" in metrics:
            print(f"❌ ERROR: {metrics['error']}")
            results[symbol] = metrics
            continue

        # Count extracted metrics
        metric_count = len([k for k in metrics.keys() if k not in ["symbol", "fiscal_year", "fiscal_period"]])

        # Check critical metrics
        critical = ["total_revenue", "net_income", "total_assets", "total_liabilities"]
        has_critical = sum(1 for m in critical if m in metrics and metrics[m] is not None)

        print(f"✓ {metric_count} metrics, {has_critical}/4 critical")

        results[symbol] = metrics
        coverage_stats.append(
            {
                "symbol": symbol,
                "total_metrics": metric_count,
                "critical_count": has_critical,
                "has_revenue": "total_revenue" in metrics and metrics["total_revenue"] is not None,
                "has_net_income": "net_income" in metrics and metrics["net_income"] is not None,
                "has_assets": "total_assets" in metrics and metrics["total_assets"] is not None,
                "has_liabilities": "total_liabilities" in metrics and metrics["total_liabilities"] is not None,
            }
        )

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)[:60]}")
        results[symbol] = {"symbol": symbol, "error": str(e)}

print()
print("=" * 80)
print("COVERAGE SUMMARY")
print("=" * 80)

successful = [s for s in coverage_stats if s["total_metrics"] > 0]
print(f"Successful extractions: {len(successful)}/{len(SP100_SAMPLE)}")
print(f"Average metrics per company: {sum(s['total_metrics'] for s in successful) / len(successful):.1f}")
print()

print("Critical Metric Coverage:")
print(f"  Revenue: {sum(1 for s in coverage_stats if s['has_revenue'])}/{len(coverage_stats)}")
print(f"  Net Income: {sum(1 for s in coverage_stats if s['has_net_income'])}/{len(coverage_stats)}")
print(f"  Assets: {sum(1 for s in coverage_stats if s['has_assets'])}/{len(coverage_stats)}")
print(f"  Liabilities: {sum(1 for s in coverage_stats if s['has_liabilities'])}/{len(coverage_stats)}")
print()

# Save results
output_file = "sp100_extraction_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=decimal_to_float)
print(f"Results saved to {output_file}")
print()

# Show companies with low coverage for investigation
low_coverage = [s for s in coverage_stats if s["total_metrics"] < 50 or s["critical_count"] < 3]
if low_coverage:
    print("=" * 80)
    print("COMPANIES WITH LOW COVERAGE (need investigation)")
    print("=" * 80)
    for s in low_coverage:
        print(f"  {s['symbol']}: {s['total_metrics']} metrics, {s['critical_count']}/4 critical")
    print()
