#!/usr/bin/env python3
"""
Analyze S&P 100 to determine qtrs availability patterns for Income vs Cash Flow statements.

This will inform the optimal data model design for sec_companyfacts_processed table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from collections import defaultdict
from typing import Dict, List, Any

DB_CONFIG = {
    "host": "${DB_HOST:-localhost}",
    "database": "sec_database",
    "user": "investigator",
    "password": "investigator",
}

# Representative tags for each statement type
STATEMENT_TAGS = {
    "income_statement": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "NetIncomeLoss"],
    "cash_flow_statement": ["NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment"],
    "balance_sheet": ["Assets", "Liabilities", "StockholdersEquity"],
}

SP100_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "UNH",
]  # Start with top 10 for faster analysis


def get_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def get_cik_for_symbol(conn, symbol: str) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT cik FROM ticker_cik_mapping WHERE ticker = %s", (symbol,))
    result = cursor.fetchone()
    cursor.close()
    return str(result["cik"]).zfill(10) if result else None


def analyze_statement_qtrs_pattern(conn, cik: str, symbol: str, statement_type: str, tags: List[str]) -> Dict:
    """
    Analyze qtrs availability for a specific statement type across all periods.

    Returns dict with pattern analysis for Q1, Q2, Q3, FY.
    """
    cursor = conn.cursor()

    results = {
        "Q1": {"qtrs_0": 0, "qtrs_1": 0, "qtrs_2": 0, "qtrs_3": 0, "qtrs_4": 0},
        "Q2": {"qtrs_0": 0, "qtrs_1": 0, "qtrs_2": 0, "qtrs_3": 0, "qtrs_4": 0},
        "Q3": {"qtrs_0": 0, "qtrs_1": 0, "qtrs_2": 0, "qtrs_3": 0, "qtrs_4": 0},
        "FY": {"qtrs_0": 0, "qtrs_1": 0, "qtrs_2": 0, "qtrs_3": 0, "qtrs_4": 0},
    }

    tag_placeholders = ",".join(["%s"] * len(tags))
    query = f"""
    SELECT 
        s.fp as fiscal_period,
        n.qtrs,
        COUNT(*) as count
    FROM sec_sub_data s
    JOIN sec_num_data n ON s.adsh = n.adsh
    WHERE s.cik = %s
        AND s.fy = 2024
        AND s.fp IN ('Q1', 'Q2', 'Q3', 'FY')
        AND n.tag IN ({tag_placeholders})
        AND n.ddate = s.period
    GROUP BY s.fp, n.qtrs
    ORDER BY s.fp, n.qtrs
    """

    cursor.execute(query, (cik, *tags))
    rows = cursor.fetchall()

    for row in rows:
        fp = row["fiscal_period"]
        qtrs = row["qtrs"]
        count = row["count"]

        if fp in results:
            results[fp][f"qtrs_{qtrs}"] = count

    cursor.close()
    return results


def main():
    print("=" * 100)
    print("S&P 100 Statement-Level qtrs Pattern Analysis")
    print("=" * 100)
    print()

    conn = get_connection()

    all_results = []

    for symbol in SP100_SYMBOLS:
        cik = get_cik_for_symbol(conn, symbol)
        if not cik:
            print(f"❌ {symbol}: No CIK found")
            continue

        print(f"\n{'='*80}")
        print(f"Analyzing: {symbol} (CIK: {cik})")
        print(f"{'='*80}")

        stock_analysis = {"symbol": symbol, "cik": cik, "statements": {}}

        for stmt_type, tags in STATEMENT_TAGS.items():
            pattern = analyze_statement_qtrs_pattern(conn, cik, symbol, stmt_type, tags)
            stock_analysis["statements"][stmt_type] = pattern

            print(f"\n{stmt_type.upper().replace('_', ' ')}:")
            print(f"  Q1: qtrs=0:{pattern['Q1']['qtrs_0']}, qtrs=1:{pattern['Q1']['qtrs_1']}")
            print(f"  Q2: qtrs=1:{pattern['Q2']['qtrs_1']}, qtrs=2:{pattern['Q2']['qtrs_2']}")
            print(f"  Q3: qtrs=1:{pattern['Q3']['qtrs_1']}, qtrs=3:{pattern['Q3']['qtrs_3']}")
            print(f"  FY: qtrs=4:{pattern['FY']['qtrs_4']}")

        # Determine recommended approach
        income_q2_has_individual = stock_analysis["statements"]["income_statement"]["Q2"]["qtrs_1"] > 0
        cashflow_q2_has_individual = stock_analysis["statements"]["cash_flow_statement"]["Q2"]["qtrs_1"] > 0

        income_q2_has_ytd = stock_analysis["statements"]["income_statement"]["Q2"]["qtrs_2"] > 0
        cashflow_q2_has_ytd = stock_analysis["statements"]["cash_flow_statement"]["Q2"]["qtrs_2"] > 0

        print(f"\n  RECOMMENDATION FOR Q2:")
        if income_q2_has_individual and cashflow_q2_has_individual:
            print(f"    ✅ Use qtrs=1 (both statements have individual quarter values)")
            stock_analysis["q2_recommendation"] = "qtrs_1"
        elif cashflow_q2_has_ytd:
            print(f"    ⚠️  Use qtrs=2 (cash flow only has YTD, income has both)")
            stock_analysis["q2_recommendation"] = "qtrs_2_mixed"
        else:
            print(f"    ❌ Insufficient data")
            stock_analysis["q2_recommendation"] = "insufficient"

        all_results.append(stock_analysis)

    conn.close()

    # Summary statistics
    print(f"\n\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}\n")

    q2_qtrs_1_count = sum(1 for r in all_results if r.get("q2_recommendation") == "qtrs_1")
    q2_qtrs_2_mixed_count = sum(1 for r in all_results if r.get("q2_recommendation") == "qtrs_2_mixed")
    q2_insufficient_count = sum(1 for r in all_results if r.get("q2_recommendation") == "insufficient")

    total = len(all_results)
    print(f"Q2 Pattern Distribution:")
    print(
        f"  ✅ Both statements have qtrs=1 (individual): {q2_qtrs_1_count}/{total} ({q2_qtrs_1_count/total*100:.1f}%)"
    )
    print(
        f"  ⚠️  Mixed (cash flow qtrs=2 YTD only): {q2_qtrs_2_mixed_count}/{total} ({q2_qtrs_2_mixed_count/total*100:.1f}%)"
    )
    print(f"  ❌ Insufficient data: {q2_insufficient_count}/{total}")
    print()

    if q2_qtrs_2_mixed_count / total > 0.8:
        print("📊 CONCLUSION: Majority (>80%) require YTD values for cash flow")
        print("   → Recommended: Store YTD values, track statement-specific qtrs")
    elif q2_qtrs_1_count / total > 0.8:
        print("📊 CONCLUSION: Majority (>80%) have individual quarter values")
        print("   → Recommended: Store individual quarter values (qtrs=1)")
    else:
        print("📊 CONCLUSION: Mixed patterns across companies")
        print("   → Recommended: Store BOTH, with fallback chain")

    # Save detailed results
    output_file = "analysis/sp100_statement_qtrs_patterns.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "analysis_date": "2025-11-04",
                "summary": {
                    "q2_both_individual": q2_qtrs_1_count,
                    "q2_mixed_ytd": q2_qtrs_2_mixed_count,
                    "q2_insufficient": q2_insufficient_count,
                    "total_analyzed": total,
                },
                "stocks": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n📄 Detailed results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
