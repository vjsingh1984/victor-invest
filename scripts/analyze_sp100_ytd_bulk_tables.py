#!/usr/bin/env python3
"""
Analyze S&P 100 stocks from SEC bulk tables to verify YTD vs point-in-time patterns.

This script queries sec_sub_data and sec_num_data joined with symbol mappings to:
1. Extract last 8 quarters for each stock
2. Analyze key XBRL tags (Revenue, OCF, COGS, Assets, Liabilities, Equity)
3. Determine if Q2/Q3 are YTD cumulative or point-in-time
4. Identify if pattern is consistent across all companies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from decimal import Decimal
from typing import Dict, List, Any
from datetime import datetime

# Database connection
DB_CONFIG = {
    "host": "${DB_HOST:-localhost}",
    "database": "sec_database",
    "user": "investigator",
    "password": "investigator",
}

# S&P 100 symbols (top 100 by market cap as of 2024)
SP100_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK.B",
    "LLY",
    "AVGO",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "WMT",
    "MA",
    "JNJ",
    "PG",
    "HD",
    "COST",
    "ORCL",
    "ABBV",
    "CVX",
    "MRK",
    "BAC",
    "CRM",
    "KO",
    "NFLX",
    "PEP",
    "AMD",
    "TMO",
    "ADBE",
    "ACN",
    "MCD",
    "CSCO",
    "LIN",
    "ABT",
    "WFC",
    "INTU",
    "DHR",
    "PM",
    "CMCSA",
    "GE",
    "TXN",
    "QCOM",
    "IBM",
    "VZ",
    "NEE",
    "CAT",
    "UNP",
    "AMGN",
    "HON",
    "RTX",
    "SPGI",
    "LOW",
    "COP",
    "BA",
    "GS",
    "AXP",
    "DE",
    "MS",
    "BLK",
    "ELV",
    "AMAT",
    "PLD",
    "SBUX",
    "ISRG",
    "T",
    "GILD",
    "SYK",
    "ADI",
    "BKNG",
    "VRTX",
    "MMC",
    "TJX",
    "REGN",
    "ADP",
    "CI",
    "MDLZ",
    "PGR",
    "SCHW",
    "CB",
    "LRCX",
    "BMY",
    "MO",
    "SO",
    "NOW",
    "C",
    "BSX",
    "ZTS",
    "ETN",
    "FI",
    "PANW",
    "DUK",
    "WM",
    "MU",
    "CME",
    "EOG",
    "ITW",
    "USB",
]

# Key XBRL tags to analyze
XBRL_TAGS = {
    "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "COGS": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "Equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "CapEx": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def get_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def get_symbol_cik_mapping(conn, symbols: List[str]) -> Dict[str, str]:
    """Get CIK for each symbol from ticker_cik_mapping."""
    cursor = conn.cursor()

    query = """
    SELECT DISTINCT ticker as symbol, cik
    FROM ticker_cik_mapping
    WHERE ticker = ANY(%s)
    ORDER BY ticker
    """

    cursor.execute(query, (symbols,))
    results = cursor.fetchall()

    mapping = {row["symbol"]: str(row["cik"]).zfill(10) for row in results}
    cursor.close()

    print(f"✅ Found CIK mappings for {len(mapping)}/{len(symbols)} symbols")
    return mapping


def get_quarterly_data(conn, cik: str, symbol: str, num_quarters: int = 8) -> List[Dict[str, Any]]:
    """
    Get last N quarters of data from sec_sub_data and sec_num_data.

    Returns list of quarters with:
    - adsh, cik, symbol, fiscal_year, fiscal_period, form, filed_date
    - XBRL tag values for Revenue, OCF, COGS, Assets, Liabilities, Equity
    """
    cursor = conn.cursor()

    # Get submission metadata for last 8 quarters
    sub_query = """
    SELECT
        adsh,
        cik,
        name,
        form,
        period,
        fy as fiscal_year,
        fp as fiscal_period,
        filed
    FROM sec_sub_data
    WHERE cik = %s
        AND form IN ('10-Q', '10-K')
        AND fp IN ('Q1', 'Q2', 'Q3', 'FY')
    ORDER BY filed DESC, period DESC
    LIMIT %s
    """

    cursor.execute(sub_query, (cik, num_quarters * 2))  # Fetch more to ensure we get enough
    submissions = cursor.fetchall()

    if not submissions:
        cursor.close()
        return []

    quarters = []
    seen_periods = set()

    for sub in submissions:
        period_key = (sub["fiscal_year"], sub["fiscal_period"])

        # Skip duplicates (sometimes multiple filings for same period)
        if period_key in seen_periods:
            continue

        seen_periods.add(period_key)

        # Get numeric data for this submission
        quarter_data = {
            "adsh": sub["adsh"],
            "cik": cik,
            "symbol": symbol,
            "fiscal_year": sub["fiscal_year"],
            "fiscal_period": sub["fiscal_period"],
            "form": sub["form"],
            "filed": str(sub["filed"]),
            "period_end": str(sub["period"]),
            "company_name": sub["name"],
        }

        # Query each XBRL tag category
        for category, tags in XBRL_TAGS.items():
            tag_placeholders = ",".join(["%s"] * len(tags))
            num_query = f"""
            SELECT tag, value, ddate, qtrs
            FROM sec_num_data
            WHERE adsh = %s
                AND tag IN ({tag_placeholders})
                AND ddate = %s
            ORDER BY
                CASE
                    WHEN qtrs = 0 THEN 1  -- Point-in-time (balance sheet)
                    WHEN qtrs = 1 THEN 2  -- 1 quarter duration
                    WHEN qtrs = 2 THEN 3  -- 2 quarters (Q2 YTD)
                    WHEN qtrs = 3 THEN 4  -- 3 quarters (Q3 YTD)
                    WHEN qtrs = 4 THEN 5  -- Full year
                    ELSE 99
                END
            LIMIT 1
            """

            cursor.execute(num_query, (sub["adsh"], *tags, sub["period"]))
            result = cursor.fetchone()

            if result:
                quarter_data[category] = {
                    "value": float(result["value"]) if result["value"] else None,
                    "tag": result["tag"],
                    "qtrs": result["qtrs"],  # Duration in quarters
                    "ddate": str(result["ddate"]),
                }
            else:
                quarter_data[category] = None

        quarters.append(quarter_data)

        if len(quarters) >= num_quarters:
            break

    cursor.close()
    return quarters


def analyze_ytd_pattern(quarters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze if Q2/Q3 values are YTD cumulative or point-in-time.

    For income statement/cash flow items (Revenue, OCF, COGS):
    - qtrs=1: Individual quarter (point-in-time)
    - qtrs=2: Q2 YTD cumulative (Q1+Q2)
    - qtrs=3: Q3 YTD cumulative (Q1+Q2+Q3)
    - qtrs=4: Full year

    For balance sheet items (Assets, Liabilities, Equity):
    - qtrs=0: Always point-in-time snapshot
    """

    analysis = {"is_consistent_ytd": True, "q2_pattern": None, "q3_pattern": None, "details": []}

    for q in quarters:
        fp = q["fiscal_period"]

        if fp == "Q2":
            # Check Revenue qtrs field
            if q.get("Revenue"):
                qtrs = q["Revenue"]["qtrs"]
                pattern = "YTD" if qtrs == 2 else "Point-in-time" if qtrs == 1 else f"Unknown (qtrs={qtrs})"
                analysis["q2_pattern"] = pattern
                analysis["details"].append(
                    {
                        "period": f"{q['fiscal_year']}-Q2",
                        "revenue_qtrs": qtrs,
                        "revenue_value": q["Revenue"]["value"],
                        "pattern": pattern,
                    }
                )

        elif fp == "Q3":
            # Check Revenue qtrs field
            if q.get("Revenue"):
                qtrs = q["Revenue"]["qtrs"]
                pattern = "YTD" if qtrs == 3 else "Point-in-time" if qtrs == 1 else f"Unknown (qtrs={qtrs})"
                analysis["q3_pattern"] = pattern
                analysis["details"].append(
                    {
                        "period": f"{q['fiscal_year']}-Q3",
                        "revenue_qtrs": qtrs,
                        "revenue_value": q["Revenue"]["value"],
                        "pattern": pattern,
                    }
                )

    # Determine if pattern is consistent
    if analysis["q2_pattern"] and "YTD" not in analysis["q2_pattern"]:
        analysis["is_consistent_ytd"] = False
    if analysis["q3_pattern"] and "YTD" not in analysis["q3_pattern"]:
        analysis["is_consistent_ytd"] = False

    return analysis


def main():
    """Main execution."""
    print("=" * 80)
    print("S&P 100 SEC Bulk Table Analysis - YTD vs Point-in-Time Pattern")
    print("=" * 80)
    print()

    conn = get_connection()

    # Get CIK mappings
    print("📊 Step 1: Fetching CIK mappings for S&P 100 stocks...")
    symbol_cik_map = get_symbol_cik_mapping(conn, SP100_SYMBOLS)
    print()

    # Analyze each stock
    print("📊 Step 2: Analyzing quarterly data patterns...")
    print()

    results = []
    ytd_consistent_count = 0
    point_in_time_count = 0
    mixed_pattern_count = 0
    no_data_count = 0

    for i, symbol in enumerate(SP100_SYMBOLS[:20], 1):  # Start with first 20 stocks
        if symbol not in symbol_cik_map:
            print(f"{i:3d}. {symbol:5s} - ❌ No CIK mapping found")
            no_data_count += 1
            continue

        cik = symbol_cik_map[symbol]

        # Get quarterly data
        quarters = get_quarterly_data(conn, cik, symbol, num_quarters=8)

        if not quarters:
            print(f"{i:3d}. {symbol:5s} - ❌ No quarterly data found")
            no_data_count += 1
            continue

        # Analyze YTD pattern
        analysis = analyze_ytd_pattern(quarters)

        # Categorize
        if analysis["q2_pattern"] == "YTD" and analysis["q3_pattern"] == "YTD":
            ytd_consistent_count += 1
            status = "✅ YTD Consistent"
        elif analysis["q2_pattern"] and "Point-in-time" in analysis["q2_pattern"]:
            point_in_time_count += 1
            status = "⚠️  Point-in-time"
        else:
            mixed_pattern_count += 1
            status = "🔶 Mixed/Unknown"

        print(
            f"{i:3d}. {symbol:5s} - {status} | Q2: {analysis['q2_pattern']} | Q3: {analysis['q3_pattern']} | Quarters: {len(quarters)}"
        )

        results.append(
            {
                "symbol": symbol,
                "cik": cik,
                "quarters_found": len(quarters),
                "q2_pattern": analysis["q2_pattern"],
                "q3_pattern": analysis["q3_pattern"],
                "is_consistent_ytd": analysis["is_consistent_ytd"],
                "details": analysis["details"],
                "sample_quarters": quarters[:3],  # Include first 3 quarters as examples
            }
        )

    conn.close()

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ YTD Consistent (Q2/Q3 are YTD cumulative): {ytd_consistent_count}")
    print(f"⚠️  Point-in-time (Q2/Q3 are individual quarters): {point_in_time_count}")
    print(f"🔶 Mixed/Unknown pattern: {mixed_pattern_count}")
    print(f"❌ No data: {no_data_count}")
    print()

    # Conclusion
    total_with_data = ytd_consistent_count + point_in_time_count + mixed_pattern_count
    if total_with_data > 0:
        ytd_pct = (ytd_consistent_count / total_with_data) * 100
        print(f"📊 Pattern Consistency: {ytd_pct:.1f}% of stocks follow YTD pattern in Q2/Q3")
        print()

        if ytd_pct > 90:
            print("✅ CONCLUSION: Q2/Q3 are CONSISTENTLY YTD cumulative across S&P 100")
            print("   → Fix should infer is_ytd=True for ALL Q2/Q3 periods")
        elif ytd_pct < 10:
            print("⚠️  CONCLUSION: Q2/Q3 are CONSISTENTLY point-in-time (NOT YTD)")
            print("   → Current assumption is WRONG, no YTD conversion needed")
        else:
            print("🔶 CONCLUSION: Mixed pattern - company-specific handling required")
            print(f"   → Need to check 'qtrs' field: qtrs=2 (Q2 YTD), qtrs=3 (Q3 YTD)")

    # Save results
    output_file = "analysis/sp100_ytd_bulk_analysis.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "analysis_date": datetime.now().isoformat(),
                "summary": {
                    "ytd_consistent": ytd_consistent_count,
                    "point_in_time": point_in_time_count,
                    "mixed": mixed_pattern_count,
                    "no_data": no_data_count,
                },
                "stocks": results,
            },
            f,
            indent=2,
        )

    print()
    print(f"📄 Detailed results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
