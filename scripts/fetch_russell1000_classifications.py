#!/usr/bin/env python3
"""
Fetch Russell 1000 sector/industry classifications from stock database.

This script:
1. Reads tickers from data/RUSSELL1000.txt
2. Queries the stock database for sector/industry data
3. Generates resources/russell1000_classifications.json
4. Updates utils/industry_classifier.py with Russell 1000 mappings
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


# Database configuration
DB_CONFIG = {
    "host": "${DB_HOST:-localhost}",
    "port": 5432,
    "database": "stock",
    "user": "stockuser",
    "password": "${STOCK_DB_PASSWORD}",
}

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
RUSSELL1000_FILE = PROJECT_ROOT / "data" / "RUSSELL1000.txt"
OUTPUT_JSON = PROJECT_ROOT / "resources" / "russell1000_classifications.json"


def read_russell1000_tickers() -> List[str]:
    """Read Russell 1000 tickers from file."""
    print(f"📖 Reading tickers from {RUSSELL1000_FILE}")

    with open(RUSSELL1000_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)

    print(f"✅ Found {len(unique_tickers)} unique tickers")
    return unique_tickers


def fetch_classifications_from_db(tickers: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Query stock database for sector/industry classifications.

    Returns:
        Dict mapping ticker -> (sector, industry)
    """
    print(f"\n🔍 Querying stock database for {len(tickers)} tickers...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Build query with all tickers
    ticker_list = ",".join([f"'{t}'" for t in tickers])

    query = f"""
    SELECT
        ticker,
        "Sector" as sector,
        "Industry" as industry,
        sec_sector,
        sec_industry,
        sic_code
    FROM symbol
    WHERE ticker IN ({ticker_list})
    ORDER BY ticker;
    """

    cur.execute(query)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    # Build classification mapping
    classifications = {}

    for row in rows:
        ticker = row["ticker"]

        # Priority: Use "Sector"/"Industry" if available, fall back to sec_sector/sec_industry
        sector = row["sector"] or row["sec_sector"]
        industry = row["industry"] or row["sec_industry"]

        if sector and industry:
            classifications[ticker] = (sector, industry)

    print(f"✅ Found classifications for {len(classifications)} tickers")
    print(f"⚠️  Missing classifications for {len(tickers) - len(classifications)} tickers")

    return classifications


def save_classifications_json(classifications: Dict[str, Tuple[str, str]]) -> None:
    """Save classifications to JSON file."""
    print(f"\n💾 Saving classifications to {OUTPUT_JSON}")

    # Ensure resources directory exists
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output = {
        "source": "stock database (${DB_HOST:-localhost})",
        "description": "Russell 1000 sector/industry classifications",
        "count": len(classifications),
        "classifications": {
            ticker: {"sector": sector, "industry": industry}
            for ticker, (sector, industry) in sorted(classifications.items())
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved {len(classifications)} classifications to JSON")


def generate_symbol_overrides_code(classifications: Dict[str, Tuple[str, str]]) -> str:
    """
    Generate Python code for RUSSELL1000_OVERRIDES dictionary.

    This can be integrated into utils/industry_classifier.py.
    """
    lines = ["# Russell 1000 sector/industry classifications (auto-generated)"]
    lines.append("RUSSELL1000_OVERRIDES = {")

    for ticker, (sector, industry) in sorted(classifications.items()):
        # Escape special characters (single quotes, newlines, etc.)
        sector_escaped = sector.replace("'", "\\'").replace("\n", " ").replace("\r", " ").strip()
        industry_escaped = industry.replace("'", "\\'").replace("\n", " ").replace("\r", " ").strip()
        lines.append(f"    '{ticker}': ('{sector_escaped}', '{industry_escaped}'),")

    lines.append("}")

    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 80)
    print("Russell 1000 Classification Fetcher")
    print("=" * 80)

    # Step 1: Read tickers
    tickers = read_russell1000_tickers()

    # Step 2: Fetch classifications from database
    classifications = fetch_classifications_from_db(tickers)

    # Step 3: Save to JSON
    save_classifications_json(classifications)

    # Step 4: Generate Python code snippet
    print("\n📝 Generating Python code for RUSSELL1000_OVERRIDES...")
    code = generate_symbol_overrides_code(classifications)

    code_file = PROJECT_ROOT / "resources" / "russell1000_overrides.py"
    with open(code_file, "w") as f:
        f.write(code)

    print(f"✅ Saved Python code to {code_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Russell 1000 tickers: {len(tickers)}")
    print(f"Tickers with classifications: {len(classifications)} ({len(classifications)/len(tickers)*100:.1f}%)")
    print(f"Tickers missing classifications: {len(tickers) - len(classifications)}")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {code_file}")
    print("\nNext steps:")
    print("  1. Review the generated files")
    print("  2. Integrate RUSSELL1000_OVERRIDES into utils/industry_classifier.py")
    print("  3. Update IndustryClassifier.classify() to use Russell 1000 mappings")
    print("=" * 80)


if __name__ == "__main__":
    main()
