#!/usr/bin/env python3
"""
Test script for ExecutiveSummaryGenerator with database persistence

Usage:
    python3 scripts/test_executive_summary.py results/NEE_FINAL_FIX.json
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.executive_summary_generator import generate_executive_summary_from_file
from core.ollama_client import OllamaClient


async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/test_executive_summary.py <input_json_file>")
        print("Example: python3 scripts/test_executive_summary.py results/NEE_FINAL_FIX.json")
        sys.exit(1)

    input_file = sys.argv[1]
    symbol = Path(input_file).stem.split("_")[0]  # Extract symbol from filename
    output_file = f"results/{symbol}_summary.json"

    print(f"Generating executive summary for {symbol}...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()

    # Initialize Ollama client
    ollama = OllamaClient()

    # Database configuration (matching SECBulkDAO expectations)
    db_config = {
        "host": "${DB_HOST:-localhost}",
        "port": 5432,
        "database": "sec_database",
        "username": "investigator",  # SECBulkDAO expects 'username', not 'user'
        "password": "investigator",
    }

    # Generate summary with database persistence
    try:
        summary = await generate_executive_summary_from_file(
            input_file=input_file,
            output_file=output_file,
            ollama_client=ollama,
            db_config=db_config,  # Enable database persistence
            fiscal_year=2024,
            fiscal_period="Q3",
        )

        print("\n✅ Executive Summary Generated Successfully!")
        print(f"   Original size: {summary['original_size_mb']:.2f}MB")
        print(f"   Summary size: {summary['summary_size_kb']:.2f}KB")
        print(
            f"   Compression: {100 * summary['summary_size_kb'] / (summary['original_size_mb'] * 1024):.1f}% of original"
        )
        print()
        print(f"📄 Summary saved to: {output_file}")
        print(f"💾 Summary saved to database: quarterly_ai_summaries table")
        print()
        print("Key Metrics:")
        print(f"  - Recommendation: {summary['recommendation']['action']}")
        print(f"  - Conviction: {summary['recommendation']['conviction']}")
        print(f"  - Price Target: ${summary['recommendation']['price_target']:.2f}")
        print(f"  - Current Price: ${summary['recommendation']['current_price']:.2f}")
        print(f"  - Upside Potential: {summary['recommendation']['upside_potential']:.1f}%")

    except Exception as e:
        print(f"\n❌ Failed to generate summary: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
