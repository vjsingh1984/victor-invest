#!/usr/bin/env python3
"""
SEC Data Extractor - Bulk fetch and process SEC filings for symbols.

This script extracts SEC Company Facts data for symbols that don't have
quarterly financial data in the database yet.

Usage:
    # Extract SEC data for Russell 1000
    python3 scripts/sec_data_extractor.py --russell1000 --batch-size 10

    # Extract for specific symbols
    python3 scripts/sec_data_extractor.py --symbols AAPL MSFT GOOGL

    # Extract for all stocks missing SEC data
    python3 scripts/sec_data_extractor.py --missing-only --batch-size 20

Author: Victor-Invest Team
Date: 2026-01-05
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from investigator.infrastructure.database.symbol_repository import SymbolRepository
from investigator.infrastructure.http.api_client import SECAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SECDataExtractor:
    """Extracts SEC Company Facts data from SEC EDGAR API and saves to database."""

    def __init__(self):
        """Initialize the SEC data extractor."""
        self.symbol_repo = SymbolRepository()
        self.sec_engine = self.symbol_repo.sec_engine
        self.stock_engine = self.symbol_repo.stock_engine

        # Get SEC User Agent from environment (REQUIRED by SEC)
        user_agent = os.environ.get("SEC_USER_AGENT", "InvestiGator/1.0 (contact@example.com)")
        print(f"  Using SEC User-Agent: {user_agent}", flush=True)

        # Initialize the actual SEC API client that makes HTTP requests
        self.sec_api_client = SECAPIClient(user_agent=user_agent)

        # Import SEC extraction components for saving data
        from investigator.infrastructure.sec.companyfacts_extractor import SECCompanyFactsExtractor
        self.facts_extractor = SECCompanyFactsExtractor()

    def get_symbols_missing_sec_data(self, order_by: str = "stockid") -> List[str]:
        """Get symbols that have CIK but no SEC quarterly data, ordered by stockid."""
        # Get all symbols with CIK from stock database (ordered by stockid)
        all_with_cik = self.symbol_repo.get_all_symbols(us_only=True, order_by=order_by)

        # Get symbols that already have quarterly data
        domestic_filers = self.symbol_repo.get_domestic_filers()

        # Return those missing SEC data, preserving the order
        missing = [s for s in all_with_cik if s not in domestic_filers]
        return missing

    def get_cik_for_symbol(self, symbol: str) -> Optional[str]:
        """Get CIK for a symbol from stock database."""
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text("SELECT cik FROM symbol WHERE ticker = :symbol"),
                {"symbol": symbol},
            )
            row = result.fetchone()
            return str(row[0]).zfill(10) if row and row[0] else None

    async def extract_for_symbol(self, symbol: str) -> dict:
        """
        Extract SEC data for a single symbol from SEC EDGAR API.

        Steps:
        1. Get CIK for the symbol
        2. Fetch company facts from SEC EDGAR API (HTTP request)
        3. Save raw data to sec_companyfacts_raw table
        4. Trigger processing to populate sec_companyfacts_processed

        Returns:
            dict with 'success', 'symbol', 'error', 'quarters_extracted'
        """
        start_time = time.time()
        result = {
            "success": False,
            "symbol": symbol,
            "error": None,
            "quarters_extracted": 0,
            "duration": 0,
        }

        try:
            # Get CIK
            cik = self.get_cik_for_symbol(symbol)
            if not cik:
                result["error"] = "No CIK found"
                return result

            # Fetch company facts directly from SEC EDGAR API (actual HTTP request)
            logger.info(f"  Fetching from SEC EDGAR API for {symbol} (CIK: {cik})...")
            try:
                api_data = self.sec_api_client.get_company_facts(cik)
            except Exception as api_error:
                # Check for 404 (foreign filer or non-existent)
                if "404" in str(api_error):
                    result["error"] = "Not a domestic filer (404)"
                else:
                    result["error"] = f"API error: {str(api_error)[:80]}"
                return result

            if not api_data or "facts" not in api_data:
                result["error"] = "No company facts in API response"
                return result

            # Check if us-gaap data exists (domestic filers only)
            us_gaap = api_data.get("facts", {}).get("us-gaap", {})
            if not us_gaap:
                result["error"] = "No us-gaap data (likely foreign filer)"
                return result

            # Count the number of tags (rough indicator of data richness)
            tag_count = len(us_gaap)
            logger.info(f"    ✓ Received {tag_count} us-gaap tags from SEC API")

            # Save raw data to database using the facts_extractor's save method
            save_data = {
                "symbol": symbol.upper(),
                "cik": api_data.get("cik", cik),
                "entityName": api_data.get("entityName", ""),
                "facts": api_data.get("facts", {}),
            }

            raw_data_id = self.facts_extractor._save_to_database(save_data)

            if raw_data_id:
                result["success"] = True
                result["quarters_extracted"] = tag_count  # Use tag count as proxy
                logger.info(f"    ✓ Saved to database (raw_data_id: {raw_data_id})")
            else:
                result["error"] = "Failed to save to database"

            result["duration"] = time.time() - start_time

        except Exception as e:
            result["error"] = str(e)[:100]
            logger.error(f"  Error extracting {symbol}: {e}")
            import traceback
            traceback.print_exc()

        return result

    async def extract_batch(
        self,
        symbols: List[str],
        batch_size: int = 10,
        delay_between_batches: float = 5.0,
    ) -> dict:
        """
        Extract SEC data for multiple symbols in batches.

        Args:
            symbols: List of symbols to extract
            batch_size: Number of symbols per batch
            delay_between_batches: Seconds to wait between batches (SEC rate limiting)

        Returns:
            Summary dict with counts and errors
        """
        total = len(symbols)
        success_count = 0
        error_count = 0
        errors = []

        print(f"\nExtracting SEC data for {total} symbols...", flush=True)
        print(f"  Batch size: {batch_size}", flush=True)
        print(f"  Delay between batches: {delay_between_batches}s", flush=True)
        print("=" * 60, flush=True)

        for i in range(0, total, batch_size):
            batch_symbols = symbols[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"\nBatch {batch_num}/{total_batches}: {', '.join(batch_symbols)}", flush=True)

            for idx, symbol in enumerate(batch_symbols):
                result = await self.extract_for_symbol(symbol)

                if result["success"]:
                    success_count += 1
                    print(
                        f"  ✓ {symbol}: {result['quarters_extracted']} tags ({result['duration']:.1f}s)",
                        flush=True,
                    )
                else:
                    error_count += 1
                    errors.append(f"{symbol}: {result['error']}")
                    print(f"  ✗ {symbol}: {result['error']}", flush=True)

                # Small delay between symbols within batch (SEC rate limit: 10 req/sec)
                if idx < len(batch_symbols) - 1:
                    await asyncio.sleep(0.5)

            # Progress
            processed = min(i + batch_size, total)
            print(
                f"  Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"Success: {success_count} | Errors: {error_count}",
                flush=True,
            )

            # Rate limit delay (SEC allows 10 requests/second)
            if i + batch_size < total:
                await asyncio.sleep(delay_between_batches)

        print("\n" + "=" * 60, flush=True)
        print("SEC DATA EXTRACTION COMPLETE", flush=True)
        print("=" * 60, flush=True)
        print(f"  Total symbols: {total}", flush=True)
        print(f"  Successful: {success_count}", flush=True)
        print(f"  Errors: {error_count}", flush=True)

        if errors:
            print(f"\nErrors ({len(errors)}):", flush=True)
            for err in errors[:20]:
                print(f"  - {err}", flush=True)
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more", flush=True)

        return {
            "total": total,
            "success": success_count,
            "errors": error_count,
            "error_details": errors,
        }


def main():
    parser = argparse.ArgumentParser(description="SEC Data Extractor - Bulk fetch SEC filings")

    # Symbol source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--russell1000", action="store_true", help="Extract for Russell 1000 symbols")
    source_group.add_argument("--sp500", action="store_true", help="Extract for S&P 500 symbols")
    source_group.add_argument("--all", action="store_true", help="Extract for ALL stocks with CIK")
    source_group.add_argument("--missing-only", action="store_true", help="Only extract for symbols missing SEC data")
    source_group.add_argument("--symbols", nargs="+", help="Specific symbols to extract")

    # Processing options
    parser.add_argument("--batch-size", type=int, default=10, help="Symbols per batch (default: 10)")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between batches in seconds (default: 5.0)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols that already have SEC data")
    parser.add_argument(
        "--order-by",
        choices=["stockid", "mktcap", "ticker"],
        default="stockid",
        help="Sort order: stockid (ascending), mktcap (descending), ticker (alphabetical). Default: stockid",
    )

    args = parser.parse_args()

    print("SEC Data Extractor", flush=True)
    print("=" * 60, flush=True)

    extractor = SECDataExtractor()

    # Get symbols based on source
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        print(f"  Processing {len(symbols)} specified symbols", flush=True)
    elif args.russell1000:
        symbols = extractor.symbol_repo.get_russell1000_symbols()
        print(f"  Found {len(symbols)} Russell 1000 symbols", flush=True)
    elif args.sp500:
        symbols = extractor.symbol_repo.get_sp500_symbols()
        print(f"  Found {len(symbols)} S&P 500 symbols", flush=True)
    elif args.all:
        symbols = extractor.symbol_repo.get_all_symbols(us_only=True, order_by=args.order_by)
        print(f"  Found {len(symbols)} total US stocks with CIK (order: {args.order_by})", flush=True)
    elif args.missing_only:
        symbols = extractor.get_symbols_missing_sec_data(order_by=args.order_by)
        print(f"  Found {len(symbols)} symbols missing SEC data (order: {args.order_by})", flush=True)

    # Filter to skip existing if requested
    if args.skip_existing and not args.missing_only:
        existing = extractor.symbol_repo.get_domestic_filers()
        original_count = len(symbols)
        symbols = [s for s in symbols if s not in existing]
        print(f"  Skipping {original_count - len(symbols)} symbols with existing data", flush=True)
        print(f"  Processing {len(symbols)} symbols", flush=True)

    if not symbols:
        print("  No symbols to process!", flush=True)
        return

    # Run extraction
    asyncio.run(
        extractor.extract_batch(
            symbols=symbols,
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
        )
    )


if __name__ == "__main__":
    main()
