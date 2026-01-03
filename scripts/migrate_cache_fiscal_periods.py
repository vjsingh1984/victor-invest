#!/usr/bin/env python3
"""
Cache Migration Script - Add Fiscal Periods to Cache Keys

Migrates existing cache files to include fiscal period in filenames.
This prevents different fiscal quarters from overwriting each other.

Phase 2.4 of Implementation Plan

Usage:
    python3 scripts/migrate_cache_fiscal_periods.py --dry-run  # Preview changes
    python3 scripts/migrate_cache_fiscal_periods.py           # Execute migration
    python3 scripts/migrate_cache_fiscal_periods.py --symbol AAPL  # Migrate specific symbol
"""

import argparse
import json
import gzip
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CacheMigrator:
    """Migrates cache files to include fiscal periods"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {"files_scanned": 0, "files_migrated": 0, "files_skipped": 0, "errors": 0}

    def migrate_all_caches(self, symbol: str = None):
        """
        Migrate all cache directories

        Args:
            symbol: Optional specific symbol to migrate (None = all symbols)
        """
        cache_dirs = [
            ("data/llm_cache", self._migrate_llm_cache),
            ("data/sec_cache/facts/processed", self._migrate_sec_facts),
            ("data/technical_cache", self._migrate_technical_cache),
        ]

        logger.info(f"Starting cache migration {'(DRY RUN)' if self.dry_run else ''}")
        if symbol:
            logger.info(f"Migrating only symbol: {symbol}")

        for cache_dir, migrate_func in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                logger.info(f"\nMigrating {cache_dir}...")
                migrate_func(cache_path, symbol)
            else:
                logger.warning(f"Cache directory not found: {cache_dir}")

        self._print_summary()

    def _migrate_llm_cache(self, cache_path: Path, symbol_filter: str = None):
        """Migrate LLM cache files to include fiscal periods"""
        for symbol_dir in cache_path.iterdir():
            if not symbol_dir.is_dir():
                continue

            symbol = symbol_dir.name
            if symbol_filter and symbol != symbol_filter:
                continue

            logger.info(f"  Processing symbol: {symbol}")

            for cache_file in symbol_dir.glob("*.json.gz"):
                self.stats["files_scanned"] += 1

                # Check if already has fiscal period
                if self._has_fiscal_period_in_name(cache_file.name):
                    logger.debug(f"    Skipping {cache_file.name} (already has period)")
                    self.stats["files_skipped"] += 1
                    continue

                # Determine fiscal period from file content or date
                fiscal_period = self._determine_fiscal_period(cache_file, symbol)

                if fiscal_period == "unknown":
                    logger.warning(f"    Cannot determine fiscal period for {cache_file.name}")
                    self.stats["files_skipped"] += 1
                    continue

                # Generate new filename
                new_filename = self._add_fiscal_period_to_filename(cache_file.name, fiscal_period)

                new_path = symbol_dir / new_filename

                logger.info(f"    {cache_file.name} → {new_filename}")

                if not self.dry_run:
                    try:
                        cache_file.rename(new_path)
                        self.stats["files_migrated"] += 1
                    except Exception as e:
                        logger.error(f"    Failed to rename {cache_file.name}: {e}")
                        self.stats["errors"] += 1
                else:
                    self.stats["files_migrated"] += 1

    def _migrate_sec_facts(self, cache_path: Path, symbol_filter: str = None):
        """Migrate SEC company facts cache files"""
        for symbol_dir in cache_path.rglob("*"):
            if not symbol_dir.is_dir():
                continue

            # Symbol directories might be nested (e.g., facts/{symbol}/)
            symbol = symbol_dir.name
            if symbol_filter and symbol != symbol_filter:
                continue

            for cache_file in symbol_dir.glob("companyfacts_*.json.gz"):
                self.stats["files_scanned"] += 1

                if self._has_fiscal_period_in_name(cache_file.name):
                    self.stats["files_skipped"] += 1
                    continue

                fiscal_period = self._determine_fiscal_period(cache_file, symbol)

                if fiscal_period == "unknown":
                    self.stats["files_skipped"] += 1
                    continue

                # For companyfacts, format is: companyfacts_{cik}_{period}.json.gz
                parts = cache_file.stem.replace(".json", "").split("_")
                if len(parts) >= 2:
                    cik = parts[1]
                    new_filename = f"companyfacts_{cik}_{fiscal_period}.json.gz"
                else:
                    new_filename = f"companyfacts_{fiscal_period}.json.gz"

                new_path = symbol_dir / new_filename

                logger.info(f"    {cache_file.name} → {new_filename}")

                if not self.dry_run:
                    try:
                        cache_file.rename(new_path)
                        self.stats["files_migrated"] += 1
                    except Exception as e:
                        logger.error(f"    Failed: {e}")
                        self.stats["errors"] += 1
                else:
                    self.stats["files_migrated"] += 1

    def _migrate_technical_cache(self, cache_path: Path, symbol_filter: str = None):
        """Migrate technical analysis cache files"""
        # Technical cache usually isn't period-specific, but migrate for consistency
        for symbol_dir in cache_path.iterdir():
            if not symbol_dir.is_dir():
                continue

            symbol = symbol_dir.name
            if symbol_filter and symbol != symbol_filter:
                continue

            logger.info(f"  Technical cache for {symbol}: No migration needed (not period-specific)")
            # Technical data is typically for a date range, not fiscal periods

    def _has_fiscal_period_in_name(self, filename: str) -> bool:
        """Check if filename already contains fiscal period"""
        # Patterns: 2024-Q1, 2024-Q2, 2025-Q4, etc.
        import re

        pattern = r"\d{4}-(Q[1-4]|FY)"
        return bool(re.search(pattern, filename))

    def _determine_fiscal_period(self, cache_file: Path, symbol: str) -> str:
        """
        Determine fiscal period from cache file content or modification time

        Args:
            cache_file: Path to cache file
            symbol: Stock symbol

        Returns:
            Fiscal period string (e.g., '2025-Q4') or 'unknown'
        """
        try:
            # Try to extract from file content
            with gzip.open(cache_file, "rt") as f:
                data = json.load(f)

                # Look for fiscal period indicators
                if isinstance(data, dict):
                    # Check direct fiscal_period field
                    if "fiscal_period" in data:
                        return data["fiscal_period"]

                    # Check for fiscal_year and fiscal_period separately
                    if "fiscal_year" in data and "fiscal_period" in data:
                        fy = data["fiscal_year"]
                        fp = data["fiscal_period"]
                        if fp in ["Q1", "Q2", "Q3", "Q4", "FY"]:
                            return f"{fy}-{fp}"

                    # Check nested response/analysis data
                    if "response" in data:
                        response = data["response"]
                        if isinstance(response, dict):
                            if "fiscal_period" in response:
                                return response["fiscal_period"]

        except Exception as e:
            logger.debug(f"Could not parse {cache_file.name} for fiscal period: {e}")

        # Fallback: Use file modification time to estimate quarter
        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            year = mtime.year
            month = mtime.month

            if month <= 3:
                quarter = "Q1"
            elif month <= 6:
                quarter = "Q2"
            elif month <= 9:
                quarter = "Q3"
            else:
                quarter = "Q4"

            logger.debug(
                f"Using file mtime for {cache_file.name}: {year}-{quarter} " f"(mtime: {mtime.strftime('%Y-%m-%d')})"
            )

            return f"{year}-{quarter}"

        except Exception as e:
            logger.warning(f"Could not determine period for {cache_file.name}: {e}")
            return "unknown"

    def _add_fiscal_period_to_filename(self, filename: str, fiscal_period: str) -> str:
        """
        Add fiscal period to filename

        Examples:
            fundamental_analysis.json.gz → fundamental_analysis_2025-Q4.json.gz
            companyfacts_0000320193.json.gz → companyfacts_0000320193_2025-Q4.json.gz
        """
        # Remove .json.gz extension
        base = filename.replace(".json.gz", "")

        # Add fiscal period before extension
        return f"{base}_{fiscal_period}.json.gz"

    def _print_summary(self):
        """Print migration summary"""
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files scanned:    {self.stats['files_scanned']}")
        logger.info(f"Files migrated:   {self.stats['files_migrated']}")
        logger.info(f"Files skipped:    {self.stats['files_skipped']}")
        logger.info(f"Errors:           {self.stats['errors']}")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("\n✓ DRY RUN COMPLETE - No files were actually modified")
            logger.info("  Run without --dry-run to execute migration")
        else:
            logger.info("\n✓ MIGRATION COMPLETE")

            if self.stats["errors"] > 0:
                logger.warning(f"⚠  {self.stats['errors']} errors occurred during migration")
            else:
                logger.info("  All files migrated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Migrate cache files to include fiscal periods in filenames")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without actually renaming files")
    parser.add_argument("--symbol", type=str, help="Migrate only specific symbol (e.g., AAPL)")

    args = parser.parse_args()

    migrator = CacheMigrator(dry_run=args.dry_run)
    migrator.migrate_all_caches(symbol=args.symbol)


if __name__ == "__main__":
    main()
