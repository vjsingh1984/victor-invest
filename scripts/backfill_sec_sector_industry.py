#!/usr/bin/env python3
"""
Backfill sec_sector and sec_industry columns in symbol table.

Strategy:
1. Update sec_sector/sec_industry from Sector/Industry columns (Yahoo Finance data)
2. Additionally backfill from sector_mapping.json for missing symbols
3. Log all updates for verification

Usage:
    python3 scripts/backfill_sec_sector_industry.py [--dry-run] [--limit N]
"""

import json
import logging
import os
import sys
from typing import Dict, Optional

from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_stock_engine():
    """Get SQLAlchemy engine for stock database."""
    # Stock database credentials (from config)
    db_host = "${DB_HOST:-localhost}"
    db_port = 5432
    db_name = "stock"  # Database name is "stock", not "stock_database"
    db_user = "stockuser"
    db_password = os.environ.get("STOCK_DB_PASSWORD", "")

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)


def load_sector_mapping() -> Dict[str, str]:
    """Load peer group sector mapping from JSON."""
    mapping_file = "data/sector_mapping.json"

    if not os.path.exists(mapping_file):
        logger.warning(f"Sector mapping file not found: {mapping_file}")
        return {}

    try:
        with open(mapping_file, "r") as f:
            sector_mapping = json.load(f)

        logger.info(f"Loaded {len(sector_mapping)} symbols from sector_mapping.json")
        return sector_mapping
    except Exception as e:
        logger.error(f"Error loading sector mapping: {e}")
        return {}


def backfill_from_yahoo(engine, dry_run=False, limit=None):
    """
    Backfill sec_sector/sec_industry from Sector/Industry columns.

    Args:
        engine: SQLAlchemy engine
        dry_run: If True, only report what would be updated
        limit: Optional limit on rows to process
    """
    logger.info("=" * 60)
    logger.info("Step 1: Backfill from Yahoo Finance (Sector/Industry columns)")
    logger.info("=" * 60)

    # Find symbols with Sector data but no sec_sector
    query = text(
        """
        SELECT
            ticker,
            CASE
                WHEN isstock THEN 'stock'
                WHEN isetf THEN 'etf'
                ELSE 'unknown'
            END as type,
            "Sector" as sector_source,
            "Industry" as industry_source
        FROM symbol
        WHERE (isstock = true OR isetf = true)
          AND sec_sector IS NULL
          AND "Sector" IS NOT NULL
        ORDER BY ticker
        {}
    """.format(
            f"LIMIT {limit}" if limit else ""
        )
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    logger.info(f"Found {len(rows)} symbols to backfill from Yahoo Finance")

    if dry_run:
        logger.info("[DRY RUN] Would update the following symbols:")
        for i, row in enumerate(rows[:20]):  # Show first 20
            logger.info(f"  {row[0]:8s} - Sector: {row[2]}, Industry: {row[3]}")
        if len(rows) > 20:
            logger.info(f"  ... and {len(rows) - 20} more")
        return len(rows)

    # Perform updates
    update_query = text(
        """
        UPDATE symbol
        SET sec_sector = :sector,
            sec_industry = :industry,
            lastupdts = NOW()
        WHERE ticker = :ticker
    """
    )

    updated_count = 0
    with engine.begin() as conn:
        for row in rows:
            ticker, type_, sector, industry = row
            conn.execute(update_query, {"ticker": ticker, "sector": sector, "industry": industry})
            updated_count += 1

            if updated_count % 100 == 0:
                logger.info(f"  Updated {updated_count}/{len(rows)} symbols...")

    logger.info(f"✅ Updated {updated_count} symbols from Yahoo Finance data")
    return updated_count


def backfill_from_json(engine, sector_mapping, dry_run=False, limit=None):
    """
    Backfill sec_sector from sector_mapping.json for remaining NULL values.

    Args:
        engine: SQLAlchemy engine
        sector_mapping: Dict mapping symbol -> sector
        dry_run: If True, only report what would be updated
        limit: Optional limit on rows to process
    """
    logger.info("=" * 60)
    logger.info("Step 2: Backfill from sector_mapping.json")
    logger.info("=" * 60)

    if not sector_mapping:
        logger.warning("No sector mapping data available, skipping JSON backfill")
        return 0

    # Find symbols still missing sec_sector
    query = text(
        """
        SELECT ticker,
               CASE
                   WHEN isstock THEN 'stock'
                   WHEN isetf THEN 'etf'
                   ELSE 'unknown'
               END as type
        FROM symbol
        WHERE (isstock = true OR isetf = true)
          AND sec_sector IS NULL
        ORDER BY ticker
        {}
    """.format(
            f"LIMIT {limit}" if limit else ""
        )
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    # Filter to symbols in mapping
    symbols_to_update = [(row[0], sector_mapping[row[0]]) for row in rows if row[0] in sector_mapping]

    logger.info(f"Found {len(symbols_to_update)} symbols to backfill from JSON")

    if dry_run:
        logger.info("[DRY RUN] Would update the following symbols:")
        for i, (symbol, sector) in enumerate(symbols_to_update[:20]):
            logger.info(f"  {symbol:8s} - Sector: {sector}")
        if len(symbols_to_update) > 20:
            logger.info(f"  ... and {len(symbols_to_update) - 20} more")
        return len(symbols_to_update)

    # Perform updates
    update_query = text(
        """
        UPDATE symbol
        SET sec_sector = :sector,
            lastupdts = NOW()
        WHERE ticker = :ticker
    """
    )

    updated_count = 0
    with engine.begin() as conn:
        for ticker, sector in symbols_to_update:
            conn.execute(update_query, {"ticker": ticker, "sector": sector})
            updated_count += 1

            if updated_count % 100 == 0:
                logger.info(f"  Updated {updated_count}/{len(symbols_to_update)} symbols...")

    logger.info(f"✅ Updated {updated_count} symbols from JSON mapping")
    return updated_count


def verify_results(engine):
    """Verify backfill results."""
    logger.info("=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)

    query = text(
        """
        SELECT
            CASE
                WHEN isstock THEN 'stock'
                WHEN isetf THEN 'etf'
                ELSE 'unknown'
            END as type,
            COUNT(*) as total_symbols,
            COUNT(sec_sector) as has_sec_sector,
            COUNT(*) - COUNT(sec_sector) as missing_sec_sector,
            ROUND(100.0 * COUNT(sec_sector) / COUNT(*), 2) as coverage_pct
        FROM symbol
        WHERE (isstock = true OR isetf = true)
        GROUP BY CASE
                     WHEN isstock THEN 'stock'
                     WHEN isetf THEN 'etf'
                     ELSE 'unknown'
                 END
        ORDER BY type
    """
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    for row in rows:
        type_, total, has_sector, missing, coverage = row
        logger.info(
            f"{type_.upper():8s}: {has_sector:5d}/{total:5d} ({coverage:5.1f}% coverage) - {missing:5d} still missing"
        )

    # Sample updated symbols
    logger.info("\nSample Updated Symbols:")
    sample_query = text(
        """
        SELECT ticker, sec_sector, sec_industry, "Sector", "Industry"
        FROM symbol
        WHERE sec_sector IS NOT NULL
          AND (isstock = true OR isetf = true)
        ORDER BY RANDOM()
        LIMIT 10
    """
    )

    with engine.connect() as conn:
        result = conn.execute(sample_query)
        rows = result.fetchall()

    for row in rows:
        symbol, sec_sector, sec_industry, yahoo_sector, yahoo_industry = row
        logger.info(f"  {symbol:8s} - sec_sector: {sec_sector:30s} | Yahoo Sector: {yahoo_sector or 'NULL'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backfill sec_sector and sec_industry columns")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without applying them")
    parser.add_argument("--limit", type=int, help="Limit number of rows to process (for testing)")
    args = parser.parse_args()

    logger.info("Starting sec_sector/sec_industry backfill")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    if args.limit:
        logger.info(f"Limit: {args.limit} rows per step")

    try:
        # Connect to stock database
        engine = get_stock_engine()
        logger.info("✅ Connected to stock database")

        # Load sector mapping
        sector_mapping = load_sector_mapping()

        # Step 1: Backfill from Yahoo Finance (Sector/Industry columns)
        yahoo_count = backfill_from_yahoo(engine, dry_run=args.dry_run, limit=args.limit)

        # Step 2: Backfill from sector_mapping.json
        json_count = backfill_from_json(engine, sector_mapping, dry_run=args.dry_run, limit=args.limit)

        # Verify results
        if not args.dry_run:
            verify_results(engine)

        logger.info("=" * 60)
        logger.info("Backfill Complete")
        logger.info("=" * 60)
        logger.info(f"Total updates: {yahoo_count + json_count}")
        logger.info(f"  - From Yahoo Finance: {yahoo_count}")
        logger.info(f"  - From JSON mapping:  {json_count}")

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
