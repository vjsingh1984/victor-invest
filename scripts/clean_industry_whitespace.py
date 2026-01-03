#!/usr/bin/env python3
"""
Clean whitespace from sec_industry values.

This script trims leading/trailing whitespace from industry names.

Usage:
    python3 scripts/clean_industry_whitespace.py [--dry-run]
"""

import logging
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_stock_engine():
    """Get SQLAlchemy engine for stock database."""
    db_host = "${DB_HOST:-localhost}"
    db_port = 5432
    db_name = "stock"
    db_user = "stockuser"
    db_password = os.environ.get("STOCK_DB_PASSWORD", "")

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string, pool_pre_ping=True)


def clean_whitespace(engine, dry_run=False):
    """
    Clean whitespace from industry values.

    Args:
        engine: SQLAlchemy engine
        dry_run: If True, only report what would be updated
    """
    logger.info("=" * 60)
    logger.info("Cleaning whitespace from industry values")
    logger.info("=" * 60)

    # Find industries with whitespace issues
    query = text(
        """
        SELECT
            sec_industry,
            TRIM(sec_industry) as trimmed,
            COUNT(*) as count
        FROM symbol
        WHERE sec_industry IS NOT NULL
          AND sec_industry != TRIM(sec_industry)
        GROUP BY sec_industry, TRIM(sec_industry)
        ORDER BY count DESC
    """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        logger.info("✅ No whitespace issues found")
        return 0

    logger.info(f"Found {len(rows)} industries with whitespace issues:")
    total_symbols = sum(row[2] for row in rows)
    logger.info(f"Total symbols affected: {total_symbols}")

    for old_industry, trimmed, count in rows:
        logger.info(f"  '{old_industry}' → '{trimmed}' ({count} symbols)")

    if dry_run:
        logger.info("\n[DRY RUN] No changes made")
        return total_symbols

    # Update all industries to trimmed versions
    logger.info("\nUpdating industries...")
    update_query = text(
        """
        UPDATE symbol
        SET sec_industry = TRIM(sec_industry),
            lastupdts = NOW()
        WHERE sec_industry IS NOT NULL
          AND sec_industry != TRIM(sec_industry)
    """
    )

    with engine.begin() as conn:
        result = conn.execute(update_query)
        updated = result.rowcount

    logger.info(f"✅ Updated {updated} symbols")
    return updated


def verify_results(engine):
    """Verify cleanup results."""
    logger.info("\n" + "=" * 60)
    logger.info("Verification: Check for remaining whitespace issues")
    logger.info("=" * 60)

    query = text(
        """
        SELECT COUNT(*) as count
        FROM symbol
        WHERE sec_industry IS NOT NULL
          AND sec_industry != TRIM(sec_industry)
    """
    )

    with engine.connect() as conn:
        count = conn.execute(query).scalar()

    if count == 0:
        logger.info("✅ No whitespace issues remaining")
    else:
        logger.warning(f"❌ Still {count} symbols with whitespace issues")

    # Show sample industries
    logger.info("\nSample industries after cleanup:")
    sample_query = text(
        """
        SELECT DISTINCT sec_industry
        FROM symbol
        WHERE sec_industry IS NOT NULL
        ORDER BY sec_industry
        LIMIT 10
    """
    )

    with engine.connect() as conn:
        industries = conn.execute(sample_query).fetchall()

    for (industry,) in industries:
        logger.info(f"  '{industry}'")


def main():
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 60)

    engine = get_stock_engine()
    logger.info("✅ Connected to stock database")

    # Clean whitespace
    updated = clean_whitespace(engine, dry_run=dry_run)

    # Verify results
    if not dry_run and updated > 0:
        verify_results(engine)


if __name__ == "__main__":
    main()
