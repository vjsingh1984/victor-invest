#!/usr/bin/env python3
"""
Normalize sec_sector values to match sector_mapping.json canonical names.

This script:
1. Reads canonical sector names from data/sector_mapping.json
2. Maps various Yahoo Finance sector names to canonical names
3. Updates sec_sector values in symbol table to use canonical names
4. Falls back to Sector column if no good mapping exists

Usage:
    python3 scripts/normalize_sectors_to_mapping.py [--dry-run]
"""

import json
import logging
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sector normalization mapping
# Maps Yahoo Finance sector names → Canonical sector names (from mapping JSON)
SECTOR_NORMALIZATION = {
    # Finance variations → Financials
    "Finance": "Financials",
    "Financial Services": "Financials",
    # Health Care variations → Healthcare
    "Health Care": "Healthcare",
    # Technology variations → Technology
    "Information Technology": "Technology",
    # Materials variations → Materials
    "Basic Materials": "Materials",
    # Communication variations → Communication Services
    "Telecommunications": "Communication Services",
    # Consumer variations → Consumer Discretionary / Consumer Staples
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    # Already canonical (no change needed, but listed for clarity)
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Technology": "Technology",
    "Industrials": "Industrials",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
}


def get_stock_engine():
    """Get SQLAlchemy engine for stock database."""
    db_host = "${DB_HOST:-localhost}"
    db_port = 5432
    db_name = "stock"
    db_user = "stockuser"
    db_password = os.environ.get("STOCK_DB_PASSWORD", "")

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string, pool_pre_ping=True)


def normalize_sectors(engine, dry_run=False):
    """
    Normalize sec_sector values to canonical names.

    Args:
        engine: SQLAlchemy engine
        dry_run: If True, only report what would be updated
    """
    logger.info("=" * 60)
    logger.info("Normalizing sec_sector values to canonical names")
    logger.info("=" * 60)

    total_updated = 0

    for old_sector, new_sector in SECTOR_NORMALIZATION.items():
        # Skip if already canonical
        if old_sector == new_sector:
            continue

        # Count symbols with this sector
        count_query = text(
            """
            SELECT COUNT(*)
            FROM symbol
            WHERE sec_sector = :old_sector
        """
        )

        with engine.connect() as conn:
            count = conn.execute(count_query, {"old_sector": old_sector}).scalar()

        if count == 0:
            continue

        logger.info(f"Normalizing: {old_sector:30s} → {new_sector:30s} ({count:5d} symbols)")

        if dry_run:
            continue

        # Update to canonical name
        update_query = text(
            """
            UPDATE symbol
            SET sec_sector = :new_sector,
                lastupdts = NOW()
            WHERE sec_sector = :old_sector
        """
        )

        with engine.begin() as conn:
            result = conn.execute(update_query, {"old_sector": old_sector, "new_sector": new_sector})
            updated = result.rowcount
            total_updated += updated
            logger.info(f"  ✅ Updated {updated} symbols")

    # Handle "Miscellaneous" - use Sector column value
    logger.info("=" * 60)
    logger.info("Handling 'Miscellaneous' sector (using Sector column)")
    logger.info("=" * 60)

    misc_query = text(
        """
        SELECT ticker, "Sector"
        FROM symbol
        WHERE sec_sector = 'Miscellaneous'
          AND "Sector" IS NOT NULL
        LIMIT 20
    """
    )

    with engine.connect() as conn:
        misc_rows = conn.execute(misc_query).fetchall()

    if misc_rows:
        logger.info(f"Found {len(misc_rows)} symbols with 'Miscellaneous' sector")

        if not dry_run:
            for ticker, yahoo_sector in misc_rows:
                # Try to map Yahoo sector to canonical
                canonical = SECTOR_NORMALIZATION.get(yahoo_sector, yahoo_sector)

                update_query = text(
                    """
                    UPDATE symbol
                    SET sec_sector = :sector,
                        lastupdts = NOW()
                    WHERE ticker = :ticker
                """
                )

                with engine.begin() as conn:
                    conn.execute(update_query, {"ticker": ticker, "sector": canonical})

                logger.info(f"  {ticker}: Miscellaneous → {canonical}")
                total_updated += 1

    return total_updated


def verify_results(engine):
    """Verify normalization results."""
    logger.info("=" * 60)
    logger.info("Verification: Sector Distribution After Normalization")
    logger.info("=" * 60)

    query = text(
        """
        SELECT sec_sector, COUNT(*) as count
        FROM symbol
        WHERE sec_sector IS NOT NULL
        GROUP BY sec_sector
        ORDER BY count DESC
    """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    canonical_sectors = set(SECTOR_NORMALIZATION.values())

    for sector, count in rows:
        is_canonical = "✅" if sector in canonical_sectors else "❌"
        logger.info(f"  {is_canonical} {sector:30s} - {count:5d} symbols")

    # Check tier companies
    logger.info("\n" + "=" * 60)
    logger.info("Verification: Tier Companies")
    logger.info("=" * 60)

    tier_query = text(
        """
        SELECT ticker, sec_sector, "Sector"
        FROM symbol
        WHERE ticker IN ('AAPL', 'MSFT', 'NVDA', 'JNJ', 'XOM', 'NEE')
        ORDER BY ticker
    """
    )

    with engine.connect() as conn:
        tier_rows = conn.execute(tier_query).fetchall()

    for ticker, sec_sector, yahoo_sector in tier_rows:
        is_canonical = "✅" if sec_sector in canonical_sectors else "❌"
        logger.info(f"  {is_canonical} {ticker:6s} - {sec_sector:30s} (Yahoo: {yahoo_sector})")


def main():
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 60)

    engine = get_stock_engine()
    logger.info("✅ Connected to stock database")

    # Normalize sectors
    total_updated = normalize_sectors(engine, dry_run=dry_run)

    # Verify results
    if not dry_run:
        logger.info("\n" + "=" * 60)
        logger.info(f"Normalization Complete - Updated {total_updated} symbols")
        logger.info("=" * 60)
        verify_results(engine)


if __name__ == "__main__":
    main()
