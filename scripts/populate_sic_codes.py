#!/usr/bin/env python3
"""
Populate SIC codes from SEC EDGAR company_tickers.json.

Downloads SEC's official CIK → SIC mapping and populates the sic_code column.

Data source: https://www.sec.gov/files/company_tickers.json

Usage:
    python3 scripts/populate_sic_codes.py [--dry-run]
"""

import logging
import requests
import json
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def get_stock_engine():
    """Get SQLAlchemy engine for stock database."""
    return create_engine("postgresql://stockuser:${STOCK_DB_PASSWORD}@${STOCK_DB_HOST}:5432/stock", pool_pre_ping=True)


def download_sec_cik_sic_mapping():
    """
    Download SEC CIK → SIC mapping from company_tickers.json.

    Returns:
        Dict mapping CIK (int) → {'ticker': str, 'title': str, 'sic_code': int}
    """
    logger.info(f"Downloading SEC company tickers from: {SEC_TICKERS_URL}")

    headers = {"User-Agent": "InvestiGator/1.0 (vijay@example.com)"}  # SEC requires user agent

    try:
        response = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse JSON
        data = response.json()

        # Convert to CIK → info mapping
        cik_mapping = {}
        for key, company in data.items():
            cik = int(company["cik_str"])
            cik_mapping[cik] = {
                "ticker": company.get("ticker", "").upper(),
                "title": company.get("title", ""),
                "sic_code": int(company.get("sic", company.get("sic_description", 0))),
            }

        logger.info(f"✅ Downloaded {len(cik_mapping)} companies with SIC codes")
        return cik_mapping

    except Exception as e:
        logger.error(f"Error downloading SEC data: {e}")
        return {}


def populate_sic_codes(engine, cik_mapping, dry_run=False):
    """
    Populate SIC codes in symbol table from CIK mapping.

    Args:
        engine: Database engine
        cik_mapping: Dict of CIK → company info
        dry_run: If True, only report without updating
    """
    logger.info("=" * 60)
    logger.info("Populating SIC Codes from SEC EDGAR")
    logger.info("=" * 60)

    # Get symbols with CIK but no SIC code
    query = text(
        """
        SELECT ticker, cik
        FROM symbol
        WHERE cik IS NOT NULL
          AND (sic_code IS NULL OR sic_code = 0)
        ORDER BY ticker
    """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    logger.info(f"Found {len(rows)} symbols with CIK but no SIC code")

    # Match with SEC data
    matches = []
    for ticker, cik in rows:
        if cik in cik_mapping:
            sic_code = cik_mapping[cik]["sic_code"]
            if sic_code and sic_code > 0:
                matches.append((ticker, cik, sic_code))

    logger.info(f"Found {len(matches)} matches in SEC data")

    if dry_run:
        logger.info("\n[DRY RUN] Sample updates:")
        for ticker, cik, sic_code in matches[:20]:
            logger.info(f"  {ticker:8s} (CIK: {cik:10d}) → SIC: {sic_code}")
        return len(matches)

    # Update SIC codes
    update_query = text(
        """
        UPDATE symbol
        SET sic_code = :sic_code,
            lastupdts = NOW()
        WHERE ticker = :ticker
    """
    )

    updated = 0
    with engine.begin() as conn:
        for ticker, cik, sic_code in matches:
            conn.execute(update_query, {"ticker": ticker, "sic_code": sic_code})
            updated += 1

            if updated % 500 == 0:
                logger.info(f"  Updated {updated}/{len(matches)} symbols...")

    logger.info(f"✅ Updated {updated} symbols with SIC codes")
    return updated


def verify_results(engine):
    """Verify SIC code population."""
    logger.info("\n" + "=" * 60)
    logger.info("Verification: SIC Code Coverage")
    logger.info("=" * 60)

    queries = [
        ("Total Symbols", "SELECT COUNT(*) FROM symbol"),
        ("Has CIK", "SELECT COUNT(*) FROM symbol WHERE cik IS NOT NULL"),
        ("Has SIC Code", "SELECT COUNT(*) FROM symbol WHERE sic_code IS NOT NULL AND sic_code > 0"),
        (
            "SIC Coverage %",
            """
            SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE sic_code IS NOT NULL AND sic_code > 0) /
                         COUNT(*) FILTER (WHERE cik IS NOT NULL), 1) as coverage_pct
            FROM symbol
        """,
        ),
    ]

    with engine.connect() as conn:
        for title, query in queries:
            result = conn.execute(text(query)).scalar()
            logger.info(f"  {title:20s}: {result}")

    # Show top SIC codes
    logger.info("\nTop 10 SIC Codes:")
    top_sic_query = text(
        """
        SELECT sic_code, COUNT(*) as count
        FROM symbol
        WHERE sic_code IS NOT NULL AND sic_code > 0
        GROUP BY sic_code
        ORDER BY count DESC
        LIMIT 10;
    """
    )

    with engine.connect() as conn:
        rows = conn.execute(top_sic_query).fetchall()
        for sic_code, count in rows:
            logger.info(f"    SIC {sic_code:4d}: {count:5d} companies")


def main():
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE")
        logger.info("=" * 60)

    # Download SEC data
    cik_mapping = download_sec_cik_sic_mapping()

    if not cik_mapping:
        logger.error("Failed to download SEC data. Exiting.")
        return

    # Connect and populate
    engine = get_stock_engine()
    logger.info("✅ Connected to stock database\n")

    updated = populate_sic_codes(engine, cik_mapping, dry_run=dry_run)

    if not dry_run and updated > 0:
        verify_results(engine)


if __name__ == "__main__":
    main()
