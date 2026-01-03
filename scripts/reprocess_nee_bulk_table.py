#!/usr/bin/env python3
"""
Re-process NEE (NextEra Energy) SEC data with utility-specific XBRL tag mappings

This script:
1. Fetches NEE's raw SEC companyfacts JSON from API
2. Processes it using the new FALLBACK_CHAINS for utility-specific tags
3. Updates the sec_companyfacts_processed bulk table with correct values
4. Verifies NEE now has non-NULL revenue and capital_expenditures

Expected Results:
- total_revenue: ~$11B-$22B per quarter (electric utility)
- capital_expenditures: ~$7B-$15B per quarter (heavy infrastructure)
- operating_cash_flow: ~$1.7B-$11.3B (already present)
- dividends_paid: ~$0.9B-$3.4B (already present)
"""

import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investigator.infrastructure.sec.data_processor import SECDataProcessor
from investigator.infrastructure.database.db import get_db_connection
from investigator.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def verify_nee_bulk_table_before():
    """Verify NEE data quality BEFORE re-processing"""
    logger.info("=" * 80)
    logger.info("STEP 1: Verify NEE bulk table data BEFORE re-processing")
    logger.info("=" * 80)

    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT
        fiscal_year,
        fiscal_quarter,
        total_revenue,
        operating_cash_flow,
        capital_expenditures,
        dividends_paid,
        free_cash_flow
    FROM sec_companyfacts_processed
    WHERE symbol = 'NEE'
    AND fiscal_year >= 2024
    ORDER BY fiscal_year DESC, fiscal_quarter DESC
    LIMIT 5;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    logger.info(f"\nFound {len(rows)} NEE quarters (2024+):")
    for row in rows:
        fy, fq, rev, ocf, capex, div, fcf = row
        logger.info(f"  {fy}-Q{fq}: Revenue={rev}, CapEx={capex}, OCF={ocf}, Div={div}, FCF={fcf}")

    # Check for NULL values
    null_count = sum(1 for row in rows if row[2] is None or row[4] is None)
    if null_count > 0:
        logger.warning(f"⚠️  Found {null_count} quarters with NULL revenue or capex")
    else:
        logger.info("✅ All quarters have non-NULL revenue and capex")

    cursor.close()
    conn.close()

    return rows


def reprocess_nee_data():
    """Re-process NEE SEC data with new utility-specific tag mappings"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Re-process NEE data with utility-specific XBRL tags")
    logger.info("=" * 80)

    processor = SECDataProcessor()
    symbol = "NEE"

    logger.info(f"\nFetching fresh SEC data for {symbol}...")

    # Fetch and process NEE data
    # This will:
    # 1. Fetch raw JSON from SEC API
    # 2. Apply FIELD_MAPPINGS (Phase 1)
    # 3. Apply FALLBACK_CHAINS for missing metrics (Phase 2)
    # 4. Save to sec_companyfacts_processed table

    try:
        result = processor.process_symbol(symbol)
        logger.info(f"✅ Successfully re-processed {symbol}")
        logger.info(f"   Quarters extracted: {result.get('quarters_count', 0)}")
        logger.info(f"   Fields extracted: {result.get('fields_count', 0)}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to re-process {symbol}: {e}")
        return False


def verify_nee_bulk_table_after():
    """Verify NEE data quality AFTER re-processing"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Verify NEE bulk table data AFTER re-processing")
    logger.info("=" * 80)

    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT
        fiscal_year,
        fiscal_quarter,
        total_revenue,
        operating_cash_flow,
        capital_expenditures,
        dividends_paid,
        free_cash_flow
    FROM sec_companyfacts_processed
    WHERE symbol = 'NEE'
    AND fiscal_year >= 2024
    ORDER BY fiscal_year DESC, fiscal_quarter DESC
    LIMIT 5;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    logger.info(f"\nFound {len(rows)} NEE quarters (2024+):")
    for row in rows:
        fy, fq, rev, ocf, capex, div, fcf = row
        logger.info(
            f"  {fy}-Q{fq}: Revenue={rev:,.0f}, CapEx={capex:,.0f}, " f"OCF={ocf:,.0f}, Div={div:,.0f}, FCF={fcf:,.0f}"
        )

    # Check for NULL values
    null_count = sum(1 for row in rows if row[2] is None or row[4] is None)
    if null_count > 0:
        logger.error(f"❌ Still have {null_count} quarters with NULL revenue or capex")
        return False
    else:
        logger.info("✅ All quarters now have non-NULL revenue and capex")
        return True

    cursor.close()
    conn.close()


def main():
    """Main re-processing workflow"""
    logger.info("🚀 Starting NEE Bulk Table Re-processing")
    logger.info(f"Using config: {get_config()}")

    # Step 1: Verify BEFORE state
    before_rows = verify_nee_bulk_table_before()

    # Step 2: Re-process NEE data
    success = reprocess_nee_data()
    if not success:
        logger.error("❌ Re-processing failed, aborting")
        sys.exit(1)

    # Step 3: Verify AFTER state
    success = verify_nee_bulk_table_after()

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ SUCCESS: NEE bulk table re-processing completed")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Run NEE analysis: python3 cli_orchestrator.py analyze NEE -m standard")
        logger.info("2. Verify DCF/GGM valuations are non-zero")
        logger.info("3. Check results/NEE_*.json for fair value estimates")
    else:
        logger.error("\n" + "=" * 80)
        logger.error("❌ FAILURE: NEE still has NULL values after re-processing")
        logger.error("=" * 80)
        logger.error("Troubleshooting:")
        logger.error("1. Check src/investigator/infrastructure/sec/data_processor.py for canonical key mappings")
        logger.error("2. Verify NEE uses expected XBRL tags in SEC API response")
        logger.error("3. Check logs for processing messages")
        sys.exit(1)


if __name__ == "__main__":
    main()
