#!/usr/bin/env python3
"""
Populate exchange data for symbols using yfinance.

This script fetches exchange information for all symbols that are missing it.
Uses yfinance API which provides reliable exchange data.

Usage:
    python3 scripts/populate_exchange_data.py [--dry-run] [--limit N]
"""

import logging
import time
from typing import Optional
import yfinance as yf
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_stock_engine():
    """Get SQLAlchemy engine for stock database."""
    return create_engine("postgresql://stockuser:${STOCK_DB_PASSWORD}@${STOCK_DB_HOST}:5432/stock", pool_pre_ping=True)


def fetch_exchange_for_ticker(ticker: str) -> Optional[str]:
    """
    Fetch exchange for a ticker using yfinance.

    Returns:
        Exchange code (e.g., 'NMS' for NASDAQ, 'NYQ' for NYSE, etc.)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # yfinance returns exchange in the 'exchange' field
        exchange = info.get("exchange", None)

        if exchange:
            # Normalize exchange codes
            exchange_map = {
                "NMS": "NASDAQ",
                "NGM": "NASDAQ",
                "NCM": "NASDAQ",
                "NYQ": "NYSE",
                "NYE": "NYSE",
                "PCX": "NYSE_ARCA",
                "ASE": "NYSE_AMERICAN",
                "BTS": "BATS",
                "PNK": "OTC",
                "OTC": "OTC",
                "TSE": "TORONTO",
                "LSE": "LONDON",
            }
            return exchange_map.get(exchange, exchange)

        return None

    except Exception as e:
        logger.debug(f"Error fetching exchange for {ticker}: {e}")
        return None


def populate_exchanges(engine, dry_run=False, limit=None):
    """
    Populate exchange data for symbols missing it.

    Args:
        engine: Database engine
        dry_run: If True, only report without updating
        limit: Optional limit on symbols to process
    """
    logger.info("=" * 60)
    logger.info("Populating Exchange Data")
    logger.info("=" * 60)

    # Get symbols missing exchange data
    query = text(
        """
        SELECT ticker, isstock, isetf
        FROM symbol
        WHERE (exchange IS NULL OR exchange = '')
          AND (isstock = true OR isetf = true)
        ORDER BY
            CASE
                WHEN ticker IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B') THEN 0
                ELSE 1
            END,
            ticker
        {}
    """.format(
            f"LIMIT {limit}" if limit else ""
        )
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    logger.info(f"Found {len(rows)} symbols without exchange data")

    if dry_run:
        logger.info("[DRY RUN] Fetching sample exchanges...")
        for ticker, isstock, isetf in rows[:10]:
            exchange = fetch_exchange_for_ticker(ticker)
            type_str = "Stock" if isstock else "ETF" if isetf else "Unknown"
            logger.info(f"  {ticker:8s} ({type_str:5s}) → {exchange or 'NOT FOUND'}")
        return 0

    # Update exchanges
    update_query = text(
        """
        UPDATE symbol
        SET exchange = :exchange,
            lastupdts = NOW()
        WHERE ticker = :ticker
    """
    )

    updated = 0
    failed = 0

    for i, (ticker, isstock, isetf) in enumerate(rows):
        if i > 0 and i % 100 == 0:
            logger.info(f"Progress: {i}/{len(rows)} ({updated} updated, {failed} failed)")

        exchange = fetch_exchange_for_ticker(ticker)

        if exchange:
            with engine.begin() as conn:
                conn.execute(update_query, {"ticker": ticker, "exchange": exchange})
            updated += 1
        else:
            failed += 1

        # Rate limiting - be nice to yfinance
        if i % 10 == 0:
            time.sleep(0.5)

    logger.info(f"✅ Updated {updated} symbols")
    logger.info(f"❌ Failed to fetch {failed} symbols")
    return updated


def verify_results(engine):
    """Verify exchange population."""
    logger.info("\n" + "=" * 60)
    logger.info("Verification: Exchange Coverage")
    logger.info("=" * 60)

    query = text(
        """
        SELECT
            exchange,
            COUNT(*) as count,
            COUNT(*) FILTER (WHERE isstock = true) as stocks,
            COUNT(*) FILTER (WHERE isetf = true) as etfs
        FROM symbol
        WHERE exchange IS NOT NULL AND exchange != ''
        GROUP BY exchange
        ORDER BY count DESC;
    """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    total = sum(row[1] for row in rows)
    logger.info(f"Total symbols with exchange: {total}")
    logger.info("")

    for exchange, count, stocks, etfs in rows[:15]:
        logger.info(f"  {exchange:15s} - {count:6d} total ({stocks:6d} stocks, {etfs:6d} ETFs)")


def main():
    import sys

    dry_run = "--dry-run" in sys.argv
    limit = None

    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE")
        logger.info("=" * 60)

    engine = get_stock_engine()
    logger.info("✅ Connected to stock database")

    updated = populate_exchanges(engine, dry_run=dry_run, limit=limit)

    if not dry_run and updated > 0:
        verify_results(engine)


if __name__ == "__main__":
    main()
