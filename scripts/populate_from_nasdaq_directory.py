#!/usr/bin/env python3
"""
Populate symbol metadata from NASDAQ symbol directory.

Downloads official NASDAQ, NYSE, and AMEX listings and populates:
- Exchange
- Description (Security Name)
- Market Category
- ETF flag

Data source: ftp://ftp.nasdaqtrader.com/symboldirectory/

Usage:
    python3 scripts/populate_from_nasdaq_directory.py [--dry-run]
"""

import logging
import pandas as pd
from io import StringIO
from sqlalchemy import create_engine, text
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NASDAQ FTP URLs for symbol directories
NASDAQ_TRADED_URL = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
NASDAQ_LISTED_URL = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
OTHER_LISTED_URL = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"


def get_stock_engine():
    """Get SQLAlchemy engine for stock database."""
    return create_engine("postgresql://stockuser:${STOCK_DB_PASSWORD}@${STOCK_DB_HOST}:5432/stock", pool_pre_ping=True)


def download_nasdaq_data():
    """
    Download and parse NASDAQ symbol directory files.

    Returns:
        DataFrame with columns: ticker, security_name, exchange, etf, test_issue
    """
    logger.info("Downloading NASDAQ symbol directory files...")

    all_symbols = []

    # Download nasdaqtraded.txt (all NASDAQ-traded securities)
    try:
        logger.info(f"Downloading: {NASDAQ_TRADED_URL}")
        response = requests.get(NASDAQ_TRADED_URL.replace("ftp://", "https://"))

        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), sep="|")

            # Filter out test issues and file trailers
            df = df[df["Test Issue"] == "N"]
            df = df[df["Symbol"].notna()]
            df = df[~df["Symbol"].str.contains("File", na=False)]

            # Extract relevant columns
            for _, row in df.iterrows():
                symbol = row.get("Symbol", row.get("NASDAQ Symbol", "")).strip()
                if not symbol:
                    continue

                # Determine exchange
                listing_ex = str(row.get("Listing Exchange", ""))
                if "Q" in listing_ex or "NASDAQ" in listing_ex:
                    exchange = "NASDAQ"
                elif "N" in listing_ex or "NYSE" in listing_ex:
                    exchange = "NYSE"
                elif "A" in listing_ex:
                    exchange = "NYSE_AMERICAN"
                elif "P" in listing_ex:
                    exchange = "NYSE_ARCA"
                elif "Z" in listing_ex:
                    exchange = "BATS"
                else:
                    exchange = "NASDAQ"  # Default for NASDAQ-traded

                all_symbols.append(
                    {
                        "ticker": symbol,
                        "security_name": str(row.get("Security Name", "")).strip(),
                        "exchange": exchange,
                        "etf": str(row.get("ETF", "N")) == "Y",
                        "test_issue": str(row.get("Test Issue", "N")) == "Y",
                    }
                )

            logger.info(f"  Loaded {len(all_symbols)} symbols from nasdaqtraded.txt")

    except Exception as e:
        logger.warning(f"Error downloading nasdaqtraded.txt: {e}")

    # Try alternative: direct NASDAQ listed file
    try:
        logger.info(f"Downloading: {NASDAQ_LISTED_URL}")
        response = requests.get(NASDAQ_LISTED_URL.replace("ftp://", "https://"))

        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), sep="|")
            df = df[df["Test Issue"] == "N"]
            df = df[df["Symbol"].notna()]
            df = df[~df["Symbol"].str.contains("File", na=False)]

            for _, row in df.iterrows():
                symbol = str(row.get("Symbol", "")).strip()
                if not symbol or any(s["ticker"] == symbol for s in all_symbols):
                    continue

                all_symbols.append(
                    {
                        "ticker": symbol,
                        "security_name": str(row.get("Security Name", "")).strip(),
                        "exchange": "NASDAQ",
                        "etf": str(row.get("ETF", "N")) == "Y",
                        "test_issue": str(row.get("Test Issue", "N")) == "Y",
                    }
                )

            logger.info(f"  Added {sum(1 for s in all_symbols if s['exchange'] == 'NASDAQ')} NASDAQ symbols")

    except Exception as e:
        logger.warning(f"Error downloading nasdaqlisted.txt: {e}")

    # Download otherlisted.txt (NYSE, AMEX, etc.)
    try:
        logger.info(f"Downloading: {OTHER_LISTED_URL}")
        response = requests.get(OTHER_LISTED_URL.replace("ftp://", "https://"))

        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), sep="|")
            df = df[df["Test Issue"] == "N"]
            df = df[~df["ACT Symbol"].str.contains("File", na=False)]

            for _, row in df.iterrows():
                symbol = str(row.get("ACT Symbol", "")).strip()
                if not symbol or any(s["ticker"] == symbol for s in all_symbols):
                    continue

                # Determine exchange from Exchange column
                exchange_code = str(row.get("Exchange", "")).strip()
                exchange_map = {
                    "N": "NYSE",
                    "A": "NYSE_AMERICAN",
                    "P": "NYSE_ARCA",
                    "Z": "BATS",
                    "V": "IEX",
                }
                exchange = exchange_map.get(exchange_code, "OTHER")

                all_symbols.append(
                    {
                        "ticker": symbol,
                        "security_name": str(row.get("Security Name", "")).strip(),
                        "exchange": exchange,
                        "etf": str(row.get("ETF", "N")) == "Y",
                        "test_issue": str(row.get("Test Issue", "N")) == "Y",
                    }
                )

            logger.info(f"  Added {sum(1 for s in all_symbols if s['exchange'] != 'NASDAQ')} other exchange symbols")

    except Exception as e:
        logger.warning(f"Error downloading otherlisted.txt: {e}")

    if not all_symbols:
        logger.error("Failed to download any symbol data!")
        return pd.DataFrame()

    df = pd.DataFrame(all_symbols)

    # Filter out test issues
    df = df[df["test_issue"] == False]
    df = df.drop(columns=["test_issue"])

    logger.info(f"✅ Total symbols loaded: {len(df)}")
    logger.info(f"   Exchanges: {df['exchange'].value_counts().to_dict()}")
    logger.info(f"   ETFs: {df['etf'].sum()}")

    return df


def populate_from_nasdaq(engine, nasdaq_df, dry_run=False):
    """
    Populate symbol table from NASDAQ data.

    Updates:
    - exchange
    - description (if missing)
    - isetf (if ETF flag is True)
    """
    logger.info("=" * 60)
    logger.info("Populating from NASDAQ Directory")
    logger.info("=" * 60)

    if nasdaq_df.empty:
        logger.error("No NASDAQ data to process")
        return 0

    # Get existing symbols from database
    query = text("SELECT ticker, exchange, description, isetf FROM symbol")

    with engine.connect() as conn:
        db_symbols = pd.read_sql(query, conn)

    logger.info(f"Database has {len(db_symbols)} symbols")
    logger.info(f"NASDAQ data has {len(nasdaq_df)} symbols")

    # Merge to find matches
    merged = db_symbols.merge(nasdaq_df, on="ticker", how="inner", suffixes=("_db", "_nasdaq"))

    logger.info(f"Found {len(merged)} matching symbols")

    if dry_run:
        logger.info("\n[DRY RUN] Sample updates:")
        for i, row in merged.head(20).iterrows():
            updates = []
            if pd.isna(row["exchange_db"]) or row["exchange_db"] == "":
                updates.append(f"exchange: NULL → {row['exchange_nasdaq']}")
            if pd.isna(row["description_db"]) or row["description_db"] == "":
                updates.append(f"description: NULL → {row['security_name'][:30]}...")
            if row["etf"] and not row["isetf"]:
                updates.append(f"isetf: False → True")

            if updates:
                logger.info(f"  {row['ticker']:8s}: {', '.join(updates)}")

        return len(merged)

    # Update records
    update_query = text(
        """
        UPDATE symbol
        SET exchange = :exchange,
            description = COALESCE(NULLIF(description, ''), :description),
            isetf = CASE WHEN :is_etf THEN true ELSE isetf END,
            lastupdts = NOW()
        WHERE ticker = :ticker
    """
    )

    updated = 0
    with engine.begin() as conn:
        for _, row in merged.iterrows():
            conn.execute(
                update_query,
                {
                    "ticker": row["ticker"],
                    "exchange": row["exchange_nasdaq"],
                    "description": row["security_name"],
                    "is_etf": row["etf"],
                },
            )
            updated += 1

            if updated % 1000 == 0:
                logger.info(f"  Updated {updated}/{len(merged)} symbols...")

    logger.info(f"✅ Updated {updated} symbols")
    return updated


def verify_results(engine):
    """Verify population results."""
    logger.info("\n" + "=" * 60)
    logger.info("Verification Results")
    logger.info("=" * 60)

    queries = {
        "Exchange Coverage": """
            SELECT
                exchange,
                COUNT(*) as count
            FROM symbol
            WHERE exchange IS NOT NULL AND exchange != ''
            GROUP BY exchange
            ORDER BY count DESC
            LIMIT 10;
        """,
        "Description Coverage": """
            SELECT
                COUNT(*) FILTER (WHERE description IS NOT NULL AND description != '') as has_desc,
                COUNT(*) as total,
                ROUND(100.0 * COUNT(*) FILTER (WHERE description IS NOT NULL AND description != '') / COUNT(*), 1) as pct
            FROM symbol;
        """,
        "ETF Count": """
            SELECT
                COUNT(*) FILTER (WHERE isetf = true) as etfs,
                COUNT(*) as total
            FROM symbol;
        """,
    }

    with engine.connect() as conn:
        for title, query in queries.items():
            logger.info(f"\n{title}:")
            result = conn.execute(text(query))
            for row in result:
                logger.info(f"  {row}")


def main():
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE")
        logger.info("=" * 60)

    # Download NASDAQ data
    nasdaq_df = download_nasdaq_data()

    if nasdaq_df.empty:
        logger.error("Failed to download NASDAQ data. Exiting.")
        return

    # Connect and populate
    engine = get_stock_engine()
    logger.info("✅ Connected to stock database\n")

    updated = populate_from_nasdaq(engine, nasdaq_df, dry_run=dry_run)

    if not dry_run and updated > 0:
        verify_results(engine)


if __name__ == "__main__":
    main()
