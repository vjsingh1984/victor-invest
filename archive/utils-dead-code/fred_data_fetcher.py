#!/usr/bin/env python3
"""
FRED Data Fetcher - Fetch and populate macro indicators from Federal Reserve Economic Data

This script fetches data from the FRED API and populates the stock database
with new macro indicators including debt metrics.
"""

import logging
import os
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from investigator.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FRED API Configuration
FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_API_KEY = None  # Set from environment or config

# Priority indicators to add to the database
PRIORITY_INDICATORS = {
    # Federal Debt Metrics
    "GFDEGDQ188S": {
        "name": "Federal Debt: Total Public Debt as Percent of GDP",
        "frequency": "Quarterly",
        "units": "Percent of GDP",
        "category": "debt",
    },
    "GFDGDPA188S": {
        "name": "Federal Debt: Gross Federal Debt as Percent of GDP",
        "frequency": "Annual",
        "units": "Percent of GDP",
        "category": "debt",
    },
    # Household Debt Metrics
    "HDTGPDUSQ163N": {
        "name": "Household Debt to GDP for United States",
        "frequency": "Quarterly",
        "units": "Percent",
        "category": "debt",
    },
    "CMDEBT": {
        "name": "Households and Nonprofit Organizations; Credit Market Instruments; Liability",
        "frequency": "Quarterly",
        "units": "Billions of Dollars",
        "category": "debt",
    },
    # Corporate Debt Metrics
    "NCBDBIQ027S": {
        "name": "Nonfinancial Corporate Business; Debt Securities; Liability",
        "frequency": "Quarterly",
        "units": "Billions of Dollars",
        "category": "debt",
    },
    "TBSDODNS": {
        "name": "Total Business: Debt Outstanding",
        "frequency": "Quarterly",
        "units": "Billions of Dollars",
        "category": "debt",
    },
    # Debt Service Metrics
    "TDSP": {
        "name": "Household Debt Service Payments as a Percent of Disposable Personal Income",
        "frequency": "Quarterly",
        "units": "Percent",
        "category": "debt",
    },
    "FODSP": {"name": "Financial Obligations Ratio", "frequency": "Quarterly", "units": "Percent", "category": "debt"},
    # Additional Economic Indicators
    "M2SL": {
        "name": "M2 Money Stock",
        "frequency": "Monthly",
        "units": "Billions of Dollars",
        "category": "money_supply",
    },
    "WALCL": {
        "name": "Federal Reserve Total Assets",
        "frequency": "Weekly",
        "units": "Billions of Dollars",
        "category": "monetary_policy",
    },
    "BAMLH0A0HYM2": {
        "name": "ICE BofA US High Yield Index Option-Adjusted Spread",
        "frequency": "Daily",
        "units": "Percent",
        "category": "credit",
    },
    "T10YIE": {
        "name": "10-Year Breakeven Inflation Rate",
        "frequency": "Daily",
        "units": "Percent",
        "category": "inflation",
    },
    "DEXUSEU": {
        "name": "U.S. / Euro Foreign Exchange Rate",
        "frequency": "Daily",
        "units": "U.S. Dollars to One Euro",
        "category": "forex",
    },
    "MORTGAGE30US": {
        "name": "30-Year Fixed Rate Mortgage Average",
        "frequency": "Weekly",
        "units": "Percent",
        "category": "housing",
    },
    "PSAVERT": {"name": "Personal Saving Rate", "frequency": "Monthly", "units": "Percent", "category": "consumer"},
}


class FREDDataFetcher:
    """Fetches data from FRED API and populates database"""

    def __init__(self, api_key: Optional[str] = None, db_user: Optional[str] = None, db_password: Optional[str] = None):
        """Initialize FRED data fetcher

        Args:
            api_key: FRED API key (or use FRED_API_KEY env var)
            db_user: Database user (default: stockuser for write, or from config)
            db_password: Database password (or use PGPASSWORD env var)
        """
        self.api_key = api_key or self._get_api_key()
        self.session = requests.Session()
        self.SessionLocal = self._get_stock_db_manager(db_user, db_password)

    def _get_api_key(self) -> Optional[str]:
        """Get FRED API key from config or environment"""
        try:
            config = get_config()
            # Try to get from config first
            if hasattr(config, "fred_api_key"):
                return config.fred_api_key
        except:
            pass

        # Try environment variable
        return os.environ.get("FRED_API_KEY")

    def _get_stock_db_manager(self, db_user: Optional[str] = None, db_password: Optional[str] = None):
        """Get database manager configured for stock database with write permissions

        Args:
            db_user: Database user (default: stockuser, or investigator for read-only)
            db_password: Database password (or use PGPASSWORD env var, or from config)

        Returns:
            SessionLocal: SQLAlchemy session maker for stock database
        """
        config = get_config()
        db_config = config.database

        # Determine database user (priority: arg > env > default 'stockuser')
        if db_user is None:
            db_user = os.environ.get("DB_USER", "stockuser")

        # Determine database password (priority: arg > env > config)
        if db_password is None:
            db_password = os.environ.get("PGPASSWORD")
            if db_password is None:
                # Fall back to config password if available
                db_password = db_config.password

        # Build connection URL
        stock_db_url = f"postgresql://{db_user}:{db_password}@{db_config.host}:{db_config.port}/stock"

        logger.info(f"Connecting to stock database as user: {db_user}")

        engine = create_engine(
            stock_db_url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            echo=False,
            pool_pre_ping=True,
        )

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal

    def fetch_series_info(self, series_id: str) -> Optional[Dict]:
        """Fetch series metadata from FRED API"""
        if not self.api_key:
            logger.warning("FRED API key not configured, skipping API call")
            return None

        url = f"{FRED_API_BASE}/series"
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "seriess" in data and len(data["seriess"]) > 0:
                return data["seriess"][0]
            return None
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return None

    def fetch_series_observations(self, series_id: str, limit: int = 1000) -> List[Dict]:
        """Fetch series observations from FRED API"""
        if not self.api_key:
            logger.warning("FRED API key not configured, skipping API call")
            return []

        url = f"{FRED_API_BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
            "sort_order": "desc",  # Get most recent first
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "observations" in data:
                return data["observations"]
            return []
        except Exception as e:
            logger.error(f"Error fetching observations for {series_id}: {e}")
            return []

    def insert_indicator_metadata(self, series_id: str, info: Dict) -> bool:
        """Insert or update indicator metadata in database"""
        try:
            with self.SessionLocal() as session:
                query = text(
                    """
                    INSERT INTO macro_indicators (id, name, frequency, units, updated_at)
                    VALUES (:id, :name, :frequency, :units, :updated_at)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        frequency = EXCLUDED.frequency,
                        units = EXCLUDED.units,
                        updated_at = EXCLUDED.updated_at
                """
                )

                session.execute(
                    query,
                    {
                        "id": series_id,
                        "name": info["name"],
                        "frequency": info.get("frequency", "Unknown"),
                        "units": info.get("units", "Unknown"),
                        "updated_at": datetime.now(),
                    },
                )
                session.commit()
                logger.info(f"✅ Inserted metadata for {series_id}")
                return True
        except Exception as e:
            logger.error(f"Error inserting metadata for {series_id}: {e}")
            return False

    def insert_observations(self, series_id: str, observations: List[Dict]) -> int:
        """Insert observations into database"""
        inserted = 0

        try:
            with self.SessionLocal() as session:
                for obs in observations:
                    # Skip if value is missing or '.'
                    if obs.get("value") in [".", "", None]:
                        continue

                    try:
                        value = float(obs["value"])
                        date = datetime.strptime(obs["date"], "%Y-%m-%d").date()

                        query = text(
                            """
                            INSERT INTO macro_indicator_values
                            (indicator_id, date, value, is_current, created_at, valid_from)
                            VALUES (:indicator_id, :date, :value, true, :created_at, :valid_from)
                            ON CONFLICT (indicator_id, date, is_current)
                            DO UPDATE SET
                                value = EXCLUDED.value,
                                valid_from = EXCLUDED.valid_from,
                                created_at = EXCLUDED.created_at
                        """
                        )

                        session.execute(
                            query,
                            {
                                "indicator_id": series_id,
                                "date": date,
                                "value": value,
                                "created_at": datetime.now(),
                                "valid_from": datetime.now(),
                            },
                        )
                        inserted += 1
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid observation: {e}")
                        continue

                session.commit()
                logger.info(f"✅ Inserted {inserted} observations for {series_id}")
        except Exception as e:
            logger.error(f"Error inserting observations for {series_id}: {e}")

        return inserted

    def get_indicators_from_database(self) -> List[Dict]:
        """Get all indicators from the macro_indicators driver table

        Returns:
            List of dicts with keys: id, name, frequency, units
        """
        indicators = []
        try:
            with self.SessionLocal() as session:
                query = text(
                    """
                    SELECT id, name, frequency, units
                    FROM macro_indicators
                    ORDER BY id
                """
                )
                result = session.execute(query)

                for row in result:
                    indicators.append({"id": row.id, "name": row.name, "frequency": row.frequency, "units": row.units})

                logger.info(f"📋 Found {len(indicators)} indicators in driver table")
        except Exception as e:
            logger.error(f"Error fetching indicators from database: {e}")

        return indicators

    def fetch_and_populate_indicator(self, series_id: str, metadata: Dict) -> bool:
        """Fetch indicator data from FRED and populate database

        Args:
            series_id: FRED series ID
            metadata: Dict with keys: name, frequency, units

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n📊 Processing {series_id}: {metadata['name']}")

        # Fetch series info from FRED API if available
        series_info = self.fetch_series_info(series_id)

        if series_info:
            # Update metadata with API data (more accurate)
            metadata_to_use = {
                "name": series_info.get("title", metadata["name"]),
                "frequency": series_info.get("frequency_short", metadata["frequency"]),
                "units": series_info.get("units", metadata["units"]),
            }

            # Update database with latest metadata from API
            self.insert_indicator_metadata(series_id, metadata_to_use)
        else:
            # Use database metadata as-is
            metadata_to_use = metadata

        # Fetch and insert observations
        observations = self.fetch_series_observations(series_id)
        if observations:
            count = self.insert_observations(series_id, observations)
            logger.info(f"✅ Successfully populated {series_id} with {count} data points")
            return count > 0
        else:
            logger.warning(f"⚠️  No observations fetched for {series_id}")
            return False

    def populate_all_indicators(self) -> Tuple[int, int]:
        """Populate all indicators from the macro_indicators driver table

        Returns:
            Tuple of (success_count, total_count)
        """
        logger.info("=" * 80)
        logger.info("FRED Data Population - Starting (Database-Driven)")
        logger.info("=" * 80)

        # Get all indicators from database
        indicators = self.get_indicators_from_database()

        if not indicators:
            logger.error("❌ No indicators found in driver table!")
            logger.error("Please populate macro_indicators table first.")
            return 0, 0

        success_count = 0
        total_count = len(indicators)

        for indicator in indicators:
            series_id = indicator["id"]
            metadata = {"name": indicator["name"], "frequency": indicator["frequency"], "units": indicator["units"]}

            if self.fetch_and_populate_indicator(series_id, metadata):
                success_count += 1

        logger.info("\n" + "=" * 80)
        logger.info(f"FRED Data Population - Complete")
        logger.info(f"Success: {success_count}/{total_count} indicators populated")
        logger.info("=" * 80)

        return success_count, total_count

    def populate_all_priority_indicators(self) -> Tuple[int, int]:
        """Legacy method - redirects to populate_all_indicators()

        DEPRECATED: Use populate_all_indicators() instead.
        This method is kept for backwards compatibility.
        """
        logger.warning("⚠️  populate_all_priority_indicators() is deprecated")
        logger.warning("⚠️  Using database-driven populate_all_indicators() instead")
        return self.populate_all_indicators()

    def verify_data(self) -> Dict[str, bool]:
        """Verify that data was populated correctly for all indicators in driver table

        Returns:
            Dict mapping series_id to bool (True if data exists)
        """
        logger.info("\n🔍 Verifying data population...")

        # Get all indicators from database
        indicators = self.get_indicators_from_database()

        if not indicators:
            logger.error("❌ No indicators found in driver table!")
            return {}

        results = {}
        try:
            with self.SessionLocal() as session:
                for indicator in indicators:
                    series_id = indicator["id"]
                    query = text(
                        """
                        SELECT COUNT(*) as count
                        FROM macro_indicator_values
                        WHERE indicator_id = :series_id
                    """
                    )
                    result = session.execute(query, {"series_id": series_id})
                    count = result.scalar()

                    results[series_id] = count > 0
                    status = "✅" if count > 0 else "❌"
                    logger.info(f"{status} {series_id}: {count} data points")
        except Exception as e:
            logger.error(f"Error verifying data: {e}")

        return results


def main():
    """Main function to populate FRED data"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and populate FRED macro indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables (recommended for stockuser)
  export FRED_API_KEY="your_fred_api_key"
  export PGPASSWORD="stockuser_password"
  python3 -m utils.fred_data_fetcher --db-user stockuser

  # Passing credentials directly
  python3 -m utils.fred_data_fetcher --api-key YOUR_KEY --db-user stockuser --db-password PASSWORD

  # Using investigator for read-only verification
  python3 -m utils.fred_data_fetcher --verify-only --db-user investigator

Note: Write operations require 'stockuser' credentials. Read-only user 'investigator'
      can be used with --verify-only flag.
        """,
    )

    parser.add_argument("--api-key", help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--db-user", help="Database user (default: stockuser for write, or DB_USER env var)")
    parser.add_argument("--db-password", help="Database password (or set PGPASSWORD env var)")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing data (read-only, works with investigator user)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without inserting data")

    args = parser.parse_args()

    fetcher = FREDDataFetcher(api_key=args.api_key, db_user=args.db_user, db_password=args.db_password)

    if args.verify_only:
        results = fetcher.verify_data()
        success = sum(1 for v in results.values() if v)
        logger.info(f"\n✅ Verified {success}/{len(results)} indicators have data")
    else:
        if not fetcher.api_key:
            logger.error("❌ FRED API key not configured!")
            logger.error("Set FRED_API_KEY environment variable or pass --api-key")
            logger.error("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            return 1

        if args.dry_run:
            logger.info("🔍 DRY RUN - No data will be inserted")
            indicators = fetcher.get_indicators_from_database()
            for indicator in indicators:
                logger.info(f"Would fetch: {indicator['id']} - {indicator['name']}")
            return 0

        success, total = fetcher.populate_all_indicators()

        # Verify
        logger.info("\n" + "=" * 80)
        results = fetcher.verify_data()
        verified = sum(1 for v in results.values() if v)

        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Populated: {success}/{total} indicators")
        logger.info(f"Verified: {verified}/{total} indicators with data")
        logger.info("=" * 80)

        return 0 if verified == total else 1


if __name__ == "__main__":
    exit(main())
