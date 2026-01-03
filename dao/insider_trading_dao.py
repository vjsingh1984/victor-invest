# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Insider Trading Data Access Object.

Provides database operations for SEC Form 4 insider trading filings
stored in the form4_filings table.

Usage:
    from dao.insider_trading_dao import InsiderTradingDAO

    dao = InsiderTradingDAO()

    # Save a filing
    dao.save_filing(form4_filing)

    # Get recent activity
    filings = dao.get_recent_activity("AAPL", days=30)

    # Get insider sentiment summary
    sentiment = dao.get_insider_sentiment("AAPL", days=90)
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class InsiderTradingDAO:
    """Data Access Object for insider trading data.

    Provides CRUD operations and analysis queries for Form 4 filings
    stored in PostgreSQL.
    """

    def __init__(self, db_config: Optional[Dict] = None, engine: Optional[Engine] = None):
        """Initialize DAO with database configuration.

        Args:
            db_config: Database configuration dict
            engine: Existing SQLAlchemy engine (takes precedence)
        """
        if engine:
            self.engine = engine
        else:
            self.db_config = db_config or {
                "host": "${DB_HOST:-localhost}",
                "port": 5432,
                "database": "sec_database",
                "username": "investigator",
                "password": "investigator",
            }
            self.engine = self._create_engine()

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine."""
        connection_string = (
            f"postgresql://{self.db_config['username']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )

    def save_filing(self, filing: "Form4Filing") -> bool:
        """Save or update a Form 4 filing.

        Args:
            filing: Parsed Form4Filing object

        Returns:
            True if saved successfully
        """
        try:
            query = text("""
                INSERT INTO form4_filings (
                    symbol, cik, accession_number, filing_date,
                    owner_name, owner_title, is_director, is_officer,
                    transaction_type, transaction_code, shares,
                    price_per_share, total_value, is_significant,
                    significance_reasons, filing_data, created_at
                )
                VALUES (
                    :symbol, :cik, :accession_number, :filing_date,
                    :owner_name, :owner_title, :is_director, :is_officer,
                    :transaction_type, :transaction_code, :shares,
                    :price_per_share, :total_value, :is_significant,
                    :significance_reasons, :filing_data, :created_at
                )
                ON CONFLICT (accession_number)
                DO UPDATE SET
                    updated_at = NOW(),
                    is_significant = EXCLUDED.is_significant,
                    significance_reasons = EXCLUDED.significance_reasons
                RETURNING id
            """)

            # Determine primary transaction type
            transaction_type = "Other"
            transaction_code = None
            if filing.transactions:
                for t in filing.transactions:
                    if t.transaction_code:
                        transaction_code = t.transaction_code
                        if t.transaction_type:
                            transaction_type = t.transaction_type.name
                        break

            # Gather significance reasons
            reasons = []
            if filing.reporting_owner and filing.reporting_owner.is_key_insider:
                reasons.append("Key insider")
            if filing.total_purchase_value > 1000000:
                reasons.append(f"Large purchase: ${filing.total_purchase_value:,.0f}")
            if filing.total_sale_value > 1000000:
                reasons.append(f"Large sale: ${filing.total_sale_value:,.0f}")

            params = {
                "symbol": filing.issuer_symbol.upper() if filing.issuer_symbol else None,
                "cik": filing.issuer_cik,
                "accession_number": filing.accession_number,
                "filing_date": filing.filing_date,
                "owner_name": filing.reporting_owner.name if filing.reporting_owner else None,
                "owner_title": filing.reporting_owner.title if filing.reporting_owner else None,
                "is_director": filing.reporting_owner.is_director if filing.reporting_owner else False,
                "is_officer": filing.reporting_owner.is_officer if filing.reporting_owner else False,
                "transaction_type": transaction_type,
                "transaction_code": transaction_code,
                "shares": sum(t.shares for t in filing.transactions),
                "price_per_share": filing.transactions[0].price_per_share if filing.transactions else 0,
                "total_value": filing.net_value,
                "is_significant": filing.is_significant,
                "significance_reasons": reasons,
                "filing_data": json.dumps(filing.to_dict()),
                "created_at": datetime.now(),
            }

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                conn.commit()
                logger.info(f"Saved Form 4 filing: {filing.accession_number}")
                return True

        except Exception as e:
            logger.error(f"Error saving Form 4 filing: {e}")
            return False

    def save_filings_batch(self, filings: List["Form4Filing"]) -> int:
        """Save multiple filings in batch.

        Args:
            filings: List of Form4Filing objects

        Returns:
            Number of filings saved
        """
        saved = 0
        for filing in filings:
            if self.save_filing(filing):
                saved += 1
        return saved

    def get_recent_activity(
        self,
        symbol: str,
        days: int = 30,
        significant_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get recent insider activity for a symbol.

        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            significant_only: Only return significant filings

        Returns:
            List of filing dictionaries
        """
        try:
            cutoff_date = date.today() - timedelta(days=days)

            query_str = """
                SELECT
                    accession_number,
                    filing_date,
                    owner_name,
                    owner_title,
                    is_director,
                    is_officer,
                    transaction_type,
                    transaction_code,
                    shares,
                    price_per_share,
                    total_value,
                    is_significant,
                    significance_reasons,
                    filing_data
                FROM form4_filings
                WHERE UPPER(symbol) = UPPER(:symbol)
                AND filing_date >= :cutoff_date
            """

            if significant_only:
                query_str += " AND is_significant = TRUE"

            query_str += " ORDER BY filing_date DESC LIMIT 100"

            query = text(query_str)

            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "cutoff_date": cutoff_date
                })

                filings = []
                for row in result.fetchall():
                    filings.append({
                        "accession_number": row[0],
                        "filing_date": str(row[1]) if row[1] else None,
                        "owner_name": row[2],
                        "owner_title": row[3],
                        "is_director": row[4],
                        "is_officer": row[5],
                        "transaction_type": row[6],
                        "transaction_code": row[7],
                        "shares": float(row[8]) if row[8] else 0,
                        "price_per_share": float(row[9]) if row[9] else 0,
                        "total_value": float(row[10]) if row[10] else 0,
                        "is_significant": row[11],
                        "significance_reasons": row[12],
                        "filing_data": row[13],
                    })

                return filings

        except Exception as e:
            logger.error(f"Error getting recent activity for {symbol}: {e}")
            return []

    def get_insider_sentiment(
        self,
        symbol: str,
        days: int = 90
    ) -> Dict[str, Any]:
        """Calculate insider sentiment for a symbol.

        Sentiment is based on the ratio of buys to sells and total value.

        Args:
            symbol: Stock ticker symbol
            days: Analysis period in days

        Returns:
            Dictionary with sentiment metrics
        """
        try:
            cutoff_date = date.today() - timedelta(days=days)

            query = text("""
                SELECT
                    COUNT(*) FILTER (WHERE transaction_code = 'P') AS purchase_count,
                    COUNT(*) FILTER (WHERE transaction_code = 'S') AS sale_count,
                    COALESCE(SUM(CASE WHEN transaction_code = 'P' THEN total_value ELSE 0 END), 0) AS purchase_value,
                    COALESCE(SUM(CASE WHEN transaction_code = 'S' THEN ABS(total_value) ELSE 0 END), 0) AS sale_value,
                    COUNT(DISTINCT owner_name) AS unique_insiders,
                    COUNT(*) FILTER (WHERE is_significant = TRUE) AS significant_filings
                FROM form4_filings
                WHERE UPPER(symbol) = UPPER(:symbol)
                AND filing_date >= :cutoff_date
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "cutoff_date": cutoff_date
                }).fetchone()

                if not result:
                    return self._empty_sentiment(symbol, days)

                purchase_count = result[0] or 0
                sale_count = result[1] or 0
                purchase_value = float(result[2] or 0)
                sale_value = float(result[3] or 0)
                unique_insiders = result[4] or 0
                significant_filings = result[5] or 0

                # Calculate sentiment score (-1 to +1)
                total_count = purchase_count + sale_count
                if total_count > 0:
                    count_sentiment = (purchase_count - sale_count) / total_count
                else:
                    count_sentiment = 0

                total_value = purchase_value + sale_value
                if total_value > 0:
                    value_sentiment = (purchase_value - sale_value) / total_value
                else:
                    value_sentiment = 0

                # Weighted sentiment (60% value, 40% count)
                sentiment_score = 0.6 * value_sentiment + 0.4 * count_sentiment

                # Classify sentiment
                if sentiment_score > 0.3:
                    sentiment_label = "bullish"
                elif sentiment_score > 0.1:
                    sentiment_label = "slightly_bullish"
                elif sentiment_score < -0.3:
                    sentiment_label = "bearish"
                elif sentiment_score < -0.1:
                    sentiment_label = "slightly_bearish"
                else:
                    sentiment_label = "neutral"

                # Detect cluster buying/selling
                cluster_detected = (
                    (purchase_count >= 3 and purchase_value > 500000) or
                    (sale_count >= 3 and sale_value > 500000)
                )

                return {
                    "symbol": symbol,
                    "period_days": days,
                    "sentiment_score": round(sentiment_score, 4),
                    "sentiment_label": sentiment_label,
                    "purchase_count": purchase_count,
                    "sale_count": sale_count,
                    "purchase_value": purchase_value,
                    "sale_value": sale_value,
                    "net_value": purchase_value - sale_value,
                    "unique_insiders": unique_insiders,
                    "significant_filings": significant_filings,
                    "cluster_detected": cluster_detected,
                    "analysis_date": str(date.today()),
                }

        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return self._empty_sentiment(symbol, days)

    def _empty_sentiment(self, symbol: str, days: int) -> Dict[str, Any]:
        """Return empty sentiment structure."""
        return {
            "symbol": symbol,
            "period_days": days,
            "sentiment_score": 0,
            "sentiment_label": "no_data",
            "purchase_count": 0,
            "sale_count": 0,
            "purchase_value": 0,
            "sale_value": 0,
            "net_value": 0,
            "unique_insiders": 0,
            "significant_filings": 0,
            "cluster_detected": False,
            "analysis_date": str(date.today()),
        }

    def get_key_insider_transactions(
        self,
        symbol: str,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get transactions from key insiders only.

        Key insiders: CEO, CFO, Directors, 10% owners

        Args:
            symbol: Stock ticker symbol
            days: Analysis period

        Returns:
            List of key insider transactions
        """
        try:
            cutoff_date = date.today() - timedelta(days=days)

            query = text("""
                SELECT
                    accession_number,
                    filing_date,
                    owner_name,
                    owner_title,
                    is_director,
                    transaction_type,
                    shares,
                    total_value
                FROM form4_filings
                WHERE UPPER(symbol) = UPPER(:symbol)
                AND filing_date >= :cutoff_date
                AND (
                    is_director = TRUE
                    OR LOWER(owner_title) LIKE '%chief executive%'
                    OR LOWER(owner_title) LIKE '%ceo%'
                    OR LOWER(owner_title) LIKE '%chief financial%'
                    OR LOWER(owner_title) LIKE '%cfo%'
                    OR LOWER(owner_title) LIKE '%president%'
                )
                ORDER BY filing_date DESC
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "cutoff_date": cutoff_date
                })

                transactions = []
                for row in result.fetchall():
                    transactions.append({
                        "accession_number": row[0],
                        "filing_date": str(row[1]) if row[1] else None,
                        "owner_name": row[2],
                        "owner_title": row[3],
                        "is_director": row[4],
                        "transaction_type": row[5],
                        "shares": float(row[6]) if row[6] else 0,
                        "total_value": float(row[7]) if row[7] else 0,
                    })

                return transactions

        except Exception as e:
            logger.error(f"Error getting key insider transactions for {symbol}: {e}")
            return []

    def get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CIK string or None
        """
        try:
            query = text("""
                SELECT cik
                FROM ticker_cik_mapping
                WHERE UPPER(ticker) = UPPER(:symbol)
                LIMIT 1
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol}).fetchone()
                if result:
                    return str(result[0]).zfill(10)

        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")

        return None


# Singleton instance
_insider_dao: Optional[InsiderTradingDAO] = None


def get_insider_trading_dao() -> InsiderTradingDAO:
    """Get or create singleton DAO instance."""
    global _insider_dao
    if _insider_dao is None:
        _insider_dao = InsiderTradingDAO()
    return _insider_dao
