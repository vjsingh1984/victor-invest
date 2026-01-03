"""
SEC Bulk Table Data Access Object

Provides unified access to SEC DERA bulk tables (sec_num_data, sec_sub_data) with
automatic XBRL tag normalization using the tag alias mapper.

Usage:
    from dao.sec_bulk_dao import SECBulkDAO

    dao = SECBulkDAO()
    metrics = dao.fetch_financial_metrics('AAPL', 2024, 'FY')
"""

import logging
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from utils.sec_data_normalizer import SECDataNormalizer
from utils.financial_calculators import FinancialCalculator

logger = logging.getLogger(__name__)


class SECBulkDAO:
    """
    Data Access Object for SEC bulk tables with tag normalization.

    Provides:
    1. Query execution against sec_num_data/sec_sub_data
    2. Automatic XBRL tag → canonical name conversion
    3. CIK ↔ Symbol mapping
    4. Fiscal period filtering
    """

    # FAANG company CIKs for quick testing
    FAANG_CIKS = {
        "AAPL": "0000320193",
        "AMZN": "0001018724",
        "META": "0001326801",
        "GOOGL": "0001652044",
        "NFLX": "0001065280",
    }

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize with database configuration.

        Args:
            db_config: Database config dict or None for default
        """
        self.db_config = db_config or {
            "host": "${DB_HOST:-localhost}",
            "port": 5432,
            "database": "sec_database",
            "username": "investigator",
            "password": "investigator",
        }
        self.engine = self._create_engine()
        self.normalizer = SECDataNormalizer()

    def _create_engine(self):
        """Create SQLAlchemy engine."""
        connection_string = (
            f"postgresql://{self.db_config['username']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string, pool_size=5, max_overflow=10, pool_pre_ping=True, echo=False)

    def get_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            CIK string or None if not found
        """
        # Check FAANG mapping first (fast path)
        if symbol.upper() in self.FAANG_CIKS:
            return self.FAANG_CIKS[symbol.upper()]

        # Query database
        query = text(
            """
            SELECT cik
            FROM ticker_cik_mapping
            WHERE UPPER(ticker) = UPPER(:symbol)
            LIMIT 1
        """
        )

        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol}).fetchone()
            return result[0] if result else None

    def fetch_financial_metrics(
        self, symbol: str, fiscal_year: int, fiscal_period: str = "FY", form_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch financial metrics from bulk tables with tag normalization.

        Args:
            symbol: Stock ticker
            fiscal_year: Fiscal year (e.g., 2024)
            fiscal_period: Fiscal period (FY, Q1, Q2, Q3, Q4)
            form_types: Form types to filter (default: ['10-K', '10-Q'])

        Returns:
            Dict with canonical snake_case keys

        Example:
            >>> dao = SECBulkDAO()
            >>> metrics = dao.fetch_financial_metrics('AAPL', 2024, 'FY')
            >>> print(metrics['total_revenue'])  # 394_328_000_000
        """
        # Get CIK
        cik = self.get_cik(symbol)
        if not cik:
            logger.error(f"CIK not found for symbol: {symbol}")
            return {"symbol": symbol, "error": "CIK not found"}

        # Default form types
        if form_types is None:
            form_types = ["10-K", "10-Q", "10-K/A", "10-Q/A"]

        # Convert fiscal period to qtrs (quarters)
        qtrs = self._fiscal_period_to_qtrs(fiscal_period)

        # Query bulk tables
        # NOTE: Balance sheet items use qtrs=0 (point-in-time snapshot)
        #       Income statement/cash flow items use qtrs=4 for annual (cumulative)
        #       We need to query for BOTH qtrs values to get complete financials
        #       Filter by submission's fiscal year/period (s.fy, s.fp)
        #       Use latest ddate for each tag (handles comparative data in filings)
        query = text(
            """
            SELECT DISTINCT ON (n.tag) n.tag, n.value, n.uom, n.ddate, s.form, s.adsh
            FROM sec_num_data n
            JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
            WHERE s.cik = :cik
              AND s.fy = :fiscal_year
              AND s.fp = :fiscal_period
              AND (n.qtrs = :qtrs OR n.qtrs = 0)
              AND s.form = ANY(:form_types)
              AND n.value IS NOT NULL
              AND (n.segments IS NULL OR n.segments = '')
              AND (n.coreg IS NULL OR n.coreg = '')
            ORDER BY n.tag, n.ddate DESC, n.qtrs DESC
        """
        )

        with self.engine.connect() as conn:
            results = conn.execute(
                query,
                {
                    "cik": cik,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "qtrs": qtrs,
                    "form_types": form_types,
                },
            ).fetchall()

        if not results:
            logger.warning(
                f"No data found for {symbol} (CIK: {cik}) " f"fiscal year {fiscal_year}, period {fiscal_period}"
            )
            return {
                "symbol": symbol,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "error": "No data found",
            }

        # Convert to list of dicts
        rows = [{"tag": r[0], "value": r[1], "uom": r[2], "ddate": r[3], "form": r[4], "adsh": r[5]} for r in results]

        # Log raw tag coverage before normalization
        unique_tags = set(r["tag"] for r in rows)
        logger.debug(f"Fetched {len(rows)} data points with {len(unique_tags)} unique XBRL tags for {symbol}")

        # Normalize using tag mapper
        metrics = self.normalizer.normalize_bulk_table_results(rows, symbol)
        metrics["fiscal_year"] = fiscal_year
        metrics["fiscal_period"] = fiscal_period

        # Enrich with calculated metrics (e.g., total_liabilities from components)
        metrics = FinancialCalculator.enrich_metrics(metrics, symbol)

        return metrics

    def fetch_metrics_multiple_periods(
        self, symbol: str, fiscal_year: int, periods: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch metrics for multiple fiscal periods.

        Args:
            symbol: Stock ticker
            fiscal_year: Fiscal year
            periods: List of periods (default: ['FY', 'Q1', 'Q2', 'Q3', 'Q4'])

        Returns:
            Dict mapping period → metrics dict

        Example:
            >>> dao = SECBulkDAO()
            >>> results = dao.fetch_metrics_multiple_periods('AAPL', 2024)
            >>> print(results['FY']['total_revenue'])
            >>> print(results['Q1']['total_revenue'])
        """
        if periods is None:
            periods = ["FY", "Q1", "Q2", "Q3", "Q4"]

        results = {}
        for period in periods:
            metrics = self.fetch_financial_metrics(symbol, fiscal_year, period)
            if "error" not in metrics:
                results[period] = metrics

        return results

    def fetch_latest_annual_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch most recent annual (10-K) metrics.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with canonical metrics

        Example:
            >>> dao = SECBulkDAO()
            >>> metrics = dao.fetch_latest_annual_metrics('AAPL')
            >>> print(metrics['total_revenue'])
        """
        # Get CIK
        cik = self.get_cik(symbol)
        if not cik:
            logger.error(f"CIK not found for symbol: {symbol}")
            return {"symbol": symbol, "error": "CIK not found"}

        # Query for latest annual filing
        query = text(
            """
            SELECT DISTINCT EXTRACT(YEAR FROM n.ddate)::int as fiscal_year
            FROM sec_num_data n
            JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
            WHERE s.cik = :cik
              AND s.form IN ('10-K', '10-K/A')
              AND n.qtrs = 4
            ORDER BY fiscal_year DESC
            LIMIT 1
        """
        )

        with self.engine.connect() as conn:
            result = conn.execute(query, {"cik": cik}).fetchone()

        if not result:
            logger.warning(f"No annual filings found for {symbol}")
            return {"symbol": symbol, "error": "No annual filings found"}

        fiscal_year = result[0]
        logger.info(f"Latest annual filing for {symbol}: {fiscal_year}-FY")

        return self.fetch_financial_metrics(symbol, fiscal_year, "FY", form_types=["10-K", "10-K/A"])

    def fetch_latest_quarterly_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch most recent quarterly (10-Q) metrics.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with canonical metrics
        """
        # Get CIK
        cik = self.get_cik(symbol)
        if not cik:
            logger.error(f"CIK not found for symbol: {symbol}")
            return {"symbol": symbol, "error": "CIK not found"}

        # Query for latest quarterly filing (Q1, Q2, Q3 only - exclude Q4)
        query = text(
            """
            SELECT DISTINCT
                EXTRACT(YEAR FROM n.ddate)::int as fiscal_year,
                s.fp as fiscal_period
            FROM sec_num_data n
            JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
            WHERE s.cik = :cik
              AND s.form IN ('10-Q', '10-Q/A')
              AND n.qtrs IN (1, 2, 3)
            ORDER BY fiscal_year DESC, fiscal_period DESC
            LIMIT 1
        """
        )

        with self.engine.connect() as conn:
            result = conn.execute(query, {"cik": cik}).fetchone()

        if not result:
            logger.warning(f"No quarterly filings found for {symbol}")
            return {"symbol": symbol, "error": "No quarterly filings found"}

        fiscal_year, fiscal_period = result[0], result[1]
        logger.info(f"Latest quarterly filing for {symbol}: {fiscal_year}-{fiscal_period}")

        return self.fetch_financial_metrics(symbol, fiscal_year, fiscal_period, form_types=["10-Q", "10-Q/A"])

    def _fiscal_period_to_qtrs(self, fiscal_period: str) -> int:
        """
        Convert fiscal period string to qtrs value.

        Args:
            fiscal_period: Fiscal period (FY, Q1, Q2, Q3, Q4)

        Returns:
            Qtrs value (1, 2, 3, 4)
        """
        mapping = {
            "FY": 4,
            "Q4": 4,
            "Q3": 3,
            "Q2": 2,
            "Q1": 1,
        }
        return mapping.get(fiscal_period.upper(), 4)

    def get_available_periods(self, symbol: str, fiscal_year: int) -> List[str]:
        """
        Get list of available fiscal periods for a symbol and year.

        Args:
            symbol: Stock ticker
            fiscal_year: Fiscal year

        Returns:
            List of available periods (e.g., ['FY', 'Q1', 'Q2', 'Q3'])
        """
        cik = self.get_cik(symbol)
        if not cik:
            return []

        query = text(
            """
            SELECT DISTINCT s.fp as fiscal_period
            FROM sec_num_data n
            JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
            WHERE s.cik = :cik
              AND EXTRACT(YEAR FROM n.ddate) = :fiscal_year
              AND s.form IN ('10-K', '10-Q', '10-K/A', '10-Q/A')
            ORDER BY fiscal_period
        """
        )

        with self.engine.connect() as conn:
            results = conn.execute(query, {"cik": cik, "fiscal_year": fiscal_year}).fetchall()

        return [r[0] for r in results]


# Convenience function
def fetch_sec_bulk_metrics(
    symbol: str, fiscal_year: int, fiscal_period: str = "FY", db_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Quick fetch of SEC bulk table metrics.

    Args:
        symbol: Stock ticker
        fiscal_year: Fiscal year
        fiscal_period: Fiscal period
        db_config: Optional database config

    Returns:
        Dict with canonical metrics
    """
    dao = SECBulkDAO(db_config)
    return dao.fetch_financial_metrics(symbol, fiscal_year, fiscal_period)
