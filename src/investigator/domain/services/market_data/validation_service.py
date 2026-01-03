# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Data Validation Service - Quality checks and anomaly detection for market data.

This service provides:
- Shares data validation (SEC vs symbol table mismatch)
- Market cap consistency checks
- Split detection warnings
- Financial data quality scoring

Example:
    service = DataValidationService()

    # Validate shares data
    warnings = service.validate_shares("AAPL", current_price=150.0)

    # Check for data quality issues
    quality = service.assess_data_quality("AAPL", financials)
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class WarningSeverity(Enum):
    """Severity levels for data quality warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DataQualityWarning:
    """Container for a data quality warning."""

    code: str
    message: str
    severity: WarningSeverity
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


class DataValidationService:
    """
    Service for validating market data quality.

    Performs consistency checks across data sources to identify
    potential issues like unaccounted splits, stale data, or
    data source mismatches.
    """

    def __init__(
        self,
        sec_db_url: str = "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database",
        stock_db_url: str = "postgresql://stockuser:${STOCK_DB_PASSWORD}@${STOCK_DB_HOST}:5432/stock",
    ):
        """
        Initialize DataValidationService with database connections.

        Args:
            sec_db_url: Connection string for SEC database
            stock_db_url: Connection string for stock database
        """
        self.sec_engine = create_engine(
            sec_db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.stock_engine = create_engine(
            stock_db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def validate_shares(
        self,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> List[DataQualityWarning]:
        """
        Validate shares data for consistency across sources.

        Checks:
        1. SEC shares vs symbol table shares (split detection)
        2. Market cap consistency
        3. Shares staleness (old data)

        Args:
            symbol: Stock ticker
            current_price: Current stock price for market cap validation

        Returns:
            List of DataQualityWarning objects
        """
        warnings = []

        # Get SEC shares (most recent)
        sec_shares = self._get_sec_shares(symbol)

        # Get symbol table shares
        symbol_data = self._get_symbol_data(symbol)
        symbol_shares = symbol_data.get("shares_outstanding") if symbol_data else None
        symbol_mktcap = symbol_data.get("market_cap") if symbol_data else None

        # Check 1: SEC vs symbol table shares mismatch (potential split)
        if sec_shares and symbol_shares:
            ratio = sec_shares / symbol_shares if symbol_shares > 0 else 0

            if ratio > 1.8:  # SEC has more shares - possible recent split not in symbol table
                warnings.append(
                    DataQualityWarning(
                        code="SHARES_MISMATCH_FORWARD_SPLIT",
                        message=f"SEC shares ({sec_shares/1e9:.2f}B) >> symbol table ({symbol_shares/1e9:.2f}B) - possible forward split",
                        severity=WarningSeverity.WARNING,
                        details={
                            "sec_shares": sec_shares,
                            "symbol_shares": symbol_shares,
                            "ratio": ratio,
                            "likely_split": f"{round(ratio)}:1",
                        },
                    )
                )
            elif ratio < 0.55:  # SEC has fewer shares - possible reverse split
                warnings.append(
                    DataQualityWarning(
                        code="SHARES_MISMATCH_REVERSE_SPLIT",
                        message=f"SEC shares ({sec_shares/1e9:.2f}B) << symbol table ({symbol_shares/1e9:.2f}B) - possible reverse split",
                        severity=WarningSeverity.WARNING,
                        details={
                            "sec_shares": sec_shares,
                            "symbol_shares": symbol_shares,
                            "ratio": ratio,
                            "likely_split": f"1:{round(1/ratio)}",
                        },
                    )
                )

        # Check 2: Market cap consistency
        if current_price and symbol_shares and symbol_mktcap:
            calc_mktcap = symbol_shares * current_price
            mktcap_ratio = calc_mktcap / symbol_mktcap if symbol_mktcap > 0 else 0

            if mktcap_ratio > 2 or mktcap_ratio < 0.5:
                warnings.append(
                    DataQualityWarning(
                        code="MKTCAP_MISMATCH",
                        message=f"Calculated mktcap (${calc_mktcap/1e9:.1f}B) differs from stored (${symbol_mktcap/1e9:.1f}B)",
                        severity=WarningSeverity.WARNING,
                        details={
                            "calculated_mktcap": calc_mktcap,
                            "stored_mktcap": symbol_mktcap,
                            "ratio": mktcap_ratio,
                            "current_price": current_price,
                            "shares_used": symbol_shares,
                        },
                    )
                )

        # Check 3: SEC data staleness
        sec_filed_date = self._get_sec_latest_filed_date(symbol)
        if sec_filed_date:
            days_old = (date.today() - sec_filed_date).days
            if days_old > 120:  # More than 4 months old
                warnings.append(
                    DataQualityWarning(
                        code="SEC_DATA_STALE",
                        message=f"SEC data is {days_old} days old (filed {sec_filed_date})",
                        severity=WarningSeverity.INFO,
                        details={
                            "filed_date": sec_filed_date,
                            "days_old": days_old,
                        },
                    )
                )

        return warnings

    def validate_financials(
        self,
        symbol: str,
        financials: Dict[str, Any],
    ) -> List[DataQualityWarning]:
        """
        Validate financial data for completeness and consistency.

        Checks:
        1. Required fields presence
        2. Value sanity (e.g., positive revenue)
        3. Ratio consistency

        Args:
            symbol: Stock ticker
            financials: Financial data dictionary

        Returns:
            List of DataQualityWarning objects
        """
        warnings = []

        # Required fields for valuation
        required_fields = ["total_revenue", "net_income"]
        for field in required_fields:
            if not financials.get(field):
                warnings.append(
                    DataQualityWarning(
                        code="MISSING_REQUIRED_FIELD",
                        message=f"Missing required field: {field}",
                        severity=WarningSeverity.ERROR,
                        details={"field": field},
                    )
                )

        # Sanity checks
        revenue = financials.get("total_revenue", 0) or 0
        net_income = financials.get("net_income", 0) or 0
        equity = financials.get("stockholders_equity", 0) or 0

        if revenue < 0:
            warnings.append(
                DataQualityWarning(
                    code="NEGATIVE_REVENUE",
                    message=f"Negative revenue: {revenue}",
                    severity=WarningSeverity.ERROR,
                    details={"revenue": revenue},
                )
            )

        if equity < 0 and net_income > 0:
            warnings.append(
                DataQualityWarning(
                    code="NEGATIVE_EQUITY_PROFITABLE",
                    message=f"Negative equity ({equity}) but positive net income ({net_income})",
                    severity=WarningSeverity.WARNING,
                    details={"equity": equity, "net_income": net_income},
                )
            )

        # Net margin sanity (> 100% is suspicious)
        if revenue > 0 and net_income > 0:
            net_margin = net_income / revenue
            if net_margin > 1.0:
                warnings.append(
                    DataQualityWarning(
                        code="SUSPICIOUS_NET_MARGIN",
                        message=f"Net margin > 100%: {net_margin:.1%}",
                        severity=WarningSeverity.WARNING,
                        details={"net_margin": net_margin, "net_income": net_income, "revenue": revenue},
                    )
                )

        return warnings

    def assess_data_quality(
        self,
        symbol: str,
        financials: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.

        Args:
            symbol: Stock ticker
            financials: Optional financial data
            current_price: Optional current price

        Returns:
            Dict with quality score and warnings
        """
        all_warnings = []

        # Validate shares
        shares_warnings = self.validate_shares(symbol, current_price)
        all_warnings.extend(shares_warnings)

        # Validate financials if provided
        if financials:
            fin_warnings = self.validate_financials(symbol, financials)
            all_warnings.extend(fin_warnings)

        # Calculate quality score
        error_count = sum(1 for w in all_warnings if w.severity == WarningSeverity.ERROR)
        warning_count = sum(1 for w in all_warnings if w.severity == WarningSeverity.WARNING)
        info_count = sum(1 for w in all_warnings if w.severity == WarningSeverity.INFO)

        # Score: 100 - (30 * errors) - (10 * warnings) - (2 * info)
        score = max(0, 100 - (30 * error_count) - (10 * warning_count) - (2 * info_count))

        return {
            "symbol": symbol,
            "quality_score": score,
            "assessment": "excellent" if score >= 90 else "good" if score >= 70 else "fair" if score >= 50 else "poor",
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "warnings": [str(w) for w in all_warnings],
            "warning_details": [
                {
                    "code": w.code,
                    "message": w.message,
                    "severity": w.severity.value,
                    "details": w.details,
                }
                for w in all_warnings
            ],
        }

    def _get_sec_shares(self, symbol: str) -> Optional[float]:
        """Get shares from most recent SEC filing."""
        with self.sec_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT shares_outstanding
                    FROM sec_companyfacts_processed
                    WHERE symbol = :symbol
                      AND shares_outstanding IS NOT NULL
                    ORDER BY filed_date DESC
                    LIMIT 1
                """
                ),
                {"symbol": symbol},
            ).fetchone()
            return float(result[0]) if result and result[0] else None

    def _get_sec_latest_filed_date(self, symbol: str) -> Optional[date]:
        """Get most recent SEC filing date."""
        with self.sec_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT filed_date
                    FROM sec_companyfacts_processed
                    WHERE symbol = :symbol
                    ORDER BY filed_date DESC
                    LIMIT 1
                """
                ),
                {"symbol": symbol},
            ).fetchone()
            return result[0] if result else None

    def _get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from symbol table."""
        with self.stock_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT outstandingshares, mktcap, "Sector", "Industry"
                    FROM symbol
                    WHERE ticker = :symbol
                """
                ),
                {"symbol": symbol},
            ).fetchone()

            if result:
                return {
                    "shares_outstanding": float(result[0]) if result[0] else None,
                    "market_cap": float(result[1]) if result[1] else None,
                    "sector": result[2],
                    "industry": result[3],
                }
            return None
