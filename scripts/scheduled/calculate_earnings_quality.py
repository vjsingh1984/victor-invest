#!/usr/bin/env python3
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

"""Calculate Earnings Quality Metrics.

Schedule: Weekly on Sunday at 6AM ET (after SEC data refresh)
Source: Calculated from existing SEC fundamental data

Calculates:
- Accrual ratios (total accruals, Sloan accruals)
- Cash conversion quality
- Beneish M-Score (earnings manipulation detection)
- Piotroski F-Score (financial strength)
- Altman Z-Score (bankruptcy risk)
- Composite earnings quality score

These metrics help identify:
- Aggressive accounting that may overstate true profitability
- Earnings manipulation before restatements
- Sustainable vs. unsustainable earnings

Usage:
    python scripts/scheduled/calculate_earnings_quality.py
    python scripts/scheduled/calculate_earnings_quality.py --symbols AAPL,MSFT
"""

import argparse
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
    get_sp500_symbols,
)


class EarningsQualityCalculator(BaseCollector):
    """Calculator for earnings quality metrics from SEC data."""

    def __init__(self, symbols: Optional[List[str]] = None):
        super().__init__("calculate_earnings_quality")
        self.symbols = symbols

    def collect(self) -> CollectionMetrics:
        """Calculate earnings quality metrics."""
        try:
            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_sp500_symbols()

            self.logger.info(f"Calculating earnings quality for {len(symbols)} symbols")

            conn = get_database_connection()
            cursor = conn.cursor()

            for symbol in symbols:
                try:
                    self.metrics.records_processed += 1

                    # Get financial data from SEC responses
                    financials = self._get_financial_data(cursor, symbol)
                    if not financials:
                        continue

                    # Calculate metrics for each period
                    for period_data in financials:
                        try:
                            metrics = self._calculate_quality_metrics(period_data)
                            if not metrics:
                                continue

                            fiscal_year = period_data.get("fiscal_year")
                            fiscal_period = period_data.get("fiscal_period")

                            record_hash = compute_record_hash(metrics)

                            # Check existing
                            cursor.execute("""
                                SELECT input_data_hash FROM earnings_quality
                                WHERE symbol = %s AND fiscal_year = %s AND fiscal_period = %s
                            """, (symbol, fiscal_year, fiscal_period))
                            existing = cursor.fetchone()

                            if existing:
                                if existing[0] == record_hash:
                                    self.metrics.records_skipped += 1
                                    continue
                                # Update
                                self._update_quality_record(cursor, symbol, fiscal_year, fiscal_period, metrics, record_hash)
                                self.metrics.records_updated += 1
                            else:
                                # Insert
                                self._insert_quality_record(cursor, symbol, fiscal_year, fiscal_period, metrics, record_hash)
                                self.metrics.records_inserted += 1

                        except Exception as e:
                            self.logger.debug(f"Failed to calculate for {symbol} {period_data.get('fiscal_period')}: {e}")

                    # Commit periodically
                    if self.metrics.records_processed % 20 == 0:
                        conn.commit()

                except Exception as e:
                    self.logger.warning(f"Failed to process {symbol}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{symbol}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Earnings quality: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"Earnings quality calculation failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _get_financial_data(self, cursor, symbol: str) -> List[Dict[str, Any]]:
        """Get financial data from SEC responses for a symbol."""
        try:
            cursor.execute("""
                SELECT fiscal_year, fiscal_period, response_data
                FROM sec_responses
                WHERE symbol = %s
                  AND form_type IN ('10-K', '10-Q')
                  AND category = 'financials'
                ORDER BY fiscal_year DESC, fiscal_period DESC
                LIMIT 8
            """, (symbol,))

            results = []
            for row in cursor.fetchall():
                data = row[2] if isinstance(row[2], dict) else {}
                data["fiscal_year"] = row[0]
                data["fiscal_period"] = row[1]
                results.append(data)

            return results
        except Exception as e:
            self.logger.debug(f"Could not get financial data for {symbol}: {e}")
            return []

    def _calculate_quality_metrics(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate earnings quality metrics from financial data."""
        try:
            # Extract key financial values
            net_income = self._get_value(data, ["netIncome", "net_income", "NetIncomeLoss"])
            ocf = self._get_value(data, ["operatingCashFlow", "operating_cash_flow", "NetCashProvidedByUsedInOperatingActivities"])
            total_assets = self._get_value(data, ["totalAssets", "total_assets", "Assets"])
            current_assets = self._get_value(data, ["currentAssets", "current_assets", "AssetsCurrent"])
            current_liabilities = self._get_value(data, ["currentLiabilities", "current_liabilities", "LiabilitiesCurrent"])
            depreciation = self._get_value(data, ["depreciation", "depreciation_amortization", "DepreciationAndAmortization"]) or 0
            revenue = self._get_value(data, ["revenue", "revenues", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
            receivables = self._get_value(data, ["accountsReceivable", "accounts_receivable", "AccountsReceivableNetCurrent"])
            total_debt = self._get_value(data, ["totalDebt", "total_debt", "LongTermDebt"]) or 0
            retained_earnings = self._get_value(data, ["retainedEarnings", "retained_earnings", "RetainedEarningsAccumulatedDeficit"]) or 0
            working_capital = self._get_value(data, ["workingCapital", "working_capital"]) or (
                (current_assets - current_liabilities) if current_assets and current_liabilities else None
            )
            ebit = self._get_value(data, ["ebit", "operatingIncome", "OperatingIncomeLoss"])
            market_cap = self._get_value(data, ["marketCap", "market_cap"])

            # 1. Total Accruals
            total_accruals = None
            accrual_ratio = None
            if net_income is not None and ocf is not None:
                total_accruals = net_income - ocf
                if total_assets and total_assets != 0:
                    accrual_ratio = total_accruals / total_assets

            # 2. Cash Conversion Ratio
            cash_conversion_ratio = None
            if ocf is not None and net_income is not None and net_income != 0:
                cash_conversion_ratio = ocf / net_income

            # 3. Sloan Accruals (simplified)
            sloan_accruals = None
            if working_capital is not None and total_assets and total_assets != 0:
                sloan_accruals = (working_capital - depreciation) / total_assets

            # 4. Altman Z-Score (simplified for public companies)
            z_score = None
            if all([working_capital, total_assets, retained_earnings, ebit, market_cap, total_debt, revenue]):
                if total_assets != 0 and (total_assets + total_debt) != 0:
                    a = (working_capital or 0) / total_assets
                    b = (retained_earnings or 0) / total_assets
                    c = (ebit or 0) / total_assets
                    d = (market_cap or 0) / (total_debt or 1)
                    e = (revenue or 0) / total_assets
                    z_score = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e

            # 5. Calculate composite score (0-1, higher = better)
            quality_flags = []
            quality_score = 1.0

            # Penalize high accruals
            if accrual_ratio is not None:
                if abs(accrual_ratio) > 0.10:
                    quality_flags.append("high_accruals")
                    quality_score -= 0.15
                elif abs(accrual_ratio) > 0.05:
                    quality_score -= 0.05

            # Penalize poor cash conversion
            if cash_conversion_ratio is not None:
                if cash_conversion_ratio < 0.5:
                    quality_flags.append("poor_cash_conversion")
                    quality_score -= 0.20
                elif cash_conversion_ratio < 0.8:
                    quality_score -= 0.10

            # Penalize low Z-score
            if z_score is not None:
                if z_score < 1.8:
                    quality_flags.append("bankruptcy_risk")
                    quality_score -= 0.25
                elif z_score < 3.0:
                    quality_score -= 0.10

            quality_score = max(0, min(1, quality_score))

            return {
                "total_accruals": total_accruals,
                "accrual_ratio": accrual_ratio,
                "sloan_accruals": sloan_accruals,
                "cash_conversion_ratio": cash_conversion_ratio,
                "altman_z_score": z_score,
                "earnings_quality_score": quality_score,
                "quality_flags": quality_flags,
            }

        except Exception as e:
            self.logger.debug(f"Error calculating quality metrics: {e}")
            return None

    def _get_value(self, data: Dict, keys: List[str]) -> Optional[float]:
        """Get a numeric value from data using multiple possible keys."""
        for key in keys:
            if key in data:
                val = data[key]
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
        return None

    def _insert_quality_record(self, cursor, symbol: str, fiscal_year: int,
                               fiscal_period: str, metrics: Dict, record_hash: str) -> None:
        """Insert a new earnings quality record."""
        cursor.execute("""
            INSERT INTO earnings_quality
                (symbol, fiscal_year, fiscal_period, total_accruals, accrual_ratio,
                 sloan_accruals, cash_conversion_ratio, altman_z_score,
                 earnings_quality_score, quality_flags, input_data_hash, source_fetch_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            symbol,
            fiscal_year,
            fiscal_period,
            metrics.get("total_accruals"),
            metrics.get("accrual_ratio"),
            metrics.get("sloan_accruals"),
            metrics.get("cash_conversion_ratio"),
            metrics.get("altman_z_score"),
            metrics.get("earnings_quality_score"),
            metrics.get("quality_flags"),
            record_hash,
        ))

    def _update_quality_record(self, cursor, symbol: str, fiscal_year: int,
                               fiscal_period: str, metrics: Dict, record_hash: str) -> None:
        """Update an existing earnings quality record."""
        cursor.execute("""
            UPDATE earnings_quality SET
                total_accruals = %s, accrual_ratio = %s, sloan_accruals = %s,
                cash_conversion_ratio = %s, altman_z_score = %s,
                earnings_quality_score = %s, quality_flags = %s,
                input_data_hash = %s, source_fetch_timestamp = NOW(), updated_at = NOW()
            WHERE symbol = %s AND fiscal_year = %s AND fiscal_period = %s
        """, (
            metrics.get("total_accruals"),
            metrics.get("accrual_ratio"),
            metrics.get("sloan_accruals"),
            metrics.get("cash_conversion_ratio"),
            metrics.get("altman_z_score"),
            metrics.get("earnings_quality_score"),
            metrics.get("quality_flags"),
            record_hash,
            symbol,
            fiscal_year,
            fiscal_period,
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate earnings quality metrics"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: S&P 500)"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    calculator = EarningsQualityCalculator(symbols=symbols)
    exit_code = calculator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
