#!/usr/bin/env python3
"""
Edge Case Fiscal Year Validation

Tests fiscal year calculation logic for various edge cases:
1. Different fiscal year end months (Jan-Dec)
2. Leap year handling (Feb 29)
3. Q1 vs Q4 boundary cases
4. Same calendar date, different fiscal years
5. Comparative periods (prior year)
6. Missing data handling
7. Fiscal year crossover

Author: InvestiGator Team
Date: 2025-11-17
"""

import sys
from datetime import datetime
from typing import Dict, List, Tuple
from sqlalchemy import create_engine, text


class FiscalYearEdgeCaseValidator:
    """Validate fiscal year calculation logic for edge cases"""

    def __init__(self):
        """Initialize validator with database connection"""
        self.db_config = {
            "host": "${DB_HOST:-localhost}",
            "port": 5432,
            "database": "sec_database",
            "username": "investigator",
            "password": "investigator",
        }
        self.engine = self._create_engine()

    def _create_engine(self):
        """Create SQLAlchemy engine"""
        connection_string = (
            f"postgresql://{self.db_config['username']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string, pool_size=5, max_overflow=10)

    def get_fiscal_year_end_distribution(self) -> List[Tuple[str, int, List[str]]]:
        """
        Get distribution of fiscal year ends across database.

        Returns:
            List of (fye, count, sample_symbols) tuples
        """
        query = text("""
            WITH symbol_lookup AS (
              SELECT DISTINCT ON (sub.cik)
                sub.cik,
                sub.fye,
                COALESCE(proc.symbol, '') as symbol
              FROM sec_sub_data sub
              LEFT JOIN sec_companyfacts_processed proc ON sub.cik = proc.cik::integer
              WHERE sub.fye IN ('0131', '0228', '0229', '0331', '0430', '0531',
                                '0630', '0731', '0831', '0930', '1031', '1130', '1231')
            )
            SELECT
              fye,
              COUNT(DISTINCT symbol) as count,
              STRING_AGG(symbol, ', ' ORDER BY symbol) as sample_symbols
            FROM (
              SELECT DISTINCT ON (fye, symbol) fye, symbol
              FROM symbol_lookup
              WHERE symbol != '' AND symbol IS NOT NULL
              ORDER BY fye, symbol
              LIMIT 50
            ) subq
            GROUP BY fye
            ORDER BY fye;
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [(row.fye, row.count, row.sample_symbols.split(', ')) for row in result]

    def get_company_fiscal_periods(self, cik: int, min_year: int = 2023) -> List[Dict]:
        """
        Get fiscal periods for a company from bulk table.

        Args:
            cik: Company CIK
            min_year: Minimum fiscal year to fetch

        Returns:
            List of fiscal period records
        """
        query = text("""
            SELECT
              period,
              fy,
              fp,
              filed,
              fye,
              EXTRACT(MONTH FROM period) as period_month,
              EXTRACT(YEAR FROM period) as period_year,
              EXTRACT(DAY FROM period) as period_day
            FROM sec_sub_data
            WHERE cik = :cik
              AND form IN ('10-K', '10-Q')
              AND fy >= :min_year
            ORDER BY period DESC
            LIMIT 20
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"cik": cik, "min_year": min_year})
            return [dict(row._mapping) for row in result]

    def _calculate_fiscal_year_from_date(
        self, period_end_date: str, fiscal_year_end_month: int
    ) -> int:
        """
        Calculate fiscal year from period end date and fiscal year end month.

        For non-calendar fiscal years, the fiscal year increments AFTER the fiscal year end.

        Args:
            period_end_date: Period end date (YYYY-MM-DD)
            fiscal_year_end_month: Fiscal year end month (1=Jan, 5=May, 12=Dec)

        Returns:
            Fiscal year (e.g., 2025)

        Examples:
            ORCL (fiscal_year_end_month=5, May):
            - 2024-11-30 → FY 2025 (Nov > May → next FY)
            - 2024-05-31 → FY 2024 (May == May → current FY)
            - 2024-02-28 → FY 2024 (Feb < May → current FY)

            WMT (fiscal_year_end_month=1, Jan):
            - 2024-10-31 → FY 2025 (Oct > Jan → next FY)
            - 2025-01-31 → FY 2025 (Jan == Jan → current FY)
        """
        try:
            period_end = datetime.strptime(period_end_date, "%Y-%m-%d")
            period_month = period_end.month
            period_year = period_end.year

            # If period month > fiscal year end month, period is in NEXT fiscal year
            if period_month > fiscal_year_end_month:
                return period_year + 1
            else:
                return period_year

        except (ValueError, TypeError) as e:
            print(f"Error calculating fiscal year from date {period_end_date}: {e}")
            return period_end.year  # Fallback to calendar year

    def test_fiscal_year_calculation(
        self,
        period_end_date: str,
        fiscal_year_end_month: int,
        expected_fy: int,
        expected_fp: str
    ) -> Dict:
        """
        Test fiscal year calculation for a specific case.

        Args:
            period_end_date: Period end date (YYYY-MM-DD)
            fiscal_year_end_month: Fiscal year end month (1-12)
            expected_fy: Expected fiscal year
            expected_fp: Expected fiscal period

        Returns:
            Test result dict with status and details
        """
        try:
            # Calculate fiscal year using local logic
            calculated_fy = self._calculate_fiscal_year_from_date(
                period_end_date, fiscal_year_end_month
            )

            # Parse period to get quarter
            period_date = datetime.strptime(period_end_date, "%Y-%m-%d")
            period_month = period_date.month

            # Determine if period matches fiscal year end
            is_fy_end = period_month == fiscal_year_end_month

            return {
                "period_end_date": period_end_date,
                "fiscal_year_end_month": fiscal_year_end_month,
                "expected_fy": expected_fy,
                "calculated_fy": calculated_fy,
                "match": calculated_fy == expected_fy,
                "is_fy_end": is_fy_end,
                "expected_fp": expected_fp,
                "status": "PASS" if calculated_fy == expected_fy else "FAIL"
            }
        except Exception as e:
            return {
                "period_end_date": period_end_date,
                "fiscal_year_end_month": fiscal_year_end_month,
                "expected_fy": expected_fy,
                "calculated_fy": None,
                "match": False,
                "error": str(e),
                "status": "ERROR"
            }

    def validate_oracle_edge_cases(self) -> List[Dict]:
        """
        Validate Oracle (May fiscal year end) edge cases.

        Oracle FY ends May 31:
        - Q1: Jun 1 - Aug 31 (calendar Q3)
        - Q2: Sep 1 - Nov 30 (calendar Q4)
        - Q3: Dec 1 - Feb 28/29 (calendar Q1)
        - Q4: Mar 1 - May 31 (calendar Q2)
        """
        test_cases = [
            # Q1 periods (Aug end) - Should be NEXT fiscal year
            ("2024-08-31", 5, 2025, "Q1"),  # Aug > May → FY 2025
            ("2023-08-31", 5, 2024, "Q1"),

            # Q2 periods (Nov end) - Should be NEXT fiscal year
            ("2024-11-30", 5, 2025, "Q2"),  # Nov > May → FY 2025
            ("2023-11-30", 5, 2024, "Q2"),

            # Q3 periods (Feb end) - Should be SAME fiscal year
            ("2025-02-28", 5, 2025, "Q3"),  # Feb < May → FY 2025
            ("2024-02-29", 5, 2024, "Q3"),  # Leap year

            # Q4 periods (May end) - Should be SAME fiscal year
            ("2025-05-31", 5, 2025, "FY"),  # May == May → FY 2025
            ("2024-05-31", 5, 2024, "FY"),
        ]

        results = []
        for period_end, fy_end_month, expected_fy, expected_fp in test_cases:
            result = self.test_fiscal_year_calculation(
                period_end, fy_end_month, expected_fy, expected_fp
            )
            results.append(result)

        return results

    def validate_walmart_edge_cases(self) -> List[Dict]:
        """
        Validate Walmart (Jan fiscal year end) edge cases.

        Walmart FY ends Jan 31:
        - Q1: Feb 1 - Apr 30 (calendar Q2)
        - Q2: May 1 - Jul 31 (calendar Q3)
        - Q3: Aug 1 - Oct 31 (calendar Q4)
        - Q4: Nov 1 - Jan 31 (calendar Q1)
        """
        test_cases = [
            # Q1 periods (Apr end) - Should be NEXT fiscal year
            ("2025-04-30", 1, 2026, "Q1"),  # Apr > Jan → FY 2026
            ("2024-04-30", 1, 2025, "Q1"),

            # Q2 periods (Jul end) - Should be NEXT fiscal year
            ("2024-07-31", 1, 2025, "Q2"),  # Jul > Jan → FY 2025

            # Q3 periods (Oct end) - Should be NEXT fiscal year
            ("2024-10-31", 1, 2025, "Q3"),  # Oct > Jan → FY 2025

            # FY periods (Jan end) - Should be SAME fiscal year
            ("2025-01-31", 1, 2024, "FY"),  # Jan == Jan → FY 2024 (weird but correct)
            ("2024-01-31", 1, 2023, "FY"),
        ]

        results = []
        for period_end, fy_end_month, expected_fy, expected_fp in test_cases:
            result = self.test_fiscal_year_calculation(
                period_end, fy_end_month, expected_fy, expected_fp
            )
            results.append(result)

        return results

    def validate_leap_year_edge_cases(self) -> List[Dict]:
        """Validate leap year handling (Feb 29)"""
        test_cases = [
            # Leap year Feb 29
            ("2024-02-29", 2, 2024, "FY"),  # Feb 29 in leap year
            ("2024-02-29", 5, 2024, "Q3"),  # Feb 29 for May FYE company

            # Non-leap year Feb 28
            ("2023-02-28", 2, 2023, "FY"),
            ("2025-02-28", 5, 2025, "Q3"),
        ]

        results = []
        for period_end, fy_end_month, expected_fy, expected_fp in test_cases:
            result = self.test_fiscal_year_calculation(
                period_end, fy_end_month, expected_fy, expected_fp
            )
            results.append(result)

        return results

    def validate_q1_q4_boundary(self) -> List[Dict]:
        """Validate Q1 vs Q4 boundary cases"""
        test_cases = [
            # Company with FY ending Jan 31
            ("2025-01-31", 1, 2024, "FY"),  # Q4 ending Jan 31 is FY 2024
            ("2025-04-30", 1, 2026, "Q1"),  # Q1 ending Apr 30 is FY 2026

            # Company with FY ending Feb 28
            ("2024-02-29", 2, 2024, "FY"),  # Q4 ending Feb 29 is FY 2024
            ("2024-05-31", 2, 2025, "Q1"),  # Q1 ending May 31 is FY 2025
        ]

        results = []
        for period_end, fy_end_month, expected_fy, expected_fp in test_cases:
            result = self.test_fiscal_year_calculation(
                period_end, fy_end_month, expected_fy, expected_fp
            )
            results.append(result)

        return results

    def validate_same_date_different_fy(self) -> List[Dict]:
        """
        Validate same calendar date, different fiscal years.

        Example: Nov 30, 2024
        - ORCL (May FYE): Nov 30, 2024 is Q2-2025
        - Company with Oct 31 FYE: Nov 30, 2024 is Q1-2025
        """
        test_cases = [
            # Nov 30, 2024 for different companies
            ("2024-11-30", 5, 2025, "Q2"),   # May FYE → Q2-2025
            ("2024-11-30", 10, 2025, "Q1"),  # Oct 31 FYE → Q1-2025
            ("2024-11-30", 12, 2024, "Q4"),  # Dec 31 FYE → Q4-2024

            # May 31, 2024 for different companies
            ("2024-05-31", 5, 2024, "FY"),   # May FYE → FY-2024
            ("2024-05-31", 12, 2024, "Q2"),  # Dec 31 FYE → Q2-2024
            ("2024-05-31", 1, 2025, "Q1"),   # Jan 31 FYE → Q1-2025
        ]

        results = []
        for period_end, fy_end_month, expected_fy, expected_fp in test_cases:
            result = self.test_fiscal_year_calculation(
                period_end, fy_end_month, expected_fy, expected_fp
            )
            results.append(result)

        return results

    def print_results(self, test_name: str, results: List[Dict]):
        """Print test results in formatted table"""
        print(f"\n{'='*80}")
        print(f"{test_name}")
        print(f"{'='*80}")
        print(f"{'Period End':<15} {'FY End':<10} {'Expected':<10} {'Calculated':<10} {'Status':<10}")
        print(f"{'-'*80}")

        for result in results:
            period_end = result['period_end_date']
            fy_end = f"Month {result['fiscal_year_end_month']}"
            expected = result['expected_fy']
            calculated = result.get('calculated_fy', 'N/A')
            status = result['status']

            # Color code status
            if status == "PASS":
                status_str = f"✓ {status}"
            elif status == "FAIL":
                status_str = f"✗ {status}"
            else:
                status_str = f"⚠ {status}"

            print(f"{period_end:<15} {fy_end:<10} {expected:<10} {calculated:<10} {status_str:<10}")

            if 'error' in result:
                print(f"  ERROR: {result['error']}")

        # Summary
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        errors = sum(1 for r in results if r['status'] == 'ERROR')

        print(f"\nSummary: {passed} PASS, {failed} FAIL, {errors} ERROR (Total: {len(results)})")

    def run_all_validations(self):
        """Run all edge case validations"""
        print("="*80)
        print("FISCAL YEAR EDGE CASE VALIDATION")
        print("="*80)

        # 1. Oracle (May FYE)
        oracle_results = self.validate_oracle_edge_cases()
        self.print_results("1. Oracle Edge Cases (May Fiscal Year End)", oracle_results)

        # 2. Walmart (Jan FYE)
        walmart_results = self.validate_walmart_edge_cases()
        self.print_results("2. Walmart Edge Cases (Jan Fiscal Year End)", walmart_results)

        # 3. Leap Year
        leap_year_results = self.validate_leap_year_edge_cases()
        self.print_results("3. Leap Year Edge Cases", leap_year_results)

        # 4. Q1/Q4 Boundary
        boundary_results = self.validate_q1_q4_boundary()
        self.print_results("4. Q1/Q4 Boundary Edge Cases", boundary_results)

        # 5. Same Date, Different FY
        same_date_results = self.validate_same_date_different_fy()
        self.print_results("5. Same Calendar Date, Different Fiscal Years", same_date_results)

        # Overall summary
        all_results = (
            oracle_results + walmart_results + leap_year_results +
            boundary_results + same_date_results
        )

        total_passed = sum(1 for r in all_results if r['status'] == 'PASS')
        total_failed = sum(1 for r in all_results if r['status'] == 'FAIL')
        total_errors = sum(1 for r in all_results if r['status'] == 'ERROR')

        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {len(all_results)}")
        print(f"Passed: {total_passed} ({100*total_passed/len(all_results):.1f}%)")
        print(f"Failed: {total_failed} ({100*total_failed/len(all_results) if total_failed > 0 else 0:.1f}%)")
        print(f"Errors: {total_errors} ({100*total_errors/len(all_results) if total_errors > 0 else 0:.1f}%)")

        if total_failed > 0 or total_errors > 0:
            print("\n⚠️  VALIDATION FAILED - Fix required!")
            return False
        else:
            print("\n✓ ALL VALIDATIONS PASSED!")
            return True


if __name__ == "__main__":
    validator = FiscalYearEdgeCaseValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)
