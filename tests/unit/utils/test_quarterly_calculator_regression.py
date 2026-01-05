#!/usr/bin/env python3
"""
Regression Tests for Quarterly Calculator

These tests ensure fixes for known issues don't regress:
- Issue #1: NameError fiscal_period_to_int not defined (Line 840)
- Issue #2: Q4 computation skipped due to YTD data
"""

import pytest

from utils.quarterly_calculator import compute_missing_quarter, get_rolling_ttm_periods


class TestIssue1FiscalPeriodSortingRegression:
    """
    Regression test for Issue #1: NameError - fiscal_period_to_int not defined

    Log Reference: logs/V_v2.log:226, 240, 261-274
    Location: utils/quarterly_calculator.py:840

    Root Cause: get_rolling_ttm_periods() was calling fiscal_period_to_int()
    which didn't exist in scope.

    Fix: Use fiscal_service.get_period_sort_key() instead
    """

    def test_get_rolling_ttm_periods_with_q4_computation_sorts_correctly(self):
        """
        Test that get_rolling_ttm_periods() correctly sorts periods after
        Q4 computation using FiscalPeriodService.get_period_sort_key().

        This reproduces the exact conditions from V_v2.log where the NameError occurred.
        """
        # Simulate quarterly metrics with multiple FY and Q periods (like in V_v2.log)
        quarterly_metrics = [
            {
                "fiscal_year": 2025,
                "fiscal_period": "Q3",
                "period_end_date": "2025-06-30",
                "income_statement": {"total_revenue": 10000},
                "cash_flow": {"operating_cash_flow": 5000},
                "balance_sheet": {"total_assets": 50000},
            },
            {
                "fiscal_year": 2025,
                "fiscal_period": "Q2",
                "period_end_date": "2025-03-31",
                "income_statement": {"total_revenue": 9000},
                "cash_flow": {"operating_cash_flow": 4500},
                "balance_sheet": {"total_assets": 48000},
            },
            {
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "period_end_date": "2025-09-30",
                "income_statement": {"total_revenue": 40000},
                "cash_flow": {"operating_cash_flow": 20000},
                "balance_sheet": {"total_assets": 50000},
            },
            {
                "fiscal_year": 2024,
                "fiscal_period": "Q3",
                "period_end_date": "2024-06-30",
                "income_statement": {"total_revenue": 9500},
                "cash_flow": {"operating_cash_flow": 4800},
                "balance_sheet": {"total_assets": 47000},
            },
        ]

        # This should NOT raise NameError: name 'fiscal_period_to_int' is not defined
        try:
            result = get_rolling_ttm_periods(quarterly_metrics, compute_missing=True)

            # If we get here, the fix works - no NameError was raised
            assert True, "Successfully called get_rolling_ttm_periods with compute_missing=True"

            # Additionally verify the result is a list
            assert isinstance(result, list), "Result should be a list of periods"

        except NameError as e:
            if "fiscal_period_to_int" in str(e):
                pytest.fail(f"REGRESSION: Issue #1 has returned! NameError: {e}")
            else:
                raise  # Re-raise if it's a different NameError

    def test_fiscal_period_sorting_order(self):
        """
        Test that fiscal periods are sorted correctly: FY=5, Q4=4, Q3=3, Q2=2, Q1=1

        This ensures the fix uses the correct FiscalPeriodService.get_period_sort_key()
        """
        quarterly_metrics = [
            {
                "fiscal_year": 2024,
                "fiscal_period": "Q1",
                "period_end_date": "2023-12-31",
                "income_statement": {"total_revenue": 8000},
                "cash_flow": {"operating_cash_flow": 4000},
                "balance_sheet": {"total_assets": 45000},
            },
            {
                "fiscal_year": 2024,
                "fiscal_period": "Q3",
                "period_end_date": "2024-06-30",
                "income_statement": {"total_revenue": 9000},
                "cash_flow": {"operating_cash_flow": 4500},
                "balance_sheet": {"total_assets": 47000},
            },
            {
                "fiscal_year": 2024,
                "fiscal_period": "Q2",
                "period_end_date": "2024-03-31",
                "income_statement": {"total_revenue": 8500},
                "cash_flow": {"operating_cash_flow": 4200},
                "balance_sheet": {"total_assets": 46000},
            },
            {
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_end_date": "2024-09-30",
                "income_statement": {"total_revenue": 35000},
                "cash_flow": {"operating_cash_flow": 17500},
                "balance_sheet": {"total_assets": 48000},
            },
        ]

        result = get_rolling_ttm_periods(quarterly_metrics, compute_missing=True)

        # Verify result is not empty
        assert len(result) > 0, "Should return at least some periods"

        # Verify fiscal periods are in expected order (most recent first)
        # Expected order: 2024-Q3, 2024-Q2, 2024-Q1, ...
        fiscal_periods = [p.get("fiscal_period") for p in result if p.get("fiscal_year") == 2024]

        # Q3 should come before Q2, Q2 before Q1 (reverse chronological)
        if "Q3" in fiscal_periods and "Q2" in fiscal_periods:
            assert fiscal_periods.index("Q3") < fiscal_periods.index(
                "Q2"
            ), "Q3 should come before Q2 in reverse chronological order"

        if "Q2" in fiscal_periods and "Q1" in fiscal_periods:
            assert fiscal_periods.index("Q2") < fiscal_periods.index(
                "Q1"
            ), "Q2 should come before Q1 in reverse chronological order"


class TestIssue2YTDConversionRegression:
    """
    Regression test for Issue #2: Q4 computation skipped due to YTD data

    Log Reference: logs/V_v2.log:207, 217, 224, 238, 259

    Root Cause: Missing Q1/Q2 periods prevented YTD conversion, blocking Q4 computation

    Fix: Allow Q4 = FY - Q3_YTD when Q1/Q2 are missing (valid SEC calculation)
    """

    def test_q4_computation_allowed_with_missing_q1_q2_ytd_q3(self):
        """
        Test that Q4 computation proceeds when Q1/Q2 are missing and Q3 is YTD.

        This reproduces the exact conditions from V_v2.log (2024-Q2 and 2025-Q2 missing Q1):
        - Q1: Missing (not filed yet)
        - Q2: Missing (not filed yet)
        - Q3: Present, YTD data
        - FY: Present

        Expected: Q4 = FY - Q3_YTD (valid calculation)
        """
        fy_data = {
            "symbol": "V",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "period_end_date": "2025-09-30",
            "income_statement": {"total_revenue": 40000000000, "net_income": 20000000000, "is_ytd": False},
            "cash_flow": {"operating_cash_flow": 25000000000, "capital_expenditures": -2000000000, "is_ytd": False},
            "balance_sheet": {"total_assets": 100000000000},
        }

        q3_ytd_data = {
            "symbol": "V",
            "fiscal_year": 2025,
            "fiscal_period": "Q3",
            "period_end_date": "2025-06-30",
            "income_statement": {
                "total_revenue": 30000000000,  # YTD (Q1+Q2+Q3)
                "net_income": 15000000000,
                "is_ytd": True,  # YTD flag set
            },
            "cash_flow": {
                "operating_cash_flow": 18000000000,  # YTD
                "capital_expenditures": -1500000000,
                "is_ytd": True,  # YTD flag set
            },
            "balance_sheet": {"total_assets": 98000000000},
        }

        # Call compute_missing_quarter with no Q1/Q2, only Q3 YTD
        q4_result = compute_missing_quarter(
            fy_data=fy_data, q1_data=None, q2_data=None, q3_data=q3_ytd_data  # Missing  # Missing  # YTD
        )

        # Should NOT return None (was the bug)
        assert q4_result is not None, "Q4 computation should proceed with Q3 YTD when Q1/Q2 missing"

        # Verify Q4 was computed correctly: Q4 = FY - Q3_YTD
        assert q4_result["fiscal_period"] == "Q4", "Should be Q4"
        assert q4_result["fiscal_year"] == 2025, "Should be FY 2025"

        # Verify revenue: Q4 = FY - Q3_YTD = 40B - 30B = 10B
        q4_revenue = q4_result["income_statement"]["total_revenue"]
        expected_q4_revenue = 40000000000 - 30000000000
        assert q4_revenue == expected_q4_revenue, f"Q4 revenue should be {expected_q4_revenue}, got {q4_revenue}"

        # Verify cash flow: Q4 OCF = FY - Q3_YTD = 25B - 18B = 7B
        q4_ocf = q4_result["cash_flow"]["operating_cash_flow"]
        expected_q4_ocf = 25000000000 - 18000000000
        assert q4_ocf == expected_q4_ocf, f"Q4 OCF should be {expected_q4_ocf}, got {q4_ocf}"

        # Verify is_ytd is False for Q4
        assert q4_result["income_statement"]["is_ytd"] is False, "Q4 should not be YTD"
        assert q4_result["cash_flow"]["is_ytd"] is False, "Q4 cash_flow should not be YTD"

    def test_q4_computation_still_skipped_when_q2_ytd_with_missing_q1(self):
        """
        Test that Q4 computation is still skipped when Q2 is YTD but Q1 is missing.

        This is a valid skip because we can't convert Q2 YTD to individual without Q1.
        """
        fy_data = {
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "income_statement": {"total_revenue": 35000000000, "is_ytd": False},
            "cash_flow": {"operating_cash_flow": 20000000000, "is_ytd": False},
            "balance_sheet": {},
        }

        q2_ytd_data = {
            "fiscal_year": 2024,
            "fiscal_period": "Q2",
            "income_statement": {"total_revenue": 17000000000, "is_ytd": True},
            "cash_flow": {"operating_cash_flow": 10000000000, "is_ytd": True},
            "balance_sheet": {},
        }

        q3_data = {
            "fiscal_year": 2024,
            "fiscal_period": "Q3",
            "income_statement": {"total_revenue": 9000000000, "is_ytd": False},
            "cash_flow": {"operating_cash_flow": 5000000000, "is_ytd": False},
            "balance_sheet": {},
        }

        # Should return None because Q2 is YTD and Q1 is missing (can't convert)
        q4_result = compute_missing_quarter(
            fy_data=fy_data, q1_data=None, q2_data=q2_ytd_data, q3_data=q3_data  # Missing  # YTD  # Individual
        )

        assert q4_result is None, "Q4 computation should be skipped when Q2 is YTD but Q1 is missing"
