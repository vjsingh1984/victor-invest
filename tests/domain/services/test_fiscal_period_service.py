"""
Unit tests for FiscalPeriodService.

This test module validates centralized fiscal period handling including:
- Period normalization (all 26 supported period variations)
- Period parsing (2024-Q1, 2024-FY, 2024)
- YTD detection based on qtrs field
- Fiscal year end detection from CompanyFacts
- Period sorting and formatting
- Q4 computation validation

Author: Claude Code (Phase 1 Architecture Redesign)
Date: 2025-11-12
"""

import pytest
from investigator.domain.services.fiscal_period_service import (
    FiscalPeriodService,
    FiscalPeriod,
    get_fiscal_period_service
)


class TestFiscalPeriodService:
    """Test suite for FiscalPeriodService"""

    @pytest.fixture
    def service(self):
        """Create a fresh FiscalPeriodService instance for each test"""
        return FiscalPeriodService()


    # =============================================================================
    # normalize_period() Tests
    # =============================================================================

    def test_normalize_period_standard_formats(self, service):
        """Test normalization of standard period formats"""
        assert service.normalize_period("Q1") == "Q1"
        assert service.normalize_period("Q2") == "Q2"
        assert service.normalize_period("Q3") == "Q3"
        assert service.normalize_period("Q4") == "Q4"
        assert service.normalize_period("FY") == "FY"

    def test_normalize_period_lowercase(self, service):
        """Test that lowercase inputs are normalized correctly"""
        assert service.normalize_period("q1") == "Q1"
        assert service.normalize_period("q2") == "Q2"
        assert service.normalize_period("q3") == "Q3"
        assert service.normalize_period("q4") == "Q4"
        assert service.normalize_period("fy") == "FY"

    def test_normalize_period_long_format(self, service):
        """Test normalization of long-form period names"""
        assert service.normalize_period("FIRST QUARTER") == "Q1"
        assert service.normalize_period("SECOND QUARTER") == "Q2"
        assert service.normalize_period("THIRD QUARTER") == "Q3"
        assert service.normalize_period("FOURTH QUARTER") == "Q4"
        assert service.normalize_period("FULL YEAR") == "FY"
        assert service.normalize_period("ANNUAL") == "FY"

    def test_normalize_period_numeric_format(self, service):
        """Test normalization of numeric period formats"""
        assert service.normalize_period("1Q") == "Q1"
        assert service.normalize_period("2Q") == "Q2"
        assert service.normalize_period("3Q") == "Q3"
        assert service.normalize_period("4Q") == "Q4"

    def test_normalize_period_ytd_suffix(self, service):
        """Test that YTD suffix is stripped and base period returned"""
        assert service.normalize_period("Q1-YTD") == "Q1"
        assert service.normalize_period("Q2-YTD") == "Q2"
        assert service.normalize_period("Q3-YTD") == "Q3"

    def test_normalize_period_sec_xbrl_format(self, service):
        """Test normalization of SEC XBRL standard forms"""
        assert service.normalize_period("QTR1") == "Q1"
        assert service.normalize_period("QTR2") == "Q2"
        assert service.normalize_period("QTR3") == "Q3"
        assert service.normalize_period("QTR4") == "Q4"

    def test_normalize_period_single_digit(self, service):
        """Test normalization of single-digit period indicators"""
        assert service.normalize_period("1") == "Q1"
        assert service.normalize_period("2") == "Q2"
        assert service.normalize_period("3") == "Q3"
        assert service.normalize_period("4") == "Q4"
        assert service.normalize_period("Y") == "FY"

    def test_normalize_period_whitespace(self, service):
        """Test that whitespace is stripped before normalization"""
        assert service.normalize_period("  Q1  ") == "Q1"
        assert service.normalize_period("\tQ2\t") == "Q2"
        assert service.normalize_period(" FIRST QUARTER ") == "Q1"

    def test_normalize_period_empty_string(self, service):
        """Test that empty string raises ValueError"""
        with pytest.raises(ValueError, match="Fiscal period cannot be empty"):
            service.normalize_period("")

    def test_normalize_period_unknown_format(self, service):
        """Test that unknown format raises ValueError"""
        with pytest.raises(ValueError, match="Unknown fiscal period format"):
            service.normalize_period("Q5")

        with pytest.raises(ValueError, match="Unknown fiscal period format"):
            service.normalize_period("INVALID")

    # =============================================================================
    # parse_period() Tests
    # =============================================================================

    def test_parse_period_standard_format(self, service):
        """Test parsing of standard YYYY-QN format"""
        result = service.parse_period("2024-Q1")
        assert result.fiscal_year == 2024
        assert result.period == "Q1"
        assert result.period_str == "2024-Q1"
        assert str(result) == "2024-Q1"

    def test_parse_period_all_quarters(self, service):
        """Test parsing all quarters"""
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            result = service.parse_period(f"2024-{quarter}")
            assert result.fiscal_year == 2024
            assert result.period == quarter
            assert result.period_str == f"2024-{quarter}"

    def test_parse_period_full_year(self, service):
        """Test parsing YYYY-FY format"""
        result = service.parse_period("2024-FY")
        assert result.fiscal_year == 2024
        assert result.period == "FY"
        assert result.period_str == "2024-FY"

    def test_parse_period_year_only(self, service):
        """Test parsing year-only format (defaults to FY)"""
        result = service.parse_period("2024")
        assert result.fiscal_year == 2024
        assert result.period == "FY"
        assert result.period_str == "2024"

    def test_parse_period_with_normalization(self, service):
        """Test that period part is normalized during parsing"""
        result = service.parse_period("2024-FIRST QUARTER")
        assert result.fiscal_year == 2024
        assert result.period == "Q1"

        result = service.parse_period("2024-1Q")
        assert result.fiscal_year == 2024
        assert result.period == "Q1"

    def test_parse_period_empty_string(self, service):
        """Test that empty string raises ValueError"""
        with pytest.raises(ValueError, match="Period string cannot be empty"):
            service.parse_period("")

    def test_parse_period_invalid_format(self, service):
        """Test that invalid format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid period string format"):
            service.parse_period("Q1-2024")  # Wrong order

        with pytest.raises(ValueError, match="Invalid period string format"):
            service.parse_period("not-a-period")

    def test_parse_period_invalid_year(self, service):
        """Test that non-numeric year raises ValueError"""
        with pytest.raises(ValueError, match="Invalid period string format"):
            service.parse_period("ABCD-Q1")

    # =============================================================================
    # is_ytd() Tests
    # =============================================================================

    def test_is_ytd_individual_quarter(self, service):
        """Test that qtrs=1 is NOT YTD (individual quarter)"""
        assert service.is_ytd(1) is False

    def test_is_ytd_cumulative_quarters(self, service):
        """Test that qtrs=2,3,4 are YTD (cumulative)"""
        assert service.is_ytd(2) is True  # YTD through Q2
        assert service.is_ytd(3) is True  # YTD through Q3
        assert service.is_ytd(4) is True  # Full year

    def test_is_ytd_invalid_type(self, service):
        """Test that non-int qtrs raises TypeError"""
        with pytest.raises(TypeError, match="qtrs must be int"):
            service.is_ytd("1")

        with pytest.raises(TypeError, match="qtrs must be int"):
            service.is_ytd(1.5)

        with pytest.raises(TypeError, match="qtrs must be int"):
            service.is_ytd(None)

    def test_is_ytd_out_of_range(self, service):
        """Test that qtrs outside 1-4 raises ValueError"""
        with pytest.raises(ValueError, match="qtrs must be 1-4"):
            service.is_ytd(0)

        with pytest.raises(ValueError, match="qtrs must be 1-4"):
            service.is_ytd(5)

        with pytest.raises(ValueError, match="qtrs must be 1-4"):
            service.is_ytd(-1)

    # =============================================================================
    # detect_fiscal_year_end() Tests
    # =============================================================================

    def test_detect_fiscal_year_end_calendar_year(self, service):
        """Test detection of calendar year end (12-31)"""
        company_facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                {
                                    'end': '2024-12-31',
                                    'val': 1000000,
                                    'fy': 2024,
                                    'fp': 'FY',
                                    'form': '10-K'
                                },
                                {
                                    'end': '2023-12-31',
                                    'val': 900000,
                                    'fy': 2023,
                                    'fp': 'FY',
                                    'form': '10-K'
                                }
                            ]
                        }
                    }
                }
            }
        }

        result = service.detect_fiscal_year_end(company_facts)
        assert result == "-12-31"

    def test_detect_fiscal_year_end_june(self, service):
        """Test detection of June fiscal year end (06-30)"""
        company_facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                {
                                    'end': '2024-06-30',
                                    'val': 1000000,
                                    'fy': 2024,
                                    'fp': 'FY',
                                    'form': '10-K'
                                },
                                {
                                    'end': '2023-06-30',
                                    'val': 900000,
                                    'fy': 2023,
                                    'fp': 'FY',
                                    'form': '10-K'
                                }
                            ]
                        }
                    }
                }
            }
        }

        result = service.detect_fiscal_year_end(company_facts)
        assert result == "-06-30"

    def test_detect_fiscal_year_end_september(self, service):
        """Test detection of September fiscal year end (09-30)"""
        company_facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                {
                                    'end': '2024-09-30',
                                    'val': 1000000,
                                    'fy': 2024,
                                    'fp': 'FY',
                                    'form': '10-K'
                                }
                            ]
                        }
                    }
                }
            }
        }

        result = service.detect_fiscal_year_end(company_facts)
        assert result == "-09-30"

    def test_detect_fiscal_year_end_most_common(self, service):
        """Test that most common fiscal year end is returned"""
        company_facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                # Three 12-31 endings
                                {'end': '2024-12-31', 'fy': 2024, 'fp': 'FY', 'form': '10-K'},
                                {'end': '2023-12-31', 'fy': 2023, 'fp': 'FY', 'form': '10-K'},
                                {'end': '2022-12-31', 'fy': 2022, 'fp': 'FY', 'form': '10-K'},
                                # One 06-30 ending (outlier)
                                {'end': '2021-06-30', 'fy': 2021, 'fp': 'FY', 'form': '10-K'},
                            ]
                        }
                    }
                }
            }
        }

        result = service.detect_fiscal_year_end(company_facts)
        assert result == "-12-31"  # Most common wins

    def test_detect_fiscal_year_end_ignores_quarterly(self, service):
        """Test that quarterly filings (10-Q) are ignored"""
        company_facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                # FY filings (10-K) - should be used
                                {'end': '2024-12-31', 'fy': 2024, 'fp': 'FY', 'form': '10-K'},
                                # Quarterly filings (10-Q) - should be ignored
                                {'end': '2024-09-30', 'fy': 2024, 'fp': 'Q3', 'form': '10-Q'},
                                {'end': '2024-06-30', 'fy': 2024, 'fp': 'Q2', 'form': '10-Q'},
                            ]
                        }
                    }
                }
            }
        }

        result = service.detect_fiscal_year_end(company_facts)
        assert result == "-12-31"  # Only 10-K used

    def test_detect_fiscal_year_end_missing_facts_key(self, service):
        """Test that missing 'facts' key raises ValueError"""
        with pytest.raises(ValueError, match="Invalid company facts data: missing 'facts' key"):
            service.detect_fiscal_year_end({})

        with pytest.raises(ValueError, match="Invalid company facts data: missing 'facts' key"):
            service.detect_fiscal_year_end(None)

    def test_detect_fiscal_year_end_no_fy_data(self, service):
        """Test that absence of FY data raises ValueError"""
        company_facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                # Only quarterly data, no 10-K
                                {'end': '2024-09-30', 'fy': 2024, 'fp': 'Q3', 'form': '10-Q'}
                            ]
                        }
                    }
                }
            }
        }

        with pytest.raises(ValueError, match="No fiscal year \\(10-K\\) data found"):
            service.detect_fiscal_year_end(company_facts)

    # =============================================================================
    # get_period_sort_key() Tests
    # =============================================================================

    def test_get_period_sort_key_chronological_order(self, service):
        """Test that sort keys produce correct chronological order"""
        assert service.get_period_sort_key("Q1") == 1
        assert service.get_period_sort_key("Q2") == 2
        assert service.get_period_sort_key("Q3") == 3
        assert service.get_period_sort_key("Q4") == 4
        assert service.get_period_sort_key("FY") == 5

    def test_get_period_sort_key_sorting(self, service):
        """Test that periods can be sorted chronologically using sort keys"""
        periods = ["FY", "Q2", "Q4", "Q1", "Q3"]
        sorted_periods = sorted(periods, key=service.get_period_sort_key)
        assert sorted_periods == ["Q1", "Q2", "Q3", "Q4", "FY"]

    def test_get_period_sort_key_with_normalization(self, service):
        """Test that sort key works with period variations"""
        assert service.get_period_sort_key("FIRST QUARTER") == 1
        assert service.get_period_sort_key("1Q") == 1
        assert service.get_period_sort_key("QTR1") == 1

    # =============================================================================
    # format_period() Tests
    # =============================================================================

    def test_format_period_standard(self, service):
        """Test standard period formatting"""
        assert service.format_period(2024, "Q1") == "2024-Q1"
        assert service.format_period(2024, "Q2") == "2024-Q2"
        assert service.format_period(2024, "Q3") == "2024-Q3"
        assert service.format_period(2024, "Q4") == "2024-Q4"
        assert service.format_period(2024, "FY") == "2024-FY"

    def test_format_period_with_normalization(self, service):
        """Test that period is normalized before formatting"""
        assert service.format_period(2024, "FIRST QUARTER") == "2024-Q1"
        assert service.format_period(2024, "1Q") == "2024-Q1"
        assert service.format_period(2024, "QTR1") == "2024-Q1"

    def test_format_period_different_years(self, service):
        """Test formatting with different fiscal years"""
        assert service.format_period(2023, "Q1") == "2023-Q1"
        assert service.format_period(2025, "Q4") == "2025-Q4"

    # =============================================================================
    # validate_q4_computation_inputs() Tests
    # =============================================================================

    def test_validate_q4_computation_valid_inputs(self, service):
        """Test validation passes for valid Q4 computation inputs"""
        # All individual quarters (qtrs=1) and FY=4
        result = service.validate_q4_computation_inputs(
            fy_qtrs=4,
            q1_qtrs=1,
            q2_qtrs=1,
            q3_qtrs=1
        )
        assert result is True

    def test_validate_q4_computation_ytd_data(self, service):
        """Test validation fails when quarters are YTD (qtrs>=2)"""
        # Q2 is YTD (qtrs=2)
        result = service.validate_q4_computation_inputs(
            fy_qtrs=4,
            q1_qtrs=1,
            q2_qtrs=2,  # YTD
            q3_qtrs=1
        )
        assert result is False

        # Q3 is YTD (qtrs=3)
        result = service.validate_q4_computation_inputs(
            fy_qtrs=4,
            q1_qtrs=1,
            q2_qtrs=1,
            q3_qtrs=3  # YTD
        )
        assert result is False

    def test_validate_q4_computation_invalid_fy(self, service):
        """Test validation fails when FY qtrs != 4"""
        result = service.validate_q4_computation_inputs(
            fy_qtrs=3,  # Should be 4
            q1_qtrs=1,
            q2_qtrs=1,
            q3_qtrs=1
        )
        assert result is False

    def test_validate_q4_computation_invalid_types(self, service):
        """Test validation raises TypeError for non-int inputs"""
        with pytest.raises(TypeError, match="fy_qtrs must be int"):
            service.validate_q4_computation_inputs(
                fy_qtrs="4",  # String instead of int
                q1_qtrs=1,
                q2_qtrs=1,
                q3_qtrs=1
            )

        with pytest.raises(TypeError, match="q1_qtrs must be int"):
            service.validate_q4_computation_inputs(
                fy_qtrs=4,
                q1_qtrs=1.5,  # Float instead of int
                q2_qtrs=1,
                q3_qtrs=1
            )

    def test_validate_q4_computation_out_of_range(self, service):
        """Test validation raises ValueError for qtrs outside 1-4"""
        with pytest.raises(ValueError, match="fy_qtrs must be 1-4"):
            service.validate_q4_computation_inputs(
                fy_qtrs=5,  # Out of range
                q1_qtrs=1,
                q2_qtrs=1,
                q3_qtrs=1
            )

        with pytest.raises(ValueError, match="q3_qtrs must be 1-4"):
            service.validate_q4_computation_inputs(
                fy_qtrs=4,
                q1_qtrs=1,
                q2_qtrs=1,
                q3_qtrs=0  # Out of range
            )

    # =============================================================================
    # Singleton Tests
    # =============================================================================

    def test_get_fiscal_period_service_singleton(self):
        """Test that get_fiscal_period_service returns singleton instance"""
        service1 = get_fiscal_period_service()
        service2 = get_fiscal_period_service()

        assert service1 is service2  # Same instance
        assert isinstance(service1, FiscalPeriodService)


    # =============================================================================
    # Integration Tests
    # =============================================================================

    def test_end_to_end_parse_and_format(self, service):
        """Test parsing and formatting round-trip"""
        original = "2024-Q1"
        parsed = service.parse_period(original)
        formatted = service.format_period(parsed.fiscal_year, parsed.period)
        assert formatted == original

    def test_end_to_end_normalize_and_sort(self, service):
        """Test normalizing various formats and sorting"""
        periods = ["FIRST QUARTER", "4Q", "Q2-YTD", "FY", "QTR3"]
        normalized = [service.normalize_period(p) for p in periods]
        sorted_periods = sorted(normalized, key=service.get_period_sort_key)

        assert sorted_periods == ["Q1", "Q2", "Q3", "Q4", "FY"]

    def test_end_to_end_ytd_detection_and_q4_validation(self, service):
        """Test YTD detection integrated with Q4 validation"""
        # Individual quarters - Q4 computation valid
        assert service.is_ytd(1) is False
        assert service.validate_q4_computation_inputs(4, 1, 1, 1) is True

        # YTD quarters - Q4 computation invalid
        assert service.is_ytd(2) is True
        assert service.validate_q4_computation_inputs(4, 1, 2, 1) is False
