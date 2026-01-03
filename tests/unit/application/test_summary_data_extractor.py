"""
Unit tests for SummaryDataExtractor.

Tests the SOLID-based data extraction framework for executive summaries,
including fallback chains, field normalization, and edge case handling.
"""

import pytest
from investigator.application.summary_data_extractor import (
    SummaryDataExtractor,
    ExtractionResult,
    ExtractionConfidence,
    PriceTargetExtractor,
    InvestmentGradeExtractor,
    CurrentPriceExtractor,
    KeyStrengthsExtractor,
    KeyRisksExtractor,
    InvestmentThesisExtractor,
    RecommendationExtractor,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

# Simulates complete agent output with proper structure
COMPLETE_ANALYSIS_RESULTS = {
    'symbol': 'AAPL',
    'timestamp': '2025-01-15T10:00:00Z',
    'agents': {
        'fundamental': {
            'valuation': {
                'fair_value': 185.50,
                'investment_grade': 'B+',
            },
            'ratios': {
                'current_price': 175.00,
            },
            'confidence': {
                'confidence_level': 'High',
            },
            'data_quality': {
                'data_quality_score': 85.5,
            },
            'multi_model_summary': {
                'blended_fair_value': 185.50,
                'blended_upside_pct': 6.0,
            },
        },
        'synthesis': {
            'synthesis': {
                'response': {
                    'recommendation_and_action_plan': {
                        'recommendation': 'BUY',
                        'time_horizon': '12-18 months',
                    },
                    'investment_thesis': {
                        'core_thesis': 'Strong product lineup with growing services revenue.',
                        'bull_case': [
                            'Growing services revenue with high margins',
                            'Strong brand loyalty and ecosystem lock-in',
                            'Consistent capital returns to shareholders',
                        ],
                    },
                    'risk_analysis': {
                        'primary_risks': [
                            'China supply chain concentration',
                            'Regulatory scrutiny in key markets',
                            'Slower iPhone upgrade cycles',
                        ],
                    },
                },
            },
        },
    },
}


# Simulates output where fair_value is used instead of price_target_12_month
ANALYSIS_WITH_FAIR_VALUE = {
    'symbol': 'STX',
    'agents': {
        'fundamental': {
            'valuation': {
                'fair_value': 95.00,  # Not price_target_12_month
            },
            'ratios': {
                'current_price': 85.00,
            },
            'multi_model_summary': {
                'blended_fair_value': 95.00,
            },
        },
    },
}


# Simulates minimal/broken output - the bug case
BROKEN_ANALYSIS_RESULTS = {
    'symbol': 'BROKEN',
    'agents': {
        'fundamental': {
            'valuation': {},  # Empty valuation
            'ratios': {},     # No price
        },
        'synthesis': {
            'synthesis': {
                'response': {},  # Empty response
            },
        },
    },
}


# Simulates bull_case as dict instead of list (edge case)
BULL_CASE_AS_DICT = {
    'symbol': 'TEST',
    'agents': {
        'synthesis': {
            'synthesis': {
                'response': {
                    'investment_thesis': {
                        'bull_case': {
                            'key_assumptions': [
                                'Revenue growth accelerates',
                                'Margin expansion continues',
                            ],
                        },
                    },
                },
            },
        },
    },
}


# Simulates investment_grade missing but upside available for calculation
NO_GRADE_WITH_UPSIDE = {
    'symbol': 'CALC',
    'agents': {
        'fundamental': {
            'valuation': {
                'fair_value': 150.00,
            },
            'ratios': {
                'current_price': 100.00,  # 50% upside
            },
            'multi_model_summary': {
                'blended_upside_pct': 50.0,
            },
        },
    },
}


# =============================================================================
# ExtractionResult Tests
# =============================================================================

class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_has_value_with_valid_string(self):
        result = ExtractionResult(
            value="BUY",
            confidence=ExtractionConfidence.HIGH,
            source_path="test.path"
        )
        assert result.has_value is True

    def test_has_value_with_none(self):
        result = ExtractionResult(
            value=None,
            confidence=ExtractionConfidence.NONE,
            source_path="test"
        )
        assert result.has_value is False

    def test_has_value_with_na_string(self):
        result = ExtractionResult(
            value="N/A",
            confidence=ExtractionConfidence.MEDIUM,
            source_path="test"
        )
        assert result.has_value is False

    def test_has_value_with_empty_list(self):
        result = ExtractionResult(
            value=[],
            confidence=ExtractionConfidence.HIGH,
            source_path="test"
        )
        assert result.has_value is False

    def test_has_value_with_valid_list(self):
        result = ExtractionResult(
            value=["item1", "item2"],
            confidence=ExtractionConfidence.HIGH,
            source_path="test"
        )
        assert result.has_value is True

    def test_not_found_factory(self):
        result = ExtractionResult.not_found("path1, path2")
        assert result.value is None
        assert result.confidence == ExtractionConfidence.NONE
        assert "not_found" in result.source_path
        assert result.has_value is False


# =============================================================================
# Individual Extractor Tests
# =============================================================================

class TestPriceTargetExtractor:
    """Tests for PriceTargetExtractor."""

    def test_extracts_from_fair_value(self):
        """Should find fair_value when price_target_12_month is missing."""
        extractor = PriceTargetExtractor()
        result = extractor.extract(ANALYSIS_WITH_FAIR_VALUE)

        assert result.has_value is True
        assert result.value == 95.00
        assert "fair_value" in result.source_path

    def test_extracts_from_blended_fair_value(self):
        """Should fallback to multi_model_summary.blended_fair_value."""
        data = {
            'agents': {
                'fundamental': {
                    'multi_model_summary': {
                        'blended_fair_value': 200.00,
                    },
                },
            },
        }
        extractor = PriceTargetExtractor()
        result = extractor.extract(data)

        assert result.has_value is True
        assert result.value == 200.00
        assert result.fallback_used is True  # Not first path

    def test_rejects_zero_price(self):
        """Should not accept zero as valid price target."""
        data = {
            'agents': {
                'fundamental': {
                    'valuation': {
                        'fair_value': 0,
                    },
                },
            },
        }
        extractor = PriceTargetExtractor()
        result = extractor.extract(data)

        assert result.has_value is False

    def test_not_found_when_missing(self):
        """Should return not_found when no price target exists."""
        extractor = PriceTargetExtractor()
        result = extractor.extract(BROKEN_ANALYSIS_RESULTS)

        assert result.has_value is False
        assert result.confidence == ExtractionConfidence.NONE


class TestInvestmentGradeExtractor:
    """Tests for InvestmentGradeExtractor."""

    def test_extracts_explicit_grade(self):
        """Should find explicit investment_grade."""
        extractor = InvestmentGradeExtractor()
        result = extractor.extract(COMPLETE_ANALYSIS_RESULTS)

        assert result.has_value is True
        assert result.value == 'B+'
        assert result.confidence == ExtractionConfidence.HIGH

    def test_calculates_grade_from_upside(self):
        """Should calculate grade when explicit grade is missing."""
        extractor = InvestmentGradeExtractor()
        result = extractor.extract(NO_GRADE_WITH_UPSIDE)

        assert result.has_value is True
        # 50% upside should give A+
        assert result.value == 'A+'
        assert result.confidence == ExtractionConfidence.LOW
        assert result.fallback_used is True
        assert "derived_from" in result.source_path

    def test_calculates_grade_from_prices(self):
        """Should calculate grade from price difference when upside_pct missing."""
        data = {
            'agents': {
                'fundamental': {
                    'valuation': {
                        'fair_value': 120.00,
                    },
                    'ratios': {
                        'current_price': 100.00,
                    },
                },
            },
        }
        extractor = InvestmentGradeExtractor()
        result = extractor.extract(data)

        assert result.has_value is True
        # 20% upside should give A
        assert result.value == 'A'


class TestCurrentPriceExtractor:
    """Tests for CurrentPriceExtractor."""

    def test_extracts_from_ratios(self):
        extractor = CurrentPriceExtractor()
        result = extractor.extract(COMPLETE_ANALYSIS_RESULTS)

        assert result.has_value is True
        assert result.value == 175.00

    def test_rejects_negative_price(self):
        data = {
            'agents': {
                'fundamental': {
                    'ratios': {
                        'current_price': -50.00,
                    },
                },
            },
        }
        extractor = CurrentPriceExtractor()
        result = extractor.extract(data)

        assert result.has_value is False


class TestKeyStrengthsExtractor:
    """Tests for KeyStrengthsExtractor."""

    def test_extracts_bull_case_list(self):
        """Should extract bull_case when it's a list."""
        extractor = KeyStrengthsExtractor()
        result = extractor.extract(COMPLETE_ANALYSIS_RESULTS)

        assert result.has_value is True
        assert len(result.value) == 3
        assert 'Growing services revenue' in result.value[0]

    def test_extracts_from_bull_case_dict(self):
        """Should handle bull_case as dict with key_assumptions."""
        extractor = KeyStrengthsExtractor()
        result = extractor.extract(BULL_CASE_AS_DICT)

        assert result.has_value is True
        assert len(result.value) == 2
        assert 'Revenue growth accelerates' in result.value[0]

    def test_limits_to_three_items(self):
        """Should return max 3 strengths."""
        data = {
            'agents': {
                'synthesis': {
                    'synthesis': {
                        'response': {
                            'investment_thesis': {
                                'bull_case': [
                                    'Strength 1',
                                    'Strength 2',
                                    'Strength 3',
                                    'Strength 4',
                                    'Strength 5',
                                ],
                            },
                        },
                    },
                },
            },
        }
        extractor = KeyStrengthsExtractor()
        result = extractor.extract(data)

        assert result.has_value is True
        assert len(result.value) == 3

    def test_empty_when_missing(self):
        extractor = KeyStrengthsExtractor()
        result = extractor.extract(BROKEN_ANALYSIS_RESULTS)

        assert result.has_value is False


class TestKeyRisksExtractor:
    """Tests for KeyRisksExtractor."""

    def test_extracts_primary_risks(self):
        extractor = KeyRisksExtractor()
        result = extractor.extract(COMPLETE_ANALYSIS_RESULTS)

        assert result.has_value is True
        assert len(result.value) == 3
        assert 'China supply chain' in result.value[0]


class TestRecommendationExtractor:
    """Tests for RecommendationExtractor."""

    def test_extracts_recommendation(self):
        extractor = RecommendationExtractor()
        result = extractor.extract(COMPLETE_ANALYSIS_RESULTS)

        assert result.has_value is True
        assert result.value == 'BUY'

    def test_uppercases_recommendation(self):
        data = {
            'agents': {
                'synthesis': {
                    'synthesis': {
                        'response': {
                            'recommendation_and_action_plan': {
                                'recommendation': 'hold',
                            },
                        },
                    },
                },
            },
        }
        extractor = RecommendationExtractor()
        result = extractor.extract(data)

        assert result.value == 'HOLD'


# =============================================================================
# SummaryDataExtractor Integration Tests
# =============================================================================

class TestSummaryDataExtractor:
    """Integration tests for SummaryDataExtractor."""

    def test_extracts_all_fields_from_complete_data(self):
        """Should extract all fields from well-formed data."""
        extractor = SummaryDataExtractor(COMPLETE_ANALYSIS_RESULTS)
        summary = extractor.extract_minimal_summary()

        assert summary['symbol'] == 'AAPL'

        # Valuation
        assert summary['valuation']['current_price'] == 175.00
        assert summary['valuation']['price_target_12m'] == 185.50
        assert summary['valuation']['investment_grade'] == 'B+'
        assert summary['valuation']['expected_return_pct'] is not None

        # Recommendation
        assert summary['recommendation']['action'] == 'BUY'
        assert summary['recommendation']['confidence'] == 'High'
        assert summary['recommendation']['time_horizon'] == '12-18 months'

        # Thesis
        assert len(summary['thesis']['key_strengths']) == 3
        assert len(summary['thesis']['key_risks']) == 3
        assert 'Strong product lineup' in summary['thesis']['investment_thesis']

        # Data quality
        assert summary['data_quality']['overall_score'] == 85.5

    def test_handles_fair_value_as_price_target(self):
        """Should map fair_value to price_target_12m."""
        extractor = SummaryDataExtractor(ANALYSIS_WITH_FAIR_VALUE)
        summary = extractor.extract_minimal_summary()

        assert summary['valuation']['price_target_12m'] == 95.00
        assert summary['valuation']['current_price'] == 85.00
        # Expected return: (95-85)/85 * 100 = 11.76%
        assert summary['valuation']['expected_return_pct'] == pytest.approx(11.76, rel=0.01)

    def test_calculates_investment_grade_when_missing(self):
        """Should calculate investment grade from upside when explicit grade missing."""
        extractor = SummaryDataExtractor(NO_GRADE_WITH_UPSIDE)
        summary = extractor.extract_minimal_summary()

        # 50% upside should yield A+
        assert summary['valuation']['investment_grade'] == 'A+'

    def test_handles_broken_data_gracefully(self):
        """Should not crash on broken/minimal data."""
        extractor = SummaryDataExtractor(BROKEN_ANALYSIS_RESULTS)
        summary = extractor.extract_minimal_summary()

        # Should have symbol
        assert summary['symbol'] == 'BROKEN'

        # Missing fields should have N/A or None
        assert summary['valuation']['price_target_12m'] is None
        assert summary['recommendation']['action'] == 'N/A'
        assert summary['thesis']['key_strengths'] == []

    def test_audit_trail_tracks_extractions(self):
        """Should track extraction attempts in audit trail."""
        extractor = SummaryDataExtractor(COMPLETE_ANALYSIS_RESULTS, enable_audit=True)
        summary = extractor.extract_minimal_summary()
        audit = extractor.get_audit()

        assert audit is not None

        # Check audit has entries
        audit_summary = audit.get_summary()
        assert 'price_target_12m' in audit_summary
        assert 'investment_grade' in audit_summary

        # Check audit details
        price_audit = audit_summary['price_target_12m']
        assert price_audit['found'] is True
        assert price_audit['confidence'] in ['high', 'medium']

    def test_audit_disabled(self):
        """Should work without audit trail."""
        extractor = SummaryDataExtractor(COMPLETE_ANALYSIS_RESULTS, enable_audit=False)
        summary = extractor.extract_minimal_summary()

        assert extractor.get_audit() is None
        assert summary['symbol'] == 'AAPL'

    def test_expected_return_calculation(self):
        """Should correctly calculate expected return percentage."""
        data = {
            'agents': {
                'fundamental': {
                    'valuation': {
                        'fair_value': 150.00,
                    },
                    'ratios': {
                        'current_price': 100.00,
                    },
                },
            },
        }
        extractor = SummaryDataExtractor(data)
        summary = extractor.extract_minimal_summary()

        # (150-100)/100 * 100 = 50%
        assert summary['valuation']['expected_return_pct'] == 50.0

    def test_handles_zero_current_price(self):
        """Should handle zero current price without division error."""
        data = {
            'agents': {
                'fundamental': {
                    'valuation': {
                        'fair_value': 100.00,
                    },
                    'ratios': {
                        'current_price': 0,
                    },
                },
            },
        }
        extractor = SummaryDataExtractor(data)
        summary = extractor.extract_minimal_summary()

        # Should not crash, expected_return should be None
        assert summary['valuation']['expected_return_pct'] is None


class TestExtractorExtensibility:
    """Tests for Open/Closed principle - adding new extractors."""

    def test_register_custom_extractor(self):
        """Should allow registering custom extractors."""
        from investigator.application.summary_data_extractor import BaseFieldExtractor

        class CustomMetricExtractor(BaseFieldExtractor):
            @property
            def field_name(self) -> str:
                return "custom_metric"

            def _get_paths(self):
                return [
                    ('agents', 'fundamental', 'custom', 'metric'),
                ]

        data = {
            'agents': {
                'fundamental': {
                    'custom': {
                        'metric': 42.0,
                    },
                },
            },
        }

        extractor = SummaryDataExtractor(data)
        extractor.register_extractor(CustomMetricExtractor())

        result = extractor.extract_field('custom_metric')
        assert result.has_value is True
        assert result.value == 42.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and unusual data formats."""

    def test_handles_string_valuation_response(self):
        """Should handle valuation.response being a string instead of dict."""
        data = {
            'agents': {
                'fundamental': {
                    'valuation': {
                        'response': "N/A",  # String instead of dict
                        'fair_value': 100.0,  # But fair_value is present
                    },
                },
            },
        }
        extractor = SummaryDataExtractor(data)
        summary = extractor.extract_minimal_summary()

        assert summary['valuation']['price_target_12m'] == 100.0

    def test_handles_nested_dicts_in_strengths(self):
        """Should extract text from dict items in strengths list."""
        data = {
            'agents': {
                'synthesis': {
                    'synthesis': {
                        'response': {
                            'investment_thesis': {
                                'bull_case': [
                                    {'description': 'Strong moat'},
                                    {'text': 'Growing market'},
                                    {'assumption': 'Margin expansion'},
                                ],
                            },
                        },
                    },
                },
            },
        }
        extractor = SummaryDataExtractor(data)
        summary = extractor.extract_minimal_summary()

        strengths = summary['thesis']['key_strengths']
        assert len(strengths) == 3
        assert 'Strong moat' in strengths
        assert 'Growing market' in strengths
        assert 'Margin expansion' in strengths

    def test_handles_without_agents_wrapper(self):
        """Should work without 'agents' wrapper (legacy format)."""
        data = {
            'symbol': 'LEGACY',
            'fundamental': {
                'valuation': {
                    'fair_value': 50.0,
                },
                'ratios': {
                    'current_price': 45.0,
                },
            },
        }
        extractor = SummaryDataExtractor(data)
        summary = extractor.extract_minimal_summary()

        assert summary['symbol'] == 'LEGACY'
        assert summary['valuation']['price_target_12m'] == 50.0
        assert summary['valuation']['current_price'] == 45.0
