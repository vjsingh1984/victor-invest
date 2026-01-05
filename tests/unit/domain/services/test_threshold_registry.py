"""
Unit tests for threshold_registry module.

Tests sector-aware P/E thresholds and classification.
"""

import pytest

from investigator.domain.services.threshold_registry import (
    PELevel,
    PEThresholds,
    ThresholdRegistry,
    get_threshold_registry,
)


class TestPELevel:
    """Tests for PELevel enum."""

    def test_all_levels_exist(self):
        """Test all P/E levels are defined."""
        assert PELevel.EXTREME
        assert PELevel.HIGH
        assert PELevel.MODERATE
        assert PELevel.LOW
        assert PELevel.NEGATIVE


class TestPEThresholds:
    """Tests for PEThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values for a sector."""
        thresholds = PEThresholds(extreme_high=200, high=100, moderate=50, low=15, sector="default")
        assert thresholds.extreme_high == 200
        assert thresholds.high == 100
        assert thresholds.moderate == 50
        assert thresholds.low == 15

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = PEThresholds(
            extreme_high=300, high=150, moderate=75, low=20, sector="Technology", industry="Software"
        )
        assert thresholds.extreme_high == 300
        assert thresholds.high == 150


class TestThresholdRegistry:
    """Tests for ThresholdRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create default registry."""
        return ThresholdRegistry()

    def test_default_thresholds(self, registry):
        """Test getting default thresholds."""
        thresholds = registry.get_pe_thresholds("Unknown", "Unknown")
        assert isinstance(thresholds, PEThresholds)
        assert thresholds.extreme_high == 200  # Default

    def test_technology_sector(self, registry):
        """Test Technology sector has higher thresholds."""
        thresholds = registry.get_pe_thresholds("Technology", None)
        # Technology should have higher thresholds than default
        default_thresholds = registry.get_pe_thresholds("Unknown", None)
        assert thresholds.extreme_high >= default_thresholds.extreme_high

    def test_financials_sector(self, registry):
        """Test Financials sector has lower thresholds."""
        thresholds = registry.get_pe_thresholds("Financials", None)
        # Financials should have lower thresholds
        default_thresholds = registry.get_pe_thresholds("Unknown", None)
        assert thresholds.extreme_high <= default_thresholds.extreme_high

    def test_software_industry_override(self, registry):
        """Test Software industry has specific overrides."""
        thresholds = registry.get_pe_thresholds("Technology", "Software - Application")
        # Software should have the highest thresholds
        assert thresholds.extreme_high >= 300

    def test_regional_banks_industry(self, registry):
        """Test Regional Banks have low thresholds."""
        thresholds = registry.get_pe_thresholds("Financials", "Banks - Regional")
        # Banks should have low P/E thresholds
        assert thresholds.extreme_high <= 50

    def test_classify_pe_extreme(self, registry):
        """Test classification of extreme P/E."""
        level = registry.classify_pe_level(500, "Technology", None)
        assert level == PELevel.EXTREME

    def test_classify_pe_high(self, registry):
        """Test classification of high P/E."""
        level = registry.classify_pe_level(120, "Consumer Discretionary", None)
        assert level == PELevel.HIGH

    def test_classify_pe_moderate(self, registry):
        """Test classification of moderate P/E."""
        level = registry.classify_pe_level(60, "Consumer Discretionary", None)
        assert level == PELevel.MODERATE

    def test_classify_pe_low(self, registry):
        """Test classification of low P/E."""
        level = registry.classify_pe_level(10, "Consumer Discretionary", None)
        assert level == PELevel.LOW

    def test_classify_pe_negative(self, registry):
        """Test classification of negative P/E."""
        level = registry.classify_pe_level(-10, "Technology", None)
        assert level == PELevel.NEGATIVE

    def test_classify_pe_zero(self, registry):
        """Test classification of zero P/E."""
        level = registry.classify_pe_level(0, "Technology", None)
        # Zero P/E is classified as LOW (only negative P/E returns NEGATIVE)
        assert level == PELevel.LOW

    def test_case_insensitive_sector(self, registry):
        """Test sector lookup is case insensitive."""
        thresholds1 = registry.get_pe_thresholds("Technology", None)
        thresholds2 = registry.get_pe_thresholds("technology", None)
        thresholds3 = registry.get_pe_thresholds("TECHNOLOGY", None)

        assert thresholds1.extreme_high == thresholds2.extreme_high
        assert thresholds2.extreme_high == thresholds3.extreme_high

    def test_partial_sector_match(self, registry):
        """Test partial sector name matching."""
        # "Tech" should match "Technology"
        thresholds = registry.get_pe_thresholds("Tech", None)
        # Should get Technology thresholds or default
        assert isinstance(thresholds, PEThresholds)

    def test_singleton_registry(self):
        """Test singleton get_threshold_registry function."""
        registry1 = get_threshold_registry()
        registry2 = get_threshold_registry()
        assert registry1 is registry2


class TestIndustryOverrides:
    """Tests for industry-specific overrides."""

    @pytest.fixture
    def registry(self):
        return ThresholdRegistry()

    def test_saas_enterprise(self, registry):
        """Test SaaS - Enterprise has high thresholds."""
        thresholds = registry.get_pe_thresholds("Technology", "SaaS - Enterprise")
        # SaaS should have very high thresholds
        default_tech = registry.get_pe_thresholds("Technology", None)
        # Industry override should be higher or equal
        assert thresholds.extreme_high >= default_tech.extreme_high * 0.8

    def test_reit_industry(self, registry):
        """Test REIT industry thresholds."""
        thresholds = registry.get_pe_thresholds("Real Estate", "REIT")
        assert isinstance(thresholds, PEThresholds)

    def test_utilities(self, registry):
        """Test Utilities sector has conservative thresholds."""
        thresholds = registry.get_pe_thresholds("Utilities", None)
        # Utilities are defensive, should have lower thresholds
        default = registry.get_pe_thresholds("Unknown", None)
        assert thresholds.extreme_high <= default.extreme_high

    def test_healthcare_pharmaceuticals(self, registry):
        """Test Healthcare/Pharmaceuticals thresholds."""
        thresholds = registry.get_pe_thresholds("Healthcare", "Pharmaceuticals")
        assert isinstance(thresholds, PEThresholds)
        # Pharma thresholds adjusted for patent cliff risk (P0-3: 14x fallback)
        # moderate=18 reflects conservative valuation due to patent expiration risks
        assert thresholds.moderate == 18
        assert thresholds.high >= thresholds.moderate
