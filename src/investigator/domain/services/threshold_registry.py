"""
Threshold Registry - Sector-aware thresholds for valuation metrics.

Provides:
1. Sector-specific P/E extremeness thresholds
2. Industry-specific overrides
3. P/E level classification (extreme/high/moderate/low)
4. Configurable threshold loading from YAML

Problem being solved:
- Hard-coded P/E thresholds (200x/100x/50x) don't work across sectors
- Technology stocks can have P/E 300x+ during growth phases
- Financial stocks rarely exceed P/E 30x
- Same threshold for all sectors causes misclassification

Solution:
- Sector-specific thresholds based on industry norms
- Industry-level overrides for finer granularity
- Dynamically loaded from config.yaml

Usage:
    from investigator.domain.services.threshold_registry import ThresholdRegistry

    registry = ThresholdRegistry()
    thresholds = registry.get_pe_thresholds('Technology', 'Software - Application')
    level = registry.classify_pe_level(150, 'Technology', 'Software - Application')
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PELevel(Enum):
    """P/E ratio classification levels."""

    EXTREME = "extreme"  # Very high/low, significant discount
    HIGH = "high"  # Above normal, moderate discount
    MODERATE = "moderate"  # Normal range, minimal impact
    LOW = "low"  # Below normal, may indicate value
    NEGATIVE = "negative"  # Negative P/E (losses)


@dataclass
class PEThresholds:
    """P/E thresholds for a sector/industry."""

    extreme_high: float
    high: float
    moderate: float
    low: float
    sector: str
    industry: Optional[str] = None

    def classify(self, pe_ratio: float) -> PELevel:
        """Classify a P/E ratio against these thresholds."""
        if pe_ratio < 0:
            return PELevel.NEGATIVE
        elif pe_ratio > self.extreme_high:
            return PELevel.EXTREME
        elif pe_ratio > self.high:
            return PELevel.HIGH
        elif pe_ratio > self.moderate:
            return PELevel.MODERATE
        else:
            return PELevel.LOW


class ThresholdRegistry:
    """
    Registry of sector-aware thresholds for valuation metrics.

    Provides P/E (and future metrics) thresholds that vary by sector
    and industry, enabling more accurate valuation adjustments.

    Default thresholds are based on historical sector averages:
    - Technology: Higher P/E acceptable due to growth expectations
    - Financials: Lower P/E due to capital-intensive nature
    - Utilities: Lower P/E due to stable, slow growth

    Example:
        registry = ThresholdRegistry()

        # Get thresholds
        thresholds = registry.get_pe_thresholds('Technology', 'Software - Application')
        print(f"Extreme P/E for SaaS: {thresholds.extreme_high}x")

        # Classify a P/E ratio
        level = registry.classify_pe_level(150, 'Technology', 'Software - Application')
        print(f"P/E 150x is: {level.value}")  # "moderate" for SaaS
    """

    # Default P/E thresholds by sector
    # Format: {sector: {'extreme': X, 'high': Y, 'moderate': Z, 'low': W}}
    DEFAULT_SECTOR_THRESHOLDS = {
        # Technology sector - higher thresholds due to growth expectations
        "Technology": {
            "extreme": 300,
            "high": 150,
            "moderate": 75,
            "low": 20,
        },
        "Information Technology": {
            "extreme": 300,
            "high": 150,
            "moderate": 75,
            "low": 20,
        },
        # Financials - lower thresholds due to capital requirements
        "Financials": {
            "extreme": 50,
            "high": 30,
            "moderate": 20,
            "low": 8,
        },
        "Financial Services": {
            "extreme": 50,
            "high": 30,
            "moderate": 20,
            "low": 8,
        },
        # Healthcare - moderate thresholds, biotech can be higher
        "Healthcare": {
            "extreme": 150,
            "high": 80,
            "moderate": 40,
            "low": 15,
        },
        "Health Care": {
            "extreme": 150,
            "high": 80,
            "moderate": 40,
            "low": 15,
        },
        # Consumer Discretionary - varies widely
        "Consumer Discretionary": {
            "extreme": 150,
            "high": 75,
            "moderate": 40,
            "low": 12,
        },
        "Consumer Cyclical": {
            "extreme": 150,
            "high": 75,
            "moderate": 40,
            "low": 12,
        },
        # Consumer Staples - stable, lower thresholds
        "Consumer Staples": {
            "extreme": 60,
            "high": 35,
            "moderate": 25,
            "low": 12,
        },
        "Consumer Defensive": {
            "extreme": 60,
            "high": 35,
            "moderate": 25,
            "low": 12,
        },
        # Industrials - moderate thresholds
        "Industrials": {
            "extreme": 100,
            "high": 50,
            "moderate": 30,
            "low": 12,
        },
        # Energy - cyclical, lower thresholds
        "Energy": {
            "extreme": 40,
            "high": 25,
            "moderate": 15,
            "low": 6,
        },
        # Utilities - stable, low growth
        "Utilities": {
            "extreme": 40,
            "high": 25,
            "moderate": 18,
            "low": 10,
        },
        # Real Estate - use P/FFO typically, but for P/E
        "Real Estate": {
            "extreme": 80,
            "high": 50,
            "moderate": 30,
            "low": 15,
        },
        # Materials - cyclical
        "Materials": {
            "extreme": 60,
            "high": 35,
            "moderate": 20,
            "low": 10,
        },
        "Basic Materials": {
            "extreme": 60,
            "high": 35,
            "moderate": 20,
            "low": 10,
        },
        # Communication Services - mixed tech/media
        "Communication Services": {
            "extreme": 150,
            "high": 80,
            "moderate": 40,
            "low": 15,
        },
    }

    # Industry-specific overrides (more granular than sector)
    DEFAULT_INDUSTRY_OVERRIDES = {
        # SaaS companies - very high P/E acceptable
        "Software - Application": {
            "extreme": 400,
            "high": 200,
            "moderate": 100,
            "low": 30,
        },
        "Software - Infrastructure": {
            "extreme": 350,
            "high": 175,
            "moderate": 90,
            "low": 25,
        },
        # Semiconductors - cyclical within tech
        "Semiconductors": {
            "extreme": 200,
            "high": 100,
            "moderate": 50,
            "low": 15,
        },
        "Semiconductor Equipment & Materials": {
            "extreme": 180,
            "high": 90,
            "moderate": 45,
            "low": 15,
        },
        # Banks - very low P/E typical
        "Banks - Regional": {
            "extreme": 30,
            "high": 18,
            "moderate": 12,
            "low": 6,
        },
        "Banks - Diversified": {
            "extreme": 35,
            "high": 20,
            "moderate": 14,
            "low": 7,
        },
        # Insurance - moderate
        "Insurance - Life": {
            "extreme": 40,
            "high": 25,
            "moderate": 15,
            "low": 8,
        },
        "Insurance - Property & Casualty": {
            "extreme": 35,
            "high": 22,
            "moderate": 14,
            "low": 8,
        },
        # Biotech - can be extreme during R&D phase
        "Biotechnology": {
            "extreme": 500,
            "high": 200,
            "moderate": 80,
            "low": 20,
        },
        # REITs - typically valued on P/FFO
        "REIT - Retail": {
            "extreme": 50,
            "high": 35,
            "moderate": 22,
            "low": 12,
        },
        "REIT - Residential": {
            "extreme": 60,
            "high": 40,
            "moderate": 25,
            "low": 15,
        },
        # Retail - moderate thresholds
        "Specialty Retail": {
            "extreme": 100,
            "high": 50,
            "moderate": 30,
            "low": 10,
        },
        "Internet Retail": {
            "extreme": 200,
            "high": 100,
            "moderate": 50,
            "low": 20,
        },
        # Auto - cyclical, low P/E (P0-3: Updated for 8x fallback)
        "Auto Manufacturers": {
            "extreme": 40,
            "high": 20,
            "moderate": 12,
            "low": 5,
        },
        "Auto Manufacturing": {
            "extreme": 40,
            "high": 20,
            "moderate": 12,
            "low": 5,
        },
        "Automobile Manufacturers": {
            "extreme": 40,
            "high": 20,
            "moderate": 12,
            "low": 5,
        },
        # Defense - stable backlog (P0-3: 16x fallback)
        "Aerospace & Defense": {
            "extreme": 60,
            "high": 35,
            "moderate": 20,
            "low": 12,
        },
        "Defense": {
            "extreme": 60,
            "high": 35,
            "moderate": 20,
            "low": 12,
        },
        # Pharmaceuticals - patent cliff risk (P0-3: 14x fallback)
        "Pharmaceuticals": {
            "extreme": 60,
            "high": 35,
            "moderate": 18,
            "low": 10,
        },
        "Drug Manufacturers - General": {
            "extreme": 60,
            "high": 35,
            "moderate": 18,
            "low": 10,
        },
        # Oil & Gas - very cyclical
        "Oil & Gas Integrated": {
            "extreme": 35,
            "high": 20,
            "moderate": 12,
            "low": 5,
        },
        "Oil & Gas E&P": {
            "extreme": 40,
            "high": 25,
            "moderate": 15,
            "low": 6,
        },
    }

    # Default thresholds when sector/industry not found
    DEFAULT_THRESHOLDS = {
        "extreme": 200,
        "high": 100,
        "moderate": 50,
        "low": 15,
    }

    def __init__(
        self,
        sector_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        industry_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize registry with optional custom thresholds.

        Args:
            sector_thresholds: Custom sector thresholds (overrides defaults)
            industry_overrides: Custom industry overrides (overrides defaults)
        """
        self.sector_thresholds = {**self.DEFAULT_SECTOR_THRESHOLDS}
        self.industry_overrides = {**self.DEFAULT_INDUSTRY_OVERRIDES}

        if sector_thresholds:
            self.sector_thresholds.update(sector_thresholds)
        if industry_overrides:
            self.industry_overrides.update(industry_overrides)

    def get_pe_thresholds(self, sector: Optional[str] = None, industry: Optional[str] = None) -> PEThresholds:
        """
        Get P/E thresholds for a sector/industry.

        Lookup order:
        1. Industry override (most specific)
        2. Sector threshold
        3. Default thresholds

        Args:
            sector: Company sector (e.g., 'Technology', 'Financials')
            industry: Company industry (e.g., 'Software - Application')

        Returns:
            PEThresholds object with extreme/high/moderate/low values
        """
        thresholds = None
        source_industry = None

        # Try industry override first (case-insensitive lookup)
        if industry:
            industry_match = self._find_key_case_insensitive(industry, self.industry_overrides)
            if industry_match:
                thresholds = self.industry_overrides[industry_match]
                source_industry = industry_match
                logger.debug(f"Using industry override for '{industry_match}'")

        # Fall back to sector (case-insensitive lookup)
        if thresholds is None and sector:
            sector_match = self._find_key_case_insensitive(sector, self.sector_thresholds)
            if sector_match:
                thresholds = self.sector_thresholds[sector_match]
                logger.debug(f"Using sector thresholds for '{sector_match}'")

        # Fall back to default
        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS
            logger.debug(f"Using default thresholds (sector='{sector}', industry='{industry}' not found)")

        return PEThresholds(
            extreme_high=thresholds["extreme"],
            high=thresholds["high"],
            moderate=thresholds["moderate"],
            low=thresholds["low"],
            sector=sector or "Unknown",
            industry=source_industry,
        )

    def _find_key_case_insensitive(self, key: str, dictionary: Dict[str, Any]) -> Optional[str]:
        """Find a key in dictionary case-insensitively."""
        key_lower = key.lower()
        for dict_key in dictionary:
            if dict_key.lower() == key_lower:
                return dict_key
        return None

    def classify_pe_level(
        self, pe_ratio: float, sector: Optional[str] = None, industry: Optional[str] = None
    ) -> PELevel:
        """
        Classify a P/E ratio against sector/industry thresholds.

        Args:
            pe_ratio: The P/E ratio to classify
            sector: Company sector
            industry: Company industry

        Returns:
            PELevel enum (EXTREME, HIGH, MODERATE, LOW, NEGATIVE)
        """
        thresholds = self.get_pe_thresholds(sector, industry)
        return thresholds.classify(pe_ratio)

    def get_pe_weight_adjustment(
        self, pe_ratio: float, sector: Optional[str] = None, industry: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Get weight adjustment multiplier based on P/E level.

        Higher P/E ratios reduce P/E model weight as it becomes
        less reliable for valuation.

        Args:
            pe_ratio: The P/E ratio
            sector: Company sector
            industry: Company industry

        Returns:
            Tuple of (weight_multiplier, reason)
        """
        level = self.classify_pe_level(pe_ratio, sector, industry)
        thresholds = self.get_pe_thresholds(sector, industry)

        adjustments = {
            PELevel.EXTREME: (0.50, f"Extreme P/E ({pe_ratio:.0f}x > {thresholds.extreme_high}x)"),
            PELevel.HIGH: (0.75, f"High P/E ({pe_ratio:.0f}x > {thresholds.high}x)"),
            PELevel.MODERATE: (1.00, f"Moderate P/E ({pe_ratio:.0f}x)"),
            PELevel.LOW: (1.10, f"Low P/E ({pe_ratio:.0f}x < {thresholds.moderate}x)"),
            PELevel.NEGATIVE: (0.30, f"Negative P/E ({pe_ratio:.0f}x) - pre-profit"),
        }

        return adjustments.get(level, (1.0, "Unknown P/E level"))


# Singleton instance
_registry: Optional[ThresholdRegistry] = None


def get_threshold_registry() -> ThresholdRegistry:
    """Get the singleton ThresholdRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ThresholdRegistry()
    return _registry
