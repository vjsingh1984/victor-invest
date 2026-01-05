# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Sector Multiples Service - Config-driven sector multiple lookups.

Provides sector-specific valuation multiples from centralized config.
Replaces hardcoded _get_sector_*_multiple() functions in rl_backtest.py.

Features:
- Sector normalization (handles variant names)
- Industry-specific overrides
- Fallback to defaults

Example:
    service = SectorMultiplesService()

    # Get single multiple
    pe = service.get_pe("Technology")  # 28
    ps = service.get_ps("Healthcare")  # 4

    # Get all multiples for a sector
    multiples = service.get_multiples("Financials")
    # {'pe': 12, 'ps': 3, 'pb': 1.2, 'ev_ebitda': 8}

    # With industry override
    multiples = service.get_multiples("Technology", industry="Software - Application")
"""

import logging
from typing import Dict, Optional

from .valuation_config_service import ValuationConfigService

logger = logging.getLogger(__name__)


class SectorMultiplesService:
    """
    Service for retrieving sector-specific valuation multiples.

    Consolidates all sector multiple lookups that were previously
    hardcoded in rl_backtest.py.
    """

    # Sector normalization mapping - handles variant names
    SECTOR_ALIASES = {
        # Technology variants
        "technology": "Technology",
        "tech": "Technology",
        "information technology": "Technology",
        "it": "Technology",
        # Healthcare variants
        "healthcare": "Healthcare",
        "health care": "Healthcare",
        "health": "Healthcare",
        # Financials variants
        "financials": "Financials",
        "financial services": "Financials",
        "financial": "Financials",
        "finance": "Financials",
        # Consumer variants
        "consumer cyclical": "Consumer Cyclical",
        "consumer discretionary": "Consumer Cyclical",
        "consumer defensive": "Consumer Defensive",
        "consumer staples": "Consumer Defensive",
        # Energy
        "energy": "Energy",
        "oil & gas": "Energy",
        # Materials
        "materials": "Materials",
        "basic materials": "Materials",
        # Industrials
        "industrials": "Industrials",
        "industrial": "Industrials",
        # Real Estate
        "real estate": "Real Estate",
        "reits": "Real Estate",
        # Utilities
        "utilities": "Utilities",
        "utility": "Utilities",
        # Communication Services
        "communication services": "Communication Services",
        "communications": "Communication Services",
        "telecom": "Communication Services",
        "telecommunications": "Communication Services",
    }

    # Industry-specific P/E overrides (from pe_multiples.industry_overrides)
    INDUSTRY_PE_OVERRIDES = {
        "Software - Application": 35.0,
        "Software - Infrastructure": 32.0,
        "Semiconductors": 18.0,
        "Semiconductor Equipment": 18.0,
        "Semiconductors & Semiconductor Equipment": 18.0,
        "Computer Hardware": 25.0,
        "Information Technology Services": 22.0,
        "Biotechnology": 18.0,
        "Medical Devices": 25.0,
        "Drug Manufacturers - General": 18.0,
        "Pharmaceuticals": 14.0,
        "Banks - Regional": 11.0,
        "Asset Management": 15.0,
        "Insurance": 12.0,
        "Internet Retail": 30.0,
        "Specialty Retail": 18.0,
        "Restaurants": 22.0,
        "Auto Manufacturing": 8.0,
        "Automobile Manufacturers": 8.0,
        "Motor Vehicles": 8.0,
        "REITs": 24.0,
        "Electric Utilities": 20.0,
        "Oil & Gas E&P": 12.0,
        "Aerospace & Defense": 16.0,
        "Defense": 16.0,
        "Defense Contractors": 16.0,
    }

    def __init__(self, config_service: Optional[ValuationConfigService] = None):
        """
        Initialize SectorMultiplesService.

        Args:
            config_service: ValuationConfigService instance. If None, creates one.
        """
        self._config = config_service or ValuationConfigService()

    def normalize_sector(self, sector: str) -> str:
        """
        Normalize sector name to standard format.

        Args:
            sector: Sector name (any format)

        Returns:
            Normalized sector name
        """
        if not sector:
            return "Unknown"

        # Check direct match first
        if sector in self.SECTOR_ALIASES.values():
            return sector

        # Try case-insensitive lookup
        normalized = self.SECTOR_ALIASES.get(sector.lower())
        if normalized:
            return normalized

        # Return as-is if not found (will use Default in lookups)
        return sector

    def get_pe(self, sector: str, industry: Optional[str] = None) -> float:
        """
        Get P/E multiple for a sector.

        Args:
            sector: Sector name
            industry: Optional industry for more specific multiple

        Returns:
            P/E multiple
        """
        # Check industry override first
        if industry:
            # Try exact match
            if industry in self.INDUSTRY_PE_OVERRIDES:
                return self.INDUSTRY_PE_OVERRIDES[industry]
            # Try from config
            industry_override = self._config.get(f"pe_multiples.industry_overrides.{industry}")
            if industry_override is not None:
                return float(industry_override)

        # Fall back to sector
        normalized = self.normalize_sector(sector)
        return self._config.get_sector_pe_multiple(normalized)

    def get_ps(self, sector: str, industry: Optional[str] = None) -> float:
        """
        Get P/S multiple for a sector.

        Args:
            sector: Sector name
            industry: Optional industry (reserved for future industry overrides)

        Returns:
            P/S multiple
        """
        normalized = self.normalize_sector(sector)
        return self._config.get_sector_ps_multiple(normalized)

    def get_pb(self, sector: str, industry: Optional[str] = None) -> float:
        """
        Get P/B multiple for a sector.

        Args:
            sector: Sector name
            industry: Optional industry (reserved for future industry overrides)

        Returns:
            P/B multiple
        """
        normalized = self.normalize_sector(sector)
        return self._config.get_sector_pb_multiple(normalized)

    def get_ev_ebitda(self, sector: str, industry: Optional[str] = None) -> float:
        """
        Get EV/EBITDA multiple for a sector.

        Args:
            sector: Sector name
            industry: Optional industry (reserved for future industry overrides)

        Returns:
            EV/EBITDA multiple
        """
        normalized = self.normalize_sector(sector)
        return self._config.get_sector_ev_ebitda_multiple(normalized)

    def get_multiples(self, sector: str, industry: Optional[str] = None) -> Dict[str, float]:
        """
        Get all multiples for a sector.

        Args:
            sector: Sector name
            industry: Optional industry for more specific P/E multiple

        Returns:
            Dict with pe, ps, pb, ev_ebitda multiples
        """
        return {
            "pe": self.get_pe(sector, industry),
            "ps": self.get_ps(sector, industry),
            "pb": self.get_pb(sector, industry),
            "ev_ebitda": self.get_ev_ebitda(sector, industry),
        }

    def get_multiple(self, multiple_type: str, sector: str, industry: Optional[str] = None) -> float:
        """
        Get a specific multiple type for a sector.

        Args:
            multiple_type: One of 'pe', 'ps', 'pb', 'ev_ebitda'
            sector: Sector name
            industry: Optional industry

        Returns:
            The requested multiple

        Raises:
            ValueError: If multiple_type is not recognized
        """
        type_lower = multiple_type.lower()

        if type_lower == "pe":
            return self.get_pe(sector, industry)
        elif type_lower == "ps":
            return self.get_ps(sector, industry)
        elif type_lower == "pb":
            return self.get_pb(sector, industry)
        elif type_lower in ("ev_ebitda", "evebitda", "ev/ebitda"):
            return self.get_ev_ebitda(sector, industry)
        else:
            raise ValueError(f"Unknown multiple type: {multiple_type}. " f"Use one of: pe, ps, pb, ev_ebitda")


# Convenience functions for backward compatibility
def get_sector_pe_multiple(sector: str, industry: Optional[str] = None) -> float:
    """Get P/E multiple for a sector (convenience function)."""
    return SectorMultiplesService().get_pe(sector, industry)


def get_sector_ps_multiple(sector: str, industry: Optional[str] = None) -> float:
    """Get P/S multiple for a sector (convenience function)."""
    return SectorMultiplesService().get_ps(sector, industry)


def get_sector_pb_multiple(sector: str, industry: Optional[str] = None) -> float:
    """Get P/B multiple for a sector (convenience function)."""
    return SectorMultiplesService().get_pb(sector, industry)


def get_sector_ev_ebitda_multiple(sector: str, industry: Optional[str] = None) -> float:
    """Get EV/EBITDA multiple for a sector (convenience function)."""
    return SectorMultiplesService().get_ev_ebitda(sector, industry)
