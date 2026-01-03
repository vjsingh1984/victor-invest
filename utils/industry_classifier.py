#!/usr/bin/env python3
"""
Industry Classifier - Maps companies to industries using SIC codes and profiles

Provides granular industry classification within sectors for accurate:
- XBRL tag selection (industry-specific revenue/cost tags)
- Valuation method selection (DCF vs P/B vs FFO multiples)
- Metric interpretation (Combined Ratio for insurance, NIM for banks)

Author: InvestiGator Team
Date: 2025-11-10
"""

from typing import Optional, Dict, Tuple
import logging
import os
from pathlib import Path
from sqlalchemy import text

logger = logging.getLogger(__name__)


# SIC Code to Industry Mapping
# Format: 'SIC_CODE': ('Sector', 'Industry')
SIC_TO_INDUSTRY = {
    # === FINANCIALS ===

    # Banks - Commercial & Savings
    '6021': ('Financials', 'Banks'),  # National Commercial Banks
    '6022': ('Financials', 'Banks'),  # State Commercial Banks
    '6029': ('Financials', 'Banks'),  # Commercial Banks, NEC
    '6035': ('Financials', 'Banks'),  # Savings Institutions, Federally Chartered
    '6036': ('Financials', 'Banks'),  # Savings Institutions, Not Federally Chartered
    '6141': ('Financials', 'Banks'),  # Personal Credit Institutions

    # Insurance - Life & Health
    '6311': ('Financials', 'Insurance'),  # Life Insurance
    '6321': ('Financials', 'Insurance'),  # Accident & Health Insurance
    '6324': ('Financials', 'Insurance'),  # Hospital & Medical Service Plans

    # Insurance - Property & Casualty
    '6331': ('Financials', 'Insurance'),  # Fire, Marine & Casualty Insurance
    '6351': ('Financials', 'Insurance'),  # Surety Insurance
    '6361': ('Financials', 'Insurance'),  # Title Insurance
    '6371': ('Financials', 'Insurance'),  # Pension, Health & Welfare Funds
    '6399': ('Financials', 'Insurance'),  # Insurance Carriers, NEC

    # Investment Management & Securities
    '6211': ('Financials', 'Investment Management'),  # Security Brokers, Dealers & Flotation Companies
    '6282': ('Financials', 'Investment Management'),  # Investment Advice
    '6289': ('Financials', 'Investment Management'),  # Services Allied With the Exchange of Securities
    '6726': ('Financials', 'Investment Management'),  # Management Investment Offices, Open-End
    '6282': ('Financials', 'Investment Management'),  # Investment Advice

    # REITs
    '6798': ('Financials', 'REITs'),  # Real Estate Investment Trusts

    # === ENERGY ===

    # Oil & Gas - Exploration & Production
    '1311': ('Energy', 'Oil & Gas Exploration'),  # Crude Petroleum & Natural Gas
    '1381': ('Energy', 'Oil & Gas Services'),  # Drilling Oil & Gas Wells
    '1382': ('Energy', 'Oil & Gas Services'),  # Oil & Gas Field Exploration Services
    '1389': ('Energy', 'Oil & Gas Services'),  # Oil & Gas Field Services, NEC

    # Oil & Gas - Refining & Distribution
    '2911': ('Energy', 'Oil & Gas Refining'),  # Petroleum Refining
    '5172': ('Energy', 'Oil & Gas Refining'),  # Petroleum & Petroleum Products Wholesalers

    # === TECHNOLOGY ===

    # Software & IT Services
    '7370': ('Technology', 'Software'),  # Computer Programming, Data Processing, etc.
    '7371': ('Technology', 'Software'),  # Computer Programming Services
    '7372': ('Technology', 'Software'),  # Prepackaged Software
    '7373': ('Technology', 'Software'),  # Computer Integrated Systems Design
    '7374': ('Technology', 'Software'),  # Computer Processing & Data Preparation
    '7379': ('Technology', 'Software'),  # Computer Related Services, NEC

    # Semiconductors
    '3674': ('Technology', 'Semiconductors'),  # Semiconductors & Related Devices

    # Computer Hardware
    '3570': ('Technology', 'Hardware'),  # Computer & Office Equipment
    '3571': ('Technology', 'Hardware'),  # Electronic Computers
    '3572': ('Technology', 'Hardware'),  # Computer Storage Devices
    '3575': ('Technology', 'Hardware'),  # Computer Terminals
    '3577': ('Technology', 'Hardware'),  # Computer Peripheral Equipment, NEC

    # === HEALTHCARE ===

    # Pharmaceuticals
    '2834': ('Healthcare', 'Pharmaceuticals'),  # Pharmaceutical Preparations
    '2835': ('Healthcare', 'Pharmaceuticals'),  # In Vitro & In Vivo Diagnostic Substances
    '2836': ('Healthcare', 'Pharmaceuticals'),  # Biological Products (No Diagnostic Substances)

    # Biotechnology
    '2836': ('Healthcare', 'Biotechnology'),  # Biological Products
    '8731': ('Healthcare', 'Biotechnology'),  # Commercial Physical & Biological Research

    # Medical Devices
    '3841': ('Healthcare', 'Medical Devices'),  # Surgical & Medical Instruments & Apparatus
    '3842': ('Healthcare', 'Medical Devices'),  # Orthopedic, Prosthetic & Surgical Appliances
    '3845': ('Healthcare', 'Medical Devices'),  # Electromedical & Electrotherapeutic Apparatus

    # Healthcare Services
    '8000': ('Healthcare', 'Healthcare Services'),  # Services-Health Services
    '8062': ('Healthcare', 'Healthcare Services'),  # General Medical & Surgical Hospitals
    '8071': ('Healthcare', 'Healthcare Services'),  # Medical Laboratories

    # === REAL ESTATE ===
    '6500': ('Real Estate', 'Real Estate'),  # Real Estate
    '6510': ('Real Estate', 'Real Estate'),  # Real Estate Operators (No Developers) & Lessors
    '6512': ('Real Estate', 'Real Estate'),  # Operators of Nonresidential Buildings
    '6513': ('Real Estate', 'Real Estate'),  # Operators of Apartment Buildings
    '6519': ('Real Estate', 'Real Estate'),  # Lessors of Real Property, NEC

    # === CONSUMER ===

    # Retail
    '5311': ('Consumer', 'Retail'),  # Department Stores
    '5331': ('Consumer', 'Retail'),  # Variety Stores
    '5399': ('Consumer', 'Retail'),  # Miscellaneous General Merchandise Stores
    '5411': ('Consumer', 'Retail'),  # Grocery Stores
    '5600': ('Consumer', 'Retail'),  # Apparel & Accessory Stores
    '5900': ('Consumer', 'Retail'),  # Miscellaneous Retail

    # E-commerce
    '5961': ('Consumer', 'E-commerce'),  # Catalog & Mail-Order Houses

    # Consumer Goods
    '2000': ('Consumer', 'Consumer Goods'),  # Food & Kindred Products
    '2080': ('Consumer', 'Consumer Goods'),  # Beverages
    '2100': ('Consumer', 'Consumer Goods'),  # Tobacco Products
}


# Known symbol overrides (for companies with misleading SIC codes)
SYMBOL_OVERRIDES = {
    'HIG': ('Financials', 'Insurance'),  # Hartford Insurance Group
    'AIG': ('Financials', 'Insurance'),  # American International Group
    'PRU': ('Financials', 'Insurance'),  # Prudential Financial
    'MET': ('Financials', 'Insurance'),  # MetLife
    'ALL': ('Financials', 'Insurance'),  # Allstate
    'TRV': ('Financials', 'Insurance'),  # Travelers
    'PGR': ('Financials', 'Insurance'),  # Progressive
    'CB': ('Financials', 'Insurance'),   # Chubb

    'JPM': ('Financials', 'Banks'),      # JPMorgan Chase
    'BAC': ('Financials', 'Banks'),      # Bank of America
    'WFC': ('Financials', 'Banks'),      # Wells Fargo
    'C': ('Financials', 'Banks'),        # Citigroup
    'USB': ('Financials', 'Banks'),      # US Bancorp
    'PNC': ('Financials', 'Banks'),      # PNC Financial
    'TFC': ('Financials', 'Banks'),      # Truist Financial

    'BLK': ('Financials', 'Investment Management'),  # BlackRock
    'SCHW': ('Financials', 'Investment Management'), # Charles Schwab
    'TROW': ('Financials', 'Investment Management'), # T. Rowe Price

    'SPG': ('Financials', 'REITs'),      # Simon Property Group
    'PLD': ('Financials', 'REITs'),      # Prologis
    'AMT': ('Financials', 'REITs'),      # American Tower
    'CCI': ('Financials', 'REITs'),      # Crown Castle
    'EQIX': ('Financials', 'REITs'),     # Equinix

    'AMZN': ('Consumer', 'E-commerce'),  # Amazon
    'EBAY': ('Consumer', 'E-commerce'),  # eBay
    'ETSY': ('Consumer', 'E-commerce'),  # Etsy

    'ZS': ('Technology', 'Security & Protection Services'),  # Zscaler - Cybersecurity
}


def _load_russell1000_overrides() -> Dict[str, Tuple[str, str]]:
    """
    Load Russell 1000 sector/industry classifications from generated file.

    Returns:
        Dict mapping ticker -> (sector, industry)
    """
    try:
        # Path to generated Russell 1000 overrides file
        project_root = Path(__file__).parent.parent
        overrides_file = project_root / 'resources' / 'russell1000_overrides.py'

        if not overrides_file.exists():
            logger.debug(f"Russell 1000 overrides file not found: {overrides_file}")
            return {}

        # Load the module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("russell1000_overrides", overrides_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'RUSSELL1000_OVERRIDES'):
                overrides = module.RUSSELL1000_OVERRIDES
                logger.debug(f"Loaded {len(overrides)} Russell 1000 classifications")
                return overrides

        logger.warning(f"RUSSELL1000_OVERRIDES not found in {overrides_file}")
        return {}

    except Exception as e:
        logger.warning(f"Failed to load Russell 1000 overrides: {e}")
        return {}


# Load Russell 1000 overrides (924 large-cap stocks with sector/industry)
RUSSELL1000_OVERRIDES = _load_russell1000_overrides()


class IndustryClassifier:
    """
    Classifies companies into sector and industry based on:
    1. Database lookup (sec_companyfacts_metadata table - PRIMARY SOURCE)
    2. Symbol overrides (for edge cases with misleading SIC codes)
    3. Russell 1000 overrides (924 large-cap stocks from stock database)
    4. SIC code mapping (fallback when database empty)
    5. Company profile industry field (if sector matches)
    6. Sector-only classification (last resort)
    """

    def __init__(self, db_engine=None):
        self.sic_map = SIC_TO_INDUSTRY
        self.symbol_overrides = SYMBOL_OVERRIDES
        self.russell1000_overrides = RUSSELL1000_OVERRIDES
        self.db_engine = db_engine

        # Lazy-load database engine if not provided
        if self.db_engine is None:
            try:
                from investigator.infrastructure.database.db import get_db_manager
                self.db_engine = get_db_manager().engine
                logger.debug("Loaded database engine for industry classification")
            except Exception as e:
                logger.warning(f"Could not load database engine: {e}. Industry classification will use SIC codes only.")
                self.db_engine = None

    def _query_database_industry(self, symbol: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Query sec_companyfacts_metadata for sector/industry/sic_code

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (sector, industry, sic_code) or (None, None, None) if not found
        """
        if not self.db_engine:
            return (None, None, None)

        try:
            with self.db_engine.connect() as conn:
                query = text(
                    "SELECT sector, industry, sic_code "
                    "FROM sec_companyfacts_metadata "
                    "WHERE symbol = :symbol"
                )
                result = conn.execute(query, {'symbol': symbol.upper()}).fetchone()

                if result:
                    sector, industry, sic = result
                    logger.debug(f"Database lookup for {symbol}: sector={sector}, industry={industry}, sic={sic}")
                    return (sector, industry, sic)

                return (None, None, None)

        except Exception as e:
            logger.warning(f"Database query failed for {symbol}: {e}")
            return (None, None, None)

    def classify(
        self,
        symbol: str,
        sic_code: Optional[str] = None,
        profile_sector: Optional[str] = None,
        profile_industry: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify company into (sector, industry)

        Args:
            symbol: Stock ticker symbol
            sic_code: Standard Industrial Classification code (e.g., '6331')
            profile_sector: Sector from company profile
            profile_industry: Industry from company profile

        Returns:
            Tuple of (sector, industry) or (None, None) if unable to classify

        Priority:
            1. Database lookup (sec_companyfacts_metadata table - PRIMARY SOURCE)
            2. Symbol overrides (for known companies with edge cases)
            3. Russell 1000 overrides (924 large-cap stocks from stock database)
            4. SIC code mapping (fallback when database empty)
            5. Profile industry (if sector matches known sector)
            6. Profile sector only (no industry)
        """
        # Priority 1: Database lookup (PRIMARY SOURCE per user requirement)
        db_sector, db_industry, db_sic = self._query_database_industry(symbol)
        if db_sector and db_industry:
            logger.debug(f"Using database industry for {symbol}: {db_sector}/{db_industry}")
            return (db_sector, db_industry)

        # If database has SIC code but no sector/industry, use it for SIC mapping below
        if db_sic and not sic_code:
            sic_code = db_sic
            logger.debug(f"Using SIC code from database for {symbol}: {sic_code}")

        # Priority 2: Symbol overrides (edge cases, known misclassifications)
        if symbol and symbol.upper() in self.symbol_overrides:
            sector, industry = self.symbol_overrides[symbol.upper()]
            logger.debug(f"Using symbol override for {symbol}: {sector}/{industry}")
            return (sector, industry)

        # Priority 3: Russell 1000 overrides (924 large-cap stocks from stock database)
        if symbol and symbol.upper() in self.russell1000_overrides:
            sector, industry = self.russell1000_overrides[symbol.upper()]
            logger.debug(f"Using Russell 1000 classification for {symbol}: {sector}/{industry}")
            return (sector, industry)

        # Priority 4: SIC code mapping (fallback when database empty)
        if sic_code:
            # Normalize SIC code (remove leading zeros, convert to string)
            sic_normalized = str(sic_code).lstrip('0').zfill(4)

            if sic_normalized in self.sic_map:
                sector, industry = self.sic_map[sic_normalized]
                logger.debug(f"Using SIC code {sic_normalized} for {symbol}: {sector}/{industry}")
                return (sector, industry)

            # Try truncated SIC codes (e.g., 6331 → 6300 → 6000)
            for prefix_len in [3, 2, 1]:
                sic_prefix = sic_normalized[:prefix_len] + '0' * (4 - prefix_len)
                if sic_prefix in self.sic_map:
                    sector, industry = self.sic_map[sic_prefix]
                    logger.debug(f"Using SIC prefix {sic_prefix} for {symbol}: {sector}/{industry}")
                    return (sector, industry)

        # Priority 5: Profile industry (if it's a recognized industry within the sector)
        if profile_sector and profile_industry:
            # Normalize industry name from profile to match our industry names
            normalized_industry = self._normalize_industry_name(profile_industry, profile_sector)
            if normalized_industry:
                logger.debug(f"Using profile industry for {symbol}: {profile_sector}/{normalized_industry}")
                return (profile_sector, normalized_industry)

        # Priority 6: Sector only (no industry specificity)
        if profile_sector:
            logger.debug(f"Using profile sector only for {symbol}: {profile_sector}/None")
            return (profile_sector, None)

        # Unable to classify
        logger.warning(f"Unable to classify {symbol} - no SIC code or profile data")
        return (None, None)

    def _normalize_industry_name(self, profile_industry: str, sector: str) -> Optional[str]:
        """
        Normalize industry name from company profile to match our taxonomy

        Args:
            profile_industry: Industry string from company profile
            sector: Sector classification

        Returns:
            Normalized industry name or None if no match
        """
        if not profile_industry:
            return None

        industry_lower = profile_industry.lower()

        # Financials industry mappings
        if sector == 'Financials':
            if any(term in industry_lower for term in ['bank', 'lending', 'credit']):
                return 'Banks'
            elif any(term in industry_lower for term in ['insurance', 'insurer', 'casualty', 'life insurance', 'property']):
                return 'Insurance'
            elif any(term in industry_lower for term in ['investment', 'asset management', 'wealth', 'broker']):
                return 'Investment Management'
            elif any(term in industry_lower for term in ['reit', 'real estate investment']):
                return 'REITs'

        # Technology industry mappings
        elif sector == 'Technology':
            if any(term in industry_lower for term in ['software', 'saas', 'application']):
                return 'Software'
            elif any(term in industry_lower for term in ['semiconductor', 'chip', 'integrated circuit']):
                return 'Semiconductors'
            elif any(term in industry_lower for term in ['hardware', 'computer', 'server']):
                return 'Hardware'

        # Healthcare industry mappings
        elif sector == 'Healthcare':
            if any(term in industry_lower for term in ['pharma', 'drug']):
                return 'Pharmaceuticals'
            elif any(term in industry_lower for term in ['biotech', 'biological']):
                return 'Biotechnology'
            elif any(term in industry_lower for term in ['medical device', 'equipment', 'instrument']):
                return 'Medical Devices'
            elif any(term in industry_lower for term in ['hospital', 'health service', 'clinic']):
                return 'Healthcare Services'

        # Energy industry mappings
        elif sector == 'Energy':
            if any(term in industry_lower for term in ['exploration', 'production', 'e&p']):
                return 'Oil & Gas Exploration'
            elif any(term in industry_lower for term in ['refining', 'refin', 'midstream']):
                return 'Oil & Gas Refining'
            elif any(term in industry_lower for term in ['service', 'drilling', 'equipment']):
                return 'Oil & Gas Services'

        # Consumer industry mappings
        elif sector == 'Consumer':
            if any(term in industry_lower for term in ['e-commerce', 'online retail', 'internet retail']):
                return 'E-commerce'
            elif any(term in industry_lower for term in ['retail', 'store', 'department']):
                return 'Retail'
            elif any(term in industry_lower for term in ['consumer goods', 'packaged', 'beverage', 'food']):
                return 'Consumer Goods'

        # No match found
        return None

    def get_supported_industries(self, sector: Optional[str] = None) -> Dict[str, list]:
        """
        Get list of supported industries, optionally filtered by sector

        Args:
            sector: Optional sector to filter industries

        Returns:
            Dict mapping sector to list of industries
        """
        industries_by_sector = {}

        for sic, (sec, ind) in self.sic_map.items():
            if sector and sec != sector:
                continue

            if sec not in industries_by_sector:
                industries_by_sector[sec] = set()
            industries_by_sector[sec].add(ind)

        # Convert sets to sorted lists
        return {sec: sorted(list(inds)) for sec, inds in industries_by_sector.items()}


# Singleton instance
_classifier = None


def get_industry_classifier() -> IndustryClassifier:
    """Get singleton IndustryClassifier instance"""
    global _classifier

    if _classifier is None:
        _classifier = IndustryClassifier()

    return _classifier


def classify_company(
    symbol: str,
    sic_code: Optional[str] = None,
    profile_sector: Optional[str] = None,
    profile_industry: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to classify a company

    Args:
        symbol: Stock ticker symbol
        sic_code: SIC code (e.g., '6331')
        profile_sector: Sector from profile
        profile_industry: Industry from profile

    Returns:
        Tuple of (sector, industry)
    """
    classifier = get_industry_classifier()
    return classifier.classify(symbol, sic_code, profile_sector, profile_industry)
