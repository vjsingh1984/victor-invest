"""
REIT Property Type Valuation Model

Implements property-type-specific FFO multiple valuation for REITs:
- Detects REIT property type from company name, symbol, or industry classification
- Applies property-specific FFO multiples (industrial/data centers trade at premium to retail/office)
- Adjusts multiples based on interest rate environment (10-year Treasury yield)

Author: Claude Code
Date: 2025-12-29
Phase: P1-C (REIT Property Type Model)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class REITPropertyType(Enum):
    """REIT property type classifications with typical growth/risk profiles."""

    INDUSTRIAL_LOGISTICS = "industrial_logistics"
    DATA_CENTERS = "data_centers"
    CELL_TOWERS = "cell_towers"
    RESIDENTIAL_SUNBELT = "residential_sunbelt"
    RESIDENTIAL_COASTAL = "residential_coastal"
    RESIDENTIAL_GENERAL = "residential_general"  # When we can't determine geography
    HEALTHCARE = "healthcare"
    OFFICE_CLASS_A = "office_class_a"
    OFFICE_CLASS_B = "office_class_b"
    OFFICE_GENERAL = "office_general"  # When we can't determine class
    REGIONAL_MALLS = "regional_malls"
    STRIP_CENTERS = "strip_centers"
    RETAIL_GENERAL = "retail_general"
    NET_LEASE = "net_lease"
    SELF_STORAGE = "self_storage"
    SPECIALTY = "specialty"
    DIVERSIFIED = "diversified"
    UNKNOWN = "unknown"


# FFO Multiple ranges by property type
# Format: (low_multiple, high_multiple)
# Based on historical trading ranges and growth profiles
REIT_FFO_MULTIPLES: Dict[REITPropertyType, Tuple[float, float]] = {
    # Premium growth sectors (secular tailwinds)
    REITPropertyType.INDUSTRIAL_LOGISTICS: (22.0, 25.0),
    REITPropertyType.DATA_CENTERS: (20.0, 24.0),
    REITPropertyType.CELL_TOWERS: (22.0, 26.0),
    # Residential (varies by geography)
    REITPropertyType.RESIDENTIAL_SUNBELT: (18.0, 20.0),
    REITPropertyType.RESIDENTIAL_COASTAL: (16.0, 18.0),
    REITPropertyType.RESIDENTIAL_GENERAL: (16.0, 19.0),
    # Healthcare (aging demographics but reimbursement risk)
    REITPropertyType.HEALTHCARE: (14.0, 16.0),
    # Office (challenged post-COVID)
    REITPropertyType.OFFICE_CLASS_A: (12.0, 14.0),
    REITPropertyType.OFFICE_CLASS_B: (8.0, 10.0),
    REITPropertyType.OFFICE_GENERAL: (10.0, 12.0),
    # Retail (varies significantly by format)
    REITPropertyType.REGIONAL_MALLS: (6.0, 10.0),
    REITPropertyType.STRIP_CENTERS: (12.0, 15.0),
    REITPropertyType.RETAIL_GENERAL: (10.0, 13.0),
    # Other property types
    REITPropertyType.NET_LEASE: (14.0, 17.0),
    REITPropertyType.SELF_STORAGE: (18.0, 21.0),
    REITPropertyType.SPECIALTY: (14.0, 18.0),
    REITPropertyType.DIVERSIFIED: (12.0, 15.0),
    # Fallback for unknown
    REITPropertyType.UNKNOWN: (12.0, 18.0),
}


# Known REIT symbols with their property types
# This provides explicit mappings for major REITs
KNOWN_REIT_MAPPINGS: Dict[str, REITPropertyType] = {
    # Industrial/Logistics
    "PLD": REITPropertyType.INDUSTRIAL_LOGISTICS,  # Prologis
    "DRE": REITPropertyType.INDUSTRIAL_LOGISTICS,  # Duke Realty (merged with Prologis)
    "REXR": REITPropertyType.INDUSTRIAL_LOGISTICS,  # Rexford Industrial
    "FR": REITPropertyType.INDUSTRIAL_LOGISTICS,  # First Industrial
    "STAG": REITPropertyType.INDUSTRIAL_LOGISTICS,  # STAG Industrial
    "EGP": REITPropertyType.INDUSTRIAL_LOGISTICS,  # EastGroup Properties
    "TRNO": REITPropertyType.INDUSTRIAL_LOGISTICS,  # Terreno Realty
    # Data Centers
    "EQIX": REITPropertyType.DATA_CENTERS,  # Equinix
    "DLR": REITPropertyType.DATA_CENTERS,  # Digital Realty
    "COR": REITPropertyType.DATA_CENTERS,  # CoreSite (acquired)
    "QTS": REITPropertyType.DATA_CENTERS,  # QTS (acquired by Blackstone)
    # Cell Towers
    "AMT": REITPropertyType.CELL_TOWERS,  # American Tower
    "CCI": REITPropertyType.CELL_TOWERS,  # Crown Castle
    "SBAC": REITPropertyType.CELL_TOWERS,  # SBA Communications
    # Residential - Sunbelt focus
    "MAA": REITPropertyType.RESIDENTIAL_SUNBELT,  # Mid-America Apartment
    "CPT": REITPropertyType.RESIDENTIAL_SUNBELT,  # Camden Property Trust
    "NNN": REITPropertyType.NET_LEASE,  # National Retail Properties
    "INVH": REITPropertyType.RESIDENTIAL_SUNBELT,  # Invitation Homes (SFR, Sunbelt focus)
    "AMH": REITPropertyType.RESIDENTIAL_SUNBELT,  # American Homes 4 Rent
    # Residential - Coastal focus
    "EQR": REITPropertyType.RESIDENTIAL_COASTAL,  # Equity Residential
    "AVB": REITPropertyType.RESIDENTIAL_COASTAL,  # AvalonBay
    "ESS": REITPropertyType.RESIDENTIAL_COASTAL,  # Essex Property Trust (West Coast)
    "AIV": REITPropertyType.RESIDENTIAL_COASTAL,  # Apartment Investment & Mgmt
    "UDR": REITPropertyType.RESIDENTIAL_GENERAL,  # UDR (diversified)
    # Healthcare
    "WELL": REITPropertyType.HEALTHCARE,  # Welltower
    "VTR": REITPropertyType.HEALTHCARE,  # Ventas
    "PEAK": REITPropertyType.HEALTHCARE,  # Healthpeak Properties
    "OHI": REITPropertyType.HEALTHCARE,  # Omega Healthcare Investors
    "HR": REITPropertyType.HEALTHCARE,  # Healthcare Realty
    "DOC": REITPropertyType.HEALTHCARE,  # Physicians Realty Trust
    "SBRA": REITPropertyType.HEALTHCARE,  # Sabra Health Care
    "LTC": REITPropertyType.HEALTHCARE,  # LTC Properties
    "CTRE": REITPropertyType.HEALTHCARE,  # CareTrust REIT
    "NHI": REITPropertyType.HEALTHCARE,  # National Health Investors
    # Office - Class A
    "BXP": REITPropertyType.OFFICE_CLASS_A,  # Boston Properties
    "VNO": REITPropertyType.OFFICE_CLASS_A,  # Vornado (NYC trophy)
    "SLG": REITPropertyType.OFFICE_CLASS_A,  # SL Green (NYC)
    "KRC": REITPropertyType.OFFICE_CLASS_A,  # Kilroy Realty (West Coast)
    # Office - General/Class B
    "ARE": REITPropertyType.OFFICE_CLASS_A,  # Alexandria Real Estate (life science)
    "DEI": REITPropertyType.OFFICE_GENERAL,  # Douglas Emmett
    "HIW": REITPropertyType.OFFICE_GENERAL,  # Highwoods Properties
    "CUZ": REITPropertyType.OFFICE_GENERAL,  # Cousins Properties
    "OFC": REITPropertyType.OFFICE_GENERAL,  # Corporate Office Properties
    "PDM": REITPropertyType.OFFICE_GENERAL,  # Piedmont Office Realty
    # Regional Malls
    "SPG": REITPropertyType.REGIONAL_MALLS,  # Simon Property Group
    "MAC": REITPropertyType.REGIONAL_MALLS,  # Macerich
    "TCO": REITPropertyType.REGIONAL_MALLS,  # Taubman Centers (acquired)
    "CBL": REITPropertyType.REGIONAL_MALLS,  # CBL & Associates
    "PEI": REITPropertyType.REGIONAL_MALLS,  # Pennsylvania REIT
    "WPG": REITPropertyType.REGIONAL_MALLS,  # Washington Prime Group
    # Strip Centers / Shopping Centers
    "REG": REITPropertyType.STRIP_CENTERS,  # Regency Centers
    "FRT": REITPropertyType.STRIP_CENTERS,  # Federal Realty
    "KIM": REITPropertyType.STRIP_CENTERS,  # Kimco Realty
    "ROIC": REITPropertyType.STRIP_CENTERS,  # Retail Opportunity Investments
    "BRX": REITPropertyType.STRIP_CENTERS,  # Brixmor Property
    "AKR": REITPropertyType.STRIP_CENTERS,  # Acadia Realty Trust
    "SITE": REITPropertyType.STRIP_CENTERS,  # Site Centers
    "UE": REITPropertyType.STRIP_CENTERS,  # Urban Edge Properties
    "RPAI": REITPropertyType.STRIP_CENTERS,  # Retail Properties of America
    # Net Lease
    "O": REITPropertyType.NET_LEASE,  # Realty Income
    "WPC": REITPropertyType.NET_LEASE,  # W.P. Carey
    "STOR": REITPropertyType.NET_LEASE,  # STORE Capital (acquired)
    "ADC": REITPropertyType.NET_LEASE,  # Agree Realty
    "EPRT": REITPropertyType.NET_LEASE,  # Essential Properties Realty Trust
    "SRC": REITPropertyType.NET_LEASE,  # Spirit Realty Capital
    "GTY": REITPropertyType.NET_LEASE,  # Getty Realty
    "FCPT": REITPropertyType.NET_LEASE,  # Four Corners Property Trust
    # Self Storage
    "PSA": REITPropertyType.SELF_STORAGE,  # Public Storage
    "EXR": REITPropertyType.SELF_STORAGE,  # Extra Space Storage
    "CUBE": REITPropertyType.SELF_STORAGE,  # CubeSmart
    "LSI": REITPropertyType.SELF_STORAGE,  # Life Storage (acquired by EXR)
    "NSA": REITPropertyType.SELF_STORAGE,  # National Storage Affiliates
    # Specialty
    "VICI": REITPropertyType.SPECIALTY,  # VICI Properties (gaming)
    "GLPI": REITPropertyType.SPECIALTY,  # Gaming and Leisure Properties
    "RHP": REITPropertyType.SPECIALTY,  # Ryman Hospitality Properties
    "EPR": REITPropertyType.SPECIALTY,  # EPR Properties (experiential)
    "IIPR": REITPropertyType.SPECIALTY,  # Innovative Industrial Properties (cannabis)
    "IRM": REITPropertyType.SPECIALTY,  # Iron Mountain (document storage)
    "COLD": REITPropertyType.SPECIALTY,  # Americold Realty Trust (cold storage)
    # Diversified
    "WY": REITPropertyType.DIVERSIFIED,  # Weyerhaeuser (timberland)
    "RYN": REITPropertyType.DIVERSIFIED,  # Rayonier (timberland)
    "PCH": REITPropertyType.DIVERSIFIED,  # PotlatchDeltic (timberland)
}


# Pattern-based detection for company names
COMPANY_NAME_PATTERNS: Dict[REITPropertyType, list] = {
    REITPropertyType.INDUSTRIAL_LOGISTICS: [
        r"industrial",
        r"logistics",
        r"warehouse",
        r"distribution",
        r"prologis",
    ],
    REITPropertyType.DATA_CENTERS: [
        r"data.?center",
        r"digital",
        r"equinix",
        r"coresite",
        r"cyrusone",
    ],
    REITPropertyType.CELL_TOWERS: [
        r"tower",
        r"wireless",
        r"cell",
        r"communications?.*(infrastructure|tower)",
        r"american tower",
        r"crown castle",
        r"sba comm",
    ],
    REITPropertyType.RESIDENTIAL_GENERAL: [
        r"apartment",
        r"residential",
        r"multifamily",
        r"single.?family",
        r"homes?.?(for)?.?rent",
    ],
    REITPropertyType.HEALTHCARE: [
        r"health",
        r"medical",
        r"senior",
        r"care",
        r"physician",
        r"hospital",
        r"life.?science",
        r"skilled.?nursing",
    ],
    REITPropertyType.OFFICE_GENERAL: [
        r"office",
        r"workplace",
        r"corporate",
    ],
    REITPropertyType.REGIONAL_MALLS: [
        r"mall",
        r"shopping.?center",
        r"retail.*(complex|destination)",
    ],
    REITPropertyType.STRIP_CENTERS: [
        r"strip.?center",
        r"neighborhood.?center",
        r"community.?center",
        r"grocery.?anchored",
        r"open.?air",
    ],
    REITPropertyType.NET_LEASE: [
        r"net.?lease",
        r"triple.?net",
        r"nnn",
        r"realty.?income",
        r"single.?tenant",
    ],
    REITPropertyType.SELF_STORAGE: [
        r"storage",
        r"self.?storage",
        r"mini.?storage",
    ],
    REITPropertyType.SPECIALTY: [
        r"gaming",
        r"casino",
        r"experiential",
        r"entertainment",
        r"timber",
        r"farmland",
        r"billboard",
        r"outdoor.?advertising",
    ],
}


@dataclass
class REITPropertyTypeResult:
    """Result from REIT property type detection."""

    property_type: REITPropertyType
    confidence: str  # "high", "medium", "low"
    detection_method: str  # "symbol_lookup", "name_pattern", "industry_classification", "default"
    details: Dict


def detect_reit_property_type(
    symbol: str,
    company_name: Optional[str] = None,
    industry: Optional[str] = None,
    sic_code: Optional[str] = None,
) -> REITPropertyTypeResult:
    """
    Detect REIT property type from available information.

    Priority:
    1. Known symbol mapping (highest confidence)
    2. Company name pattern matching
    3. Industry classification
    4. Default to UNKNOWN

    Args:
        symbol: Stock ticker symbol
        company_name: Company name (optional)
        industry: Industry classification (optional)
        sic_code: SIC code (optional, for future use)

    Returns:
        REITPropertyTypeResult with property type and confidence
    """
    symbol_upper = symbol.upper()

    # Priority 1: Check known symbol mappings
    if symbol_upper in KNOWN_REIT_MAPPINGS:
        property_type = KNOWN_REIT_MAPPINGS[symbol_upper]
        logger.info(f"{symbol} - REIT property type detected via symbol mapping: {property_type.value}")
        return REITPropertyTypeResult(
            property_type=property_type,
            confidence="high",
            detection_method="symbol_lookup",
            details={"matched_symbol": symbol_upper},
        )

    # Priority 2: Company name pattern matching
    if company_name:
        company_lower = company_name.lower()

        for prop_type, patterns in COMPANY_NAME_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, company_lower, re.IGNORECASE):
                    logger.info(
                        f"{symbol} - REIT property type detected via name pattern: "
                        f"{prop_type.value} (matched: '{pattern}')"
                    )
                    return REITPropertyTypeResult(
                        property_type=prop_type,
                        confidence="medium",
                        detection_method="name_pattern",
                        details={"company_name": company_name, "matched_pattern": pattern},
                    )

    # Priority 3: Industry classification
    if industry:
        industry_lower = industry.lower()

        # Map industry to property type
        industry_mappings = {
            "industrial": REITPropertyType.INDUSTRIAL_LOGISTICS,
            "warehouse": REITPropertyType.INDUSTRIAL_LOGISTICS,
            "logistics": REITPropertyType.INDUSTRIAL_LOGISTICS,
            "data center": REITPropertyType.DATA_CENTERS,
            "tower": REITPropertyType.CELL_TOWERS,
            "cell tower": REITPropertyType.CELL_TOWERS,
            "residential": REITPropertyType.RESIDENTIAL_GENERAL,
            "apartment": REITPropertyType.RESIDENTIAL_GENERAL,
            "multifamily": REITPropertyType.RESIDENTIAL_GENERAL,
            "healthcare": REITPropertyType.HEALTHCARE,
            "medical": REITPropertyType.HEALTHCARE,
            "senior": REITPropertyType.HEALTHCARE,
            "office": REITPropertyType.OFFICE_GENERAL,
            "mall": REITPropertyType.REGIONAL_MALLS,
            "retail": REITPropertyType.RETAIL_GENERAL,
            "shopping": REITPropertyType.RETAIL_GENERAL,
            "storage": REITPropertyType.SELF_STORAGE,
            "net lease": REITPropertyType.NET_LEASE,
            "triple net": REITPropertyType.NET_LEASE,
            "specialty": REITPropertyType.SPECIALTY,
            "diversified": REITPropertyType.DIVERSIFIED,
        }

        for keyword, prop_type in industry_mappings.items():
            if keyword in industry_lower:
                logger.info(
                    f"{symbol} - REIT property type detected via industry: "
                    f"{prop_type.value} (industry: '{industry}')"
                )
                return REITPropertyTypeResult(
                    property_type=prop_type,
                    confidence="medium",
                    detection_method="industry_classification",
                    details={"industry": industry, "matched_keyword": keyword},
                )

    # Priority 4: Default to UNKNOWN
    logger.warning(f"{symbol} - Could not detect REIT property type, defaulting to UNKNOWN")
    return REITPropertyTypeResult(
        property_type=REITPropertyType.UNKNOWN,
        confidence="low",
        detection_method="default",
        details={"company_name": company_name, "industry": industry, "sic_code": sic_code},
    )


def get_ffo_multiple_range(property_type: REITPropertyType) -> Tuple[float, float]:
    """
    Get FFO multiple range for a property type.

    Args:
        property_type: REIT property type

    Returns:
        Tuple of (low_multiple, high_multiple)
    """
    return REIT_FFO_MULTIPLES.get(property_type, REIT_FFO_MULTIPLES[REITPropertyType.UNKNOWN])


def get_base_ffo_multiple(property_type: REITPropertyType) -> float:
    """
    Get base FFO multiple (midpoint of range) for a property type.

    Args:
        property_type: REIT property type

    Returns:
        Base FFO multiple (midpoint)
    """
    low, high = get_ffo_multiple_range(property_type)
    return (low + high) / 2


def adjust_ffo_multiple_for_rates(
    base_multiple: float,
    current_10yr_yield: float,
    base_yield: float = 0.04,  # 4% is roughly the long-term "neutral" rate
) -> float:
    """
    Adjust FFO multiple based on interest rate environment.

    REITs are rate-sensitive:
    - Higher rates = lower multiples (competition from bonds, higher cap rates)
    - Lower rates = higher multiples (TINA effect, lower cap rates)

    Formula:
    - For every 50bp above base_yield, reduce multiple by 0.5x
    - For every 50bp below base_yield, increase multiple by 0.5x
    - Cap adjustment at +/- 30% of base multiple

    Args:
        base_multiple: Starting FFO multiple
        current_10yr_yield: Current 10-year Treasury yield (as decimal, e.g., 0.045 for 4.5%)
        base_yield: "Neutral" yield assumption (default 4%)

    Returns:
        Adjusted FFO multiple
    """
    # Calculate yield difference
    yield_diff = current_10yr_yield - base_yield

    # Adjustment: -0.5x per 50bp above base (or +0.5x per 50bp below)
    # yield_diff of 0.005 = 50bp
    adjustment = -0.5 * (yield_diff / 0.005)

    # Cap adjustment at +/- 30% of base multiple
    max_adjustment = base_multiple * 0.3
    adjustment = max(min(adjustment, max_adjustment), -max_adjustment)

    adjusted_multiple = base_multiple + adjustment

    # Floor at 70% of base (don't go too low even in high rate environment)
    min_multiple = base_multiple * 0.7
    adjusted_multiple = max(adjusted_multiple, min_multiple)

    logger.debug(
        f"FFO multiple adjustment: base={base_multiple:.1f}x, "
        f"10yr={current_10yr_yield*100:.2f}%, adjustment={adjustment:+.2f}x, "
        f"adjusted={adjusted_multiple:.1f}x"
    )

    return adjusted_multiple


def get_current_treasury_yield() -> Optional[float]:
    """
    Get current 10-year Treasury yield from FRED database.

    Returns:
        10-year Treasury yield as decimal (e.g., 0.045 for 4.5%) or None if unavailable
    """
    try:
        from investigator.infrastructure.external.fred import MacroIndicatorsFetcher

        fetcher = MacroIndicatorsFetcher()
        indicators = fetcher.get_latest_values(["DGS10"])

        if "DGS10" in indicators and indicators["DGS10"].get("value"):
            # FRED stores rates as percentages (e.g., 4.5 for 4.5%)
            # Convert to decimal (0.045)
            yield_pct = indicators["DGS10"]["value"]
            yield_decimal = yield_pct / 100.0
            logger.info(f"Current 10-year Treasury yield: {yield_pct:.2f}%")
            return yield_decimal
        else:
            logger.warning("10-year Treasury yield not available from FRED")
            return None

    except Exception as e:
        logger.warning(f"Error fetching Treasury yield from FRED: {e}")
        return None


@dataclass
class REITValuationResult:
    """Result from REIT FFO-based valuation."""

    fair_value: float
    ffo_per_share: float
    base_ffo_multiple: float
    adjusted_ffo_multiple: float
    property_type: REITPropertyType
    property_type_confidence: str
    detection_method: str
    current_10yr_yield: Optional[float]
    rate_adjustment: float
    warnings: list


def value_reit(
    symbol: str,
    financials: Dict,
    current_price: float,
    company_name: Optional[str] = None,
    industry: Optional[str] = None,
    use_rate_adjustment: bool = True,
) -> REITValuationResult:
    """
    Value a REIT using property-type-specific FFO multiples with rate adjustment.

    This is a comprehensive REIT valuation that:
    1. Detects property type from symbol, name, or industry
    2. Applies property-specific FFO multiples
    3. Adjusts for current interest rate environment

    Args:
        symbol: Stock ticker symbol
        financials: Dictionary of financial metrics (must include net_income, depreciation, shares)
        current_price: Current stock price
        company_name: Company name (optional, for property type detection)
        industry: Industry classification (optional)
        use_rate_adjustment: Whether to adjust for interest rates (default True)

    Returns:
        REITValuationResult with comprehensive valuation details
    """
    warnings = []

    # Step 1: Detect property type
    property_result = detect_reit_property_type(symbol=symbol, company_name=company_name, industry=industry)

    # Step 2: Calculate FFO
    net_income = financials.get("net_income", 0)
    depreciation = financials.get("depreciation_amortization", 0)
    shares_outstanding = financials.get("shares_outstanding", 0)

    if not shares_outstanding:
        raise ValueError("Missing shares_outstanding for REIT valuation")

    if not net_income:
        raise ValueError("Missing net_income for REIT valuation")

    # FFO = Net Income + Depreciation (simplified, ignoring gains/losses on property sales)
    ffo = net_income + depreciation
    ffo_per_share = ffo / shares_outstanding

    if not depreciation:
        warnings.append("Depreciation not available, FFO may be understated")

    # Step 3: Get base FFO multiple for property type
    base_multiple = get_base_ffo_multiple(property_result.property_type)

    # Step 4: Adjust for interest rates
    current_yield = None
    rate_adjustment = 0.0

    if use_rate_adjustment:
        current_yield = get_current_treasury_yield()

        if current_yield is not None:
            adjusted_multiple = adjust_ffo_multiple_for_rates(base_multiple, current_yield)
            rate_adjustment = adjusted_multiple - base_multiple
        else:
            adjusted_multiple = base_multiple
            warnings.append("Treasury yield unavailable, no rate adjustment applied")
    else:
        adjusted_multiple = base_multiple

    # Step 5: Calculate fair value
    fair_value = ffo_per_share * adjusted_multiple

    # Step 6: Add confidence warnings
    if property_result.confidence == "low":
        warnings.append(f"Property type confidence is low, using generic multiple range")

    logger.info(
        f"{symbol} - REIT valuation: property_type={property_result.property_type.value}, "
        f"FFO/share=${ffo_per_share:.2f}, base_multiple={base_multiple:.1f}x, "
        f"adjusted_multiple={adjusted_multiple:.1f}x, fair_value=${fair_value:.2f}"
    )

    return REITValuationResult(
        fair_value=fair_value,
        ffo_per_share=ffo_per_share,
        base_ffo_multiple=base_multiple,
        adjusted_ffo_multiple=adjusted_multiple,
        property_type=property_result.property_type,
        property_type_confidence=property_result.confidence,
        detection_method=property_result.detection_method,
        current_10yr_yield=current_yield,
        rate_adjustment=rate_adjustment,
        warnings=warnings,
    )
