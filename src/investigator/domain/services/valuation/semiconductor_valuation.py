"""
Semiconductor Valuation Module

Implements specialized valuation for semiconductor companies accounting for cyclicality.

Semiconductor companies are notoriously cyclical and require special valuation adjustments:
1. Inventory levels signal cycle position (high inventory = peak, low = trough)
2. Book-to-bill ratios indicate demand momentum
3. Chip type affects growth rates and margins (logic vs memory vs analog)
4. Cycle position requires margin normalization

This module provides:
- Semiconductor metrics extraction from XBRL data
- Cycle position detection from inventory and book-to-bill signals
- Chip type classification from company/industry
- Cycle-adjusted margin normalization
- Comprehensive semiconductor valuation

Key insight: Semiconductor valuations at cycle peaks should be discounted (margins will revert),
while trough valuations should carry premiums (margins will recover).

Author: Claude Code
Date: 2025-12-30
Phase: P2-C (Semiconductor Valuation Module)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================
# SEMICONDUCTOR CLASSIFICATION
# ====================

class ChipType(Enum):
    """Classification of semiconductor chip types for valuation purposes."""
    LOGIC = "logic"              # Processors, GPUs, FPGAs (NVDA, AMD, INTC)
    MEMORY = "memory"            # DRAM, NAND (MU, WDC)
    ANALOG = "analog"            # Power, signal processing (TXN, ADI)
    DISCRETE = "discrete"        # Transistors, diodes (ON, DIOD)
    MIXED_SIGNAL = "mixed_signal"  # Analog + digital (MXIM, MCHP)
    EQUIPMENT = "equipment"      # Semiconductor equipment (ASML, LRCX, AMAT)
    UNKNOWN = "unknown"


class CyclePosition(Enum):
    """Position in semiconductor industry cycle."""
    PEAK = "peak"                    # Top of cycle - elevated margins, high inventory
    PEAK_TO_NORMAL = "peak_to_normal"  # Declining from peak
    NORMAL = "normal"                # Mid-cycle equilibrium
    NORMAL_TO_TROUGH = "normal_to_trough"  # Declining toward trough
    TROUGH = "trough"                # Bottom of cycle - depressed margins, low inventory
    UNKNOWN = "unknown"


class GrowthProfile(Enum):
    """Growth profile classification for valuation purposes."""
    HYPER_GROWTH = "hyper_growth"        # >50% revenue growth (AI leaders like NVDA)
    HIGH_GROWTH = "high_growth"          # 25-50% revenue growth
    MODERATE_GROWTH = "moderate_growth"  # 10-25% revenue growth
    LOW_GROWTH = "low_growth"            # <10% revenue growth
    DECLINING = "declining"              # Negative growth
    UNKNOWN = "unknown"


# Industries that qualify for semiconductor valuation
SEMICONDUCTOR_INDUSTRIES = [
    "Semiconductors",
    "Semiconductor Equipment & Materials",
    "Semiconductor - Memory",
    "Semiconductor - Analog & Mixed-Signal",
    "Semiconductor - Broad Line",
    "Semiconductor - Specialized",
    "Semiconductor Materials & Equipment",
    "Technology Hardware, Storage & Peripherals",
    "Electronic Equipment & Instruments",
    "Semiconductors & Semiconductor Equipment",
]


# Known semiconductor symbols for explicit chip type mapping
KNOWN_SEMICONDUCTOR_COMPANIES: Dict[str, ChipType] = {
    # Logic/Processors
    "NVDA": ChipType.LOGIC,    # NVIDIA - GPUs
    "AMD": ChipType.LOGIC,     # AMD - CPUs, GPUs
    "INTC": ChipType.LOGIC,    # Intel - CPUs
    "AVGO": ChipType.LOGIC,    # Broadcom - networking chips
    "QCOM": ChipType.LOGIC,    # Qualcomm - mobile processors
    "MRVL": ChipType.LOGIC,    # Marvell - data center chips

    # Memory
    "MU": ChipType.MEMORY,     # Micron - DRAM, NAND
    "WDC": ChipType.MEMORY,    # Western Digital - NAND
    "STX": ChipType.MEMORY,    # Seagate - storage (memory-adjacent)

    # Analog
    "TXN": ChipType.ANALOG,    # Texas Instruments - analog
    "ADI": ChipType.ANALOG,    # Analog Devices - analog/mixed-signal
    "MCHP": ChipType.ANALOG,   # Microchip - analog/MCU

    # Mixed-Signal
    "NXPI": ChipType.MIXED_SIGNAL,  # NXP - automotive, IoT
    "ON": ChipType.MIXED_SIGNAL,    # ON Semi - power/mixed-signal
    "SWKS": ChipType.MIXED_SIGNAL,  # Skyworks - RF chips
    "QRVO": ChipType.MIXED_SIGNAL,  # Qorvo - RF chips

    # Equipment
    "ASML": ChipType.EQUIPMENT,  # ASML - lithography
    "LRCX": ChipType.EQUIPMENT,  # Lam Research - etch
    "AMAT": ChipType.EQUIPMENT,  # Applied Materials - deposition
    "KLAC": ChipType.EQUIPMENT,  # KLA - inspection
    "TER": ChipType.EQUIPMENT,   # Teradyne - test equipment
}


# ====================
# SEMICONDUCTOR VALUATION TIER
# ====================

SEMICONDUCTOR_TIER = {
    "name": "semiconductor",
    "description": "Specialized tier for semiconductor companies with cycle-adjusted valuation",
    "weights": {
        "ev_ebitda": 35,         # Primary multiple (accounts for capex intensity)
        "pe": 25,                # P/E on normalized earnings
        "dcf": 30,               # DCF with cycle-normalized margins
        "cycle_adjustment": 10,  # Explicit cycle position adjustment
    },
    "parameters": {
        "terminal_growth": 0.04,      # Secular growth from AI/digitization
        "peak_margin": 0.35,          # Peak cycle gross margin
        "normal_margin": 0.25,        # Mid-cycle gross margin
        "trough_margin": 0.15,        # Trough cycle gross margin
        "discount_rate_adjustment": 0.01,  # Higher risk premium for cyclicality
    },
    "industry_matches": SEMICONDUCTOR_INDUSTRIES,
}


# ====================
# GROWTH-ADJUSTED VALUATION PARAMETERS
# ====================

GROWTH_ADJUSTED_PE_MULTIPLES = {
    # Growth profile -> (base_pe, max_pe, peg_target)
    # PEG target of 1.0 means P/E should equal growth rate
    # AI/semiconductor leaders can sustain PEG up to 1.5-2.0
    GrowthProfile.HYPER_GROWTH: {
        "base_pe": 40.0,       # Base P/E for >50% growers
        "max_pe": 80.0,        # Cap even for exceptional growth
        "peg_target": 1.0,     # PEG = 1.0 (P/E = growth rate)
        "peg_premium": 0.5,    # Allow up to 1.5 PEG for leaders
        "forward_pe_discount": 0.6,  # Forward P/E typically 60% of trailing
    },
    GrowthProfile.HIGH_GROWTH: {
        "base_pe": 30.0,
        "max_pe": 50.0,
        "peg_target": 1.0,
        "peg_premium": 0.3,    # Allow up to 1.3 PEG
        "forward_pe_discount": 0.7,
    },
    GrowthProfile.MODERATE_GROWTH: {
        "base_pe": 22.0,
        "max_pe": 35.0,
        "peg_target": 1.0,
        "peg_premium": 0.2,
        "forward_pe_discount": 0.8,
    },
    GrowthProfile.LOW_GROWTH: {
        "base_pe": 15.0,
        "max_pe": 25.0,
        "peg_target": 1.2,     # Lower growth needs lower PEG
        "peg_premium": 0.0,
        "forward_pe_discount": 0.9,
    },
    GrowthProfile.DECLINING: {
        "base_pe": 10.0,
        "max_pe": 18.0,
        "peg_target": None,    # PEG not applicable for declining
        "peg_premium": 0.0,
        "forward_pe_discount": 1.0,
    },
    GrowthProfile.UNKNOWN: {
        "base_pe": 18.0,       # Conservative default
        "max_pe": 30.0,
        "peg_target": 1.0,
        "peg_premium": 0.0,
        "forward_pe_discount": 0.85,
    },
}


# Chip type growth premium adjustments
CHIP_TYPE_GROWTH_PREMIUM = {
    ChipType.LOGIC: 1.2,       # 20% premium for AI/GPU growth potential
    ChipType.MEMORY: 0.8,      # 20% discount for cyclicality
    ChipType.ANALOG: 1.0,      # No adjustment - stable
    ChipType.DISCRETE: 0.9,    # Slight discount
    ChipType.MIXED_SIGNAL: 1.0,
    ChipType.EQUIPMENT: 1.1,   # Premium for enabling role
    ChipType.UNKNOWN: 1.0,
}


# ====================
# SEMICONDUCTOR METRICS
# ====================

@dataclass
class SemiconductorMetrics:
    """Container for semiconductor-specific metrics."""
    inventory_days: Optional[float] = None
    book_to_bill: Optional[float] = None
    inventory_to_sales: Optional[float] = None
    asp_trend: Optional[float] = None  # Average selling price trend (YoY % change)
    cycle_position: CyclePosition = CyclePosition.UNKNOWN
    chip_type: ChipType = ChipType.UNKNOWN


@dataclass
class SemiconductorValuationResult:
    """Result from semiconductor company valuation."""
    fair_value: float
    cycle_position: CyclePosition
    chip_type: ChipType
    normalized_margin: float
    cycle_adjustment: float  # Multiplier applied for cycle position
    confidence: str  # "high", "medium", "low"
    warnings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


# ====================
# XBRL METRIC EXTRACTION
# ====================

def extract_semiconductor_metrics_from_xbrl(
    symbol: str,
    xbrl_data: Dict,
    financials: Dict,
) -> SemiconductorMetrics:
    """
    Extract semiconductor-specific metrics from XBRL data.

    Uses multiple XBRL tags to construct semiconductor cycle indicators:
    - Inventory days (DaysInventoryOutstanding)
    - Book-to-bill ratio (inferred from backlog changes)
    - Inventory-to-sales ratio

    Args:
        symbol: Stock ticker symbol
        xbrl_data: Raw XBRL data dictionary (us-gaap format)
        financials: Financial data dictionary with revenue, inventory, etc.

    Returns:
        SemiconductorMetrics with extracted values
    """
    from utils.xbrl_tag_aliases import XBRLTagAliasMapper

    mapper = XBRLTagAliasMapper()
    metrics = SemiconductorMetrics()

    # Define semiconductor-related canonical names to extract
    semiconductor_canonical_names = [
        'inventory_days',
        'book_to_bill_ratio',
        'inventory_to_sales',
        'inventory',
        'revenues',
        'cost_of_revenue',
    ]

    us_gaap = xbrl_data.get('facts', {}).get('us-gaap', {})
    if not us_gaap:
        logger.warning(f"{symbol} - No us-gaap data available for semiconductor metric extraction")
        return metrics

    extracted_values = {}

    for canonical_name in semiconductor_canonical_names:
        aliases = mapper.get_xbrl_aliases(canonical_name)

        for alias in aliases:
            if alias in us_gaap:
                concept = us_gaap[alias]
                units = concept.get('units', {})

                # Check for USD or pure number units
                data_sources = []
                if 'USD' in units:
                    data_sources = units.get('USD', [])
                elif 'pure' in units:
                    data_sources = units.get('pure', [])

                if data_sources:
                    # Get the latest annual value (10-K preferred, then 10-Q)
                    sorted_data = sorted(
                        [d for d in data_sources if d.get('form') in ['10-K', '10-Q', '20-F']],
                        key=lambda x: (
                            x.get('fy', 0),
                            {'FY': 5, 'Q4': 4, 'Q3': 3, 'Q2': 2, 'Q1': 1}.get(x.get('fp', ''), 0)
                        ),
                        reverse=True
                    )

                    if sorted_data:
                        value = sorted_data[0].get('val')
                        if value is not None:
                            extracted_values[canonical_name] = float(value)
                            logger.debug(
                                f"{symbol} - Extracted {canonical_name} from {alias}: {float(value)}"
                            )
                            break  # Found value, move to next canonical name

    # Calculate inventory days if we have inventory and COGS
    inventory = extracted_values.get('inventory') or financials.get('inventory')
    cost_of_revenue = extracted_values.get('cost_of_revenue') or financials.get('cost_of_revenue')
    revenues = extracted_values.get('revenues') or financials.get('total_revenue') or financials.get('revenue')

    if inventory and cost_of_revenue and cost_of_revenue > 0:
        # Inventory days = (Inventory / COGS) * 365
        metrics.inventory_days = (inventory / cost_of_revenue) * 365
        logger.info(
            f"{symbol} - Calculated inventory days: {metrics.inventory_days:.1f} days"
        )

    # Calculate inventory-to-sales ratio
    if inventory and revenues and revenues > 0:
        metrics.inventory_to_sales = inventory / revenues
        logger.info(
            f"{symbol} - Inventory-to-sales ratio: {metrics.inventory_to_sales:.2%}"
        )

    # Book-to-bill ratio (if directly available from XBRL)
    if 'book_to_bill_ratio' in extracted_values:
        metrics.book_to_bill = extracted_values['book_to_bill_ratio']
        logger.info(f"{symbol} - Book-to-bill ratio: {metrics.book_to_bill:.2f}")

    # Detect cycle position based on metrics
    metrics.cycle_position = _detect_cycle_position(metrics)

    # Detect chip type
    metrics.chip_type = _detect_chip_type(symbol, xbrl_data)

    # Log extraction summary
    logger.info(
        f"{symbol} - Semiconductor metrics extraction complete: "
        f"Inventory days={metrics.inventory_days:.1f if metrics.inventory_days else 'N/A'}, "
        f"Cycle position={metrics.cycle_position.value}, "
        f"Chip type={metrics.chip_type.value}"
    )

    return metrics


# ====================
# CYCLE POSITION DETECTION
# ====================

def _detect_cycle_position(metrics: SemiconductorMetrics) -> CyclePosition:
    """
    Detect semiconductor cycle position from inventory and book-to-bill signals.

    Heuristics:
    - High inventory days (>100) + B2B < 1.0: Peak or declining
    - Low inventory days (<60) + B2B > 1.0: Trough or recovering
    - Normal inventory (60-100) + B2B ~1.0: Normal/mid-cycle

    Args:
        metrics: SemiconductorMetrics with inventory and B2B data

    Returns:
        CyclePosition enum value
    """
    # Default to unknown if insufficient data
    if metrics.inventory_days is None and metrics.book_to_bill is None:
        return CyclePosition.UNKNOWN

    # Inventory-based cycle detection
    inv_signal = "normal"
    if metrics.inventory_days is not None:
        if metrics.inventory_days > 120:
            inv_signal = "high"  # Peak territory
        elif metrics.inventory_days > 100:
            inv_signal = "elevated"  # Peak to normal
        elif metrics.inventory_days < 50:
            inv_signal = "low"  # Trough territory
        elif metrics.inventory_days < 70:
            inv_signal = "depleted"  # Normal to trough

    # Book-to-bill signal (if available)
    btb_signal = "normal"
    if metrics.book_to_bill is not None:
        if metrics.book_to_bill > 1.15:
            btb_signal = "strong"  # Orders outpacing shipments (recovery)
        elif metrics.book_to_bill > 1.0:
            btb_signal = "healthy"
        elif metrics.book_to_bill < 0.85:
            btb_signal = "weak"  # Orders declining (downturn)
        elif metrics.book_to_bill < 1.0:
            btb_signal = "soft"

    # Combine signals to determine cycle position
    if inv_signal == "high" and btb_signal in ["weak", "soft"]:
        return CyclePosition.PEAK
    elif inv_signal == "elevated" or (inv_signal == "high" and btb_signal == "normal"):
        return CyclePosition.PEAK_TO_NORMAL
    elif inv_signal == "low" and btb_signal in ["strong", "healthy"]:
        return CyclePosition.TROUGH
    elif inv_signal == "depleted" or (inv_signal == "low" and btb_signal == "normal"):
        return CyclePosition.NORMAL_TO_TROUGH
    else:
        return CyclePosition.NORMAL


def _detect_chip_type(symbol: str, xbrl_data: Dict) -> ChipType:
    """
    Classify chip type from company symbol or industry data.

    Priority:
    1. Known symbol mapping (highest confidence)
    2. Industry string matching (from XBRL metadata)
    3. Default to UNKNOWN

    Args:
        symbol: Stock ticker symbol
        xbrl_data: XBRL data that may contain industry information

    Returns:
        ChipType enum value
    """
    symbol_upper = symbol.upper()

    # Priority 1: Check known semiconductor company mappings
    if symbol_upper in KNOWN_SEMICONDUCTOR_COMPANIES:
        chip_type = KNOWN_SEMICONDUCTOR_COMPANIES[symbol_upper]
        logger.info(f"{symbol} - Chip type detected via symbol mapping: {chip_type.value}")
        return chip_type

    # Priority 2: Try to infer from company name or description in XBRL
    entity_info = xbrl_data.get('entityName', '').lower()

    if any(keyword in entity_info for keyword in ['memory', 'micron', 'flash', 'nand', 'dram']):
        return ChipType.MEMORY
    elif any(keyword in entity_info for keyword in ['analog', 'power', 'signal']):
        return ChipType.ANALOG
    elif any(keyword in entity_info for keyword in ['equipment', 'materials', 'asml', 'lam']):
        return ChipType.EQUIPMENT
    elif any(keyword in entity_info for keyword in ['processor', 'gpu', 'nvidia', 'amd']):
        return ChipType.LOGIC

    return ChipType.UNKNOWN


# ====================
# CYCLE ADJUSTMENT CALCULATION
# ====================

def calculate_cycle_adjustment(
    cycle_position: CyclePosition,
    current_margin: float,
) -> Tuple[float, str]:
    """
    Calculate valuation adjustment based on semiconductor cycle position.

    At peak: Margins are elevated and will revert down -> discount valuation
    At trough: Margins are depressed and will revert up -> premium valuation
    At normal: No adjustment needed

    Adjustments:
    - Peak: 0.80 (-20% margin discount)
    - Peak-to-normal: 0.90 (-10% discount)
    - Normal: 1.00 (no adjustment)
    - Normal-to-trough: 1.05 (+5% premium)
    - Trough: 1.15 (+15% margin premium)

    Args:
        cycle_position: Current position in semiconductor cycle
        current_margin: Current operating/gross margin

    Returns:
        Tuple of (multiplier, reason_string)
    """
    adjustments = {
        CyclePosition.PEAK: (
            0.80,
            f"Peak cycle adjustment: -20% (margins at {current_margin:.1%} will revert down)"
        ),
        CyclePosition.PEAK_TO_NORMAL: (
            0.90,
            f"Peak-to-normal adjustment: -10% (margins still elevated at {current_margin:.1%})"
        ),
        CyclePosition.NORMAL: (
            1.00,
            f"Normal cycle: no adjustment (margins at mid-cycle {current_margin:.1%})"
        ),
        CyclePosition.NORMAL_TO_TROUGH: (
            1.05,
            f"Normal-to-trough adjustment: +5% (margins depressed at {current_margin:.1%})"
        ),
        CyclePosition.TROUGH: (
            1.15,
            f"Trough cycle adjustment: +15% (margins at {current_margin:.1%} will recover)"
        ),
        CyclePosition.UNKNOWN: (
            1.00,
            "Cycle position unknown: no adjustment applied"
        ),
    }

    multiplier, reason = adjustments.get(cycle_position, (1.00, "Unknown cycle position"))

    logger.info(f"Cycle adjustment: {multiplier:.2f}x - {reason}")

    return multiplier, reason


# ====================
# NORMALIZED MARGIN CALCULATION
# ====================

def calculate_normalized_margin(
    current_margin: float,
    cycle_position: CyclePosition,
    chip_type: ChipType,
) -> float:
    """
    Calculate normalized (mid-cycle) margin for semiconductor company.

    Different chip types have different margin profiles:
    - Logic: Higher margins (30-40% gross)
    - Memory: Highly cyclical margins (10-50% gross)
    - Analog: Stable high margins (40-55% gross)
    - Equipment: Moderate margins (35-45% gross)

    Args:
        current_margin: Current gross or operating margin
        cycle_position: Where in the cycle we are
        chip_type: Type of semiconductor business

    Returns:
        Normalized (mid-cycle) margin estimate
    """
    # Chip type normal margin baselines
    chip_margins = {
        ChipType.LOGIC: 0.35,       # Logic chips - moderate-high
        ChipType.MEMORY: 0.25,      # Memory - highly cyclical
        ChipType.ANALOG: 0.50,      # Analog - stable high margins
        ChipType.DISCRETE: 0.30,    # Discrete - moderate
        ChipType.MIXED_SIGNAL: 0.40,  # Mixed-signal - moderate-high
        ChipType.EQUIPMENT: 0.40,   # Equipment - moderate
        ChipType.UNKNOWN: 0.30,     # Conservative default
    }

    baseline = chip_margins.get(chip_type, 0.30)

    # Adjust baseline based on cycle position
    if cycle_position == CyclePosition.PEAK:
        # Current margin is probably above normal - use baseline
        normalized = baseline
    elif cycle_position == CyclePosition.TROUGH:
        # Current margin is probably below normal - use baseline
        normalized = baseline
    else:
        # Mid-cycle - blend current with baseline
        normalized = (current_margin + baseline) / 2

    logger.info(
        f"Normalized margin: {normalized:.1%} "
        f"(current={current_margin:.1%}, baseline={baseline:.1%}, cycle={cycle_position.value})"
    )

    return normalized


# ====================
# SEMICONDUCTOR INDUSTRY DETECTION
# ====================

def is_semiconductor_industry(industry: Optional[str]) -> bool:
    """
    Check if an industry string matches semiconductor industry patterns.

    Args:
        industry: Industry classification string

    Returns:
        True if industry matches semiconductor patterns
    """
    if not industry:
        return False

    industry_lower = industry.lower()

    # Check against known semiconductor industries
    for semi_industry in SEMICONDUCTOR_INDUSTRIES:
        if semi_industry.lower() in industry_lower:
            return True

    # Check keywords
    semiconductor_keywords = ["semiconductor", "chip", "fab", "foundry", "memory"]
    for keyword in semiconductor_keywords:
        if keyword in industry_lower:
            return True

    return False


def classify_semiconductor_company(
    symbol: str,
    industry: Optional[str] = None,
) -> Tuple[bool, ChipType, str]:
    """
    Classify whether a company is a semiconductor company and determine chip type.

    Args:
        symbol: Stock ticker symbol
        industry: Industry classification string

    Returns:
        Tuple of (is_semiconductor, chip_type, confidence)
    """
    symbol_upper = symbol.upper()

    # Priority 1: Known symbol mapping
    if symbol_upper in KNOWN_SEMICONDUCTOR_COMPANIES:
        chip_type = KNOWN_SEMICONDUCTOR_COMPANIES[symbol_upper]
        return True, chip_type, "high"

    # Priority 2: Industry matching
    if industry and is_semiconductor_industry(industry):
        return True, ChipType.UNKNOWN, "medium"

    return False, ChipType.UNKNOWN, "low"


# ====================
# MAIN VALUATION FUNCTION
# ====================

def value_semiconductor(
    symbol: str,
    financials: Dict,
    current_price: float,
    xbrl_data: Optional[Dict] = None,
    industry: Optional[str] = None,
) -> SemiconductorValuationResult:
    """
    Value semiconductor company with cycle-adjusted methodology.

    This function applies semiconductor-specific adjustments:
    1. Detects chip type (logic, memory, analog, etc.)
    2. Determines cycle position (peak, trough, normal)
    3. Calculates normalized margins
    4. Applies cycle adjustment to valuation

    Args:
        symbol: Stock ticker symbol
        financials: Dictionary of financial metrics
        current_price: Current stock price
        xbrl_data: Optional raw XBRL data for metric extraction
        industry: Industry classification string

    Returns:
        SemiconductorValuationResult with fair value and details
    """
    warnings = []
    details = {}

    # Classify the semiconductor company
    is_semi, chip_type, confidence = classify_semiconductor_company(symbol, industry)

    if not is_semi:
        logger.warning(f"{symbol} - Not classified as semiconductor company")
        return SemiconductorValuationResult(
            fair_value=current_price,  # No adjustment
            cycle_position=CyclePosition.UNKNOWN,
            chip_type=ChipType.UNKNOWN,
            normalized_margin=0.0,
            cycle_adjustment=1.0,
            confidence="low",
            warnings=["Company not classified as semiconductor"],
            details={"classification": "not_semiconductor"},
        )

    # Extract semiconductor metrics
    metrics = SemiconductorMetrics()
    if xbrl_data:
        metrics = extract_semiconductor_metrics_from_xbrl(symbol, xbrl_data, financials)
        if metrics.chip_type != ChipType.UNKNOWN:
            chip_type = metrics.chip_type
    else:
        warnings.append("No XBRL data provided - cycle detection limited")

    # Get current margin
    current_margin = 0.0
    gross_profit = financials.get('gross_profit', 0)
    revenue = financials.get('total_revenue') or financials.get('revenue', 0)
    if revenue > 0 and gross_profit > 0:
        current_margin = gross_profit / revenue
    else:
        # Try operating margin
        operating_income = financials.get('operating_income', 0)
        if revenue > 0 and operating_income > 0:
            current_margin = operating_income / revenue
        else:
            warnings.append("Could not calculate current margin - using default")
            current_margin = 0.25  # Default semiconductor margin

    # Calculate normalized margin
    normalized_margin = calculate_normalized_margin(
        current_margin,
        metrics.cycle_position,
        chip_type,
    )

    # Calculate cycle adjustment
    cycle_adjustment, adjustment_reason = calculate_cycle_adjustment(
        metrics.cycle_position,
        current_margin,
    )
    details["cycle_adjustment_reason"] = adjustment_reason

    # Calculate base fair value from financials
    # Use P/E or EV/EBITDA as base, then apply cycle adjustment
    eps = financials.get('eps_diluted') or financials.get('eps_basic', 0)
    base_pe = 25.0  # Typical semiconductor P/E

    # Adjust P/E for chip type
    pe_adjustments = {
        ChipType.LOGIC: 30.0,      # Premium for AI/GPU growth
        ChipType.MEMORY: 15.0,     # Discount for cyclicality
        ChipType.ANALOG: 25.0,     # Stable business
        ChipType.EQUIPMENT: 22.0,  # Cyclical but essential
        ChipType.MIXED_SIGNAL: 23.0,
        ChipType.DISCRETE: 18.0,
        ChipType.UNKNOWN: 22.0,
    }
    base_pe = pe_adjustments.get(chip_type, 22.0)
    details["base_pe"] = base_pe

    # Calculate fair value
    if eps and eps > 0:
        base_fair_value = eps * base_pe
    else:
        # Fallback to book value
        book_value = financials.get('stockholders_equity', 0)
        shares = financials.get('shares_outstanding', 0)
        if book_value > 0 and shares > 0:
            book_per_share = book_value / shares
            base_fair_value = book_per_share * 2.5  # 2.5x book for semiconductors
            warnings.append("Using book value fallback (no positive EPS)")
        else:
            base_fair_value = current_price
            warnings.append("Could not calculate base fair value")

    # Apply cycle adjustment
    fair_value = base_fair_value * cycle_adjustment

    # Determine confidence
    if confidence == "high" and metrics.cycle_position != CyclePosition.UNKNOWN:
        final_confidence = "high"
    elif confidence == "high" or metrics.inventory_days is not None:
        final_confidence = "medium"
    else:
        final_confidence = "low"

    # Build details
    details.update({
        "chip_type": chip_type.value,
        "cycle_position": metrics.cycle_position.value,
        "inventory_days": metrics.inventory_days,
        "book_to_bill": metrics.book_to_bill,
        "inventory_to_sales": metrics.inventory_to_sales,
        "current_margin": current_margin,
        "normalized_margin": normalized_margin,
        "base_fair_value": base_fair_value,
        "tier_weights": SEMICONDUCTOR_TIER["weights"],
        "tier_parameters": SEMICONDUCTOR_TIER["parameters"],
    })

    logger.info(
        f"{symbol} - Semiconductor valuation: "
        f"chip_type={chip_type.value}, cycle={metrics.cycle_position.value}, "
        f"base=${base_fair_value:.2f}, adjustment={cycle_adjustment:.2f}x, "
        f"fair_value=${fair_value:.2f}"
    )

    return SemiconductorValuationResult(
        fair_value=fair_value,
        cycle_position=metrics.cycle_position,
        chip_type=chip_type,
        normalized_margin=normalized_margin,
        cycle_adjustment=cycle_adjustment,
        confidence=final_confidence,
        warnings=warnings,
        details=details,
    )


# ====================
# UTILITY FUNCTIONS
# ====================

def get_semiconductor_tier_weights() -> Dict[str, float]:
    """
    Get the valuation model weights for semiconductor tier.

    Returns:
        Dictionary of model weights (percentages summing to 100)
    """
    return SEMICONDUCTOR_TIER["weights"].copy()


def get_semiconductor_tier_parameters() -> Dict[str, float]:
    """
    Get the valuation parameters for semiconductor tier.

    Returns:
        Dictionary of parameters (terminal growth, margins, discount rate adjustment)
    """
    return SEMICONDUCTOR_TIER["parameters"].copy()


# ====================
# GROWTH PROFILE CLASSIFICATION
# ====================

def classify_growth_profile(
    revenue_growth: Optional[float] = None,
    earnings_growth: Optional[float] = None,
) -> GrowthProfile:
    """
    Classify company's growth profile for valuation purposes.

    Priority: Use revenue growth if available, then earnings growth.

    Args:
        revenue_growth: Year-over-year revenue growth rate (e.g., 0.50 = 50%)
        earnings_growth: Year-over-year earnings growth rate

    Returns:
        GrowthProfile enum value
    """
    # Use revenue growth as primary signal
    growth = revenue_growth if revenue_growth is not None else earnings_growth

    if growth is None:
        return GrowthProfile.UNKNOWN

    if growth > 0.50:
        return GrowthProfile.HYPER_GROWTH
    elif growth > 0.25:
        return GrowthProfile.HIGH_GROWTH
    elif growth > 0.10:
        return GrowthProfile.MODERATE_GROWTH
    elif growth > 0:
        return GrowthProfile.LOW_GROWTH
    else:
        return GrowthProfile.DECLINING


# ====================
# GROWTH-ADJUSTED VALUATION FUNCTIONS
# ====================

@dataclass
class GrowthAdjustedValuation:
    """Container for growth-adjusted fair value estimates."""
    # Individual fair value estimates
    cycle_normalized_fv: float          # Traditional cycle-normalized P/E
    peg_adjusted_fv: float              # PEG-based fair value
    forward_pe_fv: Optional[float]      # Forward P/E based (if available)
    ev_ebitda_fv: float                 # EV/EBITDA based

    # Blended result
    blended_fv: float                   # Weighted average of all methods
    confidence: str                     # high, medium, low

    # Metadata
    growth_profile: GrowthProfile
    applied_pe_multiple: float
    weights_used: Dict[str, float]
    details: Dict = field(default_factory=dict)


def calculate_peg_adjusted_fair_value(
    eps: float,
    revenue_growth: float,
    earnings_growth: Optional[float],
    growth_profile: GrowthProfile,
    chip_type: ChipType,
) -> Tuple[float, float, str]:
    """
    Calculate PEG-adjusted fair value.

    PEG = P/E / Growth Rate
    Fair P/E = Growth Rate × PEG Target

    For hyper-growth companies, allow premium PEG (up to 1.5).

    Args:
        eps: Earnings per share
        revenue_growth: Revenue growth rate (e.g., 0.50 = 50%)
        earnings_growth: Earnings growth rate (if available)
        growth_profile: Classified growth profile
        chip_type: Type of semiconductor

    Returns:
        Tuple of (fair_value, applied_pe, explanation)
    """
    if eps <= 0:
        return 0.0, 0.0, "PEG not applicable: negative or zero EPS"

    # Use earnings growth if available, otherwise derive from revenue growth
    growth_rate = earnings_growth if earnings_growth else revenue_growth
    if growth_rate is None or growth_rate <= 0:
        return 0.0, 0.0, "PEG not applicable: negative or zero growth"

    # Convert to percentage for PEG calculation (e.g., 50% growth = 50 for PEG)
    growth_pct = growth_rate * 100

    # Get growth-adjusted parameters
    params = GROWTH_ADJUSTED_PE_MULTIPLES.get(growth_profile, GROWTH_ADJUSTED_PE_MULTIPLES[GrowthProfile.UNKNOWN])
    peg_target = params.get("peg_target", 1.0)
    peg_premium = params.get("peg_premium", 0.0)
    max_pe = params.get("max_pe", 50.0)

    if peg_target is None:
        return 0.0, 0.0, "PEG not applicable for declining companies"

    # Calculate fair P/E based on PEG
    # Fair P/E = Growth Rate (%) × (PEG Target + Premium)
    peg_multiplier = peg_target + peg_premium
    fair_pe = growth_pct * peg_multiplier

    # Apply chip type premium/discount
    chip_premium = CHIP_TYPE_GROWTH_PREMIUM.get(chip_type, 1.0)
    fair_pe *= chip_premium

    # Cap at maximum P/E
    fair_pe = min(fair_pe, max_pe)

    # Calculate fair value
    fair_value = eps * fair_pe

    explanation = (
        f"PEG-adjusted: {growth_pct:.0f}% growth × {peg_multiplier:.1f} PEG "
        f"× {chip_premium:.1f} chip premium = {fair_pe:.1f}x P/E"
    )

    logger.info(f"PEG calculation: {explanation} -> ${fair_value:.2f}")

    return fair_value, fair_pe, explanation


def calculate_forward_pe_fair_value(
    forward_eps: float,
    growth_profile: GrowthProfile,
    chip_type: ChipType,
) -> Tuple[float, float, str]:
    """
    Calculate fair value using forward P/E.

    Forward P/E should be lower than trailing for growing companies
    because earnings are expected to grow.

    Args:
        forward_eps: Analyst consensus forward EPS estimate
        growth_profile: Classified growth profile
        chip_type: Type of semiconductor

    Returns:
        Tuple of (fair_value, forward_pe_target, explanation)
    """
    if forward_eps <= 0:
        return 0.0, 0.0, "Forward P/E not applicable: no positive forward EPS"

    # Get parameters
    params = GROWTH_ADJUSTED_PE_MULTIPLES.get(growth_profile, GROWTH_ADJUSTED_PE_MULTIPLES[GrowthProfile.UNKNOWN])
    base_pe = params.get("base_pe", 20.0)
    forward_discount = params.get("forward_pe_discount", 0.8)

    # Forward P/E target = Base P/E × Forward discount
    forward_pe_target = base_pe * forward_discount

    # Apply chip type adjustment
    chip_premium = CHIP_TYPE_GROWTH_PREMIUM.get(chip_type, 1.0)
    forward_pe_target *= chip_premium

    # Calculate fair value
    fair_value = forward_eps * forward_pe_target

    explanation = (
        f"Forward P/E: ${forward_eps:.2f} FY EPS × {forward_pe_target:.1f}x target P/E"
    )

    logger.info(f"Forward P/E calculation: {explanation} -> ${fair_value:.2f}")

    return fair_value, forward_pe_target, explanation


def calculate_growth_adjusted_valuation(
    symbol: str,
    financials: Dict,
    market_data: Dict,
    growth_profile: GrowthProfile,
    chip_type: ChipType,
    cycle_position: CyclePosition,
) -> GrowthAdjustedValuation:
    """
    Calculate comprehensive growth-adjusted valuation for semiconductor company.

    This function combines multiple valuation methods:
    1. Cycle-normalized P/E (traditional semiconductor approach)
    2. PEG-adjusted fair value (growth-adjusted)
    3. Forward P/E (if forward EPS available)
    4. EV/EBITDA based

    Then blends them with appropriate weights based on growth profile.

    Args:
        symbol: Stock ticker
        financials: Financial data (eps, revenue, ebitda, etc.)
        market_data: Market data (current_price, forward_eps, shares_outstanding, etc.)
        growth_profile: Classified growth profile
        chip_type: Type of semiconductor
        cycle_position: Current position in semiconductor cycle

    Returns:
        GrowthAdjustedValuation with all estimates and blended result
    """
    eps = financials.get("eps") or financials.get("eps_diluted", 0)
    revenue = financials.get("revenue", 0)
    ebitda = financials.get("ebitda", 0)
    revenue_growth = financials.get("revenue_growth", 0)
    earnings_growth = financials.get("earnings_growth")
    shares = market_data.get("shares_outstanding", 0)
    forward_eps = market_data.get("forward_eps", 0)

    details = {}

    # 1. Cycle-normalized P/E (conservative baseline)
    params = GROWTH_ADJUSTED_PE_MULTIPLES.get(GrowthProfile.LOW_GROWTH)  # Use conservative baseline
    sector_pe = 18.0  # Semiconductor sector average
    cycle_normalized_fv = eps * sector_pe if eps > 0 else 0
    details["cycle_normalized"] = {
        "eps": eps,
        "pe_multiple": sector_pe,
        "fair_value": cycle_normalized_fv,
    }

    # 2. PEG-adjusted fair value
    peg_fv, peg_pe, peg_explanation = calculate_peg_adjusted_fair_value(
        eps=eps,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        growth_profile=growth_profile,
        chip_type=chip_type,
    )
    details["peg_adjusted"] = {
        "fair_value": peg_fv,
        "applied_pe": peg_pe,
        "explanation": peg_explanation,
    }

    # 3. Forward P/E (if available)
    forward_fv = None
    if forward_eps and forward_eps > 0:
        forward_fv, forward_pe, forward_explanation = calculate_forward_pe_fair_value(
            forward_eps=forward_eps,
            growth_profile=growth_profile,
            chip_type=chip_type,
        )
        details["forward_pe"] = {
            "forward_eps": forward_eps,
            "fair_value": forward_fv,
            "applied_pe": forward_pe,
            "explanation": forward_explanation,
        }

    # 4. EV/EBITDA based
    ev_ebitda_fv = 0
    if ebitda > 0 and shares > 0:
        # Determine EV/EBITDA multiple based on growth
        ev_multiples = {
            GrowthProfile.HYPER_GROWTH: 35.0,
            GrowthProfile.HIGH_GROWTH: 25.0,
            GrowthProfile.MODERATE_GROWTH: 18.0,
            GrowthProfile.LOW_GROWTH: 12.0,
            GrowthProfile.DECLINING: 8.0,
            GrowthProfile.UNKNOWN: 15.0,
        }
        ev_multiple = ev_multiples.get(growth_profile, 15.0)
        enterprise_value = ebitda * ev_multiple
        ev_ebitda_fv = enterprise_value / shares
        details["ev_ebitda"] = {
            "ebitda": ebitda,
            "multiple": ev_multiple,
            "enterprise_value": enterprise_value,
            "fair_value": ev_ebitda_fv,
        }

    # Determine weights based on growth profile
    # For hyper-growth, weight more towards PEG and Forward P/E
    # For low-growth, weight more towards cycle-normalized
    weights = _get_blended_weights(growth_profile, forward_fv is not None)

    # Calculate blended fair value
    blended_fv = 0
    total_weight = 0

    if cycle_normalized_fv > 0 and weights.get("cycle_normalized", 0) > 0:
        blended_fv += cycle_normalized_fv * weights["cycle_normalized"]
        total_weight += weights["cycle_normalized"]

    if peg_fv > 0 and weights.get("peg_adjusted", 0) > 0:
        blended_fv += peg_fv * weights["peg_adjusted"]
        total_weight += weights["peg_adjusted"]

    if forward_fv and forward_fv > 0 and weights.get("forward_pe", 0) > 0:
        blended_fv += forward_fv * weights["forward_pe"]
        total_weight += weights["forward_pe"]

    if ev_ebitda_fv > 0 and weights.get("ev_ebitda", 0) > 0:
        blended_fv += ev_ebitda_fv * weights["ev_ebitda"]
        total_weight += weights["ev_ebitda"]

    if total_weight > 0:
        blended_fv /= total_weight
    else:
        # Fallback to simple average
        valid_fvs = [fv for fv in [cycle_normalized_fv, peg_fv, forward_fv, ev_ebitda_fv] if fv and fv > 0]
        blended_fv = sum(valid_fvs) / len(valid_fvs) if valid_fvs else 0

    # Determine confidence
    confidence = _determine_valuation_confidence(
        growth_profile=growth_profile,
        has_forward_eps=forward_eps > 0,
        cycle_position=cycle_position,
    )

    # Get the primary P/E used
    applied_pe = peg_pe if peg_pe > 0 else sector_pe

    forward_fv_str = f"${forward_fv:.2f}" if forward_fv else "N/A"
    logger.info(
        f"{symbol} - Growth-adjusted valuation: "
        f"growth_profile={growth_profile.value}, "
        f"cycle_normalized=${cycle_normalized_fv:.2f}, "
        f"peg_adjusted=${peg_fv:.2f}, "
        f"forward_pe={forward_fv_str}, "
        f"ev_ebitda=${ev_ebitda_fv:.2f}, "
        f"blended=${blended_fv:.2f}"
    )

    return GrowthAdjustedValuation(
        cycle_normalized_fv=cycle_normalized_fv,
        peg_adjusted_fv=peg_fv,
        forward_pe_fv=forward_fv,
        ev_ebitda_fv=ev_ebitda_fv,
        blended_fv=blended_fv,
        confidence=confidence,
        growth_profile=growth_profile,
        applied_pe_multiple=applied_pe,
        weights_used=weights,
        details=details,
    )


def _get_blended_weights(growth_profile: GrowthProfile, has_forward_eps: bool) -> Dict[str, float]:
    """
    Get blending weights for different valuation methods based on growth profile.

    For hyper-growth: Weight towards PEG and Forward P/E (growth-adjusted methods)
    For low-growth: Weight towards cycle-normalized (traditional method)

    Args:
        growth_profile: Company's growth profile
        has_forward_eps: Whether forward EPS estimates are available

    Returns:
        Dictionary of weights for each method
    """
    if has_forward_eps:
        # With forward EPS available, we can use forward P/E
        weights_matrix = {
            GrowthProfile.HYPER_GROWTH: {
                "cycle_normalized": 0.10,
                "peg_adjusted": 0.30,
                "forward_pe": 0.40,
                "ev_ebitda": 0.20,
            },
            GrowthProfile.HIGH_GROWTH: {
                "cycle_normalized": 0.15,
                "peg_adjusted": 0.30,
                "forward_pe": 0.35,
                "ev_ebitda": 0.20,
            },
            GrowthProfile.MODERATE_GROWTH: {
                "cycle_normalized": 0.25,
                "peg_adjusted": 0.25,
                "forward_pe": 0.30,
                "ev_ebitda": 0.20,
            },
            GrowthProfile.LOW_GROWTH: {
                "cycle_normalized": 0.35,
                "peg_adjusted": 0.15,
                "forward_pe": 0.25,
                "ev_ebitda": 0.25,
            },
            GrowthProfile.DECLINING: {
                "cycle_normalized": 0.40,
                "peg_adjusted": 0.0,
                "forward_pe": 0.30,
                "ev_ebitda": 0.30,
            },
            GrowthProfile.UNKNOWN: {
                "cycle_normalized": 0.30,
                "peg_adjusted": 0.20,
                "forward_pe": 0.25,
                "ev_ebitda": 0.25,
            },
        }
    else:
        # Without forward EPS, redistribute that weight
        weights_matrix = {
            GrowthProfile.HYPER_GROWTH: {
                "cycle_normalized": 0.15,
                "peg_adjusted": 0.55,
                "forward_pe": 0.0,
                "ev_ebitda": 0.30,
            },
            GrowthProfile.HIGH_GROWTH: {
                "cycle_normalized": 0.20,
                "peg_adjusted": 0.50,
                "forward_pe": 0.0,
                "ev_ebitda": 0.30,
            },
            GrowthProfile.MODERATE_GROWTH: {
                "cycle_normalized": 0.30,
                "peg_adjusted": 0.40,
                "forward_pe": 0.0,
                "ev_ebitda": 0.30,
            },
            GrowthProfile.LOW_GROWTH: {
                "cycle_normalized": 0.45,
                "peg_adjusted": 0.20,
                "forward_pe": 0.0,
                "ev_ebitda": 0.35,
            },
            GrowthProfile.DECLINING: {
                "cycle_normalized": 0.50,
                "peg_adjusted": 0.0,
                "forward_pe": 0.0,
                "ev_ebitda": 0.50,
            },
            GrowthProfile.UNKNOWN: {
                "cycle_normalized": 0.40,
                "peg_adjusted": 0.25,
                "forward_pe": 0.0,
                "ev_ebitda": 0.35,
            },
        }

    return weights_matrix.get(growth_profile, weights_matrix[GrowthProfile.UNKNOWN])


def _determine_valuation_confidence(
    growth_profile: GrowthProfile,
    has_forward_eps: bool,
    cycle_position: CyclePosition,
) -> str:
    """
    Determine confidence level in valuation estimate.

    Args:
        growth_profile: Company's growth profile
        has_forward_eps: Whether forward EPS available
        cycle_position: Position in semiconductor cycle

    Returns:
        Confidence level: "high", "medium", or "low"
    """
    confidence_score = 0

    # Growth profile clarity
    if growth_profile in [GrowthProfile.MODERATE_GROWTH, GrowthProfile.LOW_GROWTH]:
        confidence_score += 2  # Easier to value
    elif growth_profile in [GrowthProfile.HIGH_GROWTH, GrowthProfile.HYPER_GROWTH]:
        confidence_score += 1  # More uncertain
    elif growth_profile == GrowthProfile.UNKNOWN:
        confidence_score += 0

    # Forward EPS availability
    if has_forward_eps:
        confidence_score += 2

    # Cycle position clarity
    if cycle_position == CyclePosition.NORMAL:
        confidence_score += 2
    elif cycle_position in [CyclePosition.PEAK, CyclePosition.TROUGH]:
        confidence_score += 1  # Clearer but more adjustment needed
    elif cycle_position == CyclePosition.UNKNOWN:
        confidence_score += 0

    if confidence_score >= 5:
        return "high"
    elif confidence_score >= 3:
        return "medium"
    else:
        return "low"
