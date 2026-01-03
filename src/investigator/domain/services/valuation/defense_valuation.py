"""
Defense Contractor Valuation Module (P2-B)

Implements specialized valuation for defense contractors and government services companies.

Military/Government Technical companies have historically underperformed (0.063 avg reward)
because the standard valuation approach ignores:
1. Multi-year contract visibility
2. Backlog-driven revenue predictability
3. Cost-plus vs fixed-price contract mix

This module provides:
- Backlog extraction from XBRL data
- Defense contractor tier classification
- Backlog premium calculation
- Contract mix adjustment

Key insight: Defense contractors with high backlog-to-revenue ratios (>2-3x) have
superior visibility compared to typical industrials, warranting premium valuations.

Author: Claude Code
Date: 2025-12-29
Phase: P2-B (Defense Contractor Valuation Tier)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================
# DEFENSE CONTRACTOR CLASSIFICATION
# ====================

class DefenseContractorType(Enum):
    """Classification of defense contractor types."""
    PRIME_CONTRACTOR = "prime_contractor"  # Large primes (LMT, RTX, NOC, GD, BA)
    TIER_1_SUPPLIER = "tier_1_supplier"    # Major systems integrators
    TIER_2_SUPPLIER = "tier_2_supplier"    # Component/subsystem suppliers
    GOVERNMENT_SERVICES = "government_services"  # IT/services (SAIC, LDOS, BAH)
    AEROSPACE_DEFENSE = "aerospace_defense"  # Mixed commercial/defense
    UNKNOWN = "unknown"


# Industries that qualify for defense contractor valuation
DEFENSE_INDUSTRIES = [
    "Aerospace & Defense",
    "Defense",
    "Military/Government Technical",
    "Government Services",
    "Defense Primes",
    "Defense Electronics",
    "Aerospace/Defense Products & Services",
    "Defense & Aerospace",
]

# Known defense contractor symbols for explicit mapping
KNOWN_DEFENSE_CONTRACTORS: Dict[str, DefenseContractorType] = {
    # Prime Contractors
    "LMT": DefenseContractorType.PRIME_CONTRACTOR,   # Lockheed Martin
    "RTX": DefenseContractorType.PRIME_CONTRACTOR,   # Raytheon Technologies
    "NOC": DefenseContractorType.PRIME_CONTRACTOR,   # Northrop Grumman
    "GD": DefenseContractorType.PRIME_CONTRACTOR,    # General Dynamics
    "BA": DefenseContractorType.AEROSPACE_DEFENSE,   # Boeing (mixed commercial)
    "LHX": DefenseContractorType.PRIME_CONTRACTOR,   # L3Harris Technologies

    # Government Services / IT
    "SAIC": DefenseContractorType.GOVERNMENT_SERVICES,  # Science Applications
    "LDOS": DefenseContractorType.GOVERNMENT_SERVICES,  # Leidos
    "BAH": DefenseContractorType.GOVERNMENT_SERVICES,   # Booz Allen Hamilton
    "CACI": DefenseContractorType.GOVERNMENT_SERVICES,  # CACI International
    "MANT": DefenseContractorType.GOVERNMENT_SERVICES,  # ManTech International
    "PSN": DefenseContractorType.GOVERNMENT_SERVICES,   # Parsons Corporation
    "KBR": DefenseContractorType.GOVERNMENT_SERVICES,   # KBR Inc

    # Tier 1/2 Suppliers
    "HII": DefenseContractorType.TIER_1_SUPPLIER,    # Huntington Ingalls
    "TXT": DefenseContractorType.TIER_1_SUPPLIER,    # Textron
    "TDG": DefenseContractorType.TIER_1_SUPPLIER,    # TransDigm
    "HEI": DefenseContractorType.TIER_1_SUPPLIER,    # Heico
    "MOG.A": DefenseContractorType.TIER_2_SUPPLIER,  # Moog Inc
    "CW": DefenseContractorType.TIER_2_SUPPLIER,     # Curtiss-Wright
    "DCO": DefenseContractorType.TIER_2_SUPPLIER,    # Ducommun
    "KTOS": DefenseContractorType.TIER_2_SUPPLIER,   # Kratos Defense
    "MRCY": DefenseContractorType.TIER_2_SUPPLIER,   # Mercury Systems
    "AXON": DefenseContractorType.TIER_2_SUPPLIER,   # Axon Enterprise
}


# ====================
# DEFENSE CONTRACTOR VALUATION TIER (P2-B2)
# ====================

DEFENSE_CONTRACTOR_TIER = {
    "name": "defense_contractor",
    "description": "Specialized tier for defense contractors with backlog-driven valuation",
    "weights": {
        "ev_ebitda": 35,        # Primary multiple for defense (stable margins)
        "pe": 30,               # P/E important for mature defense contractors
        "dcf": 25,              # DCF with conservative growth assumptions
        "backlog_value": 10,    # NEW: Value backlog at discount
    },
    "parameters": {
        "terminal_growth": 0.025,   # Defense spending grows slowly but steadily
        "terminal_margin": 0.10,    # 10% operating margin typical for defense
        "discount_rate_adjustment": -0.005,  # Slightly lower risk premium (gov't customer)
        "backlog_discount_rate": 0.10,  # Discount rate for backlog valuation
    },
    "industry_matches": DEFENSE_INDUSTRIES,
}


# ====================
# BACKLOG EXTRACTION (P2-B1)
# ====================

@dataclass
class BacklogMetrics:
    """Container for defense contractor backlog metrics."""
    total_backlog: Optional[float] = None
    funded_backlog: Optional[float] = None
    unfunded_backlog: Optional[float] = None
    contract_liability: Optional[float] = None
    deferred_revenue: Optional[float] = None
    unbilled_receivables: Optional[float] = None
    contract_assets: Optional[float] = None
    annual_revenue: Optional[float] = None
    backlog_ratio: Optional[float] = None  # Backlog / Annual Revenue


def extract_backlog_metrics_from_xbrl(
    symbol: str,
    xbrl_data: Dict,
    annual_revenue: Optional[float] = None,
) -> BacklogMetrics:
    """
    Extract backlog-related metrics from XBRL data for defense contractors.

    Uses multiple XBRL tags to construct comprehensive backlog picture:
    - OrderBacklog (direct backlog disclosure)
    - ContractWithCustomerLiability (advances received)
    - DeferredRevenue (unearned revenue)
    - UnbilledContractsReceivable (work performed but not billed)
    - ContractWithCustomerAsset (contract assets)

    Args:
        symbol: Stock ticker symbol
        xbrl_data: Raw XBRL data dictionary (us-gaap format)
        annual_revenue: Annual revenue for ratio calculation (optional)

    Returns:
        BacklogMetrics with extracted values
    """
    from utils.xbrl_tag_aliases import XBRLTagAliasMapper

    mapper = XBRLTagAliasMapper()
    metrics = BacklogMetrics(annual_revenue=annual_revenue)

    # Define backlog-related canonical names to extract
    backlog_canonical_names = [
        'order_backlog',
        'funded_backlog',
        'unfunded_backlog',
        'contract_liability',
        'deferred_revenue_backlog',
        'unbilled_contracts_receivable',
        'contract_assets',
    ]

    us_gaap = xbrl_data.get('facts', {}).get('us-gaap', {})
    if not us_gaap:
        logger.warning(f"{symbol} - No us-gaap data available for backlog extraction")
        return metrics

    extracted_values = {}

    for canonical_name in backlog_canonical_names:
        aliases = mapper.get_xbrl_aliases(canonical_name)

        for alias in aliases:
            if alias in us_gaap:
                concept = us_gaap[alias]
                units = concept.get('units', {})
                usd_data = units.get('USD', [])

                if usd_data:
                    # Get the latest annual value (10-K preferred, then 10-Q)
                    sorted_data = sorted(
                        [d for d in usd_data if d.get('form') in ['10-K', '10-Q', '20-F']],
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
                                f"{symbol} - Extracted {canonical_name} from {alias}: "
                                f"${float(value)/1e9:.2f}B"
                            )
                            break  # Found value, move to next canonical name

    # Map extracted values to BacklogMetrics fields
    metrics.total_backlog = extracted_values.get('order_backlog')
    metrics.funded_backlog = extracted_values.get('funded_backlog')
    metrics.unfunded_backlog = extracted_values.get('unfunded_backlog')
    metrics.contract_liability = extracted_values.get('contract_liability')
    metrics.deferred_revenue = extracted_values.get('deferred_revenue_backlog')
    metrics.unbilled_receivables = extracted_values.get('unbilled_contracts_receivable')
    metrics.contract_assets = extracted_values.get('contract_assets')

    # If no direct backlog available, estimate from contract liability + deferred revenue
    if metrics.total_backlog is None:
        estimated_backlog = 0
        if metrics.contract_liability:
            estimated_backlog += metrics.contract_liability
        if metrics.deferred_revenue:
            estimated_backlog += metrics.deferred_revenue
        if metrics.unbilled_receivables:
            estimated_backlog += metrics.unbilled_receivables

        if estimated_backlog > 0:
            metrics.total_backlog = estimated_backlog
            logger.info(
                f"{symbol} - Estimated backlog from components: ${estimated_backlog/1e9:.2f}B "
                "(contract_liability + deferred_revenue + unbilled)"
            )

    # Calculate backlog ratio if revenue available
    if metrics.total_backlog and metrics.annual_revenue and metrics.annual_revenue > 0:
        metrics.backlog_ratio = metrics.total_backlog / metrics.annual_revenue
        logger.info(
            f"{symbol} - Backlog ratio: {metrics.backlog_ratio:.2f}x "
            f"(${metrics.total_backlog/1e9:.2f}B backlog / ${metrics.annual_revenue/1e9:.2f}B revenue)"
        )

    # Log extraction summary
    found_metrics = [k for k, v in extracted_values.items() if v is not None]
    logger.info(
        f"{symbol} - Backlog extraction complete: "
        f"Found {len(found_metrics)}/{len(backlog_canonical_names)} metrics. "
        f"Total backlog: ${(metrics.total_backlog or 0)/1e9:.2f}B"
    )

    return metrics


# ====================
# BACKLOG PREMIUM CALCULATION (P2-B3)
# ====================

def calculate_backlog_premium(backlog: float, annual_revenue: float) -> float:
    """
    Calculate valuation premium based on backlog coverage.

    Defense contractors with high backlog-to-revenue ratios have superior
    visibility into future revenue, warranting premium valuations.

    Thresholds:
    - > 3.0x: 10% premium (excellent visibility, 3+ years of revenue)
    - > 2.0x: 5% premium (good visibility, 2+ years of revenue)
    - < 1.0x: 5% discount (poor visibility, less than 1 year of revenue)

    Args:
        backlog: Total backlog value
        annual_revenue: Annual revenue

    Returns:
        Premium multiplier (e.g., 1.10 for 10% premium, 0.95 for 5% discount)

    Example:
        >>> calculate_backlog_premium(150_000_000_000, 50_000_000_000)
        1.10  # 3x backlog ratio -> 10% premium
    """
    if annual_revenue <= 0:
        logger.warning("Cannot calculate backlog premium: annual_revenue <= 0")
        return 1.0

    backlog_ratio = backlog / annual_revenue

    if backlog_ratio >= 3.0:
        premium = 1.10  # 10% premium - excellent visibility
        quality = "excellent"
    elif backlog_ratio >= 2.0:
        premium = 1.05  # 5% premium - good visibility
        quality = "good"
    elif backlog_ratio < 1.0:
        premium = 0.95  # 5% discount - poor visibility
        quality = "weak"
    else:
        premium = 1.0  # No adjustment
        quality = "average"

    logger.info(
        f"Backlog premium calculated: {premium:.2f}x "
        f"(ratio={backlog_ratio:.2f}x, quality={quality})"
    )

    return premium


def calculate_backlog_value(
    backlog: float,
    annual_revenue: float,
    operating_margin: float = 0.10,
    discount_rate: float = 0.10,
    years_to_recognize: int = 3,
) -> float:
    """
    Calculate present value of backlog as a supplementary valuation metric.

    This values the backlog as a stream of future cash flows:
    - Assume backlog is recognized as revenue over N years
    - Apply operating margin to get operating income
    - Discount to present value

    Args:
        backlog: Total backlog value
        annual_revenue: Annual revenue (for backlog ratio)
        operating_margin: Expected operating margin on backlog (default 10%)
        discount_rate: Discount rate for NPV calculation (default 10%)
        years_to_recognize: Years over which backlog converts to revenue (default 3)

    Returns:
        Present value of backlog (per share if divided by shares outstanding)

    Example:
        >>> # $150B backlog, 10% margin, recognized over 3 years
        >>> pv = calculate_backlog_value(150_000_000_000, 50_000_000_000)
        >>> print(f"Backlog PV: ${pv/1e9:.2f}B")
    """
    if backlog <= 0 or years_to_recognize <= 0:
        return 0.0

    # Assume backlog converts to revenue evenly over the recognition period
    annual_backlog_revenue = backlog / years_to_recognize

    # Calculate operating income from backlog
    annual_backlog_profit = annual_backlog_revenue * operating_margin

    # Calculate NPV of backlog profits
    npv = 0.0
    for year in range(1, years_to_recognize + 1):
        discount_factor = 1 / ((1 + discount_rate) ** year)
        npv += annual_backlog_profit * discount_factor

    logger.debug(
        f"Backlog value calculation: backlog=${backlog/1e9:.2f}B, "
        f"margin={operating_margin:.1%}, years={years_to_recognize}, "
        f"NPV=${npv/1e9:.2f}B"
    )

    return npv


# ====================
# CONTRACT MIX ADJUSTMENT (P2-B4)
# ====================

def calculate_contract_mix_adjustment(cost_plus_pct: float) -> float:
    """
    Adjust valuation based on contract type mix.

    Cost-plus contracts: Lower margin but lower risk (cost reimbursement)
    Fixed-price contracts: Higher margin but higher risk (cost overruns)

    Adjustments:
    - > 70% cost-plus: 5% discount (lower margin but safer)
    - < 30% cost-plus: 5% premium (higher margin potential)
    - 30-70%: No adjustment (balanced mix)

    Args:
        cost_plus_pct: Percentage of revenue from cost-plus contracts (0.0-1.0)

    Returns:
        Adjustment multiplier (e.g., 0.95 for 5% discount, 1.05 for 5% premium)

    Example:
        >>> calculate_contract_mix_adjustment(0.80)
        0.95  # 80% cost-plus -> 5% discount for lower margins
    """
    if cost_plus_pct > 0.70:
        adjustment = 0.95  # 5% discount - lower margin business
        mix_type = "cost-plus heavy"
    elif cost_plus_pct < 0.30:
        adjustment = 1.05  # 5% premium - higher margin potential
        mix_type = "fixed-price heavy"
    else:
        adjustment = 1.0
        mix_type = "balanced"

    logger.info(
        f"Contract mix adjustment: {adjustment:.2f}x "
        f"(cost-plus={cost_plus_pct:.0%}, mix={mix_type})"
    )

    return adjustment


# ====================
# DEFENSE CONTRACTOR CLASSIFICATION
# ====================

@dataclass
class DefenseContractorClassification:
    """Result from defense contractor classification."""
    is_defense_contractor: bool
    contractor_type: DefenseContractorType
    confidence: str  # "high", "medium", "low"
    detection_method: str  # "symbol_lookup", "industry_match", "default"
    industry: Optional[str] = None


def classify_defense_contractor(
    symbol: str,
    industry: Optional[str] = None,
    sector: Optional[str] = None,
) -> DefenseContractorClassification:
    """
    Classify whether a company is a defense contractor and determine type.

    Priority:
    1. Known symbol mapping (highest confidence)
    2. Industry string matching
    3. Default to non-defense

    Args:
        symbol: Stock ticker symbol
        industry: Industry classification string
        sector: Sector classification string

    Returns:
        DefenseContractorClassification with type and confidence
    """
    symbol_upper = symbol.upper()

    # Priority 1: Check known defense contractor mappings
    if symbol_upper in KNOWN_DEFENSE_CONTRACTORS:
        contractor_type = KNOWN_DEFENSE_CONTRACTORS[symbol_upper]
        logger.info(
            f"{symbol} - Defense contractor detected via symbol mapping: {contractor_type.value}"
        )
        return DefenseContractorClassification(
            is_defense_contractor=True,
            contractor_type=contractor_type,
            confidence="high",
            detection_method="symbol_lookup",
            industry=industry,
        )

    # Priority 2: Industry string matching
    if industry:
        industry_lower = industry.lower()

        for defense_industry in DEFENSE_INDUSTRIES:
            if defense_industry.lower() in industry_lower or industry_lower in defense_industry.lower():
                logger.info(
                    f"{symbol} - Defense contractor detected via industry match: '{industry}'"
                )
                return DefenseContractorClassification(
                    is_defense_contractor=True,
                    contractor_type=DefenseContractorType.UNKNOWN,
                    confidence="medium",
                    detection_method="industry_match",
                    industry=industry,
                )

        # Check for defense-related keywords
        defense_keywords = ["defense", "military", "government", "aerospace"]
        for keyword in defense_keywords:
            if keyword in industry_lower:
                logger.info(
                    f"{symbol} - Possible defense contractor (keyword '{keyword}' in industry)"
                )
                return DefenseContractorClassification(
                    is_defense_contractor=True,
                    contractor_type=DefenseContractorType.UNKNOWN,
                    confidence="low",
                    detection_method="industry_keyword",
                    industry=industry,
                )

    # Priority 3: Not a defense contractor
    return DefenseContractorClassification(
        is_defense_contractor=False,
        contractor_type=DefenseContractorType.UNKNOWN,
        confidence="high",
        detection_method="default",
        industry=industry,
    )


# ====================
# DEFENSE CONTRACTOR VALUATION
# ====================

@dataclass
class DefenseValuationResult:
    """Result from defense contractor valuation."""
    fair_value: float
    base_fair_value: float  # Before backlog adjustments
    backlog_premium: float
    contract_mix_adjustment: float
    total_backlog: Optional[float]
    backlog_ratio: Optional[float]
    backlog_value: Optional[float]  # NPV of backlog
    contractor_type: DefenseContractorType
    confidence: str
    warnings: List[str]
    details: Dict


def value_defense_contractor(
    symbol: str,
    financials: Dict,
    current_price: float,
    base_fair_value: float,
    xbrl_data: Optional[Dict] = None,
    industry: Optional[str] = None,
    cost_plus_pct: Optional[float] = None,
) -> DefenseValuationResult:
    """
    Apply defense contractor-specific valuation adjustments.

    This function takes a base fair value (from EV/EBITDA, P/E, DCF blend)
    and applies defense-specific adjustments:
    1. Backlog premium/discount based on backlog-to-revenue ratio
    2. Contract mix adjustment based on cost-plus vs fixed-price mix
    3. Backlog value as supplementary metric

    Args:
        symbol: Stock ticker symbol
        financials: Dictionary of financial metrics
        current_price: Current stock price
        base_fair_value: Fair value from base valuation models
        xbrl_data: Optional raw XBRL data for backlog extraction
        industry: Industry classification string
        cost_plus_pct: Percentage of cost-plus contracts (0.0-1.0)

    Returns:
        DefenseValuationResult with adjusted fair value and details
    """
    warnings = []
    details = {}

    # Classify the defense contractor
    classification = classify_defense_contractor(symbol, industry)
    contractor_type = classification.contractor_type

    if not classification.is_defense_contractor:
        logger.warning(
            f"{symbol} - Not classified as defense contractor, returning base valuation"
        )
        return DefenseValuationResult(
            fair_value=base_fair_value,
            base_fair_value=base_fair_value,
            backlog_premium=1.0,
            contract_mix_adjustment=1.0,
            total_backlog=None,
            backlog_ratio=None,
            backlog_value=None,
            contractor_type=contractor_type,
            confidence="low",
            warnings=["Company not classified as defense contractor"],
            details={"classification": classification.__dict__},
        )

    # Get annual revenue
    annual_revenue = (
        financials.get('total_revenue') or
        financials.get('revenue') or
        financials.get('revenues') or
        0
    )

    # Extract backlog metrics
    backlog_metrics = None
    total_backlog = None
    backlog_ratio = None
    backlog_value = None

    if xbrl_data:
        backlog_metrics = extract_backlog_metrics_from_xbrl(
            symbol, xbrl_data, annual_revenue
        )
        total_backlog = backlog_metrics.total_backlog
        backlog_ratio = backlog_metrics.backlog_ratio

        # Calculate NPV of backlog
        if total_backlog and annual_revenue > 0:
            operating_margin = DEFENSE_CONTRACTOR_TIER["parameters"]["terminal_margin"]
            discount_rate = DEFENSE_CONTRACTOR_TIER["parameters"]["backlog_discount_rate"]
            backlog_value = calculate_backlog_value(
                total_backlog, annual_revenue, operating_margin, discount_rate
            )
    else:
        warnings.append("No XBRL data provided - backlog metrics not available")

    # Calculate backlog premium
    backlog_premium = 1.0
    if total_backlog and annual_revenue > 0:
        backlog_premium = calculate_backlog_premium(total_backlog, annual_revenue)
    else:
        warnings.append("Backlog data not available - no backlog premium applied")

    # Calculate contract mix adjustment
    contract_mix_adj = 1.0
    if cost_plus_pct is not None:
        contract_mix_adj = calculate_contract_mix_adjustment(cost_plus_pct)
        details["cost_plus_pct"] = cost_plus_pct
    else:
        warnings.append("Contract mix data not available - no mix adjustment applied")

    # Apply adjustments to base fair value
    adjusted_fair_value = base_fair_value * backlog_premium * contract_mix_adj

    # Determine confidence
    if classification.confidence == "high" and backlog_metrics and total_backlog:
        confidence = "high"
    elif classification.confidence == "high":
        confidence = "medium"
    else:
        confidence = "low"

    # Build details
    details.update({
        "classification": classification.__dict__,
        "backlog_metrics": backlog_metrics.__dict__ if backlog_metrics else None,
        "tier_weights": DEFENSE_CONTRACTOR_TIER["weights"],
        "tier_parameters": DEFENSE_CONTRACTOR_TIER["parameters"],
    })

    logger.info(
        f"{symbol} - Defense contractor valuation: "
        f"base=${base_fair_value:.2f}, backlog_premium={backlog_premium:.2f}x, "
        f"contract_mix={contract_mix_adj:.2f}x, adjusted=${adjusted_fair_value:.2f}"
    )

    return DefenseValuationResult(
        fair_value=adjusted_fair_value,
        base_fair_value=base_fair_value,
        backlog_premium=backlog_premium,
        contract_mix_adjustment=contract_mix_adj,
        total_backlog=total_backlog,
        backlog_ratio=backlog_ratio,
        backlog_value=backlog_value,
        contractor_type=contractor_type,
        confidence=confidence,
        warnings=warnings,
        details=details,
    )


def get_defense_tier_weights() -> Dict[str, float]:
    """
    Get the valuation model weights for defense contractor tier.

    Returns:
        Dictionary of model weights (percentages summing to 100)
    """
    return DEFENSE_CONTRACTOR_TIER["weights"].copy()


def get_defense_tier_parameters() -> Dict[str, float]:
    """
    Get the valuation parameters for defense contractor tier.

    Returns:
        Dictionary of parameters (terminal growth, margin, discount rate adjustment)
    """
    return DEFENSE_CONTRACTOR_TIER["parameters"].copy()


def is_defense_industry(industry: Optional[str]) -> bool:
    """
    Check if an industry string matches defense contractor industries.

    Args:
        industry: Industry classification string

    Returns:
        True if industry matches defense contractor patterns
    """
    if not industry:
        return False

    industry_lower = industry.lower()
    for defense_industry in DEFENSE_INDUSTRIES:
        if defense_industry.lower() in industry_lower:
            return True

    # Check keywords
    defense_keywords = ["defense", "military", "government services"]
    for keyword in defense_keywords:
        if keyword in industry_lower:
            return True

    return False
