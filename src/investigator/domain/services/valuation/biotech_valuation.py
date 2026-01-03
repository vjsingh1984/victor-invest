"""
Biotech Pre-Revenue Valuation Model (P2-A)

Implements probability-weighted pipeline valuation for pre-revenue biotech companies.

Pre-revenue biotech companies cannot be valued using traditional metrics:
- P/S ratio is meaningless when revenue is $0
- P/E ratio doesn't apply (no earnings)
- DCF is unreliable (highly speculative cash flows)

Instead, biotech valuation is based on:
1. Pipeline Value: Probability-weighted expected value of drug candidates
2. Cash Runway: How long before the company needs to raise capital
3. Comparable Deals: M&A transactions in similar therapeutic areas

Methodology:
- Each drug in pipeline assigned probability based on clinical phase
- Peak sales estimate multiplied by success probability
- Market size discount applied for conservative TAM estimates
- Cash runway analysis flags dilution/financing risk

Author: Claude Code
Date: 2025-12-29
Phase: P2-A (Biotech Pipeline Valuation Model)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================
# P2-A1: PHASE SUCCESS PROBABILITY WEIGHTS
# ====================

class DrugPhase(Enum):
    """Drug development phases with associated approval probabilities."""
    PRECLINICAL = "preclinical"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    FILED_NDA = "filed_nda"  # NDA/BLA filed with FDA
    APPROVED = "approved"


# Historical phase transition and approval probabilities
# Source: Industry analysis (BIO, FDA, PhRMA data)
# These represent probability of eventual FDA approval from each phase
PHASE_SUCCESS_PROBABILITIES: Dict[str, float] = {
    "preclinical": 0.05,   # 5% of preclinical candidates reach approval
    "phase_1": 0.10,       # 10% from Phase 1
    "phase_2": 0.15,       # 15% from Phase 2 (most drugs fail here)
    "phase_3": 0.50,       # 50% from Phase 3 (pivotal trials)
    "filed_nda": 0.85,     # 85% with NDA filed get approved
    "approved": 1.00,      # 100% if already approved
}

# Phase transition probabilities (for reference)
# Not used directly but helpful for understanding the model
PHASE_TRANSITION_RATES: Dict[str, float] = {
    "preclinical_to_phase_1": 0.50,  # 50% advance to Phase 1
    "phase_1_to_phase_2": 0.65,      # 65% of Phase 1 advance
    "phase_2_to_phase_3": 0.30,      # 30% of Phase 2 advance (biggest drop)
    "phase_3_to_filed": 0.70,        # 70% of Phase 3 file NDA
    "filed_to_approved": 0.90,       # 90% of NDAs approved
}


# ====================
# P2-A2: CASH RUNWAY CALCULATION
# ====================

class CashRunwayRisk(Enum):
    """Cash runway risk classification."""
    LOW = "low"        # > 24 months
    MEDIUM = "medium"  # 18-24 months
    HIGH = "high"      # < 18 months
    CRITICAL = "critical"  # < 12 months


@dataclass
class CashRunwayResult:
    """Result from cash runway analysis."""
    months: float
    risk: CashRunwayRisk
    risk_description: str
    dilution_warning: bool
    details: Dict


def calculate_cash_runway(
    cash: float,
    quarterly_burn: float,
    include_receivables: bool = False,
    receivables: float = 0,
) -> CashRunwayResult:
    """
    Calculate months of cash runway and assess financing risk.

    Cash runway indicates how long a company can operate before needing
    additional financing (equity raise, debt, partnership). Short runways
    indicate high dilution risk for shareholders.

    Risk thresholds:
    - > 24 months: Low risk (adequate runway)
    - 18-24 months: Medium risk (may need to raise within 18 months)
    - 12-18 months: High risk (likely to raise soon, dilution expected)
    - < 12 months: Critical risk (imminent financing needed)

    Args:
        cash: Total cash and cash equivalents (in dollars)
        quarterly_burn: Cash burn rate per quarter (positive = burning cash)
        include_receivables: Whether to include receivables in liquidity
        receivables: Accounts receivable (optional)

    Returns:
        CashRunwayResult with runway months and risk assessment
    """
    # Handle edge cases
    if quarterly_burn <= 0:
        # Company is cash flow positive or break-even
        logger.info("Cash flow positive or break-even - unlimited runway")
        return CashRunwayResult(
            months=float('inf'),
            risk=CashRunwayRisk.LOW,
            risk_description="Cash flow positive - no financing risk",
            dilution_warning=False,
            details={
                "cash": cash,
                "quarterly_burn": quarterly_burn,
                "monthly_burn": 0,
                "years_runway": float('inf'),
            }
        )

    # Calculate total liquidity
    total_liquidity = cash
    if include_receivables:
        total_liquidity += receivables

    # Calculate runway in months
    monthly_burn = quarterly_burn / 3
    months = total_liquidity / monthly_burn
    years = months / 12

    # Determine risk level
    if months >= 24:
        risk = CashRunwayRisk.LOW
        risk_description = f"Adequate runway ({months:.1f} months / {years:.1f} years)"
        dilution_warning = False
    elif months >= 18:
        risk = CashRunwayRisk.MEDIUM
        risk_description = f"Monitor runway ({months:.1f} months) - may need financing within 18 months"
        dilution_warning = False
    elif months >= 12:
        risk = CashRunwayRisk.HIGH
        risk_description = f"Short runway ({months:.1f} months) - financing likely needed soon"
        dilution_warning = True
    else:
        risk = CashRunwayRisk.CRITICAL
        risk_description = f"Critical runway ({months:.1f} months) - imminent financing required"
        dilution_warning = True

    logger.info(
        f"Cash runway: ${total_liquidity/1e6:.1f}M / ${monthly_burn/1e6:.1f}M/month = "
        f"{months:.1f} months ({risk.value} risk)"
    )

    return CashRunwayResult(
        months=months,
        risk=risk,
        risk_description=risk_description,
        dilution_warning=dilution_warning,
        details={
            "cash": cash,
            "receivables": receivables if include_receivables else None,
            "total_liquidity": total_liquidity,
            "quarterly_burn": quarterly_burn,
            "monthly_burn": monthly_burn,
            "years_runway": years,
        }
    )


# ====================
# P2-A3: PIPELINE PROBABILITY-WEIGHTED VALUATION
# ====================

# Biotech pre-revenue valuation tier weights
# These sum to 100 for easy percentage interpretation
BIOTECH_PRE_REVENUE_TIER = {
    "name": "biotech_pre_revenue",
    "weights": {
        "pipeline_value": 60,      # Probability-weighted pipeline (primary driver)
        "cash_runway": 25,         # Cash / burn rate (financing risk)
        "comparable_deals": 15,    # Industry benchmark comparables (implemented below)
        "ps": 0,                   # Explicitly exclude P/S (meaningless at $0 revenue)
        "pe": 0,                   # Explicitly exclude P/E (no earnings)
        "dcf": 0,                  # Explicitly exclude DCF (too speculative)
    },
    "parameters": {
        "market_size_discount": 0.30,      # 30% discount to TAM estimates (conservative)
        "cash_runway_min_months": 18,      # Flag if < 18 months runway
        "peak_sales_multiple": 3.0,        # NPV multiple for approved drugs
        "development_time_years": {        # Expected time to approval by phase
            "preclinical": 8,
            "phase_1": 6,
            "phase_2": 4,
            "phase_3": 2,
            "filed_nda": 1,
            "approved": 0,
        },
        "discount_rate": 0.15,             # 15% discount rate for biotech
    },
}


# ====================
# P2-A4: INDUSTRY BENCHMARK COMPARABLES
# ====================

# Therapeutic area valuation benchmarks
# Based on historical M&A deals and public company valuations (2020-2024)
# Source: BioPharma Dive, Evaluate Pharma, company filings
THERAPEUTIC_AREA_BENCHMARKS = {
    "oncology": {
        "avg_deal_premium": 1.40,  # 40% premium to pipeline value (hot area)
        "phase_2_ev_range": (500_000_000, 3_000_000_000),  # $500M - $3B
        "phase_3_ev_range": (1_500_000_000, 10_000_000_000),  # $1.5B - $10B
        "peak_sales_multiple": 4.0,  # Higher for oncology
        "notes": "Oncology commands premium due to unmet need and pricing power",
    },
    "rare_disease": {
        "avg_deal_premium": 1.35,  # 35% premium (orphan drug pricing)
        "phase_2_ev_range": (400_000_000, 2_500_000_000),
        "phase_3_ev_range": (1_200_000_000, 8_000_000_000),
        "peak_sales_multiple": 5.0,  # Highest due to orphan drug pricing
        "notes": "Rare disease valued higher due to orphan drug exclusivity and pricing",
    },
    "immunology": {
        "avg_deal_premium": 1.25,  # 25% premium
        "phase_2_ev_range": (400_000_000, 2_000_000_000),
        "phase_3_ev_range": (1_000_000_000, 7_000_000_000),
        "peak_sales_multiple": 3.5,
        "notes": "Large patient populations but competitive market",
    },
    "cns": {  # Central Nervous System
        "avg_deal_premium": 1.15,  # 15% premium (higher failure rate)
        "phase_2_ev_range": (300_000_000, 1_500_000_000),
        "phase_3_ev_range": (800_000_000, 5_000_000_000),
        "peak_sales_multiple": 3.0,
        "notes": "CNS drugs have higher failure rates but large markets",
    },
    "cardiovascular": {
        "avg_deal_premium": 1.10,  # 10% premium
        "phase_2_ev_range": (300_000_000, 1_500_000_000),
        "phase_3_ev_range": (700_000_000, 4_000_000_000),
        "peak_sales_multiple": 2.5,
        "notes": "Competitive market with generic pressure",
    },
    "infectious_disease": {
        "avg_deal_premium": 1.20,  # 20% premium (post-COVID awareness)
        "phase_2_ev_range": (250_000_000, 1_200_000_000),
        "phase_3_ev_range": (600_000_000, 3_500_000_000),
        "peak_sales_multiple": 2.5,
        "notes": "Cyclical interest, pandemic-related volatility",
    },
    "gene_therapy": {
        "avg_deal_premium": 1.50,  # 50% premium (cutting-edge)
        "phase_2_ev_range": (600_000_000, 4_000_000_000),
        "phase_3_ev_range": (2_000_000_000, 15_000_000_000),
        "peak_sales_multiple": 5.0,
        "notes": "Premium for platform technology and one-time cures",
    },
    "cell_therapy": {
        "avg_deal_premium": 1.45,  # 45% premium
        "phase_2_ev_range": (500_000_000, 3_500_000_000),
        "phase_3_ev_range": (1_500_000_000, 12_000_000_000),
        "peak_sales_multiple": 4.5,
        "notes": "CAR-T and similar therapies command high premiums",
    },
    "default": {
        "avg_deal_premium": 1.15,  # 15% baseline
        "phase_2_ev_range": (300_000_000, 1_500_000_000),
        "phase_3_ev_range": (700_000_000, 4_000_000_000),
        "peak_sales_multiple": 3.0,
        "notes": "Default for unclassified therapeutic areas",
    },
}

# Therapeutic area keywords for classification
THERAPEUTIC_AREA_KEYWORDS = {
    "oncology": ["oncol", "cancer", "tumor", "lymphoma", "leukemia", "melanoma", "carcinoma"],
    "rare_disease": ["rare", "orphan", "genetic disorder", "lysosomal", "enzyme replacement"],
    "immunology": ["immun", "autoimmun", "inflamm", "arthritis", "lupus", "crohn"],
    "cns": ["neuro", "alzheim", "parkinson", "epilep", "schizo", "depress", "anxiety", "brain"],
    "cardiovascular": ["cardio", "heart", "vascular", "hypertension", "cholesterol", "lipid"],
    "infectious_disease": ["infect", "antiviral", "antibact", "antifung", "vaccine", "hiv", "hepatitis"],
    "gene_therapy": ["gene therapy", "aav", "lentivir", "crispr", "gene editing", "genetic medicine"],
    "cell_therapy": ["cell therapy", "car-t", "cart", "stem cell", "cellular therapy", "adoptive"],
}


def classify_therapeutic_area(
    indication: Optional[str] = None,
    company_name: Optional[str] = None,
    pipeline: Optional[List[Dict]] = None,
) -> str:
    """
    Classify therapeutic area from indication, company name, or pipeline.

    Args:
        indication: Primary indication (e.g., "non-small cell lung cancer")
        company_name: Company name
        pipeline: List of drug candidates with indications

    Returns:
        Therapeutic area classification
    """
    # Combine all text for keyword matching
    text_sources = []
    if indication:
        text_sources.append(indication.lower())
    if company_name:
        text_sources.append(company_name.lower())
    if pipeline:
        for drug in pipeline:
            if drug.get("indication"):
                text_sources.append(drug["indication"].lower())

    combined_text = " ".join(text_sources)

    # Check each therapeutic area
    for area, keywords in THERAPEUTIC_AREA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                return area

    return "default"


@dataclass
class ComparablesBenchmark:
    """Result from industry comparable analysis."""
    therapeutic_area: str
    ev_benchmark_low: float
    ev_benchmark_high: float
    deal_premium: float
    implied_fair_value: float
    confidence: str
    notes: str


def calculate_comparables_benchmark(
    pipeline: Optional[List[Dict]] = None,
    cash: float = 0,
    shares_outstanding: float = 1,
    indication: Optional[str] = None,
    company_name: Optional[str] = None,
) -> ComparablesBenchmark:
    """
    Calculate fair value using industry benchmark comparables.

    This addresses the "comparable deals NOT IMPLEMENTED" limitation
    by using therapeutic area benchmarks based on historical M&A data.

    Args:
        pipeline: List of drug candidates
        cash: Cash position
        shares_outstanding: Shares outstanding
        indication: Primary indication
        company_name: Company name

    Returns:
        ComparablesBenchmark with implied fair value
    """
    # Classify therapeutic area
    therapeutic_area = classify_therapeutic_area(indication, company_name, pipeline)
    benchmarks = THERAPEUTIC_AREA_BENCHMARKS.get(
        therapeutic_area, THERAPEUTIC_AREA_BENCHMARKS["default"]
    )

    # Determine most advanced phase in pipeline
    most_advanced_phase = "preclinical"
    if pipeline:
        phase_order = ["preclinical", "phase_1", "phase_2", "phase_3", "filed_nda", "approved"]
        for drug in pipeline:
            phase = drug.get("phase", "preclinical").lower().replace(" ", "_")
            if phase in phase_order:
                if phase_order.index(phase) > phase_order.index(most_advanced_phase):
                    most_advanced_phase = phase

    # Get EV range based on phase
    if most_advanced_phase in ["phase_3", "filed_nda", "approved"]:
        ev_low, ev_high = benchmarks["phase_3_ev_range"]
        confidence = "medium"
    elif most_advanced_phase == "phase_2":
        ev_low, ev_high = benchmarks["phase_2_ev_range"]
        confidence = "low"
    else:
        # Early stage - use phase 2 range discounted
        base_low, base_high = benchmarks["phase_2_ev_range"]
        ev_low = base_low * 0.25  # 25% of phase 2
        ev_high = base_high * 0.40  # 40% of phase 2
        confidence = "very_low"

    # Apply deal premium
    deal_premium = benchmarks["avg_deal_premium"]
    ev_mid = ((ev_low + ev_high) / 2) * deal_premium

    # Add cash
    total_value = ev_mid + cash

    # Calculate per-share
    fair_value_per_share = total_value / shares_outstanding if shares_outstanding > 0 else 0

    logger.info(
        f"Comparable benchmark: area={therapeutic_area}, phase={most_advanced_phase}, "
        f"EV range=${ev_low/1e9:.1f}B-${ev_high/1e9:.1f}B, "
        f"premium={deal_premium:.0%}, fair value=${fair_value_per_share:.2f}"
    )

    return ComparablesBenchmark(
        therapeutic_area=therapeutic_area,
        ev_benchmark_low=ev_low,
        ev_benchmark_high=ev_high,
        deal_premium=deal_premium,
        implied_fair_value=fair_value_per_share,
        confidence=confidence,
        notes=benchmarks["notes"],
    )


@dataclass
class DrugCandidate:
    """Representation of a drug in the pipeline."""
    name: str
    phase: str  # preclinical, phase_1, phase_2, phase_3, filed_nda, approved
    indication: str
    estimated_peak_sales: float  # Estimated annual peak sales if approved
    probability_override: Optional[float] = None  # Override default probability
    launch_year: Optional[int] = None  # Expected launch year
    notes: Optional[str] = None


@dataclass
class PipelineValuationResult:
    """Result from pipeline probability-weighted valuation."""
    total_pipeline_value: float
    drug_values: List[Dict]
    probability_weighted_sales: float
    market_discount_applied: float
    methodology: str
    warnings: List[str]


def calculate_pipeline_value(
    pipeline: List[Dict],
    market_discount: float = 0.70,
    npv_multiple: float = 3.0,
) -> PipelineValuationResult:
    """
    Calculate probability-weighted pipeline valuation.

    For each drug candidate:
    1. Get approval probability based on phase
    2. Multiply peak sales by probability
    3. Apply market discount for conservative TAM
    4. Sum across all drugs

    Args:
        pipeline: List of drug dictionaries with:
            - name: Drug name/code
            - phase: Development phase (preclinical, phase_1, etc.)
            - estimated_peak_sales: Annual peak sales estimate
            - probability_override: (optional) Override default probability
        market_discount: Discount to apply to TAM estimates (default 0.70 = 30% haircut)
        npv_multiple: Multiple to apply for time value (default 3.0x)

    Returns:
        PipelineValuationResult with total value and per-drug breakdown
    """
    warnings = []
    drug_values = []
    total_probability_weighted_sales = 0
    total_value = 0

    for drug in pipeline:
        drug_name = drug.get("name", "Unknown")
        phase = drug.get("phase", "preclinical").lower().replace(" ", "_")
        peak_sales = drug.get("estimated_peak_sales", 0)
        prob_override = drug.get("probability_override")

        # Get probability (use override if provided)
        if prob_override is not None:
            prob = prob_override
        else:
            prob = PHASE_SUCCESS_PROBABILITIES.get(phase, 0.05)
            if phase not in PHASE_SUCCESS_PROBABILITIES:
                warnings.append(f"Unknown phase '{phase}' for {drug_name}, using 5% probability")

        # Calculate probability-weighted value
        pw_sales = peak_sales * prob * market_discount

        # For early-stage drugs, apply time discount
        time_years = BIOTECH_PRE_REVENUE_TIER["parameters"]["development_time_years"].get(phase, 5)
        discount_rate = BIOTECH_PRE_REVENUE_TIER["parameters"]["discount_rate"]
        time_discount = 1 / ((1 + discount_rate) ** time_years)

        # Calculate NPV contribution
        drug_npv = pw_sales * npv_multiple * time_discount

        drug_values.append({
            "name": drug_name,
            "phase": phase,
            "peak_sales": peak_sales,
            "probability": prob,
            "market_discount": market_discount,
            "probability_weighted_sales": pw_sales,
            "time_to_approval_years": time_years,
            "time_discount": time_discount,
            "npv_contribution": drug_npv,
        })

        total_probability_weighted_sales += pw_sales
        total_value += drug_npv

        logger.debug(
            f"Pipeline drug {drug_name}: Phase={phase}, Peak=${peak_sales/1e9:.2f}B, "
            f"Prob={prob:.0%}, PW Sales=${pw_sales/1e9:.2f}B, NPV=${drug_npv/1e9:.2f}B"
        )

    logger.info(
        f"Pipeline valuation: {len(pipeline)} drugs, "
        f"Total PW Sales=${total_probability_weighted_sales/1e9:.2f}B, "
        f"Total NPV=${total_value/1e9:.2f}B"
    )

    return PipelineValuationResult(
        total_pipeline_value=total_value,
        drug_values=drug_values,
        probability_weighted_sales=total_probability_weighted_sales,
        market_discount_applied=market_discount,
        methodology="probability_weighted_pipeline_npv",
        warnings=warnings,
    )


# ====================
# BIOTECH INDUSTRY DETECTION
# ====================

# Industries that should use biotech valuation
BIOTECH_INDUSTRIES = [
    "biotechnology",
    "biopharmaceuticals",
    "biological products",
    "biopharmaceutical",
    "biotech",
    "gene therapy",
    "cell therapy",
    "pharmaceutical preparations",  # Only if pre-revenue
]

# Keywords in company name that suggest biotech
BIOTECH_NAME_KEYWORDS = [
    "biotech",
    "biopharm",
    "therapeutics",
    "biopharma",
    "biologics",
    "gene therapy",
    "cell therapy",
    "oncology",
    "immuno",
]


def is_biotech_company(
    industry: Optional[str] = None,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Determine if a company should use biotech valuation.

    Args:
        industry: Industry classification
        company_name: Company name
        sector: Sector classification

    Returns:
        Tuple of (is_biotech, reason)
    """
    # Check industry classification
    if industry:
        industry_lower = industry.lower()
        for biotech_industry in BIOTECH_INDUSTRIES:
            if biotech_industry in industry_lower:
                return True, f"Industry match: {industry}"

    # Check company name for biotech keywords
    if company_name:
        company_lower = company_name.lower()
        for keyword in BIOTECH_NAME_KEYWORDS:
            if keyword in company_lower:
                return True, f"Company name contains '{keyword}'"

    # Check if sector is Healthcare
    if sector and "health" in sector.lower():
        # Healthcare sector but need more signals
        if industry and any(x in industry.lower() for x in ["drug", "pharma", "bio"]):
            return True, f"Healthcare sector with pharma/bio industry: {industry}"

    return False, "Not identified as biotech"


def is_pre_revenue_biotech(
    industry: Optional[str] = None,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    revenue: float = 0,
    revenue_threshold: float = 100_000_000,  # $100M threshold
) -> Tuple[bool, str]:
    """
    Determine if a company is a pre-revenue biotech that needs special valuation.

    Pre-revenue is defined as having less than $100M in annual revenue,
    as small revenues often come from partnerships rather than product sales.

    Args:
        industry: Industry classification
        company_name: Company name
        sector: Sector classification
        revenue: Annual revenue
        revenue_threshold: Revenue below which company is considered pre-revenue

    Returns:
        Tuple of (is_pre_revenue_biotech, reason)
    """
    # First check if it's biotech
    is_biotech, biotech_reason = is_biotech_company(industry, company_name, sector)

    if not is_biotech:
        return False, f"Not biotech: {biotech_reason}"

    # Check revenue
    if revenue < revenue_threshold:
        return True, f"Pre-revenue biotech ({biotech_reason}, revenue=${revenue/1e6:.1f}M < ${revenue_threshold/1e6:.0f}M threshold)"

    return False, f"Revenue-generating biotech (${revenue/1e6:.1f}M > ${revenue_threshold/1e6:.0f}M threshold)"


# ====================
# COMPREHENSIVE BIOTECH VALUATION
# ====================

@dataclass
class BiotechValuationResult:
    """Result from comprehensive biotech valuation."""
    fair_value_per_share: float
    total_enterprise_value: float
    pipeline_value: float
    cash_value: float
    cash_runway: CashRunwayResult
    pipeline_details: PipelineValuationResult
    methodology: str
    confidence: str
    warnings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


def value_biotech(
    symbol: str,
    financials: Dict,
    current_price: float,
    pipeline: Optional[List[Dict]] = None,
    market_discount: float = 0.70,
) -> BiotechValuationResult:
    """
    Comprehensive valuation for pre-revenue biotech companies.

    Valuation components:
    1. Pipeline Value (60%): Probability-weighted drug candidates
    2. Cash Value (25%): Cash position adjusted for runway risk
    3. Comparable Deals (15%): Industry benchmark comparables by therapeutic area

    If no pipeline data is provided, falls back to cash-based valuation
    with a significant discount.

    Args:
        symbol: Stock ticker symbol
        financials: Dictionary of financial metrics including:
            - cash: Cash and equivalents
            - quarterly_burn: Cash burn per quarter
            - shares_outstanding: Shares outstanding
            - revenue: Annual revenue (to verify pre-revenue status)
        current_price: Current stock price
        pipeline: List of drug candidate dictionaries (optional)
        market_discount: Discount to apply to TAM estimates (default 0.70)

    Returns:
        BiotechValuationResult with comprehensive valuation details
    """
    warnings = []

    # Extract financial metrics
    cash = financials.get("cash", 0) or financials.get("cash_and_equivalents", 0)
    quarterly_burn = financials.get("quarterly_burn", 0)
    shares_outstanding = financials.get("shares_outstanding", 0)
    revenue = financials.get("revenue", 0) or financials.get("total_revenue", 0)

    # Validate required data
    if not shares_outstanding:
        raise ValueError(f"{symbol} - Missing shares_outstanding for biotech valuation")

    if not cash:
        warnings.append("Cash position not available, using $0")
        cash = 0

    # Calculate cash burn if not provided
    if not quarterly_burn:
        # Try to estimate from operating cash flow
        operating_cf = financials.get("operating_cash_flow", 0)
        if operating_cf < 0:
            quarterly_burn = abs(operating_cf) / 4  # Annualize and quarterly
            warnings.append(f"Estimated quarterly burn from OCF: ${quarterly_burn/1e6:.1f}M")
        else:
            # Fallback: assume typical biotech burn
            quarterly_burn = cash * 0.08  # ~32% annual burn rate
            warnings.append("Cash burn not available, using 32% annual estimate")

    # Step 1: Calculate cash runway
    cash_runway = calculate_cash_runway(cash, quarterly_burn)

    if cash_runway.dilution_warning:
        warnings.append(cash_runway.risk_description)

    # Step 2: Calculate pipeline value (if provided)
    if pipeline and len(pipeline) > 0:
        pipeline_result = calculate_pipeline_value(
            pipeline=pipeline,
            market_discount=market_discount,
        )
        pipeline_value = pipeline_result.total_pipeline_value
        warnings.extend(pipeline_result.warnings)
        methodology = "pipeline_probability_weighted"
        confidence = "medium" if len(pipeline) >= 2 else "low"
    else:
        # No pipeline data - use cash-based valuation with discount
        pipeline_result = PipelineValuationResult(
            total_pipeline_value=0,
            drug_values=[],
            probability_weighted_sales=0,
            market_discount_applied=market_discount,
            methodology="no_pipeline_data",
            warnings=["No pipeline data provided - using cash-only valuation"],
        )
        pipeline_value = 0
        warnings.append("No pipeline data available - valuation based on cash only")
        methodology = "cash_only"
        confidence = "low"

    # Step 3: Calculate enterprise value
    # EV = Pipeline Value + Cash (no debt adjustment for simplicity)
    # For pre-revenue biotech, cash is usually the primary tangible asset

    # Apply runway discount to cash
    # If runway is short, the company will need to raise capital (dilution)
    if cash_runway.risk == CashRunwayRisk.CRITICAL:
        cash_discount = 0.60  # 40% discount for imminent dilution
    elif cash_runway.risk == CashRunwayRisk.HIGH:
        cash_discount = 0.80  # 20% discount for expected dilution
    elif cash_runway.risk == CashRunwayRisk.MEDIUM:
        cash_discount = 0.90  # 10% discount
    else:
        cash_discount = 1.0   # No discount for adequate runway

    adjusted_cash = cash * cash_discount

    # Step 4: Calculate comparables benchmark (15% weight)
    comparables_result = calculate_comparables_benchmark(
        pipeline=pipeline,
        cash=cash,
        shares_outstanding=shares_outstanding,
        indication=financials.get("primary_indication"),
        company_name=financials.get("company_name", symbol),
    )
    comparables_value = comparables_result.implied_fair_value * shares_outstanding

    # Step 5: Blend using tier weights (60% pipeline, 25% cash, 15% comparables)
    weights = BIOTECH_PRE_REVENUE_TIER["weights"]

    # Calculate weighted enterprise value
    if pipeline_value > 0:
        # Full weighting when pipeline data available
        pipeline_contribution = pipeline_value * (weights["pipeline_value"] / 100)
        cash_contribution = adjusted_cash * (weights["cash_runway"] / 100)
        comparables_contribution = comparables_value * (weights["comparable_deals"] / 100)
        total_ev = pipeline_contribution + cash_contribution + comparables_contribution

        # Normalize to ensure we're not double-counting
        # The weights should reflect relative importance, not absolute sum
        total_weight = weights["pipeline_value"] + weights["cash_runway"] + weights["comparable_deals"]
        total_ev = total_ev * (100 / total_weight)
    else:
        # No pipeline - use comparables and cash only (reweight to 60/40)
        cash_contribution = adjusted_cash * 0.60
        comparables_contribution = comparables_value * 0.40
        total_ev = cash_contribution + comparables_contribution

    # Per-share value
    fair_value_per_share = total_ev / shares_outstanding

    # Add warning if no pipeline and cash-only
    if pipeline_value == 0:
        # Cash-only valuation is essentially liquidation value
        warnings.append("Cash-only valuation - no pipeline upside reflected")

    # Log valuation summary
    logger.info(
        f"{symbol} - Biotech valuation: "
        f"Pipeline=${pipeline_value/1e9:.2f}B, Cash=${cash/1e6:.1f}M (adj ${adjusted_cash/1e6:.1f}M), "
        f"EV=${total_ev/1e9:.2f}B, Fair value=${fair_value_per_share:.2f}/share, "
        f"Runway={cash_runway.months:.1f} months ({cash_runway.risk.value})"
    )

    return BiotechValuationResult(
        fair_value_per_share=fair_value_per_share,
        total_enterprise_value=total_ev,
        pipeline_value=pipeline_value,
        cash_value=adjusted_cash,
        cash_runway=cash_runway,
        pipeline_details=pipeline_result,
        methodology=methodology,
        confidence=confidence,
        warnings=warnings,
        details={
            "symbol": symbol,
            "shares_outstanding": shares_outstanding,
            "cash_original": cash,
            "cash_discount": cash_discount,
            "cash_adjusted": adjusted_cash,
            "pipeline_drugs_count": len(pipeline) if pipeline else 0,
            "tier_weights": BIOTECH_PRE_REVENUE_TIER["weights"],
            # Comparables benchmark (15% weight)
            "comparables": {
                "therapeutic_area": comparables_result.therapeutic_area,
                "ev_benchmark_low": comparables_result.ev_benchmark_low,
                "ev_benchmark_high": comparables_result.ev_benchmark_high,
                "deal_premium": comparables_result.deal_premium,
                "implied_fair_value": comparables_result.implied_fair_value,
                "notes": comparables_result.notes,
            },
        },
    )
