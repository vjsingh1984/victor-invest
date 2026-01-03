"""
Valuation Services
==================

Multi-model valuation framework with sector-specific logic for rigorous fair value estimation.

CORE PHILOSOPHY:
    - No single model is reliable - we blend 6+ methods weighted by applicability
    - Sector-specific logic for banks, insurance, REITs, biotech, defense, semiconductors
    - Conservative bias - when uncertain, err toward lower valuations
    - Transparency - all assumptions documented in docs/VALUATION_ASSUMPTIONS.md

VALUATION METHODS:
    - DCF (Discounted Cash Flow): Intrinsic value from projected cash flows
    - GGM (Gordon Growth Model): Dividend discount for mature companies
    - PE (Price/Earnings): Earnings multiple with growth adjustments
    - PB (Price/Book): Asset-based for banks and insurance
    - EV/EBITDA: Operating earnings for cyclicals
    - PS (Price/Sales): Revenue multiple for pre-profit companies

SECTOR-SPECIFIC MODULES:
    - Insurance: Combined ratio quality, ROE-based P/BV
    - Banks: NIM, efficiency ratio, NPL ratio, Tier 1 capital
    - REITs: FFO multiples by property type, rate-adjusted
    - Biotech: Probability-weighted pipeline valuation
    - Defense: Backlog-adjusted EV/EBITDA
    - Semiconductors: Cycle-adjusted margins

RECENT ENHANCEMENTS (2025-12-30):
    - Biotech comparable deals: Therapeutic area benchmarks from M&A data
    - Earnings quality filter: Non-recurring item detection and adjustment
    - Confidence intervals: Fair value ranges based on growth uncertainty

For full assumptions and methodology, see: docs/VALUATION_ASSUMPTIONS.md

Migration History:
    2025-11-13: DCF
    2025-11-14: GGM, Sector Router
    2025-12-28: Insurance P/BV
    2025-12-29: REIT FFO, Biotech Pipeline, Defense Backlog
    2025-12-30: Bank P/B, Semiconductor Cycle-Adjusted, Growth-Adjusted Framework
"""

from investigator.domain.services.valuation.dcf import DCFValuation
from investigator.domain.services.valuation.ggm import GordonGrowthModel
from investigator.domain.services.valuation.sector_valuation_router import (
    SectorValuationRouter,
    ValuationResult,
)
from investigator.domain.services.valuation.insurance_valuation import (
    # Core valuation function
    value_insurance_company,
    calculate_insurance_specific_metrics,
    # P1-B Enhanced: Combined ratio calculation
    calculate_combined_ratio,
    calculate_loss_ratio,
    calculate_expense_ratio,
    assess_combined_ratio_quality,
    extract_insurance_metrics_from_xbrl,
    # Insurance type classification
    InsuranceType,
    TARGET_COMBINED_RATIOS,
    COMBINED_RATIO_THRESHOLDS,
)
from investigator.domain.services.valuation.reit_valuation import (
    REITPropertyType,
    REIT_FFO_MULTIPLES,
    detect_reit_property_type,
    get_ffo_multiple_range,
    get_base_ffo_multiple,
    adjust_ffo_multiple_for_rates,
    get_current_treasury_yield,
    value_reit,
    REITValuationResult,
    REITPropertyTypeResult,
)

# Biotech Pre-Revenue Valuation (P2-A)
from investigator.domain.services.valuation.biotech_valuation import (
    # Phase success probabilities
    DrugPhase,
    PHASE_SUCCESS_PROBABILITIES,
    PHASE_TRANSITION_RATES,
    # Cash runway
    CashRunwayRisk,
    CashRunwayResult,
    calculate_cash_runway,
    # Pipeline valuation
    BIOTECH_PRE_REVENUE_TIER,
    DrugCandidate,
    PipelineValuationResult,
    calculate_pipeline_value,
    # Industry detection
    BIOTECH_INDUSTRIES,
    is_biotech_company,
    is_pre_revenue_biotech,
    # Comprehensive valuation
    BiotechValuationResult,
    value_biotech,
)

# Defense Contractor Valuation (P2-B)
from investigator.domain.services.valuation.defense_valuation import (
    # Defense contractor classification
    DefenseContractorType,
    DEFENSE_INDUSTRIES,
    KNOWN_DEFENSE_CONTRACTORS,
    classify_defense_contractor,
    is_defense_industry,
    DefenseContractorClassification,
    # Backlog metrics
    BacklogMetrics,
    extract_backlog_metrics_from_xbrl,
    # Backlog premium/adjustments
    calculate_backlog_premium,
    calculate_backlog_value,
    calculate_contract_mix_adjustment,
    # Defense contractor tier
    DEFENSE_CONTRACTOR_TIER,
    get_defense_tier_weights,
    get_defense_tier_parameters,
    # Comprehensive valuation
    DefenseValuationResult,
    value_defense_contractor,
)

# Bank Valuation (Workstream 4)
from investigator.domain.services.valuation.bank_valuation import (
    # Bank type classification
    BankType,
    # Target metrics
    TARGET_NIM,
    TARGET_TIER1,
    TARGET_EFFICIENCY_RATIO,
    # Thresholds
    EFFICIENCY_THRESHOLDS,
    NPL_THRESHOLDS,
    TIER1_THRESHOLDS,
    # Dataclasses
    BankMetrics,
    BankValuationResult,
    # Functions
    extract_bank_metrics_from_xbrl,
    assess_bank_quality,
    value_bank,
)

# Semiconductor Valuation (Workstream 3)
from investigator.domain.services.valuation.semiconductor_valuation import (
    # Chip type and cycle classification
    ChipType,
    CyclePosition,
    # Known companies and industries
    SEMICONDUCTOR_INDUSTRIES,
    KNOWN_SEMICONDUCTOR_COMPANIES,
    # Semiconductor tier configuration
    SEMICONDUCTOR_TIER,
    # Dataclasses
    SemiconductorMetrics,
    SemiconductorValuationResult,
    # Functions
    extract_semiconductor_metrics_from_xbrl,
    calculate_cycle_adjustment,
    calculate_normalized_margin,
    is_semiconductor_industry,
    classify_semiconductor_company,
    value_semiconductor,
    get_semiconductor_tier_weights,
    get_semiconductor_tier_parameters,
)

__all__ = [
    # Core valuation models
    "DCFValuation",
    "GordonGrowthModel",
    "SectorValuationRouter",
    "ValuationResult",
    # Insurance valuation
    "value_insurance_company",
    "calculate_insurance_specific_metrics",
    # P1-B Enhanced: Combined ratio calculation
    "calculate_combined_ratio",
    "calculate_loss_ratio",
    "calculate_expense_ratio",
    "assess_combined_ratio_quality",
    "extract_insurance_metrics_from_xbrl",
    "InsuranceType",
    "TARGET_COMBINED_RATIOS",
    "COMBINED_RATIO_THRESHOLDS",
    # REIT valuation (P1-C)
    "REITPropertyType",
    "REIT_FFO_MULTIPLES",
    "detect_reit_property_type",
    "get_ffo_multiple_range",
    "get_base_ffo_multiple",
    "adjust_ffo_multiple_for_rates",
    "get_current_treasury_yield",
    "value_reit",
    "REITValuationResult",
    "REITPropertyTypeResult",
    # Biotech pre-revenue valuation (P2-A)
    "DrugPhase",
    "PHASE_SUCCESS_PROBABILITIES",
    "PHASE_TRANSITION_RATES",
    "CashRunwayRisk",
    "CashRunwayResult",
    "calculate_cash_runway",
    "BIOTECH_PRE_REVENUE_TIER",
    "DrugCandidate",
    "PipelineValuationResult",
    "calculate_pipeline_value",
    "BIOTECH_INDUSTRIES",
    "is_biotech_company",
    "is_pre_revenue_biotech",
    "BiotechValuationResult",
    "value_biotech",
    # Defense contractor valuation (P2-B)
    "DefenseContractorType",
    "DEFENSE_INDUSTRIES",
    "KNOWN_DEFENSE_CONTRACTORS",
    "classify_defense_contractor",
    "is_defense_industry",
    "DefenseContractorClassification",
    "BacklogMetrics",
    "extract_backlog_metrics_from_xbrl",
    "calculate_backlog_premium",
    "calculate_backlog_value",
    "calculate_contract_mix_adjustment",
    "DEFENSE_CONTRACTOR_TIER",
    "get_defense_tier_weights",
    "get_defense_tier_parameters",
    "DefenseValuationResult",
    "value_defense_contractor",
    # Bank valuation (Workstream 4)
    "BankType",
    "TARGET_NIM",
    "TARGET_TIER1",
    "TARGET_EFFICIENCY_RATIO",
    "EFFICIENCY_THRESHOLDS",
    "NPL_THRESHOLDS",
    "TIER1_THRESHOLDS",
    "BankMetrics",
    "BankValuationResult",
    "extract_bank_metrics_from_xbrl",
    "assess_bank_quality",
    "value_bank",
    # Semiconductor valuation (Workstream 3)
    "ChipType",
    "CyclePosition",
    "SEMICONDUCTOR_INDUSTRIES",
    "KNOWN_SEMICONDUCTOR_COMPANIES",
    "SEMICONDUCTOR_TIER",
    "SemiconductorMetrics",
    "SemiconductorValuationResult",
    "extract_semiconductor_metrics_from_xbrl",
    "calculate_cycle_adjustment",
    "calculate_normalized_margin",
    "is_semiconductor_industry",
    "classify_semiconductor_company",
    "value_semiconductor",
    "get_semiconductor_tier_weights",
    "get_semiconductor_tier_parameters",
]
