"""
Industry-Specific Datasets Registry

A pluggable, registry-based architecture for industry-specific metrics extraction.

Design Principles (SOLID):
- Single Responsibility: Each dataset module handles ONE industry
- Open/Closed: Registry extensible via registration without modifying core code
- Liskov Substitution: All datasets implement BaseIndustryDataset interface
- Interface Segregation: Separate extraction, metrics, and validation methods
- Dependency Inversion: Valuation code depends on registry abstraction

Usage:
    from investigator.domain.services.industry_datasets import (
        IndustryDatasetRegistry,
        get_dataset_for_industry,
        extract_industry_metrics,
        get_valuation_adjustments,
    )

    # Get dataset for a specific industry
    dataset = get_dataset_for_industry("Semiconductors")
    if dataset:
        metrics = dataset.extract_metrics(symbol, xbrl_data, financials)
        quality = dataset.assess_quality(metrics)
        adjustments = dataset.get_valuation_adjustments(metrics)

    # Or use the integration helpers for a simpler API:
    metrics = extract_industry_metrics(symbol, xbrl_data, financials, industry="Semiconductors")
    adjustments, metrics = get_valuation_adjustments(symbol, xbrl_data, financials, industry="Semiconductors")

    # Register a custom dataset
    from investigator.domain.services.industry_datasets import register_dataset
    register_dataset(MyCustomDataset())

Author: Claude Code
Date: 2025-12-30
"""

from investigator.domain.services.industry_datasets.base import (
    BaseIndustryDataset,
    IndustryMetrics,
    MetricDefinition,
    MetricQuality,
    ValuationAdjustment,
)
from investigator.domain.services.industry_datasets.registry import (
    IndustryDatasetRegistry,
    get_registry,
    get_dataset_for_industry,
    get_dataset_for_symbol,
    register_dataset,
    list_registered_industries,
)

# Import and auto-register all industry datasets
from investigator.domain.services.industry_datasets.semiconductor_dataset import SemiconductorDataset
from investigator.domain.services.industry_datasets.bank_dataset import BankDataset
from investigator.domain.services.industry_datasets.reit_dataset import REITDataset
from investigator.domain.services.industry_datasets.auto_dataset import AutoDataset
from investigator.domain.services.industry_datasets.defense_dataset import DefenseDataset
from investigator.domain.services.industry_datasets.insurance_dataset import InsuranceDataset

# Integration helpers - simpler API for using datasets
from investigator.domain.services.industry_datasets.integration import (
    extract_industry_metrics,
    get_valuation_adjustments,
    get_recommended_tier_weights,
    get_xbrl_tag_aliases,
    apply_adjustments_to_fair_value,
    get_industry_summary,
    list_available_industries,
    is_industry_covered,
    is_symbol_covered,
)

__all__ = [
    # Base classes
    "BaseIndustryDataset",
    "IndustryMetrics",
    "MetricDefinition",
    "MetricQuality",
    "ValuationAdjustment",
    # Registry
    "IndustryDatasetRegistry",
    "get_registry",
    "get_dataset_for_industry",
    "get_dataset_for_symbol",
    "register_dataset",
    "list_registered_industries",
    # Dataset implementations
    "SemiconductorDataset",
    "BankDataset",
    "REITDataset",
    "AutoDataset",
    "DefenseDataset",
    "InsuranceDataset",
    # Integration helpers
    "extract_industry_metrics",
    "get_valuation_adjustments",
    "get_recommended_tier_weights",
    "get_xbrl_tag_aliases",
    "apply_adjustments_to_fair_value",
    "get_industry_summary",
    "list_available_industries",
    "is_industry_covered",
    "is_symbol_covered",
]
