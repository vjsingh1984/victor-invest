"""
Industry Dataset Integration Module

Provides integration between the IndustryDatasetRegistry and the valuation pipeline.
This module offers helper functions that can be called during valuation to:
1. Extract industry-specific metrics
2. Get valuation adjustments
3. Get recommended tier weights

Author: Claude Code
Date: 2025-12-30
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from investigator.domain.services.industry_datasets.base import (
    BaseIndustryDataset,
    IndustryMetrics,
    MetricQuality,
    ValuationAdjustment,
)
from investigator.domain.services.industry_datasets.registry import (
    get_dataset_for_industry,
    get_dataset_for_symbol,
    get_registry,
)

logger = logging.getLogger(__name__)


def extract_industry_metrics(
    symbol: str,
    xbrl_data: Optional[Dict],
    financials: Dict,
    industry: Optional[str] = None,
    sector: Optional[str] = None,
) -> Optional[IndustryMetrics]:
    """
    Extract industry-specific metrics for a stock.

    This is the main entry point for extracting industry-specific data.
    It finds the appropriate dataset and extracts relevant metrics.

    Args:
        symbol: Stock symbol
        xbrl_data: Raw XBRL data dictionary (may be None)
        financials: Dictionary of standard financial metrics
        industry: Industry name (optional)
        sector: Sector name (optional)

    Returns:
        IndustryMetrics object if a matching dataset is found, None otherwise

    Example:
        metrics = extract_industry_metrics(
            symbol="NVDA",
            xbrl_data=xbrl_data,
            financials=financials,
            industry="Semiconductors"
        )
        if metrics:
            print(f"Inventory days: {metrics.get('inventory_days')}")
            print(f"Quality: {metrics.quality}")
    """
    registry = get_registry()

    # Try to find matching dataset
    dataset = registry.get_best_match(symbol, industry, sector)

    if not dataset:
        logger.debug(f"{symbol} - No industry dataset found for industry={industry}")
        return None

    logger.info(f"{symbol} - Using {dataset.display_name} dataset")

    try:
        metrics = dataset.extract_metrics(
            symbol=symbol,
            xbrl_data=xbrl_data,
            financials=financials,
            industry=industry,
            sector=sector,
        )

        # Validate metrics
        validation_warnings = dataset.validate_metrics(metrics)
        if validation_warnings:
            metrics.warnings.extend(validation_warnings)
            logger.debug(f"{symbol} - Validation warnings: {validation_warnings}")

        logger.info(
            f"{symbol} - Extracted {len(metrics.metrics)} industry metrics, "
            f"quality={metrics.quality.value}, coverage={metrics.coverage:.0%}"
        )

        return metrics

    except Exception as e:
        logger.warning(f"{symbol} - Failed to extract industry metrics: {e}")
        return None


def get_valuation_adjustments(
    symbol: str,
    xbrl_data: Optional[Dict],
    financials: Dict,
    industry: Optional[str] = None,
    sector: Optional[str] = None,
) -> Tuple[List[ValuationAdjustment], Optional[IndustryMetrics]]:
    """
    Get valuation adjustments based on industry-specific metrics.

    This function extracts metrics and calculates adjustments in one call.

    Args:
        symbol: Stock symbol
        xbrl_data: Raw XBRL data dictionary
        financials: Dictionary of standard financial metrics
        industry: Industry name (optional)
        sector: Sector name (optional)

    Returns:
        Tuple of (list of ValuationAdjustment, IndustryMetrics or None)

    Example:
        adjustments, metrics = get_valuation_adjustments(
            symbol="NVDA",
            xbrl_data=xbrl_data,
            financials=financials,
            industry="Semiconductors"
        )
        for adj in adjustments:
            print(f"{adj.adjustment_type}: {adj.factor:.2f}x - {adj.reason}")
    """
    # First extract metrics
    metrics = extract_industry_metrics(
        symbol=symbol,
        xbrl_data=xbrl_data,
        financials=financials,
        industry=industry,
        sector=sector,
    )

    if not metrics:
        return [], None

    # Find dataset again to get adjustments
    registry = get_registry()
    dataset = registry.get_best_match(symbol, industry, sector)

    if not dataset:
        return [], metrics

    try:
        adjustments = dataset.get_valuation_adjustments(
            metrics=metrics,
            financials=financials,
            industry=industry,
            sector=sector,
        )

        logger.info(f"{symbol} - Generated {len(adjustments)} valuation adjustments")

        return adjustments, metrics

    except Exception as e:
        logger.warning(f"{symbol} - Failed to get valuation adjustments: {e}")
        return [], metrics


def get_recommended_tier_weights(
    symbol: str,
    industry: Optional[str] = None,
    sector: Optional[str] = None,
) -> Optional[Dict[str, int]]:
    """
    Get recommended tier weights for a stock's industry.

    Args:
        symbol: Stock symbol
        industry: Industry name (optional)
        sector: Sector name (optional)

    Returns:
        Dictionary of model -> weight, or None if no recommendation

    Example:
        weights = get_recommended_tier_weights("NVDA", industry="Semiconductors")
        # Returns: {'ev_ebitda': 45, 'dcf': 25, 'pe': 20, 'pb': 10, ...}
    """
    registry = get_registry()
    dataset = registry.get_best_match(symbol, industry, sector)

    if not dataset:
        return None

    return dataset.get_tier_weights()


def get_xbrl_tag_aliases(
    industry: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Get XBRL tag aliases for an industry or all industries.

    Args:
        industry: Industry name (optional). If None, returns all aliases.

    Returns:
        Dictionary of canonical_name -> List[xbrl_tags]

    Example:
        aliases = get_xbrl_tag_aliases("Semiconductors")
        # Returns: {'inventory_days': ['DaysInventoryOutstanding', ...], ...}
    """
    registry = get_registry()

    if industry:
        dataset = get_dataset_for_industry(industry)
        if dataset:
            return dataset.get_xbrl_aliases()
        return {}

    # Collect all aliases from all datasets
    all_aliases = {}
    for name in registry.list_datasets():
        dataset = registry.get(name)
        if dataset:
            aliases = dataset.get_xbrl_aliases()
            for key, tags in aliases.items():
                if key not in all_aliases:
                    all_aliases[key] = []
                for tag in tags:
                    if tag not in all_aliases[key]:
                        all_aliases[key].append(tag)

    return all_aliases


def apply_adjustments_to_fair_value(
    base_fair_value: float,
    adjustments: List[ValuationAdjustment],
    models_to_apply: Optional[List[str]] = None,
) -> Tuple[float, List[str]]:
    """
    Apply valuation adjustments to a base fair value.

    Args:
        base_fair_value: The base fair value before adjustments
        adjustments: List of ValuationAdjustment objects
        models_to_apply: Optional list of model names to filter adjustments by.
                        If None, applies adjustments that affect any model.

    Returns:
        Tuple of (adjusted fair value, list of adjustment reasons applied)

    Example:
        adjusted_value, reasons = apply_adjustments_to_fair_value(
            base_fair_value=100.0,
            adjustments=adjustments,
            models_to_apply=["pe", "ev_ebitda"]
        )
        print(f"Adjusted value: ${adjusted_value:.2f}")
        for reason in reasons:
            print(f"  - {reason}")
    """
    adjusted_value = base_fair_value
    applied_reasons = []

    for adj in adjustments:
        # Check if this adjustment applies to the requested models
        if models_to_apply:
            if not any(m in adj.affects_models for m in models_to_apply):
                continue

        # Apply the adjustment
        adjusted_value *= adj.factor
        applied_reasons.append(f"{adj.adjustment_type}: {adj.factor:.2f}x - {adj.reason}")

    return adjusted_value, applied_reasons


def get_industry_summary(
    symbol: str,
    xbrl_data: Optional[Dict],
    financials: Dict,
    industry: Optional[str] = None,
    sector: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a comprehensive industry summary for a stock.

    This is a convenience function that combines metrics extraction,
    quality assessment, adjustments, and tier weights into one call.

    Args:
        symbol: Stock symbol
        xbrl_data: Raw XBRL data dictionary
        financials: Dictionary of standard financial metrics
        industry: Industry name (optional)
        sector: Sector name (optional)

    Returns:
        Dictionary with industry analysis, or None if no dataset found

    Example:
        summary = get_industry_summary(
            symbol="NVDA",
            xbrl_data=xbrl_data,
            financials=financials,
            industry="Semiconductors"
        )
        if summary:
            print(f"Dataset: {summary['dataset_name']}")
            print(f"Quality: {summary['quality']}")
            print(f"Key metrics: {summary['metrics']}")
    """
    registry = get_registry()
    dataset = registry.get_best_match(symbol, industry, sector)

    if not dataset:
        return None

    # Extract metrics
    metrics = extract_industry_metrics(
        symbol=symbol,
        xbrl_data=xbrl_data,
        financials=financials,
        industry=industry,
        sector=sector,
    )

    if not metrics:
        return {
            "dataset_name": dataset.name,
            "dataset_display_name": dataset.display_name,
            "dataset_version": dataset.version,
            "extraction_failed": True,
            "error": "Failed to extract metrics",
        }

    # Get adjustments
    adjustments = dataset.get_valuation_adjustments(
        metrics=metrics,
        financials=financials,
        industry=industry,
        sector=sector,
    )

    # Get tier weights
    tier_weights = dataset.get_tier_weights()

    return {
        "dataset_name": dataset.name,
        "dataset_display_name": dataset.display_name,
        "dataset_version": dataset.version,
        "symbol": symbol,
        "industry": industry,
        "sector": sector,
        "metrics": metrics.metrics,
        "quality": metrics.quality.value,
        "coverage": metrics.coverage,
        "warnings": metrics.warnings,
        "metadata": metrics.metadata,
        "adjustments": [
            {
                "type": adj.adjustment_type,
                "factor": adj.factor,
                "reason": adj.reason,
                "confidence": adj.confidence,
                "affects_models": adj.affects_models,
            }
            for adj in adjustments
        ],
        "tier_weights": tier_weights,
    }


def list_available_industries() -> List[str]:
    """
    List all industries covered by registered datasets.

    Returns:
        List of industry names
    """
    from investigator.domain.services.industry_datasets.registry import (
        list_registered_industries,
    )

    return list_registered_industries()


def is_industry_covered(industry: Optional[str]) -> bool:
    """
    Check if an industry has a specialized dataset.

    Args:
        industry: Industry name

    Returns:
        True if a dataset covers this industry
    """
    if not industry:
        return False
    dataset = get_dataset_for_industry(industry)
    return dataset is not None


def is_symbol_covered(symbol: str) -> bool:
    """
    Check if a symbol is explicitly covered by a dataset.

    Args:
        symbol: Stock symbol

    Returns:
        True if symbol is in a dataset's known symbols
    """
    dataset = get_dataset_for_symbol(symbol)
    return dataset is not None
