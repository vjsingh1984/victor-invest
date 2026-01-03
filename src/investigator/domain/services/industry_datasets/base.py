"""
Base Classes for Industry-Specific Datasets

Defines the abstract interface that all industry datasets must implement.
This ensures consistency and enables the registry pattern.

Author: Claude Code
Date: 2025-12-30
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MetricQuality(Enum):
    """Quality rating for extracted metrics."""

    EXCELLENT = "excellent"  # All key metrics available, high confidence
    GOOD = "good"  # Most metrics available
    FAIR = "fair"  # Some metrics available, can still value
    POOR = "poor"  # Few metrics, low confidence
    INSUFFICIENT = "insufficient"  # Cannot value reliably


@dataclass
class MetricDefinition:
    """
    Definition of an industry-specific metric.

    Attributes:
        name: Canonical name (e.g., 'inventory_days')
        display_name: Human-readable name (e.g., 'Inventory Days')
        description: What this metric measures
        xbrl_tags: List of XBRL tags to try (priority order)
        unit: Unit of measurement (e.g., 'days', 'ratio', 'percent')
        is_required: Whether this metric is required for valuation
        default_value: Default if not found (None = no default)
        min_value: Minimum valid value (for validation)
        max_value: Maximum valid value (for validation)
        invert_for_quality: If True, lower values indicate better quality
    """

    name: str
    display_name: str
    description: str
    xbrl_tags: List[str]
    unit: str = "value"
    is_required: bool = False
    default_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    invert_for_quality: bool = False


@dataclass
class IndustryMetrics:
    """
    Container for extracted industry-specific metrics.

    Attributes:
        industry: Industry name
        symbol: Stock symbol
        metrics: Dictionary of metric_name -> value
        quality: Overall quality rating
        coverage: Percentage of metrics successfully extracted
        warnings: List of warnings/issues
        metadata: Additional metadata (extraction timestamp, source, etc.)
    """

    industry: str
    symbol: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality: MetricQuality = MetricQuality.INSUFFICIENT
    coverage: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, metric_name: str, default: Any = None) -> Any:
        """Get a metric value with optional default."""
        return self.metrics.get(metric_name, default)

    def has(self, metric_name: str) -> bool:
        """Check if a metric is available."""
        return metric_name in self.metrics and self.metrics[metric_name] is not None


@dataclass
class ValuationAdjustment:
    """
    Valuation adjustment based on industry metrics.

    Attributes:
        adjustment_type: Type of adjustment (e.g., 'premium', 'discount', 'weight_shift')
        factor: Adjustment factor (e.g., 1.10 for 10% premium)
        reason: Explanation for the adjustment
        confidence: Confidence in this adjustment (0-1)
        affects_models: Which valuation models this affects
    """

    adjustment_type: str
    factor: float
    reason: str
    confidence: float = 1.0
    affects_models: List[str] = field(default_factory=list)


class BaseIndustryDataset(ABC):
    """
    Abstract base class for industry-specific datasets.

    All industry datasets must implement this interface to be registered
    and used by the valuation pipeline.

    Subclasses must implement:
    - get_industry_names(): Return list of industry names this dataset handles
    - get_metric_definitions(): Return list of MetricDefinition objects
    - extract_metrics(): Extract metrics from XBRL data
    - assess_quality(): Assess quality of extracted metrics
    - get_valuation_adjustments(): Calculate valuation adjustments

    Optional overrides:
    - get_known_symbols(): Return set of known symbols for this industry
    - get_tier_weights(): Return recommended tier weights
    - validate_metrics(): Validate extracted metrics
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this dataset (e.g., 'semiconductor')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Semiconductor Industry')."""
        pass

    @property
    def version(self) -> str:
        """Version of this dataset implementation."""
        return "1.0.0"

    @abstractmethod
    def get_industry_names(self) -> List[str]:
        """
        Return list of industry names this dataset handles.

        These are matched against the industry field from metadata service.
        Case-insensitive matching is used.

        Returns:
            List of industry name strings
        """
        pass

    @abstractmethod
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """
        Return list of metric definitions for this industry.

        Returns:
            List of MetricDefinition objects describing available metrics
        """
        pass

    def get_known_symbols(self) -> Set[str]:
        """
        Return set of known symbols for this industry.

        Override this to provide explicit symbol mappings for companies
        that might not be correctly classified by industry name alone.

        Returns:
            Set of uppercase stock symbols
        """
        return set()

    def matches_industry(self, industry: Optional[str]) -> bool:
        """
        Check if this dataset handles the given industry.

        Args:
            industry: Industry name from metadata service

        Returns:
            True if this dataset handles this industry
        """
        if not industry:
            return False

        industry_lower = industry.lower()
        for ind_name in self.get_industry_names():
            if ind_name.lower() in industry_lower or industry_lower in ind_name.lower():
                return True
        return False

    def matches_symbol(self, symbol: str) -> bool:
        """
        Check if this dataset handles the given symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if symbol is in known symbols set
        """
        return symbol.upper() in self.get_known_symbols()

    @abstractmethod
    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """
        Extract industry-specific metrics from XBRL data and financials.

        Args:
            symbol: Stock symbol
            xbrl_data: Raw XBRL data dictionary (may be None)
            financials: Dictionary of standard financial metrics
            **kwargs: Additional context (e.g., industry, sector)

        Returns:
            IndustryMetrics object with extracted metrics
        """
        pass

    @abstractmethod
    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """
        Assess the quality of extracted metrics.

        Args:
            metrics: Extracted IndustryMetrics object

        Returns:
            Tuple of (MetricQuality, description)
        """
        pass

    @abstractmethod
    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """
        Calculate valuation adjustments based on industry metrics.

        Args:
            metrics: Extracted IndustryMetrics object
            financials: Dictionary of standard financial metrics
            **kwargs: Additional context

        Returns:
            List of ValuationAdjustment objects
        """
        pass

    def validate_metrics(self, metrics: IndustryMetrics) -> List[str]:
        """
        Validate extracted metrics against expected ranges.

        Args:
            metrics: Extracted IndustryMetrics object

        Returns:
            List of validation warning strings
        """
        warnings = []
        definitions = {d.name: d for d in self.get_metric_definitions()}

        for metric_name, value in metrics.metrics.items():
            if value is None:
                continue

            defn = definitions.get(metric_name)
            if not defn:
                continue

            if defn.min_value is not None and value < defn.min_value:
                warnings.append(f"{defn.display_name} ({value}) below minimum ({defn.min_value})")

            if defn.max_value is not None and value > defn.max_value:
                warnings.append(f"{defn.display_name} ({value}) above maximum ({defn.max_value})")

        return warnings

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """
        Return recommended tier weights for this industry.

        Override this to provide industry-specific weight recommendations.

        Returns:
            Dictionary of model -> weight (percentages summing to 100),
            or None to use default weights
        """
        return None

    def get_xbrl_aliases(self) -> Dict[str, List[str]]:
        """
        Return XBRL tag aliases for this industry's metrics.

        These can be added to the central XBRLTagAliasMapper.

        Returns:
            Dictionary of canonical_name -> List[xbrl_tags]
        """
        aliases = {}
        for defn in self.get_metric_definitions():
            if defn.xbrl_tags:
                aliases[defn.name] = defn.xbrl_tags
        return aliases

    def _extract_from_xbrl(
        self, xbrl_data: Optional[Dict], metric_name: str, xbrl_tags: List[str], default: Optional[float] = None
    ) -> Optional[float]:
        """
        Helper to extract a value from XBRL data.

        Tries each tag in priority order until a value is found.

        Args:
            xbrl_data: Raw XBRL data
            metric_name: Canonical metric name
            xbrl_tags: List of XBRL tags to try
            default: Default value if not found

        Returns:
            Extracted value or default
        """
        if not xbrl_data:
            return default

        us_gaap = xbrl_data.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            return default

        for tag in xbrl_tags:
            if tag in us_gaap:
                concept = us_gaap[tag]
                units = concept.get("units", {})
                usd_data = units.get("USD", [])

                if usd_data:
                    # Get latest value
                    sorted_data = sorted(
                        [d for d in usd_data if d.get("form") in ["10-K", "10-Q", "20-F"]],
                        key=lambda x: (
                            x.get("fy", 0),
                            {"FY": 5, "Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}.get(x.get("fp", ""), 0),
                        ),
                        reverse=True,
                    )

                    if sorted_data:
                        value = sorted_data[0].get("val")
                        if value is not None:
                            logger.debug(f"Extracted {metric_name} from {tag}: {value}")
                            return float(value)

        return default

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
