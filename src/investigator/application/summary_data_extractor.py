"""
Summary Data Extractor - SOLID-based data normalization for executive summaries.

Provides a robust, extensible extraction framework that:
1. Maps multiple field names to canonical outputs (handles API variations)
2. Falls back to alternative data sources when primary is unavailable
3. Handles type mismatches gracefully (dict vs list, None vs empty)
4. Provides detailed extraction audit trails

Design Principles:
- Single Responsibility: Each extractor handles one field type
- Open/Closed: Add new extractors without modifying existing code
- Liskov Substitution: All extractors are interchangeable via common interface
- Interface Segregation: Small, focused extraction contracts
- Dependency Inversion: Formatter depends on abstractions, not concrete extractors

Usage:
    from investigator.application.summary_data_extractor import SummaryDataExtractor

    extractor = SummaryDataExtractor(analysis_results)
    summary = extractor.extract_all()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExtractionConfidence(Enum):
    """Confidence level for extracted data."""

    HIGH = "high"  # Found at primary location
    MEDIUM = "medium"  # Found via fallback
    LOW = "low"  # Derived/calculated
    NONE = "none"  # Not found


@dataclass
class ExtractionResult:
    """Result of a single field extraction."""

    value: Any
    confidence: ExtractionConfidence
    source_path: str  # Where the value was found (for debugging)
    fallback_used: bool = False

    @property
    def has_value(self) -> bool:
        """Check if extraction yielded a usable value."""
        if self.value is None:
            return False
        if isinstance(self.value, str) and self.value in ["", "N/A", "None"]:
            return False
        if isinstance(self.value, (list, dict)) and len(self.value) == 0:
            return False
        return True

    @classmethod
    def not_found(cls, attempted_paths: str = "") -> "ExtractionResult":
        """Factory for not-found result."""
        return cls(
            value=None,
            confidence=ExtractionConfidence.NONE,
            source_path=f"not_found:{attempted_paths}",
            fallback_used=False,
        )


class FieldExtractor(Protocol):
    """Protocol for field extractors (Interface Segregation)."""

    @property
    def field_name(self) -> str:
        """Name of the field this extractor handles."""
        ...

    def extract(self, data: Dict[str, Any]) -> ExtractionResult:
        """Extract the field value from data."""
        ...


class BaseFieldExtractor(ABC):
    """
    Base class for field extractors (Template Method Pattern).

    Subclasses implement _get_paths() and optionally _transform_value().
    """

    @property
    @abstractmethod
    def field_name(self) -> str:
        """Name of the field this extractor handles."""
        pass

    @abstractmethod
    def _get_paths(self) -> List[Tuple[str, ...]]:
        """
        Return ordered list of paths to try.

        Each path is a tuple of keys to traverse.
        Paths are tried in order until one succeeds.
        """
        pass

    def _transform_value(self, value: Any) -> Any:
        """Optional transformation after extraction. Override in subclasses."""
        return value

    def _validate_value(self, value: Any) -> bool:
        """Validate extracted value. Override for custom validation."""
        if value is None:
            return False
        if isinstance(value, str) and value in ["", "N/A", "None", "null"]:
            return False
        if isinstance(value, (int, float)) and value == 0:
            # Zero might be valid for some fields - subclass can override
            return True
        return True

    def extract(self, data: Dict[str, Any]) -> ExtractionResult:
        """
        Extract value using fallback chain (Template Method).

        Tries each path in order, returning first valid value found.
        """
        attempted_paths = []

        for i, path in enumerate(self._get_paths()):
            path_str = ".".join(path)
            attempted_paths.append(path_str)

            value = self._traverse_path(data, path)

            if value is not None:
                transformed = self._transform_value(value)

                if self._validate_value(transformed):
                    return ExtractionResult(
                        value=transformed,
                        confidence=ExtractionConfidence.HIGH if i == 0 else ExtractionConfidence.MEDIUM,
                        source_path=path_str,
                        fallback_used=(i > 0),
                    )

        return ExtractionResult.not_found(", ".join(attempted_paths))

    def _traverse_path(self, data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
        """Navigate nested dictionary using path tuple."""
        current = data

        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None

        return current


# =============================================================================
# Concrete Extractors (Open/Closed - add new ones without modifying existing)
# =============================================================================


class PriceTargetExtractor(BaseFieldExtractor):
    """
    Extracts price target from multiple possible locations.

    Fallback chain:
    1. valuation.price_target_12_month (explicit 12-month target)
    2. valuation.price_target (generic price target)
    3. valuation.fair_value (multi-model blended fair value)
    4. valuation.blended_fair_value (explicit blended value)
    5. fundamental.valuation.fair_value (nested structure)
    """

    @property
    def field_name(self) -> str:
        return "price_target_12m"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            # Direct valuation paths
            ("agents", "fundamental", "valuation", "price_target_12_month"),
            ("agents", "fundamental", "valuation", "price_target"),
            ("agents", "fundamental", "valuation", "fair_value"),
            ("agents", "fundamental", "valuation", "blended_fair_value"),
            # Response-nested paths
            ("agents", "fundamental", "valuation", "response", "price_target_12_month"),
            ("agents", "fundamental", "valuation", "response", "fair_value"),
            ("agents", "fundamental", "valuation", "response", "blended_fair_value"),
            # Multi-model summary path
            ("agents", "fundamental", "multi_model_summary", "blended_fair_value"),
            # Legacy/alternative paths (without 'agents' wrapper)
            ("fundamental", "valuation", "fair_value"),
            ("fundamental", "valuation", "price_target"),
            ("fundamental", "valuation", "blended_fair_value"),
            ("fundamental", "multi_model_summary", "blended_fair_value"),
        ]

    def _validate_value(self, value: Any) -> bool:
        """Price target must be positive number."""
        if not isinstance(value, (int, float)):
            return False
        return value > 0


class InvestmentGradeExtractor(BaseFieldExtractor):
    """
    Extracts investment grade with fallback to calculation.

    Fallback chain:
    1. valuation.investment_grade (explicit grade)
    2. Calculate from upside_pct (A=20%+, B=10-20%, C=0-10%, D=-10-0%, F=-10%+)
    """

    VALID_GRADES = {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F", "N/A"}

    @property
    def field_name(self) -> str:
        return "investment_grade"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "fundamental", "valuation", "investment_grade"),
            ("agents", "fundamental", "valuation", "response", "investment_grade"),
            ("fundamental", "valuation", "investment_grade"),
        ]

    def _validate_value(self, value: Any) -> bool:
        """Grade must be valid letter grade."""
        if not isinstance(value, str):
            return False
        return value.upper() in self.VALID_GRADES or len(value) <= 2

    def extract(self, data: Dict[str, Any]) -> ExtractionResult:
        """Override to add calculated grade fallback."""
        # Try standard extraction first
        result = super().extract(data)

        if result.has_value:
            return result

        # Fallback: Calculate from upside percentage
        return self._calculate_from_upside(data)

    def _calculate_from_upside(self, data: Dict[str, Any]) -> ExtractionResult:
        """Calculate grade from upside/downside percentage."""
        # Try to find upside percentage
        upside_paths = [
            ("agents", "fundamental", "valuation", "upside_downside_pct"),
            ("agents", "fundamental", "valuation", "upside_pct"),
            ("agents", "fundamental", "valuation", "expected_return"),
            ("agents", "fundamental", "multi_model_summary", "blended_upside_pct"),
            ("fundamental", "valuation", "upside_downside_pct"),
            ("fundamental", "multi_model_summary", "blended_upside_pct"),
        ]

        upside = None
        source_path = ""

        for path in upside_paths:
            upside = self._traverse_path(data, path)
            if upside is not None and isinstance(upside, (int, float)):
                source_path = ".".join(path)
                break

        if upside is None:
            # Try to calculate from price_target and current_price
            price_target = PriceTargetExtractor().extract(data)
            current_price = CurrentPriceExtractor().extract(data)

            if price_target.has_value and current_price.has_value:
                upside = ((price_target.value - current_price.value) / current_price.value) * 100
                source_path = "calculated_from_prices"

        if upside is None:
            return ExtractionResult.not_found("upside_pct calculation failed")

        # Map upside to grade
        if upside >= 30:
            grade = "A+"
        elif upside >= 20:
            grade = "A"
        elif upside >= 10:
            grade = "B"
        elif upside >= 0:
            grade = "C"
        elif upside >= -10:
            grade = "D"
        else:
            grade = "F"

        return ExtractionResult(
            value=grade,
            confidence=ExtractionConfidence.LOW,
            source_path=f"derived_from:{source_path}",
            fallback_used=True,
        )


class CurrentPriceExtractor(BaseFieldExtractor):
    """Extracts current stock price."""

    @property
    def field_name(self) -> str:
        return "current_price"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "fundamental", "ratios", "current_price"),
            ("agents", "fundamental", "company_data", "current_price"),
            ("agents", "technical", "current_price"),
            ("fundamental", "ratios", "current_price"),
            ("technical", "current_price"),
        ]

    def _validate_value(self, value: Any) -> bool:
        """Price must be positive number."""
        if not isinstance(value, (int, float)):
            return False
        return value > 0


class KeyStrengthsExtractor(BaseFieldExtractor):
    """
    Extracts key strengths/bull case items.

    Handles both list and dict formats for bull_case.
    Fallback chain:
    1. investment_thesis.bull_case (list of strings)
    2. investment_thesis.bull_case.key_assumptions (nested dict)
    3. investment_thesis.value_drivers (alternative field name)
    4. fundamental.analysis.strengths (fundamental agent output)
    """

    @property
    def field_name(self) -> str:
        return "key_strengths"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            # Synthesis paths
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "bull_case"),
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "value_drivers"),
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "strengths"),
            ("agents", "synthesis", "synthesis", "response", "scenarios", "bull_case", "key_assumptions"),
            # Alternative synthesis structures
            ("agents", "synthesis", "response", "investment_thesis", "bull_case"),
            ("agents", "synthesis", "investment_thesis", "bull_case"),
            # Fundamental agent fallbacks
            ("agents", "fundamental", "analysis", "response", "strengths"),
            ("agents", "fundamental", "analysis", "response", "competitive_advantages"),
            ("agents", "fundamental", "strengths"),
            # Without 'agents' wrapper
            ("synthesis", "synthesis", "response", "investment_thesis", "bull_case"),
            ("fundamental", "analysis", "response", "strengths"),
        ]

    def _transform_value(self, value: Any) -> List[str]:
        """Transform various formats to list of strings."""
        if value is None:
            return []

        # Already a list
        if isinstance(value, list):
            # Filter to strings and take first 3
            result = []
            for item in value[:5]:  # Check up to 5 items
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
                elif isinstance(item, dict):
                    # Extract text from dict items
                    text = item.get("description") or item.get("text") or item.get("assumption")
                    if text and isinstance(text, str):
                        result.append(text.strip())
                if len(result) >= 3:
                    break
            return result

        # Dict with key_assumptions or similar
        if isinstance(value, dict):
            # Try common list keys
            for key in ["key_assumptions", "assumptions", "drivers", "factors", "items"]:
                if key in value and isinstance(value[key], list):
                    return self._transform_value(value[key])

            # If dict has narrative keys, extract them
            narratives = []
            for key in ["narrative", "description", "summary"]:
                if key in value and isinstance(value[key], str):
                    narratives.append(value[key])
            if narratives:
                return narratives[:3]

            return []

        # Single string - wrap in list
        if isinstance(value, str) and value.strip():
            return [value.strip()]

        return []

    def _validate_value(self, value: Any) -> bool:
        """Must have at least one string item."""
        if not isinstance(value, list):
            return False
        return len(value) > 0


class KeyRisksExtractor(BaseFieldExtractor):
    """
    Extracts key risks from analysis.

    Fallback chain:
    1. risk_analysis.primary_risks
    2. risk_analysis.key_risks
    3. investment_thesis.bear_case
    4. fundamental.risks
    """

    @property
    def field_name(self) -> str:
        return "key_risks"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            # Synthesis paths
            ("agents", "synthesis", "synthesis", "response", "risk_analysis", "primary_risks"),
            ("agents", "synthesis", "synthesis", "response", "risk_analysis", "key_risks"),
            ("agents", "synthesis", "synthesis", "response", "risk_analysis", "risks"),
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "bear_case"),
            # Alternative structures
            ("agents", "synthesis", "response", "risk_analysis", "primary_risks"),
            ("agents", "synthesis", "risk_analysis", "primary_risks"),
            # Fundamental fallbacks
            ("agents", "fundamental", "analysis", "response", "risks"),
            ("agents", "fundamental", "risks"),
            # Without wrapper
            ("synthesis", "synthesis", "response", "risk_analysis", "primary_risks"),
        ]

    def _transform_value(self, value: Any) -> List[str]:
        """Transform to list of risk strings."""
        if value is None:
            return []

        if isinstance(value, list):
            result = []
            for item in value[:5]:
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("risk") or item.get("description") or item.get("text")
                    if text and isinstance(text, str):
                        result.append(text.strip())
                if len(result) >= 3:
                    break
            return result

        if isinstance(value, dict):
            for key in ["items", "risks", "factors"]:
                if key in value and isinstance(value[key], list):
                    return self._transform_value(value[key])
            return []

        if isinstance(value, str) and value.strip():
            return [value.strip()]

        return []

    def _validate_value(self, value: Any) -> bool:
        return isinstance(value, list) and len(value) > 0


class InvestmentThesisExtractor(BaseFieldExtractor):
    """Extracts core investment thesis statement."""

    @property
    def field_name(self) -> str:
        return "investment_thesis"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "core_thesis"),
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "thesis"),
            ("agents", "synthesis", "synthesis", "response", "investment_thesis", "summary"),
            ("agents", "synthesis", "response", "investment_thesis", "core_thesis"),
            ("agents", "synthesis", "investment_thesis", "core_thesis"),
            # Fallback to recommendation narrative
            ("agents", "synthesis", "synthesis", "response", "recommendation_and_action_plan", "rationale"),
            ("synthesis", "synthesis", "response", "investment_thesis", "core_thesis"),
        ]

    def _validate_value(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return len(value.strip()) > 10  # Must be meaningful text


class RecommendationExtractor(BaseFieldExtractor):
    """Extracts investment recommendation (BUY/HOLD/SELL)."""

    VALID_RECOMMENDATIONS = {
        "STRONG BUY",
        "BUY",
        "ACCUMULATE",
        "HOLD",
        "NEUTRAL",
        "REDUCE",
        "SELL",
        "STRONG SELL",
        "N/A",
    }

    @property
    def field_name(self) -> str:
        return "recommendation"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "synthesis", "synthesis", "response", "recommendation_and_action_plan", "recommendation"),
            ("agents", "synthesis", "synthesis", "response", "recommendation"),
            ("agents", "synthesis", "response", "recommendation"),
            ("synthesis", "synthesis", "response", "recommendation_and_action_plan", "recommendation"),
        ]

    def _transform_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value.upper().strip()
        return str(value).upper().strip() if value else None


class ConfidenceExtractor(BaseFieldExtractor):
    """Extracts confidence level."""

    @property
    def field_name(self) -> str:
        return "confidence"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "fundamental", "confidence", "confidence_level"),
            ("agents", "fundamental", "confidence", "overall"),
            ("agents", "synthesis", "synthesis", "response", "confidence_level"),
            ("fundamental", "confidence", "confidence_level"),
        ]


class DataQualityScoreExtractor(BaseFieldExtractor):
    """Extracts data quality score."""

    @property
    def field_name(self) -> str:
        return "data_quality_score"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "fundamental", "data_quality", "data_quality_score"),
            ("agents", "fundamental", "data_quality", "overall_score"),
            ("agents", "fundamental", "data_quality", "score"),
            ("fundamental", "data_quality", "data_quality_score"),
        ]

    def _validate_value(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        return 0 <= value <= 100


class TimeHorizonExtractor(BaseFieldExtractor):
    """Extracts investment time horizon."""

    @property
    def field_name(self) -> str:
        return "time_horizon"

    def _get_paths(self) -> List[Tuple[str, ...]]:
        return [
            ("agents", "synthesis", "synthesis", "response", "recommendation_and_action_plan", "time_horizon"),
            ("agents", "synthesis", "synthesis", "response", "time_horizon"),
            ("synthesis", "synthesis", "response", "recommendation_and_action_plan", "time_horizon"),
        ]


# =============================================================================
# Main Extractor Orchestrator
# =============================================================================


@dataclass
class SummaryExtractionAudit:
    """Audit trail for extraction process."""

    extractions: Dict[str, ExtractionResult] = field(default_factory=dict)

    def add(self, field_name: str, result: ExtractionResult):
        self.extractions[field_name] = result

    def get_summary(self) -> Dict[str, Any]:
        """Get extraction summary for debugging."""
        return {
            name: {
                "found": result.has_value,
                "confidence": result.confidence.value,
                "source": result.source_path,
                "fallback_used": result.fallback_used,
            }
            for name, result in self.extractions.items()
        }

    def log_summary(self):
        """Log extraction summary."""
        for name, result in self.extractions.items():
            status = "OK" if result.has_value else "MISSING"
            logger.debug(
                f"  [{status}] {name}: confidence={result.confidence.value}, "
                f"source={result.source_path}, fallback={result.fallback_used}"
            )


class SummaryDataExtractor:
    """
    Main orchestrator for summary data extraction.

    Uses Strategy Pattern to apply multiple extractors with fallback chains.
    Follows Dependency Inversion: depends on FieldExtractor protocol, not concrete classes.
    """

    def __init__(self, analysis_results: Dict[str, Any], enable_audit: bool = True):
        """
        Initialize extractor with analysis results.

        Args:
            analysis_results: Raw analysis results from orchestrator
            enable_audit: Whether to track extraction audit trail
        """
        self.data = analysis_results
        self.enable_audit = enable_audit
        self.audit = SummaryExtractionAudit() if enable_audit else None

        # Register default extractors (Open/Closed: add more without modifying)
        self._extractors: List[FieldExtractor] = [
            PriceTargetExtractor(),
            InvestmentGradeExtractor(),
            CurrentPriceExtractor(),
            KeyStrengthsExtractor(),
            KeyRisksExtractor(),
            InvestmentThesisExtractor(),
            RecommendationExtractor(),
            ConfidenceExtractor(),
            DataQualityScoreExtractor(),
            TimeHorizonExtractor(),
        ]

    def register_extractor(self, extractor: FieldExtractor) -> None:
        """Register additional extractor (Open/Closed principle)."""
        self._extractors.append(extractor)

    def extract_field(self, field_name: str) -> ExtractionResult:
        """Extract a single field by name."""
        for extractor in self._extractors:
            if extractor.field_name == field_name:
                result = extractor.extract(self.data)
                if self.audit:
                    self.audit.add(field_name, result)
                return result

        return ExtractionResult.not_found(f"no_extractor_for:{field_name}")

    def extract_all(self) -> Dict[str, Any]:
        """
        Extract all summary fields.

        Returns dict compatible with result_formatter's _format_minimal output.
        """
        results = {}

        for extractor in self._extractors:
            result = extractor.extract(self.data)
            if self.audit:
                self.audit.add(extractor.field_name, result)

            # Only include if has value
            if result.has_value:
                results[extractor.field_name] = result.value

        return results

    def extract_minimal_summary(self) -> Dict[str, Any]:
        """
        Extract fields specifically for minimal summary format.

        Returns structure matching _format_minimal() expectations.
        """
        # Extract all fields
        price_target = self.extract_field("price_target_12m")
        current_price = self.extract_field("current_price")
        investment_grade = self.extract_field("investment_grade")
        key_strengths = self.extract_field("key_strengths")
        key_risks = self.extract_field("key_risks")
        thesis = self.extract_field("investment_thesis")
        recommendation = self.extract_field("recommendation")
        confidence = self.extract_field("confidence")
        time_horizon = self.extract_field("time_horizon")
        data_quality = self.extract_field("data_quality_score")

        # Calculate expected return
        expected_return = None
        if price_target.has_value and current_price.has_value:
            if current_price.value > 0:
                expected_return = round((price_target.value - current_price.value) / current_price.value * 100, 2)

        return {
            "symbol": self.data.get("symbol"),
            "timestamp": self.data.get("timestamp", self.data.get("completed_at")),
            "detail_level": "minimal",
            "recommendation": {
                "action": recommendation.value if recommendation.has_value else "N/A",
                "confidence": confidence.value if confidence.has_value else "N/A",
                "time_horizon": time_horizon.value if time_horizon.has_value else "N/A",
            },
            "valuation": {
                "current_price": current_price.value if current_price.has_value else None,
                "price_target_12m": price_target.value if price_target.has_value else None,
                "expected_return_pct": expected_return,
                "investment_grade": investment_grade.value if investment_grade.has_value else "N/A",
            },
            "thesis": {
                "investment_thesis": thesis.value if thesis.has_value else "N/A",
                "key_strengths": key_strengths.value if key_strengths.has_value else [],
                "key_risks": key_risks.value if key_risks.has_value else [],
            },
            "data_quality": {
                "overall_score": data_quality.value if data_quality.has_value else None,
            },
            # Include extraction metadata for debugging
            "_extraction_audit": self.audit.get_summary() if self.audit else None,
        }

    def get_audit(self) -> Optional[SummaryExtractionAudit]:
        """Get extraction audit trail."""
        return self.audit


# Export public interface
__all__ = [
    "SummaryDataExtractor",
    "ExtractionResult",
    "ExtractionConfidence",
    "SummaryExtractionAudit",
    # Extractors (for extension)
    "FieldExtractor",
    "BaseFieldExtractor",
    "PriceTargetExtractor",
    "InvestmentGradeExtractor",
    "CurrentPriceExtractor",
    "KeyStrengthsExtractor",
    "KeyRisksExtractor",
]
