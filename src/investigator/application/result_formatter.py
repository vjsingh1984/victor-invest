"""
Result Formatter - Configurable Output Detail Levels

Provides configurable output formatting to reduce duplication and verbosity
in analysis results. Supports three detail levels:

- MINIMAL: Executive summary only (for quick decisions)
- STANDARD: Investor decision-making details (default, removes duplicates/metadata)
- VERBOSE: Full analysis with all metadata and prompts

Usage:
    from investigator.application.result_formatter import format_analysis_output, OutputDetailLevel

    formatted = format_analysis_output(raw_results, OutputDetailLevel.STANDARD)
"""

import copy
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from investigator.application.summary_data_extractor import SummaryDataExtractor

logger = logging.getLogger(__name__)


def _is_empty_value(value: Any) -> bool:
    """
    Check if a value should be considered "empty" for removal.

    Handles numpy arrays safely (they raise ValueError on ambiguous truth checks).

    Empty values:
    - None
    - Empty string ""
    - Empty list []
    - Empty dict {}
    - Empty numpy array (size == 0)

    NOT empty (retained):
    - Non-empty numpy arrays
    - Zero values (0, 0.0)
    - False boolean
    - Non-empty collections

    Returns:
        True if value should be removed, False otherwise.
    """
    # Handle None first (simplest case)
    if value is None:
        return True

    # Handle numpy arrays and array-like objects (they fail on ambiguous truth comparisons)
    # Check for ndarray explicitly first
    if isinstance(value, np.ndarray):
        return value.size == 0

    # Check for any object with __array__ method (numpy-compatible objects)
    # This catches masked arrays, memoryviews, etc.
    if hasattr(value, "__array__") or hasattr(value, "size"):
        try:
            # Try to check size attribute for array-like objects
            if hasattr(value, "size"):
                return value.size == 0
            # Convert to numpy array and check
            arr = np.asarray(value)
            return arr.size == 0
        except (ValueError, TypeError):
            # If conversion fails, keep the value
            return False

    # Handle pandas objects if present (they also have ambiguous truth values)
    # Check for pandas-like objects by duck typing (has 'empty' attribute)
    if hasattr(value, "empty"):
        try:
            empty_attr = getattr(value, "empty", None)
            # Check if it's a property/attribute (not callable)
            if not callable(empty_attr):
                return bool(empty_attr)
        except (ValueError, AttributeError):
            # If empty check fails, keep the value
            return False

    # Standard Python types - use type checks to avoid __eq__ issues
    if isinstance(value, str):
        return value == ""
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0

    return False


class OutputDetailLevel(Enum):
    """Output detail level for analysis results"""

    MINIMAL = "minimal"  # Executive summary only
    STANDARD = "standard"  # Investor decision-making (default, no duplicates)
    VERBOSE = "verbose"  # Full analysis with all metadata


def format_analysis_output(
    analysis_results: Dict[str, Any], detail_level: OutputDetailLevel = OutputDetailLevel.STANDARD
) -> Dict[str, Any]:
    """
    Format analysis results according to specified detail level.

    Args:
        analysis_results: Raw analysis results from orchestrator
        detail_level: Desired output detail level

    Returns:
        Formatted analysis results

    Example:
        >>> results = await orchestrator.get_results(task_id)
        >>> formatted = format_analysis_output(results, OutputDetailLevel.STANDARD)
        >>> # Output is ~95% smaller, removes duplicates and metadata
    """
    if detail_level == OutputDetailLevel.VERBOSE:
        # Return full analysis unchanged
        return analysis_results

    elif detail_level == OutputDetailLevel.MINIMAL:
        # Return executive summary only
        return _format_minimal(analysis_results)

    else:  # STANDARD (default)
        # Remove duplicates, prompts, and metadata
        return _format_standard(analysis_results)


def _format_minimal(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract minimal executive summary only.

    Uses SummaryDataExtractor for robust field extraction with fallback chains.

    Returns only critical decision-making data:
    - Symbol, timestamp
    - Recommendation, confidence, price target
    - Key strengths and risks
    - Investment thesis

    The extractor handles:
    - Field name variations (fair_value vs price_target_12_month)
    - Nested structure differences
    - Missing data with fallback calculations (e.g., investment_grade from upside%)
    """
    # Use SOLID-based extractor with fallback chains
    extractor = SummaryDataExtractor(analysis_results, enable_audit=True)
    summary = extractor.extract_minimal_summary()

    # Log extraction audit for debugging if issues occur
    audit = extractor.get_audit()
    if audit:
        missing_fields = [name for name, result in audit.extractions.items() if not result.has_value]
        if missing_fields:
            logger.debug(f"Summary extraction missing fields: {missing_fields}")
            audit.log_summary()

    # Remove internal audit from output (keep it clean for display)
    summary.pop("_extraction_audit", None)

    # Add data quality assessment if score is available
    if summary.get("data_quality", {}).get("overall_score") is not None:
        score = summary["data_quality"]["overall_score"]
        if score >= 80:
            summary["data_quality"]["assessment"] = "Excellent"
        elif score >= 60:
            summary["data_quality"]["assessment"] = "Good"
        elif score >= 40:
            summary["data_quality"]["assessment"] = "Fair"
        else:
            summary["data_quality"]["assessment"] = "Limited"

    return summary


def _format_standard(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format for investor decision-making (removes duplicates and metadata).

    Removes:
    - All prompts
    - Internal metadata (cached_at, agent_id, raw_thinking, prompt_length, model_info)
    - Duplicate data points (signals, valuation, company data repeated across sections)
    - Empty/null fields

    Keeps:
    - All analysis conclusions and insights
    - Financial data and ratios
    - Recommendations and targets
    - Risk analysis
    - Data quality scores
    """
    # Deep copy to avoid modifying original
    result = copy.deepcopy(analysis_results)

    # Remove top-level metadata
    _remove_keys(result, ["task_id", "execution_metadata", "execution_trace"])

    # Clean each agent section (handle both direct and 'agents' wrapper)
    agents_dict = result.get("agents", result)  # Support both structures
    for agent_name in ["fundamental", "technical", "synthesis", "market_context", "sec", "symbol_update"]:
        if agent_name in agents_dict:
            _clean_agent_section(agents_dict[agent_name])

    # Consolidate duplicate data (keep single source of truth)
    _consolidate_duplicates(result)

    # Remove empty/null values
    result = _remove_empty_values(result)

    # Add detail level indicator
    result["detail_level"] = "standard"

    return result


def _clean_agent_section(agent_data: Dict[str, Any]) -> None:
    """
    Clean agent section by removing metadata and prompts.

    Modifies agent_data in-place.
    """
    # Remove agent metadata
    _remove_keys(
        agent_data,
        [
            "agent_id",
            "task_id",
            "cached_at",
            "cache_hit",
            "execution_time",
            "model_info",
            "prompt_length",
            "raw_thinking",
            "full_prompt",
            "system_prompt",
            "user_prompt",
        ],
    )

    # Clean ALL nested sections (analysis, valuation, ratios, confidence, data_quality, etc.)
    for key, value in list(agent_data.items()):
        if isinstance(value, dict):
            # Remove prompts and metadata from this section
            _remove_keys(
                value,
                [
                    "prompt",
                    "raw_thinking",
                    "cached_at",
                    "agent_id",
                    "model_info",
                    "metadata",
                    "prompt_length",
                ],
            )

            # If this section has a 'response' dict, clean it recursively
            if "response" in value and isinstance(value["response"], dict):
                _clean_analysis_section(value["response"])


def _clean_analysis_section(analysis_data: Dict[str, Any]) -> None:
    """
    Clean nested analysis sections.

    Modifies analysis_data in-place.
    """
    # Remove prompts and metadata from nested response objects
    for key, value in list(analysis_data.items()):
        if isinstance(value, dict):
            _remove_keys(
                value,
                [
                    "prompt",
                    "raw_thinking",
                    "cached_at",
                    "agent_id",
                    "model",
                    "model_info",
                    "metadata",
                    "temperature",
                    "max_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ],
            )

            # Recursively clean nested 'response' objects
            if "response" in value and isinstance(value["response"], dict):
                _clean_analysis_section(value["response"])


def _consolidate_duplicates(result: Dict[str, Any]) -> None:
    """
    Remove duplicate data points across sections.

    Strategy:
    - Keep company data in fundamental only
    - Keep valuation in fundamental only
    - Keep signals in technical only
    - Synthesis references but doesn't duplicate

    Modifies result in-place.
    """
    # Remove company data duplicates from synthesis
    if "synthesis" in result:
        synthesis = result["synthesis"]
        if isinstance(synthesis, dict) and "synthesis" in synthesis:
            synth_data = synthesis["synthesis"]
            if isinstance(synth_data, dict):
                _remove_keys(synth_data, ["company_data", "market_data"])

                # Keep only references in response, not full data
                if "response" in synth_data and isinstance(synth_data["response"], dict):
                    synth_response = synth_data["response"]
                    _remove_keys(
                        synth_response,
                        [
                            "financial_data",
                            "technical_signals",
                            "valuation_details",
                            "complete_ratios",
                        ],
                    )


def _remove_keys(data: Dict[str, Any], keys: List[str]) -> None:
    """
    Remove specified keys from dictionary.

    Modifies data in-place.
    """
    for key in keys:
        data.pop(key, None)


def _remove_empty_values(data: Any) -> Any:
    """
    Recursively remove empty/null values from data structure.

    Handles numpy arrays and pandas objects safely (they raise ValueError
    on ambiguous truth comparisons like `arr in [None, [], {}]`).

    Returns cleaned data structure.
    """
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            # Skip empty values
            if _is_empty_value(v):
                continue
            # Skip zero scores (but keep other zero values)
            if isinstance(v, (int, float)) and v == 0 and k.endswith("_score"):
                continue
            # Recursively clean and add
            cleaned[k] = _remove_empty_values(v)
        return cleaned
    elif isinstance(data, list):
        return [_remove_empty_values(item) for item in data if not _is_empty_value(item)]
    elif isinstance(data, np.ndarray):
        # Convert numpy arrays to lists for JSON serialization
        return data.tolist() if data.size > 0 else []
    else:
        return data


def _calculate_expected_return(target_price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    """Calculate expected return percentage."""
    if target_price and current_price and current_price > 0:
        return round((target_price - current_price) / current_price * 100, 2)
    return None


def _extract_list(data: Dict[str, Any], key: str, max_items: int = 3) -> List[str]:
    """Extract list from dict, limiting to max_items."""
    items = data.get(key, [])
    if isinstance(items, list):
        return items[:max_items]
    return []


# Export public interface
__all__ = [
    "OutputDetailLevel",
    "format_analysis_output",
]
