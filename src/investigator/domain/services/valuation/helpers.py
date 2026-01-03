"""
Valuation Helper Functions

Utilities for normalizing and serializing valuation model outputs.

These functions were extracted from the legacy orchestrator to support clean
architecture migration while maintaining backward compatibility with existing
fundamental agent code.

Migration Date: 2025-11-14
Phase: Phase 6 (Orchestrator Helpers Migration)
Source: utils/valuation/framework/orchestrator.py (lines 36-86)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from investigator.domain.services.valuation.models.base import (
    ModelNotApplicable,
    ValuationModelResult,
)
from investigator.domain.services.valuation.models.company_profile import (
    CompanyArchetype,
    CompanyProfile,
)


def normalize_model_output(result: ValuationModelResult | ModelNotApplicable) -> Dict[str, Any]:
    """
    Convert valuation model results into a JSON-friendly dictionary.

    Ensures every model output shares a common structure so downstream code can
    rely on consistent keys regardless of the originating model.

    Args:
        result: ValuationModelResult or ModelNotApplicable from a valuation model

    Returns:
        Dictionary with normalized structure containing:
        - model: Model identifier
        - methodology: Methodology description
        - applicable: Whether model produced a valuation
        - fair_value_per_share: Fair value estimate (None if not applicable)
        - confidence_score: Confidence score (0.0-1.0)
        - weight: Model weight in blended valuation
        - assumptions: Model assumptions dict
        - diagnostics: ModelDiagnostics as dict
        - metadata: Additional metadata (if applicable)
    """
    if isinstance(result, ModelNotApplicable):
        return {
            "model": result.model_name,
            "methodology": result.model_name,
            "applicable": False,
            "reason": result.reason,
            "fair_value_per_share": None,
            "confidence_score": 0.0,
            "weight": 0.0,
            "assumptions": {},
            "diagnostics": asdict(result.diagnostics),
        }

    return {
        "model": result.model_name,
        "methodology": result.methodology or result.model_name,
        "applicable": True,
        "fair_value_per_share": result.fair_value,
        "confidence_score": result.confidence_score,
        "weight": result.weight,
        "assumptions": result.assumptions or {},
        "diagnostics": asdict(result.diagnostics),
        "metadata": result.metadata or {},
    }


def serialize_company_profile(profile: CompanyProfile) -> Dict[str, Any]:
    """
    Serialize CompanyProfile to a dictionary with human-readable archetype and data flags.

    Converts enum values to their string names and includes convenience fields
    for archetype labels.

    Args:
        profile: CompanyProfile instance to serialize

    Returns:
        Dictionary with all profile fields, where:
        - primary_archetype: Archetype name as string (or None)
        - secondary_archetype: Archetype name as string (or None)
        - data_quality_flags: List of flag names as strings
        - archetype_labels: List of human-readable archetype labels
    """
    payload = asdict(profile)

    def _enum_name(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, CompanyArchetype):
            return value.name
        return value

    payload["primary_archetype"] = _enum_name(profile.primary_archetype)
    payload["secondary_archetype"] = _enum_name(profile.secondary_archetype)
    payload["data_quality_flags"] = [flag.name for flag in profile.data_quality_flags]
    payload["archetype_labels"] = profile.archetype_labels()
    return payload
