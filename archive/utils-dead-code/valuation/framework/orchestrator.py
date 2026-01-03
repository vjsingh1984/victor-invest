"""
Utilities for combining valuation model outputs into a blended fair value.

This module provides lightweight helpers that operate on the normalized
valuation results emitted by the wrapper classes so that consumers (agents,
CLI, API) can obtain a consistent blended output without reimplementing the
weighting logic.

Updated: 2025-11-07 - Refactored to use shared WeightNormalizer service
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Updated to use clean architecture imports (Phase 6 migration)
from investigator.domain.services.valuation.models.base import ModelNotApplicable, ValuationModelResult
from investigator.domain.services.valuation.models.company_profile import CompanyArchetype, CompanyProfile

logger = logging.getLogger(__name__)

# Import shared WeightNormalizer service
# Handle both legacy utils/ and new src/investigator/ paths
try:
    from investigator.domain.services.weight_normalizer import WeightNormalizer
except ImportError:
    # Fallback for when running without src/ in path
    sys.path.insert(0, "src")
    from investigator.domain.services.weight_normalizer import WeightNormalizer


def normalize_model_output(result: ValuationModelResult | ModelNotApplicable) -> Dict[str, Any]:
    """
    Convert valuation model results into a JSON-friendly dictionary.

    Ensures every model output shares a common structure so downstream code can
    rely on consistent keys regardless of the originating model.
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


class MultiModelValuationOrchestrator:
    """
    Combine individual model outputs into a blended valuation summary.

    The orchestrator expects already-normalized model outputs (typically from
    :func:`normalize_model_output`) so it can focus on weighting, agreement
    scoring, and diagnostics.

    Uses shared WeightNormalizer service for standardized weight normalization
    (5% increments, sum=100%).
    """

    def __init__(self, divergence_threshold: float = 0.35) -> None:
        self.divergence_threshold = divergence_threshold
        self.weight_normalizer = WeightNormalizer(rounding_increment=5)

    def combine(
        self,
        company_profile: CompanyProfile,
        model_outputs: Sequence[Dict[str, Any]],
        *,
        fallback_weights: Optional[Dict[str, float]] = None,
        tier_classification: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Blend applicable model outputs and return a consolidated summary.

        Args:
            company_profile: Company profile with financial metrics
            model_outputs: List of model results (normalized)
            fallback_weights: Optional tier-based weights (percentages: 0-100)
            tier_classification: Optional tier name from DynamicModelWeightingService
        """
        models = [copy.deepcopy(model) for model in model_outputs]
        applicable = [
            model
            for model in models
            if model.get("applicable") and isinstance(model.get("fair_value_per_share"), (int, float))
        ]

        for model in models:
            model.setdefault("weight", 0.0)

        if not applicable:
            return {
                "models": models,
                "blended_fair_value": None,
                "overall_confidence": 0.0,
                "model_agreement_score": None,
                "divergence_flag": False,
                "applicable_models": 0,
                "notes": ["No applicable valuation models produced fair values."],
            }

        confidences: List[float] = []
        for model in applicable:
            confidence = model.get("confidence_score") or 0.0
            confidences.append(max(float(confidence), 0.0))

        total_confidence = sum(confidences)

        # Build weights dict for normalization
        weights_dict = {}
        fallback_applied = False
        applied_weights: Dict[str, float] = {}

        missing_weight_targets: List[str] = []
        if fallback_weights:
            desired_models = {name: weight for name, weight in fallback_weights.items() if (weight or 0) > 0}
            for model_name, weight in desired_models.items():
                if not any(app.get("model") == model_name for app in applicable):
                    missing_weight_targets.append(f"{model_name.upper()} ({weight:.0f}%)")

        if total_confidence <= 0 and fallback_weights:
            # Use fallback weights (tier-based from DynamicWeightingService)
            matched = {
                model.get("model"): float(fallback_weights.get(model.get("model"), 0.0))
                for model in applicable
                if fallback_weights.get(model.get("model")) is not None
            }
            if sum(matched.values()) > 0:
                fallback_applied = True
                weights_dict = matched
                applied_weights = matched
        elif total_confidence <= 0:
            # Equal weighting fallback
            for model in applicable:
                weights_dict[model.get("model")] = 1.0 / len(applicable) * 100
        else:
            # Confidence-based weighting
            for model, confidence in zip(applicable, confidences):
                model_name = model.get("model")
                weights_dict[model_name] = (confidence / total_confidence) * 100

        # Normalize using shared service (standardize to 5% increments, sum=100%)
        try:
            normalized_weights = self.weight_normalizer.normalize(weights_dict)
            # Apply normalized weights to models
            for model in applicable:
                model_name = model.get("model")
                model["weight"] = normalized_weights.get(model_name, 0.0) / 100.0  # Convert back to 0-1 range
        except ValueError as e:
            # Fallback if normalization fails (shouldn't happen but handle gracefully)
            import logging

            logging.warning(f"Weight normalization failed: {e}, using equal weights")
            equal_weight = 1.0 / len(applicable)
            for model in applicable:
                model["weight"] = equal_weight

        blended_fair_value = sum(
            (model.get("fair_value_per_share") or 0.0) * model.get("weight", 0.0) for model in applicable
        )
        overall_confidence = sum(
            (model.get("confidence_score") or 0.0) * model.get("weight", 0.0) for model in applicable
        )

        # Enhanced blended valuation logging for visibility
        symbol = company_profile.symbol if hasattr(company_profile, "symbol") else "UNKNOWN"
        sector = company_profile.sector if hasattr(company_profile, "sector") else "N/A"
        industry = company_profile.industry if hasattr(company_profile, "industry") else None

        logger.info(f"💰 {symbol} - Blended Valuation Breakdown:")
        logger.info(f"   Tier: {tier_classification or 'N/A'} | Sector: {sector} | Industry: {industry or 'N/A'}")
        logger.info("")
        logger.info("   Model Contributions:")

        # Calculate total weight for normalization display
        total_weight = sum(model.get("weight", 0.0) * 100 for model in applicable)

        for model in applicable:
            model_name = model.get("model", "unknown").upper()
            fair_value = model.get("fair_value_per_share")
            weight_decimal = model.get("weight", 0.0)
            weight_pct = weight_decimal * 100
            contribution = fair_value * weight_decimal if fair_value else 0.0

            if fair_value is not None and fair_value > 0:
                status = f"${fair_value:.2f}"
            else:
                status = "N/A"

            logger.info(f"   - {model_name:<12} {status:>10} × {weight_pct:>5.1f}% = ${contribution:>8.2f}")

        logger.info("")
        logger.info(f"   Blended Fair Value: ${blended_fair_value:.2f}")
        if abs(total_weight - 100.0) > 0.01:
            logger.info(f"   Weight Normalization: {total_weight:.1f}% → 100%")
        logger.info("")

        fair_values = [float(model.get("fair_value_per_share")) for model in applicable]
        dispersion_ratio = self._calculate_dispersion(fair_values)

        model_agreement_score = None
        divergence_flag = False
        if dispersion_ratio is not None:
            model_agreement_score = max(0.0, 1.0 - dispersion_ratio)
            divergence_flag = dispersion_ratio > self.divergence_threshold

        notes: List[str] = []
        if missing_weight_targets:
            notes.append("Tier targets ignored for missing fair values → " + ", ".join(missing_weight_targets))
        if divergence_flag:
            notes.append(
                "Model fair values diverge beyond threshold "
                f"({dispersion_ratio:.2f} > {self.divergence_threshold:.2f}); investigate assumptions."
            )
        if overall_confidence < 0.5:
            notes.append("Overall confidence below 0.5; consider gathering additional data before acting.")
        if fallback_applied:
            applied_keys = [key for key, value in applied_weights.items() if value > 0]
            notes.append("Applied fallback weights from configuration to models: " + ", ".join(applied_keys))

        return {
            "models": models,
            "blended_fair_value": blended_fair_value,
            "overall_confidence": round(overall_confidence, 4),
            "model_agreement_score": None if model_agreement_score is None else round(model_agreement_score, 4),
            "dispersion_ratio": dispersion_ratio,
            "divergence_flag": divergence_flag,
            "applicable_models": len(applicable),
            "notes": notes,
            "primary_archetype": company_profile.primary_archetype.name if company_profile.primary_archetype else None,
            "fallback_applied": fallback_applied,
            "tier_classification": tier_classification,  # NEW: Pass through tier classification
        }

    @staticmethod
    def _calculate_dispersion(values: Iterable[float]) -> float | None:
        values = list(values)
        if len(values) < 2:
            return None
        mean_value = sum(values) / len(values)
        if math.isclose(mean_value, 0.0, abs_tol=1e-9):
            return float("inf")
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        stdev = math.sqrt(max(variance, 0.0))
        return stdev / abs(mean_value)
