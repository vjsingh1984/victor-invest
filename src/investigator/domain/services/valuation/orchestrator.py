"""
Multi-Model Valuation Orchestrator (Clean Architecture)

Combines individual valuation model outputs into a blended fair value with
confidence-based and tier-based weighting. Uses shared WeightNormalizer service
for standardized weight normalization (5% increments, sum=100%).

Migration History:
- 2025-11-07: Refactored to use shared WeightNormalizer service
- 2025-11-14: Migrated to clean architecture (Phase 7)
- 2025-11-27: Integrated ModelAgreementScorer for divergence handling (M6)

Source: utils/valuation/framework/orchestrator.py (MultiModelValuationOrchestrator class)
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from investigator.domain.services.model_agreement_scorer import (
    AgreementConfig,
    AgreementLevel,
    ModelAgreementScorer,
)
from investigator.domain.services.valuation.bounds_checker import (
    BoundsChecker,
    ValidationSeverity,
    get_bounds_checker,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile
from investigator.domain.services.weight_normalizer import WeightNormalizer

logger = logging.getLogger(__name__)


class MultiModelValuationOrchestrator:
    """
    Combine individual model outputs into a blended valuation summary.

    The orchestrator expects already-normalized model outputs (typically from
    helpers.normalize_model_output()) so it can focus on weighting, agreement
    scoring, and diagnostics.

    Uses shared WeightNormalizer service for standardized weight normalization
    (5% increments, sum=100%).
    """

    def __init__(
        self,
        divergence_threshold: float = 0.35,
        agreement_config: Optional[Dict[str, Any]] = None,
        bounds_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.divergence_threshold = divergence_threshold
        self.weight_normalizer = WeightNormalizer(rounding_increment=5)

        # Initialize ModelAgreementScorer with config from valuation settings
        config_dict = agreement_config or {}
        self.agreement_scorer = ModelAgreementScorer(
            config=AgreementConfig(
                divergence_threshold=config_dict.get("divergence_threshold", divergence_threshold),
                high_agreement_threshold=config_dict.get("high_agreement_threshold", 0.15),
                zscore_threshold=config_dict.get("outlier_detection", {}).get("zscore_threshold", 2.0),
                outlier_weight_penalty=config_dict.get("outlier_detection", {}).get("outlier_weight_penalty", 0.50),
                divergence_penalty=config_dict.get("confidence_adjustments", {}).get("divergence_penalty", -0.15),
                high_agreement_bonus=config_dict.get("confidence_adjustments", {}).get("high_agreement_bonus", 0.10),
            )
        )
        self.apply_outlier_penalties = config_dict.get("outlier_detection", {}).get("enabled", True)

        # Initialize BoundsChecker for output validation (M7)
        self.bounds_checker = get_bounds_checker()
        self.validate_outputs = bounds_config.get("validate_outputs", True) if bounds_config else True

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

        Returns:
            Dictionary containing:
            - models: List of all model outputs with weights
            - blended_fair_value: Weighted average fair value
            - overall_confidence: Weighted average confidence
            - model_agreement_score: Agreement score (0.0-1.0)
            - divergence_flag: True if dispersion exceeds threshold
            - applicable_models: Count of applicable models
            - notes: List of diagnostic notes
            - primary_archetype: Primary company archetype name
            - fallback_applied: Whether fallback weights were used
            - tier_classification: Tier name (pass-through)
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

        # CRITICAL FIX: Prioritize dynamic weights (tier-based) over confidence-based weighting
        # Dynamic weights from DynamicModelWeightingService are more appropriate for company stage/sector
        # than generic confidence scores from individual models
        if fallback_weights:
            # Use dynamic weights (tier-based from DynamicWeightingService) - HIGHEST PRIORITY
            matched = {
                model.get("model"): float(fallback_weights.get(model.get("model"), 0.0))
                for model in applicable
                if fallback_weights.get(model.get("model")) is not None
            }
            if sum(matched.values()) > 0:
                fallback_applied = True
                weights_dict = matched
                applied_weights = matched
                logger.info(f"✅ Using tier-based dynamic weights from DynamicModelWeightingService")
        elif total_confidence > 0:
            # Confidence-based weighting (fallback when no dynamic weights provided)
            for model, confidence in zip(applicable, confidences):
                model_name = model.get("model")
                weights_dict[model_name] = (confidence / total_confidence) * 100
            logger.info(f"⚠️  Using confidence-based weights (no tier-based weights provided)")
        else:
            # Equal weighting fallback (when no dynamic weights and zero confidence)
            for model in applicable:
                weights_dict[model.get("model")] = 1.0 / len(applicable) * 100
            logger.info(f"⚠️  Using equal weights (no tier-based weights, zero confidence)")

        # Normalize using shared service (standardize to 5% increments, sum=100%)
        try:
            normalized_weights = self.weight_normalizer.normalize(weights_dict)
            # Apply normalized weights to models
            for model in applicable:
                model_name = model.get("model")
                model["weight"] = normalized_weights.get(model_name, 0.0) / 100.0  # Convert back to 0-1 range
        except ValueError as e:
            # Fallback if normalization fails (shouldn't happen but handle gracefully)
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

        # Use ModelAgreementScorer for enhanced divergence analysis (M6)
        model_fair_values = {
            model.get("model"): float(model.get("fair_value_per_share"))
            for model in applicable
            if model.get("fair_value_per_share") is not None
        }
        model_weights_for_agreement = {
            model.get("model"): model.get("weight", 0.0) * 100 for model in applicable  # Convert to percentage
        }

        agreement_result = self.agreement_scorer.analyze(
            model_fair_values=model_fair_values, symbol=symbol, model_weights=model_weights_for_agreement
        )

        # Apply outlier penalties if enabled
        effective_weights = model_weights_for_agreement
        if self.apply_outlier_penalties and agreement_result.outlier_models:
            effective_weights = self.agreement_scorer.apply_outlier_penalty(
                model_weights_for_agreement, agreement_result.outlier_models
            )

            # Re-apply adjusted weights to models
            for model in applicable:
                model_name = model.get("model")
                model["weight"] = effective_weights.get(model_name, 0.0) / 100.0

            # Recalculate blended fair value with adjusted weights
            blended_fair_value = sum(
                (model.get("fair_value_per_share") or 0.0) * model.get("weight", 0.0) for model in applicable
            )

            logger.info(
                f"[{symbol}] Outlier penalties applied to: {agreement_result.outlier_models}, "
                f"new blended fair value: ${blended_fair_value:.2f}"
            )

        # Apply confidence adjustment based on agreement
        overall_confidence = overall_confidence + agreement_result.confidence_adjustment
        overall_confidence = max(0.0, min(1.0, overall_confidence))  # Clamp 0-1

        # Use agreement scorer results
        dispersion_ratio = agreement_result.cv
        model_agreement_score = agreement_result.agreement_score
        divergence_flag = agreement_result.divergence_flag
        agreement_level = agreement_result.agreement_level

        notes: List[str] = []
        if missing_weight_targets:
            notes.append("Tier targets ignored for missing fair values → " + ", ".join(missing_weight_targets))
        if divergence_flag:
            notes.append(
                f"Model fair values diverge beyond threshold "
                f"(CV={dispersion_ratio:.0%} > {self.divergence_threshold:.0%}); investigate assumptions."
            )
        if agreement_result.outlier_models:
            notes.append(
                f"Outlier models detected: {', '.join(agreement_result.outlier_models)} "
                f"(z-score > {self.agreement_scorer.config.zscore_threshold})"
            )
        if agreement_result.confidence_adjustment != 0:
            notes.append(
                f"Confidence adjusted by {agreement_result.confidence_adjustment:+.0%} "
                f"due to {agreement_level.value} model agreement"
            )
        if overall_confidence < 0.5:
            notes.append("Overall confidence below 0.5; consider gathering additional data before acting.")
        if fallback_applied:
            applied_keys = [key for key, value in applied_weights.items() if value > 0]
            notes.append("Applied fallback weights from configuration to models: " + ", ".join(applied_keys))

        # Add agreement scorer notes
        notes.extend(agreement_result.notes)

        # Validate blended fair value against bounds (M7)
        bounds_validation = None
        if self.validate_outputs and blended_fair_value is not None:
            current_price = getattr(company_profile, "current_price", None)
            if current_price and current_price > 0:
                bounds_validation = self.bounds_checker.validate_output(
                    fair_value=blended_fair_value, current_price=current_price, model_type="blended", symbol=symbol
                )

                # Add validation issues to notes
                for issue in bounds_validation.issues:
                    if issue.severity == ValidationSeverity.WARNING:
                        notes.append(f"⚠️ Bounds warning: {issue.message}")
                    elif issue.severity == ValidationSeverity.ERROR:
                        notes.append(f"🚨 Bounds error: {issue.message}")
                        # Reduce confidence for bounds errors
                        overall_confidence = max(0.0, overall_confidence - 0.10)

                if not bounds_validation.is_valid:
                    logger.warning(
                        f"[{symbol}] Blended fair value ${blended_fair_value:.2f} "
                        f"failed bounds validation: {bounds_validation.summary()}"
                    )

        return {
            "models": models,
            "blended_fair_value": blended_fair_value,
            "overall_confidence": round(overall_confidence, 4),
            "model_agreement_score": round(model_agreement_score, 4) if model_agreement_score is not None else None,
            "dispersion_ratio": dispersion_ratio,
            "divergence_flag": divergence_flag,
            "agreement_level": agreement_level.value,
            "outlier_models": agreement_result.outlier_models,
            "applicable_models": len(applicable),
            "notes": notes,
            "primary_archetype": company_profile.primary_archetype.name if company_profile.primary_archetype else None,
            "fallback_applied": fallback_applied,
            "tier_classification": tier_classification,
            "model_z_scores": agreement_result.model_z_scores,
            "bounds_valid": bounds_validation.is_valid if bounds_validation else None,
        }

    @staticmethod
    def _calculate_dispersion(values: Iterable[float]) -> float | None:
        """
        Calculate coefficient of variation (CV) for model fair values.

        Returns the ratio of standard deviation to mean, which measures the
        degree of variation relative to the mean value.

        Args:
            values: Iterable of fair value estimates

        Returns:
            Coefficient of variation (CV), or None if < 2 values, or inf if mean is ~0
        """
        values = list(values)
        if len(values) < 2:
            return None
        mean_value = sum(values) / len(values)
        if math.isclose(mean_value, 0.0, abs_tol=1e-9):
            return float("inf")
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        stdev = math.sqrt(max(variance, 0.0))
        return stdev / abs(mean_value)
