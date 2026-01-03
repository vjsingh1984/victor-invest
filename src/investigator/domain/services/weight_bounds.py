"""
Weight Bounds System - Prevent weight collapse in dynamic model weighting.

Provides:
1. Cumulative multiplier bounds (floor/ceiling)
2. Per-model minimum floors
3. Graduated clamping with warnings
4. Integration with audit trail

Problem being solved:
- Market context multipliers can compound to extreme values
- e.g., 0.7 × 0.8 × 0.9 = 0.504 (nearly 50% reduction)
- This collapses model weights to unusable levels

Solution:
- Bound cumulative multipliers to configurable range (default: 0.50 - 1.50)
- Ensure minimum per-model weight (default: 5%)
- Log warnings when bounds are applied

Usage:
    from investigator.domain.services.weight_bounds import BoundedMultiplierApplicator

    applicator = BoundedMultiplierApplicator()
    adjusted, audit = applicator.apply_multipliers(
        base_weights={'dcf': 30, 'pe': 25, 'ps': 45},
        multiplier_groups={...},
        symbol='AAPL'
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class BoundAction(Enum):
    """Action taken when a bound is applied."""
    NONE = "none"
    FLOOR_APPLIED = "floor_applied"
    CEILING_APPLIED = "ceiling_applied"
    MINIMUM_APPLIED = "minimum_applied"


@dataclass
class MultiplierAuditEntry:
    """Audit entry for a single multiplier application."""
    model: str
    group: str
    original_multiplier: float
    applied_multiplier: float
    action: BoundAction
    reason: Optional[str] = None


@dataclass
class BoundedMultiplierResult:
    """Result of bounded multiplier application."""
    adjusted_weights: Dict[str, float]
    cumulative_multipliers: Dict[str, float]
    audit_entries: List[MultiplierAuditEntry]
    warnings: List[str]
    bounds_applied: bool

    def summary(self) -> str:
        """Get a summary of the bounded application."""
        num_bounded = sum(1 for e in self.audit_entries if e.action != BoundAction.NONE)
        return (
            f"Weights adjusted: {len(self.adjusted_weights)}, "
            f"Bounds applied: {num_bounded}, "
            f"Warnings: {len(self.warnings)}"
        )


@dataclass
class BoundConfig:
    """Configuration for weight bounds."""
    cumulative_floor: float = 0.50      # Never below 50% of base
    cumulative_ceiling: float = 1.50    # Never above 150% of base
    per_model_minimum: float = 5.0      # Minimum 5% weight per model
    warning_threshold: float = 0.70     # Warn if approaching bounds
    normalization_target: float = 100.0 # Target sum for normalized weights


class BoundedMultiplierApplicator:
    """
    Applies multipliers with bounded cumulative effects.

    Prevents weight collapse by enforcing:
    1. Cumulative multiplier bounds (0.50 - 1.50 default)
    2. Per-model minimum weights (5% default)
    3. Graduated clamping with warnings

    Example:
        applicator = BoundedMultiplierApplicator()

        base_weights = {'dcf': 30, 'pe': 25, 'ps': 45}
        multiplier_groups = {
            'market_context': {'dcf': 0.8, 'pe': 0.9, 'ps': 1.1},
            'data_quality': {'dcf': 0.9, 'pe': 1.0, 'ps': 0.85},
        }

        result = applicator.apply_multipliers(base_weights, multiplier_groups, 'AAPL')
        # Result enforces bounds even if cumulative multiplier would collapse weights
    """

    def __init__(self, config: Optional[BoundConfig] = None):
        """
        Initialize with optional custom configuration.

        Args:
            config: BoundConfig with custom bounds (uses defaults if not provided)
        """
        self.config = config or BoundConfig()

    def apply_multipliers(
        self,
        base_weights: Dict[str, float],
        multiplier_groups: Dict[str, Dict[str, float]],
        symbol: str
    ) -> BoundedMultiplierResult:
        """
        Apply multipliers with bounded cumulative effects.

        Args:
            base_weights: Base model weights (e.g., {'dcf': 30, 'pe': 25, 'ps': 45})
            multiplier_groups: Grouped multipliers by source
                e.g., {
                    'market_context': {'dcf': 0.8, 'pe': 0.9, 'ps': 1.1},
                    'data_quality': {'dcf': 0.9, 'pe': 1.0, 'ps': 0.85},
                }
            symbol: Stock symbol for logging

        Returns:
            BoundedMultiplierResult with adjusted weights and audit info
        """
        audit_entries: List[MultiplierAuditEntry] = []
        warnings: List[str] = []
        bounds_applied = False

        # Calculate cumulative multipliers for each model
        cumulative_multipliers: Dict[str, float] = {}

        for model in base_weights:
            cumulative = 1.0

            for group_name, group_multipliers in multiplier_groups.items():
                if model in group_multipliers:
                    multiplier = group_multipliers[model]
                    original_cumulative = cumulative
                    cumulative *= multiplier

                    # Track each step
                    audit_entries.append(MultiplierAuditEntry(
                        model=model,
                        group=group_name,
                        original_multiplier=multiplier,
                        applied_multiplier=multiplier,
                        action=BoundAction.NONE
                    ))

            cumulative_multipliers[model] = cumulative

        # Apply bounds to cumulative multipliers
        bounded_multipliers: Dict[str, float] = {}

        for model, cumulative in cumulative_multipliers.items():
            bounded = cumulative
            action = BoundAction.NONE
            reason = None

            # Check floor
            if cumulative < self.config.cumulative_floor:
                bounded = self.config.cumulative_floor
                action = BoundAction.FLOOR_APPLIED
                reason = (
                    f"Cumulative multiplier {cumulative:.3f} below floor {self.config.cumulative_floor}"
                )
                bounds_applied = True
                warnings.append(
                    f"[{symbol}] {model}: {reason}"
                )
                logger.warning(f"[{symbol}] {model}: {reason}")

            # Check ceiling
            elif cumulative > self.config.cumulative_ceiling:
                bounded = self.config.cumulative_ceiling
                action = BoundAction.CEILING_APPLIED
                reason = (
                    f"Cumulative multiplier {cumulative:.3f} above ceiling {self.config.cumulative_ceiling}"
                )
                bounds_applied = True
                warnings.append(
                    f"[{symbol}] {model}: {reason}"
                )
                logger.warning(f"[{symbol}] {model}: {reason}")

            # Check warning threshold (approaching bounds)
            elif cumulative < self.config.warning_threshold:
                warnings.append(
                    f"[{symbol}] {model}: Approaching floor (multiplier: {cumulative:.3f})"
                )
                logger.debug(f"[{symbol}] {model}: Approaching floor (multiplier: {cumulative:.3f})")

            bounded_multipliers[model] = bounded

            if action != BoundAction.NONE:
                audit_entries.append(MultiplierAuditEntry(
                    model=model,
                    group="cumulative_bound",
                    original_multiplier=cumulative,
                    applied_multiplier=bounded,
                    action=action,
                    reason=reason
                ))

        # Apply bounded multipliers to base weights
        adjusted_weights: Dict[str, float] = {}

        for model, base_weight in base_weights.items():
            multiplier = bounded_multipliers.get(model, 1.0)
            adjusted = base_weight * multiplier
            adjusted_weights[model] = adjusted

        # Enforce per-model minimum
        for model, weight in adjusted_weights.items():
            if weight < self.config.per_model_minimum:
                reason = (
                    f"Weight {weight:.2f}% below minimum {self.config.per_model_minimum}%"
                )
                warnings.append(f"[{symbol}] {model}: {reason}")
                logger.warning(f"[{symbol}] {model}: {reason}")

                adjusted_weights[model] = self.config.per_model_minimum
                bounds_applied = True

                audit_entries.append(MultiplierAuditEntry(
                    model=model,
                    group="minimum_weight",
                    original_multiplier=weight / base_weights[model] if base_weights[model] > 0 else 0,
                    applied_multiplier=self.config.per_model_minimum / base_weights[model] if base_weights[model] > 0 else 0,
                    action=BoundAction.MINIMUM_APPLIED,
                    reason=reason
                ))

        # Normalize to target (default 100%)
        total = sum(adjusted_weights.values())
        if total > 0 and abs(total - self.config.normalization_target) > 0.1:
            scale = self.config.normalization_target / total
            adjusted_weights = {
                model: weight * scale
                for model, weight in adjusted_weights.items()
            }

        return BoundedMultiplierResult(
            adjusted_weights=adjusted_weights,
            cumulative_multipliers=cumulative_multipliers,
            audit_entries=audit_entries,
            warnings=warnings,
            bounds_applied=bounds_applied
        )

    def validate_weights(
        self,
        weights: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that weights meet all bounds constraints.

        Args:
            weights: Model weights to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check per-model minimum
        for model, weight in weights.items():
            if weight < self.config.per_model_minimum:
                issues.append(
                    f"{model}: Weight {weight:.2f}% below minimum {self.config.per_model_minimum}%"
                )

        # Check sum is approximately 100%
        total = sum(weights.values())
        if abs(total - self.config.normalization_target) > 1.0:
            issues.append(
                f"Total weight {total:.2f}% deviates from target {self.config.normalization_target}%"
            )

        return (len(issues) == 0, issues)


# Default config for import
DEFAULT_BOUND_CONFIG = BoundConfig()

# Singleton instance
_applicator: Optional[BoundedMultiplierApplicator] = None


def get_bounded_multiplier_applicator() -> BoundedMultiplierApplicator:
    """Get the singleton BoundedMultiplierApplicator instance."""
    global _applicator
    if _applicator is None:
        _applicator = BoundedMultiplierApplicator()
    return _applicator
