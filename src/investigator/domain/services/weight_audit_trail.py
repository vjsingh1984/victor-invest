"""
Weight Audit Trail - Full traceability of weight evolution.

Provides:
1. Before/after snapshots at every adjustment step
2. Adjustment reason tracking
3. Summary generation for debugging
4. JSON export for analysis

Problem being solved:
- Dynamic model weighting applies multiple adjustment steps
- Hard to debug why weights ended up at certain values
- No visibility into intermediate calculations

Solution:
- Capture weights before and after each adjustment
- Track adjustment sources and multipliers
- Generate human-readable summaries

Usage:
    from investigator.domain.services.weight_audit_trail import WeightAuditTrail

    trail = WeightAuditTrail(symbol='AAPL')
    trail.capture(1, before_weights, after_weights, adjustments)
    trail.log_summary()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WeightAdjustment:
    """Single weight adjustment record."""
    model: str
    source: str  # e.g., 'market_context', 'data_quality', 'sector'
    multiplier: float
    reason: Optional[str] = None


@dataclass
class AuditStep:
    """A single step in the weight audit trail."""
    step_number: int
    step_name: str
    timestamp: str
    weights_before: Dict[str, float]
    weights_after: Dict[str, float]
    adjustments: List[WeightAdjustment]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_changes(self) -> Dict[str, Dict[str, float]]:
        """Get changes between before and after weights."""
        changes = {}
        for model in set(self.weights_before) | set(self.weights_after):
            before = self.weights_before.get(model, 0)
            after = self.weights_after.get(model, 0)
            if abs(after - before) > 0.01:
                changes[model] = {
                    'before': round(before, 2),
                    'after': round(after, 2),
                    'delta': round(after - before, 2),
                    'delta_pct': round((after - before) / before * 100, 1) if before > 0 else 0
                }
        return changes


@dataclass
class AuditSummary:
    """Summary of the full audit trail."""
    symbol: str
    total_steps: int
    initial_weights: Dict[str, float]
    final_weights: Dict[str, float]
    total_adjustments: int
    bounds_applied: bool
    largest_change: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


class WeightAuditTrail:
    """
    Tracks weight evolution through all adjustment steps.

    Captures before/after snapshots at each step with full
    adjustment details for debugging and analysis.

    Example:
        trail = WeightAuditTrail(symbol='AAPL')

        # Step 1: Base weights
        trail.capture(
            step_number=1,
            step_name='base_weights',
            weights_before={},
            weights_after={'dcf': 30, 'pe': 25, 'ps': 45},
            adjustments=[]
        )

        # Step 2: Market context
        trail.capture(
            step_number=2,
            step_name='market_context',
            weights_before={'dcf': 30, 'pe': 25, 'ps': 45},
            weights_after={'dcf': 24, 'pe': 22.5, 'ps': 49.5},
            adjustments=[
                WeightAdjustment('dcf', 'trend', 0.8, 'Bearish trend'),
                WeightAdjustment('pe', 'sentiment', 0.9, 'Neutral sentiment'),
            ]
        )

        trail.log_summary()
    """

    def __init__(self, symbol: str):
        """
        Initialize audit trail for a symbol.

        Args:
            symbol: Stock symbol being analyzed
        """
        self.symbol = symbol
        self.steps: List[AuditStep] = []
        self.bounds_applied = False
        self.started_at = datetime.now().isoformat()

    def capture(
        self,
        step_number: int,
        step_name: str,
        weights_before: Dict[str, float],
        weights_after: Dict[str, float],
        adjustments: List[WeightAdjustment],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Capture a snapshot of weights at a step.

        Args:
            step_number: Sequential step number
            step_name: Name of the adjustment step
            weights_before: Weights before this step
            weights_after: Weights after this step
            adjustments: List of adjustments applied
            metadata: Optional additional context
        """
        step = AuditStep(
            step_number=step_number,
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            weights_before=weights_before.copy(),
            weights_after=weights_after.copy(),
            adjustments=adjustments,
            metadata=metadata or {}
        )

        self.steps.append(step)

        # Log step summary
        changes = step.get_changes()
        if changes:
            logger.debug(
                f"[{self.symbol}] Step {step_number} ({step_name}): "
                f"{len(changes)} models changed"
            )
            for model, change in changes.items():
                logger.debug(
                    f"  {model}: {change['before']:.1f}% → {change['after']:.1f}% "
                    f"({change['delta']:+.1f}%)"
                )

    def mark_bounds_applied(self) -> None:
        """Mark that bounds were applied during weighting."""
        self.bounds_applied = True

    def get_summary(self) -> AuditSummary:
        """
        Get a summary of the full audit trail.

        Returns:
            AuditSummary with key metrics
        """
        if not self.steps:
            return AuditSummary(
                symbol=self.symbol,
                total_steps=0,
                initial_weights={},
                final_weights={},
                total_adjustments=0,
                bounds_applied=self.bounds_applied
            )

        # Get initial and final weights
        initial_weights = self.steps[0].weights_after if self.steps else {}
        final_weights = self.steps[-1].weights_after if self.steps else {}

        # Count total adjustments
        total_adjustments = sum(len(step.adjustments) for step in self.steps)

        # Find largest change
        largest_change = None
        max_delta = 0

        for step in self.steps:
            changes = step.get_changes()
            for model, change in changes.items():
                if abs(change['delta']) > max_delta:
                    max_delta = abs(change['delta'])
                    largest_change = {
                        'model': model,
                        'step': step.step_name,
                        **change
                    }

        # Collect warnings
        warnings = []
        for model in set(initial_weights) | set(final_weights):
            initial = initial_weights.get(model, 0)
            final = final_weights.get(model, 0)
            if initial > 0:
                change_pct = (final - initial) / initial * 100
                if abs(change_pct) > 50:
                    warnings.append(
                        f"{model}: Weight changed by {change_pct:+.0f}% "
                        f"({initial:.1f}% → {final:.1f}%)"
                    )

        return AuditSummary(
            symbol=self.symbol,
            total_steps=len(self.steps),
            initial_weights=initial_weights,
            final_weights=final_weights,
            total_adjustments=total_adjustments,
            bounds_applied=self.bounds_applied,
            largest_change=largest_change,
            warnings=warnings
        )

    def log_summary(self) -> None:
        """Log a human-readable summary of the audit trail."""
        summary = self.get_summary()

        logger.info(
            f"╔══════════════════════════════════════════════════════════╗"
        )
        logger.info(
            f"║ WEIGHT AUDIT TRAIL: {self.symbol:^38} ║"
        )
        logger.info(
            f"╠══════════════════════════════════════════════════════════╣"
        )

        # Steps overview
        logger.info(
            f"║ Steps: {summary.total_steps:<3}  Adjustments: {summary.total_adjustments:<3}  "
            f"Bounds: {'Yes' if summary.bounds_applied else 'No ':3}     ║"
        )

        # Initial weights
        logger.info(f"╠══════════════════════════════════════════════════════════╣")
        logger.info(f"║ INITIAL WEIGHTS:                                         ║")
        for model, weight in summary.initial_weights.items():
            logger.info(f"║   {model:12}: {weight:6.1f}%                               ║")

        # Final weights
        logger.info(f"╠══════════════════════════════════════════════════════════╣")
        logger.info(f"║ FINAL WEIGHTS:                                           ║")
        for model, weight in summary.final_weights.items():
            initial = summary.initial_weights.get(model, 0)
            delta = weight - initial
            logger.info(
                f"║   {model:12}: {weight:6.1f}% ({delta:+5.1f}%)                     ║"
            )

        # Largest change
        if summary.largest_change:
            lc = summary.largest_change
            logger.info(f"╠══════════════════════════════════════════════════════════╣")
            logger.info(
                f"║ LARGEST CHANGE: {lc['model']} in {lc['step'][:20]:20}       ║"
            )
            logger.info(
                f"║   {lc['before']:.1f}% → {lc['after']:.1f}% ({lc['delta']:+.1f}%)                            ║"
            )

        # Warnings
        if summary.warnings:
            logger.info(f"╠══════════════════════════════════════════════════════════╣")
            logger.info(f"║ WARNINGS ({len(summary.warnings)}):                                          ║")
            for warning in summary.warnings[:3]:  # Limit to 3
                logger.info(f"║   ⚠️  {warning[:50]:50} ║")

        logger.info(
            f"╚══════════════════════════════════════════════════════════╝"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Export audit trail as dictionary for JSON serialization.

        Returns:
            Dictionary representation of audit trail
        """
        summary = self.get_summary()

        return {
            'symbol': self.symbol,
            'started_at': self.started_at,
            'summary': {
                'total_steps': summary.total_steps,
                'total_adjustments': summary.total_adjustments,
                'bounds_applied': summary.bounds_applied,
                'initial_weights': summary.initial_weights,
                'final_weights': summary.final_weights,
                'largest_change': summary.largest_change,
                'warnings': summary.warnings,
            },
            'steps': [
                {
                    'step_number': step.step_number,
                    'step_name': step.step_name,
                    'timestamp': step.timestamp,
                    'weights_before': step.weights_before,
                    'weights_after': step.weights_after,
                    'changes': step.get_changes(),
                    'adjustments': [
                        {
                            'model': adj.model,
                            'source': adj.source,
                            'multiplier': adj.multiplier,
                            'reason': adj.reason,
                        }
                        for adj in step.adjustments
                    ],
                    'metadata': step.metadata,
                }
                for step in self.steps
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Export audit trail as JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def get_step(self, step_number: int) -> Optional[AuditStep]:
        """
        Get a specific step by number.

        Args:
            step_number: Step number to retrieve

        Returns:
            AuditStep or None if not found
        """
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_steps_for_model(self, model: str) -> List[AuditStep]:
        """
        Get all steps that affected a specific model.

        Args:
            model: Model name (e.g., 'dcf', 'pe')

        Returns:
            List of steps where this model's weight changed
        """
        result = []
        for step in self.steps:
            changes = step.get_changes()
            if model in changes:
                result.append(step)
        return result
