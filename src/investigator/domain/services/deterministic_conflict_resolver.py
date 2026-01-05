"""
Deterministic Conflict Resolver - Replaces LLM-based conflict reconciliation.

This module provides rule-based conflict detection and resolution that replaces
the LLM call in `_reconcile_conflicts()`. It uses predefined rules and patterns
to identify, explain, and resolve analytical conflicts.

Conflict Types Handled:
- Recommendation conflicts (fundamental vs technical)
- Time horizon conflicts (short-term vs long-term)
- Valuation vs momentum conflicts
- Sentiment vs fundamentals conflicts
- Data quality conflicts

Design Principles (SOLID):
- Single Responsibility: Each resolver handles one conflict type
- Open/Closed: New conflict types can be added via registry
- Liskov Substitution: All resolvers implement common protocol
- Interface Segregation: Focused interfaces for detection vs resolution
- Dependency Inversion: Depends on abstractions, not concretions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of analytical conflicts."""

    RECOMMENDATION = "recommendation_conflict"
    TIME_HORIZON = "time_horizon_conflict"
    VALUATION_MOMENTUM = "valuation_momentum_conflict"
    SENTIMENT_FUNDAMENTALS = "sentiment_fundamentals_conflict"
    DATA_QUALITY = "data_quality_conflict"
    MODEL_DIVERGENCE = "model_divergence_conflict"
    SECTOR_CONTEXT = "sector_context_conflict"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""

    LOW = "low"  # Minor discrepancy, doesn't affect recommendation
    MEDIUM = "medium"  # Notable conflict, requires explanation
    HIGH = "high"  # Significant conflict, may change recommendation
    CRITICAL = "critical"  # Severe conflict, warrants caution


@dataclass
class Conflict:
    """Represents a detected analytical conflict."""

    conflict_type: ConflictType
    severity: ConflictSeverity
    sources: List[str]  # e.g., ["fundamental", "technical"]
    description: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Resolution for a detected conflict."""

    conflict_type: str
    explanation: str
    prioritization: str
    weight_adjustments: Dict[str, float]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "conflict_type": self.conflict_type,
            "explanation": self.explanation,
            "prioritization": self.prioritization,
            "weight_adjustments": self.weight_adjustments,
            "rationale": self.rationale,
        }


@dataclass
class ReconciliationResult:
    """Complete result of conflict reconciliation."""

    overall_coherence: str
    reconciled_recommendation: str
    confidence_impact: float  # Adjustment to overall confidence (-1 to 1)
    resolutions: List[ConflictResolution]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "overall_coherence": self.overall_coherence,
            "reconciled_recommendation": self.reconciled_recommendation,
            "confidence_impact": self.confidence_impact,
            "resolutions": [r.to_dict() for r in self.resolutions],
        }


class ConflictDetector(Protocol):
    """Protocol for detecting conflicts in analysis results."""

    def detect(
        self,
        fundamental: Optional[Dict[str, Any]],
        technical: Optional[Dict[str, Any]],
        sec: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> List[Conflict]:
        """Detect conflicts between analyses."""
        ...


class ConflictResolver(Protocol):
    """Protocol for resolving a specific type of conflict."""

    def can_resolve(self, conflict: Conflict) -> bool:
        """Check if this resolver handles the given conflict type."""
        ...

    def resolve(self, conflict: Conflict) -> ConflictResolution:
        """Resolve the conflict and return resolution."""
        ...


# ============================================================================
# Conflict Detectors
# ============================================================================


class RecommendationConflictDetector:
    """Detects conflicts between recommendation signals."""

    # Signal interpretation
    BULLISH_SIGNALS = {"buy", "strong_buy", "outperform", "overweight", "bullish"}
    BEARISH_SIGNALS = {"sell", "strong_sell", "underperform", "underweight", "bearish"}
    NEUTRAL_SIGNALS = {"hold", "neutral", "market_perform", "equal_weight"}

    def detect(
        self,
        fundamental: Optional[Dict[str, Any]],
        technical: Optional[Dict[str, Any]],
        sec: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> List[Conflict]:
        """Detect recommendation conflicts between analyses."""
        conflicts = []
        signals = self._extract_signals(fundamental, technical, sec)

        # Check for fundamental vs technical conflict
        if signals.get("fundamental") and signals.get("technical"):
            fund_type = self._classify_signal(signals["fundamental"])
            tech_type = self._classify_signal(signals["technical"])

            if fund_type == "bullish" and tech_type == "bearish":
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.RECOMMENDATION,
                        severity=ConflictSeverity.HIGH,
                        sources=["fundamental", "technical"],
                        description=(
                            f"Fundamental analysis suggests {signals['fundamental']} "
                            f"while technical analysis indicates {signals['technical']}"
                        ),
                        data={
                            "fundamental_signal": signals["fundamental"],
                            "technical_signal": signals["technical"],
                            "direction": "fundamental_bullish_technical_bearish",
                        },
                    )
                )
            elif fund_type == "bearish" and tech_type == "bullish":
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.RECOMMENDATION,
                        severity=ConflictSeverity.HIGH,
                        sources=["fundamental", "technical"],
                        description=(
                            f"Fundamental analysis suggests {signals['fundamental']} "
                            f"while technical analysis indicates {signals['technical']}"
                        ),
                        data={
                            "fundamental_signal": signals["fundamental"],
                            "technical_signal": signals["technical"],
                            "direction": "fundamental_bearish_technical_bullish",
                        },
                    )
                )

        return conflicts

    def _extract_signals(
        self, fundamental: Optional[Dict[str, Any]], technical: Optional[Dict[str, Any]], sec: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Extract recommendation signals from each analysis."""
        signals = {}

        if fundamental:
            # Try multiple paths for recommendation
            rec = (
                fundamental.get("recommendation")
                or fundamental.get("analysis", {}).get("recommendation")
                or fundamental.get("valuation", {}).get("recommendation")
            )
            if rec:
                signals["fundamental"] = str(rec).lower()

        if technical:
            # Technical signal from trend or rating
            signal = (
                technical.get("signal")
                or technical.get("overall_signal")
                or technical.get("analysis", {}).get("signal")
            )
            if signal:
                signals["technical"] = str(signal).lower()

        if sec:
            # SEC analysis may have quality-based signal
            rating = sec.get("analysis", {}).get("overall_rating")
            if rating:
                if rating >= 7:
                    signals["sec"] = "positive"
                elif rating <= 4:
                    signals["sec"] = "negative"
                else:
                    signals["sec"] = "neutral"

        return signals

    def _classify_signal(self, signal: str) -> str:
        """Classify signal as bullish, bearish, or neutral."""
        signal_lower = signal.lower()
        if any(s in signal_lower for s in self.BULLISH_SIGNALS):
            return "bullish"
        elif any(s in signal_lower for s in self.BEARISH_SIGNALS):
            return "bearish"
        return "neutral"


class TimeHorizonConflictDetector:
    """Detects conflicts between short-term and long-term outlooks."""

    def detect(
        self,
        fundamental: Optional[Dict[str, Any]],
        technical: Optional[Dict[str, Any]],
        sec: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> List[Conflict]:
        """Detect time horizon conflicts."""
        conflicts = []

        # Technical = short-term, Fundamental = long-term
        if not fundamental or not technical:
            return conflicts

        fund_valuation = fundamental.get("valuation", {})
        tech_signals = technical.get("signals", technical.get("analysis", {}))

        # Check for valuation vs momentum conflict
        fair_value = fund_valuation.get("fair_value", 0)
        current_price = fund_valuation.get("current_price", 0)
        tech_trend = tech_signals.get("trend", "neutral")

        if current_price > 0 and fair_value > 0:
            upside = (fair_value - current_price) / current_price

            # Fundamentally undervalued but technically bearish
            if upside > 0.15 and "bearish" in str(tech_trend).lower():
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.TIME_HORIZON,
                        severity=ConflictSeverity.MEDIUM,
                        sources=["fundamental", "technical"],
                        description=(
                            f"Long-term valuation suggests {upside:.0%} upside, "
                            f"but short-term technicals show bearish trend"
                        ),
                        data={
                            "upside": upside,
                            "tech_trend": tech_trend,
                            "fair_value": fair_value,
                            "current_price": current_price,
                        },
                    )
                )

            # Fundamentally overvalued but technically bullish
            elif upside < -0.15 and "bullish" in str(tech_trend).lower():
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.TIME_HORIZON,
                        severity=ConflictSeverity.MEDIUM,
                        sources=["fundamental", "technical"],
                        description=(
                            f"Long-term valuation suggests {abs(upside):.0%} downside, "
                            f"but short-term technicals show bullish trend"
                        ),
                        data={
                            "upside": upside,
                            "tech_trend": tech_trend,
                            "fair_value": fair_value,
                            "current_price": current_price,
                        },
                    )
                )

        return conflicts


class DataQualityConflictDetector:
    """Detects conflicts related to data quality differences."""

    def detect(
        self,
        fundamental: Optional[Dict[str, Any]],
        technical: Optional[Dict[str, Any]],
        sec: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
    ) -> List[Conflict]:
        """Detect data quality conflicts."""
        conflicts = []

        # Check for significant data quality differences
        quality_scores = {}

        if fundamental:
            dq = fundamental.get("data_quality", {})
            quality_scores["fundamental"] = dq.get("data_quality_score", 100)

        if sec:
            dq = sec.get("data_quality", {})
            quality_scores["sec"] = dq.get("data_quality_score", 100)

        if technical:
            # Technical data quality based on data completeness
            quality_scores["technical"] = 80  # Default, usually reliable

        if len(quality_scores) >= 2:
            scores = list(quality_scores.values())
            max_score = max(scores)
            min_score = min(scores)

            if max_score - min_score > 30:
                low_quality = [k for k, v in quality_scores.items() if v == min_score]
                high_quality = [k for k, v in quality_scores.items() if v == max_score]

                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.DATA_QUALITY,
                        severity=ConflictSeverity.MEDIUM,
                        sources=low_quality + high_quality,
                        description=(
                            f"Significant data quality gap: "
                            f"{high_quality[0]} ({max_score:.0f}/100) vs "
                            f"{low_quality[0]} ({min_score:.0f}/100)"
                        ),
                        data={"quality_scores": quality_scores, "gap": max_score - min_score},
                    )
                )

        return conflicts


# ============================================================================
# Conflict Resolvers
# ============================================================================


class RecommendationConflictResolver:
    """Resolves recommendation conflicts between analyses."""

    # Default weight adjustments based on conflict direction
    WEIGHT_ADJUSTMENTS = {
        "fundamental_bullish_technical_bearish": {
            "short_term": {"fundamental": 0.4, "technical": 0.6},
            "long_term": {"fundamental": 0.7, "technical": 0.3},
        },
        "fundamental_bearish_technical_bullish": {
            "short_term": {"fundamental": 0.3, "technical": 0.7},
            "long_term": {"fundamental": 0.75, "technical": 0.25},
        },
    }

    def __init__(self, time_horizon: str = "long_term"):
        self.time_horizon = time_horizon

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.RECOMMENDATION

    def resolve(self, conflict: Conflict) -> ConflictResolution:
        direction = conflict.data.get("direction", "fundamental_bullish_technical_bearish")
        weights = self.WEIGHT_ADJUSTMENTS.get(direction, {}).get(
            self.time_horizon, {"fundamental": 0.5, "technical": 0.5}
        )

        if direction == "fundamental_bullish_technical_bearish":
            explanation = (
                "Fundamental analysis indicates value opportunity while "
                "technical analysis shows near-term weakness. This is common "
                "when a stock is temporarily out of favor but fundamentally sound."
            )
            prioritization = (
                f"Prioritize {'fundamentals' if self.time_horizon == 'long_term' else 'technicals'} "
                f"for {self.time_horizon.replace('_', ' ')} investors."
            )
            rationale = (
                "Technical weakness may present buying opportunities for "
                "long-term investors, while short-term traders should wait "
                "for technical confirmation before entry."
            )
        else:
            explanation = (
                "Fundamental analysis suggests caution while technical momentum "
                "is positive. This often occurs in late-stage rallies or "
                "during speculative market phases."
            )
            prioritization = (
                f"Prioritize {'fundamentals' if self.time_horizon == 'long_term' else 'technicals'} "
                f"for {self.time_horizon.replace('_', ' ')} investors."
            )
            rationale = (
                "Strong technicals with weak fundamentals warrant caution. "
                "Momentum may continue short-term, but fundamental concerns "
                "typically prevail over longer periods."
            )

        return ConflictResolution(
            conflict_type=conflict.conflict_type.value,
            explanation=explanation,
            prioritization=prioritization,
            weight_adjustments=weights,
            rationale=rationale,
        )


class TimeHorizonConflictResolver:
    """Resolves time horizon conflicts between analyses."""

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.TIME_HORIZON

    def resolve(self, conflict: Conflict) -> ConflictResolution:
        upside = conflict.data.get("upside", 0)

        if upside > 0:
            # Fundamentally undervalued but technically weak
            explanation = (
                "The stock appears undervalued on a fundamental basis "
                "but is experiencing short-term technical weakness. "
                "This divergence often resolves in favor of fundamentals "
                "over 6-12 month periods."
            )
            prioritization = (
                "Long-term investors may view technical weakness as an "
                "entry opportunity; short-term traders should wait for "
                "technical confirmation."
            )
            rationale = (
                "Mean reversion tends to favor fundamentally undervalued "
                "stocks over time, though timing can be challenging. "
                "Dollar-cost averaging may be appropriate."
            )
            weights = {"fundamental": 0.65, "technical": 0.35}
        else:
            # Fundamentally overvalued but technically strong
            explanation = (
                "The stock appears overvalued on a fundamental basis "
                "but is benefiting from positive momentum. "
                "Such divergences can persist but typically resolve "
                "toward fundamental value."
            )
            prioritization = (
                "Short-term traders may ride momentum with tight stops; "
                "long-term investors should be cautious about new positions."
            )
            rationale = (
                "Momentum can extend overvaluation, but fundamental gravity "
                "eventually prevails. Risk management is critical."
            )
            weights = {"fundamental": 0.55, "technical": 0.45}

        return ConflictResolution(
            conflict_type=conflict.conflict_type.value,
            explanation=explanation,
            prioritization=prioritization,
            weight_adjustments=weights,
            rationale=rationale,
        )


class DataQualityConflictResolver:
    """Resolves data quality conflicts between analyses."""

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.DATA_QUALITY

    def resolve(self, conflict: Conflict) -> ConflictResolution:
        quality_scores = conflict.data.get("quality_scores", {})
        gap = conflict.data.get("gap", 0)

        # Weight adjustments proportional to quality
        total_quality = sum(quality_scores.values())
        if total_quality > 0:
            weights = {k: v / total_quality for k, v in quality_scores.items()}
        else:
            weights = {k: 1 / len(quality_scores) for k in quality_scores}

        explanation = (
            f"Data quality varies significantly across analyses, "
            f"with a {gap:.0f}-point gap. This affects the reliability "
            f"of conclusions and warrants increased margin of safety."
        )

        high_quality_sources = [k for k, v in quality_scores.items() if v >= 70]
        prioritization = (
            f"Prioritize insights from {', '.join(high_quality_sources) or 'sources with better data'} "
            f"given data quality differences."
        )

        rationale = (
            "Analyses based on incomplete or inconsistent data carry "
            "higher uncertainty. Weight adjustments reflect data reliability."
        )

        return ConflictResolution(
            conflict_type=conflict.conflict_type.value,
            explanation=explanation,
            prioritization=prioritization,
            weight_adjustments=weights,
            rationale=rationale,
        )


# ============================================================================
# Main Orchestrator
# ============================================================================


class DeterministicConflictResolver:
    """
    Main orchestrator for deterministic conflict detection and resolution.

    Replaces LLM-based `_reconcile_conflicts()` with rule-based resolution.

    Usage:
        resolver = DeterministicConflictResolver()
        result = resolver.reconcile(
            conflicts=[...],
            fundamental=...,
            technical=...,
            sec=...,
            market_context=...
        )
    """

    def __init__(
        self,
        time_horizon: str = "long_term",
        detectors: Optional[List[ConflictDetector]] = None,
        resolvers: Optional[List[ConflictResolver]] = None,
    ):
        self.time_horizon = time_horizon

        # Default detectors
        self.detectors = detectors or [
            RecommendationConflictDetector(),
            TimeHorizonConflictDetector(),
            DataQualityConflictDetector(),
        ]

        # Default resolvers
        self.resolvers = resolvers or [
            RecommendationConflictResolver(time_horizon=time_horizon),
            TimeHorizonConflictResolver(),
            DataQualityConflictResolver(),
        ]

    def detect_conflicts(
        self,
        fundamental: Optional[Dict[str, Any]] = None,
        technical: Optional[Dict[str, Any]] = None,
        sec: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> List[Conflict]:
        """
        Detect all conflicts between analyses.

        Args:
            fundamental: Fundamental analysis results
            technical: Technical analysis results
            sec: SEC analysis results
            market_context: Market context data

        Returns:
            List of detected conflicts
        """
        all_conflicts = []

        for detector in self.detectors:
            try:
                conflicts = detector.detect(fundamental, technical, sec, market_context)
                all_conflicts.extend(conflicts)
            except Exception as e:
                logger.warning(f"Conflict detector failed: {e}")

        return all_conflicts

    def reconcile(
        self,
        conflicts: Optional[List[Dict[str, Any]]] = None,
        fundamental: Optional[Dict[str, Any]] = None,
        technical: Optional[Dict[str, Any]] = None,
        sec: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> ReconciliationResult:
        """
        Detect and resolve all conflicts.

        Args:
            conflicts: Pre-detected conflicts (optional)
            fundamental: Fundamental analysis results
            technical: Technical analysis results
            sec: SEC analysis results
            market_context: Market context data

        Returns:
            ReconciliationResult with all resolutions
        """
        # Detect conflicts if not provided
        if conflicts is None:
            detected = self.detect_conflicts(fundamental, technical, sec, market_context)
        else:
            # Convert dict conflicts to Conflict objects
            detected = []
            for c in conflicts:
                try:
                    detected.append(
                        Conflict(
                            conflict_type=ConflictType(c.get("conflict_type", "recommendation_conflict")),
                            severity=ConflictSeverity(c.get("severity", "medium")),
                            sources=c.get("sources", []),
                            description=c.get("description", ""),
                            data=c.get("data", {}),
                        )
                    )
                except (ValueError, KeyError):
                    continue

        # Resolve each conflict
        resolutions = []
        for conflict in detected:
            for resolver in self.resolvers:
                if resolver.can_resolve(conflict):
                    try:
                        resolution = resolver.resolve(conflict)
                        resolutions.append(resolution)
                    except Exception as e:
                        logger.warning(f"Conflict resolution failed: {e}")
                    break

        # Calculate overall coherence and confidence impact
        if not resolutions:
            overall_coherence = "High - No significant conflicts detected"
            confidence_impact = 0.0
            reconciled_recommendation = self._determine_recommendation(fundamental, technical, sec)
        else:
            # Coherence based on severity
            high_severity = sum(1 for c in detected if c.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL])
            if high_severity >= 2:
                overall_coherence = "Low - Multiple significant conflicts require resolution"
                confidence_impact = -0.15
            elif high_severity == 1:
                overall_coherence = "Moderate - One significant conflict identified"
                confidence_impact = -0.08
            else:
                overall_coherence = "Good - Minor conflicts manageable"
                confidence_impact = -0.03

            # Blend weights from resolutions
            blended_weights = self._blend_resolution_weights(resolutions)
            reconciled_recommendation = self._weighted_recommendation(fundamental, technical, sec, blended_weights)

        return ReconciliationResult(
            overall_coherence=overall_coherence,
            reconciled_recommendation=reconciled_recommendation,
            confidence_impact=confidence_impact,
            resolutions=resolutions,
        )

    def _blend_resolution_weights(self, resolutions: List[ConflictResolution]) -> Dict[str, float]:
        """Blend weight adjustments from multiple resolutions."""
        if not resolutions:
            return {"fundamental": 0.5, "technical": 0.3, "sec": 0.2}

        # Average weights across resolutions
        weight_sums = {}
        weight_counts = {}

        for resolution in resolutions:
            for source, weight in resolution.weight_adjustments.items():
                weight_sums[source] = weight_sums.get(source, 0) + weight
                weight_counts[source] = weight_counts.get(source, 0) + 1

        blended = {}
        for source in weight_sums:
            blended[source] = weight_sums[source] / weight_counts[source]

        # Normalize to sum to 1
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def _determine_recommendation(
        self, fundamental: Optional[Dict[str, Any]], technical: Optional[Dict[str, Any]], sec: Optional[Dict[str, Any]]
    ) -> str:
        """Determine recommendation when no conflicts exist."""
        # Extract recommendations
        fund_rec = None
        if fundamental:
            fund_rec = (
                fundamental.get("recommendation")
                or fundamental.get("analysis", {}).get("recommendation")
                or fundamental.get("valuation", {}).get("recommendation")
            )

        # Default to fundamental if available
        if fund_rec:
            return str(fund_rec)

        return "Hold - Insufficient data for strong recommendation"

    def _weighted_recommendation(
        self,
        fundamental: Optional[Dict[str, Any]],
        technical: Optional[Dict[str, Any]],
        sec: Optional[Dict[str, Any]],
        weights: Dict[str, float],
    ) -> str:
        """Generate weighted recommendation based on conflict resolution."""
        # Score each source (-1 bearish, 0 neutral, +1 bullish)
        scores = {}

        if fundamental:
            rec = str(fundamental.get("recommendation", "")).lower()
            if any(s in rec for s in ["buy", "bullish", "undervalued"]):
                scores["fundamental"] = 1
            elif any(s in rec for s in ["sell", "bearish", "overvalued"]):
                scores["fundamental"] = -1
            else:
                scores["fundamental"] = 0

        if technical:
            signal = str(technical.get("signal", technical.get("overall_signal", ""))).lower()
            if any(s in signal for s in ["buy", "bullish"]):
                scores["technical"] = 1
            elif any(s in signal for s in ["sell", "bearish"]):
                scores["technical"] = -1
            else:
                scores["technical"] = 0

        if sec:
            rating = sec.get("analysis", {}).get("overall_rating", 5)
            if rating >= 7:
                scores["sec"] = 1
            elif rating <= 4:
                scores["sec"] = -1
            else:
                scores["sec"] = 0

        # Calculate weighted score
        weighted_score = 0
        for source, score in scores.items():
            weight = weights.get(source, 0.33)
            weighted_score += score * weight

        # Map to recommendation
        if weighted_score >= 0.5:
            return "Buy - Weighted analysis favors bullish outlook"
        elif weighted_score >= 0.2:
            return "Lean Buy - Moderately positive weighted signal"
        elif weighted_score <= -0.5:
            return "Sell - Weighted analysis favors bearish outlook"
        elif weighted_score <= -0.2:
            return "Lean Sell - Moderately negative weighted signal"
        else:
            return "Hold - Mixed signals across weighted analyses"


# ============================================================================
# Convenience function for drop-in replacement
# ============================================================================


def reconcile_conflicts(
    conflicts: Optional[List[Dict[str, Any]]] = None,
    fundamental: Optional[Dict[str, Any]] = None,
    technical: Optional[Dict[str, Any]] = None,
    sec: Optional[Dict[str, Any]] = None,
    market_context: Optional[Dict[str, Any]] = None,
    time_horizon: str = "long_term",
) -> Dict[str, Any]:
    """
    Drop-in replacement for LLM-based conflict reconciliation.

    Returns dict with same structure as LLM response for API compatibility.

    Example:
        # Before (LLM):
        response = await self.ollama.generate(model=..., prompt=...)

        # After (deterministic):
        response = reconcile_conflicts(
            conflicts=detected_conflicts,
            fundamental=synthesis_input.fundamental_analysis,
            technical=synthesis_input.technical_analysis
        )
    """
    resolver = DeterministicConflictResolver(time_horizon=time_horizon)
    result = resolver.reconcile(
        conflicts=conflicts, fundamental=fundamental, technical=technical, sec=sec, market_context=market_context
    )
    return result.to_dict()


__all__ = [
    "DeterministicConflictResolver",
    "ReconciliationResult",
    "ConflictResolution",
    "Conflict",
    "ConflictType",
    "ConflictSeverity",
    "RecommendationConflictDetector",
    "TimeHorizonConflictDetector",
    "DataQualityConflictDetector",
    "RecommendationConflictResolver",
    "TimeHorizonConflictResolver",
    "DataQualityConflictResolver",
    "reconcile_conflicts",
]
