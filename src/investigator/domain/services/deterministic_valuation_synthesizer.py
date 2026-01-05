"""
Deterministic Valuation Synthesizer - Replaces LLM-based valuation synthesis.

This module provides deterministic, rule-based valuation synthesis that replaces
the LLM call in `_perform_valuation_synthesis()`. It uses mathematical formulas
and template-based text generation instead of LLM inference.

Benefits:
- Zero token cost
- Instant response (no network latency)
- Fully reproducible results
- Easier to test and debug

Design Principles (SOLID):
- Single Responsibility: Each component handles one aspect of synthesis
- Open/Closed: New stance rules/templates can be added without modifying core
- Liskov Substitution: All synthesizers implement common protocol
- Interface Segregation: Focused interfaces for each concern
- Dependency Inversion: Depends on abstractions (protocols), not concretions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ValuationStance(Enum):
    """Investment stance based on valuation analysis."""

    SIGNIFICANTLY_UNDERVALUED = "Significantly Undervalued"
    UNDERVALUED = "Undervalued"
    SLIGHTLY_UNDERVALUED = "Slightly Undervalued"
    FAIRLY_VALUED = "Fairly Valued"
    SLIGHTLY_OVERVALUED = "Slightly Overvalued"
    OVERVALUED = "Overvalued"
    SIGNIFICANTLY_OVERVALUED = "Significantly Overvalued"


@dataclass
class ModelContribution:
    """Represents a valuation model's contribution to the blended value."""

    model_name: str
    fair_value: Optional[float]
    weight: float
    is_applicable: bool
    reason: Optional[str] = None
    assumptions: Optional[Dict[str, Any]] = None
    weighted_contribution: float = 0.0

    def __post_init__(self):
        if self.is_applicable and self.fair_value and self.weight > 0:
            self.weighted_contribution = self.fair_value * self.weight


@dataclass
class SynthesisContext:
    """Context for valuation synthesis."""

    symbol: str
    current_price: float
    blended_fair_value: float
    overall_confidence: float
    model_agreement_score: float
    divergence_flag: bool
    data_quality_score: float
    quality_grade: str
    sector: str
    industry: Optional[str]
    model_contributions: List[ModelContribution]
    notes: List[str] = field(default_factory=list)
    archetypes: List[str] = field(default_factory=list)


@dataclass
class ValuationSynthesisResult:
    """Result of deterministic valuation synthesis."""

    fair_value_estimate: float
    implied_upside_downside: float
    model_influence_explanation: str
    confidence_and_caution: str
    key_drivers_and_assumptions: str
    valuation_risks: str
    valuation_stance: str
    margin_of_safety_target: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "fair_value_estimate": self.fair_value_estimate,
            "implied_upside_downside": self.implied_upside_downside,
            "model_influence_explanation": self.model_influence_explanation,
            "confidence_and_caution": self.confidence_and_caution,
            "key_drivers_and_assumptions": self.key_drivers_and_assumptions,
            "valuation_risks": self.valuation_risks,
            "valuation_stance": self.valuation_stance,
            "margin_of_safety_target": self.margin_of_safety_target,
        }


class StanceDeterminer(Protocol):
    """Protocol for determining valuation stance."""

    def determine_stance(self, upside_pct: float, confidence: float) -> ValuationStance:
        """Determine valuation stance based on upside and confidence."""
        ...


class MarginOfSafetyCalculator(Protocol):
    """Protocol for calculating margin of safety target."""

    def calculate(
        self, model_agreement: float, data_quality: float, confidence: float, volatility: Optional[float] = None
    ) -> float:
        """Calculate recommended margin of safety target."""
        ...


class ExplanationGenerator(Protocol):
    """Protocol for generating text explanations."""

    def generate_model_influence(self, contributions: List[ModelContribution]) -> str: ...

    def generate_confidence_caution(
        self, confidence: float, data_quality: float, divergence_flag: bool, quality_grade: str
    ) -> str: ...

    def generate_key_drivers(self, contributions: List[ModelContribution]) -> str: ...

    def generate_valuation_risks(self, model_agreement: float, data_quality: float, notes: List[str]) -> str: ...


# ============================================================================
# Implementations
# ============================================================================


class ThresholdBasedStanceDeterminer:
    """
    Determines valuation stance using configurable thresholds.

    The stance is determined by:
    1. Upside/downside percentage relative to current price
    2. Confidence level (adjusts thresholds for edge cases)
    """

    DEFAULT_THRESHOLDS = {
        "significantly_undervalued": 0.30,  # >30% upside
        "undervalued": 0.15,  # >15% upside
        "slightly_undervalued": 0.05,  # >5% upside
        "fairly_valued": (-0.05, 0.05),  # -5% to +5%
        "slightly_overvalued": -0.05,  # <-5% (downside)
        "overvalued": -0.15,  # <-15%
        "significantly_overvalued": -0.30,  # <-30%
    }

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def determine_stance(self, upside_pct: float, confidence: float) -> ValuationStance:
        """
        Determine stance based on upside percentage.

        Args:
            upside_pct: Expected upside as decimal (0.15 = 15%)
            confidence: Confidence level 0-1

        Returns:
            ValuationStance enum value
        """
        # Adjust thresholds for low confidence (widen "fairly valued" band)
        confidence_factor = max(0.5, confidence)  # Don't over-adjust

        sig_under = self.thresholds["significantly_undervalued"] * confidence_factor
        under = self.thresholds["undervalued"] * confidence_factor
        slight_under = self.thresholds["slightly_undervalued"] * confidence_factor
        slight_over = self.thresholds["slightly_overvalued"] * confidence_factor
        over = self.thresholds["overvalued"] * confidence_factor
        sig_over = self.thresholds["significantly_overvalued"] * confidence_factor

        if upside_pct >= sig_under:
            return ValuationStance.SIGNIFICANTLY_UNDERVALUED
        elif upside_pct >= under:
            return ValuationStance.UNDERVALUED
        elif upside_pct >= slight_under:
            return ValuationStance.SLIGHTLY_UNDERVALUED
        elif upside_pct >= slight_over:
            return ValuationStance.FAIRLY_VALUED
        elif upside_pct >= over:
            return ValuationStance.SLIGHTLY_OVERVALUED
        elif upside_pct >= sig_over:
            return ValuationStance.OVERVALUED
        else:
            return ValuationStance.SIGNIFICANTLY_OVERVALUED


class RiskBasedMarginOfSafetyCalculator:
    """
    Calculates margin of safety based on model agreement and data quality.

    Higher margin of safety when:
    - Model agreement is low (divergence)
    - Data quality is poor
    - Confidence is low
    """

    # Base margin of safety targets by risk level
    BASE_MARGINS = {
        "low_risk": 0.10,  # 10% for high-quality, high-agreement
        "medium_risk": 0.20,  # 20% default
        "high_risk": 0.30,  # 30% for divergence or quality issues
        "very_high_risk": 0.40,  # 40% for significant concerns
    }

    def calculate(
        self, model_agreement: float, data_quality: float, confidence: float, volatility: Optional[float] = None
    ) -> float:
        """
        Calculate recommended margin of safety.

        Args:
            model_agreement: Score 0-1 (1 = perfect agreement)
            data_quality: Score 0-100
            confidence: Overall confidence 0-1
            volatility: Optional volatility factor

        Returns:
            Margin of safety as decimal (0.20 = 20%)
        """
        # Normalize data quality to 0-1
        dq_normalized = data_quality / 100.0 if data_quality > 1 else data_quality

        # Calculate risk score (0 = low risk, 1 = high risk)
        risk_score = 0.0

        # Model agreement contribution (40% weight)
        if model_agreement < 0.3:
            risk_score += 0.4  # High divergence = high risk
        elif model_agreement < 0.6:
            risk_score += 0.2

        # Data quality contribution (30% weight)
        if dq_normalized < 0.5:
            risk_score += 0.3  # Poor quality = more margin needed
        elif dq_normalized < 0.75:
            risk_score += 0.15

        # Confidence contribution (30% weight)
        if confidence < 0.5:
            risk_score += 0.3
        elif confidence < 0.75:
            risk_score += 0.15

        # Map risk score to margin of safety
        if risk_score >= 0.7:
            base_margin = self.BASE_MARGINS["very_high_risk"]
        elif risk_score >= 0.5:
            base_margin = self.BASE_MARGINS["high_risk"]
        elif risk_score >= 0.25:
            base_margin = self.BASE_MARGINS["medium_risk"]
        else:
            base_margin = self.BASE_MARGINS["low_risk"]

        # Apply volatility adjustment if available
        if volatility and volatility > 0.4:  # High volatility
            base_margin = min(0.50, base_margin * 1.25)

        return round(base_margin, 2)


class TemplateBasedExplanationGenerator:
    """
    Generates text explanations using templates with variable substitution.

    Templates are designed to be professional, clear, and actionable.
    """

    def generate_model_influence(self, contributions: List[ModelContribution]) -> str:
        """Generate explanation of how each model influences the final value."""
        applicable = [c for c in contributions if c.is_applicable and c.fair_value]

        if not applicable:
            return "No valuation models produced applicable outputs due to data limitations."

        # Sort by weight (highest first)
        sorted_contrib = sorted(applicable, key=lambda x: x.weight, reverse=True)

        # Build explanation
        parts = []
        primary = sorted_contrib[0]

        parts.append(
            f"The {primary.model_name.upper()} model has the highest weight "
            f"({primary.weight:.0%}) and is the primary driver of the fair value estimate "
            f"at ${primary.fair_value:,.2f}."
        )

        if len(sorted_contrib) > 1:
            secondary_names = [c.model_name.upper() for c in sorted_contrib[1:3]]
            secondary_weights = [f"{c.weight:.0%}" for c in sorted_contrib[1:3]]

            if len(secondary_names) == 1:
                parts.append(
                    f"The {secondary_names[0]} model ({secondary_weights[0]}) " f"provides secondary validation."
                )
            else:
                parts.append(
                    f"The {' and '.join(secondary_names)} models "
                    f"({', '.join(secondary_weights)}) provide additional perspectives."
                )

        # Note any significant divergence
        if len(applicable) >= 2:
            values = [c.fair_value for c in applicable]
            max_val, min_val = max(values), min(values)
            if min_val > 0:
                spread = (max_val - min_val) / min_val
                if spread > 0.3:
                    parts.append(f"Note: Model outputs show {spread:.0%} spread, " f"suggesting valuation uncertainty.")

        return " ".join(parts)

    def generate_confidence_caution(
        self, confidence: float, data_quality: float, divergence_flag: bool, quality_grade: str
    ) -> str:
        """Generate confidence assessment and cautionary notes."""
        parts = []

        # Confidence statement
        if confidence >= 0.8:
            conf_desc = "high"
        elif confidence >= 0.6:
            conf_desc = "moderate"
        elif confidence >= 0.4:
            conf_desc = "limited"
        else:
            conf_desc = "low"

        parts.append(f"Confidence is {conf_desc} ({confidence:.0%}).")

        # Data quality impact
        if data_quality >= 80:
            parts.append(f"Data quality is {quality_grade} ({data_quality:.0f}/100), " f"supporting reliable analysis.")
        elif data_quality >= 60:
            parts.append(
                f"Data quality is {quality_grade} ({data_quality:.0f}/100). " f"Some metrics may require verification."
            )
        else:
            parts.append(
                f"Data quality is {quality_grade} ({data_quality:.0f}/100). "
                f"Results should be interpreted with caution due to data gaps."
            )

        # Divergence warning
        if divergence_flag:
            parts.append(
                "Model divergence flag raised - multiple valuation approaches "
                "yield materially different results, warranting additional scrutiny."
            )

        return " ".join(parts)

    def generate_key_drivers(self, contributions: List[ModelContribution]) -> str:
        """Generate summary of key drivers and assumptions."""
        drivers = []

        for c in contributions:
            if not c.is_applicable or not c.assumptions:
                continue

            model_drivers = []
            assumptions = c.assumptions

            # DCF/GGM drivers
            if "growth_rate" in assumptions:
                model_drivers.append(f"growth rate {assumptions['growth_rate']:.1%}")
            if "discount_rate" in assumptions:
                model_drivers.append(f"discount rate {assumptions['discount_rate']:.1%}")
            if "terminal_growth" in assumptions:
                model_drivers.append(f"terminal growth {assumptions['terminal_growth']:.1%}")

            # Multiple-based drivers
            if "target_pe" in assumptions:
                model_drivers.append(f"target P/E {assumptions['target_pe']:.1f}x")
            if "target_ps" in assumptions:
                model_drivers.append(f"target P/S {assumptions['target_ps']:.1f}x")
            if "target_ev_ebitda" in assumptions:
                model_drivers.append(f"target EV/EBITDA {assumptions['target_ev_ebitda']:.1f}x")

            if model_drivers:
                drivers.append(f"{c.model_name.upper()}: {', '.join(model_drivers)}")

        if not drivers:
            return "Key assumptions vary by model; see individual model details for specifics."

        return "Key drivers: " + "; ".join(drivers) + "."

    def generate_valuation_risks(self, model_agreement: float, data_quality: float, notes: List[str]) -> str:
        """Generate valuation risks and scenario considerations."""
        risks = []

        # Model agreement risks
        if model_agreement < 0.4:
            risks.append(
                "High model divergence suggests significant uncertainty - "
                "actual fair value could deviate substantially from estimate"
            )
        elif model_agreement < 0.7:
            risks.append("Moderate model divergence indicates some valuation uncertainty")

        # Data quality risks
        if data_quality < 60:
            risks.append("Limited data quality may affect accuracy of key inputs")

        # Include relevant notes
        for note in notes[:3]:  # Limit to top 3 notes
            if note and len(note) > 10:
                risks.append(note)

        # Add standard risk considerations if list is short
        if len(risks) < 2:
            risks.append(
                "Changes in interest rates, sector multiples, or "
                "company fundamentals could shift valuation materially"
            )

        return " ".join(risks) if risks else "Standard valuation risks apply."


class DeterministicValuationSynthesizer:
    """
    Main orchestrator for deterministic valuation synthesis.

    Replaces LLM-based `_perform_valuation_synthesis()` with deterministic
    computation and template-based text generation.

    Usage:
        synthesizer = DeterministicValuationSynthesizer()
        result = synthesizer.synthesize(context)
        output_dict = result.to_dict()  # Compatible with existing API
    """

    def __init__(
        self,
        stance_determiner: Optional[StanceDeterminer] = None,
        margin_calculator: Optional[MarginOfSafetyCalculator] = None,
        explanation_generator: Optional[ExplanationGenerator] = None,
    ):
        self.stance_determiner = stance_determiner or ThresholdBasedStanceDeterminer()
        self.margin_calculator = margin_calculator or RiskBasedMarginOfSafetyCalculator()
        self.explanation_generator = explanation_generator or TemplateBasedExplanationGenerator()

    def synthesize(self, context: SynthesisContext) -> ValuationSynthesisResult:
        """
        Perform deterministic valuation synthesis.

        Args:
            context: SynthesisContext with all required inputs

        Returns:
            ValuationSynthesisResult with complete synthesis output
        """
        # Calculate upside/downside
        if context.current_price > 0:
            upside = (context.blended_fair_value - context.current_price) / context.current_price
        else:
            upside = 0.0

        # Determine stance
        stance = self.stance_determiner.determine_stance(upside, context.overall_confidence)

        # Calculate margin of safety target
        margin = self.margin_calculator.calculate(
            context.model_agreement_score, context.data_quality_score, context.overall_confidence
        )

        # Generate explanations
        model_influence = self.explanation_generator.generate_model_influence(context.model_contributions)

        confidence_caution = self.explanation_generator.generate_confidence_caution(
            context.overall_confidence, context.data_quality_score, context.divergence_flag, context.quality_grade
        )

        key_drivers = self.explanation_generator.generate_key_drivers(context.model_contributions)

        valuation_risks = self.explanation_generator.generate_valuation_risks(
            context.model_agreement_score, context.data_quality_score, context.notes
        )

        return ValuationSynthesisResult(
            fair_value_estimate=round(context.blended_fair_value, 2),
            implied_upside_downside=round(upside, 4),
            model_influence_explanation=model_influence,
            confidence_and_caution=confidence_caution,
            key_drivers_and_assumptions=key_drivers,
            valuation_risks=valuation_risks,
            valuation_stance=stance.value,
            margin_of_safety_target=margin,
        )

    @classmethod
    def from_valuation_results(
        cls,
        symbol: str,
        current_price: float,
        valuation_results: Dict[str, Any],
        multi_model_summary: Dict[str, Any],
        data_quality: Dict[str, Any],
        company_profile: Dict[str, Any],
        notes: Optional[List[str]] = None,
    ) -> "ValuationSynthesisResult":
        """
        Factory method to create synthesis from existing valuation outputs.

        This method bridges the existing data structures to SynthesisContext.

        Args:
            symbol: Stock ticker
            current_price: Current market price
            valuation_results: Dict with model outputs (dcf, pe, ps, etc.)
            multi_model_summary: Summary from dynamic model weighting
            data_quality: Data quality assessment dict
            company_profile: Company profile dict
            notes: Optional list of notes/warnings

        Returns:
            ValuationSynthesisResult ready for use
        """
        # Build model contributions from valuation results
        contributions = []

        for model_name, model_data in valuation_results.items():
            if not isinstance(model_data, dict):
                continue

            contribution = ModelContribution(
                model_name=model_name,
                fair_value=model_data.get("fair_value"),
                weight=model_data.get("weight", 0),
                is_applicable=model_data.get("applicable", False),
                reason=model_data.get("reason"),
                assumptions=model_data.get("assumptions", {}),
            )
            contributions.append(contribution)

        # Build context
        context = SynthesisContext(
            symbol=symbol,
            current_price=current_price,
            blended_fair_value=multi_model_summary.get("blended_fair_value", 0),
            overall_confidence=multi_model_summary.get("overall_confidence", 0.5),
            model_agreement_score=multi_model_summary.get("model_agreement_score", 0.5),
            divergence_flag=multi_model_summary.get("divergence_flag", False),
            data_quality_score=data_quality.get("data_quality_score", 50),
            quality_grade=data_quality.get("quality_grade", "Unknown"),
            sector=company_profile.get("sector", "Unknown"),
            industry=company_profile.get("industry"),
            model_contributions=contributions,
            notes=notes or [],
            archetypes=company_profile.get("archetypes", []),
        )

        # Create synthesizer and run
        synthesizer = cls()
        return synthesizer.synthesize(context)


# ============================================================================
# Convenience function for drop-in replacement
# ============================================================================


def synthesize_valuation(
    symbol: str,
    current_price: float,
    valuation_results: Dict[str, Any],
    multi_model_summary: Dict[str, Any],
    data_quality: Dict[str, Any],
    company_profile: Dict[str, Any],
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for LLM-based valuation synthesis.

    Returns dict with same structure as LLM response for API compatibility.

    Example:
        # Before (LLM):
        response = await self.ollama.generate(model=..., prompt=...)

        # After (deterministic):
        response = synthesize_valuation(
            symbol=symbol,
            current_price=market_data.get('price', 0),
            valuation_results=valuation_results,
            multi_model_summary=multi_model_summary,
            data_quality=company_data.get('data_quality', {}),
            company_profile=company_profile_payload
        )
    """
    result = DeterministicValuationSynthesizer.from_valuation_results(
        symbol=symbol,
        current_price=current_price,
        valuation_results=valuation_results,
        multi_model_summary=multi_model_summary,
        data_quality=data_quality,
        company_profile=company_profile,
        notes=notes,
    )
    return result.to_dict()


__all__ = [
    "DeterministicValuationSynthesizer",
    "ValuationSynthesisResult",
    "SynthesisContext",
    "ModelContribution",
    "ValuationStance",
    "ThresholdBasedStanceDeterminer",
    "RiskBasedMarginOfSafetyCalculator",
    "TemplateBasedExplanationGenerator",
    "synthesize_valuation",
]
