"""
Template-Based Thesis Generator - Replaces LLM-based investment thesis generation.

This module provides template-based investment thesis generation that replaces
the LLM call in `_generate_investment_thesis()`. It uses structured templates
with variable substitution to create professional investment narratives.

Benefits:
- Zero token cost
- Instant response
- Consistent formatting
- Reproducible outputs
- Easier to customize per sector/style

Design Principles (SOLID):
- Single Responsibility: Each component handles one thesis element
- Open/Closed: New templates can be added without modifying core
- Liskov Substitution: All generators implement common protocol
- Interface Segregation: Focused interfaces for each thesis section
- Dependency Inversion: Depends on abstractions, not concretions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class InvestmentStance(Enum):
    """Investment stance derived from analysis."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TimeHorizon(Enum):
    """Investment time horizon."""

    SHORT_TERM = "6-12 months"
    MEDIUM_TERM = "1-3 years"
    LONG_TERM = "3-5 years"


@dataclass
class ThesisContext:
    """Context for thesis generation."""

    symbol: str
    company_name: str
    sector: str
    industry: Optional[str]
    overall_score: float  # 0-100
    confidence: float  # 0-100
    upside: float  # Decimal (0.15 = 15%)
    current_price: float
    fair_value: float
    # From key insights
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)
    critical_metrics: Dict[str, Any] = field(default_factory=dict)
    # From analysis
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    # Quality and risk
    data_quality_score: float = 50.0
    model_agreement: float = 0.5
    risk_level: str = "medium"


@dataclass
class InvestmentThesis:
    """Complete investment thesis output."""

    core_investment_narrative: str
    key_value_drivers: List[str]
    competitive_advantages: List[str]
    growth_catalysts: List[str]
    bear_case_considerations: List[str]
    time_horizon: str
    key_metrics_to_monitor: List[str]
    thesis_invalidation_triggers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "core_investment_narrative": self.core_investment_narrative,
            "key_value_drivers": self.key_value_drivers,
            "competitive_advantages": self.competitive_advantages,
            "growth_catalysts": self.growth_catalysts,
            "bear_case_considerations": self.bear_case_considerations,
            "time_horizon": self.time_horizon,
            "key_metrics_to_monitor": self.key_metrics_to_monitor,
            "thesis_invalidation_triggers": self.thesis_invalidation_triggers,
        }


class NarrativeGenerator(Protocol):
    """Protocol for generating narrative sections."""

    def generate(self, context: ThesisContext) -> str: ...


class ListGenerator(Protocol):
    """Protocol for generating list sections."""

    def generate(self, context: ThesisContext) -> List[str]: ...


# ============================================================================
# Narrative Generators
# ============================================================================


class CoreNarrativeGenerator:
    """Generates the core investment narrative based on analysis context."""

    # Templates by stance
    TEMPLATES = {
        InvestmentStance.STRONG_BUY: (
            "{symbol} presents a compelling investment opportunity in the {sector} sector. "
            "Trading at ${current_price:.2f}, the stock offers {upside_pct:.0%} upside to our "
            "fair value estimate of ${fair_value:.2f}. {positive_summary} "
            "With a quality score of {score:.0f}/100 and {confidence:.0f}% confidence in our "
            "analysis, we see strong risk-adjusted return potential."
        ),
        InvestmentStance.BUY: (
            "{symbol} offers attractive value in the {sector} sector at current levels. "
            "At ${current_price:.2f}, the stock trades at a discount to our "
            "${fair_value:.2f} fair value estimate, implying {upside_pct:.0%} upside. "
            "{positive_summary} "
            "Our analysis confidence of {confidence:.0f}% supports a constructive outlook."
        ),
        InvestmentStance.HOLD: (
            "{symbol} appears fairly valued at ${current_price:.2f} in the {sector} sector. "
            "Our fair value estimate of ${fair_value:.2f} suggests limited near-term "
            "upside of {upside_pct:.0%}. {balanced_summary} "
            "Current holders may maintain positions; new investors should await "
            "better entry points."
        ),
        InvestmentStance.SELL: (
            "{symbol} appears overvalued at ${current_price:.2f} in the {sector} sector. "
            "Our analysis indicates {downside_pct:.0%} downside risk to fair value of "
            "${fair_value:.2f}. {negative_summary} "
            "We recommend reducing exposure or avoiding new positions."
        ),
        InvestmentStance.STRONG_SELL: (
            "{symbol} presents significant downside risk at ${current_price:.2f}. "
            "Our fair value estimate of ${fair_value:.2f} implies {downside_pct:.0%} "
            "potential loss. {negative_summary} "
            "We strongly recommend avoiding or exiting positions."
        ),
    }

    def generate(self, context: ThesisContext) -> str:
        """Generate the core investment narrative."""
        stance = self._determine_stance(context)
        template = self.TEMPLATES[stance]

        # Build summaries
        if context.positive_factors:
            positive_summary = f"Key strengths include {self._format_list(context.positive_factors[:2])}."
        else:
            positive_summary = "The company demonstrates solid fundamentals."

        if context.negative_factors:
            negative_summary = f"Concerns include {self._format_list(context.negative_factors[:2])}."
        else:
            negative_summary = "Risk factors warrant monitoring."

        balanced_summary = (
            f"While {context.positive_factors[0].lower() if context.positive_factors else 'fundamentals are stable'}, "
            f"{context.negative_factors[0].lower() if context.negative_factors else 'some caution is warranted'}."
        )

        return template.format(
            symbol=context.symbol,
            sector=context.sector,
            current_price=context.current_price,
            fair_value=context.fair_value,
            upside_pct=abs(context.upside),
            downside_pct=abs(context.upside),
            score=context.overall_score,
            confidence=context.confidence,
            positive_summary=positive_summary,
            negative_summary=negative_summary,
            balanced_summary=balanced_summary,
        )

    def _determine_stance(self, context: ThesisContext) -> InvestmentStance:
        """Determine investment stance from context."""
        upside = context.upside
        confidence = context.confidence / 100.0

        # Adjust thresholds by confidence
        strong_buy_threshold = 0.25 * confidence
        buy_threshold = 0.10 * confidence
        sell_threshold = -0.10 * confidence
        strong_sell_threshold = -0.25 * confidence

        if upside >= strong_buy_threshold:
            return InvestmentStance.STRONG_BUY
        elif upside >= buy_threshold:
            return InvestmentStance.BUY
        elif upside <= strong_sell_threshold:
            return InvestmentStance.STRONG_SELL
        elif upside <= sell_threshold:
            return InvestmentStance.SELL
        else:
            return InvestmentStance.HOLD

    def _format_list(self, items: List[str]) -> str:
        """Format list items as natural language."""
        if not items:
            return ""
        elif len(items) == 1:
            return items[0].lower()
        else:
            return f"{items[0].lower()} and {items[1].lower()}"


# ============================================================================
# List Generators
# ============================================================================


class ValueDriversGenerator:
    """Generates key value drivers based on financial metrics."""

    # Sector-specific value drivers
    SECTOR_DRIVERS = {
        "Technology": [
            "Innovation pipeline and R&D productivity",
            "Recurring revenue and subscription growth",
            "Market share expansion in core segments",
            "Cloud/digital transformation tailwinds",
        ],
        "Healthcare": [
            "Drug pipeline and FDA approval potential",
            "Demographic tailwinds from aging population",
            "Healthcare spending growth",
            "Pricing power in specialty therapeutics",
        ],
        "Financials": [
            "Net interest margin expansion",
            "Credit quality and loan growth",
            "Fee income diversification",
            "Capital return programs",
        ],
        "Consumer": [
            "Brand strength and pricing power",
            "Market share gains",
            "Geographic expansion",
            "E-commerce growth",
        ],
        "Industrials": [
            "Order backlog and book-to-bill ratio",
            "Margin expansion through operational efficiency",
            "Infrastructure spending tailwinds",
            "Aftermarket services growth",
        ],
    }

    DEFAULT_DRIVERS = [
        "Revenue growth sustainability",
        "Margin expansion opportunity",
        "Competitive positioning",
        "Capital allocation efficiency",
    ]

    def generate(self, context: ThesisContext) -> List[str]:
        """Generate key value drivers."""
        drivers = []

        # Add metrics-based drivers
        if context.revenue_growth and context.revenue_growth > 0.10:
            drivers.append(f"Strong revenue growth of {context.revenue_growth:.0%} demonstrates market share gains")
        if context.profit_margin and context.profit_margin > 0.15:
            drivers.append(f"Above-average profit margins of {context.profit_margin:.0%} support valuation premium")
        if context.dividend_yield and context.dividend_yield > 0.02:
            drivers.append(f"Dividend yield of {context.dividend_yield:.1%} provides income component")

        # Add sector-specific drivers
        sector_specific = self.SECTOR_DRIVERS.get(context.sector, self.DEFAULT_DRIVERS)
        for driver in sector_specific[:2]:
            if driver not in drivers:
                drivers.append(driver)

        # Add from positive factors if available
        for factor in context.positive_factors[:2]:
            if len(drivers) < 4 and factor not in drivers:
                drivers.append(factor)

        # Ensure minimum drivers
        while len(drivers) < 3:
            for driver in self.DEFAULT_DRIVERS:
                if driver not in drivers:
                    drivers.append(driver)
                    break

        return drivers[:4]


class CompetitiveAdvantagesGenerator:
    """Generates competitive advantages based on moat analysis."""

    # Common competitive advantages by type
    MOAT_TEMPLATES = {
        "brand": "Strong brand recognition and customer loyalty",
        "network_effects": "Network effects create switching costs and barriers to entry",
        "cost_advantage": "Cost leadership through scale and operational efficiency",
        "intangibles": "Proprietary technology and intellectual property",
        "switching_costs": "High customer switching costs support retention",
        "regulatory": "Regulatory barriers limit competitive threats",
    }

    SECTOR_MOATS = {
        "Technology": ["intangibles", "network_effects", "switching_costs"],
        "Healthcare": ["intangibles", "regulatory", "brand"],
        "Financials": ["switching_costs", "regulatory", "brand"],
        "Consumer": ["brand", "cost_advantage", "network_effects"],
        "Industrials": ["cost_advantage", "switching_costs", "intangibles"],
    }

    def generate(self, context: ThesisContext) -> List[str]:
        """Generate competitive advantages."""
        advantages = []

        # Get sector-appropriate moats
        sector_moats = self.SECTOR_MOATS.get(context.sector, ["brand", "cost_advantage"])

        for moat_type in sector_moats:
            if moat_type in self.MOAT_TEMPLATES:
                advantages.append(self.MOAT_TEMPLATES[moat_type])

        # Add from positive factors
        for factor in context.positive_factors:
            if "moat" in factor.lower() or "advantage" in factor.lower():
                if factor not in advantages:
                    advantages.append(factor)
                    break

        return advantages[:3]


class GrowthCatalystsGenerator:
    """Generates growth catalysts based on sector and metrics."""

    SECTOR_CATALYSTS = {
        "Technology": [
            "AI and machine learning integration driving new revenue streams",
            "Cloud adoption acceleration across enterprise customers",
            "Geographic expansion into emerging markets",
            "Strategic acquisitions to expand product portfolio",
        ],
        "Healthcare": [
            "Pipeline developments and regulatory approvals",
            "Medicare/Medicaid policy changes benefiting segment",
            "Biosimilar opportunities in key therapeutic areas",
            "Value-based care transition driving utilization",
        ],
        "Financials": [
            "Interest rate environment supporting net interest income",
            "Credit normalization improving loss provisions",
            "Wealth management AUM growth",
            "Digital banking adoption reducing cost-to-income",
        ],
        "Consumer": [
            "Product innovation driving market share gains",
            "International expansion into underpenetrated markets",
            "E-commerce channel optimization",
            "Premiumization strategy supporting margins",
        ],
        "Industrials": [
            "Infrastructure spending legislation benefits",
            "Reshoring and supply chain localization trends",
            "Electrification and decarbonization investments",
            "Aftermarket services expansion",
        ],
    }

    DEFAULT_CATALYSTS = [
        "Market share expansion opportunities",
        "Operational efficiency improvements",
        "Strategic capital deployment",
        "Industry consolidation potential",
    ]

    def generate(self, context: ThesisContext) -> List[str]:
        """Generate growth catalysts."""
        catalysts = self.SECTOR_CATALYSTS.get(context.sector, self.DEFAULT_CATALYSTS)

        # Add revenue-growth specific catalyst if applicable
        if context.revenue_growth and context.revenue_growth > 0.15:
            catalysts = [f"Momentum in core business with {context.revenue_growth:.0%} revenue growth"] + catalysts

        return catalysts[:4]


class BearCaseGenerator:
    """Generates bear case considerations based on risks."""

    SECTOR_RISKS = {
        "Technology": [
            "Increased competition from well-funded rivals",
            "Technology obsolescence or disruption risk",
            "Customer concentration in key accounts",
            "Regulatory scrutiny and antitrust concerns",
        ],
        "Healthcare": [
            "Drug pricing pressure from policy changes",
            "Clinical trial failures or delays",
            "Patent cliff exposure",
            "Reimbursement rate reductions",
        ],
        "Financials": [
            "Credit cycle deterioration increasing losses",
            "Interest rate volatility compressing margins",
            "Regulatory capital requirements tightening",
            "Fintech disruption in core businesses",
        ],
        "Consumer": [
            "Consumer spending weakness in downturn",
            "Private label and discount competition",
            "Supply chain cost inflation",
            "Changing consumer preferences",
        ],
        "Industrials": [
            "Economic cycle sensitivity",
            "Raw material cost inflation",
            "Labor availability and cost pressures",
            "Project execution and backlog risks",
        ],
    }

    DEFAULT_RISKS = [
        "Macroeconomic uncertainty affecting demand",
        "Competitive pressure on margins",
        "Execution risk in growth initiatives",
        "Valuation premium at risk if growth disappoints",
    ]

    def generate(self, context: ThesisContext) -> List[str]:
        """Generate bear case considerations."""
        risks = []

        # Add from negative factors first
        for factor in context.negative_factors[:2]:
            risks.append(factor)

        # Add sector-specific risks
        sector_risks = self.SECTOR_RISKS.get(context.sector, self.DEFAULT_RISKS)
        for risk in sector_risks:
            if len(risks) >= 4:
                break
            if risk not in risks:
                risks.append(risk)

        # Add valuation risk if overvalued
        if context.upside < 0:
            risks.append(f"Valuation risk with {abs(context.upside):.0%} implied downside to fair value")

        return risks[:4]


class KeyMetricsGenerator:
    """Generates key metrics to monitor based on sector and analysis."""

    SECTOR_METRICS = {
        "Technology": [
            "Revenue growth and ARR/MRR trends",
            "Operating margin and R&D efficiency",
            "Customer acquisition cost and LTV",
            "Net revenue retention rate",
        ],
        "Healthcare": [
            "Revenue growth by therapeutic area",
            "Gross margin and SG&A efficiency",
            "Pipeline milestones and approval dates",
            "Payer mix and reimbursement trends",
        ],
        "Financials": [
            "Net interest margin and spread trends",
            "Credit quality metrics (NPL, NCO, provisions)",
            "Efficiency ratio and operating leverage",
            "Capital ratios and return on equity",
        ],
        "Consumer": [
            "Same-store sales and comparable growth",
            "Gross margin and promotional intensity",
            "Inventory turns and working capital",
            "Customer acquisition and retention",
        ],
        "Industrials": [
            "Order book and book-to-bill ratio",
            "Operating margin and productivity",
            "Free cash flow conversion",
            "Backlog duration and quality",
        ],
    }

    DEFAULT_METRICS = [
        "Revenue growth trajectory",
        "Operating margin trends",
        "Free cash flow generation",
        "Return on invested capital",
    ]

    def generate(self, context: ThesisContext) -> List[str]:
        """Generate key metrics to monitor."""
        metrics = self.SECTOR_METRICS.get(context.sector, self.DEFAULT_METRICS)

        # Add valuation metric
        if context.pe_ratio:
            metrics = metrics + [f"P/E ratio vs. historical average ({context.pe_ratio:.1f}x current)"]

        return metrics[:5]


class InvalidationTriggersGenerator:
    """Generates thesis invalidation triggers."""

    BASE_TRIGGERS = [
        "Revenue growth decelerates below {growth_threshold}%",
        "Operating margin contracts more than {margin_threshold} basis points",
        "Debt-to-equity ratio exceeds {leverage_threshold}x",
        "Key management departures without clear succession",
    ]

    def generate(self, context: ThesisContext) -> List[str]:
        """Generate thesis invalidation triggers."""
        triggers = []

        # Revenue growth trigger
        if context.revenue_growth:
            min_growth = max(0, context.revenue_growth - 0.10)
            triggers.append(f"Revenue growth decelerates below {min_growth:.0%} for two consecutive quarters")
        else:
            triggers.append("Revenue growth turns negative for two consecutive quarters")

        # Margin trigger
        if context.profit_margin:
            margin_floor = context.profit_margin * 0.7
            triggers.append(f"Operating margin contracts below {margin_floor:.0%} (30% compression from current)")
        else:
            triggers.append("Operating margin declines materially from current levels")

        # Leverage trigger
        if context.debt_to_equity:
            leverage_ceiling = context.debt_to_equity * 1.5
            triggers.append(f"Debt-to-equity ratio exceeds {leverage_ceiling:.1f}x (50% increase from current)")
        else:
            triggers.append("Significant deterioration in balance sheet leverage")

        # Fair value trigger
        triggers.append(f"Stock price exceeds fair value estimate of ${context.fair_value:.2f} by more than 20%")

        return triggers[:4]


# ============================================================================
# Time Horizon Determiner
# ============================================================================


class TimeHorizonDeterminer:
    """Determines appropriate investment time horizon."""

    def determine(self, context: ThesisContext) -> str:
        """Determine time horizon based on context."""
        # Factors affecting time horizon
        is_growth = context.revenue_growth and context.revenue_growth > 0.15
        is_value = context.upside > 0.20
        is_high_quality = context.data_quality_score >= 70 and context.model_agreement >= 0.6

        if is_growth:
            # Growth stocks need longer horizon
            return TimeHorizon.LONG_TERM.value
        elif is_value and is_high_quality:
            # Clear value with high confidence = medium term
            return TimeHorizon.MEDIUM_TERM.value
        elif context.upside < 0:
            # Overvalued = shorter horizon for potential exit
            return TimeHorizon.SHORT_TERM.value
        else:
            return TimeHorizon.MEDIUM_TERM.value


# ============================================================================
# Main Generator
# ============================================================================


class TemplateBasedThesisGenerator:
    """
    Main orchestrator for template-based investment thesis generation.

    Replaces LLM-based `_generate_investment_thesis()` with deterministic
    template-based generation.

    Usage:
        generator = TemplateBasedThesisGenerator()
        thesis = generator.generate(context)
        output_dict = thesis.to_dict()  # Compatible with existing API
    """

    def __init__(
        self,
        narrative_generator: Optional[NarrativeGenerator] = None,
        value_drivers_generator: Optional[ListGenerator] = None,
        competitive_advantages_generator: Optional[ListGenerator] = None,
        growth_catalysts_generator: Optional[ListGenerator] = None,
        bear_case_generator: Optional[ListGenerator] = None,
        key_metrics_generator: Optional[ListGenerator] = None,
        invalidation_triggers_generator: Optional[ListGenerator] = None,
        time_horizon_determiner: Optional[TimeHorizonDeterminer] = None,
    ):
        self.narrative_generator = narrative_generator or CoreNarrativeGenerator()
        self.value_drivers_generator = value_drivers_generator or ValueDriversGenerator()
        self.competitive_advantages_generator = competitive_advantages_generator or CompetitiveAdvantagesGenerator()
        self.growth_catalysts_generator = growth_catalysts_generator or GrowthCatalystsGenerator()
        self.bear_case_generator = bear_case_generator or BearCaseGenerator()
        self.key_metrics_generator = key_metrics_generator or KeyMetricsGenerator()
        self.invalidation_triggers_generator = invalidation_triggers_generator or InvalidationTriggersGenerator()
        self.time_horizon_determiner = time_horizon_determiner or TimeHorizonDeterminer()

    def generate(self, context: ThesisContext) -> InvestmentThesis:
        """
        Generate complete investment thesis.

        Args:
            context: ThesisContext with all required inputs

        Returns:
            InvestmentThesis with all sections populated
        """
        return InvestmentThesis(
            core_investment_narrative=self.narrative_generator.generate(context),
            key_value_drivers=self.value_drivers_generator.generate(context),
            competitive_advantages=self.competitive_advantages_generator.generate(context),
            growth_catalysts=self.growth_catalysts_generator.generate(context),
            bear_case_considerations=self.bear_case_generator.generate(context),
            time_horizon=self.time_horizon_determiner.determine(context),
            key_metrics_to_monitor=self.key_metrics_generator.generate(context),
            thesis_invalidation_triggers=self.invalidation_triggers_generator.generate(context),
        )

    @classmethod
    def from_synthesis_data(
        cls,
        symbol: str,
        key_insights: Dict[str, Any],
        composite_scores: Dict[str, Any],
        fundamental_analysis: Optional[Dict[str, Any]] = None,
        company_profile: Optional[Dict[str, Any]] = None,
    ) -> InvestmentThesis:
        """
        Factory method to create thesis from existing synthesis data.

        This method bridges the existing data structures to ThesisContext.

        Args:
            symbol: Stock ticker
            key_insights: Key insights from synthesis
            composite_scores: Composite scores from synthesis
            fundamental_analysis: Fundamental analysis results
            company_profile: Company profile data

        Returns:
            InvestmentThesis ready for use
        """
        # Extract positive and negative factors
        positive_factors = []
        negative_factors = []

        for source in ["fundamental", "technical", "sec"]:
            source_insights = key_insights.get(source, {})
            if isinstance(source_insights, dict):
                positive_factors.extend(source_insights.get("positive_factors", [])[:2])
                negative_factors.extend(source_insights.get("negative_factors", [])[:2])

        # Extract financial metrics from fundamental analysis
        valuation = {}
        revenue_growth = None
        profit_margin = None
        dividend_yield = None
        pe_ratio = None
        debt_to_equity = None

        if fundamental_analysis:
            valuation = fundamental_analysis.get("valuation", {})
            ratios = fundamental_analysis.get("ratios", {})
            revenue_growth = ratios.get("revenue_growth")
            profit_margin = ratios.get("profit_margin") or ratios.get("operating_margin")
            dividend_yield = valuation.get("dividend_yield")
            pe_ratio = valuation.get("pe_ratio")
            debt_to_equity = ratios.get("debt_to_equity")

        # Calculate upside
        current_price = valuation.get("current_price", 0)
        fair_value = valuation.get("fair_value", 0)
        if current_price > 0 and fair_value > 0:
            upside = (fair_value - current_price) / current_price
        else:
            upside = 0

        # Build context
        context = ThesisContext(
            symbol=symbol,
            company_name=company_profile.get("company_name", symbol) if company_profile else symbol,
            sector=company_profile.get("sector", "Unknown") if company_profile else "Unknown",
            industry=company_profile.get("industry") if company_profile else None,
            overall_score=composite_scores.get("overall_score", 50),
            confidence=composite_scores.get("confidence", 50),
            upside=upside,
            current_price=current_price,
            fair_value=fair_value,
            positive_factors=positive_factors[:4],
            negative_factors=negative_factors[:4],
            revenue_growth=revenue_growth,
            profit_margin=profit_margin,
            dividend_yield=dividend_yield,
            pe_ratio=pe_ratio,
            debt_to_equity=debt_to_equity,
            data_quality_score=(
                fundamental_analysis.get("data_quality", {}).get("data_quality_score", 50)
                if fundamental_analysis
                else 50
            ),
            model_agreement=(
                fundamental_analysis.get("multi_model_summary", {}).get("model_agreement_score", 0.5)
                if fundamental_analysis
                else 0.5
            ),
        )

        # Generate thesis
        generator = cls()
        return generator.generate(context)


# ============================================================================
# Convenience function for drop-in replacement
# ============================================================================


def generate_investment_thesis(
    symbol: str,
    key_insights: Dict[str, Any],
    composite_scores: Dict[str, Any],
    fundamental_analysis: Optional[Dict[str, Any]] = None,
    company_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for LLM-based investment thesis generation.

    Returns dict with same structure as LLM response for API compatibility.

    Example:
        # Before (LLM):
        response = await self.ollama.generate(model=..., prompt=...)

        # After (deterministic):
        response = generate_investment_thesis(
            symbol=synthesis_input.symbol,
            key_insights=key_insights,
            composite_scores=composite_scores,
            fundamental_analysis=synthesis_input.fundamental_analysis
        )
    """
    thesis = TemplateBasedThesisGenerator.from_synthesis_data(
        symbol=symbol,
        key_insights=key_insights,
        composite_scores=composite_scores,
        fundamental_analysis=fundamental_analysis,
        company_profile=company_profile,
    )
    return thesis.to_dict()


__all__ = [
    "TemplateBasedThesisGenerator",
    "InvestmentThesis",
    "ThesisContext",
    "InvestmentStance",
    "TimeHorizon",
    "CoreNarrativeGenerator",
    "ValueDriversGenerator",
    "CompetitiveAdvantagesGenerator",
    "GrowthCatalystsGenerator",
    "BearCaseGenerator",
    "KeyMetricsGenerator",
    "InvalidationTriggersGenerator",
    "TimeHorizonDeterminer",
    "generate_investment_thesis",
]
