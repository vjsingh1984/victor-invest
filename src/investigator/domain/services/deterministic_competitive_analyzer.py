"""
Deterministic Competitive Analyzer - Replaces LLM-based competitive position analysis.

This module provides rule-based competitive position analysis using Porter's
Five Forces framework, replacing the LLM call in `_analyze_competitive_position()`.
It uses sector/industry heuristics and financial metrics to assess competitive dynamics.

Benefits:
- Zero token cost
- Instant response
- Consistent, sector-appropriate analysis
- Objective scoring based on metrics

Design Principles (SOLID):
- Single Responsibility: Each analyzer handles one competitive force
- Open/Closed: New sector profiles can be added via registry
- Liskov Substitution: All analyzers implement common protocol
- Interface Segregation: Focused interfaces for each force analysis
- Dependency Inversion: Depends on abstractions, not concretions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class AssessmentLevel(Enum):
    """Assessment levels for competitive factors."""

    VERY_HIGH = "Very High"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


class MoatWidth(Enum):
    """Economic moat classification."""

    WIDE = "Wide"
    NARROW = "Narrow"
    NONE = "None"


class MarketPosition(Enum):
    """Market position classification."""

    LEADER = "Leader"
    CHALLENGER = "Challenger"
    FOLLOWER = "Follower"
    NICHE = "Niche Player"


@dataclass
class ForceAssessment:
    """Assessment of a single Porter's Force."""

    assessment: str
    commentary: str
    score: int  # 0-100 (higher = more favorable for company)


@dataclass
class CompetitiveAnalysis:
    """Complete competitive position analysis."""

    market_position_and_share: ForceAssessment
    competitive_advantages_moat: ForceAssessment
    industry_dynamics_and_trends: ForceAssessment
    barriers_to_entry: ForceAssessment
    supplier_and_customer_power: ForceAssessment
    threat_of_substitutes: ForceAssessment
    competitive_risks: List[str]
    strategic_positioning_score: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "market_position_and_share": {
                "assessment": self.market_position_and_share.assessment,
                "commentary": self.market_position_and_share.commentary,
            },
            "competitive_advantages_moat": {
                "assessment": self.competitive_advantages_moat.assessment,
                "commentary": self.competitive_advantages_moat.commentary,
            },
            "industry_dynamics_and_trends": {
                "assessment": self.industry_dynamics_and_trends.assessment,
                "commentary": self.industry_dynamics_and_trends.commentary,
            },
            "barriers_to_entry": {
                "assessment": self.barriers_to_entry.assessment,
                "commentary": self.barriers_to_entry.commentary,
            },
            "supplier_and_customer_power": {
                "assessment": self.supplier_and_customer_power.assessment,
                "commentary": self.supplier_and_customer_power.commentary,
            },
            "threat_of_substitutes": {
                "assessment": self.threat_of_substitutes.assessment,
                "commentary": self.threat_of_substitutes.commentary,
            },
            "competitive_risks": self.competitive_risks,
            "strategic_positioning_score": self.strategic_positioning_score,
        }


@dataclass
class CompetitiveContext:
    """Context for competitive analysis."""

    symbol: str
    sector: str
    industry: Optional[str]
    market_cap: Optional[float]
    revenue: Optional[float]
    profit_margin: Optional[float]
    revenue_growth: Optional[float]
    roe: Optional[float]
    debt_to_equity: Optional[float]
    data_quality_score: float = 50.0


class ForceAnalyzer(Protocol):
    """Protocol for analyzing a Porter's Force."""

    def analyze(self, context: CompetitiveContext, sector_profile: "SectorProfile") -> ForceAssessment: ...


# ============================================================================
# Sector Profiles
# ============================================================================


@dataclass
class SectorProfile:
    """Competitive dynamics profile for a sector."""

    sector: str
    typical_barriers: str
    typical_moat_sources: List[str]
    supplier_power: str
    customer_power: str
    substitute_threat: str
    competitive_intensity: str
    key_success_factors: List[str]
    common_risks: List[str]
    industry_growth: str
    consolidation_trend: str


# Sector profile registry
SECTOR_PROFILES: Dict[str, SectorProfile] = {
    "Technology": SectorProfile(
        sector="Technology",
        typical_barriers="High",
        typical_moat_sources=["Network effects", "Switching costs", "Intellectual property", "Platform economics"],
        supplier_power="Low",
        customer_power="Moderate",
        substitute_threat="Moderate to High",
        competitive_intensity="High",
        key_success_factors=[
            "Innovation velocity",
            "Developer ecosystem",
            "Customer retention",
            "Cloud/AI capabilities",
        ],
        common_risks=[
            "Rapid technology obsolescence",
            "Intense competition from well-funded rivals",
            "Regulatory/antitrust scrutiny",
            "Cybersecurity threats",
        ],
        industry_growth="Favorable",
        consolidation_trend="Active M&A",
    ),
    "Healthcare": SectorProfile(
        sector="Healthcare",
        typical_barriers="Very High",
        typical_moat_sources=["Patents and IP", "FDA regulatory approvals", "Clinical expertise", "Scale in R&D"],
        supplier_power="Moderate",
        customer_power="High (payors)",
        substitute_threat="Low to Moderate",
        competitive_intensity="Moderate",
        key_success_factors=["Pipeline depth", "Regulatory expertise", "Commercial execution", "Pricing power"],
        common_risks=[
            "Clinical trial failures",
            "Patent cliff exposure",
            "Drug pricing pressure",
            "Regulatory changes",
        ],
        industry_growth="Favorable (aging demographics)",
        consolidation_trend="Selective M&A",
    ),
    "Financials": SectorProfile(
        sector="Financials",
        typical_barriers="High",
        typical_moat_sources=["Switching costs", "Regulatory licenses", "Scale economies", "Distribution networks"],
        supplier_power="Low",
        customer_power="Moderate",
        substitute_threat="Moderate (fintech)",
        competitive_intensity="Moderate to High",
        key_success_factors=["Credit discipline", "Digital capabilities", "Cost efficiency", "Capital management"],
        common_risks=[
            "Credit cycle deterioration",
            "Interest rate volatility",
            "Fintech disruption",
            "Regulatory compliance costs",
        ],
        industry_growth="Moderate",
        consolidation_trend="Continued consolidation",
    ),
    "Consumer": SectorProfile(
        sector="Consumer",
        typical_barriers="Moderate",
        typical_moat_sources=["Brand strength", "Distribution networks", "Scale advantages", "Customer loyalty"],
        supplier_power="Moderate",
        customer_power="High",
        substitute_threat="High",
        competitive_intensity="High",
        key_success_factors=[
            "Brand equity",
            "E-commerce capabilities",
            "Supply chain efficiency",
            "Product innovation",
        ],
        common_risks=[
            "Consumer spending weakness",
            "Private label competition",
            "Changing consumer preferences",
            "Input cost inflation",
        ],
        industry_growth="Moderate",
        consolidation_trend="Active M&A",
    ),
    "Industrials": SectorProfile(
        sector="Industrials",
        typical_barriers="Moderate to High",
        typical_moat_sources=["Scale economics", "Technical expertise", "Customer relationships", "Installed base"],
        supplier_power="Moderate",
        customer_power="Moderate to High",
        substitute_threat="Low to Moderate",
        competitive_intensity="Moderate",
        key_success_factors=[
            "Operational efficiency",
            "Aftermarket services",
            "Global footprint",
            "Engineering capabilities",
        ],
        common_risks=[
            "Economic cycle sensitivity",
            "Raw material cost inflation",
            "Project execution risk",
            "Labor availability",
        ],
        industry_growth="Cyclical",
        consolidation_trend="Steady M&A",
    ),
    "Energy": SectorProfile(
        sector="Energy",
        typical_barriers="Very High",
        typical_moat_sources=["Asset quality", "Scale economics", "Resource access", "Infrastructure"],
        supplier_power="Low to Moderate",
        customer_power="Low",
        substitute_threat="Moderate (renewables)",
        competitive_intensity="Moderate",
        key_success_factors=["Cost structure", "Reserve replacement", "Capital discipline", "ESG transition"],
        common_risks=[
            "Commodity price volatility",
            "Energy transition risk",
            "Regulatory changes",
            "Geopolitical factors",
        ],
        industry_growth="Transitioning",
        consolidation_trend="Selective M&A",
    ),
}

DEFAULT_PROFILE = SectorProfile(
    sector="Default",
    typical_barriers="Moderate",
    typical_moat_sources=["Scale", "Brand", "Cost advantages"],
    supplier_power="Moderate",
    customer_power="Moderate",
    substitute_threat="Moderate",
    competitive_intensity="Moderate",
    key_success_factors=["Market share", "Profitability", "Innovation"],
    common_risks=["Competition", "Economic sensitivity", "Execution risk"],
    industry_growth="Moderate",
    consolidation_trend="Varies",
)


# ============================================================================
# Force Analyzers
# ============================================================================


class MarketPositionAnalyzer:
    """Analyzes market position and share."""

    def analyze(self, context: CompetitiveContext, sector_profile: SectorProfile) -> ForceAssessment:
        """Assess market position based on financial metrics."""
        # Estimate position from market cap (simplified heuristic)
        market_cap = context.market_cap or 0

        if market_cap > 100_000_000_000:  # >$100B
            position = MarketPosition.LEADER
            assessment = "Leader"
            base_score = 85
        elif market_cap > 20_000_000_000:  # >$20B
            position = MarketPosition.CHALLENGER
            assessment = "Challenger"
            base_score = 70
        elif market_cap > 5_000_000_000:  # >$5B
            position = MarketPosition.FOLLOWER
            assessment = "Follower"
            base_score = 55
        else:
            position = MarketPosition.NICHE
            assessment = "Niche Player"
            base_score = 45

        # Adjust for revenue growth (strong growth = gaining share)
        if context.revenue_growth and context.revenue_growth > 0.15:
            base_score = min(95, base_score + 10)
            growth_comment = "with market share gains evident from strong revenue growth"
        elif context.revenue_growth and context.revenue_growth < 0:
            base_score = max(20, base_score - 15)
            growth_comment = "though market share may be under pressure given revenue decline"
        else:
            growth_comment = "with stable market positioning"

        # Format market cap safely
        if context.market_cap and context.market_cap > 0:
            market_cap_str = f"${context.market_cap / 1e9:.1f}B"
        else:
            market_cap_str = "undisclosed"

        commentary = (
            f"The company operates as a {assessment.lower()} in the {context.sector} sector "
            f"(market cap: {market_cap_str}), {growth_comment}."
        )

        return ForceAssessment(assessment=assessment, commentary=commentary, score=base_score)


class MoatAnalyzer:
    """Analyzes competitive advantages and economic moat."""

    def analyze(self, context: CompetitiveContext, sector_profile: SectorProfile) -> ForceAssessment:
        """Assess economic moat from financial metrics."""
        moat_indicators = 0
        moat_sources = []

        # High ROE suggests moat (returns above cost of capital)
        if context.roe and context.roe > 0.20:
            moat_indicators += 2
            moat_sources.append("high return on equity")
        elif context.roe and context.roe > 0.15:
            moat_indicators += 1

        # High margins suggest pricing power
        if context.profit_margin and context.profit_margin > 0.20:
            moat_indicators += 2
            moat_sources.append("strong pricing power (high margins)")
        elif context.profit_margin and context.profit_margin > 0.12:
            moat_indicators += 1

        # Consistent revenue growth suggests competitive advantage
        if context.revenue_growth and context.revenue_growth > 0.15:
            moat_indicators += 1
            moat_sources.append("consistent revenue growth")

        # Large market cap in sector suggests scale advantages
        if context.market_cap and context.market_cap > 50_000_000_000:
            moat_indicators += 1
            moat_sources.append("scale advantages")

        # Determine moat width
        if moat_indicators >= 4:
            moat = MoatWidth.WIDE
            assessment = "Wide"
            base_score = 85
        elif moat_indicators >= 2:
            moat = MoatWidth.NARROW
            assessment = "Narrow"
            base_score = 65
        else:
            moat = MoatWidth.NONE
            assessment = "Limited"
            base_score = 40

        # Build commentary
        if moat_sources:
            sources_text = ", ".join(moat_sources)
            commentary = (
                f"The company appears to have a {assessment.lower()} economic moat, "
                f"evidenced by {sources_text}. "
                f"Typical moat sources in {context.sector} include "
                f"{', '.join(sector_profile.typical_moat_sources[:2])}."
            )
        else:
            commentary = (
                f"Limited evidence of durable competitive advantage based on current metrics. "
                f"In {context.sector}, sustainable moats typically come from "
                f"{', '.join(sector_profile.typical_moat_sources[:2])}."
            )

        return ForceAssessment(assessment=assessment, commentary=commentary, score=base_score)


class IndustryDynamicsAnalyzer:
    """Analyzes industry dynamics and trends."""

    def analyze(self, context: CompetitiveContext, sector_profile: SectorProfile) -> ForceAssessment:
        """Assess industry dynamics from sector profile."""
        growth = sector_profile.industry_growth
        consolidation = sector_profile.consolidation_trend

        # Map growth to assessment
        if "favorable" in growth.lower():
            assessment = "Favorable"
            base_score = 75
        elif "transition" in growth.lower() or "cyclical" in growth.lower():
            assessment = "Mixed"
            base_score = 55
        else:
            assessment = "Moderate"
            base_score = 60

        # Adjust for company-specific growth
        if context.revenue_growth:
            if context.revenue_growth > 0.20:
                base_score = min(90, base_score + 10)
            elif context.revenue_growth < 0:
                base_score = max(30, base_score - 15)

        commentary = (
            f"The {context.sector} sector shows {growth.lower()} growth dynamics, "
            f"with {consolidation.lower()} creating both opportunities and challenges. "
            f"Key success factors include {', '.join(sector_profile.key_success_factors[:2])}."
        )

        return ForceAssessment(assessment=assessment, commentary=commentary, score=base_score)


class BarriersToEntryAnalyzer:
    """Analyzes barriers to entry."""

    def analyze(self, context: CompetitiveContext, sector_profile: SectorProfile) -> ForceAssessment:
        """Assess barriers to entry from sector profile."""
        barriers = sector_profile.typical_barriers

        if "very high" in barriers.lower():
            assessment = "Very High"
            base_score = 85
        elif "high" in barriers.lower():
            assessment = "High"
            base_score = 75
        elif "low" in barriers.lower():
            assessment = "Low"
            base_score = 40
        else:
            assessment = "Moderate"
            base_score = 60

        # Large companies benefit more from barriers
        if context.market_cap and context.market_cap > 20_000_000_000:
            base_score = min(95, base_score + 5)

        commentary = (
            f"The {context.sector} sector has {barriers.lower()} barriers to entry, "
            f"providing {'strong' if base_score > 70 else 'moderate'} protection "
            f"against new competition. Key barriers include "
            f"{', '.join(sector_profile.typical_moat_sources[:2])}."
        )

        return ForceAssessment(assessment=assessment, commentary=commentary, score=base_score)


class BuyerSupplierPowerAnalyzer:
    """Analyzes supplier and customer power."""

    def analyze(self, context: CompetitiveContext, sector_profile: SectorProfile) -> ForceAssessment:
        """Assess supplier and customer bargaining power."""
        supplier_power = sector_profile.supplier_power.lower()
        customer_power = sector_profile.customer_power.lower()

        # Score based on combined power (lower is better for company)
        score = 60

        if "low" in supplier_power:
            score += 10
        elif "high" in supplier_power:
            score -= 10

        if "low" in customer_power:
            score += 10
        elif "high" in customer_power:
            score -= 10

        # High margins suggest low bargaining power of others
        if context.profit_margin and context.profit_margin > 0.20:
            score = min(85, score + 10)
        elif context.profit_margin and context.profit_margin < 0.08:
            score = max(30, score - 10)

        # Determine assessment
        if score >= 70:
            assessment = "Low"
        elif score >= 50:
            assessment = "Moderate"
        else:
            assessment = "High"

        commentary = (
            f"Supplier power is {supplier_power} and customer power is {customer_power} "
            f"in the {context.sector} sector. "
            f"{'Strong profit margins suggest effective bargaining position.' if context.profit_margin and context.profit_margin > 0.15 else 'Margin profile suggests typical sector dynamics.'}"
        )

        return ForceAssessment(assessment=assessment, commentary=commentary, score=score)


class SubstituteThreatAnalyzer:
    """Analyzes threat of substitutes."""

    def analyze(self, context: CompetitiveContext, sector_profile: SectorProfile) -> ForceAssessment:
        """Assess threat of substitutes from sector profile."""
        threat = sector_profile.substitute_threat.lower()

        if "very" in threat and "high" in threat:
            assessment = "Very High"
            base_score = 30
        elif "high" in threat:
            assessment = "High"
            base_score = 40
        elif "low" in threat:
            assessment = "Low"
            base_score = 75
        else:
            assessment = "Moderate"
            base_score = 55

        # Strong revenue growth suggests limited substitution
        if context.revenue_growth and context.revenue_growth > 0.15:
            base_score = min(85, base_score + 10)

        commentary = (
            f"The threat of substitutes in {context.sector} is {threat}. "
            f"{'Strong revenue growth suggests limited near-term substitution risk.' if base_score > 60 else 'Companies must continuously innovate to maintain relevance.'}"
        )

        return ForceAssessment(assessment=assessment, commentary=commentary, score=base_score)


# ============================================================================
# Risk Generator
# ============================================================================


class CompetitiveRiskGenerator:
    """Generates competitive risks based on context and sector."""

    def generate(self, context: CompetitiveContext, sector_profile: SectorProfile) -> List[str]:
        """Generate relevant competitive risks."""
        risks = []

        # Add sector-specific risks
        for risk in sector_profile.common_risks[:2]:
            risks.append(risk)

        # Add context-specific risks
        if context.profit_margin and context.profit_margin < 0.10:
            risks.append("Margin compression from competitive pressure")

        if context.revenue_growth and context.revenue_growth < 0.05:
            risks.append("Slowing growth may indicate market share losses")

        if context.debt_to_equity and context.debt_to_equity > 1.5:
            risks.append("High leverage limits competitive flexibility")

        # Add general competitive risks
        if len(risks) < 3:
            risks.append("Potential intensification of competition from well-resourced rivals")

        if len(risks) < 4:
            risks.append("Technology or business model disruption risk")

        return risks[:4]


# ============================================================================
# Main Analyzer
# ============================================================================


class DeterministicCompetitiveAnalyzer:
    """
    Main orchestrator for deterministic competitive position analysis.

    Replaces LLM-based `_analyze_competitive_position()` with rule-based
    Porter's Five Forces analysis.

    Usage:
        analyzer = DeterministicCompetitiveAnalyzer()
        analysis = analyzer.analyze(context)
        output_dict = analysis.to_dict()
    """

    def __init__(
        self,
        market_position_analyzer: Optional[ForceAnalyzer] = None,
        moat_analyzer: Optional[ForceAnalyzer] = None,
        industry_dynamics_analyzer: Optional[ForceAnalyzer] = None,
        barriers_analyzer: Optional[ForceAnalyzer] = None,
        power_analyzer: Optional[ForceAnalyzer] = None,
        substitute_analyzer: Optional[ForceAnalyzer] = None,
        risk_generator: Optional[CompetitiveRiskGenerator] = None,
    ):
        self.market_position_analyzer = market_position_analyzer or MarketPositionAnalyzer()
        self.moat_analyzer = moat_analyzer or MoatAnalyzer()
        self.industry_dynamics_analyzer = industry_dynamics_analyzer or IndustryDynamicsAnalyzer()
        self.barriers_analyzer = barriers_analyzer or BarriersToEntryAnalyzer()
        self.power_analyzer = power_analyzer or BuyerSupplierPowerAnalyzer()
        self.substitute_analyzer = substitute_analyzer or SubstituteThreatAnalyzer()
        self.risk_generator = risk_generator or CompetitiveRiskGenerator()

    def analyze(self, context: CompetitiveContext) -> CompetitiveAnalysis:
        """
        Perform complete competitive analysis.

        Args:
            context: CompetitiveContext with company and sector data

        Returns:
            CompetitiveAnalysis with all force assessments
        """
        # Get sector profile
        sector_profile = SECTOR_PROFILES.get(context.sector, DEFAULT_PROFILE)

        # Analyze each force
        market_position = self.market_position_analyzer.analyze(context, sector_profile)
        moat = self.moat_analyzer.analyze(context, sector_profile)
        industry_dynamics = self.industry_dynamics_analyzer.analyze(context, sector_profile)
        barriers = self.barriers_analyzer.analyze(context, sector_profile)
        power = self.power_analyzer.analyze(context, sector_profile)
        substitutes = self.substitute_analyzer.analyze(context, sector_profile)

        # Generate risks
        risks = self.risk_generator.generate(context, sector_profile)

        # Calculate strategic positioning score (weighted average of forces)
        scores = [
            market_position.score * 0.20,
            moat.score * 0.25,
            industry_dynamics.score * 0.15,
            barriers.score * 0.15,
            power.score * 0.15,
            substitutes.score * 0.10,
        ]
        strategic_score = int(sum(scores))

        # Adjust for data quality
        if context.data_quality_score < 50:
            strategic_score = int(strategic_score * 0.85)

        return CompetitiveAnalysis(
            market_position_and_share=market_position,
            competitive_advantages_moat=moat,
            industry_dynamics_and_trends=industry_dynamics,
            barriers_to_entry=barriers,
            supplier_and_customer_power=power,
            threat_of_substitutes=substitutes,
            competitive_risks=risks,
            strategic_positioning_score=max(0, min(100, strategic_score)),
        )

    @classmethod
    def from_company_data(cls, symbol: str, company_data: Dict[str, Any]) -> CompetitiveAnalysis:
        """
        Factory method to create analysis from company_data dict.

        Args:
            symbol: Stock ticker
            company_data: Company data dict with financials and market data

        Returns:
            CompetitiveAnalysis ready for use
        """
        # Extract market data
        market_data = company_data.get("market_data", {})

        # Extract financials
        financials = company_data.get("financials", company_data.get("data", {}))
        ratios = company_data.get("ratios", financials.get("ratios", {}))

        # Extract profile
        profile = company_data.get("profile", company_data.get("company_profile", {}))

        # Build context
        context = CompetitiveContext(
            symbol=symbol,
            sector=profile.get("sector", company_data.get("sector", "Unknown")),
            industry=profile.get("industry", company_data.get("industry")),
            market_cap=market_data.get("market_cap"),
            revenue=financials.get("revenues", financials.get("revenue")),
            profit_margin=ratios.get("profit_margin") or ratios.get("operating_margin"),
            revenue_growth=ratios.get("revenue_growth"),
            roe=ratios.get("roe") or ratios.get("return_on_equity"),
            debt_to_equity=ratios.get("debt_to_equity"),
            data_quality_score=company_data.get("data_quality", {}).get("data_quality_score", 50),
        )

        # Create analyzer and run
        analyzer = cls()
        return analyzer.analyze(context)


# ============================================================================
# Convenience function for drop-in replacement
# ============================================================================


def analyze_competitive_position(symbol: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop-in replacement for LLM-based competitive position analysis.

    Returns dict with same structure as LLM response for API compatibility.

    Example:
        # Before (LLM):
        response = await self.ollama.generate(model=..., prompt=...)

        # After (deterministic):
        response = analyze_competitive_position(
            symbol=company_data['symbol'],
            company_data=company_data
        )
    """
    analysis = DeterministicCompetitiveAnalyzer.from_company_data(symbol=symbol, company_data=company_data)
    return analysis.to_dict()


__all__ = [
    "DeterministicCompetitiveAnalyzer",
    "CompetitiveAnalysis",
    "CompetitiveContext",
    "ForceAssessment",
    "AssessmentLevel",
    "MoatWidth",
    "MarketPosition",
    "SectorProfile",
    "SECTOR_PROFILES",
    "MarketPositionAnalyzer",
    "MoatAnalyzer",
    "IndustryDynamicsAnalyzer",
    "BarriersToEntryAnalyzer",
    "BuyerSupplierPowerAnalyzer",
    "SubstituteThreatAnalyzer",
    "CompetitiveRiskGenerator",
    "analyze_competitive_position",
]
