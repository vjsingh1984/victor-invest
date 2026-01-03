"""
Market context models for dynamic weight adjustments.

This module provides enums and dataclasses for representing market conditions
that affect valuation model weighting. Market context is derived from technical
and market context agent outputs and used to apply dynamic multipliers to
tier-based static weights.

Author: InvestiGator Team
Date: 2025-11-14
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TechnicalTrend(Enum):
    """Technical trend classification based on price action and momentum."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class MarketSentiment(Enum):
    """Market sentiment classification based on VIX, breadth, and qualitative factors."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class RiskLevel(Enum):
    """Risk environment classification based on volatility and uncertainty."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class QualityTier(Enum):
    """Company quality classification (may also come from tier classification)."""

    EXCEPTIONAL = "exceptional"
    HIGH = "high"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"


class CreditCyclePhase(Enum):
    """Credit cycle phase classification based on spreads and economic conditions.

    The credit cycle typically moves through these phases:
    - EARLY_EXPANSION: Post-recession recovery, spreads tightening, growth accelerating
    - MID_CYCLE: Healthy expansion, stable spreads, moderate growth
    - LATE_CYCLE: Slowing growth, spreads starting to widen, caution warranted
    - CREDIT_STRESS: Elevated risk, widening spreads, defensive positioning needed
    - CREDIT_CRISIS: Severe stress, extremely wide spreads, maximum defensive
    """
    EARLY_EXPANSION = "early_expansion"
    MID_CYCLE = "mid_cycle"
    LATE_CYCLE = "late_cycle"
    CREDIT_STRESS = "credit_stress"
    CREDIT_CRISIS = "credit_crisis"
    UNKNOWN = "unknown"


class RecessionProbability(Enum):
    """Recession probability classification based on multiple indicators.

    Uses yield curve, credit spreads, unemployment trends, and other factors
    to estimate recession probability within 12 months.
    """
    VERY_LOW = "very_low"        # <10%
    LOW = "low"                   # 10-25%
    ELEVATED = "elevated"         # 25-50%
    HIGH = "high"                 # 50-75%
    IMMINENT = "imminent"         # >75%
    UNKNOWN = "unknown"


class VolatilityRegime(Enum):
    """Volatility regime classification based on VIX and realized volatility.

    Different volatility regimes require different valuation approaches
    and risk management strategies.
    """
    VERY_LOW = "very_low"         # VIX < 12
    LOW = "low"                    # VIX 12-16
    NORMAL = "normal"              # VIX 16-20
    ELEVATED = "elevated"          # VIX 20-25
    HIGH = "high"                  # VIX 25-35
    EXTREME = "extreme"            # VIX > 35
    UNKNOWN = "unknown"


class FedPolicyStance(Enum):
    """Federal Reserve monetary policy stance.

    Derived from Fed funds rate trajectory and forward guidance.
    """
    VERY_DOVISH = "very_dovish"   # Aggressive easing
    DOVISH = "dovish"              # Easing bias
    NEUTRAL = "neutral"            # Data dependent
    HAWKISH = "hawkish"            # Tightening bias
    VERY_HAWKISH = "very_hawkish"  # Aggressive tightening
    UNKNOWN = "unknown"


@dataclass
class MarketContext:
    """
    Market condition context for dynamic weight adjustments.

    This dataclass encapsulates all market-related factors that influence
    how much weight each valuation model should receive. These factors are
    extracted from technical and market context agent outputs.

    Attributes:
        technical_trend: Price trend classification (strong_uptrend to strong_downtrend)
        market_sentiment: Overall market sentiment (very_bullish to very_bearish)
        risk_level: Risk environment (very_low to very_high)
        quality_tier: Company quality tier (optional, may come from tier classification)
        volatility_regime: Volatility classification (very_low to extreme)
        credit_cycle_phase: Credit cycle phase (early_expansion to credit_crisis)
        recession_probability: Recession probability classification
        fed_policy_stance: Federal Reserve policy stance

    Example:
        >>> context = MarketContext(
        ...     technical_trend=TechnicalTrend.STRONG_DOWNTREND,
        ...     market_sentiment=MarketSentiment.VERY_BEARISH,
        ...     risk_level=RiskLevel.HIGH,
        ...     quality_tier="below_average",
        ...     credit_cycle_phase=CreditCyclePhase.LATE_CYCLE
        ... )
        >>> context.to_dict()
        {
            'technical_trend': 'strong_downtrend',
            'market_sentiment': 'very_bearish',
            'risk_level': 'high',
            'quality_tier': 'below_average',
            ...
        }
    """

    technical_trend: TechnicalTrend = TechnicalTrend.SIDEWAYS
    market_sentiment: MarketSentiment = MarketSentiment.NEUTRAL
    risk_level: RiskLevel = RiskLevel.MEDIUM
    quality_tier: Optional[str] = None
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    credit_cycle_phase: CreditCyclePhase = CreditCyclePhase.MID_CYCLE
    recession_probability: RecessionProbability = RecessionProbability.LOW
    fed_policy_stance: FedPolicyStance = FedPolicyStance.NEUTRAL

    def to_dict(self) -> dict:
        """
        Convert MarketContext to dictionary for logging and serialization.

        Returns:
            dict: Dictionary representation with enum values as strings
        """
        return {
            "technical_trend": self.technical_trend.value,
            "market_sentiment": self.market_sentiment.value,
            "risk_level": self.risk_level.value,
            "quality_tier": self.quality_tier,
            "volatility_regime": self.volatility_regime.value,
            "credit_cycle_phase": self.credit_cycle_phase.value,
            "recession_probability": self.recession_probability.value,
            "fed_policy_stance": self.fed_policy_stance.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarketContext":
        """
        Create MarketContext from dictionary (e.g., from agent outputs).

        Args:
            data: Dictionary with string enum values

        Returns:
            MarketContext: Initialized market context

        Example:
            >>> data = {
            ...     'technical_trend': 'uptrend',
            ...     'market_sentiment': 'bullish',
            ...     'risk_level': 'low'
            ... }
            >>> context = MarketContext.from_dict(data)
        """
        # Helper to safely get enum value
        def get_enum(enum_cls, key, default):
            val = data.get(key)
            if val is None:
                return default
            try:
                return enum_cls(val)
            except ValueError:
                return default

        return cls(
            technical_trend=get_enum(TechnicalTrend, "technical_trend", TechnicalTrend.SIDEWAYS),
            market_sentiment=get_enum(MarketSentiment, "market_sentiment", MarketSentiment.NEUTRAL),
            risk_level=get_enum(RiskLevel, "risk_level", RiskLevel.MEDIUM),
            quality_tier=data.get("quality_tier"),
            volatility_regime=get_enum(VolatilityRegime, "volatility_regime", VolatilityRegime.NORMAL),
            credit_cycle_phase=get_enum(CreditCyclePhase, "credit_cycle_phase", CreditCyclePhase.MID_CYCLE),
            recession_probability=get_enum(RecessionProbability, "recession_probability", RecessionProbability.LOW),
            fed_policy_stance=get_enum(FedPolicyStance, "fed_policy_stance", FedPolicyStance.NEUTRAL),
        )

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"MarketContext(trend={self.technical_trend.value}, "
            f"sentiment={self.market_sentiment.value}, "
            f"risk={self.risk_level.value}, "
            f"credit_cycle={self.credit_cycle_phase.value}, "
            f"recession={self.recession_probability.value})"
        )

    @property
    def is_defensive_regime(self) -> bool:
        """Check if current regime suggests defensive positioning."""
        return (
            self.credit_cycle_phase in (CreditCyclePhase.CREDIT_STRESS, CreditCyclePhase.CREDIT_CRISIS)
            or self.recession_probability in (RecessionProbability.HIGH, RecessionProbability.IMMINENT)
            or self.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)
        )

    @property
    def is_risk_on_regime(self) -> bool:
        """Check if current regime supports risk-on positioning."""
        return (
            self.credit_cycle_phase in (CreditCyclePhase.EARLY_EXPANSION, CreditCyclePhase.MID_CYCLE)
            and self.recession_probability in (RecessionProbability.VERY_LOW, RecessionProbability.LOW)
            and self.volatility_regime in (VolatilityRegime.VERY_LOW, VolatilityRegime.LOW, VolatilityRegime.NORMAL)
        )

    def get_valuation_adjustment_factor(self) -> float:
        """
        Get valuation adjustment factor based on market regime.

        Returns:
            float: Adjustment factor (0.8-1.2) to apply to valuations
                   <1.0 means apply discount, >1.0 means premium justified
        """
        factor = 1.0

        # Credit cycle adjustment
        cycle_adjustments = {
            CreditCyclePhase.EARLY_EXPANSION: 1.05,
            CreditCyclePhase.MID_CYCLE: 1.0,
            CreditCyclePhase.LATE_CYCLE: 0.95,
            CreditCyclePhase.CREDIT_STRESS: 0.85,
            CreditCyclePhase.CREDIT_CRISIS: 0.75,
        }
        factor *= cycle_adjustments.get(self.credit_cycle_phase, 1.0)

        # Recession probability adjustment
        recession_adjustments = {
            RecessionProbability.VERY_LOW: 1.02,
            RecessionProbability.LOW: 1.0,
            RecessionProbability.ELEVATED: 0.95,
            RecessionProbability.HIGH: 0.88,
            RecessionProbability.IMMINENT: 0.80,
        }
        factor *= recession_adjustments.get(self.recession_probability, 1.0)

        # Volatility adjustment
        vol_adjustments = {
            VolatilityRegime.VERY_LOW: 1.02,
            VolatilityRegime.LOW: 1.01,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.ELEVATED: 0.97,
            VolatilityRegime.HIGH: 0.93,
            VolatilityRegime.EXTREME: 0.85,
        }
        factor *= vol_adjustments.get(self.volatility_regime, 1.0)

        return round(factor, 3)
