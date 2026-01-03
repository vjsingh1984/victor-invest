# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Valuation Signal Integrator.

Integrates external signals (credit risk, insider sentiment, short interest,
market regime) into the valuation process to provide more accurate fair value
estimates.

Integration Points:
1. Credit Risk → Valuation discount (5-50% based on distress tier)
2. Insider Sentiment → Confidence adjustment
3. Short Interest → Contrarian signal / risk flag
4. Market Regime → WACC adjustment, equity allocation

Author: InvestiGator Team
Date: 2025-01-02
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DistressTier(Enum):
    """Company distress tier based on credit risk models."""
    HEALTHY = "healthy"              # No discount
    WATCH = "watch"                  # 5% discount
    CONCERN = "concern"              # 15% discount
    DISTRESSED = "distressed"        # 30% discount
    SEVERE_DISTRESS = "severe_distress"  # 50% discount


class InsiderSignal(Enum):
    """Insider trading sentiment signal."""
    STRONG_BUY = "strong_buy"        # +10% confidence boost
    BUY = "buy"                      # +5% confidence boost
    NEUTRAL = "neutral"              # No adjustment
    SELL = "sell"                    # -5% confidence
    STRONG_SELL = "strong_sell"      # -10% confidence


class ShortInterestSignal(Enum):
    """Short interest signal."""
    SQUEEZE_RISK = "squeeze_risk"    # High squeeze potential (contrarian bullish)
    ELEVATED = "elevated"            # Warning flag
    NORMAL = "normal"                # No signal
    LOW = "low"                      # Low short interest


@dataclass
class CreditRiskSignal:
    """Credit risk signal for valuation adjustment."""
    altman_zscore: Optional[float] = None
    altman_zone: Optional[str] = None  # "safe", "grey", "distress"
    beneish_mscore: Optional[float] = None
    manipulation_flag: bool = False
    piotroski_fscore: Optional[int] = None
    piotroski_grade: Optional[str] = None  # "strong", "moderate", "weak"
    distress_tier: DistressTier = DistressTier.HEALTHY
    discount_pct: float = 0.0
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "altman_zscore": self.altman_zscore,
            "altman_zone": self.altman_zone,
            "beneish_mscore": self.beneish_mscore,
            "manipulation_flag": self.manipulation_flag,
            "piotroski_fscore": self.piotroski_fscore,
            "piotroski_grade": self.piotroski_grade,
            "distress_tier": self.distress_tier.value,
            "discount_pct": self.discount_pct,
            "factors": self.factors,
        }


@dataclass
class InsiderSentimentSignal:
    """Insider sentiment signal for confidence adjustment."""
    signal: InsiderSignal = InsiderSignal.NEUTRAL
    buy_sell_ratio: Optional[float] = None
    net_shares_change: Optional[int] = None
    cluster_detected: bool = False
    confidence_adjustment: float = 0.0
    interpretation: str = ""
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "buy_sell_ratio": self.buy_sell_ratio,
            "net_shares_change": self.net_shares_change,
            "cluster_detected": self.cluster_detected,
            "confidence_adjustment": self.confidence_adjustment,
            "interpretation": self.interpretation,
            "factors": self.factors,
        }


@dataclass
class ShortInterestAdjustment:
    """Short interest signal for valuation."""
    signal: ShortInterestSignal = ShortInterestSignal.NORMAL
    short_percent_float: Optional[float] = None
    days_to_cover: Optional[float] = None
    squeeze_score: Optional[float] = None
    is_contrarian_signal: bool = False
    warning_flag: bool = False
    interpretation: str = ""
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "short_percent_float": self.short_percent_float,
            "days_to_cover": self.days_to_cover,
            "squeeze_score": self.squeeze_score,
            "is_contrarian_signal": self.is_contrarian_signal,
            "warning_flag": self.warning_flag,
            "interpretation": self.interpretation,
            "factors": self.factors,
        }


@dataclass
class MarketRegimeAdjustment:
    """Market regime adjustment for WACC and valuations."""
    credit_cycle_phase: str = "mid_cycle"
    volatility_regime: str = "normal"
    recession_probability: str = "low"
    fed_policy_stance: str = "neutral"
    risk_free_rate: float = 0.04  # Current 10Y yield
    wacc_spread_adjustment_bps: int = 0
    equity_allocation_adjustment: float = 0.0
    valuation_adjustment_factor: float = 1.0
    interpretation: str = ""
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credit_cycle_phase": self.credit_cycle_phase,
            "volatility_regime": self.volatility_regime,
            "recession_probability": self.recession_probability,
            "fed_policy_stance": self.fed_policy_stance,
            "risk_free_rate": self.risk_free_rate,
            "wacc_spread_adjustment_bps": self.wacc_spread_adjustment_bps,
            "equity_allocation_adjustment": self.equity_allocation_adjustment,
            "valuation_adjustment_factor": self.valuation_adjustment_factor,
            "interpretation": self.interpretation,
            "factors": self.factors,
        }


@dataclass
class IntegratedValuationSignals:
    """Combined signals for valuation adjustment."""
    symbol: str
    base_fair_value: float
    adjusted_fair_value: float
    current_price: float
    credit_risk: Optional[CreditRiskSignal] = None
    insider_sentiment: Optional[InsiderSentimentSignal] = None
    short_interest: Optional[ShortInterestAdjustment] = None
    market_regime: Optional[MarketRegimeAdjustment] = None
    total_adjustment_pct: float = 0.0
    confidence_adjustment: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_fair_value": self.base_fair_value,
            "adjusted_fair_value": self.adjusted_fair_value,
            "current_price": self.current_price,
            "upside_pct": ((self.adjusted_fair_value - self.current_price) / self.current_price) * 100,
            "credit_risk": self.credit_risk.to_dict() if self.credit_risk else None,
            "insider_sentiment": self.insider_sentiment.to_dict() if self.insider_sentiment else None,
            "short_interest": self.short_interest.to_dict() if self.short_interest else None,
            "market_regime": self.market_regime.to_dict() if self.market_regime else None,
            "total_adjustment_pct": self.total_adjustment_pct,
            "confidence_adjustment": self.confidence_adjustment,
            "warnings": self.warnings,
        }


class ValuationSignalIntegrator:
    """Integrates external signals into valuation adjustments.

    This class provides methods to:
    1. Calculate credit risk-based valuation discounts
    2. Adjust confidence based on insider sentiment
    3. Flag short squeeze risks and contrarian signals
    4. Apply market regime adjustments to WACC

    Example:
        integrator = ValuationSignalIntegrator()

        # Get credit risk signal
        credit_signal = integrator.calculate_credit_risk_signal(
            altman_zscore=2.5,
            beneish_mscore=-2.1,
            piotroski_fscore=7
        )

        # Apply all adjustments
        result = integrator.integrate_signals(
            symbol="AAPL",
            base_fair_value=150.0,
            current_price=145.0,
            credit_risk_data={...},
            insider_data={...},
            short_interest_data={...},
            market_regime_data={...}
        )
    """

    # Credit risk discount tiers
    DISTRESS_DISCOUNTS = {
        DistressTier.HEALTHY: 0.0,
        DistressTier.WATCH: 0.05,
        DistressTier.CONCERN: 0.15,
        DistressTier.DISTRESSED: 0.30,
        DistressTier.SEVERE_DISTRESS: 0.50,
    }

    # Insider sentiment confidence adjustments
    INSIDER_CONFIDENCE_ADJUSTMENTS = {
        InsiderSignal.STRONG_BUY: 0.10,
        InsiderSignal.BUY: 0.05,
        InsiderSignal.NEUTRAL: 0.0,
        InsiderSignal.SELL: -0.05,
        InsiderSignal.STRONG_SELL: -0.10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the signal integrator.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}

        # Override defaults from config if provided
        self.distress_discounts = self.config.get(
            "distress_discounts", self.DISTRESS_DISCOUNTS
        )
        self.insider_adjustments = self.config.get(
            "insider_confidence_adjustments", self.INSIDER_CONFIDENCE_ADJUSTMENTS
        )

    def calculate_credit_risk_signal(
        self,
        altman_zscore: Optional[float] = None,
        beneish_mscore: Optional[float] = None,
        piotroski_fscore: Optional[int] = None,
    ) -> CreditRiskSignal:
        """Calculate credit risk signal from credit models.

        Args:
            altman_zscore: Altman Z-Score (>2.99 safe, 1.81-2.99 grey, <1.81 distress)
            beneish_mscore: Beneish M-Score (>-1.78 indicates manipulation)
            piotroski_fscore: Piotroski F-Score (0-9, higher is better)

        Returns:
            CreditRiskSignal with distress tier and discount
        """
        factors = []
        distress_points = 0

        # Analyze Altman Z-Score
        altman_zone = None
        if altman_zscore is not None:
            if altman_zscore > 2.99:
                altman_zone = "safe"
                factors.append(f"Altman Z-Score {altman_zscore:.2f} in safe zone (>2.99)")
            elif altman_zscore >= 1.81:
                altman_zone = "grey"
                distress_points += 1
                factors.append(f"Altman Z-Score {altman_zscore:.2f} in grey zone (1.81-2.99)")
            else:
                altman_zone = "distress"
                distress_points += 3
                factors.append(f"Altman Z-Score {altman_zscore:.2f} in distress zone (<1.81)")

        # Analyze Beneish M-Score
        manipulation_flag = False
        if beneish_mscore is not None:
            if beneish_mscore > -1.78:
                manipulation_flag = True
                distress_points += 2
                factors.append(f"Beneish M-Score {beneish_mscore:.2f} indicates manipulation risk (>-1.78)")
            else:
                factors.append(f"Beneish M-Score {beneish_mscore:.2f} - no manipulation signal")

        # Analyze Piotroski F-Score
        piotroski_grade = None
        if piotroski_fscore is not None:
            if piotroski_fscore >= 8:
                piotroski_grade = "strong"
                factors.append(f"Piotroski F-Score {piotroski_fscore} indicates strong fundamentals")
            elif piotroski_fscore >= 5:
                piotroski_grade = "moderate"
                factors.append(f"Piotroski F-Score {piotroski_fscore} indicates moderate fundamentals")
            else:
                piotroski_grade = "weak"
                distress_points += 2
                factors.append(f"Piotroski F-Score {piotroski_fscore} indicates weak fundamentals")

        # Determine distress tier based on cumulative points
        if distress_points >= 5:
            distress_tier = DistressTier.SEVERE_DISTRESS
        elif distress_points >= 4:
            distress_tier = DistressTier.DISTRESSED
        elif distress_points >= 2:
            distress_tier = DistressTier.CONCERN
        elif distress_points >= 1:
            distress_tier = DistressTier.WATCH
        else:
            distress_tier = DistressTier.HEALTHY

        discount_pct = self.distress_discounts.get(distress_tier, 0.0)
        if isinstance(discount_pct, DistressTier):
            discount_pct = self.DISTRESS_DISCOUNTS.get(discount_pct, 0.0)

        return CreditRiskSignal(
            altman_zscore=altman_zscore,
            altman_zone=altman_zone,
            beneish_mscore=beneish_mscore,
            manipulation_flag=manipulation_flag,
            piotroski_fscore=piotroski_fscore,
            piotroski_grade=piotroski_grade,
            distress_tier=distress_tier,
            discount_pct=discount_pct,
            factors=factors,
        )

    def calculate_insider_sentiment_signal(
        self,
        buy_sell_ratio: Optional[float] = None,
        net_shares_change: Optional[int] = None,
        cluster_detected: bool = False,
        sentiment_score: Optional[float] = None,
    ) -> InsiderSentimentSignal:
        """Calculate insider sentiment signal.

        Args:
            buy_sell_ratio: Ratio of buy to sell transactions
            net_shares_change: Net change in shares held by insiders
            cluster_detected: Whether coordinated buying/selling detected
            sentiment_score: Pre-calculated sentiment score (-1 to +1)

        Returns:
            InsiderSentimentSignal with confidence adjustment
        """
        factors = []

        # Determine signal level
        if sentiment_score is not None:
            if sentiment_score >= 0.7:
                signal = InsiderSignal.STRONG_BUY
                factors.append(f"Sentiment score {sentiment_score:.2f} indicates strong insider buying")
            elif sentiment_score >= 0.3:
                signal = InsiderSignal.BUY
                factors.append(f"Sentiment score {sentiment_score:.2f} indicates insider buying")
            elif sentiment_score >= -0.3:
                signal = InsiderSignal.NEUTRAL
                factors.append(f"Sentiment score {sentiment_score:.2f} indicates neutral insider activity")
            elif sentiment_score >= -0.7:
                signal = InsiderSignal.SELL
                factors.append(f"Sentiment score {sentiment_score:.2f} indicates insider selling")
            else:
                signal = InsiderSignal.STRONG_SELL
                factors.append(f"Sentiment score {sentiment_score:.2f} indicates heavy insider selling")
        elif buy_sell_ratio is not None:
            if buy_sell_ratio >= 3.0:
                signal = InsiderSignal.STRONG_BUY
                factors.append(f"Buy/sell ratio {buy_sell_ratio:.1f} indicates strong insider buying")
            elif buy_sell_ratio >= 1.5:
                signal = InsiderSignal.BUY
                factors.append(f"Buy/sell ratio {buy_sell_ratio:.1f} indicates insider buying")
            elif buy_sell_ratio >= 0.67:
                signal = InsiderSignal.NEUTRAL
                factors.append(f"Buy/sell ratio {buy_sell_ratio:.1f} indicates neutral activity")
            elif buy_sell_ratio >= 0.33:
                signal = InsiderSignal.SELL
                factors.append(f"Buy/sell ratio {buy_sell_ratio:.1f} indicates insider selling")
            else:
                signal = InsiderSignal.STRONG_SELL
                factors.append(f"Buy/sell ratio {buy_sell_ratio:.1f} indicates heavy selling")
        else:
            signal = InsiderSignal.NEUTRAL
            factors.append("Insufficient insider data for signal")

        # Cluster detection amplifies signal
        if cluster_detected:
            factors.append("Coordinated insider activity detected - signal amplified")
            # Amplify the signal if cluster detected
            if signal == InsiderSignal.BUY:
                signal = InsiderSignal.STRONG_BUY
            elif signal == InsiderSignal.SELL:
                signal = InsiderSignal.STRONG_SELL

        confidence_adjustment = self.insider_adjustments.get(signal, 0.0)
        if isinstance(confidence_adjustment, InsiderSignal):
            confidence_adjustment = self.INSIDER_CONFIDENCE_ADJUSTMENTS.get(confidence_adjustment, 0.0)

        # Generate interpretation
        interpretations = {
            InsiderSignal.STRONG_BUY: "Strong insider buying supports bullish thesis",
            InsiderSignal.BUY: "Insider buying suggests confidence in company",
            InsiderSignal.NEUTRAL: "Insider activity is neutral - no clear signal",
            InsiderSignal.SELL: "Insider selling warrants caution",
            InsiderSignal.STRONG_SELL: "Heavy insider selling is a significant warning sign",
        }

        return InsiderSentimentSignal(
            signal=signal,
            buy_sell_ratio=buy_sell_ratio,
            net_shares_change=net_shares_change,
            cluster_detected=cluster_detected,
            confidence_adjustment=confidence_adjustment,
            interpretation=interpretations.get(signal, ""),
            factors=factors,
        )

    def calculate_short_interest_signal(
        self,
        short_percent_float: Optional[float] = None,
        days_to_cover: Optional[float] = None,
        squeeze_score: Optional[float] = None,
    ) -> ShortInterestAdjustment:
        """Calculate short interest signal.

        Args:
            short_percent_float: Short interest as % of float
            days_to_cover: Days to cover short position
            squeeze_score: Pre-calculated squeeze risk score (0-100)

        Returns:
            ShortInterestAdjustment with signal and flags
        """
        factors = []
        is_contrarian = False
        warning_flag = False

        # Determine signal based on short metrics
        if squeeze_score is not None and squeeze_score >= 70:
            signal = ShortInterestSignal.SQUEEZE_RISK
            is_contrarian = True
            factors.append(f"Squeeze score {squeeze_score:.0f} indicates high squeeze potential")
        elif short_percent_float is not None:
            if short_percent_float >= 30:
                signal = ShortInterestSignal.SQUEEZE_RISK
                is_contrarian = True
                warning_flag = True
                factors.append(f"Short % of float {short_percent_float:.1f}% is extremely elevated")
            elif short_percent_float >= 15:
                signal = ShortInterestSignal.ELEVATED
                warning_flag = True
                factors.append(f"Short % of float {short_percent_float:.1f}% is elevated")
            elif short_percent_float >= 5:
                signal = ShortInterestSignal.NORMAL
                factors.append(f"Short % of float {short_percent_float:.1f}% is normal")
            else:
                signal = ShortInterestSignal.LOW
                factors.append(f"Short % of float {short_percent_float:.1f}% is low")
        else:
            signal = ShortInterestSignal.NORMAL
            factors.append("Insufficient short interest data")

        # Days to cover analysis
        if days_to_cover is not None:
            if days_to_cover >= 10:
                factors.append(f"Days to cover {days_to_cover:.1f} suggests squeeze risk")
                if signal == ShortInterestSignal.ELEVATED:
                    signal = ShortInterestSignal.SQUEEZE_RISK
                    is_contrarian = True
            elif days_to_cover >= 5:
                factors.append(f"Days to cover {days_to_cover:.1f} is elevated")

        # Generate interpretation
        interpretations = {
            ShortInterestSignal.SQUEEZE_RISK:
                "High short interest creates potential squeeze opportunity - contrarian bullish signal",
            ShortInterestSignal.ELEVATED:
                "Elevated short interest - market skepticism or hedging activity",
            ShortInterestSignal.NORMAL:
                "Short interest at normal levels - no significant signal",
            ShortInterestSignal.LOW:
                "Low short interest - limited bearish positioning",
        }

        return ShortInterestAdjustment(
            signal=signal,
            short_percent_float=short_percent_float,
            days_to_cover=days_to_cover,
            squeeze_score=squeeze_score,
            is_contrarian_signal=is_contrarian,
            warning_flag=warning_flag,
            interpretation=interpretations.get(signal, ""),
            factors=factors,
        )

    def calculate_market_regime_adjustment(
        self,
        credit_cycle_phase: str = "mid_cycle",
        volatility_regime: str = "normal",
        recession_probability: str = "low",
        fed_policy_stance: str = "neutral",
        risk_free_rate: float = 0.04,
        yield_curve_spread_bps: Optional[int] = None,
    ) -> MarketRegimeAdjustment:
        """Calculate market regime adjustment for valuation.

        Args:
            credit_cycle_phase: Credit cycle phase
            volatility_regime: VIX-based volatility regime
            recession_probability: Recession probability level
            fed_policy_stance: Fed monetary policy stance
            risk_free_rate: Current risk-free rate (10Y yield)
            yield_curve_spread_bps: 10Y-2Y spread in basis points

        Returns:
            MarketRegimeAdjustment with WACC and valuation adjustments
        """
        factors = []
        wacc_adjustment_bps = 0
        valuation_factor = 1.0

        # Credit cycle phase adjustments
        cycle_adjustments = {
            "early_expansion": (0, 1.05, "Early expansion favors risk assets"),
            "mid_cycle": (0, 1.0, "Mid-cycle is neutral for valuations"),
            "late_cycle": (25, 0.95, "Late cycle warrants caution - WACC +25bps"),
            "credit_stress": (75, 0.85, "Credit stress requires higher discount - WACC +75bps"),
            "credit_crisis": (150, 0.75, "Credit crisis - maximum defensive - WACC +150bps"),
        }

        cycle_info = cycle_adjustments.get(credit_cycle_phase, (0, 1.0, ""))
        wacc_adjustment_bps += cycle_info[0]
        valuation_factor *= cycle_info[1]
        if cycle_info[2]:
            factors.append(cycle_info[2])

        # Volatility regime adjustments
        vol_adjustments = {
            "very_low": (-10, 1.02, "Low volatility supports higher valuations"),
            "low": (-5, 1.01, "Below-average volatility"),
            "normal": (0, 1.0, "Normal volatility environment"),
            "elevated": (15, 0.97, "Elevated volatility - WACC +15bps"),
            "high": (35, 0.93, "High volatility - WACC +35bps"),
            "extreme": (75, 0.85, "Extreme volatility - WACC +75bps"),
        }

        vol_info = vol_adjustments.get(volatility_regime, (0, 1.0, ""))
        wacc_adjustment_bps += vol_info[0]
        valuation_factor *= vol_info[1]
        if vol_info[2]:
            factors.append(vol_info[2])

        # Recession probability adjustments
        recession_adjustments = {
            "very_low": (0, 1.02, "Very low recession risk"),
            "low": (0, 1.0, "Low recession risk"),
            "elevated": (20, 0.95, "Elevated recession risk - WACC +20bps"),
            "high": (50, 0.88, "High recession risk - WACC +50bps"),
            "imminent": (100, 0.80, "Imminent recession - WACC +100bps"),
        }

        recession_info = recession_adjustments.get(recession_probability, (0, 1.0, ""))
        wacc_adjustment_bps += recession_info[0]
        valuation_factor *= recession_info[1]
        if recession_info[2]:
            factors.append(recession_info[2])

        # Yield curve inversion warning
        if yield_curve_spread_bps is not None and yield_curve_spread_bps < 0:
            factors.append(f"Yield curve inverted ({yield_curve_spread_bps}bps) - recession warning")
            wacc_adjustment_bps += 25
            valuation_factor *= 0.95

        # Calculate equity allocation adjustment
        equity_adjustment = 0.0
        if credit_cycle_phase in ["credit_stress", "credit_crisis"]:
            equity_adjustment = -0.15
        elif credit_cycle_phase == "late_cycle":
            equity_adjustment = -0.05
        elif credit_cycle_phase == "early_expansion":
            equity_adjustment = 0.10

        # Generate interpretation
        interpretation = f"Market regime: {credit_cycle_phase.replace('_', ' ').title()}"
        if wacc_adjustment_bps != 0:
            interpretation += f" with WACC adjustment of {wacc_adjustment_bps:+d}bps"
        if valuation_factor != 1.0:
            interpretation += f" and valuation factor of {valuation_factor:.2f}x"

        return MarketRegimeAdjustment(
            credit_cycle_phase=credit_cycle_phase,
            volatility_regime=volatility_regime,
            recession_probability=recession_probability,
            fed_policy_stance=fed_policy_stance,
            risk_free_rate=risk_free_rate,
            wacc_spread_adjustment_bps=wacc_adjustment_bps,
            equity_allocation_adjustment=equity_adjustment,
            valuation_adjustment_factor=valuation_factor,
            interpretation=interpretation,
            factors=factors,
        )

    def integrate_signals(
        self,
        symbol: str,
        base_fair_value: float,
        current_price: float,
        credit_risk_data: Optional[Dict[str, Any]] = None,
        insider_data: Optional[Dict[str, Any]] = None,
        short_interest_data: Optional[Dict[str, Any]] = None,
        market_regime_data: Optional[Dict[str, Any]] = None,
    ) -> IntegratedValuationSignals:
        """Integrate all signals to produce adjusted fair value.

        Args:
            symbol: Stock symbol
            base_fair_value: Base fair value from valuation models
            current_price: Current stock price
            credit_risk_data: Credit risk scores (altman_zscore, beneish_mscore, piotroski_fscore)
            insider_data: Insider sentiment data (buy_sell_ratio, sentiment_score, etc.)
            short_interest_data: Short interest data (short_percent_float, days_to_cover, etc.)
            market_regime_data: Market regime data (credit_cycle_phase, volatility_regime, etc.)

        Returns:
            IntegratedValuationSignals with adjusted fair value and all signals
        """
        warnings = []
        total_adjustment_factor = 1.0
        confidence_adjustment = 0.0

        # Process credit risk
        credit_risk = None
        if credit_risk_data:
            credit_risk = self.calculate_credit_risk_signal(
                altman_zscore=credit_risk_data.get("altman_zscore"),
                beneish_mscore=credit_risk_data.get("beneish_mscore"),
                piotroski_fscore=credit_risk_data.get("piotroski_fscore"),
            )

            # Apply credit risk discount
            if credit_risk.discount_pct > 0:
                total_adjustment_factor *= (1 - credit_risk.discount_pct)
                warnings.append(
                    f"Credit risk discount of {credit_risk.discount_pct*100:.0f}% applied "
                    f"({credit_risk.distress_tier.value})"
                )

            # Manipulation flag is a warning
            if credit_risk.manipulation_flag:
                warnings.append("Earnings manipulation risk detected (Beneish M-Score)")

        # Process insider sentiment
        insider_sentiment = None
        if insider_data:
            insider_sentiment = self.calculate_insider_sentiment_signal(
                buy_sell_ratio=insider_data.get("buy_sell_ratio"),
                net_shares_change=insider_data.get("net_shares_change"),
                cluster_detected=insider_data.get("cluster_detected", False),
                sentiment_score=insider_data.get("sentiment_score"),
            )

            # Apply confidence adjustment
            confidence_adjustment += insider_sentiment.confidence_adjustment

            if insider_sentiment.signal in [InsiderSignal.SELL, InsiderSignal.STRONG_SELL]:
                warnings.append(f"Insider selling detected: {insider_sentiment.interpretation}")

        # Process short interest
        short_interest = None
        if short_interest_data:
            short_interest = self.calculate_short_interest_signal(
                short_percent_float=short_interest_data.get("short_percent_float"),
                days_to_cover=short_interest_data.get("days_to_cover"),
                squeeze_score=short_interest_data.get("squeeze_score"),
            )

            if short_interest.warning_flag:
                warnings.append(f"Short interest warning: {short_interest.interpretation}")

            if short_interest.is_contrarian_signal:
                warnings.append("Potential short squeeze - contrarian bullish signal")

        # Process market regime
        market_regime = None
        if market_regime_data:
            market_regime = self.calculate_market_regime_adjustment(
                credit_cycle_phase=market_regime_data.get("credit_cycle_phase", "mid_cycle"),
                volatility_regime=market_regime_data.get("volatility_regime", "normal"),
                recession_probability=market_regime_data.get("recession_probability", "low"),
                fed_policy_stance=market_regime_data.get("fed_policy_stance", "neutral"),
                risk_free_rate=market_regime_data.get("risk_free_rate", 0.04),
                yield_curve_spread_bps=market_regime_data.get("yield_curve_spread_bps"),
            )

            # Apply market regime valuation adjustment
            total_adjustment_factor *= market_regime.valuation_adjustment_factor

            if market_regime.credit_cycle_phase in ["credit_stress", "credit_crisis"]:
                warnings.append(f"Defensive market regime: {market_regime.interpretation}")

        # Calculate adjusted fair value
        adjusted_fair_value = base_fair_value * total_adjustment_factor
        total_adjustment_pct = (total_adjustment_factor - 1) * 100

        logger.info(
            f"{symbol} - Signal integration: Base FV ${base_fair_value:.2f} → "
            f"Adjusted FV ${adjusted_fair_value:.2f} ({total_adjustment_pct:+.1f}%)"
        )

        return IntegratedValuationSignals(
            symbol=symbol,
            base_fair_value=base_fair_value,
            adjusted_fair_value=adjusted_fair_value,
            current_price=current_price,
            credit_risk=credit_risk,
            insider_sentiment=insider_sentiment,
            short_interest=short_interest,
            market_regime=market_regime,
            total_adjustment_pct=total_adjustment_pct,
            confidence_adjustment=confidence_adjustment,
            warnings=warnings,
        )


# Singleton accessor
_signal_integrator: Optional[ValuationSignalIntegrator] = None


def get_signal_integrator(config: Optional[Dict[str, Any]] = None) -> ValuationSignalIntegrator:
    """Get or create the signal integrator singleton.

    Args:
        config: Optional configuration overrides

    Returns:
        ValuationSignalIntegrator instance
    """
    global _signal_integrator
    if _signal_integrator is None:
        _signal_integrator = ValuationSignalIntegrator(config)
    return _signal_integrator
