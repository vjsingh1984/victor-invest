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

"""Credit Cycle Analyzer Service.

This module provides analysis of the credit cycle based on multiple indicators
including credit spreads, yield curve shape, VIX levels, and Fed policy.

Credit Cycle Phases:
- EARLY_EXPANSION: Post-recession, spreads tightening, growth accelerating
- MID_CYCLE: Healthy expansion, stable spreads, moderate growth
- LATE_CYCLE: Slowing growth, spreads widening, caution warranted
- CREDIT_STRESS: Elevated risk, widening spreads, defensive positioning
- CREDIT_CRISIS: Severe stress, extremely wide spreads, max defensive

Key Indicators:
- BAA-10Y Spread: Investment grade credit spread (normal ~200 bps)
- VIX: Market volatility expectation
- Fed Funds Rate: Monetary policy stance
- Yield Curve: Shape indicates growth expectations

Investment Implications:
- Early Expansion: Favor cyclicals, high-yield bonds, small caps
- Mid-Cycle: Balanced approach, quality growth
- Late Cycle: Favor quality, reduce leverage exposure
- Credit Stress: Defensive sectors, reduce high-yield
- Credit Crisis: Cash, treasuries, quality defensives

Example:
    analyzer = get_credit_cycle_analyzer()

    # Get current analysis
    analysis = await analyzer.analyze()
    print(f"Credit cycle phase: {analysis.phase}")
    print(f"Credit spread: {analysis.baa_spread_bps} bps")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from investigator.domain.models.market_context import (
    CreditCyclePhase,
    FedPolicyStance,
    RecessionProbability,
    VolatilityRegime,
)

logger = logging.getLogger(__name__)


@dataclass
class CreditCycleAnalysis:
    """Complete credit cycle analysis result.

    Attributes:
        date: Analysis date
        phase: Credit cycle phase classification
        baa_spread_bps: BAA-10Y credit spread in basis points
        baa_spread_percentile: Current spread vs historical (0-100)
        vix_level: Current VIX level
        volatility_regime: Volatility regime classification
        fed_funds_rate: Current Fed funds rate
        fed_policy_stance: Fed policy stance classification
        recession_probability: Recession probability classification
        confidence: Confidence in the analysis (0-100)
        factors: Contributing factors to the analysis
        interpretation: Human-readable interpretation
        warnings: Any data quality warnings
    """

    date: date
    phase: CreditCyclePhase = CreditCyclePhase.UNKNOWN
    baa_spread_bps: Optional[float] = None
    baa_spread_percentile: Optional[float] = None
    vix_level: Optional[float] = None
    volatility_regime: VolatilityRegime = VolatilityRegime.UNKNOWN
    fed_funds_rate: Optional[float] = None
    fed_policy_stance: FedPolicyStance = FedPolicyStance.UNKNOWN
    recession_probability: RecessionProbability = RecessionProbability.UNKNOWN
    confidence: float = 0.0
    factors: List[str] = field(default_factory=list)
    interpretation: str = ""
    sector_recommendations: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": str(self.date),
            "phase": self.phase.value,
            "credit_indicators": {
                "baa_spread_bps": self.baa_spread_bps,
                "baa_spread_percentile": self.baa_spread_percentile,
            },
            "volatility": {
                "vix_level": self.vix_level,
                "regime": self.volatility_regime.value,
            },
            "fed_policy": {
                "fed_funds_rate": self.fed_funds_rate,
                "stance": self.fed_policy_stance.value,
            },
            "recession_probability": self.recession_probability.value,
            "confidence": self.confidence,
            "factors": self.factors,
            "interpretation": self.interpretation,
            "sector_recommendations": self.sector_recommendations,
            "warnings": self.warnings,
        }


class CreditCycleAnalyzer:
    """Service for analyzing the credit cycle.

    Combines multiple indicators to determine the current phase
    of the credit cycle and provide investment guidance.

    SOLID: Single Responsibility - only handles credit cycle analysis
    """

    # Historical thresholds for BAA-10Y spread
    # Normal: ~200 bps, Stressed: >300 bps, Crisis: >500 bps
    SPREAD_THRESHOLDS = {
        "tight": 150,  # Early expansion
        "normal": 200,  # Mid cycle
        "wide": 300,  # Late cycle
        "stressed": 400,  # Credit stress
        "crisis": 500,  # Credit crisis
    }

    # VIX thresholds
    VIX_THRESHOLDS = {
        "very_low": 12,
        "low": 16,
        "normal": 20,
        "elevated": 25,
        "high": 35,
    }

    def __init__(self, fred_client=None, yield_curve_analyzer=None):
        """Initialize analyzer.

        Args:
            fred_client: FRED data client (created if not provided)
            yield_curve_analyzer: YieldCurveAnalyzer instance
        """
        self._fred_client = fred_client
        self._yield_curve_analyzer = yield_curve_analyzer

    def _get_fred_client(self):
        """Lazy-load FRED client."""
        if self._fred_client is None:
            try:
                from investigator.infrastructure.external.fred.macro_indicators import (
                    get_macro_indicators_service,
                )

                self._fred_client = get_macro_indicators_service()
            except ImportError:
                logger.warning("FRED client not available")
        return self._fred_client

    def _get_yield_curve_analyzer(self):
        """Lazy-load yield curve analyzer."""
        if self._yield_curve_analyzer is None:
            from investigator.domain.services.market_regime.yield_curve_analyzer import (
                get_yield_curve_analyzer,
            )

            self._yield_curve_analyzer = get_yield_curve_analyzer()
        return self._yield_curve_analyzer

    async def analyze(self) -> CreditCycleAnalysis:
        """Analyze current credit cycle.

        Returns:
            CreditCycleAnalysis with phase and recommendations
        """
        analysis = CreditCycleAnalysis(date=date.today())
        factors = []
        scores = {
            CreditCyclePhase.EARLY_EXPANSION: 0,
            CreditCyclePhase.MID_CYCLE: 0,
            CreditCyclePhase.LATE_CYCLE: 0,
            CreditCyclePhase.CREDIT_STRESS: 0,
            CreditCyclePhase.CREDIT_CRISIS: 0,
        }

        # Get credit spread data
        spread_data = await self._get_credit_spread()
        if spread_data:
            analysis.baa_spread_bps = spread_data.get("spread_bps")
            analysis.baa_spread_percentile = spread_data.get("percentile")
            spread_scores, spread_factors = self._score_credit_spread(spread_data)
            for phase, score in spread_scores.items():
                scores[phase] += score
            factors.extend(spread_factors)

        # Get VIX data
        vix_data = await self._get_vix()
        if vix_data:
            analysis.vix_level = vix_data.get("level")
            analysis.volatility_regime = self._classify_volatility(vix_data.get("level"))
            vix_scores, vix_factors = self._score_vix(vix_data)
            for phase, score in vix_scores.items():
                scores[phase] += score
            factors.extend(vix_factors)

        # Get Fed funds rate
        fed_data = await self._get_fed_funds()
        if fed_data:
            analysis.fed_funds_rate = fed_data.get("rate")
            analysis.fed_policy_stance = self._classify_fed_stance(fed_data)
            fed_scores, fed_factors = self._score_fed_policy(fed_data)
            for phase, score in fed_scores.items():
                scores[phase] += score
            factors.extend(fed_factors)

        # Get yield curve analysis
        try:
            yc_analyzer = self._get_yield_curve_analyzer()
            yc_analysis = await yc_analyzer.analyze()
            yc_scores, yc_factors = self._score_yield_curve(yc_analysis)
            for phase, score in yc_scores.items():
                scores[phase] += score
            factors.extend(yc_factors)
        except Exception as e:
            logger.debug(f"Yield curve analysis unavailable: {e}")
            analysis.warnings.append("Yield curve data unavailable")

        # Determine phase from highest score
        analysis.phase = max(scores, key=scores.get)
        analysis.factors = factors

        # Calculate confidence (how clear is the signal)
        total_score = sum(scores.values())
        if total_score > 0:
            max_score = max(scores.values())
            analysis.confidence = round((max_score / total_score) * 100, 1)
        else:
            analysis.confidence = 0
            analysis.phase = CreditCyclePhase.UNKNOWN

        # Set interpretation and recommendations
        analysis.interpretation = self._get_interpretation(analysis.phase)
        analysis.sector_recommendations = self._get_sector_recommendations(analysis.phase)

        # Derive recession probability from phase
        analysis.recession_probability = self._derive_recession_probability(analysis)

        return analysis

    async def _get_credit_spread(self) -> Optional[Dict[str, Any]]:
        """Get BAA-10Y credit spread from FRED."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            # Get BAA10Y spread (BAA corporate bond yield minus 10Y Treasury)
            query = text(
                """
                SELECT value, series_date
                FROM macro_indicator_values
                WHERE indicator_id = (
                    SELECT id FROM macro_indicators WHERE series_id = 'BAA10Y'
                )
                ORDER BY series_date DESC
                LIMIT 1
            """
            )

            with engine.connect() as conn:
                result = conn.execute(query).fetchone()
                if result:
                    spread_pct = float(result[0])
                    spread_bps = spread_pct * 100  # Convert to basis points

                    # Get historical percentile
                    percentile_query = text(
                        """
                        SELECT
                            COUNT(*) FILTER (WHERE value < :current) * 100.0 / COUNT(*)
                        FROM macro_indicator_values
                        WHERE indicator_id = (
                            SELECT id FROM macro_indicators WHERE series_id = 'BAA10Y'
                        )
                        AND series_date >= CURRENT_DATE - INTERVAL '10 years'
                    """
                    )
                    percentile_result = conn.execute(percentile_query, {"current": spread_pct}).fetchone()
                    percentile = float(percentile_result[0]) if percentile_result else None

                    return {
                        "spread_bps": spread_bps,
                        "spread_pct": spread_pct,
                        "percentile": percentile,
                        "date": result[1],
                    }

        except Exception as e:
            logger.debug(f"Error getting credit spread: {e}")

        return None

    async def _get_vix(self) -> Optional[Dict[str, Any]]:
        """Get VIX level from FRED."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            query = text(
                """
                SELECT value, series_date
                FROM macro_indicator_values
                WHERE indicator_id = (
                    SELECT id FROM macro_indicators WHERE series_id = 'VIXCLS'
                )
                ORDER BY series_date DESC
                LIMIT 1
            """
            )

            with engine.connect() as conn:
                result = conn.execute(query).fetchone()
                if result:
                    return {
                        "level": float(result[0]),
                        "date": result[1],
                    }

        except Exception as e:
            logger.debug(f"Error getting VIX: {e}")

        return None

    async def _get_fed_funds(self) -> Optional[Dict[str, Any]]:
        """Get Fed funds rate from FRED."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine("postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}:5432/sec_database")

            # Get current and historical rates
            query = text(
                """
                SELECT value, series_date
                FROM macro_indicator_values
                WHERE indicator_id = (
                    SELECT id FROM macro_indicators WHERE series_id = 'FEDFUNDS'
                )
                ORDER BY series_date DESC
                LIMIT 13
            """
            )

            with engine.connect() as conn:
                results = conn.execute(query).fetchall()
                if results:
                    current = float(results[0][0])
                    year_ago = float(results[-1][0]) if len(results) > 12 else current

                    return {
                        "rate": current,
                        "rate_1y_ago": year_ago,
                        "change_1y": current - year_ago,
                        "date": results[0][1],
                    }

        except Exception as e:
            logger.debug(f"Error getting Fed funds: {e}")

        return None

    def _score_credit_spread(self, data: Dict[str, Any]) -> tuple:
        """Score credit spread indicator."""
        scores = {phase: 0 for phase in CreditCyclePhase if phase != CreditCyclePhase.UNKNOWN}
        factors = []

        spread = data.get("spread_bps", 0)
        percentile = data.get("percentile")

        if spread < self.SPREAD_THRESHOLDS["tight"]:
            scores[CreditCyclePhase.EARLY_EXPANSION] += 30
            scores[CreditCyclePhase.MID_CYCLE] += 20
            factors.append(f"Tight credit spreads ({spread:.0f} bps) - risk appetite healthy")
        elif spread < self.SPREAD_THRESHOLDS["normal"]:
            scores[CreditCyclePhase.MID_CYCLE] += 30
            scores[CreditCyclePhase.EARLY_EXPANSION] += 15
            factors.append(f"Normal credit spreads ({spread:.0f} bps) - stable conditions")
        elif spread < self.SPREAD_THRESHOLDS["wide"]:
            scores[CreditCyclePhase.LATE_CYCLE] += 30
            scores[CreditCyclePhase.MID_CYCLE] += 10
            factors.append(f"Widening credit spreads ({spread:.0f} bps) - late cycle concerns")
        elif spread < self.SPREAD_THRESHOLDS["stressed"]:
            scores[CreditCyclePhase.CREDIT_STRESS] += 35
            scores[CreditCyclePhase.LATE_CYCLE] += 15
            factors.append(f"Elevated credit spreads ({spread:.0f} bps) - credit stress")
        else:
            scores[CreditCyclePhase.CREDIT_CRISIS] += 40
            scores[CreditCyclePhase.CREDIT_STRESS] += 15
            factors.append(f"Crisis-level credit spreads ({spread:.0f} bps) - severe stress")

        # Percentile adjustment
        if percentile is not None:
            if percentile > 90:
                scores[CreditCyclePhase.CREDIT_CRISIS] += 10
                factors.append(f"Credit spreads at {percentile:.0f}th percentile (10Y)")
            elif percentile < 20:
                scores[CreditCyclePhase.EARLY_EXPANSION] += 10
                factors.append(f"Credit spreads at {percentile:.0f}th percentile (10Y)")

        return scores, factors

    def _score_vix(self, data: Dict[str, Any]) -> tuple:
        """Score VIX indicator."""
        scores = {phase: 0 for phase in CreditCyclePhase if phase != CreditCyclePhase.UNKNOWN}
        factors = []

        vix = data.get("level", 20)

        if vix < self.VIX_THRESHOLDS["very_low"]:
            scores[CreditCyclePhase.EARLY_EXPANSION] += 20
            scores[CreditCyclePhase.MID_CYCLE] += 15
            factors.append(f"Very low VIX ({vix:.1f}) - market complacency")
        elif vix < self.VIX_THRESHOLDS["low"]:
            scores[CreditCyclePhase.MID_CYCLE] += 20
            scores[CreditCyclePhase.EARLY_EXPANSION] += 10
            factors.append(f"Low VIX ({vix:.1f}) - calm markets")
        elif vix < self.VIX_THRESHOLDS["normal"]:
            scores[CreditCyclePhase.MID_CYCLE] += 15
            scores[CreditCyclePhase.LATE_CYCLE] += 10
            factors.append(f"Normal VIX ({vix:.1f}) - balanced uncertainty")
        elif vix < self.VIX_THRESHOLDS["elevated"]:
            scores[CreditCyclePhase.LATE_CYCLE] += 20
            scores[CreditCyclePhase.CREDIT_STRESS] += 10
            factors.append(f"Elevated VIX ({vix:.1f}) - rising uncertainty")
        elif vix < self.VIX_THRESHOLDS["high"]:
            scores[CreditCyclePhase.CREDIT_STRESS] += 25
            scores[CreditCyclePhase.LATE_CYCLE] += 10
            factors.append(f"High VIX ({vix:.1f}) - significant stress")
        else:
            scores[CreditCyclePhase.CREDIT_CRISIS] += 30
            scores[CreditCyclePhase.CREDIT_STRESS] += 15
            factors.append(f"Extreme VIX ({vix:.1f}) - panic conditions")

        return scores, factors

    def _score_fed_policy(self, data: Dict[str, Any]) -> tuple:
        """Score Fed policy indicator."""
        scores = {phase: 0 for phase in CreditCyclePhase if phase != CreditCyclePhase.UNKNOWN}
        factors = []

        rate = data.get("rate", 0)
        change_1y = data.get("change_1y", 0)

        # Rate level
        if rate < 1:
            scores[CreditCyclePhase.EARLY_EXPANSION] += 15
            scores[CreditCyclePhase.CREDIT_CRISIS] += 10  # Could also indicate crisis response
            factors.append(f"Near-zero Fed funds ({rate:.2f}%) - accommodative")
        elif rate < 3:
            scores[CreditCyclePhase.MID_CYCLE] += 15
            factors.append(f"Moderate Fed funds ({rate:.2f}%) - normalizing")
        elif rate < 5:
            scores[CreditCyclePhase.LATE_CYCLE] += 15
            factors.append(f"Elevated Fed funds ({rate:.2f}%) - restrictive")
        else:
            scores[CreditCyclePhase.LATE_CYCLE] += 20
            scores[CreditCyclePhase.CREDIT_STRESS] += 10
            factors.append(f"High Fed funds ({rate:.2f}%) - very restrictive")

        # Rate trajectory
        if change_1y > 1.5:
            scores[CreditCyclePhase.LATE_CYCLE] += 10
            factors.append(f"Fed tightening aggressively (+{change_1y:.2f}% YoY)")
        elif change_1y > 0.5:
            scores[CreditCyclePhase.MID_CYCLE] += 10
            factors.append(f"Fed tightening gradually (+{change_1y:.2f}% YoY)")
        elif change_1y < -1.0:
            scores[CreditCyclePhase.EARLY_EXPANSION] += 10
            scores[CreditCyclePhase.CREDIT_CRISIS] += 5
            factors.append(f"Fed easing significantly ({change_1y:.2f}% YoY)")

        return scores, factors

    def _score_yield_curve(self, yc_analysis) -> tuple:
        """Score yield curve indicator."""
        from investigator.domain.services.market_regime.yield_curve_analyzer import YieldCurveShape

        scores = {phase: 0 for phase in CreditCyclePhase if phase != CreditCyclePhase.UNKNOWN}
        factors = []

        shape = yc_analysis.shape

        if shape == YieldCurveShape.STEEP:
            scores[CreditCyclePhase.EARLY_EXPANSION] += 25
            factors.append("Steep yield curve - growth expectations strong")
        elif shape == YieldCurveShape.NORMAL:
            scores[CreditCyclePhase.MID_CYCLE] += 25
            factors.append("Normal yield curve - balanced outlook")
        elif shape == YieldCurveShape.FLAT:
            scores[CreditCyclePhase.LATE_CYCLE] += 25
            factors.append("Flat yield curve - late cycle warning")
        elif shape == YieldCurveShape.INVERTED:
            scores[CreditCyclePhase.LATE_CYCLE] += 20
            scores[CreditCyclePhase.CREDIT_STRESS] += 15
            factors.append("Inverted yield curve - recession signal")
        elif shape == YieldCurveShape.DEEPLY_INVERTED:
            scores[CreditCyclePhase.CREDIT_STRESS] += 25
            scores[CreditCyclePhase.CREDIT_CRISIS] += 10
            factors.append("Deeply inverted curve - high recession risk")

        return scores, factors

    def _classify_volatility(self, vix: Optional[float]) -> VolatilityRegime:
        """Classify volatility regime from VIX level."""
        if vix is None:
            return VolatilityRegime.UNKNOWN
        if vix < self.VIX_THRESHOLDS["very_low"]:
            return VolatilityRegime.VERY_LOW
        if vix < self.VIX_THRESHOLDS["low"]:
            return VolatilityRegime.LOW
        if vix < self.VIX_THRESHOLDS["normal"]:
            return VolatilityRegime.NORMAL
        if vix < self.VIX_THRESHOLDS["elevated"]:
            return VolatilityRegime.ELEVATED
        if vix < self.VIX_THRESHOLDS["high"]:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME

    def _classify_fed_stance(self, data: Dict[str, Any]) -> FedPolicyStance:
        """Classify Fed policy stance."""
        rate = data.get("rate", 0)
        change_1y = data.get("change_1y", 0)

        if change_1y > 2:
            return FedPolicyStance.VERY_HAWKISH
        if change_1y > 0.75:
            return FedPolicyStance.HAWKISH
        if change_1y < -1.5:
            return FedPolicyStance.VERY_DOVISH
        if change_1y < -0.5:
            return FedPolicyStance.DOVISH
        return FedPolicyStance.NEUTRAL

    def _derive_recession_probability(self, analysis: CreditCycleAnalysis) -> RecessionProbability:
        """Derive recession probability from analysis."""
        phase = analysis.phase

        probability_map = {
            CreditCyclePhase.EARLY_EXPANSION: RecessionProbability.VERY_LOW,
            CreditCyclePhase.MID_CYCLE: RecessionProbability.LOW,
            CreditCyclePhase.LATE_CYCLE: RecessionProbability.ELEVATED,
            CreditCyclePhase.CREDIT_STRESS: RecessionProbability.HIGH,
            CreditCyclePhase.CREDIT_CRISIS: RecessionProbability.IMMINENT,
            CreditCyclePhase.UNKNOWN: RecessionProbability.UNKNOWN,
        }
        return probability_map.get(phase, RecessionProbability.UNKNOWN)

    def _get_interpretation(self, phase: CreditCyclePhase) -> str:
        """Get human-readable interpretation for phase."""
        interpretations = {
            CreditCyclePhase.EARLY_EXPANSION: (
                "Economy in early expansion phase following recession. Credit spreads "
                "tightening as confidence returns. Favor cyclical sectors, high-yield "
                "bonds, and small-cap stocks. Opportune time for risk-on positioning."
            ),
            CreditCyclePhase.MID_CYCLE: (
                "Economy in healthy mid-cycle expansion. Credit conditions stable with "
                "moderate spreads. Balanced portfolio approach appropriate. Quality "
                "growth stocks and investment-grade credit offer good risk/reward."
            ),
            CreditCyclePhase.LATE_CYCLE: (
                "Economy showing late-cycle characteristics. Credit spreads beginning "
                "to widen as risks accumulate. Favor quality over leverage. Reduce "
                "exposure to cyclical and highly leveraged names. Defensive rotation advised."
            ),
            CreditCyclePhase.CREDIT_STRESS: (
                "Credit market stress elevated. Spreads widening materially. Reduce "
                "high-yield exposure. Favor investment-grade and defensive sectors. "
                "Increase cash allocation. Recession risk significantly elevated."
            ),
            CreditCyclePhase.CREDIT_CRISIS: (
                "Severe credit market stress indicating crisis conditions. Wide spreads "
                "and high volatility. Maximum defensive positioning. Favor treasuries, "
                "cash, and only highest-quality equities. Preserve capital priority."
            ),
        }
        return interpretations.get(phase, "Credit cycle phase undetermined")

    def _get_sector_recommendations(self, phase: CreditCyclePhase) -> Dict[str, str]:
        """Get sector recommendations for phase."""
        recommendations = {
            CreditCyclePhase.EARLY_EXPANSION: {
                "overweight": "Financials, Industrials, Consumer Discretionary, Small Caps",
                "neutral": "Technology, Communication Services",
                "underweight": "Utilities, Consumer Staples, REITs",
                "fixed_income": "High Yield, Emerging Market Debt",
            },
            CreditCyclePhase.MID_CYCLE: {
                "overweight": "Technology, Healthcare, Industrials",
                "neutral": "Financials, Consumer Discretionary, Energy",
                "underweight": "Utilities, High-Yield vulnerable credits",
                "fixed_income": "Investment Grade, Balanced Duration",
            },
            CreditCyclePhase.LATE_CYCLE: {
                "overweight": "Healthcare, Utilities, Consumer Staples",
                "neutral": "Technology (quality only), Energy",
                "underweight": "Financials, Consumer Discretionary, Small Caps",
                "fixed_income": "Investment Grade, Reduce High Yield",
            },
            CreditCyclePhase.CREDIT_STRESS: {
                "overweight": "Utilities, Consumer Staples, Healthcare",
                "neutral": "Technology (mega-cap quality)",
                "underweight": "Financials, Industrials, Small Caps, High Yield",
                "fixed_income": "Treasuries, Investment Grade only",
            },
            CreditCyclePhase.CREDIT_CRISIS: {
                "overweight": "Treasuries, Cash, Gold",
                "neutral": "Utilities, Consumer Staples (quality)",
                "underweight": "All cyclicals, Financials, High Yield, EM",
                "fixed_income": "Short-duration Treasuries, Cash",
            },
        }
        return recommendations.get(phase, {})


# Singleton instance
_credit_cycle_analyzer: Optional[CreditCycleAnalyzer] = None


def get_credit_cycle_analyzer() -> CreditCycleAnalyzer:
    """Get or create singleton analyzer instance."""
    global _credit_cycle_analyzer
    if _credit_cycle_analyzer is None:
        _credit_cycle_analyzer = CreditCycleAnalyzer()
    return _credit_cycle_analyzer
