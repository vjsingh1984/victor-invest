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

"""Recession Indicator Service.

This module provides recession probability assessment and economic phase
classification using multiple indicators:
- NY Fed recession probability model
- Yield curve inversion duration
- GSCPI (supply chain stress)

Investment Implications:
- EXPANSION: Full equity allocation, favor cyclicals
- LATE_CYCLE: Reduce risk, favor quality
- PRE_RECESSION: Defensive positioning, increase cash
- RECESSION: Min equity, max defensives/bonds
- RECOVERY: Increase equity, favor cyclicals

Example:
    indicator = get_recession_indicator()

    # Get recession assessment
    assessment = await indicator.assess()
    print(f"Economic phase: {assessment.phase}")
    print(f"Recession probability: {assessment.probability}%")
    print(f"Investment posture: {assessment.investment_posture}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EconomicPhase(Enum):
    """Economic cycle phase classification."""
    EXPANSION = "expansion"           # Strong growth, low recession risk
    LATE_CYCLE = "late_cycle"         # Maturing expansion, rising risks
    PRE_RECESSION = "pre_recession"   # High recession probability
    RECESSION = "recession"           # Economic contraction
    RECOVERY = "recovery"             # Early expansion from trough
    UNKNOWN = "unknown"


class InvestmentPosture(Enum):
    """Recommended investment posture."""
    AGGRESSIVE = "aggressive"         # Max equity, growth/cyclicals
    GROWTH = "growth"                 # Above-avg equity, balanced sectors
    BALANCED = "balanced"             # Neutral positioning
    CAUTIOUS = "cautious"             # Below-avg equity, quality focus
    DEFENSIVE = "defensive"           # Low equity, defensives/bonds
    STRONGLY_DEFENSIVE = "strongly_defensive"  # Min equity, max safety


@dataclass
class RecessionAssessment:
    """Comprehensive recession and economic cycle assessment.

    Attributes:
        date: Assessment date
        probability: Recession probability (0-100%)
        phase: Current economic phase
        investment_posture: Recommended investment posture
        yield_curve_inverted: Whether yield curve is inverted
        inversion_days: Days yield curve has been inverted
        gscpi_stress: Whether GSCPI shows elevated stress
        leading_indicators: Status of leading economic indicators
        confidence: Confidence in the assessment (0-1)
        warnings: Any data quality warnings
    """
    date: date
    probability: float = 0.0
    phase: EconomicPhase = EconomicPhase.UNKNOWN
    investment_posture: InvestmentPosture = InvestmentPosture.BALANCED
    yield_curve_inverted: bool = False
    inversion_days: int = 0
    gscpi_stress: bool = False
    gscpi_value: Optional[float] = None
    leading_indicators: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived values."""
        self._classify_phase()
        self._set_investment_posture()
        self._calculate_confidence()

    def _classify_phase(self):
        """Classify economic phase from indicators."""
        # Use recession probability as primary signal
        if self.probability >= 50:
            # High recession probability
            if self.probability >= 70:
                self.phase = EconomicPhase.RECESSION
            else:
                self.phase = EconomicPhase.PRE_RECESSION
        elif self.probability >= 30:
            self.phase = EconomicPhase.LATE_CYCLE
        elif self.probability >= 15:
            self.phase = EconomicPhase.EXPANSION
        else:
            # Very low probability could be expansion or recovery
            # Use yield curve as secondary signal
            if self.yield_curve_inverted:
                self.phase = EconomicPhase.LATE_CYCLE
            else:
                self.phase = EconomicPhase.EXPANSION

    def _set_investment_posture(self):
        """Set investment posture based on phase and indicators."""
        posture_map = {
            EconomicPhase.EXPANSION: InvestmentPosture.GROWTH,
            EconomicPhase.LATE_CYCLE: InvestmentPosture.CAUTIOUS,
            EconomicPhase.PRE_RECESSION: InvestmentPosture.DEFENSIVE,
            EconomicPhase.RECESSION: InvestmentPosture.STRONGLY_DEFENSIVE,
            EconomicPhase.RECOVERY: InvestmentPosture.AGGRESSIVE,
            EconomicPhase.UNKNOWN: InvestmentPosture.BALANCED,
        }
        self.investment_posture = posture_map.get(self.phase, InvestmentPosture.BALANCED)

        # Adjust for supply chain stress
        if self.gscpi_stress and self.investment_posture == InvestmentPosture.GROWTH:
            self.investment_posture = InvestmentPosture.BALANCED

    def _calculate_confidence(self):
        """Calculate confidence in the assessment."""
        # Base confidence from data availability
        confidence = 0.0

        # Recession probability available
        if self.probability > 0 or self.probability == 0:  # Has data
            confidence += 0.4

        # Yield curve data available
        if self.inversion_days >= 0:  # Has data
            confidence += 0.3

        # GSCPI available
        if self.gscpi_value is not None:
            confidence += 0.2

        # Leading indicators available
        if self.leading_indicators:
            confidence += 0.1

        self.confidence = min(confidence, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": str(self.date),
            "recession": {
                "probability_pct": round(self.probability, 2),
                "phase": self.phase.value,
                "investment_posture": self.investment_posture.value,
            },
            "yield_curve": {
                "is_inverted": self.yield_curve_inverted,
                "inversion_days": self.inversion_days,
            },
            "supply_chain": {
                "gscpi_value": self.gscpi_value,
                "is_stressed": self.gscpi_stress,
            },
            "leading_indicators": self.leading_indicators,
            "confidence": round(self.confidence, 2),
            "interpretation": self._get_interpretation(),
            "warnings": self.warnings,
        }

    def _get_interpretation(self) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            EconomicPhase.EXPANSION: (
                f"Economy in expansion phase with {self.probability:.1f}% recession probability. "
                "Favorable conditions for equity investments. "
                "Consider growth and cyclical sector exposure."
            ),
            EconomicPhase.LATE_CYCLE: (
                f"Late-cycle conditions with {self.probability:.1f}% recession probability. "
                "Elevated caution warranted. Favor quality over growth. "
                "Consider reducing cyclical exposure."
            ),
            EconomicPhase.PRE_RECESSION: (
                f"Pre-recession conditions with {self.probability:.1f}% recession probability. "
                "Defensive positioning recommended. "
                "Favor utilities, staples, healthcare. Increase cash/bonds."
            ),
            EconomicPhase.RECESSION: (
                f"Recessionary conditions with {self.probability:.1f}% recession probability. "
                "Strongly defensive positioning. "
                "Minimize equity exposure, maximize safety."
            ),
            EconomicPhase.RECOVERY: (
                f"Economic recovery phase with {self.probability:.1f}% recession probability. "
                "Opportunistic positioning for growth. "
                "Consider increasing equity, favor cyclicals."
            ),
        }
        return interpretations.get(
            self.phase,
            "Economic conditions uncertain. Maintain balanced positioning."
        )

    @property
    def equity_allocation_range(self) -> tuple:
        """Suggested equity allocation range (min, max) as percentages."""
        ranges = {
            EconomicPhase.EXPANSION: (60, 80),
            EconomicPhase.LATE_CYCLE: (40, 60),
            EconomicPhase.PRE_RECESSION: (25, 45),
            EconomicPhase.RECESSION: (15, 30),
            EconomicPhase.RECOVERY: (50, 75),
            EconomicPhase.UNKNOWN: (40, 60),
        }
        return ranges.get(self.phase, (40, 60))

    @property
    def sector_recommendations(self) -> Dict[str, str]:
        """Sector allocation recommendations."""
        if self.phase in (EconomicPhase.PRE_RECESSION, EconomicPhase.RECESSION):
            return {
                "overweight": "Utilities, Consumer Staples, Healthcare",
                "neutral": "Communication Services, Real Estate",
                "underweight": "Consumer Discretionary, Industrials, Materials, Financials",
            }
        elif self.phase == EconomicPhase.EXPANSION:
            return {
                "overweight": "Technology, Consumer Discretionary, Industrials",
                "neutral": "Financials, Healthcare, Materials",
                "underweight": "Utilities, Consumer Staples",
            }
        elif self.phase == EconomicPhase.LATE_CYCLE:
            return {
                "overweight": "Healthcare, Consumer Staples, Energy",
                "neutral": "Technology, Financials",
                "underweight": "Consumer Discretionary, Materials",
            }
        else:
            return {
                "overweight": "Balanced across sectors",
                "neutral": "All sectors",
                "underweight": "None specifically",
            }


class RecessionIndicator:
    """Service for assessing recession probability and economic phase.

    Combines multiple indicators to provide comprehensive economic
    cycle assessment and investment guidance.

    SOLID: Single Responsibility - only handles recession assessment
    """

    def __init__(self, nyfed_client=None, yield_analyzer=None):
        """Initialize indicator with optional dependencies.

        Args:
            nyfed_client: NYFedDataClient instance
            yield_analyzer: YieldCurveAnalyzer instance
        """
        self._nyfed_client = nyfed_client
        self._yield_analyzer = yield_analyzer

    def _get_nyfed_client(self):
        """Lazy-load NY Fed client."""
        if self._nyfed_client is None:
            from investigator.infrastructure.external.nyfed import get_nyfed_client
            self._nyfed_client = get_nyfed_client()
        return self._nyfed_client

    def _get_yield_analyzer(self):
        """Lazy-load yield curve analyzer."""
        if self._yield_analyzer is None:
            from investigator.domain.services.market_regime.yield_curve_analyzer import (
                get_yield_curve_analyzer
            )
            self._yield_analyzer = get_yield_curve_analyzer()
        return self._yield_analyzer

    async def assess(self) -> RecessionAssessment:
        """Get comprehensive recession assessment.

        Returns:
            RecessionAssessment with probability, phase, and recommendations
        """
        assessment = RecessionAssessment(date=date.today())

        try:
            # Get recession probability from NY Fed
            nyfed = self._get_nyfed_client()
            recession_prob = await nyfed.get_recession_probability()

            if recession_prob:
                assessment.probability = recession_prob.probability
            else:
                assessment.warnings.append("Recession probability data unavailable")

            # Get GSCPI
            gscpi = await nyfed.get_gscpi()
            if gscpi:
                assessment.gscpi_value = gscpi.value
                assessment.gscpi_stress = gscpi.value > 1.0  # > 1 std dev = stressed

            # Get yield curve analysis
            yield_analyzer = self._get_yield_analyzer()
            curve_analysis = await yield_analyzer.analyze()

            if curve_analysis:
                assessment.yield_curve_inverted = curve_analysis.shape.value in ('inverted', 'deeply_inverted')
                assessment.inversion_days = curve_analysis.days_inverted

            # Build leading indicators summary
            assessment.leading_indicators = self._build_leading_indicators(
                recession_prob,
                gscpi,
                curve_analysis
            )

            # Recalculate derived values after setting all inputs
            assessment._classify_phase()
            assessment._set_investment_posture()
            assessment._calculate_confidence()

            return assessment

        except Exception as e:
            logger.error(f"Error in recession assessment: {e}")
            assessment.warnings.append(f"Assessment error: {str(e)}")
            return assessment

    def _build_leading_indicators(
        self,
        recession_prob,
        gscpi,
        curve_analysis
    ) -> Dict[str, str]:
        """Build leading indicators summary."""
        indicators = {}

        # Recession probability indicator
        if recession_prob:
            if recession_prob.probability < 20:
                indicators["recession_probability"] = "low_risk"
            elif recession_prob.probability < 40:
                indicators["recession_probability"] = "moderate_risk"
            else:
                indicators["recession_probability"] = "high_risk"

        # Yield curve indicator
        if curve_analysis:
            indicators["yield_curve"] = curve_analysis.shape.value

        # Supply chain indicator
        if gscpi:
            if gscpi.value < 0:
                indicators["supply_chain"] = "favorable"
            elif gscpi.value < 1.5:
                indicators["supply_chain"] = "normal"
            else:
                indicators["supply_chain"] = "stressed"

        return indicators

    async def get_history(self, months: int = 24) -> List[Dict[str, Any]]:
        """Get historical recession probability.

        Args:
            months: Number of months of history

        Returns:
            List of historical assessments
        """
        try:
            nyfed = self._get_nyfed_client()
            history = await nyfed.get_recession_probability_history(months)
            return history
        except Exception as e:
            logger.error(f"Error fetching recession history: {e}")
            return []

    async def get_market_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive market regime summary.

        Combines yield curve, recession probability, and supply chain
        data into a unified market regime assessment.

        Returns:
            Dictionary with complete market regime analysis
        """
        assessment = await self.assess()
        yield_analyzer = self._get_yield_analyzer()
        curve_analysis = await yield_analyzer.analyze()

        return {
            "date": str(date.today()),
            "economic_phase": assessment.phase.value,
            "investment_posture": assessment.investment_posture.value,
            "recession_assessment": assessment.to_dict(),
            "yield_curve_analysis": curve_analysis.to_dict() if curve_analysis else None,
            "allocation_guidance": {
                "equity_range": assessment.equity_allocation_range,
                "sector_recommendations": assessment.sector_recommendations,
            },
            "risk_metrics": {
                "recession_probability": assessment.probability,
                "yield_curve_inverted": assessment.yield_curve_inverted,
                "supply_chain_stress": assessment.gscpi_stress,
            },
        }


# Singleton instance
_recession_indicator: Optional[RecessionIndicator] = None


def get_recession_indicator() -> RecessionIndicator:
    """Get or create singleton indicator instance."""
    global _recession_indicator
    if _recession_indicator is None:
        _recession_indicator = RecessionIndicator()
    return _recession_indicator
