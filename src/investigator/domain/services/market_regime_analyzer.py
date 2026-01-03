#!/usr/bin/env python3
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

"""Unified Market Regime Analyzer.

Combines yield curve, credit cycle, and recession indicators into a
comprehensive market regime assessment for scheduled data collection.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from investigator.domain.services.market_regime import (
    get_yield_curve_analyzer,
    get_recession_indicator,
    get_credit_cycle_analyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveRegime:
    """Comprehensive market regime assessment."""

    # Overall regime
    regime: str  # e.g., "expansion", "contraction", "transition"

    # Component assessments
    yield_curve_shape: str
    yield_curve_inverted: bool
    credit_cycle_phase: str
    volatility_regime: str
    recession_probability: float

    # Risk signals
    risk_off_signal: bool
    vix_level: Optional[float] = None
    credit_spread: Optional[float] = None

    # Metadata
    snapshot_date: datetime = None
    recommendations: Optional[str] = None

    def __post_init__(self):
        if self.snapshot_date is None:
            self.snapshot_date = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime,
            'yield_curve_shape': self.yield_curve_shape,
            'yield_curve_inverted': self.yield_curve_inverted,
            'credit_cycle_phase': self.credit_cycle_phase,
            'volatility_regime': self.volatility_regime,
            'recession_probability': self.recession_probability,
            'risk_off_signal': self.risk_off_signal,
            'vix_level': self.vix_level,
            'credit_spread': self.credit_spread,
            'snapshot_date': self.snapshot_date.isoformat() if self.snapshot_date else None,
            'recommendations': self.recommendations,
        }


class MarketRegimeAnalyzer:
    """Unified market regime analyzer combining multiple indicators."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._yield_curve_analyzer = None
        self._recession_indicator = None
        self._credit_cycle_analyzer = None

    @property
    def yield_curve_analyzer(self):
        if self._yield_curve_analyzer is None:
            self._yield_curve_analyzer = get_yield_curve_analyzer()
        return self._yield_curve_analyzer

    @property
    def recession_indicator(self):
        if self._recession_indicator is None:
            self._recession_indicator = get_recession_indicator()
        return self._recession_indicator

    @property
    def credit_cycle_analyzer(self):
        if self._credit_cycle_analyzer is None:
            self._credit_cycle_analyzer = get_credit_cycle_analyzer()
        return self._credit_cycle_analyzer

    def get_comprehensive_regime(self) -> Optional[ComprehensiveRegime]:
        """Get comprehensive market regime assessment.

        Combines:
        - Yield curve analysis (shape, inversion)
        - Credit cycle analysis (phase, spreads)
        - Recession probability
        - Volatility regime

        Returns:
            ComprehensiveRegime with full assessment
        """
        try:
            # Default values
            yield_curve_shape = "normal"
            yield_curve_inverted = False
            credit_cycle_phase = "unknown"
            volatility_regime = "normal"
            recession_probability = 0.0
            vix_level = None
            credit_spread = None
            risk_off_signal = False

            # Try to get yield curve analysis
            try:
                yc_analysis = asyncio.run(self.yield_curve_analyzer.analyze())
                if yc_analysis:
                    yield_curve_shape = yc_analysis.shape.value if hasattr(yc_analysis.shape, 'value') else str(yc_analysis.shape)
                    yield_curve_inverted = yc_analysis.is_inverted if hasattr(yc_analysis, 'is_inverted') else False
            except Exception as e:
                self.logger.warning(f"Yield curve analysis failed: {e}")

            # Try to get recession assessment
            try:
                recession = asyncio.run(self.recession_indicator.assess())
                if recession:
                    recession_probability = recession.probability if hasattr(recession, 'probability') else 0.0
            except Exception as e:
                self.logger.warning(f"Recession assessment failed: {e}")

            # Try to get credit cycle analysis
            try:
                cc_analysis = asyncio.run(self.credit_cycle_analyzer.analyze())
                if cc_analysis:
                    credit_cycle_phase = cc_analysis.phase if hasattr(cc_analysis, 'phase') else "unknown"
                    credit_spread = cc_analysis.high_yield_spread if hasattr(cc_analysis, 'high_yield_spread') else None
            except Exception as e:
                self.logger.warning(f"Credit cycle analysis failed: {e}")

            # Determine volatility regime from VIX if available
            try:
                from investigator.infrastructure.external.fred.macro_indicators import (
                    get_macro_indicator_fetcher,
                )
                fetcher = get_macro_indicator_fetcher()
                vix_data = fetcher.get_latest_values(['VIXCLS'])
                if vix_data and 'VIXCLS' in vix_data:
                    vix_level = vix_data['VIXCLS'].get('value')
                    if vix_level:
                        if vix_level > 30:
                            volatility_regime = "high"
                        elif vix_level > 20:
                            volatility_regime = "elevated"
                        else:
                            volatility_regime = "normal"
            except Exception as e:
                self.logger.debug(f"VIX fetch failed: {e}")

            # Determine overall regime
            regime = self._determine_overall_regime(
                yield_curve_inverted=yield_curve_inverted,
                recession_probability=recession_probability,
                volatility_regime=volatility_regime,
                credit_cycle_phase=credit_cycle_phase,
            )

            # Determine risk-off signal
            risk_off_signal = (
                yield_curve_inverted or
                recession_probability > 0.5 or
                volatility_regime == "high" or
                (credit_spread and credit_spread > 500)  # 500 bps
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                regime=regime,
                yield_curve_inverted=yield_curve_inverted,
                recession_probability=recession_probability,
                volatility_regime=volatility_regime,
            )

            return ComprehensiveRegime(
                regime=regime,
                yield_curve_shape=yield_curve_shape,
                yield_curve_inverted=yield_curve_inverted,
                credit_cycle_phase=credit_cycle_phase,
                volatility_regime=volatility_regime,
                recession_probability=recession_probability,
                risk_off_signal=risk_off_signal,
                vix_level=vix_level,
                credit_spread=credit_spread,
                recommendations=recommendations,
            )

        except Exception as e:
            self.logger.error(f"Comprehensive regime analysis failed: {e}")
            return None

    def _determine_overall_regime(
        self,
        yield_curve_inverted: bool,
        recession_probability: float,
        volatility_regime: str,
        credit_cycle_phase: str,
    ) -> str:
        """Determine overall market regime from component indicators."""

        # High recession probability indicates contraction
        if recession_probability > 0.7:
            return "contraction"

        # Inverted yield curve with elevated volatility suggests transition
        if yield_curve_inverted and volatility_regime in ["elevated", "high"]:
            return "late_cycle"

        # Inverted curve alone suggests caution
        if yield_curve_inverted:
            return "late_cycle_warning"

        # High volatility without inversion
        if volatility_regime == "high":
            return "volatility_spike"

        # Credit tightening phase
        if credit_cycle_phase in ["tightening", "distress"]:
            return "credit_tightening"

        # Low probability, normal conditions
        if recession_probability < 0.3 and volatility_regime == "normal":
            return "expansion"

        return "transition"

    def _generate_recommendations(
        self,
        regime: str,
        yield_curve_inverted: bool,
        recession_probability: float,
        volatility_regime: str,
    ) -> str:
        """Generate investment recommendations based on regime."""

        recommendations = []

        if regime == "contraction":
            recommendations.append("Reduce equity exposure")
            recommendations.append("Favor defensive sectors (utilities, consumer staples)")
            recommendations.append("Consider increasing cash allocation")

        elif regime == "late_cycle":
            recommendations.append("Shift to quality stocks with strong balance sheets")
            recommendations.append("Reduce cyclical exposure")
            recommendations.append("Consider duration in fixed income")

        elif regime == "late_cycle_warning":
            recommendations.append("Monitor for further deterioration")
            recommendations.append("Review portfolio risk exposure")
            recommendations.append("Consider hedging strategies")

        elif regime == "volatility_spike":
            recommendations.append("Avoid panic selling")
            recommendations.append("Look for opportunistic entry points")
            recommendations.append("Maintain diversification")

        elif regime == "credit_tightening":
            recommendations.append("Favor companies with low leverage")
            recommendations.append("Avoid high-yield credit exposure")
            recommendations.append("Focus on cash flow positive businesses")

        elif regime == "expansion":
            recommendations.append("Maintain equity exposure")
            recommendations.append("Consider growth-oriented investments")
            recommendations.append("Monitor for signs of overheating")

        else:
            recommendations.append("Stay diversified across asset classes")
            recommendations.append("Monitor key indicators for direction")

        return "; ".join(recommendations)


# Singleton instance
_market_regime_analyzer: Optional[MarketRegimeAnalyzer] = None


def get_market_regime_analyzer() -> MarketRegimeAnalyzer:
    """Get singleton MarketRegimeAnalyzer instance.

    Returns:
        MarketRegimeAnalyzer instance
    """
    global _market_regime_analyzer
    if _market_regime_analyzer is None:
        _market_regime_analyzer = MarketRegimeAnalyzer()
    return _market_regime_analyzer
