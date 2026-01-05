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

"""Market Regime Tool for Victor Invest.

This tool provides comprehensive market regime analysis combining:
- Yield curve analysis
- Credit cycle assessment
- Recession probability
- Volatility regime
- Investment recommendations

Available Actions:
- summary: Complete market regime summary
- yield_curve: Yield curve analysis only
- credit_cycle: Credit cycle phase analysis
- recession: Recession probability assessment
- volatility: Volatility regime analysis
- recommendations: Sector and allocation recommendations

Example:
    tool = MarketRegimeTool()

    # Get comprehensive summary
    result = await tool.execute(action="summary")

    # Get credit cycle analysis
    result = await tool.execute(action="credit_cycle")

    # Get sector recommendations
    result = await tool.execute(action="recommendations")
"""

import logging
from typing import Any, Dict, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MarketRegimeTool(BaseTool):
    """Tool for comprehensive market regime analysis.

    Provides CLI and agent access to market regime indicators including
    yield curve, credit cycle, recession probability, and investment
    recommendations based on current conditions.

    Supported actions:
    - summary: Complete market regime summary
    - yield_curve: Yield curve shape and signals
    - credit_cycle: Credit cycle phase analysis
    - recession: Recession probability assessment
    - volatility: Volatility regime classification
    - recommendations: Sector and allocation recommendations

    Attributes:
        name: "market_regime"
        description: Tool description for agent discovery
    """

    name = "market_regime"
    description = """Comprehensive market regime analysis and investment recommendations.

Actions:
- summary: Complete market regime summary (all indicators combined)
- yield_curve: Yield curve shape and investment signals
- credit_cycle: Credit cycle phase (early expansion to credit crisis)
- recession: Recession probability assessment
- volatility: Volatility regime classification
- recommendations: Sector allocation recommendations based on regime

Returns:
- Current market regime classification
- Credit cycle phase with confidence score
- Recession probability estimate
- Volatility regime and VIX level
- Investment signal (risk-on to strongly defensive)
- Sector overweight/underweight recommendations
- Valuation adjustment factors

Investment Signals by Regime:
- Early Expansion: Risk-on, favor cyclicals, high-yield
- Mid-Cycle: Balanced, quality growth, investment grade
- Late Cycle: Reduce risk, favor quality, defensive rotation
- Credit Stress: Defensive, reduce high-yield, increase cash
- Credit Crisis: Maximum defensive, treasuries, capital preservation
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Market Regime Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._yield_curve_analyzer = None
        self._credit_cycle_analyzer = None
        self._recession_indicator = None

    async def initialize(self) -> None:
        """Initialize market regime analyzers."""
        try:
            from investigator.domain.services.market_regime import (
                get_yield_curve_analyzer,
                get_credit_cycle_analyzer,
                get_recession_indicator,
            )

            self._yield_curve_analyzer = get_yield_curve_analyzer()
            self._credit_cycle_analyzer = get_credit_cycle_analyzer()
            self._recession_indicator = get_recession_indicator()

            self._initialized = True
            logger.info("MarketRegimeTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MarketRegimeTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Dict[str, Any],
        action: str = "summary",
        **kwargs
    ) -> ToolResult:
        """Execute market regime query.

        Args:
            action: Query type:
                - "summary": Complete market regime summary
                - "yield_curve": Yield curve analysis
                - "credit_cycle": Credit cycle analysis
                - "recession": Recession assessment
                - "volatility": Volatility regime
                - "recommendations": Investment recommendations
            **kwargs: Additional parameters

        Returns:
            ToolResult with market regime data
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "summary":
                return await self._get_summary()

            elif action == "yield_curve":
                return await self._get_yield_curve()

            elif action == "credit_cycle":
                return await self._get_credit_cycle()

            elif action == "recession":
                return await self._get_recession()

            elif action == "volatility":
                return await self._get_volatility()

            elif action == "recommendations":
                return await self._get_recommendations()

            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "summary, yield_curve, credit_cycle, recession, volatility, recommendations"
                )

        except Exception as e:
            logger.error(f"MarketRegimeTool execute error: {e}")
            return ToolResult.error_result(
                f"Market regime query failed: {str(e)}",
                metadata={"action": action}
            )

    async def _get_summary(self) -> ToolResult:
        """Get comprehensive market regime summary."""
        # Get all analyses
        yc_analysis = await self._yield_curve_analyzer.analyze()
        cc_analysis = await self._credit_cycle_analyzer.analyze()
        recession_assessment = await self._recession_indicator.assess()

        # Combine into comprehensive summary
        summary = {
            "date": str(cc_analysis.date),
            "overall_regime": {
                "credit_cycle_phase": cc_analysis.phase.value,
                "yield_curve_shape": yc_analysis.shape.value,
                "investment_signal": yc_analysis.investment_signal.value,
                "confidence": cc_analysis.confidence,
            },
            "indicators": {
                "baa_credit_spread_bps": cc_analysis.baa_spread_bps,
                "vix_level": cc_analysis.vix_level,
                "fed_funds_rate": cc_analysis.fed_funds_rate,
                "yield_10y_2y_spread_bps": yc_analysis.spread_10y_2y_bps,
            },
            "classifications": {
                "credit_cycle": cc_analysis.phase.value,
                "volatility_regime": cc_analysis.volatility_regime.value,
                "fed_policy_stance": cc_analysis.fed_policy_stance.value,
                "recession_probability": cc_analysis.recession_probability.value,
            },
            "valuation_impacts": {
                "risk_free_rate": yc_analysis.risk_free_rate,
                "wacc_spread_adjustment_bps": yc_analysis.wacc_spread_adjustment,
                "equity_allocation_adjustment": yc_analysis.equity_adjustment,
            },
            "interpretation": cc_analysis.interpretation,
            "sector_recommendations": cc_analysis.sector_recommendations,
            "factors": cc_analysis.factors,
        }

        # Determine overall signal
        overall_signal = self._derive_overall_signal(yc_analysis, cc_analysis)
        summary["overall_signal"] = overall_signal

        warnings = yc_analysis.warnings + cc_analysis.warnings

        return ToolResult.success_result(
            data=summary,
            warnings=warnings if warnings else None,
            metadata={
                "source": "market_regime_services",
                "credit_cycle": cc_analysis.phase.value,
                "signal": overall_signal["level"],
            }
        )

    async def _get_yield_curve(self) -> ToolResult:
        """Get yield curve analysis."""
        analysis = await self._yield_curve_analyzer.analyze()

        return ToolResult.success_result(
            data=analysis.to_dict(),
            warnings=analysis.warnings if analysis.warnings else None,
            metadata={
                "source": "treasury_yield_curve",
                "shape": analysis.shape.value,
            }
        )

    async def _get_credit_cycle(self) -> ToolResult:
        """Get credit cycle analysis."""
        analysis = await self._credit_cycle_analyzer.analyze()

        return ToolResult.success_result(
            data=analysis.to_dict(),
            warnings=analysis.warnings if analysis.warnings else None,
            metadata={
                "source": "credit_cycle_analyzer",
                "phase": analysis.phase.value,
                "confidence": analysis.confidence,
            }
        )

    async def _get_recession(self) -> ToolResult:
        """Get recession probability assessment."""
        assessment = await self._recession_indicator.assess()

        return ToolResult.success_result(
            data=assessment.to_dict(),
            warnings=assessment.warnings if assessment.warnings else None,
            metadata={
                "source": "recession_indicator",
                "phase": assessment.phase.value,
                "probability": assessment.probability_pct,
            }
        )

    async def _get_volatility(self) -> ToolResult:
        """Get volatility regime analysis."""
        cc_analysis = await self._credit_cycle_analyzer.analyze()

        return ToolResult.success_result(
            data={
                "date": str(cc_analysis.date),
                "vix_level": cc_analysis.vix_level,
                "volatility_regime": cc_analysis.volatility_regime.value,
                "interpretation": self._get_volatility_interpretation(cc_analysis.volatility_regime),
            },
            metadata={
                "source": "vix_analysis",
                "regime": cc_analysis.volatility_regime.value,
            }
        )

    async def _get_recommendations(self) -> ToolResult:
        """Get investment recommendations based on regime."""
        cc_analysis = await self._credit_cycle_analyzer.analyze()
        yc_analysis = await self._yield_curve_analyzer.analyze()

        recommendations = {
            "date": str(cc_analysis.date),
            "credit_cycle_phase": cc_analysis.phase.value,
            "investment_signal": yc_analysis.investment_signal.value,
            "sector_recommendations": cc_analysis.sector_recommendations,
            "allocation_guidance": {
                "equity_adjustment": f"{yc_analysis.equity_adjustment * 100:+.0f}%",
                "risk_posture": self._get_risk_posture(cc_analysis.phase),
                "duration_guidance": self._get_duration_guidance(yc_analysis.shape),
            },
            "interpretation": cc_analysis.interpretation,
        }

        return ToolResult.success_result(
            data=recommendations,
            metadata={
                "source": "market_regime_services",
                "phase": cc_analysis.phase.value,
            }
        )

    def _derive_overall_signal(self, yc_analysis, cc_analysis) -> Dict[str, Any]:
        """Derive overall investment signal from all analyses."""
        from investigator.domain.models.market_context import CreditCyclePhase

        # Combine signals
        credit_phase = cc_analysis.phase
        yc_signal = yc_analysis.investment_signal.value

        # Map to overall signal
        if credit_phase == CreditCyclePhase.CREDIT_CRISIS:
            level = "maximum_defensive"
            description = "Crisis conditions - capital preservation paramount"
        elif credit_phase == CreditCyclePhase.CREDIT_STRESS:
            level = "strongly_defensive"
            description = "Elevated stress - defensive positioning required"
        elif credit_phase == CreditCyclePhase.LATE_CYCLE:
            level = "cautious"
            description = "Late cycle - favor quality, reduce risk"
        elif credit_phase == CreditCyclePhase.EARLY_EXPANSION:
            level = "risk_on"
            description = "Early expansion - favor cyclicals and risk assets"
        else:  # MID_CYCLE
            level = "balanced"
            description = "Mid-cycle - balanced approach appropriate"

        return {
            "level": level,
            "description": description,
            "credit_cycle": credit_phase.value,
            "yield_curve_signal": yc_signal,
        }

    def _get_volatility_interpretation(self, regime) -> str:
        """Get interpretation for volatility regime."""
        from investigator.domain.models.market_context import VolatilityRegime

        interpretations = {
            VolatilityRegime.VERY_LOW: "Very low volatility - market complacency, potential for sharp moves",
            VolatilityRegime.LOW: "Low volatility - calm markets, favorable for position building",
            VolatilityRegime.NORMAL: "Normal volatility - typical market conditions",
            VolatilityRegime.ELEVATED: "Elevated volatility - increased uncertainty, wider stops advised",
            VolatilityRegime.HIGH: "High volatility - significant stress, reduce position sizes",
            VolatilityRegime.EXTREME: "Extreme volatility - crisis conditions, capital preservation",
        }
        return interpretations.get(regime, "Volatility regime undetermined")

    def _get_risk_posture(self, phase) -> str:
        """Get recommended risk posture for credit cycle phase."""
        from investigator.domain.models.market_context import CreditCyclePhase

        postures = {
            CreditCyclePhase.EARLY_EXPANSION: "Aggressive - favor risk assets",
            CreditCyclePhase.MID_CYCLE: "Balanced - moderate risk exposure",
            CreditCyclePhase.LATE_CYCLE: "Conservative - reduce risk, favor quality",
            CreditCyclePhase.CREDIT_STRESS: "Defensive - minimize risk exposure",
            CreditCyclePhase.CREDIT_CRISIS: "Maximum defensive - capital preservation",
        }
        return postures.get(phase, "Balanced")

    def _get_duration_guidance(self, shape) -> str:
        """Get fixed income duration guidance."""
        from investigator.domain.services.market_regime.yield_curve_analyzer import YieldCurveShape

        guidance = {
            YieldCurveShape.STEEP: "Extend duration - rates likely to fall",
            YieldCurveShape.NORMAL: "Neutral duration - balanced positioning",
            YieldCurveShape.FLAT: "Reduce duration - curve may steepen",
            YieldCurveShape.INVERTED: "Short duration - recession risk elevated",
            YieldCurveShape.DEEPLY_INVERTED: "Ultra-short duration - high recession risk",
        }
        return guidance.get(shape, "Neutral duration")

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Market Regime Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "yield_curve", "credit_cycle", "recession", "volatility", "recommendations"],
                    "description": "Type of market regime query",
                    "default": "summary"
                }
            },
            "required": []
        }
