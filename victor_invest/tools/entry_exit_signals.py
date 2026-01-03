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

"""Entry/Exit Signal Tool for Victor Invest.

Provides entry/exit signal generation combining:
- Programmatic signal generation from technical indicators
- LLM-based signal validation and enhancement
- Optimal entry zone calculation with position sizing

Uses the EntryExitEngine from investigator.domain.services.signals.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class EntryExitSignalTool(BaseTool):
    """Tool for generating entry/exit signals for stock positions.

    This tool analyzes technical indicators, price data, and valuation
    to generate actionable entry and exit signals with risk management.

    Features:
    - Entry signal detection (oversold, support bounce, MACD crossover, etc.)
    - Exit signal detection (overbought, resistance, stop loss, etc.)
    - Optimal entry zone calculation with timing recommendations
    - Position sizing guidance based on conviction and volatility
    - Signal confidence scoring

    Actions:
        generate_signals: Generate all entry/exit signals for a symbol
        get_entry_signals: Get only entry signals
        get_exit_signals: Get only exit signals
        get_entry_zone: Calculate optimal entry zone
        score_signal: Score a specific signal

    Example:
        tool = EntryExitSignalTool()
        result = await tool.execute(
            symbol="AAPL",
            action="generate_signals",
            current_price=175.50,
            fair_value=195.00
        )
    """

    name = "entry_exit_signals"
    description = """Generate entry/exit signals for stock positions including:
    - Entry signals (oversold reversal, support bounce, MACD crossover, golden cross, breakout)
    - Exit signals (overbought, resistance rejection, stop loss, target hit)
    - Optimal entry zone with timing and scaling recommendations
    - Position sizing guidance based on conviction and volatility"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize the entry/exit signal tool."""
        super().__init__(config)
        self._engine = None
        self._integrator = None

    async def initialize(self) -> None:
        """Initialize the signal engine and integrator."""
        try:
            from investigator.domain.services.signals import (
                get_entry_exit_engine,
                get_signal_integrator,
            )
            self._engine = get_entry_exit_engine()
            self._integrator = get_signal_integrator()
            self._initialized = True
            logger.info("EntryExitSignalTool initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not import signal services: {e}")
            self._initialized = True  # Allow fallback behavior

    async def execute(
        self,
        action: str = "generate_signals",
        symbol: str = "",
        current_price: float = 0.0,
        fair_value: Optional[float] = None,
        price_data: Optional[Dict] = None,
        indicators: Optional[Dict] = None,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
        volatility: float = 0.25,
        atr: Optional[float] = None,
        llm_analysis: Optional[Dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute entry/exit signal analysis.

        Args:
            action: Action to perform:
                - "generate_signals": Generate all entry/exit signals
                - "get_entry_signals": Get entry signals only
                - "get_exit_signals": Get exit signals only
                - "get_entry_zone": Calculate optimal entry zone
                - "integrate_signals": Integrate programmatic + LLM signals
            symbol: Stock symbol (for logging/context)
            current_price: Current stock price
            fair_value: Estimated fair value (optional)
            price_data: Dict with OHLCV data (or DataFrame)
            indicators: Technical indicators dict
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
            volatility: 20-day volatility (default 0.25)
            atr: Average True Range (optional)
            llm_analysis: LLM technical analysis output (for integration)

        Returns:
            ToolResult with signals data
        """
        await self.ensure_initialized()

        try:
            if action == "generate_signals":
                return await self._generate_all_signals(
                    symbol=symbol,
                    current_price=current_price,
                    fair_value=fair_value,
                    price_data=price_data,
                    indicators=indicators or {},
                    support_levels=support_levels or [],
                    resistance_levels=resistance_levels or [],
                    volatility=volatility,
                    atr=atr,
                )
            elif action == "get_entry_signals":
                return await self._get_entry_signals(
                    current_price=current_price,
                    fair_value=fair_value,
                    price_data=price_data,
                    indicators=indicators or {},
                    support_levels=support_levels or [],
                    resistance_levels=resistance_levels or [],
                )
            elif action == "get_exit_signals":
                return await self._get_exit_signals(
                    current_price=current_price,
                    price_data=price_data,
                    indicators=indicators or {},
                )
            elif action == "get_entry_zone":
                return await self._get_entry_zone(
                    current_price=current_price,
                    fair_value=fair_value,
                    support_levels=support_levels or [],
                    resistance_levels=resistance_levels or [],
                    volatility=volatility,
                    atr=atr,
                )
            elif action == "integrate_signals":
                return await self._integrate_signals(
                    price_data=price_data,
                    indicators=indicators or {},
                    valuation={"blended_fair_value": fair_value} if fair_value else {},
                    llm_analysis=llm_analysis,
                )
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: generate_signals, "
                    "get_entry_signals, get_exit_signals, get_entry_zone, integrate_signals"
                )
        except Exception as e:
            logger.error(f"Error in EntryExitSignalTool: {e}")
            return ToolResult.error_result(str(e))

    async def _generate_all_signals(
        self,
        symbol: str,
        current_price: float,
        fair_value: Optional[float],
        price_data: Optional[Dict],
        indicators: Dict,
        support_levels: List[float],
        resistance_levels: List[float],
        volatility: float,
        atr: Optional[float],
    ) -> ToolResult:
        """Generate all entry/exit signals."""
        if not self._engine:
            return self._fallback_signals(current_price, fair_value, support_levels, resistance_levels)

        # Convert price_data to DataFrame if needed
        df = self._to_dataframe(price_data, current_price)

        # Build support/resistance dict
        sr = {
            "immediate_support": support_levels[0] if support_levels else current_price * 0.95,
            "immediate_resistance": resistance_levels[0] if resistance_levels else current_price * 1.05,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
        }

        # Generate entry signals
        entry_signals = self._engine.generate_entry_signals(
            price_data=df,
            indicators=indicators,
            valuation={"blended_fair_value": fair_value} if fair_value else {},
            support_resistance=sr,
        )

        # Generate exit signals
        exit_signals = self._engine.generate_exit_signals(
            price_data=df,
            indicators=indicators,
            position_info=None,
        )

        # Calculate entry zone
        effective_atr = atr if atr else current_price * 0.02
        entry_zone = self._engine.calculate_optimal_entry_zone(
            current_price=current_price,
            fair_value=fair_value or current_price,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            volatility=volatility,
            atr=effective_atr,
        )

        # Convert to serializable format
        entry_signals_data = [self._signal_to_dict(s) for s in entry_signals[:5]]
        exit_signals_data = [self._exit_signal_to_dict(s) for s in exit_signals[:5]]
        entry_zone_data = self._entry_zone_to_dict(entry_zone) if entry_zone else None

        return ToolResult.success_result(
            data={
                "symbol": symbol,
                "current_price": current_price,
                "fair_value": fair_value,
                "entry_signals": entry_signals_data,
                "exit_signals": exit_signals_data,
                "optimal_entry_zone": entry_zone_data,
                "total_entry_signals": len(entry_signals),
                "total_exit_signals": len(exit_signals),
            },
            metadata={
                "tool": "entry_exit_signals",
                "action": "generate_signals",
            }
        )

    async def _get_entry_signals(
        self,
        current_price: float,
        fair_value: Optional[float],
        price_data: Optional[Dict],
        indicators: Dict,
        support_levels: List[float],
        resistance_levels: List[float],
    ) -> ToolResult:
        """Get entry signals only."""
        if not self._engine:
            return self._fallback_signals(current_price, fair_value, support_levels, resistance_levels)

        df = self._to_dataframe(price_data, current_price)
        sr = {
            "immediate_support": support_levels[0] if support_levels else current_price * 0.95,
            "immediate_resistance": resistance_levels[0] if resistance_levels else current_price * 1.05,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
        }

        entry_signals = self._engine.generate_entry_signals(
            price_data=df,
            indicators=indicators,
            valuation={"blended_fair_value": fair_value} if fair_value else {},
            support_resistance=sr,
        )

        return ToolResult.success_result(
            data={
                "entry_signals": [self._signal_to_dict(s) for s in entry_signals[:5]],
                "total_count": len(entry_signals),
            }
        )

    async def _get_exit_signals(
        self,
        current_price: float,
        price_data: Optional[Dict],
        indicators: Dict,
    ) -> ToolResult:
        """Get exit signals only."""
        if not self._engine:
            return ToolResult.success_result(data={"exit_signals": [], "total_count": 0})

        df = self._to_dataframe(price_data, current_price)

        exit_signals = self._engine.generate_exit_signals(
            price_data=df,
            indicators=indicators,
            position_info=None,
        )

        return ToolResult.success_result(
            data={
                "exit_signals": [self._exit_signal_to_dict(s) for s in exit_signals[:5]],
                "total_count": len(exit_signals),
            }
        )

    async def _get_entry_zone(
        self,
        current_price: float,
        fair_value: Optional[float],
        support_levels: List[float],
        resistance_levels: List[float],
        volatility: float,
        atr: Optional[float],
    ) -> ToolResult:
        """Calculate optimal entry zone."""
        if not self._engine:
            # Fallback calculation
            lower = current_price * 0.97
            upper = current_price * 1.02
            return ToolResult.success_result(
                data={
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "ideal_entry": round(current_price * 0.98, 2),
                    "timing": "WAIT_PULLBACK",
                    "scaling_strategy": "SCALE_IN_THIRDS",
                    "confidence": "MEDIUM",
                }
            )

        effective_atr = atr if atr else current_price * 0.02
        entry_zone = self._engine.calculate_optimal_entry_zone(
            current_price=current_price,
            fair_value=fair_value or current_price,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            volatility=volatility,
            atr=effective_atr,
        )

        return ToolResult.success_result(
            data=self._entry_zone_to_dict(entry_zone) if entry_zone else {}
        )

    async def _integrate_signals(
        self,
        price_data: Optional[Dict],
        indicators: Dict,
        valuation: Dict,
        llm_analysis: Optional[Dict],
    ) -> ToolResult:
        """Integrate programmatic and LLM signals."""
        if not self._integrator:
            return ToolResult.error_result("Signal integrator not available")

        df = self._to_dataframe(price_data, 0)
        if df.empty:
            return ToolResult.error_result("Price data required for signal integration")

        integrated = self._integrator.integrate_signals(
            price_data=df,
            indicators=indicators,
            valuation=valuation,
            llm_technical_analysis=llm_analysis,
        )

        report_data = self._integrator.to_report_format(integrated)
        report_data["signal_agreement_score"] = integrated.signal_agreement_score
        report_data["confidence_boost"] = integrated.confidence_boost

        return ToolResult.success_result(data=report_data)

    def _to_dataframe(self, price_data: Optional[Dict], current_price: float) -> pd.DataFrame:
        """Convert price data to DataFrame."""
        if price_data is None:
            # Create minimal DataFrame with current price
            return pd.DataFrame({"close": [current_price], "Close": [current_price]})

        if isinstance(price_data, pd.DataFrame):
            return price_data

        if isinstance(price_data, dict):
            return pd.DataFrame(price_data)

        return pd.DataFrame({"close": [current_price], "Close": [current_price]})

    def _signal_to_dict(self, signal) -> Dict[str, Any]:
        """Convert EntrySignal to dict."""
        return {
            "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
            "price_level": round(signal.price_level, 2),
            "confidence": signal.confidence.value if hasattr(signal.confidence, 'value') else str(signal.confidence),
            "rationale": signal.rationale,
            "risk_reward_ratio": round(signal.risk_reward_ratio, 2),
            "stop_loss": round(signal.stop_loss, 2),
            "stop_loss_pct": round(signal.stop_loss_pct, 2),
            "target_price": round(signal.target_price, 2),
            "target_pct": round(signal.target_pct, 2),
            "expected_holding_days": signal.expected_holding_days,
            "volume_confirmation": signal.volume_confirmation,
            "trend_alignment": signal.trend_alignment,
        }

    def _exit_signal_to_dict(self, signal) -> Dict[str, Any]:
        """Convert ExitSignal to dict."""
        return {
            "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
            "price_level": round(signal.price_level, 2),
            "confidence": signal.confidence.value if hasattr(signal.confidence, 'value') else str(signal.confidence),
            "rationale": signal.rationale,
            "urgency": signal.urgency,
            "partial_exit_pct": round(signal.partial_exit_pct, 1),
        }

    def _entry_zone_to_dict(self, zone) -> Dict[str, Any]:
        """Convert OptimalEntryZone to dict."""
        return {
            "lower_bound": round(zone.lower_bound, 2),
            "upper_bound": round(zone.upper_bound, 2),
            "ideal_entry": round(zone.ideal_entry, 2),
            "timing": zone.timing.value if hasattr(zone.timing, 'value') else str(zone.timing),
            "scaling_strategy": zone.scaling_strategy.value if hasattr(zone.scaling_strategy, 'value') else str(zone.scaling_strategy),
            "confidence": zone.confidence.value if hasattr(zone.confidence, 'value') else str(zone.confidence),
            "rationale": zone.rationale,
            "recommended_allocation_pct": round(zone.recommended_allocation_pct, 1),
            "max_position_size_pct": round(zone.max_position_size_pct, 1),
        }

    def _fallback_signals(
        self,
        current_price: float,
        fair_value: Optional[float],
        support_levels: List[float],
        resistance_levels: List[float],
    ) -> ToolResult:
        """Provide fallback signals when engine not available."""
        fv = fair_value or current_price
        upside = (fv - current_price) / current_price if current_price > 0 else 0

        entry_signals = []
        if upside > 0.10:
            entry_signals.append({
                "signal_type": "VALUATION_BASED",
                "price_level": round(current_price, 2),
                "confidence": "MEDIUM",
                "rationale": f"Trading at {abs(upside)*100:.1f}% discount to fair value",
                "risk_reward_ratio": round(upside / 0.05, 2),  # Assume 5% stop loss
                "stop_loss": round(current_price * 0.95, 2),
                "stop_loss_pct": 5.0,
                "target_price": round(fv, 2),
                "target_pct": round(upside * 100, 1),
                "expected_holding_days": 90,
                "volume_confirmation": False,
                "trend_alignment": False,
            })

        return ToolResult.success_result(
            data={
                "entry_signals": entry_signals,
                "exit_signals": [],
                "optimal_entry_zone": {
                    "lower_bound": round(current_price * 0.97, 2),
                    "upper_bound": round(current_price * 1.02, 2),
                    "ideal_entry": round(current_price * 0.98, 2),
                    "timing": "WAIT_PULLBACK",
                    "scaling_strategy": "SCALE_IN_THIRDS",
                    "confidence": "LOW",
                    "rationale": "Fallback calculation - engine not available",
                    "recommended_allocation_pct": 3.0,
                    "max_position_size_pct": 5.0,
                },
                "total_entry_signals": len(entry_signals),
                "total_exit_signals": 0,
            },
            warnings=["Using fallback signal generation - full engine not available"],
        )

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["generate_signals", "get_entry_signals", "get_exit_signals", "get_entry_zone", "integrate_signals"],
                    "description": "Action to perform",
                    "default": "generate_signals",
                },
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL)",
                },
                "current_price": {
                    "type": "number",
                    "description": "Current stock price",
                },
                "fair_value": {
                    "type": "number",
                    "description": "Estimated fair value from valuation models",
                },
                "support_levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Support price levels",
                },
                "resistance_levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Resistance price levels",
                },
                "volatility": {
                    "type": "number",
                    "description": "20-day volatility (decimal, e.g., 0.25 for 25%)",
                    "default": 0.25,
                },
                "atr": {
                    "type": "number",
                    "description": "Average True Range",
                },
            },
            "required": ["current_price"],
        }
