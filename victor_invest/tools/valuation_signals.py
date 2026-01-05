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

"""Valuation Signals Tool for Victor Invest.

This tool provides integrated valuation signal analysis combining:
- Credit risk (Altman Z, Beneish M, Piotroski F) → Valuation discounts
- Insider sentiment → Confidence adjustments
- Short interest → Contrarian signals
- Market regime → WACC adjustments

Available Actions:
- integrate: Full signal integration with adjusted fair value
- credit_risk: Credit risk signal analysis only
- insider: Insider sentiment signal only
- short_interest: Short interest signal only
- market_regime: Market regime adjustment only

Example:
    tool = ValuationSignalsTool()

    # Get fully integrated signals
    result = await tool.execute(
        action="integrate",
        symbol="AAPL",
        base_fair_value=190.0,
        current_price=185.0
    )

    # Get credit risk signal only
    result = await tool.execute(
        action="credit_risk",
        symbol="AAPL"
    )
"""

import logging
from typing import Any, Dict, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ValuationSignalsTool(BaseTool):
    """Tool for integrated valuation signal analysis.

    Provides CLI and agent access to valuation signal integration,
    combining credit risk, insider sentiment, short interest, and
    market regime data into valuation adjustments.

    Supported actions:
    - integrate: Full signal integration with adjusted fair value
    - credit_risk: Credit risk signal analysis
    - insider: Insider sentiment signal
    - short_interest: Short interest signal
    - market_regime: Market regime adjustment

    Attributes:
        name: "valuation_signals"
        description: Tool description for agent discovery
    """

    name = "valuation_signals"
    description = """Integrated valuation signal analysis for fair value adjustments.

Actions:
- integrate: Full signal integration (credit risk, insider, short, regime)
- credit_risk: Credit risk analysis (Altman Z, Beneish M, Piotroski F)
- insider: Insider sentiment signal
- short_interest: Short interest signal
- market_regime: Market regime adjustment

Parameters:
- symbol: Stock symbol (required for most actions)
- base_fair_value: Base fair value from models (for integrate action)
- current_price: Current stock price (for integrate action)

Returns:
- Adjusted fair value with signal-based adjustments
- Credit risk discount (5-50% based on distress tier)
- Insider sentiment confidence adjustment
- Short squeeze/contrarian signals
- Market regime WACC adjustments
- Combined warnings and factors

Signal Integration:
1. Credit Risk → Valuation discount (5-50%)
2. Insider Sentiment → Confidence adjustment (+/-10%)
3. Short Interest → Contrarian signal / risk flag
4. Market Regime → WACC adjustment, valuation factor
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Valuation Signals Tool.

        Args:
            config: Optional investigator config object.
        """
        super().__init__(config)
        self._signal_integrator = None
        self._credit_risk_tool = None
        self._insider_tool = None
        self._short_interest_tool = None
        self._market_regime_tool = None

    async def initialize(self) -> None:
        """Initialize valuation signal services."""
        try:
            from investigator.domain.services.valuation.signal_integrator import (
                get_signal_integrator,
            )

            self._signal_integrator = get_signal_integrator()
            self._initialized = True
            logger.info("ValuationSignalsTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ValuationSignalsTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Dict[str, Any],
        action: str = "integrate",
        symbol: Optional[str] = None,
        base_fair_value: Optional[float] = None,
        current_price: Optional[float] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute valuation signal query.

        Args:
            action: Query type:
                - "integrate": Full signal integration
                - "credit_risk": Credit risk signal only
                - "insider": Insider sentiment only
                - "short_interest": Short interest only
                - "market_regime": Market regime only
            symbol: Stock symbol
            base_fair_value: Base fair value (for integrate)
            current_price: Current price (for integrate)
            **kwargs: Additional parameters

        Returns:
            ToolResult with signal data
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "integrate":
                return await self._integrate_signals(symbol, base_fair_value, current_price, **kwargs)

            elif action == "credit_risk":
                return await self._get_credit_risk_signal(symbol, **kwargs)

            elif action == "insider":
                return await self._get_insider_signal(symbol, **kwargs)

            elif action == "short_interest":
                return await self._get_short_interest_signal(symbol, **kwargs)

            elif action == "market_regime":
                return await self._get_market_regime_adjustment(**kwargs)

            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "integrate, credit_risk, insider, short_interest, market_regime"
                )

        except Exception as e:
            logger.error(f"ValuationSignalsTool execute error: {e}")
            return ToolResult.error_result(
                f"Valuation signal query failed: {str(e)}", metadata={"action": action, "symbol": symbol}
            )

    async def _integrate_signals(
        self, symbol: Optional[str], base_fair_value: Optional[float], current_price: Optional[float], **kwargs
    ) -> ToolResult:
        """Integrate all signals for adjusted fair value."""
        if not symbol:
            return ToolResult.error_result("Symbol is required for integration")
        if base_fair_value is None:
            return ToolResult.error_result("base_fair_value is required for integration")
        if current_price is None:
            return ToolResult.error_result("current_price is required for integration")

        # Collect data from all sources
        credit_risk_data = await self._fetch_credit_risk_data(symbol)
        insider_data = await self._fetch_insider_data(symbol)
        short_interest_data = await self._fetch_short_interest_data(symbol)
        market_regime_data = await self._fetch_market_regime_data()

        # Integrate signals
        result = self._signal_integrator.integrate_signals(
            symbol=symbol,
            base_fair_value=base_fair_value,
            current_price=current_price,
            credit_risk_data=credit_risk_data,
            insider_data=insider_data,
            short_interest_data=short_interest_data,
            market_regime_data=market_regime_data,
        )

        return ToolResult.success_result(
            data=result.to_dict(),
            warnings=result.warnings if result.warnings else None,
            metadata={
                "source": "valuation_signal_integrator",
                "symbol": symbol,
                "adjustment_pct": result.total_adjustment_pct,
            },
        )

    async def _get_credit_risk_signal(self, symbol: Optional[str], **kwargs) -> ToolResult:
        """Get credit risk signal only."""
        if not symbol:
            return ToolResult.error_result("Symbol is required for credit risk signal")

        # Use provided data or fetch
        altman_zscore = kwargs.get("altman_zscore")
        beneish_mscore = kwargs.get("beneish_mscore")
        piotroski_fscore = kwargs.get("piotroski_fscore")

        # If no data provided, try to fetch
        if altman_zscore is None and beneish_mscore is None and piotroski_fscore is None:
            credit_data = await self._fetch_credit_risk_data(symbol)
            if credit_data:
                altman_zscore = credit_data.get("altman_zscore")
                beneish_mscore = credit_data.get("beneish_mscore")
                piotroski_fscore = credit_data.get("piotroski_fscore")

        signal = self._signal_integrator.calculate_credit_risk_signal(
            altman_zscore=altman_zscore,
            beneish_mscore=beneish_mscore,
            piotroski_fscore=piotroski_fscore,
        )

        warnings = []
        if signal.manipulation_flag:
            warnings.append("Earnings manipulation risk detected")
        if signal.distress_tier.value in ["distressed", "severe_distress"]:
            warnings.append(f"Company in {signal.distress_tier.value} tier")

        return ToolResult.success_result(
            data=signal.to_dict(),
            warnings=warnings if warnings else None,
            metadata={
                "source": "credit_risk_signal",
                "symbol": symbol,
                "distress_tier": signal.distress_tier.value,
            },
        )

    async def _get_insider_signal(self, symbol: Optional[str], **kwargs) -> ToolResult:
        """Get insider sentiment signal only."""
        if not symbol:
            return ToolResult.error_result("Symbol is required for insider signal")

        # Use provided data or fetch
        buy_sell_ratio = kwargs.get("buy_sell_ratio")
        sentiment_score = kwargs.get("sentiment_score")
        cluster_detected = kwargs.get("cluster_detected", False)
        net_shares_change = kwargs.get("net_shares_change")

        # If no data provided, try to fetch
        if buy_sell_ratio is None and sentiment_score is None:
            insider_data = await self._fetch_insider_data(symbol)
            if insider_data:
                buy_sell_ratio = insider_data.get("buy_sell_ratio")
                sentiment_score = insider_data.get("sentiment_score")
                cluster_detected = insider_data.get("cluster_detected", False)
                net_shares_change = insider_data.get("net_shares_change")

        signal = self._signal_integrator.calculate_insider_sentiment_signal(
            buy_sell_ratio=buy_sell_ratio,
            net_shares_change=net_shares_change,
            cluster_detected=cluster_detected,
            sentiment_score=sentiment_score,
        )

        return ToolResult.success_result(
            data=signal.to_dict(),
            metadata={
                "source": "insider_sentiment_signal",
                "symbol": symbol,
                "signal": signal.signal.value,
            },
        )

    async def _get_short_interest_signal(self, symbol: Optional[str], **kwargs) -> ToolResult:
        """Get short interest signal only."""
        if not symbol:
            return ToolResult.error_result("Symbol is required for short interest signal")

        # Use provided data or fetch
        short_percent_float = kwargs.get("short_percent_float")
        days_to_cover = kwargs.get("days_to_cover")
        squeeze_score = kwargs.get("squeeze_score")

        # If no data provided, try to fetch
        if short_percent_float is None and squeeze_score is None:
            short_data = await self._fetch_short_interest_data(symbol)
            if short_data:
                short_percent_float = short_data.get("short_percent_float")
                days_to_cover = short_data.get("days_to_cover")
                squeeze_score = short_data.get("squeeze_score")

        signal = self._signal_integrator.calculate_short_interest_signal(
            short_percent_float=short_percent_float,
            days_to_cover=days_to_cover,
            squeeze_score=squeeze_score,
        )

        warnings = []
        if signal.warning_flag:
            warnings.append(signal.interpretation)

        return ToolResult.success_result(
            data=signal.to_dict(),
            warnings=warnings if warnings else None,
            metadata={
                "source": "short_interest_signal",
                "symbol": symbol,
                "signal": signal.signal.value,
            },
        )

    async def _get_market_regime_adjustment(self, **kwargs) -> ToolResult:
        """Get market regime adjustment."""
        # Use provided data or fetch
        credit_cycle_phase = kwargs.get("credit_cycle_phase")
        volatility_regime = kwargs.get("volatility_regime")
        recession_probability = kwargs.get("recession_probability")

        # If no data provided, try to fetch
        if credit_cycle_phase is None:
            regime_data = await self._fetch_market_regime_data()
            if regime_data:
                credit_cycle_phase = regime_data.get("credit_cycle_phase", "mid_cycle")
                volatility_regime = regime_data.get("volatility_regime", "normal")
                recession_probability = regime_data.get("recession_probability", "low")
            else:
                credit_cycle_phase = "mid_cycle"
                volatility_regime = "normal"
                recession_probability = "low"

        signal = self._signal_integrator.calculate_market_regime_adjustment(
            credit_cycle_phase=credit_cycle_phase,
            volatility_regime=volatility_regime,
            recession_probability=recession_probability,
            fed_policy_stance=kwargs.get("fed_policy_stance", "neutral"),
            risk_free_rate=kwargs.get("risk_free_rate", 0.04),
            yield_curve_spread_bps=kwargs.get("yield_curve_spread_bps"),
        )

        return ToolResult.success_result(
            data=signal.to_dict(),
            metadata={
                "source": "market_regime_adjustment",
                "phase": signal.credit_cycle_phase,
                "wacc_adjustment_bps": signal.wacc_spread_adjustment_bps,
            },
        )

    async def _fetch_credit_risk_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch credit risk data from CreditRiskTool."""
        try:
            from victor_invest.tools.credit_risk import CreditRiskTool

            tool = CreditRiskTool()
            await tool.initialize()
            result = await tool.execute(action="all", symbol=symbol)

            if result.success and result.data:
                return {
                    "altman_zscore": result.data.get("altman_z", {}).get("zscore"),
                    "beneish_mscore": result.data.get("beneish_m", {}).get("mscore"),
                    "piotroski_fscore": result.data.get("piotroski_f", {}).get("fscore"),
                }
        except Exception as e:
            logger.debug(f"Could not fetch credit risk data for {symbol}: {e}")
        return None

    async def _fetch_insider_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch insider sentiment data from InsiderTradingTool."""
        try:
            from victor_invest.tools.insider_trading import InsiderTradingTool

            tool = InsiderTradingTool()
            await tool.initialize()
            result = await tool.execute(action="sentiment", symbol=symbol, days=90)

            if result.success and result.data:
                sentiment = result.data.get("sentiment", {})
                return {
                    "sentiment_score": sentiment.get("score"),
                    "buy_sell_ratio": sentiment.get("buy_sell_ratio"),
                    "cluster_detected": sentiment.get("cluster_detected", False),
                    "net_shares_change": sentiment.get("net_shares_change"),
                }
        except Exception as e:
            logger.debug(f"Could not fetch insider data for {symbol}: {e}")
        return None

    async def _fetch_short_interest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch short interest data from ShortInterestTool."""
        try:
            from victor_invest.tools.short_interest import ShortInterestTool

            tool = ShortInterestTool()
            await tool.initialize()
            result = await tool.execute(action="squeeze", symbol=symbol)

            if result.success and result.data:
                return {
                    "short_percent_float": result.data.get("short_percent_float"),
                    "days_to_cover": result.data.get("days_to_cover"),
                    "squeeze_score": result.data.get("squeeze_score"),
                }
        except Exception as e:
            logger.debug(f"Could not fetch short interest data for {symbol}: {e}")
        return None

    async def _fetch_market_regime_data(self) -> Optional[Dict[str, Any]]:
        """Fetch market regime data from MarketRegimeTool."""
        try:
            from victor_invest.tools.market_regime import MarketRegimeTool

            tool = MarketRegimeTool()
            await tool.initialize()
            result = await tool.execute(action="summary")

            if result.success and result.data:
                classifications = result.data.get("classifications", {})
                indicators = result.data.get("indicators", {})
                return {
                    "credit_cycle_phase": classifications.get("credit_cycle", "mid_cycle"),
                    "volatility_regime": classifications.get("volatility_regime", "normal"),
                    "recession_probability": classifications.get("recession_probability", "low"),
                    "fed_policy_stance": classifications.get("fed_policy_stance", "neutral"),
                    "yield_curve_spread_bps": indicators.get("yield_10y_2y_spread_bps"),
                }
        except Exception as e:
            logger.debug(f"Could not fetch market regime data: {e}")
        return None

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Valuation Signals Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["integrate", "credit_risk", "insider", "short_interest", "market_regime"],
                    "description": "Type of signal query",
                    "default": "integrate",
                },
                "symbol": {"type": "string", "description": "Stock symbol"},
                "base_fair_value": {"type": "number", "description": "Base fair value from valuation models"},
                "current_price": {"type": "number", "description": "Current stock price"},
            },
            "required": [],
        }
