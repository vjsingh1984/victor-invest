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

"""Technical Indicators Tool for Victor Invest.

This tool wraps the existing technical indicators infrastructure to provide
comprehensive technical analysis including moving averages, momentum indicators,
volatility measures, and support/resistance levels.

Infrastructure wrapped:
- investigator.infrastructure.indicators.technical_indicators.TechnicalIndicatorCalculator

Example:
    tool = TechnicalIndicatorsTool()

    # Calculate all indicators for a symbol
    result = await tool.execute(
        symbol="AAPL",
        action="calculate_all"
    )

    # Get specific indicators
    result = await tool.execute(
        symbol="AAPL",
        action="get_momentum",
        period=14
    )

    # Get recent data for analysis
    result = await tool.execute(
        symbol="AAPL",
        action="get_recent",
        days=30
    )
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TechnicalIndicatorsTool(BaseTool):
    """Tool for calculating technical indicators on market data.

    Provides comprehensive technical analysis including:
    - Moving Averages (SMA, EMA for various periods)
    - Momentum (RSI, MACD, Stochastic, Williams %R, MFI)
    - Volatility (Bollinger Bands, ATR, Historical Volatility)
    - Volume (OBV, VPT, A/D Line, VWAP)
    - Support/Resistance (Pivot Points, Fibonacci, S/R Levels)

    Attributes:
        name: "technical_indicators"
        description: Tool description for agent discovery
    """

    name = "technical_indicators"
    description = """Calculate technical indicators for stock analysis.

Actions:
- calculate_all: Calculate all indicators on historical data
- get_momentum: Get momentum indicators (RSI, MACD, Stochastic)
- get_volatility: Get volatility indicators (Bollinger, ATR)
- get_moving_averages: Get SMA/EMA for various periods
- get_volume_indicators: Get volume-based indicators (OBV, VWAP)
- get_support_resistance: Get support/resistance and Fibonacci levels
- get_recent: Get recent N days with all indicators calculated
- get_summary: Get technical analysis summary for trading signals

Parameters:
- symbol: Stock ticker symbol (required)
- action: One of the actions above (default: "calculate_all")
- days: Number of days of data (default: 365)
- recent_days: Days to return for get_recent (default: 30)
- period: Indicator period for specific calculations

Returns calculated indicators as structured data suitable for analysis.
"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Technical Indicators Tool.

        Args:
            config: Optional investigator config object
        """
        super().__init__(config)
        self._calculator = None
        self._market_data_fetcher = None

    async def initialize(self) -> None:
        """Initialize technical analysis infrastructure."""
        try:
            from investigator.infrastructure.indicators.technical_indicators import (
                get_technical_calculator
            )
            from investigator.infrastructure.database.market_data import (
                get_market_data_fetcher
            )

            if self.config is None:
                from investigator.config import get_config
                self.config = get_config()

            self._calculator = get_technical_calculator()
            self._market_data_fetcher = get_market_data_fetcher(self.config)

            self._initialized = True
            logger.info("TechnicalIndicatorsTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TechnicalIndicatorsTool: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Dict[str, Any],
        symbol: str = "",
        action: str = "calculate_all",
        days: int = 365,
        recent_days: int = 30,
        period: int = 14,
        **kwargs
    ) -> ToolResult:
        """Execute technical indicator calculation.

        Args:
            symbol: Stock ticker symbol
            action: Operation to perform:
                - "calculate_all": Calculate all indicators
                - "get_momentum": RSI, MACD, Stochastic, etc.
                - "get_volatility": Bollinger Bands, ATR, etc.
                - "get_moving_averages": SMA/EMA for various periods
                - "get_volume_indicators": OBV, VWAP, VPT, etc.
                - "get_support_resistance": Pivot, Fib, S/R levels
                - "get_recent": Recent data with indicators
                - "get_summary": Technical analysis summary
            days: Historical data lookback period
            recent_days: Days to return for get_recent action
            period: Indicator calculation period
            **kwargs: Additional parameters

        Returns:
            ToolResult with calculated indicators or error
        """
        try:
            await self.ensure_initialized()

            symbol = symbol.upper().strip()
            if not symbol:
                return ToolResult.error_result("Symbol is required")

            action = action.lower().strip()

            # Fetch market data first
            df = await self._fetch_market_data(symbol, days)
            if df is None or df.empty:
                return ToolResult.error_result(
                    f"No market data available for {symbol}",
                    metadata={"symbol": symbol, "days": days}
                )

            # Calculate all indicators on full dataset
            enhanced_df = await self._calculate_indicators(df, symbol)

            if action == "calculate_all":
                return self._format_all_indicators(symbol, enhanced_df)
            elif action == "get_momentum":
                return self._format_momentum(symbol, enhanced_df, period)
            elif action == "get_volatility":
                return self._format_volatility(symbol, enhanced_df)
            elif action == "get_moving_averages":
                return self._format_moving_averages(symbol, enhanced_df)
            elif action == "get_volume_indicators":
                return self._format_volume_indicators(symbol, enhanced_df)
            elif action == "get_support_resistance":
                return self._format_support_resistance(symbol, enhanced_df)
            elif action == "get_recent":
                return self._format_recent(symbol, enhanced_df, recent_days)
            elif action == "get_summary":
                return self._format_summary(symbol, enhanced_df)
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: "
                    "calculate_all, get_momentum, get_volatility, get_moving_averages, "
                    "get_volume_indicators, get_support_resistance, get_recent, get_summary"
                )

        except Exception as e:
            logger.error(f"TechnicalIndicatorsTool execute error for {symbol}: {e}")
            return ToolResult.error_result(
                f"Technical analysis failed: {str(e)}",
                metadata={"symbol": symbol, "action": action}
            )

    async def _fetch_market_data(
        self,
        symbol: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """Fetch market data for technical analysis.

        Args:
            symbol: Stock ticker
            days: Number of days

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self._market_data_fetcher.get_stock_data,
                symbol,
                days
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def _calculate_indicators(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Calculate all technical indicators.

        Args:
            df: OHLCV DataFrame
            symbol: Stock ticker

        Returns:
            Enhanced DataFrame with indicators
        """
        try:
            loop = asyncio.get_event_loop()
            enhanced_df = await loop.run_in_executor(
                None,
                self._calculator.calculate_all_indicators,
                df,
                symbol
            )
            return enhanced_df
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return df

    def _format_all_indicators(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> ToolResult:
        """Format all indicators for response."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}

            # Clean up any NaN values
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "data_points": len(df),
                    "date_range": {
                        "start": str(df.index[0]) if not df.empty else None,
                        "end": str(df.index[-1]) if not df.empty else None,
                    },
                    "latest": {
                        "price": {
                            "open": latest.get("Open"),
                            "high": latest.get("High"),
                            "low": latest.get("Low"),
                            "close": latest.get("Close"),
                            "volume": latest.get("Volume"),
                        },
                        "moving_averages": {
                            "sma_20": latest.get("SMA_20"),
                            "sma_50": latest.get("SMA_50"),
                            "sma_200": latest.get("SMA_200"),
                            "ema_12": latest.get("EMA_12"),
                            "ema_26": latest.get("EMA_26"),
                        },
                        "momentum": {
                            "rsi_14": latest.get("RSI_14"),
                            "macd": latest.get("MACD"),
                            "macd_signal": latest.get("MACD_Signal"),
                            "macd_histogram": latest.get("MACD_Histogram"),
                            "stoch_k": latest.get("Stoch_K"),
                            "stoch_d": latest.get("Stoch_D"),
                            "williams_r": latest.get("Williams_R"),
                            "mfi_14": latest.get("MFI_14"),
                        },
                        "volatility": {
                            "bb_upper": latest.get("BB_Upper"),
                            "bb_middle": latest.get("BB_Middle"),
                            "bb_lower": latest.get("BB_Lower"),
                            "bb_position": latest.get("BB_Position"),
                            "atr_14": latest.get("ATR_14"),
                            "volatility_20": latest.get("Volatility_20"),
                        },
                        "volume": {
                            "obv": latest.get("OBV"),
                            "vwap": latest.get("VWAP"),
                            "volume_sma_20": latest.get("Volume_SMA_20"),
                            "volume_ratio": latest.get("Volume_Ratio"),
                        },
                        "levels": {
                            "high_52w": latest.get("High_52w"),
                            "low_52w": latest.get("Low_52w"),
                            "support_1": latest.get("Support_1"),
                            "resistance_1": latest.get("Resistance_1"),
                            "pivot_point": latest.get("Pivot_Point"),
                            "fib_38_2": latest.get("Fib_38_2"),
                            "fib_50_0": latest.get("Fib_50_0"),
                            "fib_61_8": latest.get("Fib_61_8"),
                        }
                    },
                    "columns_calculated": len(df.columns)
                },
                metadata={
                    "source": "technical_calculator",
                    "data_points": len(df)
                }
            )

        except Exception as e:
            logger.error(f"Error formatting all indicators: {e}")
            return ToolResult.error_result(f"Failed to format indicators: {str(e)}")

    def _format_momentum(
        self,
        symbol: str,
        df: pd.DataFrame,
        period: int
    ) -> ToolResult:
        """Format momentum indicators."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "rsi": {
                        f"rsi_{p}": latest.get(f"RSI_{p}")
                        for p in [9, 14, 21]
                    },
                    "macd": {
                        "macd": latest.get("MACD"),
                        "signal": latest.get("MACD_Signal"),
                        "histogram": latest.get("MACD_Histogram"),
                    },
                    "stochastic": {
                        "k": latest.get("Stoch_K"),
                        "d": latest.get("Stoch_D"),
                    },
                    "williams_r": latest.get("Williams_R"),
                    "mfi": latest.get("MFI_14"),
                    "roc": {
                        "roc_10": latest.get("ROC_10"),
                        "roc_20": latest.get("ROC_20"),
                    },
                    "signals": self._interpret_momentum_signals(latest)
                },
                metadata={"indicator_type": "momentum"}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to format momentum: {str(e)}")

    def _format_volatility(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> ToolResult:
        """Format volatility indicators."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "bollinger_bands": {
                        "upper": latest.get("BB_Upper"),
                        "middle": latest.get("BB_Middle"),
                        "lower": latest.get("BB_Lower"),
                        "width": latest.get("BB_Width"),
                        "position": latest.get("BB_Position"),
                    },
                    "atr": latest.get("ATR_14"),
                    "volatility_20d": latest.get("Volatility_20"),
                    "signals": self._interpret_volatility_signals(latest)
                },
                metadata={"indicator_type": "volatility"}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to format volatility: {str(e)}")

    def _format_moving_averages(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> ToolResult:
        """Format moving averages."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            current_price = latest.get("Close")

            sma_data = {f"sma_{p}": latest.get(f"SMA_{p}") for p in [5, 10, 20, 50, 100, 200]}
            ema_data = {f"ema_{p}": latest.get(f"EMA_{p}") for p in [5, 10, 12, 20, 26, 50, 100, 200]}

            # Calculate price vs MA signals
            signals = {}
            if current_price:
                for key, value in {**sma_data, **ema_data}.items():
                    if value:
                        signals[key] = "above" if current_price > value else "below"

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "current_price": current_price,
                    "sma": sma_data,
                    "ema": ema_data,
                    "price_vs_ma": signals,
                    "golden_cross": self._check_golden_cross(df),
                    "death_cross": self._check_death_cross(df),
                },
                metadata={"indicator_type": "moving_averages"}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to format moving averages: {str(e)}")

    def _format_volume_indicators(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> ToolResult:
        """Format volume indicators."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "current_volume": latest.get("Volume"),
                    "volume_sma_20": latest.get("Volume_SMA_20"),
                    "volume_ratio": latest.get("Volume_Ratio"),
                    "obv": latest.get("OBV"),
                    "vpt": latest.get("VPT"),
                    "ad_line": latest.get("AD"),
                    "vwap": latest.get("VWAP"),
                    "signals": self._interpret_volume_signals(latest)
                },
                metadata={"indicator_type": "volume"}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to format volume: {str(e)}")

    def _format_support_resistance(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> ToolResult:
        """Format support/resistance levels."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "current_price": latest.get("Close"),
                    "52_week": {
                        "high": latest.get("High_52w"),
                        "low": latest.get("Low_52w"),
                    },
                    "support_levels": {
                        "support_1": latest.get("Support_1"),
                        "support_2": latest.get("Support_2"),
                        "pivot_s1": latest.get("Pivot_S1"),
                        "pivot_s2": latest.get("Pivot_S2"),
                    },
                    "resistance_levels": {
                        "resistance_1": latest.get("Resistance_1"),
                        "resistance_2": latest.get("Resistance_2"),
                        "pivot_r1": latest.get("Pivot_R1"),
                        "pivot_r2": latest.get("Pivot_R2"),
                    },
                    "pivot_point": latest.get("Pivot_Point"),
                    "fibonacci_levels": {
                        "0.0%": latest.get("Fib_0"),
                        "23.6%": latest.get("Fib_23_6"),
                        "38.2%": latest.get("Fib_38_2"),
                        "50.0%": latest.get("Fib_50_0"),
                        "61.8%": latest.get("Fib_61_8"),
                        "78.6%": latest.get("Fib_78_6"),
                        "100%": latest.get("Fib_100"),
                    }
                },
                metadata={"indicator_type": "support_resistance"}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to format S/R levels: {str(e)}")

    def _format_recent(
        self,
        symbol: str,
        df: pd.DataFrame,
        days: int
    ) -> ToolResult:
        """Format recent data with indicators."""
        try:
            recent_df = self._calculator.extract_recent_data_for_llm(df, days)

            # Convert to list of records
            records = []
            for idx, row in recent_df.iterrows():
                record = row.to_dict()
                record["date"] = str(idx)
                # Clean NaN values
                record = {k: (v if pd.notna(v) else None) for k, v in record.items()}
                records.append(record)

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "days": len(records),
                    "data": records
                },
                metadata={"indicator_type": "recent_data", "requested_days": days}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to format recent data: {str(e)}")

    def _format_summary(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> ToolResult:
        """Generate technical analysis summary with trading signals."""
        try:
            latest = df.iloc[-1].to_dict() if not df.empty else {}
            latest = {k: (v if pd.notna(v) else None) for k, v in latest.items()}

            current_price = latest.get("Close")

            # Gather signals
            momentum_signals = self._interpret_momentum_signals(latest)
            volatility_signals = self._interpret_volatility_signals(latest)
            volume_signals = self._interpret_volume_signals(latest)
            ma_signals = self._interpret_ma_signals(latest, current_price)

            # Calculate overall score
            bullish = 0
            bearish = 0
            neutral = 0

            all_signals = {**momentum_signals, **volatility_signals, **volume_signals, **ma_signals}
            for signal in all_signals.values():
                if signal == "bullish":
                    bullish += 1
                elif signal == "bearish":
                    bearish += 1
                else:
                    neutral += 1

            total = bullish + bearish + neutral
            if total > 0:
                bullish_pct = (bullish / total) * 100
                bearish_pct = (bearish / total) * 100
            else:
                bullish_pct = bearish_pct = 0

            # Determine overall bias
            if bullish_pct > 60:
                overall = "bullish"
            elif bearish_pct > 60:
                overall = "bearish"
            else:
                overall = "neutral"

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "current_price": current_price,
                    "overall_signal": overall,
                    "signal_counts": {
                        "bullish": bullish,
                        "bearish": bearish,
                        "neutral": neutral,
                    },
                    "signal_percentages": {
                        "bullish_pct": round(bullish_pct, 1),
                        "bearish_pct": round(bearish_pct, 1),
                    },
                    "signals": {
                        "momentum": momentum_signals,
                        "volatility": volatility_signals,
                        "volume": volume_signals,
                        "moving_averages": ma_signals,
                    },
                    "key_levels": {
                        "support": latest.get("Support_1"),
                        "resistance": latest.get("Resistance_1"),
                        "pivot": latest.get("Pivot_Point"),
                    }
                },
                metadata={"indicator_type": "summary"}
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to generate summary: {str(e)}")

    def _interpret_momentum_signals(self, latest: Dict) -> Dict[str, str]:
        """Interpret momentum indicator signals."""
        signals = {}

        # RSI
        rsi = latest.get("RSI_14")
        if rsi is not None:
            if rsi < 30:
                signals["rsi"] = "bullish"  # Oversold
            elif rsi > 70:
                signals["rsi"] = "bearish"  # Overbought
            else:
                signals["rsi"] = "neutral"

        # MACD
        macd = latest.get("MACD")
        macd_signal = latest.get("MACD_Signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                signals["macd"] = "bullish"
            else:
                signals["macd"] = "bearish"

        # Stochastic
        stoch_k = latest.get("Stoch_K")
        if stoch_k is not None:
            if stoch_k < 20:
                signals["stochastic"] = "bullish"
            elif stoch_k > 80:
                signals["stochastic"] = "bearish"
            else:
                signals["stochastic"] = "neutral"

        return signals

    def _interpret_volatility_signals(self, latest: Dict) -> Dict[str, str]:
        """Interpret volatility indicator signals."""
        signals = {}

        bb_position = latest.get("BB_Position")
        if bb_position is not None:
            if bb_position < 0.2:
                signals["bollinger"] = "bullish"  # Near lower band
            elif bb_position > 0.8:
                signals["bollinger"] = "bearish"  # Near upper band
            else:
                signals["bollinger"] = "neutral"

        return signals

    def _interpret_volume_signals(self, latest: Dict) -> Dict[str, str]:
        """Interpret volume indicator signals."""
        signals = {}

        volume_ratio = latest.get("Volume_Ratio")
        if volume_ratio is not None:
            if volume_ratio > 1.5:
                signals["volume"] = "high_activity"
            elif volume_ratio < 0.5:
                signals["volume"] = "low_activity"
            else:
                signals["volume"] = "normal"

        return signals

    def _interpret_ma_signals(
        self,
        latest: Dict,
        current_price: Optional[float]
    ) -> Dict[str, str]:
        """Interpret moving average signals."""
        signals = {}

        if current_price is None:
            return signals

        sma_50 = latest.get("SMA_50")
        sma_200 = latest.get("SMA_200")

        if sma_50:
            signals["sma_50"] = "bullish" if current_price > sma_50 else "bearish"
        if sma_200:
            signals["sma_200"] = "bullish" if current_price > sma_200 else "bearish"

        return signals

    def _check_golden_cross(self, df: pd.DataFrame) -> Optional[bool]:
        """Check for golden cross (SMA50 crosses above SMA200)."""
        try:
            if len(df) < 2:
                return None
            sma50_prev = df["SMA_50"].iloc[-2]
            sma200_prev = df["SMA_200"].iloc[-2]
            sma50_curr = df["SMA_50"].iloc[-1]
            sma200_curr = df["SMA_200"].iloc[-1]

            if pd.isna(sma50_prev) or pd.isna(sma200_prev) or pd.isna(sma50_curr) or pd.isna(sma200_curr):
                return None

            # Golden cross: SMA50 was below SMA200, now above
            return sma50_prev < sma200_prev and sma50_curr > sma200_curr
        except Exception:
            return None

    def _check_death_cross(self, df: pd.DataFrame) -> Optional[bool]:
        """Check for death cross (SMA50 crosses below SMA200)."""
        try:
            if len(df) < 2:
                return None
            sma50_prev = df["SMA_50"].iloc[-2]
            sma200_prev = df["SMA_200"].iloc[-2]
            sma50_curr = df["SMA_50"].iloc[-1]
            sma200_curr = df["SMA_200"].iloc[-1]

            if pd.isna(sma50_prev) or pd.isna(sma200_prev) or pd.isna(sma50_curr) or pd.isna(sma200_curr):
                return None

            # Death cross: SMA50 was above SMA200, now below
            return sma50_prev > sma200_prev and sma50_curr < sma200_curr
        except Exception:
            return None

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Technical Indicators Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "calculate_all", "get_momentum", "get_volatility",
                        "get_moving_averages", "get_volume_indicators",
                        "get_support_resistance", "get_recent", "get_summary"
                    ],
                    "description": "Action to perform",
                    "default": "calculate_all"
                },
                "days": {
                    "type": "integer",
                    "description": "Days of historical data",
                    "default": 365,
                    "minimum": 30,
                    "maximum": 1825
                },
                "recent_days": {
                    "type": "integer",
                    "description": "Days to return for get_recent",
                    "default": 30,
                    "minimum": 5,
                    "maximum": 90
                },
                "period": {
                    "type": "integer",
                    "description": "Indicator calculation period",
                    "default": 14
                }
            },
            "required": ["symbol"]
        }
