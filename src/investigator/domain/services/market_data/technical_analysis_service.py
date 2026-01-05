# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Technical Analysis Service - Shared module for technical analysis operations.

This service provides a consistent interface for technical analysis across:
- rl_backtest.py (RL training data generation)
- batch_analysis_runner.py (production analysis)
- test_entry_exit_db.py (signal testing)
- entry_exit_engine.py (signal generation)

Features:
- Technical indicator calculation (RSI, MACD, OBV, ADX, Stochastic, MFI)
- Entry/exit signal generation using EntryExitEngine
- Feature extraction for RL state representation
- Caching for performance

Example:
    from investigator.domain.services.market_data import TechnicalAnalysisService

    ta_service = TechnicalAnalysisService()

    # Get technical features for RL
    features = ta_service.get_technical_features(symbol, analysis_date)

    # Get full indicator DataFrame
    df = ta_service.calculate_indicators(symbol, start_date, end_date)
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from investigator.domain.services.market_data.price_service import PriceService
from investigator.infrastructure.indicators.technical_indicators import (
    TechnicalIndicatorCalculator,
    get_technical_calculator,
)

logger = logging.getLogger(__name__)


@dataclass
class TechnicalFeatures:
    """Technical features for RL state representation."""

    # Core momentum indicators
    rsi_14: float = 50.0
    macd_histogram: float = 0.0
    obv_trend: float = 0.0  # -1 (bearish) to +1 (bullish)
    adx_14: float = 25.0
    stoch_k: float = 50.0
    mfi_14: float = 50.0

    # Entry/Exit signal features
    entry_signal_strength: float = 0.0  # -1 (avoid) to +1 (strong buy)
    exit_signal_strength: float = 0.0  # -1 (hold) to +1 (strong sell)
    signal_confluence: float = 0.0  # How many signals agree
    days_from_support: float = 0.5  # 0 (at support) to 1 (at resistance)
    risk_reward_ratio: float = 2.0

    # Price relative to moving averages
    price_vs_sma_20: float = 0.0
    price_vs_sma_50: float = 0.0
    price_vs_sma_200: float = 0.0

    # Volatility
    volatility: float = 0.5  # Normalized ATR

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for RL context."""
        return {
            "rsi_14": self.rsi_14,
            "macd_histogram": self.macd_histogram,
            "obv_trend": self.obv_trend,
            "adx_14": self.adx_14,
            "stoch_k": self.stoch_k,
            "mfi_14": self.mfi_14,
            "entry_signal_strength": self.entry_signal_strength,
            "exit_signal_strength": self.exit_signal_strength,
            "signal_confluence": self.signal_confluence,
            "days_from_support": self.days_from_support,
            "risk_reward_ratio": self.risk_reward_ratio,
            "price_vs_sma_20": self.price_vs_sma_20,
            "price_vs_sma_50": self.price_vs_sma_50,
            "price_vs_sma_200": self.price_vs_sma_200,
            "volatility": self.volatility,
        }


class TechnicalAnalysisService:
    """
    Shared service for technical analysis across the codebase.

    Wraps TechnicalIndicatorCalculator and EntryExitEngine to provide
    consistent features for RL training and production analysis.
    """

    def __init__(
        self,
        price_service: Optional[PriceService] = None,
        indicator_calculator: Optional[TechnicalIndicatorCalculator] = None,
    ):
        """
        Initialize TechnicalAnalysisService.

        Args:
            price_service: Optional PriceService instance (creates one if not provided)
            indicator_calculator: Optional calculator (uses shared instance if not provided)
        """
        self.price_service = price_service or PriceService()
        self.calculator = indicator_calculator or get_technical_calculator()
        self._entry_exit_engine = None  # Lazy load

    @property
    def entry_exit_engine(self):
        """Lazy load entry/exit engine to avoid circular imports."""
        if self._entry_exit_engine is None:
            from investigator.domain.services.signals.entry_exit_engine import EntryExitEngine

            self._entry_exit_engine = EntryExitEngine()
        return self._entry_exit_engine

    def get_technical_features(
        self,
        symbol: str,
        analysis_date: date,
        lookback_days: int = 365,
        fair_value: Optional[float] = None,
    ) -> TechnicalFeatures:
        """
        Get technical features for RL state representation.

        Args:
            symbol: Stock ticker symbol
            analysis_date: Date of analysis (uses data available up to this date)
            lookback_days: Days of history to use for indicator calculation
            fair_value: Optional fair value for entry/exit signal calculation

        Returns:
            TechnicalFeatures with normalized values for RL
        """
        try:
            # Get price history
            start_date = analysis_date - timedelta(days=lookback_days)
            df = self.price_service.get_price_history(symbol, start_date, analysis_date)

            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient price data for {symbol}: {len(df)} rows")
                return TechnicalFeatures()  # Return defaults

            # Calculate indicators
            df_enhanced = self._calculate_indicators(df, symbol)

            # Extract features from latest row
            return self._extract_features(df_enhanced, fair_value)

        except Exception as e:
            logger.error(f"Error getting technical features for {symbol}: {e}")
            return TechnicalFeatures()

    def calculate_indicators(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Calculate full technical indicators DataFrame.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Enhanced DataFrame with all technical indicators
        """
        df = self.price_service.get_price_history(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame()
        return self._calculate_indicators(df, symbol)

    def get_entry_exit_signals(
        self,
        symbol: str,
        analysis_date: date,
        fair_value: float,
        lookback_days: int = 365,
    ) -> Dict[str, Any]:
        """
        Get entry/exit signals for a symbol.

        Args:
            symbol: Stock ticker symbol
            analysis_date: Date of analysis
            fair_value: Fair value estimate for the stock
            lookback_days: Days of history to use

        Returns:
            Dictionary with entry_signals, exit_signals, and optimal_entry_zone
        """
        try:
            start_date = analysis_date - timedelta(days=lookback_days)
            df = self.price_service.get_price_history(symbol, start_date, analysis_date)

            if df.empty or len(df) < 50:
                return {"entry_signals": [], "exit_signals": [], "optimal_entry_zone": None}

            df_enhanced = self._calculate_indicators(df, symbol)
            current_price = float(df_enhanced["Close"].iloc[-1])

            # Build indicators dict for entry/exit engine
            indicators = self._build_indicators_dict(df_enhanced)

            # Get support/resistance levels
            support_resistance = self._get_support_resistance(df_enhanced, current_price)

            # Valuation info
            valuation = {
                "fair_value": fair_value,
                "upside": (fair_value - current_price) / current_price if current_price > 0 else 0,
            }

            # Generate signals
            entry_signals = self.entry_exit_engine.generate_entry_signals(
                price_data=df_enhanced,
                indicators=indicators,
                valuation=valuation,
                support_resistance=support_resistance,
            )

            exit_signals = self.entry_exit_engine.generate_exit_signals(
                price_data=df_enhanced,
                indicators=indicators,
                position_info={"entry_price": current_price, "current_price": current_price, "fair_value": fair_value},
            )

            # Calculate optimal entry zone
            support_levels = [support_resistance["support_2"], support_resistance["support_1"]]
            resistance_levels = [support_resistance["resistance_1"], support_resistance["resistance_2"]]
            entry_zone = self.entry_exit_engine.calculate_optimal_entry_zone(
                current_price=current_price,
                fair_value=fair_value,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volatility=indicators.get("atr_14", 0) / current_price if current_price > 0 else 0.02,
                atr=indicators.get("atr_14", 0),
            )

            return {
                "entry_signals": entry_signals,
                "exit_signals": exit_signals,
                "optimal_entry_zone": entry_zone,
                "support_resistance": support_resistance,
                "indicators": indicators,
            }

        except Exception as e:
            logger.error(f"Error getting entry/exit signals for {symbol}: {e}")
            return {"entry_signals": [], "exit_signals": [], "optimal_entry_zone": None}

    def get_market_context_for_rl(
        self,
        symbol: str,
        analysis_date: date,
        fair_value: Optional[float] = None,
        lookback_days: int = 365,
    ) -> Dict[str, Any]:
        """
        Get full market context dictionary for RL feature extraction.

        This is the format expected by ValuationContextExtractor.extract()

        Args:
            symbol: Stock ticker symbol
            analysis_date: Date of analysis
            fair_value: Optional fair value for signal calculation
            lookback_days: Days of history

        Returns:
            Dictionary with technical_indicators and entry_exit_signals
        """
        features = self.get_technical_features(symbol, analysis_date, lookback_days, fair_value)

        # Compute trend score from price vs moving averages
        trend_score = (features.price_vs_sma_20 + features.price_vs_sma_50 + features.price_vs_sma_200) / 3
        trend_score = max(-1.0, min(1.0, trend_score))

        # Compute sentiment from momentum indicators
        rsi_sentiment = (features.rsi_14 - 50) / 50  # -1 to +1
        stoch_sentiment = (features.stoch_k - 50) / 50
        mfi_sentiment = (features.mfi_14 - 50) / 50
        sentiment_score = (rsi_sentiment + stoch_sentiment + mfi_sentiment) / 3

        return {
            "trend_score": trend_score,
            "sentiment_score": sentiment_score,
            "volatility": features.volatility,
            "technical_indicators": {
                "rsi_14": features.rsi_14,
                "macd_histogram": features.macd_histogram,
                "obv_trend": features.obv_trend,
                "adx_14": features.adx_14,
                "stoch_k": features.stoch_k,
                "mfi_14": features.mfi_14,
            },
            "entry_exit_signals": {
                "entry_signal_strength": features.entry_signal_strength,
                "exit_signal_strength": features.exit_signal_strength,
                "signal_confluence": features.signal_confluence,
                "days_from_support": features.days_from_support,
                "risk_reward_ratio": features.risk_reward_ratio,
            },
        }

    def _calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators on price DataFrame."""
        # Convert Decimal to float if needed (database returns Decimal)
        df_calc = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_calc.columns:
                df_calc[col] = df_calc[col].astype(float)

        # Rename to expected format
        df_calc = df_calc.rename(
            columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )

        return self.calculator.calculate_all_indicators(df_calc, symbol)

    def _extract_features(
        self,
        df: pd.DataFrame,
        fair_value: Optional[float] = None,
    ) -> TechnicalFeatures:
        """Extract TechnicalFeatures from enhanced DataFrame."""
        try:
            latest = df.iloc[-1]
            current_price = float(latest["Close"])

            # Core momentum indicators
            rsi_14 = float(latest.get("RSI_14", 50))
            macd_histogram = float(latest.get("MACD_Histogram", 0))
            adx_14 = float(latest.get("ADX_14", 25))
            stoch_k = float(latest.get("Stoch_K", 50))
            mfi_14 = float(latest.get("MFI_14", 50))

            # OBV trend (-1, 0, +1)
            obv_trend = float(latest.get("OBV_Trend", 0))

            # Price vs moving averages (normalized deviation)
            sma_20 = float(latest.get("SMA_20", current_price))
            sma_50 = float(latest.get("SMA_50", current_price))
            sma_200 = float(latest.get("SMA_200", current_price))

            price_vs_sma_20 = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            price_vs_sma_50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            price_vs_sma_200 = (current_price - sma_200) / sma_200 if sma_200 > 0 else 0

            # Volatility (ATR as % of price)
            atr = float(latest.get("ATR_14", 0))
            volatility = min(1.0, atr / current_price * 5) if current_price > 0 else 0.5

            # Entry/Exit signal features
            entry_strength = 0.0
            exit_strength = 0.0
            signal_confluence = 0.0
            days_from_support = 0.5
            risk_reward = 2.0

            # Calculate signal confluence from indicators
            bullish_signals = 0
            bearish_signals = 0

            if rsi_14 < 30:
                bullish_signals += 1
            elif rsi_14 > 70:
                bearish_signals += 1

            if stoch_k < 20:
                bullish_signals += 1
            elif stoch_k > 80:
                bearish_signals += 1

            if mfi_14 < 20:
                bullish_signals += 1
            elif mfi_14 > 80:
                bearish_signals += 1

            if macd_histogram > 0:
                bullish_signals += 1
            elif macd_histogram < 0:
                bearish_signals += 1

            if obv_trend > 0:
                bullish_signals += 1
            elif obv_trend < 0:
                bearish_signals += 1

            # Calculate signal confluence (-1 to +1)
            total_signals = 5
            signal_confluence = (bullish_signals - bearish_signals) / total_signals

            # Entry strength based on confluence and valuation
            entry_strength = signal_confluence
            if fair_value and current_price > 0:
                upside = (fair_value - current_price) / current_price
                # Boost entry if undervalued
                if upside > 0.2:
                    entry_strength = min(1.0, entry_strength + 0.3)
                elif upside < -0.1:
                    entry_strength = max(-1.0, entry_strength - 0.3)

            # Exit strength (inverse of entry)
            exit_strength = -signal_confluence

            # Days from support (based on BB position)
            bb_upper = float(latest.get("BB_Upper", current_price * 1.1))
            bb_lower = float(latest.get("BB_Lower", current_price * 0.9))
            if bb_upper != bb_lower:
                days_from_support = (current_price - bb_lower) / (bb_upper - bb_lower)
            days_from_support = max(0, min(1, days_from_support))

            # Risk/reward ratio (simple estimate)
            if fair_value and current_price > 0:
                upside = fair_value - current_price
                downside = atr * 2 if atr > 0 else current_price * 0.05
                risk_reward = upside / downside if downside > 0 else 2.0
                risk_reward = max(0.5, min(5.0, risk_reward))

            return TechnicalFeatures(
                rsi_14=rsi_14,
                macd_histogram=macd_histogram,
                obv_trend=obv_trend,
                adx_14=adx_14,
                stoch_k=stoch_k,
                mfi_14=mfi_14,
                entry_signal_strength=entry_strength,
                exit_signal_strength=exit_strength,
                signal_confluence=signal_confluence,
                days_from_support=days_from_support,
                risk_reward_ratio=risk_reward,
                price_vs_sma_20=price_vs_sma_20,
                price_vs_sma_50=price_vs_sma_50,
                price_vs_sma_200=price_vs_sma_200,
                volatility=volatility,
            )

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return TechnicalFeatures()

    def _build_indicators_dict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build indicators dictionary for entry/exit engine."""
        latest = df.iloc[-1]
        current_price = float(latest["Close"])

        return {
            "rsi_14": float(latest.get("RSI_14", 50)),
            "macd": float(latest.get("MACD", 0)),
            "macd_signal": float(latest.get("MACD_Signal", 0)),
            "sma_20": float(latest.get("SMA_20", current_price)),
            "sma_50": float(latest.get("SMA_50", current_price)),
            "sma_200": float(latest.get("SMA_200", current_price)),
            "bb_upper": float(latest.get("BB_Upper", current_price * 1.1)),
            "bb_lower": float(latest.get("BB_Lower", current_price * 0.9)),
            "bb_middle": float(latest.get("BB_Middle", current_price)),
            "atr_14": float(latest.get("ATR_14", 0)),
            "obv": float(latest.get("OBV", 0)),
            "aobv_20": float(latest.get("AOBV_20", 0)),
            "obv_trend": (
                "bullish"
                if latest.get("OBV_Trend", 0) > 0
                else ("bearish" if latest.get("OBV_Trend", 0) < 0 else "neutral")
            ),
            "obv_divergence": "none",
            "volume_trend": (
                "increasing"
                if latest.get("Volume_Trend", 0) > 0
                else ("decreasing" if latest.get("Volume_Trend", 0) < 0 else "neutral")
            ),
            "ADX_14": float(latest.get("ADX_14", 25)),
            "Plus_DI": float(latest.get("Plus_DI", 0)),
            "Minus_DI": float(latest.get("Minus_DI", 0)),
            "MFI_14": float(latest.get("MFI_14", 50)),
            "Stoch_K": float(latest.get("Stoch_K", 50)),
            "Stoch_D": float(latest.get("Stoch_D", 50)),
            "Williams_R": float(latest.get("Williams_R", -50)),
            "EMA_8": float(latest.get("EMA_8", current_price)),
            "EMA_12": float(latest.get("EMA_12", current_price)),
            "EMA_21": float(latest.get("EMA_21", current_price)),
            "EMA_26": float(latest.get("EMA_26", current_price)),
            "EMA_50": float(latest.get("EMA_50", current_price)),
            "EMA_200": float(latest.get("EMA_200", current_price)),
        }

    def _get_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Get support/resistance levels from DataFrame."""
        latest = df.iloc[-1]

        support_1 = latest.get("Support_1", current_price * 0.95)
        support_2 = latest.get("Support_2", current_price * 0.90)
        resistance_1 = latest.get("Resistance_1", current_price * 1.05)
        resistance_2 = latest.get("Resistance_2", current_price * 1.10)

        # Fallback to BB if columns don't exist
        if pd.isna(support_1):
            support_1 = latest.get("BB_Lower", current_price * 0.95)
        if pd.isna(resistance_1):
            resistance_1 = latest.get("BB_Upper", current_price * 1.05)

        return {
            "support_1": float(support_1) if not pd.isna(support_1) else current_price * 0.95,
            "support_2": float(support_2) if not pd.isna(support_2) else current_price * 0.90,
            "resistance_1": float(resistance_1) if not pd.isna(resistance_1) else current_price * 1.05,
            "resistance_2": float(resistance_2) if not pd.isna(resistance_2) else current_price * 1.10,
        }


# Singleton instance for shared use
_technical_analysis_service: Optional[TechnicalAnalysisService] = None


def get_technical_analysis_service() -> TechnicalAnalysisService:
    """Get shared TechnicalAnalysisService instance."""
    global _technical_analysis_service
    if _technical_analysis_service is None:
        _technical_analysis_service = TechnicalAnalysisService()
    return _technical_analysis_service
