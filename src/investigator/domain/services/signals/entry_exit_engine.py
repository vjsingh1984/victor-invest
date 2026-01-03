"""
Entry/Exit Signal Generation Engine

Generates actionable entry and exit signals based on:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Support/resistance levels
- Valuation metrics (fair value vs current price)
- Volume confirmation
- Pattern recognition

Provides:
- Specific entry price levels with stop losses
- Exit triggers with urgency levels
- Optimal entry zones with timing guidance
- Position sizing recommendations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of entry/exit signals"""
    # Entry signals
    BREAKOUT = "breakout"
    SUPPORT_BOUNCE = "support_bounce"
    MOMENTUM = "momentum"
    DIVERGENCE = "divergence"
    PATTERN = "pattern"
    GOLDEN_CROSS = "golden_cross"
    OVERSOLD_REVERSAL = "oversold_reversal"
    VOLUME_SURGE = "volume_surge"
    OBV_ACCUMULATION = "obv_accumulation"  # AOBV bullish crossover
    OBV_BULLISH_DIVERGENCE = "obv_bullish_divergence"  # Price down, OBV up
    BB_SQUEEZE_BREAKOUT = "bb_squeeze_breakout"  # Low volatility then breakout
    BB_MEAN_REVERSION = "bb_mean_reversion"  # Reversion to middle band
    STOCHASTIC_OVERSOLD = "stochastic_oversold"  # Stochastic cross in oversold
    EMA_CROSS_SHORT = "ema_cross_short"  # 8/21 EMA bullish cross
    EMA_CROSS_MEDIUM = "ema_cross_medium"  # 12/26 EMA bullish cross
    EMA_CROSS_LONG = "ema_cross_long"  # 50/200 EMA golden cross
    ADX_TREND_START = "adx_trend_start"  # ADX crossing above 25 with +DI > -DI
    MFI_OVERSOLD = "mfi_oversold"  # MFI < 20 reversal
    DI_CROSS_BULLISH = "di_cross_bullish"  # +DI crosses above -DI

    # Exit signals
    RESISTANCE = "resistance"
    MOMENTUM_LOSS = "momentum_loss"
    PATTERN_COMPLETE = "pattern_complete"
    TARGET_HIT = "target_hit"
    STOP_LOSS = "stop_loss"
    DEATH_CROSS = "death_cross"
    OVERBOUGHT = "overbought"
    VOLUME_DIVERGENCE = "volume_divergence"
    OBV_DISTRIBUTION = "obv_distribution"  # AOBV bearish crossover
    OBV_BEARISH_DIVERGENCE = "obv_bearish_divergence"  # Price up, OBV down
    BB_UPPER_REJECTION = "bb_upper_rejection"  # Rejection at upper band
    STOCHASTIC_OVERBOUGHT = "stochastic_overbought"  # Stochastic cross in overbought
    EMA_CROSS_BEARISH_SHORT = "ema_cross_bearish_short"  # 8/21 EMA bearish cross
    EMA_CROSS_BEARISH_MEDIUM = "ema_cross_bearish_medium"  # 12/26 EMA bearish cross
    ADX_TREND_END = "adx_trend_end"  # ADX falling below 20 (trend exhaustion)
    MFI_OVERBOUGHT = "mfi_overbought"  # MFI > 80 reversal
    DI_CROSS_BEARISH = "di_cross_bearish"  # -DI crosses above +DI


class SignalConfidence(Enum):
    """Confidence level of signals"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SignalTiming(Enum):
    """Timing recommendation for entry"""
    IMMEDIATE = "immediate"
    WAIT_PULLBACK = "wait_pullback"
    AVOID_NOW = "avoid_now"
    SCALE_IN = "scale_in"


class ScalingStrategy(Enum):
    """Position scaling strategy"""
    FULL_POSITION = "full_position"
    SCALE_IN = "scale_in"
    PYRAMID = "pyramid"
    DCA = "dca"


@dataclass
class EntrySignal:
    """Represents an entry (buy) signal"""
    signal_type: SignalType
    price_level: float
    confidence: SignalConfidence
    rationale: str
    risk_reward_ratio: float
    stop_loss: float
    stop_loss_pct: float
    target_price: float
    target_pct: float
    expected_holding_days: int
    volume_confirmation: bool = False
    trend_alignment: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "price_level": round(self.price_level, 2),
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "stop_loss": round(self.stop_loss, 2),
            "stop_loss_pct": round(self.stop_loss_pct, 2),
            "target_price": round(self.target_price, 2),
            "target_pct": round(self.target_pct, 2),
            "expected_holding_days": self.expected_holding_days,
            "volume_confirmation": self.volume_confirmation,
            "trend_alignment": self.trend_alignment,
        }


@dataclass
class ExitSignal:
    """Represents an exit (sell) signal"""
    signal_type: SignalType
    price_level: float
    confidence: SignalConfidence
    rationale: str
    urgency: str  # "immediate", "staged", "watch"
    partial_exit_pct: float = 100.0  # Percentage to exit
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "price_level": round(self.price_level, 2),
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "urgency": self.urgency,
            "partial_exit_pct": round(self.partial_exit_pct, 1),
        }


@dataclass
class OptimalEntryZone:
    """Optimal entry zone with bounds and timing"""
    lower_bound: float
    upper_bound: float
    ideal_entry: float
    timing: SignalTiming
    scaling_strategy: ScalingStrategy
    confidence: SignalConfidence
    rationale: str
    recommended_allocation_pct: float  # % of portfolio
    max_position_size_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bound": round(self.lower_bound, 2),
            "upper_bound": round(self.upper_bound, 2),
            "ideal_entry": round(self.ideal_entry, 2),
            "timing": self.timing.value,
            "scaling_strategy": self.scaling_strategy.value,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "recommended_allocation_pct": round(self.recommended_allocation_pct, 2),
            "max_position_size_pct": round(self.max_position_size_pct, 2),
        }


class EntryExitEngine:
    """
    Generates entry and exit signals based on technical and fundamental analysis.

    Combines multiple signal sources:
    - Momentum indicators (RSI, MACD, Stochastic)
    - Trend indicators (Moving averages, ADX)
    - Volatility indicators (Bollinger Bands, ATR)
    - Support/Resistance levels
    - Volume analysis
    - Valuation metrics
    """

    # Signal scoring weights
    WEIGHTS = {
        "rsi": 0.12,
        "macd": 0.12,
        "moving_average": 0.12,
        "bollinger": 0.08,
        "support_resistance": 0.18,
        "volume": 0.08,
        "obv": 0.15,  # OBV/AOBV signals
        "valuation": 0.15,
    }

    # Thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_NEUTRAL_LOW = 40
    RSI_NEUTRAL_HIGH = 60

    def __init__(self):
        self.logger = logger

    def generate_entry_signals(
        self,
        price_data: pd.DataFrame,
        indicators: Optional[Dict[str, Any]] = None,
        valuation: Optional[Dict[str, Any]] = None,
        support_resistance: Optional[Dict[str, float]] = None,
    ) -> List[EntrySignal]:
        """
        Generate entry signals based on technical and fundamental confluence.

        Args:
            price_data: DataFrame with OHLCV and calculated indicators
            indicators: Pre-calculated indicator values (optional)
            valuation: Valuation data with fair_value, upside, etc.
            support_resistance: Support/resistance levels

        Returns:
            List of EntrySignal objects, sorted by confidence
        """
        signals = []

        if price_data is None or len(price_data) < 20:
            self.logger.warning("Insufficient price data for signal generation")
            return signals

        # Get latest values
        latest = price_data.iloc[-1]
        current_price = float(latest.get('Close', latest.get('close', 0)))

        if current_price <= 0:
            return signals

        # Extract indicator values
        rsi = self._get_indicator(price_data, 'RSI_14', indicators)
        macd = self._get_indicator(price_data, 'MACD', indicators)
        macd_signal = self._get_indicator(price_data, 'MACD_Signal', indicators)
        macd_hist = self._get_indicator(price_data, 'MACD_Histogram', indicators)
        sma_20 = self._get_indicator(price_data, 'SMA_20', indicators)
        sma_50 = self._get_indicator(price_data, 'SMA_50', indicators)
        sma_200 = self._get_indicator(price_data, 'SMA_200', indicators)
        bb_lower = self._get_indicator(price_data, 'BB_Lower', indicators)
        bb_upper = self._get_indicator(price_data, 'BB_Upper', indicators)
        atr = self._get_indicator(price_data, 'ATR_14', indicators)
        volume_ratio = self._get_indicator(price_data, 'Volume_Ratio', indicators)
        stoch_k = self._get_indicator(price_data, 'Stoch_K', indicators)

        # Get support/resistance from data or parameters
        support_1 = support_resistance.get('support_1') if support_resistance else self._get_indicator(price_data, 'Support_1', indicators)
        support_2 = support_resistance.get('support_2') if support_resistance else self._get_indicator(price_data, 'Support_2', indicators)
        resistance_1 = support_resistance.get('resistance_1') if support_resistance else self._get_indicator(price_data, 'Resistance_1', indicators)

        # Get valuation data
        fair_value = valuation.get('fair_value', current_price) if valuation else current_price
        upside_pct = ((fair_value - current_price) / current_price * 100) if fair_value else 0

        # Default ATR if not available
        if atr is None or atr <= 0:
            atr = current_price * 0.02  # 2% as fallback

        # Check trend alignment
        trend_bullish = self._is_trend_bullish(current_price, sma_20, sma_50, sma_200)

        # 1. RSI Oversold Reversal Signal
        if rsi is not None and rsi < self.RSI_OVERSOLD:
            # Check for reversal (RSI turning up)
            prev_rsi = self._get_prev_indicator(price_data, 'RSI_14', 1)
            if prev_rsi is not None and rsi > prev_rsi:
                stop_loss = current_price - (2 * atr)
                target = current_price + (3 * atr)
                signals.append(EntrySignal(
                    signal_type=SignalType.OVERSOLD_REVERSAL,
                    price_level=current_price,
                    confidence=SignalConfidence.HIGH if rsi < 25 else SignalConfidence.MEDIUM,
                    rationale=f"RSI at {rsi:.1f} (oversold) with reversal signal",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=((target - current_price) / current_price) * 100,
                    expected_holding_days=20,
                    volume_confirmation=volume_ratio is not None and volume_ratio > 1.2,
                    trend_alignment=trend_bullish,
                ))

        # 2. Support Bounce Signal
        if support_1 is not None:
            distance_to_support = (current_price - support_1) / current_price * 100
            if 0 < distance_to_support < 3:  # Within 3% of support
                stop_loss = support_1 * 0.97  # 3% below support
                target = resistance_1 if resistance_1 else current_price * 1.10
                signals.append(EntrySignal(
                    signal_type=SignalType.SUPPORT_BOUNCE,
                    price_level=support_1,
                    confidence=SignalConfidence.HIGH if distance_to_support < 1.5 else SignalConfidence.MEDIUM,
                    rationale=f"Price near support at ${support_1:.2f} ({distance_to_support:.1f}% above)",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=((target - current_price) / current_price) * 100,
                    expected_holding_days=30,
                    volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                    trend_alignment=trend_bullish,
                ))

        # 3. MACD Bullish Crossover
        if macd is not None and macd_signal is not None:
            prev_macd = self._get_prev_indicator(price_data, 'MACD', 1)
            prev_signal = self._get_prev_indicator(price_data, 'MACD_Signal', 1)

            if prev_macd is not None and prev_signal is not None:
                # Bullish crossover: MACD crosses above signal
                if prev_macd < prev_signal and macd > macd_signal:
                    stop_loss = current_price - (2 * atr)
                    target = current_price + (2.5 * atr)
                    signals.append(EntrySignal(
                        signal_type=SignalType.MOMENTUM,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH if macd_hist > 0 else SignalConfidence.MEDIUM,
                        rationale=f"MACD bullish crossover (MACD: {macd:.2f}, Signal: {macd_signal:.2f})",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=((target - current_price) / current_price) * 100,
                        expected_holding_days=15,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 1.2,
                        trend_alignment=trend_bullish,
                    ))

        # 4. Golden Cross (50 SMA crosses above 200 SMA)
        if sma_50 is not None and sma_200 is not None:
            prev_sma_50 = self._get_prev_indicator(price_data, 'SMA_50', 1)
            prev_sma_200 = self._get_prev_indicator(price_data, 'SMA_200', 1)

            if prev_sma_50 is not None and prev_sma_200 is not None:
                if prev_sma_50 < prev_sma_200 and sma_50 > sma_200:
                    stop_loss = sma_200 * 0.95
                    target = current_price * 1.20  # 20% target for golden cross
                    signals.append(EntrySignal(
                        signal_type=SignalType.GOLDEN_CROSS,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH,
                        rationale=f"Golden Cross: 50 SMA (${sma_50:.2f}) crossed above 200 SMA (${sma_200:.2f})",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=((target - current_price) / current_price) * 100,
                        expected_holding_days=90,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                        trend_alignment=True,  # Golden cross is bullish by definition
                    ))

        # 5. Bollinger Band Bounce (price touches lower band)
        if bb_lower is not None:
            distance_to_bb = (current_price - bb_lower) / current_price * 100
            if 0 < distance_to_bb < 2:  # Within 2% of lower band
                stop_loss = bb_lower * 0.97
                target = bb_upper if bb_upper else current_price * 1.08
                signals.append(EntrySignal(
                    signal_type=SignalType.SUPPORT_BOUNCE,
                    price_level=current_price,
                    confidence=SignalConfidence.MEDIUM,
                    rationale=f"Price near Bollinger lower band at ${bb_lower:.2f}",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=((target - current_price) / current_price) * 100,
                    expected_holding_days=10,
                    volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                    trend_alignment=trend_bullish,
                ))

        # 6. Breakout Signal (price breaks above resistance with volume)
        if resistance_1 is not None:
            if current_price > resistance_1 and volume_ratio is not None and volume_ratio > 1.5:
                stop_loss = resistance_1 * 0.97  # Previous resistance becomes support
                target = current_price + (resistance_1 - support_1) if support_1 else current_price * 1.10
                signals.append(EntrySignal(
                    signal_type=SignalType.BREAKOUT,
                    price_level=current_price,
                    confidence=SignalConfidence.HIGH if volume_ratio > 2.0 else SignalConfidence.MEDIUM,
                    rationale=f"Breakout above resistance ${resistance_1:.2f} with {volume_ratio:.1f}x volume",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=((target - current_price) / current_price) * 100,
                    expected_holding_days=20,
                    volume_confirmation=True,
                    trend_alignment=trend_bullish,
                ))

        # 7. Valuation-based entry (significant undervaluation)
        if valuation and upside_pct > 15:
            stop_loss = current_price * 0.90  # 10% stop
            target = fair_value
            signals.append(EntrySignal(
                signal_type=SignalType.MOMENTUM,  # Fundamental momentum
                price_level=current_price,
                confidence=SignalConfidence.HIGH if upside_pct > 25 else SignalConfidence.MEDIUM,
                rationale=f"Trading at ${current_price:.2f} vs fair value ${fair_value:.2f} ({upside_pct:.1f}% upside)",
                risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                stop_loss=stop_loss,
                stop_loss_pct=10.0,
                target_price=target,
                target_pct=upside_pct,
                expected_holding_days=90,
                volume_confirmation=False,
                trend_alignment=trend_bullish,
            ))

        # 8. OBV Accumulation Signal (AOBV bullish crossover)
        obv_trend = self._get_indicator(price_data, 'OBV_Trend', indicators)
        aobv_20 = self._get_indicator(price_data, 'AOBV_20', indicators)
        aobv_50 = self._get_indicator(price_data, 'AOBV_50', indicators)
        obv_histogram = self._get_indicator(price_data, 'OBV_Histogram', indicators)

        if obv_trend is not None and aobv_20 is not None and aobv_50 is not None:
            # Check for AOBV bullish crossover (AOBV_20 crosses above AOBV_50)
            prev_aobv_20 = self._get_prev_indicator(price_data, 'AOBV_20', 1)
            prev_aobv_50 = self._get_prev_indicator(price_data, 'AOBV_50', 1)

            if prev_aobv_20 is not None and prev_aobv_50 is not None:
                if prev_aobv_20 < prev_aobv_50 and aobv_20 > aobv_50:
                    # Bullish AOBV crossover detected
                    stop_loss = current_price - (2 * atr)
                    target = resistance_1 if resistance_1 else current_price * 1.12
                    signals.append(EntrySignal(
                        signal_type=SignalType.OBV_ACCUMULATION,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH if obv_histogram and obv_histogram > 0 else SignalConfidence.MEDIUM,
                        rationale=f"OBV accumulation: AOBV(20) crossed above AOBV(50) - institutional buying",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=((target - current_price) / current_price) * 100,
                        expected_holding_days=30,
                        volume_confirmation=True,
                        trend_alignment=trend_bullish,
                    ))

        # 9. OBV Bullish Divergence (price down, OBV up - accumulation)
        obv_roc_20 = self._get_indicator(price_data, 'OBV_ROC_20', indicators)
        price_change_10d = self._get_indicator(price_data, 'Price_Change_10D', indicators)

        if obv_roc_20 is not None and price_change_10d is not None:
            # Bullish divergence: price falling but OBV rising (smart money accumulating)
            if price_change_10d < -5 and obv_roc_20 > 5:
                stop_loss = current_price - (2.5 * atr)
                target = current_price * 1.15  # 15% target for divergence plays
                signals.append(EntrySignal(
                    signal_type=SignalType.OBV_BULLISH_DIVERGENCE,
                    price_level=current_price,
                    confidence=SignalConfidence.HIGH if obv_roc_20 > 10 else SignalConfidence.MEDIUM,
                    rationale=f"Bullish OBV divergence: price down {price_change_10d:.1f}% but OBV up {obv_roc_20:.1f}% - accumulation",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=15.0,
                    expected_holding_days=45,
                    volume_confirmation=True,
                    trend_alignment=False,  # Divergence often occurs against trend
                ))

        # 10. OBV Momentum Confirmation (strong OBV with price breakout)
        obv = self._get_indicator(price_data, 'OBV', indicators)
        obv_signal = self._get_indicator(price_data, 'OBV_Signal', indicators)

        if obv is not None and obv_signal is not None and obv_roc_20 is not None:
            # OBV above signal line with strong momentum confirms breakout
            if obv > obv_signal and obv_roc_20 > 8 and volume_ratio is not None and volume_ratio > 1.3:
                # Check if price is also breaking out
                if price_change_10d is not None and price_change_10d > 3:
                    stop_loss = current_price - (1.5 * atr)
                    target = current_price * 1.12
                    signals.append(EntrySignal(
                        signal_type=SignalType.VOLUME_SURGE,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH,
                        rationale=f"OBV momentum surge: {obv_roc_20:.1f}% OBV increase with {volume_ratio:.1f}x volume confirms breakout",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=12.0,
                        expected_holding_days=20,
                        volume_confirmation=True,
                        trend_alignment=trend_bullish,
                    ))

        # 11. Bollinger Band Squeeze Breakout (low volatility then breakout)
        bb_width = self._get_indicator(price_data, 'BB_Width', indicators)
        bb_middle = self._get_indicator(price_data, 'BB_Middle', indicators)
        bb_position = self._get_indicator(price_data, 'BB_Position', indicators)

        if bb_width is not None and bb_middle is not None:
            # Calculate BB Width percentile (squeeze detection)
            # Low BB Width indicates consolidation - potential breakout coming
            avg_bb_width = self._get_rolling_avg(price_data, 'BB_Width', 50)

            if avg_bb_width is not None and bb_width < avg_bb_width * 0.6:
                # Squeeze detected - BB Width below 60% of 50-day average
                # Check for breakout direction
                if current_price > bb_middle and volume_ratio is not None and volume_ratio > 1.2:
                    stop_loss = bb_middle * 0.98  # Just below middle band
                    target = bb_upper if bb_upper else current_price * 1.08
                    signals.append(EntrySignal(
                        signal_type=SignalType.BB_SQUEEZE_BREAKOUT,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH if volume_ratio > 1.5 else SignalConfidence.MEDIUM,
                        rationale=f"BB squeeze breakout: width at {(bb_width/avg_bb_width*100):.0f}% of avg, breaking above middle band with volume",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=((target - current_price) / current_price) * 100,
                        expected_holding_days=15,
                        volume_confirmation=True,
                        trend_alignment=trend_bullish,
                    ))

        # 12. Bollinger Band Mean Reversion (oversold bounce toward middle)
        if bb_lower is not None and bb_middle is not None and bb_position is not None:
            # Price near lower band with momentum turning up
            if bb_position < 0.2:  # In lower 20% of band
                prev_bb_position = self._get_prev_indicator(price_data, 'BB_Position', 1)
                if prev_bb_position is not None and bb_position > prev_bb_position:
                    # Bouncing off lower band
                    stop_loss = bb_lower * 0.97
                    target = bb_middle  # Target middle band
                    signals.append(EntrySignal(
                        signal_type=SignalType.BB_MEAN_REVERSION,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"BB mean reversion: price at {bb_position*100:.0f}% of band, reverting to middle (${bb_middle:.2f})",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=((target - current_price) / current_price) * 100,
                        expected_holding_days=10,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 0.8,
                        trend_alignment=False,  # Mean reversion can be counter-trend
                    ))

        # 13. Stochastic Oversold Bullish Cross
        if stoch_k is not None:
            stoch_d = self._get_indicator(price_data, 'Stoch_D', indicators)
            if stoch_d is not None and stoch_k < 20 and stoch_d < 20:
                # Both in oversold territory
                prev_stoch_k = self._get_prev_indicator(price_data, 'Stoch_K', 1)
                prev_stoch_d = self._get_prev_indicator(price_data, 'Stoch_D', 1)

                if prev_stoch_k is not None and prev_stoch_d is not None:
                    if prev_stoch_k < prev_stoch_d and stoch_k > stoch_d:
                        # Bullish crossover in oversold zone
                        stop_loss = current_price - (2 * atr)
                        target = current_price * 1.08
                        signals.append(EntrySignal(
                            signal_type=SignalType.STOCHASTIC_OVERSOLD,
                            price_level=current_price,
                            confidence=SignalConfidence.HIGH if stoch_k < 15 else SignalConfidence.MEDIUM,
                            rationale=f"Stochastic bullish cross in oversold: %K({stoch_k:.1f}) crossed above %D({stoch_d:.1f})",
                            risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                            stop_loss=stop_loss,
                            stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                            target_price=target,
                            target_pct=8.0,
                            expected_holding_days=12,
                            volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                            trend_alignment=trend_bullish,
                        ))

        # 14. Williams %R Oversold Reversal
        williams_r = self._get_indicator(price_data, 'Williams_R', indicators)
        if williams_r is not None and williams_r < -80:
            prev_williams = self._get_prev_indicator(price_data, 'Williams_R', 1)
            if prev_williams is not None and williams_r > prev_williams:
                # Turning up from oversold
                stop_loss = current_price - (1.5 * atr)
                target = current_price * 1.06
                signals.append(EntrySignal(
                    signal_type=SignalType.OVERSOLD_REVERSAL,
                    price_level=current_price,
                    confidence=SignalConfidence.MEDIUM,
                    rationale=f"Williams %R reversal from oversold: {williams_r:.1f} (turning up)",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=6.0,
                    expected_holding_days=10,
                    volume_confirmation=volume_ratio is not None and volume_ratio > 0.8,
                    trend_alignment=trend_bullish,
                ))

        # 15. EMA 8/21 Bullish Cross (Short-term swing trading)
        ema_8 = self._get_indicator(price_data, 'EMA_8', indicators)
        ema_21 = self._get_indicator(price_data, 'EMA_21', indicators)

        if ema_8 is not None and ema_21 is not None:
            prev_ema_8 = self._get_prev_indicator(price_data, 'EMA_8', 1)
            prev_ema_21 = self._get_prev_indicator(price_data, 'EMA_21', 1)

            if prev_ema_8 is not None and prev_ema_21 is not None:
                if prev_ema_8 < prev_ema_21 and ema_8 > ema_21:
                    # Short-term bullish crossover
                    stop_loss = ema_21 * 0.98
                    target = current_price * 1.08
                    signals.append(EntrySignal(
                        signal_type=SignalType.EMA_CROSS_SHORT,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"EMA 8/21 bullish cross: EMA(8)=${ema_8:.2f} crossed above EMA(21)=${ema_21:.2f}",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=8.0,
                        expected_holding_days=10,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                        trend_alignment=trend_bullish,
                    ))

        # 16. EMA 12/26 Bullish Cross (Medium-term, MACD-based)
        ema_12 = self._get_indicator(price_data, 'EMA_12', indicators)
        ema_26 = self._get_indicator(price_data, 'EMA_26', indicators)

        if ema_12 is not None and ema_26 is not None:
            prev_ema_12 = self._get_prev_indicator(price_data, 'EMA_12', 1)
            prev_ema_26 = self._get_prev_indicator(price_data, 'EMA_26', 1)

            if prev_ema_12 is not None and prev_ema_26 is not None:
                if prev_ema_12 < prev_ema_26 and ema_12 > ema_26:
                    # Medium-term bullish crossover (confirms MACD)
                    stop_loss = ema_26 * 0.97
                    target = current_price * 1.12
                    signals.append(EntrySignal(
                        signal_type=SignalType.EMA_CROSS_MEDIUM,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH if trend_bullish else SignalConfidence.MEDIUM,
                        rationale=f"EMA 12/26 bullish cross: EMA(12)=${ema_12:.2f} crossed above EMA(26)=${ema_26:.2f}",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=12.0,
                        expected_holding_days=25,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                        trend_alignment=trend_bullish,
                    ))

        # 17. EMA 50/200 Bullish Cross (Long-term Golden Cross - already have SMA version)
        ema_50 = self._get_indicator(price_data, 'EMA_50', indicators)
        ema_200 = self._get_indicator(price_data, 'EMA_200', indicators)

        if ema_50 is not None and ema_200 is not None:
            prev_ema_50 = self._get_prev_indicator(price_data, 'EMA_50', 1)
            prev_ema_200 = self._get_prev_indicator(price_data, 'EMA_200', 1)

            if prev_ema_50 is not None and prev_ema_200 is not None:
                if prev_ema_50 < prev_ema_200 and ema_50 > ema_200:
                    # Long-term bullish crossover (EMA Golden Cross)
                    stop_loss = ema_200 * 0.95
                    target = current_price * 1.20
                    signals.append(EntrySignal(
                        signal_type=SignalType.EMA_CROSS_LONG,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH,
                        rationale=f"EMA Golden Cross: EMA(50)=${ema_50:.2f} crossed above EMA(200)=${ema_200:.2f}",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=20.0,
                        expected_holding_days=90,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                        trend_alignment=True,  # EMA Golden Cross is bullish by definition
                    ))

        # 18. ADX Trend Start (strong trend beginning with bullish direction)
        adx = self._get_indicator(price_data, 'ADX_14', indicators)
        plus_di = self._get_indicator(price_data, 'Plus_DI', indicators)
        minus_di = self._get_indicator(price_data, 'Minus_DI', indicators)

        if adx is not None and plus_di is not None and minus_di is not None:
            prev_adx = self._get_prev_indicator(price_data, 'ADX_14', 1)

            if prev_adx is not None:
                # ADX crossing above 25 with bullish direction (+DI > -DI)
                if prev_adx < 25 and adx >= 25 and plus_di > minus_di:
                    stop_loss = current_price - (2 * atr)
                    target = current_price * 1.12
                    signals.append(EntrySignal(
                        signal_type=SignalType.ADX_TREND_START,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH if adx > 30 else SignalConfidence.MEDIUM,
                        rationale=f"ADX trend start: ADX={adx:.1f} (strong trend), +DI={plus_di:.1f} > -DI={minus_di:.1f}",
                        risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                        stop_loss=stop_loss,
                        stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                        target_price=target,
                        target_pct=12.0,
                        expected_holding_days=20,
                        volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                        trend_alignment=True,
                    ))

        # 19. +DI/-DI Bullish Crossover (directional change)
        if plus_di is not None and minus_di is not None and adx is not None:
            prev_plus_di = self._get_prev_indicator(price_data, 'Plus_DI', 1)
            prev_minus_di = self._get_prev_indicator(price_data, 'Minus_DI', 1)

            if prev_plus_di is not None and prev_minus_di is not None:
                if prev_plus_di < prev_minus_di and plus_di > minus_di:
                    # Bullish DI crossover in a trending market
                    if adx > 20:  # Only signal if there's some trend
                        stop_loss = current_price - (2 * atr)
                        target = current_price * 1.10
                        signals.append(EntrySignal(
                            signal_type=SignalType.DI_CROSS_BULLISH,
                            price_level=current_price,
                            confidence=SignalConfidence.HIGH if adx > 25 else SignalConfidence.MEDIUM,
                            rationale=f"Bullish DI cross: +DI={plus_di:.1f} crossed above -DI={minus_di:.1f}, ADX={adx:.1f}",
                            risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                            stop_loss=stop_loss,
                            stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                            target_price=target,
                            target_pct=10.0,
                            expected_holding_days=15,
                            volume_confirmation=volume_ratio is not None and volume_ratio > 1.0,
                            trend_alignment=True,
                        ))

        # 20. MFI Oversold Reversal (volume-weighted RSI)
        mfi = self._get_indicator(price_data, 'MFI_14', indicators)

        if mfi is not None and mfi < 20:
            prev_mfi = self._get_prev_indicator(price_data, 'MFI_14', 1)
            if prev_mfi is not None and mfi > prev_mfi:
                # MFI turning up from oversold
                stop_loss = current_price - (2 * atr)
                target = current_price * 1.08
                signals.append(EntrySignal(
                    signal_type=SignalType.MFI_OVERSOLD,
                    price_level=current_price,
                    confidence=SignalConfidence.HIGH if mfi < 15 else SignalConfidence.MEDIUM,
                    rationale=f"MFI oversold reversal: MFI={mfi:.1f} (volume-weighted) turning up",
                    risk_reward_ratio=self._calculate_rr(current_price, stop_loss, target),
                    stop_loss=stop_loss,
                    stop_loss_pct=((current_price - stop_loss) / current_price) * 100,
                    target_price=target,
                    target_pct=8.0,
                    expected_holding_days=12,
                    volume_confirmation=True,  # MFI inherently uses volume
                    trend_alignment=trend_bullish,
                ))

        # Sort by confidence (HIGH first) and risk/reward ratio
        signals.sort(key=lambda s: (
            0 if s.confidence == SignalConfidence.HIGH else 1 if s.confidence == SignalConfidence.MEDIUM else 2,
            -s.risk_reward_ratio
        ))

        return signals

    def generate_exit_signals(
        self,
        price_data: pd.DataFrame,
        indicators: Optional[Dict[str, Any]] = None,
        position_info: Optional[Dict[str, Any]] = None,
    ) -> List[ExitSignal]:
        """
        Generate exit signals for existing or hypothetical positions.

        Args:
            price_data: DataFrame with OHLCV and calculated indicators
            indicators: Pre-calculated indicator values
            position_info: Optional position details (entry_price, stop_loss, target)

        Returns:
            List of ExitSignal objects
        """
        signals = []

        if price_data is None or len(price_data) < 20:
            return signals

        latest = price_data.iloc[-1]
        current_price = float(latest.get('Close', latest.get('close', 0)))

        if current_price <= 0:
            return signals

        # Get indicators
        rsi = self._get_indicator(price_data, 'RSI_14', indicators)
        macd = self._get_indicator(price_data, 'MACD', indicators)
        macd_signal = self._get_indicator(price_data, 'MACD_Signal', indicators)
        sma_20 = self._get_indicator(price_data, 'SMA_20', indicators)
        sma_50 = self._get_indicator(price_data, 'SMA_50', indicators)
        sma_200 = self._get_indicator(price_data, 'SMA_200', indicators)
        bb_upper = self._get_indicator(price_data, 'BB_Upper', indicators)
        resistance_1 = self._get_indicator(price_data, 'Resistance_1', indicators)
        volume_ratio = self._get_indicator(price_data, 'Volume_Ratio', indicators)

        # Position info
        entry_price = position_info.get('entry_price', current_price * 0.9) if position_info else None
        stop_loss = position_info.get('stop_loss') if position_info else None
        target_price = position_info.get('target_price') if position_info else None

        # 1. RSI Overbought
        if rsi is not None and rsi > self.RSI_OVERBOUGHT:
            signals.append(ExitSignal(
                signal_type=SignalType.OVERBOUGHT,
                price_level=current_price,
                confidence=SignalConfidence.HIGH if rsi > 80 else SignalConfidence.MEDIUM,
                rationale=f"RSI at {rsi:.1f} (overbought territory)",
                urgency="staged" if rsi < 80 else "immediate",
                partial_exit_pct=50.0 if rsi < 80 else 100.0,
            ))

        # 2. MACD Bearish Crossover
        if macd is not None and macd_signal is not None:
            prev_macd = self._get_prev_indicator(price_data, 'MACD', 1)
            prev_signal = self._get_prev_indicator(price_data, 'MACD_Signal', 1)

            if prev_macd is not None and prev_signal is not None:
                if prev_macd > prev_signal and macd < macd_signal:
                    signals.append(ExitSignal(
                        signal_type=SignalType.MOMENTUM_LOSS,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"MACD bearish crossover (MACD: {macd:.2f} < Signal: {macd_signal:.2f})",
                        urgency="staged",
                        partial_exit_pct=50.0,
                    ))

        # 3. Death Cross (50 SMA crosses below 200 SMA)
        if sma_50 is not None and sma_200 is not None:
            prev_sma_50 = self._get_prev_indicator(price_data, 'SMA_50', 1)
            prev_sma_200 = self._get_prev_indicator(price_data, 'SMA_200', 1)

            if prev_sma_50 is not None and prev_sma_200 is not None:
                if prev_sma_50 > prev_sma_200 and sma_50 < sma_200:
                    signals.append(ExitSignal(
                        signal_type=SignalType.DEATH_CROSS,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH,
                        rationale=f"Death Cross: 50 SMA crossed below 200 SMA",
                        urgency="immediate",
                        partial_exit_pct=100.0,
                    ))

        # 4. Resistance Rejection
        if resistance_1 is not None:
            distance_to_resistance = (resistance_1 - current_price) / current_price * 100
            if -2 < distance_to_resistance < 2:  # Near resistance
                # Check for rejection (price failing at resistance)
                prev_close = self._get_prev_indicator(price_data, 'Close', 1)
                if prev_close is not None and current_price < prev_close:
                    signals.append(ExitSignal(
                        signal_type=SignalType.RESISTANCE,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"Price rejected at resistance ${resistance_1:.2f}",
                        urgency="watch",
                        partial_exit_pct=30.0,
                    ))

        # 5. Bollinger Upper Band Touch (potential reversal)
        if bb_upper is not None:
            if current_price > bb_upper:
                signals.append(ExitSignal(
                    signal_type=SignalType.OVERBOUGHT,
                    price_level=current_price,
                    confidence=SignalConfidence.MEDIUM,
                    rationale=f"Price above Bollinger upper band (${bb_upper:.2f})",
                    urgency="staged",
                    partial_exit_pct=30.0,
                ))

        # 6. Stop Loss Hit
        if stop_loss is not None and current_price <= stop_loss:
            signals.append(ExitSignal(
                signal_type=SignalType.STOP_LOSS,
                price_level=current_price,
                confidence=SignalConfidence.HIGH,
                rationale=f"Stop loss triggered at ${stop_loss:.2f}",
                urgency="immediate",
                partial_exit_pct=100.0,
            ))

        # 7. Target Hit
        if target_price is not None and current_price >= target_price:
            signals.append(ExitSignal(
                signal_type=SignalType.TARGET_HIT,
                price_level=current_price,
                confidence=SignalConfidence.HIGH,
                rationale=f"Target price ${target_price:.2f} reached",
                urgency="staged",
                partial_exit_pct=75.0,  # Take 75% profits, let rest ride
            ))

        # 8. Volume Divergence (price up but volume declining)
        if volume_ratio is not None and volume_ratio < 0.7:
            # Check if price is up but volume is weak
            price_change_5d = self._get_indicator(price_data, 'Price_Change_5D', indicators)
            if price_change_5d is not None and price_change_5d > 3:
                signals.append(ExitSignal(
                    signal_type=SignalType.VOLUME_DIVERGENCE,
                    price_level=current_price,
                    confidence=SignalConfidence.LOW,
                    rationale=f"Price up {price_change_5d:.1f}% but volume at {volume_ratio:.1f}x average (weak)",
                    urgency="watch",
                    partial_exit_pct=25.0,
                ))

        # 9. OBV Distribution Signal (AOBV bearish crossover)
        obv_trend = self._get_indicator(price_data, 'OBV_Trend', indicators)
        aobv_20 = self._get_indicator(price_data, 'AOBV_20', indicators)
        aobv_50 = self._get_indicator(price_data, 'AOBV_50', indicators)
        obv_histogram = self._get_indicator(price_data, 'OBV_Histogram', indicators)

        if obv_trend is not None and aobv_20 is not None and aobv_50 is not None:
            # Check for AOBV bearish crossover (AOBV_20 crosses below AOBV_50)
            prev_aobv_20 = self._get_prev_indicator(price_data, 'AOBV_20', 1)
            prev_aobv_50 = self._get_prev_indicator(price_data, 'AOBV_50', 1)

            if prev_aobv_20 is not None and prev_aobv_50 is not None:
                if prev_aobv_20 > prev_aobv_50 and aobv_20 < aobv_50:
                    # Bearish AOBV crossover - institutional selling
                    signals.append(ExitSignal(
                        signal_type=SignalType.OBV_DISTRIBUTION,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH if obv_histogram and obv_histogram < 0 else SignalConfidence.MEDIUM,
                        rationale=f"OBV distribution: AOBV(20) crossed below AOBV(50) - institutional selling",
                        urgency="staged",
                        partial_exit_pct=50.0,
                    ))

        # 10. OBV Bearish Divergence (price up, OBV down - distribution/exhaustion)
        obv_roc_20 = self._get_indicator(price_data, 'OBV_ROC_20', indicators)
        price_change_10d = self._get_indicator(price_data, 'Price_Change_10D', indicators)

        if obv_roc_20 is not None and price_change_10d is not None:
            # Bearish divergence: price rising but OBV falling (smart money distributing)
            if price_change_10d > 5 and obv_roc_20 < -5:
                signals.append(ExitSignal(
                    signal_type=SignalType.OBV_BEARISH_DIVERGENCE,
                    price_level=current_price,
                    confidence=SignalConfidence.HIGH if obv_roc_20 < -10 else SignalConfidence.MEDIUM,
                    rationale=f"Bearish OBV divergence: price up {price_change_10d:.1f}% but OBV down {obv_roc_20:.1f}% - distribution",
                    urgency="staged" if obv_roc_20 > -10 else "immediate",
                    partial_exit_pct=50.0 if obv_roc_20 > -10 else 75.0,
                ))

        # 11. OBV Momentum Loss (OBV breaks below signal line after uptrend)
        obv = self._get_indicator(price_data, 'OBV', indicators)
        obv_signal = self._get_indicator(price_data, 'OBV_Signal', indicators)

        if obv is not None and obv_signal is not None:
            prev_obv = self._get_prev_indicator(price_data, 'OBV', 1)
            prev_obv_signal = self._get_prev_indicator(price_data, 'OBV_Signal', 1)

            if prev_obv is not None and prev_obv_signal is not None:
                # OBV crossing below signal line (like MACD bearish cross)
                if prev_obv > prev_obv_signal and obv < obv_signal:
                    signals.append(ExitSignal(
                        signal_type=SignalType.MOMENTUM_LOSS,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"OBV crossed below signal line - buying momentum fading",
                        urgency="watch",
                        partial_exit_pct=30.0,
                    ))

        # 12. Bollinger Band Upper Rejection (price rejected at upper band)
        bb_upper = self._get_indicator(price_data, 'BB_Upper', indicators)
        bb_position = self._get_indicator(price_data, 'BB_Position', indicators)
        bb_middle = self._get_indicator(price_data, 'BB_Middle', indicators)

        if bb_upper is not None and bb_position is not None:
            # Price in upper 20% of band and turning down
            if bb_position > 0.8:
                prev_bb_position = self._get_prev_indicator(price_data, 'BB_Position', 1)
                prev_close = self._get_prev_indicator(price_data, 'Close', 1)

                if prev_bb_position is not None and prev_close is not None:
                    if bb_position < prev_bb_position and current_price < prev_close:
                        # Price turning down from upper band
                        signals.append(ExitSignal(
                            signal_type=SignalType.BB_UPPER_REJECTION,
                            price_level=current_price,
                            confidence=SignalConfidence.MEDIUM,
                            rationale=f"BB upper rejection: price at {bb_position*100:.0f}% of band, turning down from ${bb_upper:.2f}",
                            urgency="staged",
                            partial_exit_pct=40.0,
                        ))

        # 13. Bollinger Band Mean Reversion Exit (reached middle band target)
        if bb_middle is not None and entry_price is not None:
            # If we entered on lower band bounce, middle band is target
            if current_price >= bb_middle * 0.99 and entry_price < bb_middle * 0.95:
                signals.append(ExitSignal(
                    signal_type=SignalType.TARGET_HIT,
                    price_level=current_price,
                    confidence=SignalConfidence.MEDIUM,
                    rationale=f"BB mean reversion target reached: middle band at ${bb_middle:.2f}",
                    urgency="staged",
                    partial_exit_pct=50.0,
                ))

        # 14. Stochastic Overbought Bearish Cross
        stoch_k = self._get_indicator(price_data, 'Stoch_K', indicators)
        stoch_d = self._get_indicator(price_data, 'Stoch_D', indicators)

        if stoch_k is not None and stoch_d is not None:
            if stoch_k > 80 and stoch_d > 80:
                # Both in overbought territory
                prev_stoch_k = self._get_prev_indicator(price_data, 'Stoch_K', 1)
                prev_stoch_d = self._get_prev_indicator(price_data, 'Stoch_D', 1)

                if prev_stoch_k is not None and prev_stoch_d is not None:
                    if prev_stoch_k > prev_stoch_d and stoch_k < stoch_d:
                        # Bearish crossover in overbought zone
                        signals.append(ExitSignal(
                            signal_type=SignalType.STOCHASTIC_OVERBOUGHT,
                            price_level=current_price,
                            confidence=SignalConfidence.HIGH if stoch_k > 85 else SignalConfidence.MEDIUM,
                            rationale=f"Stochastic bearish cross in overbought: %K({stoch_k:.1f}) crossed below %D({stoch_d:.1f})",
                            urgency="staged" if stoch_k < 85 else "immediate",
                            partial_exit_pct=50.0,
                        ))

        # 15. Williams %R Overbought Exit
        williams_r = self._get_indicator(price_data, 'Williams_R', indicators)
        if williams_r is not None and williams_r > -20:
            prev_williams = self._get_prev_indicator(price_data, 'Williams_R', 1)
            if prev_williams is not None and williams_r < prev_williams:
                # Turning down from overbought
                signals.append(ExitSignal(
                    signal_type=SignalType.OVERBOUGHT,
                    price_level=current_price,
                    confidence=SignalConfidence.MEDIUM,
                    rationale=f"Williams %R reversal from overbought: {williams_r:.1f} (turning down)",
                    urgency="watch",
                    partial_exit_pct=30.0,
                ))

        # 16. EMA 8/21 Bearish Cross (Short-term exit)
        ema_8 = self._get_indicator(price_data, 'EMA_8', indicators)
        ema_21 = self._get_indicator(price_data, 'EMA_21', indicators)

        if ema_8 is not None and ema_21 is not None:
            prev_ema_8 = self._get_prev_indicator(price_data, 'EMA_8', 1)
            prev_ema_21 = self._get_prev_indicator(price_data, 'EMA_21', 1)

            if prev_ema_8 is not None and prev_ema_21 is not None:
                if prev_ema_8 > prev_ema_21 and ema_8 < ema_21:
                    # Short-term bearish crossover
                    signals.append(ExitSignal(
                        signal_type=SignalType.EMA_CROSS_BEARISH_SHORT,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"EMA 8/21 bearish cross: EMA(8)=${ema_8:.2f} crossed below EMA(21)=${ema_21:.2f}",
                        urgency="staged",
                        partial_exit_pct=40.0,
                    ))

        # 17. EMA 12/26 Bearish Cross (Medium-term exit, confirms MACD)
        ema_12 = self._get_indicator(price_data, 'EMA_12', indicators)
        ema_26 = self._get_indicator(price_data, 'EMA_26', indicators)

        if ema_12 is not None and ema_26 is not None:
            prev_ema_12 = self._get_prev_indicator(price_data, 'EMA_12', 1)
            prev_ema_26 = self._get_prev_indicator(price_data, 'EMA_26', 1)

            if prev_ema_12 is not None and prev_ema_26 is not None:
                if prev_ema_12 > prev_ema_26 and ema_12 < ema_26:
                    # Medium-term bearish crossover
                    signals.append(ExitSignal(
                        signal_type=SignalType.EMA_CROSS_BEARISH_MEDIUM,
                        price_level=current_price,
                        confidence=SignalConfidence.HIGH,
                        rationale=f"EMA 12/26 bearish cross: EMA(12)=${ema_12:.2f} crossed below EMA(26)=${ema_26:.2f}",
                        urgency="immediate",
                        partial_exit_pct=60.0,
                    ))

        # 18. ADX Trend End (trend exhaustion)
        adx = self._get_indicator(price_data, 'ADX_14', indicators)
        plus_di = self._get_indicator(price_data, 'Plus_DI', indicators)
        minus_di = self._get_indicator(price_data, 'Minus_DI', indicators)

        if adx is not None:
            prev_adx = self._get_prev_indicator(price_data, 'ADX_14', 1)

            if prev_adx is not None:
                # ADX falling below 20 (trend losing strength)
                if prev_adx >= 20 and adx < 20:
                    signals.append(ExitSignal(
                        signal_type=SignalType.ADX_TREND_END,
                        price_level=current_price,
                        confidence=SignalConfidence.MEDIUM,
                        rationale=f"ADX trend exhaustion: ADX={adx:.1f} fell below 20 (was {prev_adx:.1f})",
                        urgency="staged",
                        partial_exit_pct=40.0,
                    ))

        # 19. -DI/+DI Bearish Crossover (directional change)
        if plus_di is not None and minus_di is not None and adx is not None:
            prev_plus_di = self._get_prev_indicator(price_data, 'Plus_DI', 1)
            prev_minus_di = self._get_prev_indicator(price_data, 'Minus_DI', 1)

            if prev_plus_di is not None and prev_minus_di is not None:
                if prev_minus_di < prev_plus_di and minus_di > plus_di:
                    # Bearish DI crossover
                    if adx > 20:  # Only signal if there's trend strength
                        signals.append(ExitSignal(
                            signal_type=SignalType.DI_CROSS_BEARISH,
                            price_level=current_price,
                            confidence=SignalConfidence.HIGH if adx > 25 else SignalConfidence.MEDIUM,
                            rationale=f"Bearish DI cross: -DI={minus_di:.1f} crossed above +DI={plus_di:.1f}, ADX={adx:.1f}",
                            urgency="immediate" if adx > 25 else "staged",
                            partial_exit_pct=60.0 if adx > 25 else 40.0,
                        ))

        # 20. MFI Overbought Exit
        mfi = self._get_indicator(price_data, 'MFI_14', indicators)

        if mfi is not None and mfi > 80:
            prev_mfi = self._get_prev_indicator(price_data, 'MFI_14', 1)
            if prev_mfi is not None and mfi < prev_mfi:
                # MFI turning down from overbought
                signals.append(ExitSignal(
                    signal_type=SignalType.MFI_OVERBOUGHT,
                    price_level=current_price,
                    confidence=SignalConfidence.HIGH if mfi > 85 else SignalConfidence.MEDIUM,
                    rationale=f"MFI overbought reversal: MFI={mfi:.1f} (volume-weighted) turning down",
                    urgency="staged" if mfi < 85 else "immediate",
                    partial_exit_pct=50.0,
                ))

        # Sort by urgency (immediate first) and confidence
        urgency_order = {"immediate": 0, "staged": 1, "watch": 2}
        signals.sort(key=lambda s: (
            urgency_order.get(s.urgency, 3),
            0 if s.confidence == SignalConfidence.HIGH else 1 if s.confidence == SignalConfidence.MEDIUM else 2,
        ))

        return signals

    def calculate_optimal_entry_zone(
        self,
        current_price: float,
        fair_value: float,
        support_levels: List[float],
        resistance_levels: List[float],
        volatility: float,
        atr: Optional[float] = None,
    ) -> OptimalEntryZone:
        """
        Calculate optimal entry zone with bounds and timing.

        Args:
            current_price: Current stock price
            fair_value: Calculated fair value
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
            volatility: Annualized volatility (as decimal, e.g., 0.25 for 25%)
            atr: Average True Range (optional)

        Returns:
            OptimalEntryZone with entry recommendations
        """
        if current_price <= 0:
            raise ValueError("Current price must be positive")

        # Calculate upside/downside
        upside_pct = ((fair_value - current_price) / current_price) * 100

        # Find nearest support and resistance
        supports = sorted([s for s in support_levels if s < current_price], reverse=True)
        resistances = sorted([r for r in resistance_levels if r > current_price])

        nearest_support = supports[0] if supports else current_price * 0.95
        nearest_resistance = resistances[0] if resistances else current_price * 1.10

        # Default ATR if not provided
        if atr is None:
            atr = current_price * volatility / 16  # Approximate daily volatility

        # Calculate entry zone bounds
        # Lower bound: Near support level or current price minus 1 ATR
        lower_bound = max(nearest_support, current_price - atr)

        # Upper bound: Current price or slight premium for momentum
        upper_bound = min(current_price + (0.5 * atr), nearest_resistance * 0.98)

        # Ideal entry: Weighted toward support
        ideal_entry = lower_bound + (upper_bound - lower_bound) * 0.3

        # Determine timing
        if upside_pct > 20:
            if current_price < nearest_support * 1.02:
                timing = SignalTiming.IMMEDIATE
            else:
                timing = SignalTiming.WAIT_PULLBACK
        elif upside_pct > 10:
            timing = SignalTiming.SCALE_IN
        elif upside_pct < -10:
            timing = SignalTiming.AVOID_NOW
        else:
            timing = SignalTiming.WAIT_PULLBACK

        # Determine scaling strategy based on conviction and volatility
        if upside_pct > 25 and volatility < 0.3:
            scaling = ScalingStrategy.FULL_POSITION
        elif upside_pct > 15:
            scaling = ScalingStrategy.SCALE_IN
        elif volatility > 0.4:
            scaling = ScalingStrategy.DCA
        else:
            scaling = ScalingStrategy.SCALE_IN

        # Confidence based on confluence
        if upside_pct > 20 and current_price < nearest_support * 1.05:
            confidence = SignalConfidence.HIGH
        elif upside_pct > 10:
            confidence = SignalConfidence.MEDIUM
        else:
            confidence = SignalConfidence.LOW

        # Position sizing based on volatility and conviction
        base_allocation = 3.0  # Base 3% of portfolio
        volatility_adjustment = max(0.5, 1.0 - volatility)  # Reduce for high vol
        conviction_adjustment = 1.0 + (upside_pct / 50)  # Increase for high upside

        recommended_allocation = min(5.0, base_allocation * volatility_adjustment * conviction_adjustment)
        max_position = min(8.0, recommended_allocation * 1.5)

        # Generate rationale
        rationale_parts = []
        if upside_pct > 15:
            rationale_parts.append(f"{upside_pct:.0f}% upside to fair value ${fair_value:.2f}")
        if nearest_support:
            rationale_parts.append(f"Support at ${nearest_support:.2f}")
        if volatility > 0.35:
            rationale_parts.append(f"High volatility ({volatility*100:.0f}%)")

        rationale = "; ".join(rationale_parts) if rationale_parts else "Standard entry conditions"

        return OptimalEntryZone(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            ideal_entry=ideal_entry,
            timing=timing,
            scaling_strategy=scaling,
            confidence=confidence,
            rationale=rationale,
            recommended_allocation_pct=recommended_allocation,
            max_position_size_pct=max_position,
        )

    def score_entry_signal(
        self,
        signal: EntrySignal,
        price_data: pd.DataFrame,
        valuation: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Score an entry signal from 0-100 based on multiple factors.

        Returns:
            Composite score from 0-100
        """
        score = 50.0  # Base score

        # Confidence adjustment
        if signal.confidence == SignalConfidence.HIGH:
            score += 20
        elif signal.confidence == SignalConfidence.MEDIUM:
            score += 10

        # Risk/reward adjustment
        if signal.risk_reward_ratio >= 3.0:
            score += 15
        elif signal.risk_reward_ratio >= 2.0:
            score += 10
        elif signal.risk_reward_ratio >= 1.5:
            score += 5
        elif signal.risk_reward_ratio < 1.0:
            score -= 15

        # Volume confirmation
        if signal.volume_confirmation:
            score += 10

        # Trend alignment
        if signal.trend_alignment:
            score += 10

        # Valuation support
        if valuation:
            upside = valuation.get('upside_pct', 0)
            if upside > 20:
                score += 15
            elif upside > 10:
                score += 10
            elif upside < -10:
                score -= 20

        return max(0, min(100, score))

    def _get_indicator(
        self,
        df: pd.DataFrame,
        name: str,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Get indicator value from DataFrame or indicators dict"""
        # Try indicators dict first
        if indicators and name in indicators:
            return float(indicators[name])

        # Try DataFrame
        if name in df.columns:
            val = df[name].iloc[-1]
            if pd.notna(val):
                return float(val)

        # Try lowercase
        name_lower = name.lower()
        if name_lower in df.columns:
            val = df[name_lower].iloc[-1]
            if pd.notna(val):
                return float(val)

        return None

    def _get_prev_indicator(
        self,
        df: pd.DataFrame,
        name: str,
        periods_back: int = 1,
    ) -> Optional[float]:
        """Get previous indicator value"""
        if name in df.columns and len(df) > periods_back:
            val = df[name].iloc[-(periods_back + 1)]
            if pd.notna(val):
                return float(val)
        return None

    def _get_rolling_avg(
        self,
        df: pd.DataFrame,
        name: str,
        window: int = 50,
    ) -> Optional[float]:
        """Get rolling average of an indicator"""
        if name in df.columns and len(df) >= window:
            avg = df[name].tail(window).mean()
            if pd.notna(avg):
                return float(avg)
        return None

    def _calculate_rr(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0.0

    def _is_trend_bullish(
        self,
        price: float,
        sma_20: Optional[float],
        sma_50: Optional[float],
        sma_200: Optional[float],
    ) -> bool:
        """Check if trend is bullish based on moving average alignment"""
        bullish_count = 0

        if sma_20 is not None and price > sma_20:
            bullish_count += 1
        if sma_50 is not None and price > sma_50:
            bullish_count += 1
        if sma_200 is not None and price > sma_200:
            bullish_count += 1
        if sma_20 is not None and sma_50 is not None and sma_20 > sma_50:
            bullish_count += 1
        if sma_50 is not None and sma_200 is not None and sma_50 > sma_200:
            bullish_count += 1

        return bullish_count >= 3


# Singleton instance
_engine: Optional[EntryExitEngine] = None


def get_entry_exit_engine() -> EntryExitEngine:
    """Get singleton EntryExitEngine instance"""
    global _engine
    if _engine is None:
        _engine = EntryExitEngine()
    return _engine
