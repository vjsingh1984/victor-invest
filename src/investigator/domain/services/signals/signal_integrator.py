"""
Signal Integrator Service

Bridges LLM-based technical analysis with programmatic EntryExitEngine.
Validates LLM signals and provides standardized output for PDF reports.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from investigator.domain.services.signals.entry_exit_engine import (
    EntryExitEngine,
    EntrySignal,
    ExitSignal,
    OptimalEntryZone,
    ScalingStrategy,
    SignalConfidence,
    SignalTiming,
    SignalType,
    get_entry_exit_engine,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSignals:
    """Combined signals from LLM and programmatic analysis"""

    # Programmatic signals (always available)
    programmatic_entry_signals: List[EntrySignal]
    programmatic_exit_signals: List[ExitSignal]
    programmatic_entry_zone: Optional[OptimalEntryZone]

    # LLM signals (when available)
    llm_entry_signals: Optional[List[Dict]]
    llm_exit_signals: Optional[List[Dict]]
    llm_entry_zone: Optional[Dict]

    # Merged/validated signals (best of both)
    final_entry_signals: List[EntrySignal]
    final_exit_signals: List[ExitSignal]
    final_entry_zone: Optional[OptimalEntryZone]

    # Metadata
    signal_agreement_score: float  # 0-1: how much LLM and programmatic agree
    confidence_boost: bool  # True if LLM confirms programmatic signals


class SignalIntegrator:
    """
    Integrates LLM-based technical analysis with programmatic signal generation.

    The integrator:
    1. Runs programmatic signal generation (fast, consistent)
    2. Parses LLM signals from technical analysis
    3. Validates LLM signals against programmatic checks
    4. Merges signals, boosting confidence when they agree
    5. Produces standardized output for PDF reports
    """

    def __init__(self):
        self.engine = get_entry_exit_engine()

    def integrate_signals(
        self,
        price_data: pd.DataFrame,
        indicators: Dict[str, Any],
        valuation: Optional[Dict[str, Any]] = None,
        llm_technical_analysis: Optional[Dict[str, Any]] = None,
    ) -> IntegratedSignals:
        """
        Integrate programmatic and LLM-based signals.

        Args:
            price_data: OHLCV DataFrame with technical data
            indicators: Technical indicators dict
            valuation: Optional valuation data (fair_value, etc.)
            llm_technical_analysis: Optional LLM technical analysis response

        Returns:
            IntegratedSignals with merged and validated signals
        """
        # Get current price
        close_col = "close" if "close" in price_data.columns else "Close"
        current_price = float(price_data[close_col].iloc[-1])

        # Build support/resistance from indicators
        support_resistance = self._build_support_resistance(indicators, price_data)

        # Generate programmatic signals
        prog_entry_signals = self.engine.generate_entry_signals(
            price_data=price_data,
            indicators=indicators,
            valuation=valuation or {},
            support_resistance=support_resistance,
        )

        prog_exit_signals = self.engine.generate_exit_signals(
            price_data=price_data,
            indicators=indicators,
            position_info=None,
        )

        # Calculate ATR for entry zone
        atr = indicators.get("atr", indicators.get("ATR", current_price * 0.02))
        volatility = indicators.get("volatility_20d", 0.25)

        prog_entry_zone = self.engine.calculate_optimal_entry_zone(
            current_price=current_price,
            fair_value=valuation.get("blended_fair_value", current_price) if valuation else current_price,
            support_levels=support_resistance.get("support_levels", []),
            resistance_levels=support_resistance.get("resistance_levels", []),
            volatility=volatility,
            atr=atr,
        )

        # Parse LLM signals if available
        llm_entry_signals = None
        llm_exit_signals = None
        llm_entry_zone = None

        if llm_technical_analysis:
            llm_entry_signals = llm_technical_analysis.get("entry_signals", [])
            llm_exit_signals = llm_technical_analysis.get("exit_signals", [])
            llm_entry_zone = llm_technical_analysis.get("optimal_entry_zone")

        # Merge and validate signals
        final_entry_signals, final_exit_signals, final_entry_zone, agreement_score, confidence_boost = (
            self._merge_signals(
                prog_entry_signals,
                prog_exit_signals,
                prog_entry_zone,
                llm_entry_signals,
                llm_exit_signals,
                llm_entry_zone,
                current_price,
            )
        )

        return IntegratedSignals(
            programmatic_entry_signals=prog_entry_signals,
            programmatic_exit_signals=prog_exit_signals,
            programmatic_entry_zone=prog_entry_zone,
            llm_entry_signals=llm_entry_signals,
            llm_exit_signals=llm_exit_signals,
            llm_entry_zone=llm_entry_zone,
            final_entry_signals=final_entry_signals,
            final_exit_signals=final_exit_signals,
            final_entry_zone=final_entry_zone,
            signal_agreement_score=agreement_score,
            confidence_boost=confidence_boost,
        )

    def _build_support_resistance(
        self,
        indicators: Dict[str, Any],
        price_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build support/resistance dict from indicators and price data."""
        close_col = "close" if "close" in price_data.columns else "Close"
        high_col = "high" if "high" in price_data.columns else "High"
        low_col = "low" if "low" in price_data.columns else "Low"

        current_price = float(price_data[close_col].iloc[-1])

        # Extract from indicators if available
        immediate_support = indicators.get("support_1", current_price * 0.95)
        immediate_resistance = indicators.get("resistance_1", current_price * 1.05)

        # Calculate from price data if not in indicators
        if "support_1" not in indicators:
            recent_low = float(price_data[low_col].tail(20).min())
            immediate_support = recent_low

        if "resistance_1" not in indicators:
            recent_high = float(price_data[high_col].tail(20).max())
            immediate_resistance = recent_high

        # Build support levels
        support_levels = [
            immediate_support,
            indicators.get("support_2", immediate_support * 0.97),
            indicators.get("support_3", immediate_support * 0.94),
        ]

        # Build resistance levels
        resistance_levels = [
            immediate_resistance,
            indicators.get("resistance_2", immediate_resistance * 1.03),
            indicators.get("resistance_3", immediate_resistance * 1.06),
        ]

        # Add Bollinger Bands as support/resistance
        bb_lower = indicators.get("bb_lower", indicators.get("BB_Lower"))
        bb_upper = indicators.get("bb_upper", indicators.get("BB_Upper"))

        if bb_lower and bb_lower < current_price:
            support_levels.append(float(bb_lower))
        if bb_upper and bb_upper > current_price:
            resistance_levels.append(float(bb_upper))

        # Add moving averages as support/resistance
        for ma_key in ["sma_50", "SMA_50", "sma_200", "SMA_200"]:
            ma_value = indicators.get(ma_key)
            if ma_value:
                if ma_value < current_price:
                    support_levels.append(float(ma_value))
                else:
                    resistance_levels.append(float(ma_value))

        return {
            "immediate_support": float(immediate_support),
            "immediate_resistance": float(immediate_resistance),
            "support_levels": sorted(set(support_levels), reverse=True)[:5],
            "resistance_levels": sorted(set(resistance_levels))[:5],
        }

    def _merge_signals(
        self,
        prog_entry: List[EntrySignal],
        prog_exit: List[ExitSignal],
        prog_zone: Optional[OptimalEntryZone],
        llm_entry: Optional[List[Dict]],
        llm_exit: Optional[List[Dict]],
        llm_zone: Optional[Dict],
        current_price: float,
    ) -> tuple:
        """
        Merge programmatic and LLM signals.

        Returns:
            Tuple of (final_entry, final_exit, final_zone, agreement_score, confidence_boost)
        """
        # Start with programmatic signals as base
        final_entry = list(prog_entry)
        final_exit = list(prog_exit)
        final_zone = prog_zone

        agreement_score = 0.0
        confidence_boost = False

        if not llm_entry and not llm_exit:
            # No LLM signals, use programmatic only
            return final_entry, final_exit, final_zone, 0.5, False

        # Calculate agreement between programmatic and LLM signals
        agreements = 0
        total_checks = 0

        if llm_entry and prog_entry:
            # Check if entry signal directions agree
            llm_buy_signals = sum(1 for s in llm_entry if self._is_buy_signal(s))
            prog_buy_signals = len(
                [
                    s
                    for s in prog_entry
                    if s.signal_type
                    in [
                        SignalType.OVERSOLD_REVERSAL,
                        SignalType.SUPPORT_BOUNCE,
                        SignalType.GOLDEN_CROSS,
                        SignalType.MOMENTUM,
                    ]
                ]
            )

            if (llm_buy_signals > 0 and prog_buy_signals > 0) or (llm_buy_signals == 0 and prog_buy_signals == 0):
                agreements += 1
            total_checks += 1

            # Check if price levels are similar (within 3%)
            for llm_sig in llm_entry[:3]:
                llm_price = llm_sig.get("price_level", 0)
                if llm_price > 0:
                    for prog_sig in prog_entry:
                        if abs(prog_sig.price_level - llm_price) / llm_price < 0.03:
                            agreements += 1
                            break
                    total_checks += 1

        if llm_zone and prog_zone:
            # Check if entry zones overlap
            llm_lower = llm_zone.get("lower_bound", 0)
            llm_upper = llm_zone.get("upper_bound", 0)

            if llm_lower and llm_upper and llm_lower < llm_upper:
                # Check for overlap
                if not (prog_zone.upper_bound < llm_lower or prog_zone.lower_bound > llm_upper):
                    agreements += 2  # Strong agreement
                total_checks += 2

        if total_checks > 0:
            agreement_score = agreements / total_checks
            confidence_boost = agreement_score > 0.6

        # Boost confidence of signals that both agree on
        if confidence_boost and llm_entry:
            for i, prog_sig in enumerate(final_entry):
                for llm_sig in llm_entry:
                    llm_price = llm_sig.get("price_level", 0)
                    if llm_price > 0 and abs(prog_sig.price_level - llm_price) / llm_price < 0.03:
                        # Both agree on this signal - boost confidence
                        if prog_sig.confidence == SignalConfidence.MEDIUM:
                            final_entry[i] = EntrySignal(
                                signal_type=prog_sig.signal_type,
                                price_level=prog_sig.price_level,
                                confidence=SignalConfidence.HIGH,  # Boosted
                                rationale=prog_sig.rationale + " (confirmed by LLM analysis)",
                                risk_reward_ratio=prog_sig.risk_reward_ratio,
                                stop_loss=prog_sig.stop_loss,
                                stop_loss_pct=prog_sig.stop_loss_pct,
                                target_price=prog_sig.target_price,
                                target_pct=prog_sig.target_pct,
                                expected_holding_days=prog_sig.expected_holding_days,
                                volume_confirmation=prog_sig.volume_confirmation,
                                trend_alignment=prog_sig.trend_alignment,
                            )
                        break

        # Merge LLM-unique signals
        if llm_entry:
            for llm_sig in llm_entry:
                if not self._signal_exists(llm_sig, final_entry, current_price):
                    # Convert LLM signal to EntrySignal if valid
                    converted = self._convert_llm_entry_signal(llm_sig, current_price)
                    if converted:
                        final_entry.append(converted)

        # Merge optimal entry zone (prefer programmatic if available)
        if prog_zone is None and llm_zone:
            final_zone = self._convert_llm_entry_zone(llm_zone, current_price)
        elif prog_zone and llm_zone and confidence_boost:
            # Average the zones when they agree
            final_zone = OptimalEntryZone(
                lower_bound=(prog_zone.lower_bound + llm_zone.get("lower_bound", prog_zone.lower_bound)) / 2,
                upper_bound=(prog_zone.upper_bound + llm_zone.get("upper_bound", prog_zone.upper_bound)) / 2,
                ideal_entry=(prog_zone.ideal_entry + llm_zone.get("ideal_entry", prog_zone.ideal_entry)) / 2,
                timing=prog_zone.timing,
                scaling_strategy=prog_zone.scaling_strategy,
                confidence=SignalConfidence.HIGH if confidence_boost else prog_zone.confidence,
                rationale=prog_zone.rationale + " (confirmed by LLM)",
                recommended_allocation_pct=prog_zone.recommended_allocation_pct,
                max_position_size_pct=prog_zone.max_position_size_pct,
            )

        return final_entry, final_exit, final_zone, agreement_score, confidence_boost

    def _is_buy_signal(self, llm_signal: Dict) -> bool:
        """Check if LLM signal is a buy signal."""
        signal_type = llm_signal.get("signal_type", "").upper()
        buy_types = [
            "OVERSOLD",
            "SUPPORT",
            "MACD_CROSSOVER",
            "GOLDEN_CROSS",
            "BOLLINGER_BOUNCE",
            "BREAKOUT",
            "VALUATION",
            "MOMENTUM",
            "BUY",
            "LONG",
        ]
        return any(bt in signal_type for bt in buy_types)

    def _signal_exists(
        self,
        llm_signal: Dict,
        existing_signals: List[EntrySignal],
        current_price: float,
    ) -> bool:
        """Check if LLM signal already exists in programmatic signals."""
        llm_price = llm_signal.get("price_level", 0)
        if llm_price <= 0:
            return False

        for sig in existing_signals:
            if abs(sig.price_level - llm_price) / current_price < 0.02:
                return True
        return False

    def _convert_llm_entry_signal(
        self,
        llm_signal: Dict,
        current_price: float,
    ) -> Optional[EntrySignal]:
        """Convert LLM signal dict to EntrySignal."""
        try:
            price_level = float(llm_signal.get("price_level", 0))
            if price_level <= 0:
                return None

            stop_loss = float(llm_signal.get("stop_loss", price_level * 0.95))
            target_price = float(llm_signal.get("target_price", price_level * 1.10))

            # Parse signal type
            signal_type_str = llm_signal.get("signal_type", "MOMENTUM").upper()
            signal_type = SignalType.MOMENTUM
            for st in SignalType:
                if st.value.upper() in signal_type_str or signal_type_str in st.value.upper():
                    signal_type = st
                    break

            # Parse confidence
            confidence_str = llm_signal.get("confidence", "MEDIUM").upper()
            confidence = SignalConfidence.MEDIUM
            if "HIGH" in confidence_str:
                confidence = SignalConfidence.HIGH
            elif "LOW" in confidence_str:
                confidence = SignalConfidence.LOW

            return EntrySignal(
                signal_type=signal_type,
                price_level=price_level,
                confidence=confidence,
                rationale=llm_signal.get("rationale", "LLM-generated signal"),
                risk_reward_ratio=float(llm_signal.get("risk_reward_ratio", 2.0)),
                stop_loss=stop_loss,
                stop_loss_pct=abs((price_level - stop_loss) / price_level) * 100,
                target_price=target_price,
                target_pct=abs((target_price - price_level) / price_level) * 100,
                expected_holding_days=int(llm_signal.get("expected_holding_days", 30)),
                volume_confirmation=llm_signal.get("volume_confirmation", False),
                trend_alignment=llm_signal.get("trend_alignment", False),
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert LLM signal: {e}")
            return None

    def _convert_llm_entry_zone(
        self,
        llm_zone: Dict,
        current_price: float,
    ) -> Optional[OptimalEntryZone]:
        """Convert LLM entry zone dict to OptimalEntryZone."""
        try:
            lower = float(llm_zone.get("lower_bound", current_price * 0.97))
            upper = float(llm_zone.get("upper_bound", current_price * 1.02))
            ideal = float(llm_zone.get("ideal_entry", current_price))

            # Parse timing
            timing_str = llm_zone.get("timing", "WAIT_PULLBACK").upper()
            timing = SignalTiming.WAIT_PULLBACK
            if "IMMEDIATE" in timing_str:
                timing = SignalTiming.IMMEDIATE
            elif "AVOID" in timing_str:
                timing = SignalTiming.AVOID_NOW

            # Parse scaling strategy
            scaling_str = llm_zone.get("scaling_strategy", "SCALE_IN_THIRDS").upper()
            scaling = ScalingStrategy.SCALE_IN_THIRDS
            if "FULL" in scaling_str:
                scaling = ScalingStrategy.FULL_POSITION
            elif "HALVES" in scaling_str:
                scaling = ScalingStrategy.SCALE_IN_HALVES
            elif "PYRAMID" in scaling_str:
                scaling = ScalingStrategy.PYRAMID
            elif "DOLLAR" in scaling_str or "DCA" in scaling_str:
                scaling = ScalingStrategy.DOLLAR_COST_AVERAGE

            # Parse confidence
            conf_str = llm_zone.get("confidence", "MEDIUM").upper()
            confidence = SignalConfidence.MEDIUM
            if "HIGH" in conf_str:
                confidence = SignalConfidence.HIGH
            elif "LOW" in conf_str:
                confidence = SignalConfidence.LOW

            return OptimalEntryZone(
                lower_bound=lower,
                upper_bound=upper,
                ideal_entry=ideal,
                timing=timing,
                scaling_strategy=scaling,
                confidence=confidence,
                rationale=llm_zone.get("rationale", "LLM-generated entry zone"),
                recommended_allocation_pct=float(llm_zone.get("recommended_allocation_pct", 5.0)),
                max_position_size_pct=float(llm_zone.get("max_position_size_pct", 10.0)),
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert LLM entry zone: {e}")
            return None

    def to_report_format(self, signals: IntegratedSignals) -> Dict[str, Any]:
        """
        Convert integrated signals to report-friendly format for PDF generation.

        Returns:
            Dict suitable for report_payload_builder.py
        """

        def signal_to_dict(sig: EntrySignal) -> Dict:
            return {
                "signal_type": sig.signal_type.value,
                "price_level": round(sig.price_level, 2),
                "confidence": sig.confidence.value,
                "rationale": sig.rationale,
                "risk_reward_ratio": round(sig.risk_reward_ratio, 2),
                "stop_loss": round(sig.stop_loss, 2),
                "stop_loss_pct": round(sig.stop_loss_pct, 2),
                "target_price": round(sig.target_price, 2),
                "target_pct": round(sig.target_pct, 2),
                "expected_holding_days": sig.expected_holding_days,
                "volume_confirmation": sig.volume_confirmation,
                "trend_alignment": sig.trend_alignment,
            }

        def exit_to_dict(sig: ExitSignal) -> Dict:
            return {
                "signal_type": sig.signal_type.value,
                "price_level": round(sig.price_level, 2),
                "confidence": sig.confidence.value,
                "rationale": sig.rationale,
                "urgency": sig.urgency,
                "partial_exit_pct": round(sig.partial_exit_pct, 1),
            }

        def zone_to_dict(zone: OptimalEntryZone) -> Dict:
            return {
                "lower_bound": round(zone.lower_bound, 2),
                "upper_bound": round(zone.upper_bound, 2),
                "ideal_entry": round(zone.ideal_entry, 2),
                "timing": zone.timing.value,
                "scaling_strategy": zone.scaling_strategy.value,
                "confidence": zone.confidence.value,
                "rationale": zone.rationale,
                "recommended_allocation_pct": round(zone.recommended_allocation_pct, 1),
                "max_position_size_pct": round(zone.max_position_size_pct, 1),
            }

        return {
            "entry_signals": [signal_to_dict(s) for s in signals.final_entry_signals[:5]],
            "exit_signals": [exit_to_dict(s) for s in signals.final_exit_signals[:5]],
            "optimal_entry_zone": zone_to_dict(signals.final_entry_zone) if signals.final_entry_zone else None,
            "signal_agreement_score": round(signals.signal_agreement_score, 2),
            "confidence_boost": signals.confidence_boost,
            "total_entry_signals": len(signals.final_entry_signals),
            "total_exit_signals": len(signals.final_exit_signals),
        }


# Singleton
_signal_integrator: Optional[SignalIntegrator] = None


def get_signal_integrator() -> SignalIntegrator:
    """Get singleton SignalIntegrator instance."""
    global _signal_integrator
    if _signal_integrator is None:
        _signal_integrator = SignalIntegrator()
    return _signal_integrator
