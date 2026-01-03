"""
Chart Pattern Recognition Module

Detects technical chart patterns in price data for investment analysis.
Foundation for Tier 4 advanced analytics.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from scipy import signal

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Chart pattern types"""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULLISH_FLAG = "bullish_flag"
    BEARISH_FLAG = "bearish_flag"
    CONSOLIDATION = "consolidation"


@dataclass
class PatternResult:
    """Result from pattern detection"""
    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    start_date: datetime
    end_date: datetime
    direction: str  # 'bullish', 'bearish', 'neutral'
    key_points: Dict  # Important points in the pattern
    price_target: float
    stop_loss: float
    description: str


class PatternRecognizer:
    """Chart pattern recognition engine"""

    def __init__(
        self,
        min_pattern_length: int = 5,
        peak_prominence: float = 1.0,
        similarity_threshold: float = 0.10
    ):
        """
        Initialize pattern recognizer

        Args:
            min_pattern_length: Minimum number of bars for a pattern
            peak_prominence: Prominence threshold for peak detection
            similarity_threshold: Maximum price difference ratio for similar peaks/troughs
        """
        self.min_pattern_length = min_pattern_length
        self.peak_prominence = peak_prominence
        self.similarity_threshold = similarity_threshold

        logger.info(f"Initialized PatternRecognizer (min_length={min_pattern_length})")

    def detect_patterns(self, price_data: pd.DataFrame) -> List[PatternResult]:
        """
        Detect all patterns in price data

        Args:
            price_data: DataFrame with columns: date, open, high, low, close, volume

        Returns:
            List of detected patterns
        """
        logger.info(f"Detecting patterns in {len(price_data)} bars")

        patterns = []

        # Ensure we have enough data
        if len(price_data) < self.min_pattern_length:
            logger.warning(f"Insufficient data for pattern detection: {len(price_data)} bars")
            return patterns

        # Find peaks and troughs
        prices = price_data['close'].values
        peaks, troughs = find_peaks_and_troughs(prices, prominence=self.peak_prominence)

        # Check for double top
        double_tops = self._detect_double_top(price_data, peaks, troughs)
        patterns.extend(double_tops)

        # Check for double bottom
        double_bottoms = self._detect_double_bottom(price_data, peaks, troughs)
        patterns.extend(double_bottoms)

        # Check for head and shoulders
        head_shoulders = self._detect_head_and_shoulders(price_data, peaks, troughs)
        patterns.extend(head_shoulders)

        # Check for inverse head and shoulders
        inverse_head_shoulders = self._detect_inverse_head_and_shoulders(price_data, peaks, troughs)
        patterns.extend(inverse_head_shoulders)

        # Check for ascending triangle
        ascending_triangles = self._detect_ascending_triangle(price_data, peaks, troughs)
        patterns.extend(ascending_triangles)

        # Check for descending triangle
        descending_triangles = self._detect_descending_triangle(price_data, peaks, troughs)
        patterns.extend(descending_triangles)

        # Check for symmetrical triangle
        symmetrical_triangles = self._detect_symmetrical_triangle(price_data, peaks, troughs)
        patterns.extend(symmetrical_triangles)

        # Check for bullish flag
        bullish_flags = self._detect_bullish_flag(price_data, peaks, troughs)
        patterns.extend(bullish_flags)

        # Check for bearish flag
        bearish_flags = self._detect_bearish_flag(price_data, peaks, troughs)
        patterns.extend(bearish_flags)

        # Check for consolidation
        consolidations = self._detect_consolidation(price_data, peaks, troughs)
        patterns.extend(consolidations)

        logger.info(f"Detected {len(patterns)} patterns")

        return patterns

    def _detect_double_top(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect double top patterns

        Double Top Pattern:
        - Two peaks at similar price levels
        - Trough between them
        - Second peak followed by decline
        - Bearish reversal signal
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 2 peaks
        if len(peaks) < 2:
            return patterns

        # Check consecutive peak pairs
        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]

            peak1_price = prices[peak1_idx]
            peak2_price = prices[peak2_idx]

            # Check if peaks are at similar levels
            price_diff_ratio = abs(peak1_price - peak2_price) / peak1_price

            if price_diff_ratio > self.similarity_threshold:
                continue  # Peaks too different

            # Find trough between peaks
            troughs_between = troughs[(troughs > peak1_idx) & (troughs < peak2_idx)]

            if len(troughs_between) == 0:
                continue  # No trough between peaks

            trough_idx = troughs_between[0]
            trough_price = prices[trough_idx]

            # Trough should be significantly below peaks
            depth = (peak1_price - trough_price) / peak1_price

            if depth < 0.03:  # At least 3% depth
                continue

            # Check if pattern is complete (decline after second peak)
            if peak2_idx >= len(prices) - 5:
                continue  # Not enough data after second peak

            # Check for decline after second peak
            prices_after = prices[peak2_idx:min(peak2_idx + 10, len(prices))]
            if len(prices_after) > 0 and prices_after[-1] >= peak2_price * 0.98:
                continue  # No significant decline

            # Calculate confidence
            volume_confirmation = True  # Simplified for now
            peaks_prices = np.array([peak1_price, peak2_price])
            confidence = calculate_pattern_confidence(
                peaks_prices,
                pattern_type='double_top',
                volume_confirmation=volume_confirmation
            )

            # Calculate price target (depth of pattern projected down)
            price_target = trough_price - (peak1_price - trough_price)

            # Stop loss above second peak
            stop_loss = peak2_price * 1.02

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.DOUBLE_TOP,
                confidence=confidence,
                start_date=price_data['date'].iloc[peak1_idx],
                end_date=price_data['date'].iloc[peak2_idx],
                direction='bearish',
                key_points={
                    'peak1': (peak1_idx, peak1_price),
                    'trough': (trough_idx, trough_price),
                    'peak2': (peak2_idx, peak2_price)
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Double top at ${peak1_price:.2f}, targeting ${price_target:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected double top: {peak1_price:.2f} -> {trough_price:.2f} -> {peak2_price:.2f}")

        return patterns

    def _detect_double_bottom(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect double bottom patterns

        Double Bottom Pattern:
        - Two troughs at similar price levels
        - Peak between them
        - Second trough followed by rise
        - Bullish reversal signal
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 2 troughs
        if len(troughs) < 2:
            return patterns

        # Check consecutive trough pairs
        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough2_idx = troughs[i + 1]

            trough1_price = prices[trough1_idx]
            trough2_price = prices[trough2_idx]

            # Check if troughs are at similar levels
            price_diff_ratio = abs(trough1_price - trough2_price) / trough1_price

            if price_diff_ratio > self.similarity_threshold:
                continue  # Troughs too different

            # Find peak between troughs
            peaks_between = peaks[(peaks > trough1_idx) & (peaks < trough2_idx)]

            if len(peaks_between) == 0:
                continue  # No peak between troughs

            peak_idx = peaks_between[0]
            peak_price = prices[peak_idx]

            # Peak should be significantly above troughs
            height = (peak_price - trough1_price) / trough1_price

            if height < 0.03:  # At least 3% height
                continue

            # Check if pattern is complete (rise after second trough)
            if trough2_idx >= len(prices) - 5:
                continue  # Not enough data after second trough

            # Check for rise after second trough
            prices_after = prices[trough2_idx:min(trough2_idx + 10, len(prices))]
            if len(prices_after) > 0 and prices_after[-1] <= trough2_price * 1.02:
                continue  # No significant rise

            # Calculate confidence
            volume_confirmation = True  # Simplified for now
            troughs_prices = np.array([trough1_price, trough2_price])
            confidence = calculate_pattern_confidence(
                troughs_prices,
                pattern_type='double_bottom',
                volume_confirmation=volume_confirmation
            )

            # Calculate price target (height of pattern projected up)
            price_target = peak_price + (peak_price - trough1_price)

            # Stop loss below second trough
            stop_loss = trough2_price * 0.98

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.DOUBLE_BOTTOM,
                confidence=confidence,
                start_date=price_data['date'].iloc[trough1_idx],
                end_date=price_data['date'].iloc[trough2_idx],
                direction='bullish',
                key_points={
                    'trough1': (trough1_idx, trough1_price),
                    'peak': (peak_idx, peak_price),
                    'trough2': (trough2_idx, trough2_price)
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Double bottom at ${trough1_price:.2f}, targeting ${price_target:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected double bottom: {trough1_price:.2f} -> {peak_price:.2f} -> {trough2_price:.2f}")

        return patterns

    def _detect_head_and_shoulders(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect head and shoulders patterns

        Head and Shoulders Pattern:
        - Three peaks: left shoulder, head (highest), right shoulder
        - Head is significantly higher than shoulders
        - Shoulders at similar levels
        - Neckline connects troughs between peaks
        - Bearish reversal signal
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 3 peaks
        if len(peaks) < 3:
            return patterns

        # Check consecutive peak triplets
        for i in range(len(peaks) - 2):
            left_shoulder_idx = peaks[i]
            head_idx = peaks[i + 1]
            right_shoulder_idx = peaks[i + 2]

            left_shoulder_price = prices[left_shoulder_idx]
            head_price = prices[head_idx]
            right_shoulder_price = prices[right_shoulder_idx]

            # Head must be higher than both shoulders
            if head_price <= left_shoulder_price or head_price <= right_shoulder_price:
                continue

            # Head should be significantly higher (at least 5%)
            if (head_price - left_shoulder_price) / left_shoulder_price < 0.05:
                continue
            if (head_price - right_shoulder_price) / right_shoulder_price < 0.05:
                continue

            # Shoulders should be at similar levels
            shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price

            if shoulder_diff > self.similarity_threshold * 2:  # Allow more variation
                continue

            # Find troughs (neckline points)
            troughs_between = troughs[(troughs > left_shoulder_idx) & (troughs < right_shoulder_idx)]

            if len(troughs_between) < 2:
                continue  # Need at least 2 troughs for neckline

            left_trough_idx = troughs_between[0]
            right_trough_idx = troughs_between[-1]

            left_trough_price = prices[left_trough_idx]
            right_trough_price = prices[right_trough_idx]

            # Calculate neckline
            neckline_avg = (left_trough_price + right_trough_price) / 2

            # Check if pattern is complete (breakdown below neckline)
            if right_shoulder_idx >= len(prices) - 5:
                continue

            prices_after = prices[right_shoulder_idx:min(right_shoulder_idx + 10, len(prices))]
            if len(prices_after) > 0 and prices_after[-1] >= neckline_avg * 0.97:
                continue  # No breakdown

            # Calculate confidence
            shoulder_similarity = 1.0 - shoulder_diff
            confidence = 0.6 + shoulder_similarity * 0.3

            # Pattern height (head to neckline)
            pattern_height = head_price - neckline_avg

            # Price target: neckline minus pattern height
            price_target = neckline_avg - pattern_height

            # Stop loss above head
            stop_loss = head_price * 1.02

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                confidence=confidence,
                start_date=price_data['date'].iloc[left_shoulder_idx],
                end_date=price_data['date'].iloc[right_shoulder_idx],
                direction='bearish',
                key_points={
                    'left_shoulder': (left_shoulder_idx, left_shoulder_price),
                    'head': (head_idx, head_price),
                    'right_shoulder': (right_shoulder_idx, right_shoulder_price),
                    'neckline': neckline_avg
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Head and shoulders: shoulders at ${left_shoulder_price:.2f}, head at ${head_price:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected head and shoulders: {left_shoulder_price:.2f} -> {head_price:.2f} -> {right_shoulder_price:.2f}")

        return patterns

    def _detect_inverse_head_and_shoulders(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect inverse head and shoulders patterns

        Inverse Head and Shoulders Pattern:
        - Three troughs: left shoulder, head (lowest), right shoulder
        - Head is significantly lower than shoulders
        - Shoulders at similar levels
        - Neckline connects peaks between troughs
        - Bullish reversal signal
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 3 troughs
        if len(troughs) < 3:
            return patterns

        # Check consecutive trough triplets
        for i in range(len(troughs) - 2):
            left_shoulder_idx = troughs[i]
            head_idx = troughs[i + 1]
            right_shoulder_idx = troughs[i + 2]

            left_shoulder_price = prices[left_shoulder_idx]
            head_price = prices[head_idx]
            right_shoulder_price = prices[right_shoulder_idx]

            # Head must be lower than both shoulders
            if head_price >= left_shoulder_price or head_price >= right_shoulder_price:
                continue

            # Head should be significantly lower (at least 5%)
            if (left_shoulder_price - head_price) / left_shoulder_price < 0.05:
                continue
            if (right_shoulder_price - head_price) / right_shoulder_price < 0.05:
                continue

            # Shoulders should be at similar levels
            shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price

            if shoulder_diff > self.similarity_threshold * 2:
                continue

            # Find peaks (neckline points)
            peaks_between = peaks[(peaks > left_shoulder_idx) & (peaks < right_shoulder_idx)]

            if len(peaks_between) < 2:
                continue

            left_peak_idx = peaks_between[0]
            right_peak_idx = peaks_between[-1]

            left_peak_price = prices[left_peak_idx]
            right_peak_price = prices[right_peak_idx]

            # Calculate neckline
            neckline_avg = (left_peak_price + right_peak_price) / 2

            # Check if pattern is complete (breakout above neckline)
            if right_shoulder_idx >= len(prices) - 5:
                continue

            prices_after = prices[right_shoulder_idx:min(right_shoulder_idx + 10, len(prices))]
            if len(prices_after) > 0 and prices_after[-1] <= neckline_avg * 1.03:
                continue  # No breakout

            # Calculate confidence
            shoulder_similarity = 1.0 - shoulder_diff
            confidence = 0.6 + shoulder_similarity * 0.3

            # Pattern height (neckline to head)
            pattern_height = neckline_avg - head_price

            # Price target: neckline plus pattern height
            price_target = neckline_avg + pattern_height

            # Stop loss below head
            stop_loss = head_price * 0.98

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                confidence=confidence,
                start_date=price_data['date'].iloc[left_shoulder_idx],
                end_date=price_data['date'].iloc[right_shoulder_idx],
                direction='bullish',
                key_points={
                    'left_shoulder': (left_shoulder_idx, left_shoulder_price),
                    'head': (head_idx, head_price),
                    'right_shoulder': (right_shoulder_idx, right_shoulder_price),
                    'neckline': neckline_avg
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Inverse head and shoulders: shoulders at ${left_shoulder_price:.2f}, head at ${head_price:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected inverse head and shoulders: {left_shoulder_price:.2f} -> {head_price:.2f} -> {right_shoulder_price:.2f}")

        return patterns

    def _detect_ascending_triangle(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect ascending triangle patterns

        Ascending Triangle:
        - Flat/horizontal resistance (peaks at similar levels)
        - Rising support (higher lows/troughs)
        - Bullish breakout pattern
        - Usually breaks upward
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 3 peaks and 3 troughs
        if len(peaks) < 3 or len(troughs) < 3:
            return patterns

        # Look for flat resistance and rising support
        for i in range(len(peaks) - 2):
            # Get 3 consecutive peaks
            peak_indices = peaks[i:i+3]
            peak_prices = prices[peak_indices]

            # Check if peaks are at similar level (flat resistance)
            resistance_level = np.mean(peak_prices)
            peak_variance = np.std(peak_prices) / resistance_level

            if peak_variance > 0.05:  # More than 5% variance (triangles are less strict than double tops)
                continue

            # Find troughs within this range
            troughs_in_range = troughs[(troughs > peak_indices[0]) & (troughs < peak_indices[-1])]

            if len(troughs_in_range) < 2:
                continue

            trough_prices = prices[troughs_in_range]

            # Check if troughs are rising (ascending support)
            if len(trough_prices) >= 2:
                # Simple linear fit to check rising trend
                trough_slope = (trough_prices[-1] - trough_prices[0]) / len(trough_prices)

                if trough_slope <= 0:  # Not rising
                    continue

            # Check for breakout above resistance
            last_peak_idx = peak_indices[-1]
            if last_peak_idx >= len(prices) - 5:
                continue

            prices_after = prices[last_peak_idx:min(last_peak_idx + 10, len(prices))]
            if len(prices_after) > 0 and prices_after[-1] <= resistance_level * 1.02:
                continue  # No breakout

            # Calculate confidence
            confidence = 0.6 + (0.4 * min(1.0, trough_slope / 2))  # Higher slope = higher confidence

            # Price target: height of triangle projected upward
            pattern_height = resistance_level - trough_prices[0]
            price_target = resistance_level + pattern_height

            # Stop loss below recent support
            stop_loss = trough_prices[-1] * 0.98

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                confidence=confidence,
                start_date=price_data['date'].iloc[peak_indices[0]],
                end_date=price_data['date'].iloc[last_peak_idx],
                direction='bullish',
                key_points={
                    'resistance_level': resistance_level,
                    'support_slope': trough_slope,
                    'peak_indices': peak_indices.tolist(),
                    'trough_indices': troughs_in_range.tolist()
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Ascending triangle with resistance at ${resistance_level:.2f}, targeting ${price_target:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected ascending triangle: resistance ${resistance_level:.2f}")

        return patterns

    def _detect_descending_triangle(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect descending triangle patterns

        Descending Triangle:
        - Descending resistance (lower highs/peaks)
        - Flat/horizontal support (troughs at similar levels)
        - Bearish breakdown pattern
        - Usually breaks downward
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 3 peaks and 3 troughs
        if len(peaks) < 3 or len(troughs) < 3:
            return patterns

        # Look for descending resistance and flat support
        for i in range(len(troughs) - 2):
            # Get 3 consecutive troughs
            trough_indices = troughs[i:i+3]
            trough_prices = prices[trough_indices]

            # Check if troughs are at similar level (flat support)
            support_level = np.mean(trough_prices)
            trough_variance = np.std(trough_prices) / support_level

            if trough_variance > 0.05:  # More than 5% variance (triangles are less strict)
                continue

            # Find peaks within this range
            peaks_in_range = peaks[(peaks > trough_indices[0]) & (peaks < trough_indices[-1])]

            if len(peaks_in_range) < 2:
                continue

            peak_prices = prices[peaks_in_range]

            # Check if peaks are descending
            if len(peak_prices) >= 2:
                peak_slope = (peak_prices[-1] - peak_prices[0]) / len(peak_prices)

                if peak_slope >= 0:  # Not descending
                    continue

            # Check for breakdown below support
            last_trough_idx = trough_indices[-1]
            if last_trough_idx >= len(prices) - 5:
                continue

            prices_after = prices[last_trough_idx:min(last_trough_idx + 10, len(prices))]
            if len(prices_after) > 0 and prices_after[-1] >= support_level * 0.98:
                continue  # No breakdown

            # Calculate confidence
            confidence = 0.6 + (0.4 * min(1.0, abs(peak_slope) / 2))

            # Price target: height of triangle projected downward
            pattern_height = peak_prices[0] - support_level
            price_target = support_level - pattern_height

            # Stop loss above recent resistance
            stop_loss = peak_prices[-1] * 1.02

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.DESCENDING_TRIANGLE,
                confidence=confidence,
                start_date=price_data['date'].iloc[trough_indices[0]],
                end_date=price_data['date'].iloc[last_trough_idx],
                direction='bearish',
                key_points={
                    'support_level': support_level,
                    'resistance_slope': peak_slope,
                    'trough_indices': trough_indices.tolist(),
                    'peak_indices': peaks_in_range.tolist()
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Descending triangle with support at ${support_level:.2f}, targeting ${price_target:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected descending triangle: support ${support_level:.2f}")

        return patterns

    def _detect_symmetrical_triangle(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect symmetrical triangle patterns

        Symmetrical Triangle:
        - Descending resistance (lower highs)
        - Ascending support (higher lows)
        - Converging trendlines
        - Neutral pattern - can break either direction
        """
        patterns = []
        prices = price_data['close'].values

        # Need at least 3 peaks and 3 troughs
        if len(peaks) < 3 or len(troughs) < 3:
            return patterns

        # Look for converging trendlines
        for i in range(min(len(peaks) - 2, len(troughs) - 2)):
            # Check if we can find overlapping peaks and troughs
            if i >= len(peaks) - 2 or i >= len(troughs) - 2:
                continue

            peak_indices = peaks[i:i+3]
            trough_indices = troughs[i:i+3]

            # Ensure they interleave properly
            if peak_indices[0] > trough_indices[-1] or trough_indices[0] > peak_indices[-1]:
                continue

            peak_prices = prices[peak_indices]
            trough_prices = prices[trough_indices]

            # Check for descending peaks
            peak_slope = (peak_prices[-1] - peak_prices[0]) / len(peak_prices)
            if peak_slope >= 0:
                continue

            # Check for ascending troughs
            trough_slope = (trough_prices[-1] - trough_prices[0]) / len(trough_prices)
            if trough_slope <= 0:
                continue

            # Check convergence (slopes should be similar magnitude)
            slope_ratio = abs(peak_slope) / abs(trough_slope) if trough_slope != 0 else 0
            if slope_ratio < 0.5 or slope_ratio > 2.0:  # Not converging symmetrically
                continue

            # Find breakout direction
            last_point_idx = max(peak_indices[-1], trough_indices[-1])
            if last_point_idx >= len(prices) - 5:
                continue

            # Calculate apex (where lines would meet)
            mid_price = (peak_prices[-1] + trough_prices[-1]) / 2

            # Determine breakout direction
            prices_after = prices[last_point_idx:min(last_point_idx + 10, len(prices))]
            if len(prices_after) == 0:
                continue

            final_price = prices_after[-1]
            if final_price > mid_price * 1.02:
                direction = 'bullish'
                price_target = mid_price + (peak_prices[0] - trough_prices[0])
            elif final_price < mid_price * 0.98:
                direction = 'bearish'
                price_target = mid_price - (peak_prices[0] - trough_prices[0])
            else:
                direction = 'neutral'
                price_target = mid_price

            # Calculate confidence (lower for symmetrical as direction is uncertain until breakout)
            confidence = 0.5 + (0.3 * min(1.0, 1.0 / abs(1.0 - slope_ratio)))

            # Stop loss on opposite side of breakout
            if direction == 'bullish':
                stop_loss = trough_prices[-1] * 0.98
            elif direction == 'bearish':
                stop_loss = peak_prices[-1] * 1.02
            else:
                stop_loss = mid_price * 0.98

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                confidence=confidence,
                start_date=price_data['date'].iloc[min(peak_indices[0], trough_indices[0])],
                end_date=price_data['date'].iloc[last_point_idx],
                direction=direction,
                key_points={
                    'apex': mid_price,
                    'converging_point': (last_point_idx, mid_price),
                    'peak_slope': peak_slope,
                    'trough_slope': trough_slope,
                    'peak_indices': peak_indices.tolist(),
                    'trough_indices': trough_indices.tolist()
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Symmetrical triangle converging at ${mid_price:.2f}, {direction} breakout"
            )

            patterns.append(pattern)
            logger.info(f"Detected symmetrical triangle: {direction} breakout from ${mid_price:.2f}")

        return patterns

    def _detect_bullish_flag(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect bullish flag patterns

        Bullish Flag:
        - Strong upward move (pole)
        - Consolidation in a slight downward channel (flag)
        - Continuation breakout upward
        - Bullish continuation pattern
        """
        patterns = []
        prices = price_data['close'].values

        # Need sufficient data
        if len(prices) < 30:
            return patterns

        # Look for sharp upward moves followed by consolidation
        for i in range(10, len(prices) - 20):
            # Check for pole: strong upward move
            pole_start = i - 10
            pole_end = i
            pole_prices = prices[pole_start:pole_end]

            if len(pole_prices) < 10:
                continue

            pole_gain = (pole_prices[-1] - pole_prices[0]) / pole_prices[0]

            # Need at least 10% gain in the pole
            if pole_gain < 0.10:
                continue

            # Check pole strength (consistent upward movement)
            pole_slope = (pole_prices[-1] - pole_prices[0]) / len(pole_prices)
            if pole_slope <= 0:
                continue

            # Look for flag consolidation after pole
            flag_start = pole_end
            flag_end = min(pole_end + 15, len(prices))
            flag_prices = prices[flag_start:flag_end]

            if len(flag_prices) < 10:
                continue

            # Flag should be relatively flat or slightly downward
            flag_slope = (flag_prices[-1] - flag_prices[0]) / len(flag_prices)
            flag_range = np.max(flag_prices) - np.min(flag_prices)
            flag_volatility = flag_range / flag_prices[0]

            # Flag should be consolidating (low volatility compared to pole)
            if flag_volatility > 0.05:  # More than 5% range
                continue

            # Flag can slope slightly down or be flat, but not up
            if flag_slope > pole_slope * 0.3:  # Not a flag if sloping up too much
                continue

            # Check for breakout continuation
            if flag_end >= len(prices) - 5:
                continue

            breakout_prices = prices[flag_end:min(flag_end + 5, len(prices))]
            if len(breakout_prices) == 0:
                continue

            # Breakout should exceed flag high
            flag_high = np.max(flag_prices)
            if breakout_prices[-1] <= flag_high * 1.00:
                continue  # No breakout

            # Calculate confidence based on pole strength and flag tightness
            pole_strength = min(1.0, pole_gain / 0.20)  # Up to 20% gain
            flag_tightness = 1.0 - min(1.0, flag_volatility / 0.05)
            confidence = 0.6 + (0.2 * pole_strength) + (0.2 * flag_tightness)

            # Price target: pole height projected from breakout
            pole_height = pole_prices[-1] - pole_prices[0]
            price_target = flag_high + pole_height

            # Stop loss below flag low
            flag_low = np.min(flag_prices)
            stop_loss = flag_low * 0.98

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.BULLISH_FLAG,
                confidence=confidence,
                start_date=price_data['date'].iloc[pole_start],
                end_date=price_data['date'].iloc[flag_end],
                direction='bullish',
                key_points={
                    'pole_start': pole_start,
                    'pole_end': pole_end,
                    'pole_gain': pole_gain,
                    'flag_high': flag_high,
                    'flag_low': flag_low,
                    'pole_height': pole_height
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Bullish flag with {pole_gain*100:.1f}% pole, targeting ${price_target:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected bullish flag: pole gain {pole_gain*100:.1f}%, target ${price_target:.2f}")

        return patterns

    def _detect_bearish_flag(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect bearish flag patterns

        Bearish Flag:
        - Strong downward move (pole)
        - Consolidation in a slight upward channel (flag)
        - Continuation breakdown downward
        - Bearish continuation pattern
        """
        patterns = []
        prices = price_data['close'].values

        # Need sufficient data
        if len(prices) < 30:
            return patterns

        # Look for sharp downward moves followed by consolidation
        for i in range(10, len(prices) - 20):
            # Check for pole: strong downward move
            pole_start = i - 10
            pole_end = i
            pole_prices = prices[pole_start:pole_end]

            if len(pole_prices) < 10:
                continue

            pole_loss = (pole_prices[0] - pole_prices[-1]) / pole_prices[0]

            # Need at least 10% decline in the pole
            if pole_loss < 0.10:
                continue

            # Check pole strength (consistent downward movement)
            pole_slope = (pole_prices[-1] - pole_prices[0]) / len(pole_prices)
            if pole_slope >= 0:
                continue

            # Look for flag consolidation after pole
            flag_start = pole_end
            flag_end = min(pole_end + 15, len(prices))
            flag_prices = prices[flag_start:flag_end]

            if len(flag_prices) < 10:
                continue

            # Flag should be relatively flat or slightly upward
            flag_slope = (flag_prices[-1] - flag_prices[0]) / len(flag_prices)
            flag_range = np.max(flag_prices) - np.min(flag_prices)
            flag_volatility = flag_range / flag_prices[0]

            # Flag should be consolidating (low volatility compared to pole)
            if flag_volatility > 0.05:  # More than 5% range
                continue

            # Flag can slope slightly up or be flat, but not down
            if flag_slope < pole_slope * 0.3:  # Not a flag if sloping down too much
                continue

            # Check for breakdown continuation
            if flag_end >= len(prices) - 5:
                continue

            breakdown_prices = prices[flag_end:min(flag_end + 5, len(prices))]
            if len(breakdown_prices) == 0:
                continue

            # Breakdown should fall below flag low
            flag_low = np.min(flag_prices)
            if breakdown_prices[-1] >= flag_low * 1.00:
                continue  # No breakdown

            # Calculate confidence based on pole strength and flag tightness
            pole_strength = min(1.0, pole_loss / 0.20)  # Up to 20% loss
            flag_tightness = 1.0 - min(1.0, flag_volatility / 0.05)
            confidence = 0.6 + (0.2 * pole_strength) + (0.2 * flag_tightness)

            # Price target: pole height projected from breakdown
            pole_height = pole_prices[0] - pole_prices[-1]
            price_target = flag_low - pole_height

            # Stop loss above flag high
            flag_high = np.max(flag_prices)
            stop_loss = flag_high * 1.02

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.BEARISH_FLAG,
                confidence=confidence,
                start_date=price_data['date'].iloc[pole_start],
                end_date=price_data['date'].iloc[flag_end],
                direction='bearish',
                key_points={
                    'pole_start': pole_start,
                    'pole_end': pole_end,
                    'pole_loss': pole_loss,
                    'flag_high': flag_high,
                    'flag_low': flag_low,
                    'pole_height': pole_height
                },
                price_target=price_target,
                stop_loss=stop_loss,
                description=f"Bearish flag with {pole_loss*100:.1f}% pole, targeting ${price_target:.2f}"
            )

            patterns.append(pattern)
            logger.info(f"Detected bearish flag: pole loss {pole_loss*100:.1f}%, target ${price_target:.2f}")

        return patterns

    def _detect_consolidation(
        self,
        price_data: pd.DataFrame,
        peaks: np.ndarray,
        troughs: np.ndarray
    ) -> List[PatternResult]:
        """
        Detect consolidation/range-bound patterns

        Consolidation:
        - Sideways price movement
        - Clear support and resistance levels
        - Low volatility and trending
        - Neutral pattern awaiting breakout
        """
        patterns = []
        prices = price_data['close'].values

        # Need sufficient data
        if len(prices) < 30:
            return patterns

        # Use rolling window to detect consolidation
        window_size = 20

        for i in range(window_size, len(prices) - 10):
            window_start = i - window_size
            window_end = i
            window_prices = prices[window_start:window_end]

            # Calculate range statistics
            price_high = np.max(window_prices)
            price_low = np.min(window_prices)
            price_range = price_high - price_low
            price_mid = (price_high + price_low) / 2

            # Check if range is tight (consolidating)
            range_pct = price_range / price_mid

            # Consolidation should be in a relatively tight range (5-10%)
            if range_pct < 0.05 or range_pct > 0.15:
                continue

            # Check for horizontal movement (low slope)
            window_slope = abs((window_prices[-1] - window_prices[0]) / len(window_prices))
            avg_price = np.mean(window_prices)

            # Slope should be minimal for consolidation
            if window_slope > avg_price * 0.005:  # More than 0.5% per bar
                continue

            # Count touches of support and resistance
            resistance_touches = np.sum(window_prices > price_high * 0.98)
            support_touches = np.sum(window_prices < price_low * 1.02)

            # Need multiple touches of both levels
            if resistance_touches < 2 or support_touches < 2:
                continue

            # Calculate price stability (how much time spent in middle)
            middle_range = (price_mid * 0.97, price_mid * 1.03)
            in_middle = np.sum((window_prices >= middle_range[0]) & (window_prices <= middle_range[1]))
            stability = in_middle / len(window_prices)

            # Higher stability = better consolidation
            if stability < 0.3:
                continue

            # Calculate confidence based on tightness and stability
            tightness_score = 1.0 - min(1.0, (range_pct - 0.05) / 0.10)
            confidence = 0.5 + (0.25 * tightness_score) + (0.25 * stability)

            # Price targets at breakout levels
            price_target_up = price_high * 1.02
            price_target_down = price_low * 0.98

            # Stop loss on opposite side
            stop_loss_long = price_low * 0.98
            stop_loss_short = price_high * 1.02

            # Create pattern result
            pattern = PatternResult(
                pattern_type=PatternType.CONSOLIDATION,
                confidence=confidence,
                start_date=price_data['date'].iloc[window_start],
                end_date=price_data['date'].iloc[window_end],
                direction='neutral',
                key_points={
                    'resistance': price_high,
                    'support': price_low,
                    'midpoint': price_mid,
                    'range_pct': range_pct,
                    'stability': stability,
                    'resistance_touches': int(resistance_touches),
                    'support_touches': int(support_touches)
                },
                price_target=price_mid,  # Neutral - could go either way
                stop_loss=stop_loss_long,
                description=f"Consolidation between ${price_low:.2f} and ${price_high:.2f} ({range_pct*100:.1f}% range)"
            )

            patterns.append(pattern)
            logger.info(f"Detected consolidation: ${price_low:.2f} - ${price_high:.2f}, {range_pct*100:.1f}% range")

        return patterns


def find_peaks_and_troughs(
    prices: np.ndarray,
    prominence: float = 1.0,
    distance: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks (local maxima) and troughs (local minima) in price data

    Args:
        prices: Array of prices
        prominence: Required prominence of peaks/troughs
        distance: Minimum distance between peaks/troughs

    Returns:
        Tuple of (peaks_indices, troughs_indices)
    """
    # Find peaks (local maxima)
    peaks, _ = signal.find_peaks(prices, prominence=prominence, distance=distance)

    # Find troughs (local minima) by inverting prices
    troughs, _ = signal.find_peaks(-prices, prominence=prominence, distance=distance)

    logger.debug(f"Found {len(peaks)} peaks and {len(troughs)} troughs")

    return peaks, troughs


def calculate_pattern_confidence(
    key_prices: np.ndarray,
    pattern_type: str,
    volume_confirmation: bool = False
) -> float:
    """
    Calculate confidence score for a detected pattern

    Args:
        key_prices: Array of key price points in pattern (e.g., two peaks)
        pattern_type: Type of pattern
        volume_confirmation: Whether volume confirms the pattern

    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.5  # Base confidence

    # Check price similarity (for double patterns)
    if len(key_prices) >= 2:
        mean_price = np.mean(key_prices)
        price_std = np.std(key_prices)
        price_cv = price_std / mean_price if mean_price > 0 else 1.0

        # Lower coefficient of variation = higher confidence
        similarity_score = max(0, 1.0 - price_cv * 10)  # Scale CV to 0-1 range
        confidence += similarity_score * 0.3

    # Volume confirmation adds confidence
    if volume_confirmation:
        confidence += 0.2

    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, confidence))

    return confidence
