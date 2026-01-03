#!/usr/bin/env python3
"""
Support and Resistance Level Detection

Detects key price levels using local extrema analysis and clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def detect_support_resistance_levels(price_data: pd.DataFrame,
                                     window: int = 20,
                                     num_levels: int = 5,
                                     tolerance: float = 0.02) -> Dict:
    """
    Detect support and resistance levels from price data

    Uses local maxima/minima detection with clustering to identify
    key price levels that act as support (floor) or resistance (ceiling).

    Args:
        price_data: DataFrame with OHLCV data (columns: high, low, close)
        window: Window size for local extrema detection (default: 20)
        num_levels: Number of top levels to return for each type (default: 5)
        tolerance: Percentage tolerance for clustering nearby levels (default: 0.02 = 2%)

    Returns:
        Dictionary with:
            - current_price: Latest closing price
            - resistance_levels: List of resistance prices above current
            - support_levels: List of support prices below current
            - nearest_resistance: Closest resistance level
            - nearest_support: Closest support level
            - distance_to_resistance: Percentage distance to nearest resistance
            - distance_to_support: Percentage distance to nearest support
    """
    try:
        from scipy.signal import argrelextrema

        if len(price_data) < window * 2:
            logger.warning(f"Insufficient data for support/resistance detection. Need at least {window * 2} bars, have {len(price_data)}")
            return _empty_result(price_data)

        # Extract price arrays
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        current_price = closes[-1]

        # Find local maxima (resistance candidates)
        max_idx = argrelextrema(highs, np.greater, order=window)[0]
        resistance_prices = highs[max_idx]

        # Find local minima (support candidates)
        min_idx = argrelextrema(lows, np.less, order=window)[0]
        support_prices = lows[min_idx]

        logger.info(f"Found {len(resistance_prices)} resistance candidates and {len(support_prices)} support candidates")

        # Cluster nearby levels
        resistance_levels = _cluster_levels(resistance_prices, tolerance)
        support_levels = _cluster_levels(support_prices, tolerance)

        # Filter and sort by proximity to current price
        resistance_levels = sorted(
            [r for r in resistance_levels if r > current_price],
            key=lambda x: x - current_price
        )[:num_levels]

        support_levels = sorted(
            [s for s in support_levels if s < current_price],
            key=lambda x: current_price - x
        )[:num_levels]

        # Calculate distances
        nearest_resistance = resistance_levels[0] if resistance_levels else None
        nearest_support = support_levels[0] if support_levels else None

        result = {
            'current_price': float(current_price),
            'resistance_levels': [float(r) for r in resistance_levels],
            'support_levels': [float(s) for s in support_levels],
            'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
            'nearest_support': float(nearest_support) if nearest_support else None,
        }

        if nearest_resistance:
            result['distance_to_resistance'] = float(((nearest_resistance / current_price) - 1) * 100)
            result['resistance_pct'] = float(((nearest_resistance / current_price) - 1) * 100)

        if nearest_support:
            result['distance_to_support'] = float((1 - (nearest_support / current_price)) * 100)
            result['support_pct'] = float((1 - (nearest_support / current_price)) * 100)

        logger.info(f"Detected {len(resistance_levels)} resistance and {len(support_levels)} support levels")

        return result

    except ImportError:
        logger.error("scipy not available - cannot detect support/resistance levels")
        return _empty_result(price_data)

    except Exception as e:
        logger.error(f"Error detecting support/resistance levels: {e}")
        return _empty_result(price_data)


def _cluster_levels(levels: np.ndarray, tolerance: float = 0.02) -> List[float]:
    """
    Cluster nearby price levels into single representative levels

    Args:
        levels: Array of price levels
        tolerance: Percentage tolerance for clustering (default: 0.02 = 2%)

    Returns:
        List of clustered price levels
    """
    if len(levels) == 0:
        return []

    clustered = []
    levels_sorted = sorted(levels)
    current_cluster = [levels_sorted[0]]

    for level in levels_sorted[1:]:
        # If within tolerance of current cluster, add to it
        if (level - current_cluster[-1]) / current_cluster[-1] < tolerance:
            current_cluster.append(level)
        else:
            # Otherwise, save current cluster and start new one
            clustered.append(np.mean(current_cluster))
            current_cluster = [level]

    # Don't forget the last cluster
    clustered.append(np.mean(current_cluster))

    return clustered


def _empty_result(price_data: pd.DataFrame) -> Dict:
    """Return empty result structure with current price"""
    try:
        current_price = float(price_data['close'].iloc[-1])
    except:
        current_price = 0.0

    return {
        'current_price': current_price,
        'resistance_levels': [],
        'support_levels': [],
        'nearest_resistance': None,
        'nearest_support': None,
        'distance_to_resistance': None,
        'distance_to_support': None
    }


def format_support_resistance_text(sr_levels: Dict) -> str:
    """
    Format support/resistance levels as human-readable text

    Args:
        sr_levels: Dictionary from detect_support_resistance_levels()

    Returns:
        Formatted text description
    """
    if not sr_levels or not (sr_levels.get('support_levels') or sr_levels.get('resistance_levels')):
        return "No significant support/resistance levels detected."

    lines = []
    current_price = sr_levels.get('current_price', 0)

    lines.append(f"Current Price: ${current_price:.2f}")

    # Nearest support
    if nearest_support := sr_levels.get('nearest_support'):
        distance = sr_levels.get('distance_to_support', 0)
        lines.append(f"Nearest Support: ${nearest_support:.2f} ({distance:.1f}% below)")

    # Nearest resistance
    if nearest_resistance := sr_levels.get('nearest_resistance'):
        distance = sr_levels.get('distance_to_resistance', 0)
        lines.append(f"Nearest Resistance: ${nearest_resistance:.2f} ({distance:.1f}% above)")

    # All support levels
    if support_levels := sr_levels.get('support_levels'):
        lines.append(f"Support Levels: {', '.join([f'${s:.2f}' for s in support_levels])}")

    # All resistance levels
    if resistance_levels := sr_levels.get('resistance_levels'):
        lines.append(f"Resistance Levels: {', '.join([f'${r:.2f}' for r in resistance_levels])}")

    return "\n".join(lines)
