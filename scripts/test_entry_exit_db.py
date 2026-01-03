#!/usr/bin/env python3
"""
Test Entry/Exit Engine using tickerdata table with split-normalized data.

Uses PriceService for database access instead of yfinance.
Includes OBV and AOBV (OBV SMA) for volume-based signals.

Usage:
    PYTHONPATH=./src:. python scripts/test_entry_exit_db.py AAPL
    PYTHONPATH=./src:. python scripts/test_entry_exit_db.py NVDA --days 365
"""

import argparse
import sys
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Database access
from investigator.domain.services.market_data.price_service import PriceService

# Technical indicators
from investigator.infrastructure.indicators.technical_indicators import get_technical_calculator

# Entry/Exit engine
from investigator.domain.services.signals.entry_exit_engine import EntryExitEngine


def calculate_obv_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract OBV and AOBV metrics from enhanced DataFrame.

    Uses built-in indicators from TechnicalIndicatorCalculator:
    - OBV: On-Balance Volume
    - AOBV_20/AOBV_50: OBV SMA for trend detection
    - OBV_Trend: 1=bullish, -1=bearish, 0=neutral
    - OBV_Signal/OBV_Histogram: MACD-style signal line
    - OBV_ROC_10/20: Rate of change for momentum

    Args:
        df: Enhanced DataFrame with pre-calculated OBV indicators

    Returns:
        Dict with OBV metrics and divergence analysis
    """
    obv = df['OBV'].iloc[-1]
    aobv_20 = df['AOBV_20'].iloc[-1]
    aobv_50 = df['AOBV_50'].iloc[-1]
    obv_trend_val = df['OBV_Trend'].iloc[-1]
    volume_trend_val = df['Volume_Trend'].iloc[-1]

    # Convert numeric trend to string
    obv_trend = "bullish" if obv_trend_val == 1 else ("bearish" if obv_trend_val == -1 else "neutral")
    volume_trend = "increasing" if volume_trend_val == 1 else ("decreasing" if volume_trend_val == -1 else "neutral")

    # Get OBV ROC for divergence detection
    obv_roc_20 = df['OBV_ROC_20'].iloc[-1] if 'OBV_ROC_20' in df.columns else 0
    price_change_20d = df['Price_Change_10D'].iloc[-1] * 2 if 'Price_Change_10D' in df.columns else 0

    # Divergence detection
    divergence = "none"
    if price_change_20d > 5 and obv_roc_20 < -5:
        divergence = "bearish"  # Price up but OBV down - distribution
    elif price_change_20d < -5 and obv_roc_20 > 5:
        divergence = "bullish"  # Price down but OBV up - accumulation

    return {
        "obv": obv,
        "aobv_20": aobv_20,
        "aobv_50": aobv_50,
        "obv_signal": df['OBV_Signal'].iloc[-1],
        "obv_histogram": df['OBV_Histogram'].iloc[-1],
        "obv_trend": obv_trend,
        "obv_roc_10": df['OBV_ROC_10'].iloc[-1],
        "obv_roc_20": obv_roc_20,
        "price_change_20d": round(price_change_20d, 2),
        "divergence": divergence,
        "volume_trend": volume_trend,
        "vol_sma_20": df['Volume_SMA_20'].iloc[-1],
        "vol_sma_50": df['Volume_SMA_50'].iloc[-1],
    }


def get_support_resistance(df: pd.DataFrame, current_price: float) -> Dict[str, float]:
    """
    Calculate support and resistance levels from technical indicators DataFrame.
    Returns a dict matching the format expected by EntryExitEngine.
    """
    # Get from DataFrame if available
    support_1 = df.get('Support_1', pd.Series([current_price * 0.95])).iloc[-1]
    support_2 = df.get('Support_2', pd.Series([current_price * 0.90])).iloc[-1]
    resistance_1 = df.get('Resistance_1', pd.Series([current_price * 1.05])).iloc[-1]
    resistance_2 = df.get('Resistance_2', pd.Series([current_price * 1.10])).iloc[-1]

    # Fallback to SMA/BB if columns don't exist
    if pd.isna(support_1):
        support_1 = df.get('BB_Lower', pd.Series([current_price * 0.95])).iloc[-1]
    if pd.isna(resistance_1):
        resistance_1 = df.get('BB_Upper', pd.Series([current_price * 1.05])).iloc[-1]

    return {
        'support_1': float(support_1) if not pd.isna(support_1) else current_price * 0.95,
        'support_2': float(support_2) if not pd.isna(support_2) else current_price * 0.90,
        'resistance_1': float(resistance_1) if not pd.isna(resistance_1) else current_price * 1.05,
        'resistance_2': float(resistance_2) if not pd.isna(resistance_2) else current_price * 1.10,
    }


def main():
    parser = argparse.ArgumentParser(description='Test Entry/Exit Engine with database data')
    parser.add_argument('symbol', type=str, help='Stock symbol to analyze')
    parser.add_argument('--days', type=int, default=252, help='Days of history (default: 252)')
    parser.add_argument('--fair-value', type=float, default=None, help='Override fair value')
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print(f"\n{'='*60}")
    print(f"Entry/Exit Analysis: {symbol}")
    print(f"Data Source: tickerdata table (split-adjusted)")
    print(f"{'='*60}\n")

    # 1. Fetch data from database
    print("Fetching price history from database...")
    price_service = PriceService()
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    df = price_service.get_price_history(symbol, start_date, end_date)

    if df.empty:
        print(f"ERROR: No data found for {symbol}")
        sys.exit(1)

    print(f"  Retrieved {len(df)} trading days")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # 2. Calculate technical indicators
    print("\nCalculating technical indicators...")
    calculator = get_technical_calculator()

    # Convert Decimal types to float (database returns Decimal)
    df_calc = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_calc.columns:
            df_calc[col] = df_calc[col].astype(float)

    # Rename columns to match expected format
    df_calc = df_calc.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Calculate all indicators
    df_enhanced = calculator.calculate_all_indicators(df_calc, symbol)

    # Get current values
    current_price = float(df_enhanced['Close'].iloc[-1])

    print(f"\n  Current Price: ${current_price:.2f}")
    print(f"  SMA 20: ${df_enhanced['SMA_20'].iloc[-1]:.2f}")
    print(f"  SMA 50: ${df_enhanced['SMA_50'].iloc[-1]:.2f}")
    print(f"  SMA 200: ${df_enhanced['SMA_200'].iloc[-1]:.2f}")
    print(f"  RSI(14): {df_enhanced['RSI_14'].iloc[-1]:.1f}")
    print(f"  MACD: {df_enhanced['MACD'].iloc[-1]:.2f}")
    print(f"  MACD Signal: {df_enhanced['MACD_Signal'].iloc[-1]:.2f}")

    # 3. Calculate OBV metrics with AOBV
    print("\nCalculating OBV/AOBV metrics...")
    obv_metrics = calculate_obv_metrics(df_enhanced)

    print(f"\n  OBV: {obv_metrics['obv']:,.0f}")
    print(f"  AOBV(20): {obv_metrics['aobv_20']:,.0f}")
    print(f"  AOBV(50): {obv_metrics['aobv_50']:,.0f}")
    print(f"  OBV Signal: {obv_metrics['obv_signal']:,.0f}")
    print(f"  OBV Histogram: {obv_metrics['obv_histogram']:,.0f}")
    print(f"  OBV Trend: {obv_metrics['obv_trend']}")
    print(f"  OBV ROC(10): {obv_metrics['obv_roc_10']:.1f}%")
    print(f"  OBV ROC(20): {obv_metrics['obv_roc_20']:.1f}%")
    print(f"  Price Change (20d): {obv_metrics['price_change_20d']:.1f}%")
    print(f"  Divergence: {obv_metrics['divergence']}")
    print(f"  Volume Trend: {obv_metrics['volume_trend']}")

    # 4. Build indicators dict for entry/exit engine
    indicators = {
        "rsi_14": float(df_enhanced['RSI_14'].iloc[-1]),
        "macd": float(df_enhanced['MACD'].iloc[-1]),
        "macd_signal": float(df_enhanced['MACD_Signal'].iloc[-1]),
        "sma_20": float(df_enhanced['SMA_20'].iloc[-1]),
        "sma_50": float(df_enhanced['SMA_50'].iloc[-1]),
        "sma_200": float(df_enhanced['SMA_200'].iloc[-1]),
        "bb_upper": float(df_enhanced.get('BB_Upper', pd.Series([current_price * 1.1])).iloc[-1]),
        "bb_lower": float(df_enhanced.get('BB_Lower', pd.Series([current_price * 0.9])).iloc[-1]),
        "bb_middle": float(df_enhanced.get('BB_Middle', pd.Series([current_price])).iloc[-1]),
        "atr_14": float(df_enhanced['ATR_14'].iloc[-1]),
        # OBV metrics
        "obv": obv_metrics['obv'],
        "aobv_20": obv_metrics['aobv_20'],
        "obv_trend": obv_metrics['obv_trend'],
        "obv_divergence": obv_metrics['divergence'],
        "volume_trend": obv_metrics['volume_trend'],
        # ADX and Directional Indicators (keys match DataFrame column names)
        "ADX_14": float(df_enhanced['ADX_14'].iloc[-1]),
        "Plus_DI": float(df_enhanced['Plus_DI'].iloc[-1]),
        "Minus_DI": float(df_enhanced['Minus_DI'].iloc[-1]),
        # MFI
        "MFI_14": float(df_enhanced['MFI_14'].iloc[-1]),
        # Stochastic and Williams %R
        "Stoch_K": float(df_enhanced['Stoch_K'].iloc[-1]),
        "Stoch_D": float(df_enhanced['Stoch_D'].iloc[-1]),
        "Williams_R": float(df_enhanced['Williams_R'].iloc[-1]),
        # EMA for crossover signals
        "EMA_8": float(df_enhanced['EMA_8'].iloc[-1]),
        "EMA_12": float(df_enhanced['EMA_12'].iloc[-1]),
        "EMA_21": float(df_enhanced['EMA_21'].iloc[-1]),
        "EMA_26": float(df_enhanced['EMA_26'].iloc[-1]),
        "EMA_50": float(df_enhanced['EMA_50'].iloc[-1]),
        "EMA_200": float(df_enhanced['EMA_200'].iloc[-1]),
    }

    # 5. Get support/resistance
    support_resistance = get_support_resistance(df_enhanced, current_price)

    print(f"\n  Support 1: ${support_resistance['support_1']:.2f}")
    print(f"  Support 2: ${support_resistance['support_2']:.2f}")
    print(f"  Resistance 1: ${support_resistance['resistance_1']:.2f}")
    print(f"  Resistance 2: ${support_resistance['resistance_2']:.2f}")

    # 6. Run Entry/Exit Engine
    print("\n" + "="*60)
    print("ENTRY/EXIT SIGNAL ANALYSIS")
    print("="*60 + "\n")

    engine = EntryExitEngine()

    # Use provided fair value or estimate from SMA 200
    fair_value = args.fair_value or float(df_enhanced['SMA_200'].iloc[-1]) * 1.05
    print(f"Fair Value Estimate: ${fair_value:.2f}")

    # Create valuation dict for the engine
    valuation = {
        'fair_value': fair_value,
        'upside': (fair_value - current_price) / current_price,
    }

    # Generate entry signals (expects price_data DataFrame)
    entry_signals = engine.generate_entry_signals(
        price_data=df_enhanced,
        indicators=indicators,
        valuation=valuation,
        support_resistance=support_resistance,
    )

    print(f"\n--- ENTRY SIGNALS ({len(entry_signals)} found) ---")
    for signal in entry_signals:
        print(f"\n  Type: {signal.signal_type.value if hasattr(signal.signal_type, 'value') else signal.signal_type}")
        print(f"  Price: ${signal.price_level:.2f}")
        print(f"  Confidence: {signal.confidence.value if hasattr(signal.confidence, 'value') else signal.confidence}")
        print(f"  Rationale: {signal.rationale}")
        print(f"  Risk/Reward: {signal.risk_reward_ratio:.1f}:1")
        print(f"  Stop Loss: ${signal.stop_loss:.2f} ({signal.stop_loss_pct:.1f}%)")
        print(f"  Target: ${signal.target_price:.2f}")

    # Generate exit signals (expects price_data DataFrame)
    position_info = {
        'entry_price': current_price * 0.95,  # Assume entered 5% lower
        'current_price': current_price,
        'fair_value': fair_value,
    }
    exit_signals = engine.generate_exit_signals(
        price_data=df_enhanced,
        indicators=indicators,
        position_info=position_info,
    )

    print(f"\n--- EXIT SIGNALS ({len(exit_signals)} found) ---")
    for signal in exit_signals:
        print(f"\n  Type: {signal.signal_type.value if hasattr(signal.signal_type, 'value') else signal.signal_type}")
        print(f"  Price: ${signal.price_level:.2f}")
        print(f"  Confidence: {signal.confidence.value if hasattr(signal.confidence, 'value') else signal.confidence}")
        print(f"  Rationale: {signal.rationale}")
        print(f"  Urgency: {signal.urgency}")

    # Calculate optimal entry zone
    support_levels = [support_resistance['support_2'], support_resistance['support_1']]
    resistance_levels = [support_resistance['resistance_1'], support_resistance['resistance_2']]
    entry_zone = engine.calculate_optimal_entry_zone(
        current_price=current_price,
        fair_value=fair_value,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        volatility=indicators['atr_14'] / current_price,
        atr=indicators['atr_14'],
    )

    print(f"\n--- OPTIMAL ENTRY ZONE ---")
    print(f"  Lower Bound: ${entry_zone.lower_bound:.2f}")
    print(f"  Ideal Entry: ${entry_zone.ideal_entry:.2f}")
    print(f"  Upper Bound: ${entry_zone.upper_bound:.2f}")
    print(f"  Timing: {entry_zone.timing.value if hasattr(entry_zone.timing, 'value') else entry_zone.timing}")

    # OBV-based recommendation
    print(f"\n--- OBV ANALYSIS ---")
    if obv_metrics['divergence'] == 'bullish':
        print("  BULLISH DIVERGENCE: Price down but OBV up - potential accumulation")
        print("  Consider: Increasing position on pullbacks")
    elif obv_metrics['divergence'] == 'bearish':
        print("  BEARISH DIVERGENCE: Price up but OBV down - potential distribution")
        print("  Consider: Taking profits or reducing position")
    else:
        if obv_metrics['obv_trend'] == 'bullish' and obv_metrics['volume_trend'] == 'increasing':
            print("  STRONG BULLISH: OBV trending up with increasing volume")
        elif obv_metrics['obv_trend'] == 'bearish' and obv_metrics['volume_trend'] == 'increasing':
            print("  STRONG BEARISH: OBV trending down with increasing volume")
        else:
            print(f"  NEUTRAL: OBV {obv_metrics['obv_trend']}, Volume {obv_metrics['volume_trend']}")

    print(f"\n{'='*60}")
    print("Analysis complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
