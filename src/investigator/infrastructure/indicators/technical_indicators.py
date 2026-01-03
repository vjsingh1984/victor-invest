#!/usr/bin/env python3
"""
Technical Indicators Calculation Module
Centralized calculation of all technical analysis indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """
    Centralized calculator for all technical indicators
    Ensures consistent calculations across all data flows (cache and direct)
    """
    
    def __init__(self):
        self.logger = logger
        
    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate ALL technical indicators on the full dataset
        Returns enhanced DataFrame with all indicators as columns
        
        Args:
            df: Raw OHLCV DataFrame with 365 days of data
            symbol: Stock symbol for logging
            
        Returns:
            Enhanced DataFrame with all technical indicators calculated
        """
        try:
            # Make a copy to avoid modifying original
            enhanced_df = df.copy()
            
            # Ensure we have required columns (handle both uppercase and lowercase)
            # Map lowercase to uppercase if needed
            column_mapping = {}
            if 'open' in enhanced_df.columns and 'Open' not in enhanced_df.columns:
                column_mapping.update({
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                })
                enhanced_df = enhanced_df.rename(columns=column_mapping)
                self.logger.debug(f"Mapped lowercase columns to uppercase for {symbol}")
            
            # Check for required columns after mapping
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in enhanced_df.columns:
                    enhanced_df[col] = 0.0
                    self.logger.warning(f"Missing {col} column for {symbol}, filled with 0.0")
            
            # Fill any NaN values
            enhanced_df = enhanced_df.ffill().bfill()
            
            # Calculate Moving Averages
            self._calculate_moving_averages(enhanced_df)
            
            # Calculate Momentum Indicators  
            self._calculate_momentum_indicators(enhanced_df)
            
            # Calculate Volatility Indicators
            self._calculate_volatility_indicators(enhanced_df)
            
            # Calculate Volume Indicators
            self._calculate_volume_indicators(enhanced_df)
            
            # Calculate Support/Resistance and Fibonacci Levels
            self._calculate_support_resistance_fibonacci(enhanced_df)
            
            # Add metadata
            enhanced_df['Symbol'] = symbol
            enhanced_df['Calculation_Date'] = datetime.now()
            
            self.logger.info(f"Calculated {len(enhanced_df.columns)} indicators for {symbol} ({len(enhanced_df)} data points)")
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            return df  # Return original if calculation fails
    
    def _calculate_moving_averages(self, df: pd.DataFrame):
        """Calculate Simple and Exponential Moving Averages"""
        # Simple Moving Averages
        sma_periods = [5, 10, 12, 20, 26, 50, 100, 200]
        for period in sma_periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
        
        # Exponential Moving Averages
        # Include Fibonacci-based periods (8, 13, 21) for swing trading
        ema_periods = [5, 8, 9, 10, 12, 13, 20, 21, 26, 50, 100, 200]
        for period in ema_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # EMA Trend indicators (for quick reference)
        df['EMA_8_21_Trend'] = np.where(df['EMA_8'] > df['EMA_21'], 1, -1)  # Short-term
        df['EMA_12_26_Trend'] = np.where(df['EMA_12'] > df['EMA_26'], 1, -1)  # Medium-term (MACD)
        df['EMA_50_200_Trend'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)  # Long-term
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame):
        """Calculate momentum-based indicators"""
        # RSI (Relative Strength Index)
        for period in [9, 14, 21]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df, 14)
        
        # Rate of Change
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
        
        # Money Flow Index
        df['MFI_14'] = self._calculate_mfi(df, 14)

        # ADX (Average Directional Index) - Trend Strength
        df['ADX_14'], df['Plus_DI'], df['Minus_DI'] = self._calculate_adx(df, 14)
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame):
        """Calculate volatility-based indicators"""
        # Bollinger Bands
        for period in [20]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = sma + (std * 2)
            df[f'BB_Middle_{period}'] = sma
            df[f'BB_Lower_{period}'] = sma - (std * 2)
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / df[f'BB_Width_{period}']
        
        # Simplified column names for backward compatibility
        df['BB_Upper'] = df['BB_Upper_20']
        df['BB_Middle'] = df['BB_Middle_20']
        df['BB_Lower'] = df['BB_Lower_20']
        df['BB_Width'] = df['BB_Width_20']
        df['BB_Position'] = df['BB_Position_20']
        
        # Average True Range
        df['ATR_14'] = self._calculate_atr(df, 14)
        
        # Volatility (20-day)
        df['Volatility_20'] = df['Close'].rolling(window=20).std() * np.sqrt(252) * 100
    
    def _calculate_volume_indicators(self, df: pd.DataFrame):
        """Calculate volume-based indicators"""
        # Volume SMA
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_SMA_50'] = df['Volume'].rolling(window=50, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # On-Balance Volume
        df['OBV'] = self._calculate_obv(df)

        # AOBV (Accumulated OBV / OBV SMA) - smoothed OBV for trend detection
        df['AOBV_20'] = df['OBV'].rolling(window=20, min_periods=1).mean()
        df['AOBV_50'] = df['OBV'].rolling(window=50, min_periods=1).mean()

        # OBV Signal line (9-period EMA of OBV, similar to MACD signal)
        df['OBV_Signal'] = df['OBV'].ewm(span=9, adjust=False).mean()
        df['OBV_Histogram'] = df['OBV'] - df['OBV_Signal']

        # OBV Trend (1 = bullish, -1 = bearish, 0 = neutral)
        df['OBV_Trend'] = np.where(
            df['AOBV_20'] > df['AOBV_50'], 1,
            np.where(df['AOBV_20'] < df['AOBV_50'], -1, 0)
        )

        # OBV Rate of Change (momentum of OBV)
        df['OBV_ROC_10'] = df['OBV'].pct_change(periods=10) * 100
        df['OBV_ROC_20'] = df['OBV'].pct_change(periods=20) * 100

        # Volume Price Trend
        df['VPT'] = self._calculate_vpt(df)

        # Accumulation/Distribution Line
        df['AD'] = self._calculate_ad(df)

        # Volume Weighted Average Price
        df['VWAP'] = self._calculate_vwap(df)

        # Price change indicators
        df['Price_Change_1D'] = df['Close'].pct_change() * 100
        df['Price_Change_5D'] = df['Close'].pct_change(periods=5) * 100
        df['Price_Change_10D'] = df['Close'].pct_change(periods=10) * 100

        # Volume trend indicator (1 = increasing, -1 = decreasing)
        df['Volume_Trend'] = np.where(
            df['Volume_SMA_20'] > df['Volume_SMA_50'], 1,
            np.where(df['Volume_SMA_20'] < df['Volume_SMA_50'], -1, 0)
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['High'].rolling(window=k_period, min_periods=1).max()
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.inf)
        stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
        return stoch_k, stoch_d
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['High'].rolling(window=period, min_periods=1).max()
        low_min = df['Low'].rolling(window=period, min_periods=1).min()
        return -100 * (high_max - df['Close']) / (high_max - low_min).replace(0, np.inf)
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']

        money_flow_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(money_flow_diff > 0, 0).rolling(window=period, min_periods=1).sum()
        negative_flow = raw_money_flow.where(money_flow_diff < 0, 0).rolling(window=period, min_periods=1).sum()

        money_ratio = positive_flow / negative_flow.replace(0, np.inf)
        return 100 - (100 / (1 + money_ratio))

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX) and Directional Indicators.

        ADX measures trend strength (not direction):
        - ADX > 25: Strong trend (trade with trend)
        - ADX < 20: Weak/no trend (range-bound, mean reversion)
        - ADX 20-25: Developing trend

        +DI > -DI: Bullish trend
        -DI > +DI: Bearish trend

        Returns:
            Tuple of (ADX, +DI, -DI) series
        """
        # Calculate True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # Calculate Directional Movement
        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']

        # +DM and -DM
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth TR, +DM, -DM with Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1 / period
        atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        smooth_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        smooth_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (smooth_plus_dm / atr.replace(0, np.inf))
        minus_di = 100 * (smooth_minus_dm / atr.replace(0, np.inf))

        # Calculate DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * (di_diff / di_sum.replace(0, np.inf))

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx, plus_di, minus_di
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period, min_periods=1).mean()
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        price_change_pct = df['Close'].pct_change()
        vpt = (price_change_pct * df['Volume']).cumsum()
        return vpt.fillna(0)
    
    def _calculate_ad(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.inf)
        ad = (clv * df['Volume']).cumsum()
        return ad.fillna(0)
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    def _calculate_support_resistance_fibonacci(self, df: pd.DataFrame):
        """Calculate support/resistance levels and Fibonacci retracements"""
        # Calculate rolling highs and lows for different periods
        periods = [20, 50, 100, 200]
        
        for period in periods:
            df[f'High_{period}d'] = df['High'].rolling(window=period, min_periods=1).max()
            df[f'Low_{period}d'] = df['Low'].rolling(window=period, min_periods=1).min()
        
        # Calculate Fibonacci retracement levels based on 52-week high/low
        period_52w = min(252, len(df))  # Use 252 trading days or available data
        high_52w = df['High'].tail(period_52w).max()
        low_52w = df['Low'].tail(period_52w).min()
        
        # Calculate Fibonacci levels
        diff = high_52w - low_52w
        df['Fib_0'] = high_52w  # 0% retracement (high)
        df['Fib_23_6'] = high_52w - (diff * 0.236)
        df['Fib_38_2'] = high_52w - (diff * 0.382)
        df['Fib_50_0'] = high_52w - (diff * 0.500)
        df['Fib_61_8'] = high_52w - (diff * 0.618)
        df['Fib_78_6'] = high_52w - (diff * 0.786)
        df['Fib_100'] = low_52w  # 100% retracement (low)
        
        # Add 52-week high/low for reference
        df['High_52w'] = high_52w
        df['Low_52w'] = low_52w
        
        # Calculate pivot points (Traditional)
        df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Pivot_R1'] = 2 * df['Pivot_Point'] - df['Low']
        df['Pivot_R2'] = df['Pivot_Point'] + (df['High'] - df['Low'])
        df['Pivot_R3'] = df['High'] + 2 * (df['Pivot_Point'] - df['Low'])
        df['Pivot_S1'] = 2 * df['Pivot_Point'] - df['High']
        df['Pivot_S2'] = df['Pivot_Point'] - (df['High'] - df['Low'])
        df['Pivot_S3'] = df['Low'] - 2 * (df['High'] - df['Pivot_Point'])
        
        # Identify recent support and resistance levels
        lookback = min(20, len(df))
        recent_data = df.tail(lookback)
        
        # Simple support/resistance based on recent highs/lows
        df['Support_1'] = recent_data['Low'].min()
        df['Support_2'] = recent_data['Low'].nsmallest(2).iloc[-1] if len(recent_data) > 1 else df['Support_1']
        df['Resistance_1'] = recent_data['High'].max()
        df['Resistance_2'] = recent_data['High'].nlargest(2).iloc[-1] if len(recent_data) > 1 else df['Resistance_1']
    
    def extract_recent_data_for_llm(self, enhanced_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Extract recent N days from enhanced DataFrame for LLM analysis
        All indicators are already calculated on the full dataset
        
        Args:
            enhanced_df: DataFrame with all indicators calculated
            days: Number of recent days to extract (default 30)
            
        Returns:
            DataFrame with recent data and all pre-calculated indicators
        """
        return enhanced_df.tail(days).copy()


# Singleton instance
_calculator = None

def get_technical_calculator() -> TechnicalIndicatorCalculator:
    """Get singleton instance of TechnicalIndicatorCalculator"""
    global _calculator
    if _calculator is None:
        _calculator = TechnicalIndicatorCalculator()
    return _calculator