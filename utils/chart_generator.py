#!/usr/bin/env python3
"""
InvestiGator - Chart Generation Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Chart Generation Module for InvestiGator
Handles technical and fundamental analysis chart creation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
import json

try:
    import matplotlib
    # Force a non-interactive backend to avoid GUI dependencies that can segfault in headless mode
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib unavailable or failed to initialize - chart generation disabled")

# Try to import talib for technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available - some technical indicators may be limited")

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Handles chart generation for technical and fundamental analysis"""
    
    def __init__(self, charts_dir: Path):
        """Initialize chart generator"""
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - chart generation disabled")
            return
        
        # Set matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                logger.warning("Seaborn style not available, using default")
    
    def generate_technical_chart(self, symbol: str, price_data: pd.DataFrame,
                                sr_levels: dict = None) -> str:
        """
        Generate comprehensive technical analysis chart with Fibonacci levels

        Args:
            symbol: Stock symbol
            price_data: DataFrame with OHLCV and indicator data
            sr_levels: Optional dict with detected support/resistance levels

        Returns:
            Path to generated chart or empty string if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""
            
        try:
            # Normalize column names to handle both uppercase and lowercase
            price_data = self._normalize_column_names(price_data)
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Main price chart with candlesticks/line
            ax1.plot(price_data.index, price_data['close'], 'b-', linewidth=1.5, label='Close')
            
            # Add moving averages if available
            if 'sma_20' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_20'], 'g-', alpha=0.7, label='SMA 20')
            if 'sma_50' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_50'], 'r-', alpha=0.7, label='SMA 50')
            if 'sma_200' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_200'], 'm-', alpha=0.7, label='SMA 200')
            
            # Add Bollinger Bands if available
            if all(col in price_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                ax1.fill_between(price_data.index, price_data['bb_upper'], price_data['bb_lower'],
                               alpha=0.1, color='gray', label='Bollinger Bands')
                ax1.plot(price_data.index, price_data['bb_middle'], 'k--', alpha=0.5, linewidth=0.8)
            
            # Add support/resistance levels
            self._add_support_resistance_levels(ax1, price_data, sr_levels)
            
            # Add Fibonacci retracement levels
            self._add_fibonacci_levels(ax1, price_data)
            
            # Volume subplot
            colors = ['g' if price_data['close'].iloc[i] >= price_data['open'].iloc[i] else 'r' 
                     for i in range(len(price_data))]
            ax2.bar(price_data.index, price_data['volume'], color=colors, alpha=0.7)
            
            # Add OBV line if available
            if 'obv' in price_data.columns:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(price_data.index, price_data['obv'], 'b-', linewidth=1, label='OBV')
                ax2_twin.set_ylabel('OBV')
                ax2_twin.yaxis.label.set_color('blue')
                ax2_twin.tick_params(axis='y', colors='blue')
            
            # MACD subplot
            if all(col in price_data.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                ax3.plot(price_data.index, price_data['macd'], 'b-', label='MACD')
                ax3.plot(price_data.index, price_data['macd_signal'], 'r-', label='Signal')
                ax3.bar(price_data.index, price_data['macd_histogram'], alpha=0.3, label='Histogram')
                ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax3.legend(loc='upper left', fontsize=8)
            
            # Formatting
            ax1.set_title(f'{symbol} Technical Analysis', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            ax3.set_ylabel('MACD')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)
            
            # Format x-axis dates
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.charts_dir / f"{symbol}_technical_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated technical chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating technical chart: {e}")
            return ""
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to lowercase for consistent access
        
        Args:
            df: DataFrame with potentially mixed case column names
            
        Returns:
            DataFrame with normalized lowercase column names
        """
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Create mapping for common column variations
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower()
            # Handle common variations
            if lower_col in ['close', 'high', 'low', 'open', 'volume']:
                column_mapping[col] = lower_col
            elif lower_col.startswith('sma_'):
                column_mapping[col] = lower_col
            elif lower_col.startswith('ema_'):
                column_mapping[col] = lower_col
            elif lower_col.startswith('bb_'):
                column_mapping[col] = lower_col
            elif lower_col in ['macd', 'macd_signal', 'macd_histogram']:
                column_mapping[col] = lower_col
            elif lower_col == 'obv':
                column_mapping[col] = 'obv'
            else:
                column_mapping[col] = col
        
        # Rename columns
        normalized_df = normalized_df.rename(columns=column_mapping)
        
        return normalized_df
    
    def _add_support_resistance_levels(self, ax, price_data: pd.DataFrame, sr_levels: dict = None):
        """Add support and resistance levels to chart using detected levels or fallback calculation"""
        try:
            # Use detected levels if available
            if sr_levels and (sr_levels.get('support_levels') or sr_levels.get('resistance_levels')):
                # Plot resistance levels (red)
                resistance_levels = sr_levels.get('resistance_levels', [])
                for level in resistance_levels:
                    ax.axhline(y=level, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.text(price_data.index[0], level, f' R: ${level:.2f}',
                           verticalalignment='bottom', horizontalalignment='left',
                           color='red', fontsize=8, fontweight='bold')

                # Plot support levels (green)
                support_levels = sr_levels.get('support_levels', [])
                for level in support_levels:
                    ax.axhline(y=level, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.text(price_data.index[0], level, f' S: ${level:.2f}',
                           verticalalignment='top', horizontalalignment='left',
                           color='green', fontsize=8, fontweight='bold')

                logger.info(f"Added {len(support_levels)} support and {len(resistance_levels)} resistance levels to chart")

            else:
                # Fallback: Calculate support/resistance from recent highs/lows
                recent_data = price_data.tail(60)  # Last 60 periods

                # Find local maxima/minima
                if 'high' in recent_data.columns and 'low' in recent_data.columns:
                    highs = recent_data['high'].rolling(window=5, center=True).max() == recent_data['high']
                    lows = recent_data['low'].rolling(window=5, center=True).min() == recent_data['low']

                    # Plot resistance levels (red)
                    resistance_levels = recent_data[highs]['high'].unique()[-3:]  # Top 3 levels
                    for level in resistance_levels:
                        ax.axhline(y=level, color='red', linestyle='--', alpha=0.5, linewidth=1)
                        ax.text(price_data.index[-1], level, f'R: ${level:.2f}',
                               verticalalignment='bottom', horizontalalignment='right',
                               color='red', fontsize=8)

                    # Plot support levels (green)
                    support_levels = recent_data[lows]['low'].unique()[:3]  # Bottom 3 levels
                    for level in support_levels:
                        ax.axhline(y=level, color='green', linestyle='--', alpha=0.5, linewidth=1)
                        ax.text(price_data.index[-1], level, f'S: ${level:.2f}',
                               verticalalignment='top', horizontalalignment='right',
                               color='green', fontsize=8)

        except Exception as e:
            logger.warning(f"Could not add support/resistance levels: {e}")
    
    def _add_fibonacci_levels(self, ax, price_data: pd.DataFrame):
        """Add Fibonacci retracement levels to chart"""
        try:
            # Calculate Fibonacci levels from recent high/low
            if 'high' in price_data.columns and 'low' in price_data.columns:
                recent_high = price_data['high'].tail(120).max()
                recent_low = price_data['low'].tail(120).min()
                diff = recent_high - recent_low
            
            # Fibonacci ratios
            fib_levels = {
                '0.0%': recent_high,
                '23.6%': recent_high - diff * 0.236,
                '38.2%': recent_high - diff * 0.382,
                '50.0%': recent_high - diff * 0.500,
                '61.8%': recent_high - diff * 0.618,
                '78.6%': recent_high - diff * 0.786,
                '100.0%': recent_low
            }
            
            # Colors for Fibonacci levels
            colors = ['#FF0000', '#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#00CED1', '#0000FF']
            
            # Plot Fibonacci levels
            for i, (level, price) in enumerate(fib_levels.items()):
                ax.axhline(y=price, color=colors[i], linestyle=':', alpha=0.6, 
                          linewidth=1, label=f'Fib {level}')
            
        except Exception as e:
            logger.warning(f"Could not add Fibonacci levels: {e}")
    
    def generate_3d_fundamental_plot(self, recommendations: List[Dict]) -> str:
        """Generate 3D plot showing income/cashflow/balance sheet dimensions"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract data for plotting
            symbols = []
            income_scores = []
            cashflow_scores = []
            balance_scores = []
            overall_scores = []
            colors = []
            
            for rec in recommendations:
                symbol = rec.get('symbol', 'UNK')
                symbols.append(symbol)
                
                # Extract fundamental scores from recommendation
                income_score = rec.get('income_score', 5.0)
                cashflow_score = rec.get('cashflow_score', 5.0)
                balance_score = rec.get('balance_score', 5.0)
                overall_score = rec.get('overall_score', 5.0)
                
                income_scores.append(income_score)
                cashflow_scores.append(cashflow_score)
                balance_scores.append(balance_score)
                overall_scores.append(overall_score)
                
                # Color based on recommendation
                recommendation = rec.get('recommendation', 'HOLD')
                if 'BUY' in recommendation.upper():
                    colors.append('green')
                elif 'SELL' in recommendation.upper():
                    colors.append('red')
                else:
                    colors.append('blue')
            
            # Create 3D scatter plot with size based on overall score
            scatter = ax.scatter(income_scores, cashflow_scores, balance_scores,
                               s=[score * 20 for score in overall_scores],  # Size by overall score
                               c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.text(income_scores[i], cashflow_scores[i], balance_scores[i], 
                       f'  {symbol}', fontsize=8)
            
            # Set labels and title
            ax.set_xlabel('Income Statement Score')
            ax.set_ylabel('Cash Flow Score')
            ax.set_zlabel('Balance Sheet Score')
            ax.set_title('3D Fundamental Analysis\nSize = Overall Score, Color = Recommendation')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='BUY'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                          markersize=10, label='HOLD'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='SELL')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Save plot
            plot_path = self.charts_dir / "3d_fundamental_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated 3D fundamental plot: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating 3D fundamental plot: {e}")
            return ""
    
    def generate_2d_technical_fundamental_plot(self, recommendations: List[Dict]) -> str:
        """Generate 2D plot showing fundamental vs technical scores with data quality as size"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract data for plotting
            symbols = []
            fundamental_scores = []
            technical_scores = []
            data_quality_scores = []
            colors = []
            
            for rec in recommendations:
                symbol = rec.get('symbol', 'UNK')
                symbols.append(symbol)
                
                # Get scores
                fundamental = rec.get('fundamental_score', 5.0)
                technical = rec.get('technical_score', 5.0)
                data_quality = rec.get('data_quality_score', 0.5)
                
                fundamental_scores.append(fundamental)
                technical_scores.append(technical)
                data_quality_scores.append(data_quality * 100)  # Scale for visibility
                
                # Color based on overall score
                overall = rec.get('overall_score', 5.0)
                if overall >= 7:
                    colors.append('green')
                elif overall >= 4:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            # Create scatter plot
            scatter = ax.scatter(fundamental_scores, technical_scores,
                               s=data_quality_scores,  # Size by data quality
                               c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (fundamental_scores[i], technical_scores[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add diagonal reference line
            ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='Equal Weight')
            
            # Set labels and title
            ax.set_xlabel('Fundamental Score', fontsize=12)
            ax.set_ylabel('Technical Score', fontsize=12)
            ax.set_title('Technical vs Fundamental Analysis\nSize = Data Quality, Color = Overall Score', fontsize=14)
            
            # Set axis limits
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='High Score (≥7)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=10, label='Medium Score (4-7)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='Low Score (<4)')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Save plot
            plot_path = self.charts_dir / "2d_technical_fundamental_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated 2D technical/fundamental plot: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating 2D plot: {e}")
            return ""
    
    def generate_growth_value_plot(self, recommendations: List[Dict]) -> str:
        """Generate 2D plot showing growth vs value positioning"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Extract data
            symbols = []
            growth_scores = []
            value_scores = []
            overall_scores = []
            colors = []
            
            for rec in recommendations:
                symbol = rec.get('symbol', 'UNK')
                symbols.append(symbol)
                
                # Extract scores
                growth_score = rec.get('growth_score', 5.0)
                value_score = rec.get('value_score', 5.0)
                overall_score = rec.get('overall_score', 5.0)
                
                growth_scores.append(growth_score)
                value_scores.append(value_score)
                overall_scores.append(overall_score)
                
                # Color based on overall score
                if overall_score >= 7:
                    colors.append('green')
                elif overall_score >= 4:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            # Create scatter plot with size based on overall score
            scatter = ax.scatter(value_scores, growth_scores, 
                               s=[score * 30 for score in overall_scores],
                               c=colors, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (value_scores[i], growth_scores[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add quadrant lines
            ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(2.5, 8.5, 'Growth\nStocks', ha='center', va='center', 
                   fontsize=12, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax.text(7.5, 8.5, 'Quality\nGrowth', ha='center', va='center', 
                   fontsize=12, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax.text(2.5, 2.5, 'Deep\nValue', ha='center', va='center', 
                   fontsize=12, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax.text(7.5, 2.5, 'Value\nTraps', ha='center', va='center', 
                   fontsize=12, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            
            # Set labels and title
            ax.set_xlabel('Value Score', fontsize=12)
            ax.set_ylabel('Growth Score', fontsize=12)
            ax.set_title('Growth vs Value Analysis\nSize = Overall Score', fontsize=14)
            
            # Set axis limits
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.charts_dir / "growth_value_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated growth/value plot: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating growth/value plot: {e}")
            return ""
    
    def _extract_income_score(self, rec: Dict) -> float:
        """Extract income statement score from recommendation"""
        # If explicitly provided, use it
        if 'income_score' in rec:
            return rec['income_score']
        
        # Try to extract from full analysis
        fundamental_score = rec.get('fundamental_score', 5.0)
        
        # Look for income-related keywords in analysis
        full_analysis = rec.get('full_analysis', {})
        synthesis = full_analysis.get('synthesis', {})
        insights = synthesis.get('key_insights', [])
        
        income_keywords = ['revenue', 'income', 'earnings', 'profit', 'margin']
        income_mentions = sum(1 for insight in insights 
                            if any(keyword in str(insight).lower() for keyword in income_keywords))
        
        # Adjust score based on mentions
        if income_mentions > 2:
            return min(fundamental_score + 0.5, 10.0)
        elif income_mentions > 0:
            return fundamental_score
        else:
            return max(fundamental_score - 0.5, 1.0)
    
    def _extract_cashflow_score(self, rec: Dict) -> float:
        """Extract cash flow score from recommendation"""
        # If explicitly provided, use it
        if 'cashflow_score' in rec:
            return rec['cashflow_score']
        
        # Try to extract from full analysis
        fundamental_score = rec.get('fundamental_score', 5.0)
        
        # Look for cash flow keywords
        full_analysis = rec.get('full_analysis', {})
        synthesis = full_analysis.get('synthesis', {})
        insights = synthesis.get('key_insights', [])
        
        cashflow_keywords = ['cash flow', 'cash', 'liquidity', 'fcf', 'working capital']
        cashflow_mentions = sum(1 for insight in insights 
                              if any(keyword in str(insight).lower() for keyword in cashflow_keywords))
        
        # Adjust score based on mentions
        if cashflow_mentions > 2:
            return min(fundamental_score + 0.5, 10.0)
        elif cashflow_mentions > 0:
            return fundamental_score
        else:
            return max(fundamental_score - 0.5, 1.0)
    
    def _extract_balance_score(self, rec: Dict) -> float:
        """Extract balance sheet score from recommendation"""
        # If explicitly provided, use it
        if 'balance_score' in rec:
            return rec['balance_score']
        
        # Try to extract from full analysis
        fundamental_score = rec.get('fundamental_score', 5.0)
        
        # Look for balance sheet keywords
        full_analysis = rec.get('full_analysis', {})
        synthesis = full_analysis.get('synthesis', {})
        insights = synthesis.get('key_insights', [])
        
        balance_keywords = ['asset', 'liability', 'equity', 'debt', 'balance sheet', 'leverage']
        balance_mentions = sum(1 for insight in insights 
                             if any(keyword in str(insight).lower() for keyword in balance_keywords))
        
        # Adjust score based on mentions
        if balance_mentions > 2:
            return min(fundamental_score + 0.5, 10.0)
        elif balance_mentions > 0:
            return fundamental_score
        else:
            return max(fundamental_score - 0.5, 1.0)
    
    def generate_comprehensive_3d_plot(self, chart_data: List[Dict]) -> str:
        """Generate comprehensive 3D plot for peer group positioning"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot generate 3D plot - matplotlib not available")
            return ""
            
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            if not chart_data:
                logger.warning("No chart data provided for comprehensive 3D plot")
                return ""
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract data for plotting
            symbols = []
            overall_scores = []
            technical_scores = []
            fundamental_scores = []
            sectors = []
            colors_map = {
                'financials': 'blue',
                'technology': 'green', 
                'healthcare': 'red',
                'consumer_discretionary': 'orange',
                'industrials': 'purple',
                'energy': 'brown',
                'utilities': 'pink',
                'materials': 'gray',
                'real_estate': 'olive',
                'communication_services': 'cyan',
                'consumer_staples': 'magenta',
                'unknown': 'black'
            }
            
            for data in chart_data:
                symbols.append(data.get('symbol', 'UNK'))
                overall_scores.append(data.get('overall_score', 5.0))
                technical_scores.append(data.get('technical_score', 5.0))
                fundamental_scores.append(data.get('fundamental_score', 5.0))
                sectors.append(data.get('sector', 'unknown'))
            
            # Map sectors to colors
            colors = [colors_map.get(sector.lower(), 'black') for sector in sectors]
            
            # Create 3D scatter plot
            scatter = ax.scatter(overall_scores, technical_scores, fundamental_scores,
                               s=100, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.text(overall_scores[i], technical_scores[i], fundamental_scores[i], 
                       f'  {symbol}', fontsize=8)
            
            # Set labels and title
            ax.set_xlabel('Overall Score', fontsize=12)
            ax.set_ylabel('Technical Score', fontsize=12)
            ax.set_zlabel('Fundamental Score', fontsize=12)
            ax.set_title('3D Comprehensive Investment Universe\nPositioning Analysis', fontsize=14)
            
            # Add sector legend
            unique_sectors = list(set(sectors))
            legend_elements = []
            for sector in unique_sectors:
                color = colors_map.get(sector.lower(), 'black')
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=8, label=sector.title())
                )
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
            
            # Save plot
            plot_path = self.charts_dir / "comprehensive_3d_universe.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated comprehensive 3D plot: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive 3D plot: {e}")
            return ""
    
    def generate_sector_comparison_plot(self, chart_data: List[Dict]) -> str:
        """Generate sector comparison chart"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot generate sector comparison plot - matplotlib not available")
            return ""
            
        try:
            if not chart_data:
                logger.warning("No chart data provided for sector comparison plot")
                return ""
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Group data by sector
            sector_data = {}
            for data in chart_data:
                sector = data.get('sector', 'unknown').title()
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(data)
            
            # Calculate sector averages
            sector_names = []
            avg_scores = []
            symbol_counts = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, (sector, symbols) in enumerate(sector_data.items()):
                sector_names.append(sector)
                avg_score = sum(s.get('overall_score', 5.0) for s in symbols) / len(symbols)
                avg_scores.append(avg_score)
                symbol_counts.append(len(symbols))
            
            # Create bar chart
            bars = ax.bar(sector_names, avg_scores, 
                         color=[colors[i % len(colors)] for i in range(len(sector_names))],
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, symbol_counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}\n({count} stocks)', 
                       ha='center', va='bottom', fontsize=9)
            
            # Formatting
            ax.set_ylabel('Average Investment Score', fontsize=12)
            ax.set_xlabel('Sector', fontsize=12)
            ax.set_title('Sector Performance Comparison\nAverage Investment Scores by Sector', fontsize=14)
            ax.set_ylim(0, 10)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels if needed
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.charts_dir / "sector_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated sector comparison plot: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating sector comparison plot: {e}")
            return ""

    def generate_quarterly_revenue_trend(self, symbol: str, quarterly_trends: Dict) -> str:
        """
        Generate quarterly revenue trend chart with Q-o-Q comparison
        Grouped by fiscal quarter for better year-over-year comparison

        Args:
            symbol: Stock symbol
            quarterly_trends: Dictionary with quarterly trend data

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""

        if not quarterly_trends or 'revenue_trend' not in quarterly_trends:
            logger.warning(f"No revenue trend data available for {symbol}")
            return ""

        try:
            revenue_data = quarterly_trends['revenue_trend']
            if not revenue_data:
                return ""

            # Parse periods and group by quarter
            from collections import defaultdict
            quarterly_data = defaultdict(list)

            for item in revenue_data:
                period = item['period']
                value = item['value']

                # Parse period like "2024-Q1" or "Q1 2024" or "2024-FY" or "FY 2024"
                if '-Q' in period:
                    year, quarter = period.split('-Q')
                    quarter = f"Q{quarter}"
                elif '-FY' in period or '-fy' in period.lower():
                    # Handle FY format like "2024-FY"
                    year = period.split('-')[0]
                    quarter = 'FY'
                elif 'Q' in period:
                    parts = period.split()
                    if len(parts) == 2:
                        quarter, year = parts[0], parts[1]
                    else:
                        continue
                elif 'FY' in period.upper():
                    # Handle FY format like "FY 2024" or "2024 FY"
                    parts = period.split()
                    if len(parts) == 2:
                        if parts[0].upper() == 'FY':
                            quarter, year = 'FY', parts[1]
                        else:
                            year, quarter = parts[0], 'FY'
                    else:
                        continue
                else:
                    continue

                quarterly_data[quarter].append({
                    'year': year,
                    'value': value,
                    'period': period
                })

            # Sort by quarter - include FY at the end
            quarter_order = ['Q1', 'Q2', 'Q3', 'Q4', 'FY']
            sorted_quarters = [q for q in quarter_order if q in quarterly_data]

            # Log what quarters we found
            logger.info(f"Quarterly revenue chart for {symbol}: Found periods {sorted_quarters}")
            if 'FY' in sorted_quarters:
                fy_years = [item['year'] for item in quarterly_data['FY']]
                logger.info(f"  Including FY data for years: {fy_years}")

            if not sorted_quarters:
                # Fallback to original chronological view
                logger.warning(f"No quarterly or FY data found for {symbol}, falling back to chronological view")
                return self._generate_chronological_revenue_chart(symbol, revenue_data)

            # Get unique years and assign colors
            all_years = sorted(set(item['year'] for q in sorted_quarters for item in quarterly_data[q]))
            year_colors = {
                all_years[-1]: '#2E86AB' if len(all_years) >= 1 else '#2E86AB',  # Current year - blue
                all_years[-2]: '#F77F00' if len(all_years) >= 2 else '#F77F00',  # Previous year - orange
                all_years[-3]: '#06A77D' if len(all_years) >= 3 else '#06A77D',  # 2 years ago - green
            }
            # Default color for older years
            for year in all_years[:-3]:
                year_colors[year] = '#A0A0A0'  # Gray

            fig, ax = plt.subplots(figsize=(14, 7))

            # Calculate bar positions
            num_quarters = len(sorted_quarters)
            bar_width = 0.8 / len(all_years)
            x_positions = range(num_quarters)

            # Plot bars grouped by quarter
            for year_idx, year in enumerate(all_years):
                values = []
                for quarter in sorted_quarters:
                    quarter_items = quarterly_data[quarter]
                    year_data = [item for item in quarter_items if item['year'] == year]
                    if year_data:
                        values.append(year_data[0]['value'])
                    else:
                        values.append(0)

                # Calculate offset for this year's bars
                offset = (year_idx - len(all_years)/2) * bar_width + bar_width/2

                bars = ax.bar([x + offset for x in x_positions], values,
                             width=bar_width, label=year,
                             color=year_colors.get(year, '#A0A0A0'),
                             alpha=0.8, edgecolor='black', linewidth=0.5)

                # Add value labels and YoY growth percentages
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0:
                        height = bar.get_height()
                        # Show value
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(v for v in values if v > 0)*0.01,
                               f'${value:.0f}M', ha='center', va='bottom', fontsize=8)

                        # Calculate YoY growth if previous year data exists
                        if year_idx > 0:
                            prev_year = all_years[year_idx - 1]
                            prev_quarter_items = quarterly_data[sorted_quarters[i]]
                            prev_year_data = [item for item in prev_quarter_items if item['year'] == prev_year]

                            if prev_year_data and prev_year_data[0]['value'] > 0:
                                yoy_growth = ((value - prev_year_data[0]['value']) / prev_year_data[0]['value']) * 100
                                growth_color = 'green' if yoy_growth > 0 else 'red'
                                ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                                       f'{yoy_growth:+.1f}%', ha='center', va='center',
                                       fontsize=7, color=growth_color, fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=growth_color))

            # Formatting
            ax.set_ylabel('Revenue ($ Millions)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Fiscal Quarter', fontsize=12, fontweight='bold')
            ax.set_title(f'{symbol} - Quarterly Revenue Comparison\nGrouped by Quarter with Year-over-Year Growth',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(sorted_quarters, fontsize=11)
            ax.legend(loc='upper left', framealpha=0.9, title='Fiscal Year')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_revenue_trend.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated quarter-grouped revenue trend chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating revenue trend chart: {e}")
            return ""

    def _generate_chronological_revenue_chart(self, symbol: str, revenue_data: List[Dict]) -> str:
        """Fallback: Generate chronological revenue chart if quarter grouping fails"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            periods = [item['period'] for item in revenue_data]
            values = [item['value'] for item in revenue_data]

            bars = ax.bar(range(len(periods)), values, color='#2E86AB', alpha=0.7, edgecolor='black')

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                label = f'${value:.0f}M'
                if i > 0:
                    growth = ((values[i] - values[i-1]) / values[i-1]) * 100
                    color = 'green' if growth > 0 else 'red'
                    label += f'\n({growth:+.1f}%)'
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                           label, ha='center', va='bottom', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                           label, ha='center', va='bottom', fontsize=10)

            ax.set_ylabel('Revenue ($ Millions)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Fiscal Period', fontsize=12, fontweight='bold')
            ax.set_title(f'{symbol} - Quarterly Revenue Trend', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels(periods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()

            plot_path = self.charts_dir / f"{symbol}_revenue_trend.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating chronological revenue chart: {e}")
            return ""

    def generate_quarterly_profitability_chart(self, symbol: str, quarterly_trends: Dict) -> str:
        """
        Generate quarterly profitability chart showing net income and margins

        Args:
            symbol: Stock symbol
            quarterly_trends: Dictionary with quarterly trend data

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""

        if not quarterly_trends or 'net_income_trend' not in quarterly_trends:
            logger.warning(f"No profitability data available for {symbol}")
            return ""

        try:
            net_income_data = quarterly_trends.get('net_income_trend', [])
            margin_data = quarterly_trends.get('margin_trends', [])

            if not net_income_data:
                return ""

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                          gridspec_kw={'height_ratios': [1, 1]})

            # Top chart: Net Income
            periods = [item['period'] for item in net_income_data]
            net_income_values = [item['value'] for item in net_income_data]

            colors = ['green' if v >= 0 else 'red' for v in net_income_values]
            bars1 = ax1.bar(range(len(periods)), net_income_values, color=colors, alpha=0.7, edgecolor='black')

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars1, net_income_values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (max(net_income_values)*0.02 if height >= 0 else -max(net_income_values)*0.02),
                        f'${value:.0f}M', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax1.set_ylabel('Net Income ($ Millions)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{symbol} - Quarterly Net Income', fontsize=13, fontweight='bold')
            ax1.set_xticks(range(len(periods)))
            ax1.set_xticklabels(periods, rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')

            # Bottom chart: Profit Margins
            if margin_data:
                margin_periods = [item['period'] for item in margin_data]
                net_margins = [item['net_margin'] for item in margin_data]
                op_margins = [item['operating_margin'] for item in margin_data]

                x = range(len(margin_periods))
                width = 0.35

                bars2 = ax2.bar([i - width/2 for i in x], net_margins, width,
                              label='Net Margin %', color='#2E86AB', alpha=0.7, edgecolor='black')
                bars3 = ax2.bar([i + width/2 for i in x], op_margins, width,
                              label='Operating Margin %', color='#A23B72', alpha=0.7, edgecolor='black')

                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                ax2.set_ylabel('Margin %', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Fiscal Period', fontsize=12, fontweight='bold')
                ax2.set_title(f'{symbol} - Quarterly Profit Margins', fontsize=13, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(margin_periods, rotation=45, ha='right')
                ax2.legend(loc='best')
                ax2.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_profitability.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated profitability chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating profitability chart: {e}")
            return ""

    def generate_quarterly_cashflow_chart(self, symbol: str, quarterly_trends: Dict) -> str:
        """
        Generate quarterly operating cash flow chart

        Args:
            symbol: Stock symbol
            quarterly_trends: Dictionary with quarterly trend data

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""

        if not quarterly_trends or 'operating_cash_flow_trend' not in quarterly_trends:
            logger.warning(f"No cash flow data available for {symbol}")
            return ""

        try:
            ocf_data = quarterly_trends['operating_cash_flow_trend']
            if not ocf_data:
                return ""

            fig, ax = plt.subplots(figsize=(12, 6))

            periods = [item['period'] for item in ocf_data]
            values = [item['value'] for item in ocf_data]

            # Create bar chart
            colors = ['green' if v >= 0 else 'red' for v in values]
            bars = ax.bar(range(len(periods)), values, color=colors, alpha=0.7, edgecolor='black')

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                label = f'${value:.0f}M'
                if i > 0:
                    growth = ((values[i] - values[i-1]) / abs(values[i-1])) * 100 if values[i-1] != 0 else 0
                    color_text = 'green' if growth > 0 else 'red'
                    label += f'\n({growth:+.1f}%)'
                    ax.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.02 if height >= 0 else -max(values)*0.02),
                           label, ha='center', va='bottom' if height >= 0 else 'top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor=color_text, alpha=0.2))
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.02 if height >= 0 else -max(values)*0.02),
                           label, ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

            # Formatting
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('Operating Cash Flow ($ Millions)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Fiscal Period', fontsize=12, fontweight='bold')
            ax.set_title(f'{symbol} - Quarterly Operating Cash Flow\nWith Quarter-over-Quarter Growth Rates',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels(periods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_cash_flow.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated cash flow chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating cash flow chart: {e}")
            return ""

    def generate_score_history_chart(self, symbol: str, score_history: List[Dict], score_trend: Dict) -> str:
        """
        Generate investment score history chart showing trend over time

        Args:
            symbol: Stock symbol
            score_history: List of historical scores with dates
            score_trend: Trend analysis dictionary

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""

        if not score_history or len(score_history) < 2:
            logger.warning(f"Insufficient score history for {symbol}")
            return ""

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={'height_ratios': [2, 1]})

            dates = [item['date'] for item in score_history]
            overall_scores = [item['overall_score'] for item in score_history]
            fundamental_scores = [item.get('fundamental_score', 0) for item in score_history]
            technical_scores = [item.get('technical_score', 0) for item in score_history]

            # Top chart: Overall score trend
            ax1.plot(dates, overall_scores, 'b-', linewidth=2.5, marker='o', markersize=8,
                    label='Overall Score', color='#2E86AB')

            # Add trend line
            from numpy.polynomial import Polynomial
            if len(dates) >= 3:
                x_numeric = list(range(len(dates)))
                p = Polynomial.fit(x_numeric, overall_scores, 1)
                trend_line = p(x_numeric)
                trend_color = 'green' if trend_line[-1] > trend_line[0] else 'red'
                ax1.plot(dates, trend_line, '--', alpha=0.5, linewidth=2,
                        color=trend_color, label='Trend')

            # Add reference lines
            ax1.axhline(y=7.0, color='green', linestyle=':', alpha=0.3, linewidth=1)
            ax1.axhline(y=4.0, color='red', linestyle=':', alpha=0.3, linewidth=1)
            ax1.text(dates[0], 7.0, ' BUY Threshold', va='bottom', fontsize=8, color='green')
            ax1.text(dates[0], 4.0, ' SELL Threshold', va='top', fontsize=8, color='red')

            # Formatting
            ax1.set_ylabel('Investment Score (0-10)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{symbol} - Investment Score History\nTrend: {score_trend.get("trend", "N/A").upper()} '
                         f'({score_trend.get("change", 0):+.1f} points over {score_trend.get("num_analyses", 0)} analyses)',
                         fontsize=14, fontweight='bold')
            ax1.set_ylim(0, 10)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Bottom chart: Component scores
            ax2.plot(dates, fundamental_scores, '-', linewidth=2, marker='s', markersize=6,
                    label='Fundamental', color='#A23B72', alpha=0.7)
            ax2.plot(dates, technical_scores, '-', linewidth=2, marker='^', markersize=6,
                    label='Technical', color='#F18F01', alpha=0.7)

            ax2.set_ylabel('Component Scores', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Analysis Date', fontsize=11, fontweight='bold')
            ax2.set_ylim(0, 10)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3, linestyle='--')

            # Format dates on x-axis
            fig.autofmt_xdate()

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_score_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated score history chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating score history chart: {e}")
            return ""

    def generate_valuation_comparison_chart(self, symbol: str, peer_valuation: Dict) -> str:
        """
        Generate valuation comparison vs peers

        Args:
            symbol: Stock symbol
            peer_valuation: Dictionary with peer valuation metrics

        Returns:
            Path to generated chart
        """
        try:
            if not peer_valuation or 'relative_valuation' not in peer_valuation:
                logger.warning(f"No peer valuation data available for {symbol}")
                return ""

            fig, ax = plt.subplots(figsize=(10, 6))

            rel_val = peer_valuation['relative_valuation']
            metrics = ['P/E', 'P/B', 'P/S', 'PEG']
            keys = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio']

            target_values = []
            peer_medians = []

            for key in keys:
                if key in rel_val and rel_val[key].get('target'):
                    target_values.append(rel_val[key]['target'])
                    peer_medians.append(rel_val[key]['peer_median'])
                else:
                    target_values.append(0)
                    peer_medians.append(0)

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax.bar(x - width/2, target_values, width, label=symbol, color='#2E86AB')
            bars2 = ax.bar(x + width/2, peer_medians, width, label='Peer Median', color='#A23B72')

            ax.set_ylabel('Ratio', fontsize=11, fontweight='bold')
            ax.set_title(f'{symbol} - Valuation vs Peers\nOverall: {rel_val.get("overall_assessment", "").upper()}',
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plot_path = self.charts_dir / f"{symbol}_valuation_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📊 Generated valuation comparison chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating valuation comparison chart: {e}")
            return ""

    def generate_multi_year_trends_chart(self, symbol: str, multi_year_trends: Dict) -> str:
        """
        Generate multi-year historical trends chart showing revenue and earnings with CAGR

        Args:
            symbol: Stock symbol
            multi_year_trends: Dictionary with 'data' (yearly financials) and 'metrics' (CAGR, trends)

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""

        if not multi_year_trends or 'data' not in multi_year_trends or 'metrics' not in multi_year_trends:
            logger.warning(f"No multi-year trends data available for {symbol}")
            return ""

        try:
            yearly_data = multi_year_trends['data']
            metrics = multi_year_trends['metrics']

            # Log received data for debugging
            logger.info(f"generate_multi_year_trends_chart called for {symbol}")
            logger.info(f"  Yearly data points: {len(yearly_data) if yearly_data else 0}")
            logger.info(f"  Metrics received: {list(metrics.keys()) if metrics else 'None'}")
            if metrics:
                logger.info(f"  Revenue CAGR: {metrics.get('revenue_cagr')}")
                logger.info(f"  Cyclical pattern: {metrics.get('cyclical_pattern')}")

            if not yearly_data or len(yearly_data) < 3:
                logger.warning(f"Insufficient multi-year data for {symbol} - need at least 3 years, got {len(yearly_data) if yearly_data else 0}")
                return ""

            # Create dual-axis chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Extract data
            years = [int(item.get('fiscal_year', 0)) for item in yearly_data]
            revenues = [item.get('revenue', 0) / 1_000_000 for item in yearly_data]  # Convert to millions
            earnings = [item.get('net_income', 0) / 1_000_000 for item in yearly_data]  # Convert to millions

            # Filter out zero values
            valid_indices = [i for i, (r, e) in enumerate(zip(revenues, earnings)) if r > 0 or e != 0]
            if not valid_indices:
                logger.warning(f"No valid data points for {symbol}")
                return ""

            years = [years[i] for i in valid_indices]
            revenues = [revenues[i] for i in valid_indices]
            earnings = [earnings[i] for i in valid_indices]

            # Panel 1: Revenue Trend
            ax1.plot(years, revenues, marker='o', linewidth=2.5, markersize=8,
                    color='#2E86AB', label='Revenue')

            # Add trend line for revenue
            if len(years) >= 2:
                z = np.polyfit(range(len(years)), revenues, 1)
                p = np.poly1d(z)
                ax1.plot(years, p(range(len(years))), "--", color='#1a5276', alpha=0.6,
                        linewidth=1.5, label='Trend')

            # Add value labels on revenue points
            for year, rev in zip(years, revenues):
                ax1.annotate(f'${rev:.0f}M', xy=(year, rev),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            ax1.set_ylabel('Revenue ($ Millions)', fontsize=12, fontweight='bold')

            # Format CAGR safely (handle None/missing values)
            cagr_value = metrics.get("revenue_cagr")
            cagr_text = f"{cagr_value:.1f}%" if cagr_value is not None else "N/A"
            pattern = metrics.get("cyclical_pattern", "N/A")
            pattern_display = pattern.replace("_", " ").title() if pattern != "N/A" else "N/A"

            ax1.set_title(f'{symbol} - Multi-Year Financial Trends\n'
                         f'Revenue CAGR: {cagr_text} | Pattern: {pattern_display}',
                         fontsize=14, fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper left', fontsize=10)

            # Panel 2: Earnings Trend
            earnings_colors = ['green' if e >= 0 else 'red' for e in earnings]
            ax2.bar(years, earnings, color=earnings_colors, alpha=0.7, edgecolor='black')

            # Add zero line
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

            # Add value labels on earnings bars
            for year, earn in zip(years, earnings):
                y_pos = earn + (max(earnings) * 0.03 if earn >= 0 else min(earnings) * 0.03)
                ax2.annotate(f'${earn:.0f}M', xy=(year, earn),
                           xytext=(0, 5 if earn >= 0 else -5), textcoords='offset points',
                           ha='center', va='bottom' if earn >= 0 else 'top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            ax2.set_xlabel('Fiscal Year', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Net Income ($ Millions)', fontsize=12, fontweight='bold')

            # Format earnings CAGR and volatility safely
            earnings_cagr = metrics.get("earnings_cagr")
            earnings_cagr_text = f"{earnings_cagr:.1f}%" if earnings_cagr is not None else "N/A"
            volatility = metrics.get("revenue_volatility")
            volatility_text = f"{volatility:.1f}%" if volatility is not None else "N/A"

            ax2.set_title(f'Earnings CAGR: {earnings_cagr_text} | Volatility: {volatility_text}',
                         fontsize=12, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xticks(years)
            ax2.set_xticklabels(years, rotation=45, ha='right')

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_multi_year_trends.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Successfully generated multi-year trends chart: {plot_path}")
            logger.info(f"  Chart includes {len(yearly_data)} years of data with CAGR: {cagr_text}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating multi-year trends chart: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def generate_risk_scores_radar_chart(self, symbol: str, risk_scores: Dict) -> str:
        """
        Generate radar/spider chart showing multi-dimensional risk scores

        Args:
            symbol: Stock symbol
            risk_scores: Dictionary with risk scores across 5 dimensions

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate chart for {symbol} - matplotlib not available")
            return ""

        if not risk_scores or 'overall_risk' not in risk_scores:
            logger.warning(f"No risk scores data available for {symbol}")
            return ""

        try:
            # Define the 5 risk dimensions
            categories = [
                'Financial\nHealth',
                'Market\nRisk',
                'Operational\nRisk',
                'Business\nModel',
                'Growth\nRisk'
            ]

            # Extract risk scores (0-10 scale)
            values = [
                risk_scores.get('financial_health_risk', 5),
                risk_scores.get('market_risk', 5),
                risk_scores.get('operational_risk', 5),
                risk_scores.get('business_model_risk', 5),
                risk_scores.get('growth_risk', 5)
            ]

            # Number of variables
            N = len(categories)

            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1]  # Complete the circle
            angles += angles[:1]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # Draw the plot
            ax.plot(angles, values, 'o-', linewidth=2.5, color='#2E86AB', markersize=8)
            ax.fill(angles, values, alpha=0.25, color='#2E86AB')

            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=11, fontweight='bold')

            # Set y-axis limits and labels
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], size=9, color='gray')

            # Add risk zones with color coding
            ax.fill_between(angles, 0, 3, alpha=0.1, color='green')  # Low risk
            ax.fill_between(angles, 3, 6, alpha=0.1, color='yellow')  # Medium risk
            ax.fill_between(angles, 6, 10, alpha=0.1, color='red')  # High risk

            # Add title with overall risk rating
            overall_risk = risk_scores.get('overall_risk', 'N/A')
            risk_rating = risk_scores.get('risk_rating', 'Unknown')

            # Color code the rating
            rating_colors = {
                'Very Low': 'green',
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red',
                'Very High': 'darkred'
            }
            rating_color = rating_colors.get(risk_rating, 'black')

            ax.set_title(f'{symbol} - Multi-Dimensional Risk Assessment\n'
                        f'Overall Risk: {overall_risk}/10 ({risk_rating})',
                        size=14, fontweight='bold', pad=30)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_risk_radar.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated risk radar chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating risk radar chart: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def generate_competitive_positioning_matrix(self, competitive_positioning: Dict) -> str:
        """
        Generate 2D scatter plot showing competitive positioning

        Args:
            competitive_positioning: Dictionary with target and peer positioning data

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot generate competitive positioning matrix - matplotlib not available")
            return ""

        if not competitive_positioning or not competitive_positioning.get('target'):
            logger.warning("No competitive positioning data available")
            return ""

        try:
            target = competitive_positioning['target']
            peers = competitive_positioning.get('peers', [])
            industry = competitive_positioning.get('industry', 'Industry')
            sector = competitive_positioning.get('sector', '')

            if not peers:
                logger.warning("No peer data available for competitive positioning matrix")
                return ""

            # Extract data for plotting
            peer_growth = [p['revenue_growth'] for p in peers]
            peer_margin = [p['profit_margin'] for p in peers]
            peer_labels = [p['symbol'] for p in peers]

            target_growth = target['revenue_growth']
            target_margin = target['profit_margin']
            target_symbol = target['symbol']

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot peers as blue dots
            ax.scatter(peer_growth, peer_margin, s=100, c='#A0B0C0', alpha=0.6,
                      label='Peers', edgecolors='black', linewidth=0.5)

            # Plot target as highlighted red star
            ax.scatter([target_growth], [target_margin], s=500, c='#D32F2F',
                      marker='*', label=target_symbol, edgecolors='black', linewidth=2, zorder=5)

            # Add labels for top peers and target
            ax.annotate(target_symbol, xy=(target_growth, target_margin),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='#D32F2F',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#D32F2F', alpha=0.8))

            # Label a few notable peers (avoid overcrowding)
            labeled_count = 0
            for i, (growth, margin, label) in enumerate(zip(peer_growth, peer_margin, peer_labels)):
                # Label peers at extremes
                if labeled_count < 5:
                    distance_from_target = ((growth - target_growth)**2 + (margin - target_margin)**2)**0.5
                    if distance_from_target > 5:  # Only label if sufficiently far from target
                        ax.annotate(label, xy=(growth, margin),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))
                        labeled_count += 1

            # Add quadrant lines at median
            if peer_growth and peer_margin:
                median_growth = np.median(peer_growth)
                median_margin = np.median(peer_margin)

                ax.axvline(x=median_growth, color='gray', linestyle='--', alpha=0.3, linewidth=1)
                ax.axhline(y=median_margin, color='gray', linestyle='--', alpha=0.3, linewidth=1)

                # Add quadrant labels
                x_range = max(peer_growth + [target_growth]) - min(peer_growth + [target_growth])
                y_range = max(peer_margin + [target_margin]) - min(peer_margin + [target_margin])

                x_offset = x_range * 0.1
                y_offset = y_range * 0.1

                # Top-right quadrant
                ax.text(max(peer_growth + [target_growth]) - x_offset,
                       max(peer_margin + [target_margin]) - y_offset,
                       'High Growth\nHigh Margin', fontsize=10, alpha=0.5,
                       ha='right', va='top', style='italic')

                # Bottom-right quadrant
                ax.text(max(peer_growth + [target_growth]) - x_offset,
                       min(peer_margin + [target_margin]) + y_offset,
                       'High Growth\nLow Margin', fontsize=10, alpha=0.5,
                       ha='right', va='bottom', style='italic')

                # Top-left quadrant
                ax.text(min(peer_growth + [target_growth]) + x_offset,
                       max(peer_margin + [target_margin]) - y_offset,
                       'Low Growth\nHigh Margin', fontsize=10, alpha=0.5,
                       ha='left', va='top', style='italic')

                # Bottom-left quadrant
                ax.text(min(peer_growth + [target_growth]) + x_offset,
                       min(peer_margin + [target_margin]) + y_offset,
                       'Low Growth\nLow Margin', fontsize=10, alpha=0.5,
                       ha='left', va='bottom', style='italic')

            # Formatting
            ax.set_xlabel('Revenue Growth (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Competitive Positioning Matrix - {industry}\n'
                        f'{target_symbol} vs {len(peers)} Industry Peers',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11, loc='best')

            # Add zero lines
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{target_symbol}_competitive_positioning.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated competitive positioning matrix: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating competitive positioning matrix: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def generate_volume_profile_chart(self, symbol: str, volume_profile: Dict) -> str:
        """
        Generate volume profile chart showing volume distribution at price levels

        Args:
            symbol: Stock symbol
            volume_profile: Dictionary with volume profile data

        Returns:
            Path to generated chart or empty string if no data/matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot generate volume profile chart - matplotlib not available")
            return ""

        if not volume_profile or not volume_profile.get('profile_bins'):
            logger.warning(f"No volume profile data available for {symbol}")
            return ""

        try:
            profile_bins = volume_profile['profile_bins']
            current_price = volume_profile.get('current_price', 0)
            poc_price = volume_profile.get('poc_price', 0)
            value_area_low = volume_profile.get('value_area_low', 0)
            value_area_high = volume_profile.get('value_area_high', 0)
            days_analyzed = volume_profile.get('days_analyzed', 90)

            # Extract data for plotting
            prices = [bin['price_mid'] for bin in profile_bins]
            volumes = [bin['volume'] for bin in profile_bins]
            volume_pcts = [bin['volume_pct'] for bin in profile_bins]

            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot volume bars (horizontal)
            colors = []
            for price in prices:
                if value_area_low <= price <= value_area_high:
                    colors.append('#2E86AB')  # Value area in blue
                else:
                    colors.append('#A0B0C0')  # Outside value area in gray

            bars = ax.barh(prices, volumes, height=(prices[1] - prices[0]) * 0.9,
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add Point of Control line
            ax.axhline(y=poc_price, color='red', linestyle='--', linewidth=2.5,
                      label=f'POC: ${poc_price:.2f}', alpha=0.8, zorder=5)

            # Add current price line
            ax.axhline(y=current_price, color='green', linestyle='-', linewidth=2.5,
                      label=f'Current: ${current_price:.2f}', alpha=0.8, zorder=5)

            # Add value area boundaries
            ax.axhline(y=value_area_low, color='blue', linestyle=':', linewidth=1.5,
                      alpha=0.6, label='Value Area')
            ax.axhline(y=value_area_high, color='blue', linestyle=':', linewidth=1.5,
                      alpha=0.6)

            # Add value area shading
            ax.axhspan(value_area_low, value_area_high, alpha=0.1, color='blue', zorder=1)

            # Labels and formatting
            ax.set_xlabel('Trading Volume', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price Level ($)', fontsize=12, fontweight='bold')
            ax.set_title(f'{symbol} - Volume Profile Analysis ({days_analyzed} Days)\n'
                        f'Point of Control: ${poc_price:.2f} | Value Area: ${value_area_low:.2f} - ${value_area_high:.2f}',
                        fontsize=14, fontweight='bold', pad=20)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='x')

            # Legend
            ax.legend(fontsize=11, loc='upper right')

            # Format x-axis to show volume in millions
            max_volume = max(volumes) if volumes else 1
            if max_volume > 1e6:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            elif max_volume > 1e3:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))

            # Add annotations for key levels
            # Annotate POC
            max_vol = max(volumes) if volumes else 1
            ax.annotate(f'Highest Volume\n{volume_profile.get("poc_volume_pct", 0):.1f}% of total',
                       xy=(max_vol * 0.7, poc_price),
                       xytext=(max_vol * 0.8, poc_price + (max(prices) - min(prices)) * 0.05),
                       fontsize=9, color='red',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

            # Price position relative to POC
            if current_price > poc_price:
                position = "above"
                color = 'green'
            elif current_price < poc_price:
                position = "below"
                color = 'red'
            else:
                position = "at"
                color = 'blue'

            price_diff_pct = ((current_price - poc_price) / poc_price * 100) if poc_price > 0 else 0

            ax.text(0.02, 0.98, f'Price is {abs(price_diff_pct):.1f}% {position} POC',
                   transform=ax.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

            plt.tight_layout()

            # Save plot
            plot_path = self.charts_dir / f"{symbol}_volume_profile.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"📊 Generated volume profile chart: {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Error generating volume profile chart: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
