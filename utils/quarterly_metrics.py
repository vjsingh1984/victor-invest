#!/usr/bin/env python3
"""
Quarterly Metrics Calculation Module
Centralized calculation of all quarterly financial metrics and ratios
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class QuarterlyMetricsCalculator:
    """
    Centralized calculator for all quarterly financial metrics
    Ensures consistent calculations across all data flows (cache and direct)
    """
    
    def __init__(self):
        self.logger = logger
        
    def calculate_all_metrics(self, quarterly_data: List[Dict], symbol: str) -> pd.DataFrame:
        """
        Calculate ALL quarterly financial metrics on the full dataset
        Returns enhanced DataFrame with all calculated ratios and metrics
        
        Args:
            quarterly_data: List of quarterly data dictionaries from company facts
            symbol: Stock symbol for logging
            
        Returns:
            Enhanced DataFrame with all quarterly metrics calculated
        """
        try:
            if not quarterly_data:
                self.logger.warning(f"No quarterly data provided for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(quarterly_data)
            
            # Ensure we have required columns
            required_cols = ['period', 'fiscal_year', 'fiscal_quarter']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
                    self.logger.warning(f"Missing {col} column for {symbol}")
            
            # Sort by fiscal year and quarter
            df = df.sort_values(['fiscal_year', 'fiscal_quarter'], ascending=[True, True])
            
            # Calculate Growth Metrics
            self._calculate_growth_metrics(df)
            
            # Calculate Profitability Ratios
            self._calculate_profitability_ratios(df)
            
            # Calculate Efficiency Ratios
            self._calculate_efficiency_ratios(df)
            
            # Calculate Liquidity Ratios
            self._calculate_liquidity_ratios(df)
            
            # Calculate Leverage Ratios
            self._calculate_leverage_ratios(df)
            
            # Calculate Valuation Metrics
            self._calculate_valuation_metrics(df)
            
            # Calculate Quality Scores
            self._calculate_quality_scores(df)
            
            # Add metadata
            df['Symbol'] = symbol
            df['Calculation_Date'] = datetime.now()
            df['Data_Quality'] = self._assess_data_quality(df)
            
            self.logger.info(f"Calculated quarterly metrics for {symbol}: {len(df)} quarters, {len(df.columns)} total columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating quarterly metrics for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_growth_metrics(self, df: pd.DataFrame):
        """Calculate quarter-over-quarter and year-over-year growth metrics"""
        # Revenue growth metrics
        if 'revenue' in df.columns:
            df['Revenue_QoQ_Growth'] = df['revenue'].pct_change() * 100
            df['Revenue_YoY_Growth'] = df['revenue'].pct_change(periods=4) * 100  # 4 quarters = 1 year
        
        # Net income growth
        if 'net_income' in df.columns:
            df['NetIncome_QoQ_Growth'] = df['net_income'].pct_change() * 100
            df['NetIncome_YoY_Growth'] = df['net_income'].pct_change(periods=4) * 100
        
        # EPS growth
        if 'earnings_per_share' in df.columns:
            df['EPS_QoQ_Growth'] = df['earnings_per_share'].pct_change() * 100
            df['EPS_YoY_Growth'] = df['earnings_per_share'].pct_change(periods=4) * 100
        
        # Assets growth
        if 'total_assets' in df.columns:
            df['Assets_QoQ_Growth'] = df['total_assets'].pct_change() * 100
            df['Assets_YoY_Growth'] = df['total_assets'].pct_change(periods=4) * 100
    
    def _calculate_profitability_ratios(self, df: pd.DataFrame):
        """Calculate profitability ratios and margins"""
        # Gross Margin
        if 'revenue' in df.columns and 'cost_of_revenue' in df.columns:
            df['Gross_Margin'] = ((df['revenue'] - df['cost_of_revenue']) / df['revenue'] * 100).fillna(0)
        
        # Operating Margin
        if 'operating_income' in df.columns and 'revenue' in df.columns:
            df['Operating_Margin'] = (df['operating_income'] / df['revenue'] * 100).fillna(0)
        
        # Net Margin
        if 'net_income' in df.columns and 'revenue' in df.columns:
            df['Net_Margin'] = (df['net_income'] / df['revenue'] * 100).fillna(0)
        
        # Return on Assets (ROA)
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df['ROA'] = (df['net_income'] / df['total_assets'] * 100).fillna(0)
        
        # Return on Equity (ROE)
        if 'net_income' in df.columns and 'stockholders_equity' in df.columns:
            df['ROE'] = (df['net_income'] / df['stockholders_equity'] * 100).fillna(0)
    
    def _calculate_efficiency_ratios(self, df: pd.DataFrame):
        """Calculate efficiency and turnover ratios"""
        # Asset Turnover (annualized quarterly revenue / total assets)
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            df['Asset_Turnover'] = ((df['revenue'] * 4) / df['total_assets']).fillna(0)
        
        # Inventory Turnover (annualized quarterly COGS / inventory)
        if 'cost_of_revenue' in df.columns and 'inventory' in df.columns:
            df['Inventory_Turnover'] = ((df['cost_of_revenue'] * 4) / df['inventory']).fillna(0)
        
        # Receivables Turnover (annualized quarterly revenue / accounts receivable)
        if 'revenue' in df.columns and 'accounts_receivable' in df.columns:
            df['Receivables_Turnover'] = ((df['revenue'] * 4) / df['accounts_receivable']).fillna(0)
    
    def _calculate_liquidity_ratios(self, df: pd.DataFrame):
        """Calculate liquidity ratios"""
        # Current Ratio
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            df['Current_Ratio'] = (df['current_assets'] / df['current_liabilities']).fillna(0)
        
        # Quick Ratio
        if all(col in df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            df['Quick_Ratio'] = ((df['current_assets'] - df['inventory']) / df['current_liabilities']).fillna(0)
        
        # Cash Ratio
        if 'cash_and_equivalents' in df.columns and 'current_liabilities' in df.columns:
            df['Cash_Ratio'] = (df['cash_and_equivalents'] / df['current_liabilities']).fillna(0)
    
    def _calculate_leverage_ratios(self, df: pd.DataFrame):
        """Calculate leverage and solvency ratios"""
        # Debt-to-Equity Ratio
        if 'total_debt' in df.columns and 'stockholders_equity' in df.columns:
            df['Debt_to_Equity'] = (df['total_debt'] / df['stockholders_equity']).fillna(0)
        
        # Debt-to-Assets Ratio
        if 'total_debt' in df.columns and 'total_assets' in df.columns:
            df['Debt_to_Assets'] = (df['total_debt'] / df['total_assets']).fillna(0)
        
        # Equity Ratio
        if 'stockholders_equity' in df.columns and 'total_assets' in df.columns:
            df['Equity_Ratio'] = (df['stockholders_equity'] / df['total_assets']).fillna(0)
        
        # Interest Coverage Ratio
        if 'operating_income' in df.columns and 'interest_expense' in df.columns:
            df['Interest_Coverage'] = (df['operating_income'] / df['interest_expense']).fillna(0)
    
    def _calculate_valuation_metrics(self, df: pd.DataFrame):
        """Calculate valuation-related metrics"""
        # Price-to-Book (requires current market cap - will be calculated elsewhere)
        # Book Value per Share
        if 'stockholders_equity' in df.columns and 'shares_outstanding' in df.columns:
            df['Book_Value_Per_Share'] = (df['stockholders_equity'] / df['shares_outstanding']).fillna(0)
        
        # Revenue per Share
        if 'revenue' in df.columns and 'shares_outstanding' in df.columns:
            df['Revenue_Per_Share'] = (df['revenue'] / df['shares_outstanding']).fillna(0)
        
        # Working Capital per Share
        if all(col in df.columns for col in ['current_assets', 'current_liabilities', 'shares_outstanding']):
            working_capital = df['current_assets'] - df['current_liabilities']
            df['Working_Capital_Per_Share'] = (working_capital / df['shares_outstanding']).fillna(0)
    
    def _calculate_quality_scores(self, df: pd.DataFrame):
        """Calculate data quality and financial health scores"""
        # Revenue Quality (consistency of revenue growth)
        if 'Revenue_QoQ_Growth' in df.columns:
            df['Revenue_Growth_Consistency'] = df['Revenue_QoQ_Growth'].rolling(window=4, min_periods=2).std().fillna(0)
        
        # Earnings Quality (cash flow vs earnings)
        if 'net_income' in df.columns and 'operating_cash_flow' in df.columns:
            df['Earnings_Quality'] = (df['operating_cash_flow'] / df['net_income']).fillna(0)
        
        # Financial Strength Score (composite score based on multiple ratios)
        score_components = []
        
        if 'Current_Ratio' in df.columns:
            score_components.append(np.clip(df['Current_Ratio'] / 2.0, 0, 1))  # Normalize current ratio
        
        if 'ROE' in df.columns:
            score_components.append(np.clip(df['ROE'] / 15.0, 0, 1))  # Normalize ROE
        
        if 'Debt_to_Equity' in df.columns:
            score_components.append(np.clip(1 - (df['Debt_to_Equity'] / 2.0), 0, 1))  # Lower debt is better
        
        if score_components:
            df['Financial_Strength_Score'] = np.mean(score_components, axis=0) * 100
        else:
            df['Financial_Strength_Score'] = 0
    
    def _assess_data_quality(self, df: pd.DataFrame) -> List[float]:
        """Assess data quality for each quarter"""
        quality_scores = []
        
        # Key financial statement items
        key_items = ['revenue', 'net_income', 'total_assets', 'stockholders_equity', 'operating_cash_flow']
        
        for _, row in df.iterrows():
            available_items = sum(1 for item in key_items if item in df.columns and pd.notna(row[item]) and row[item] != 0)
            quality_score = (available_items / len(key_items)) * 100
            quality_scores.append(quality_score)
        
        return quality_scores
    
    def extract_recent_quarters_for_llm(self, enhanced_df: pd.DataFrame, quarters: int = 3) -> pd.DataFrame:
        """
        Extract recent N quarters from enhanced DataFrame for LLM analysis
        All metrics are already calculated on the full dataset
        
        Args:
            enhanced_df: DataFrame with all quarterly metrics calculated
            quarters: Number of recent quarters to extract (default 3)
            
        Returns:
            DataFrame with recent quarters and all pre-calculated metrics
        """
        if enhanced_df.empty:
            return enhanced_df
        
        # Sort by fiscal year and quarter to ensure proper ordering
        sorted_df = enhanced_df.sort_values(['fiscal_year', 'fiscal_quarter'], ascending=[True, True])
        
        return sorted_df.tail(quarters).copy()
    
    def safe_convert_to_float(self, value: Any) -> float:
        """Safely convert various types to float"""
        if pd.isna(value) or value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            try:
                # Remove commas and convert
                cleaned = value.replace(',', '').replace('$', '').replace('%', '')
                return float(cleaned)
            except (ValueError, AttributeError):
                return 0.0
        
        if isinstance(value, Decimal):
            try:
                return float(value)
            except (InvalidOperation, ValueError):
                return 0.0
        
        return 0.0


# Singleton instance
_quarterly_calculator = None

def get_quarterly_calculator() -> QuarterlyMetricsCalculator:
    """Get singleton instance of QuarterlyMetricsCalculator"""
    global _quarterly_calculator
    if _quarterly_calculator is None:
        _quarterly_calculator = QuarterlyMetricsCalculator()
    return _quarterly_calculator