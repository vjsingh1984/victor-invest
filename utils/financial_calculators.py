"""
Financial Metric Calculators

Provides robust fallback logic for calculating financial metrics when direct
XBRL tags are not available. Uses accounting identities and alternative
calculation methods.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class FinancialCalculator:
    """
    Calculates missing financial metrics using accounting identities and
    alternative data sources.
    """

    @staticmethod
    def calculate_total_liabilities(data: Dict[str, Any], symbol: str = '') -> Optional[float]:
        """
        Calculate total liabilities using multiple fallback strategies.

        Strategy 1: Direct tag (Liabilities)
        Strategy 2: Current + Noncurrent liabilities
        Strategy 3: Total Assets - Stockholders Equity (balance sheet identity)
        Strategy 4: LiabilitiesAndStockholdersEquity - StockholdersEquity

        Args:
            data: Financial data dict with extracted metrics
            symbol: Stock ticker (for logging)

        Returns:
            Total liabilities or None if cannot be calculated
        """
        # Strategy 1: Direct tag (already tried by normalizer)
        if 'total_liabilities' in data and data['total_liabilities'] is not None:
            return data['total_liabilities']

        # Strategy 2: Current + Noncurrent liabilities
        current_liab = data.get('current_liabilities')
        noncurrent_liab = data.get('noncurrent_liabilities')

        if current_liab is not None and noncurrent_liab is not None:
            result = float(current_liab) + float(noncurrent_liab)
            logger.debug(f"{symbol}: Calculated total_liabilities = current + noncurrent = {result:,.0f}")
            return result

        # Strategy 3: Assets - Equity (fundamental accounting equation)
        total_assets = data.get('total_assets')
        stockholders_equity = data.get('stockholders_equity')

        if total_assets is not None and stockholders_equity is not None:
            result = float(total_assets) - float(stockholders_equity)
            logger.debug(f"{symbol}: Calculated total_liabilities = assets - equity = {result:,.0f}")
            return result

        # Strategy 4: Check for raw XBRL tags that weren't normalized
        # LiabilitiesAndStockholdersEquity - StockholdersEquity
        liab_and_equity = data.get('LiabilitiesAndStockholdersEquity')
        if liab_and_equity is not None and stockholders_equity is not None:
            result = float(liab_and_equity) - float(stockholders_equity)
            logger.debug(f"{symbol}: Calculated total_liabilities from LiabAndEquity - Equity = {result:,.0f}")
            return result

        logger.warning(f"{symbol}: Unable to calculate total_liabilities with any strategy")
        return None

    @staticmethod
    def calculate_total_revenue(data: Dict[str, Any], symbol: str = '') -> Optional[float]:
        """
        Calculate total revenue using fallback strategies.

        Strategy 1: Direct tag (RevenueFromContractWithCustomer, Revenues, etc.)
        Strategy 2: SalesRevenueNet
        Strategy 3: For financial institutions: InterestAndDividendIncomeOperating
        Strategy 4: RevenuesNetOfInterestExpense (for some financials)

        Args:
            data: Financial data dict
            symbol: Stock ticker (for logging)

        Returns:
            Total revenue or None
        """
        # Strategy 1: Already tried by normalizer
        if 'total_revenue' in data and data['total_revenue'] is not None:
            return data['total_revenue']

        # Strategy 2: Check for InterestAndDividendIncomeOperating (financial institutions)
        interest_income = data.get('InterestAndDividendIncomeOperating')
        if interest_income is not None:
            logger.debug(f"{symbol}: Using InterestAndDividendIncomeOperating as revenue = {interest_income:,.0f}")
            return float(interest_income)

        # Strategy 3: RevenuesNetOfInterestExpense (some banks)
        revenue_net_interest = data.get('RevenuesNetOfInterestExpense')
        if revenue_net_interest is not None:
            logger.debug(f"{symbol}: Using RevenuesNetOfInterestExpense as revenue = {revenue_net_interest:,.0f}")
            return float(revenue_net_interest)

        # Strategy 4: For investment banks - FeesAndCommissions
        fees_and_commissions = data.get('FeesAndCommissions')
        if fees_and_commissions is not None:
            logger.debug(f"{symbol}: Using FeesAndCommissions as revenue = {fees_and_commissions:,.0f}")
            return float(fees_and_commissions)

        # Strategy 5: For utilities - RegulatedAndUnregulatedOperatingRevenue
        utility_revenue = data.get('RegulatedAndUnregulatedOperatingRevenue')
        if utility_revenue is not None:
            logger.debug(f"{symbol}: Using RegulatedAndUnregulatedOperatingRevenue as revenue = {utility_revenue:,.0f}")
            return float(utility_revenue)

        logger.warning(f"{symbol}: Unable to calculate total_revenue with any strategy")
        return None

    @staticmethod
    def calculate_net_income(data: Dict[str, Any], symbol: str = '') -> Optional[float]:
        """
        Calculate net income using fallback strategies.

        Strategy 1: Direct tag (NetIncomeLoss)
        Strategy 2: For banks: NetIncomeLossAvailableToCommonStockholdersBasic
        Strategy 3: Comprehensive income (less other comprehensive income)

        Args:
            data: Financial data dict
            symbol: Stock ticker (for logging)

        Returns:
            Net income or None
        """
        # Strategy 1: Already tried
        if 'net_income' in data and data['net_income'] is not None:
            return data['net_income']

        # Strategy 2: Net income available to common stockholders
        ni_common = data.get('NetIncomeLossAvailableToCommonStockholdersBasic')
        if ni_common is not None:
            logger.debug(f"{symbol}: Using NetIncomeLossAvailableToCommonStockholdersBasic = {ni_common:,.0f}")
            return float(ni_common)

        # Strategy 3: Comprehensive income minus other comprehensive income
        comprehensive_income = data.get('ComprehensiveIncomeNetOfTax')
        other_comprehensive = data.get('OtherComprehensiveIncomeLossNetOfTax')

        if comprehensive_income is not None and other_comprehensive is not None:
            result = float(comprehensive_income) - float(other_comprehensive)
            logger.debug(f"{symbol}: Calculated net_income from comprehensive income = {result:,.0f}")
            return result

        logger.warning(f"{symbol}: Unable to calculate net_income with any strategy")
        return None

    @classmethod
    def enrich_metrics(cls, data: Dict[str, Any], symbol: str = '') -> Dict[str, Any]:
        """
        Enrich financial data by calculating missing critical metrics.

        Args:
            data: Financial data dict (will be modified in-place)
            symbol: Stock ticker (for logging)

        Returns:
            Enriched data dict (same object, modified in-place)
        """
        # Calculate missing total_liabilities
        if 'total_liabilities' not in data or data['total_liabilities'] is None:
            calculated = cls.calculate_total_liabilities(data, symbol)
            if calculated is not None:
                data['total_liabilities'] = calculated
                data['_total_liabilities_calculated'] = True

        # Calculate missing total_revenue
        if 'total_revenue' not in data or data['total_revenue'] is None:
            calculated = cls.calculate_total_revenue(data, symbol)
            if calculated is not None:
                data['total_revenue'] = calculated
                data['_total_revenue_calculated'] = True

        # Calculate missing net_income
        if 'net_income' not in data or data['net_income'] is None:
            calculated = cls.calculate_net_income(data, symbol)
            if calculated is not None:
                data['net_income'] = calculated
                data['_net_income_calculated'] = True

        return data


def enrich_financial_data(data: Dict[str, Any], symbol: str = '') -> Dict[str, Any]:
    """
    Convenience function to enrich financial data.

    Args:
        data: Financial data dict
        symbol: Stock ticker

    Returns:
        Enriched data dict
    """
    return FinancialCalculator.enrich_metrics(data, symbol)
