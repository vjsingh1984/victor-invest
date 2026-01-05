# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Ratio Calculator - Standardized financial ratio calculations.

This service centralizes ratio calculations that were previously duplicated in:
- scripts/rl_backtest.py (calculate_ratios)
- victor_invest/tools/valuation.py

Key features:
- Consistent ratio calculation logic
- Handles different data formats (nested vs flat)
- Validates inputs and provides meaningful defaults
- Calculates derived metrics (Rule of 40, enterprise value, etc.)

Example:
    from investigator.domain.services.valuation_shared import RatioCalculator

    calc = RatioCalculator()

    # Calculate all ratios
    ratios = calc.calculate_all_ratios(
        financials=ttm_data,
        current_price=150.0,
        shares=1_000_000_000,
    )

    # Calculate specific ratio
    pe = calc.calculate_pe_ratio(current_price=150.0, eps=5.0)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RatioCalculator:
    """
    Standardized financial ratio calculator.

    Provides consistent ratio calculations across all valuation consumers.
    """

    def calculate_pe_ratio(
        self,
        current_price: float,
        eps: Optional[float] = None,
        net_income: Optional[float] = None,
        shares: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate Price-to-Earnings ratio.

        Args:
            current_price: Current stock price
            eps: Earnings per share (if available)
            net_income: TTM net income (used if eps not provided)
            shares: Shares outstanding (used with net_income)

        Returns:
            P/E ratio or None if not calculable
        """
        if eps is not None and eps > 0:
            return current_price / eps

        if net_income is not None and shares is not None and shares > 0:
            calculated_eps = net_income / shares
            if calculated_eps > 0:
                return current_price / calculated_eps

        return None

    def calculate_ps_ratio(
        self,
        market_cap: Optional[float] = None,
        revenue: Optional[float] = None,
        current_price: Optional[float] = None,
        shares: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate Price-to-Sales ratio.

        Args:
            market_cap: Market capitalization
            revenue: TTM revenue
            current_price: Current stock price (used if market_cap not provided)
            shares: Shares outstanding (used with current_price)

        Returns:
            P/S ratio or None if not calculable
        """
        if market_cap is None and current_price and shares:
            market_cap = current_price * shares

        if market_cap and revenue and revenue > 0:
            return market_cap / revenue

        return None

    def calculate_pb_ratio(
        self,
        current_price: float,
        book_value_per_share: Optional[float] = None,
        stockholders_equity: Optional[float] = None,
        shares: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate Price-to-Book ratio.

        Args:
            current_price: Current stock price
            book_value_per_share: Book value per share (if available)
            stockholders_equity: Total equity (used if bvps not provided)
            shares: Shares outstanding (used with stockholders_equity)

        Returns:
            P/B ratio or None if not calculable
        """
        if book_value_per_share is not None and book_value_per_share > 0:
            return current_price / book_value_per_share

        if stockholders_equity is not None and shares is not None and shares > 0:
            bvps = stockholders_equity / shares
            if bvps > 0:
                return current_price / bvps

        return None

    def calculate_ev_ebitda(
        self,
        market_cap: float,
        ebitda: float,
        total_debt: Optional[float] = None,
        cash: Optional[float] = None,
        long_term_debt: Optional[float] = None,
        short_term_debt: Optional[float] = None,
        cash_and_equivalents: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate EV/EBITDA ratio.

        Enterprise Value = Market Cap + Total Debt - Cash

        Args:
            market_cap: Market capitalization
            ebitda: TTM EBITDA
            total_debt: Total debt (if available)
            cash: Cash and equivalents
            long_term_debt: Long-term debt (used if total_debt not provided)
            short_term_debt: Short-term debt (used if total_debt not provided)
            cash_and_equivalents: Cash (used if cash not provided)

        Returns:
            EV/EBITDA ratio or None if not calculable
        """
        if ebitda is None or ebitda <= 0:
            return None

        # Calculate total debt
        if total_debt is None:
            total_debt = (long_term_debt or 0) + (short_term_debt or 0)

        # Get cash
        if cash is None:
            cash = cash_and_equivalents or 0

        # Enterprise Value
        ev = market_cap + total_debt - cash

        return ev / ebitda if ebitda > 0 else None

    def calculate_payout_ratio(
        self,
        dividends_paid: float,
        net_income: float,
    ) -> Optional[float]:
        """
        Calculate dividend payout ratio.

        Args:
            dividends_paid: TTM dividends paid (absolute value)
            net_income: TTM net income

        Returns:
            Payout ratio (0.0-1.0+) or None if not calculable
        """
        dividends = abs(dividends_paid) if dividends_paid else 0

        if net_income and net_income > 0:
            return dividends / net_income

        return None

    def calculate_margins(
        self,
        revenue: float,
        gross_profit: Optional[float] = None,
        operating_income: Optional[float] = None,
        net_income: Optional[float] = None,
        free_cash_flow: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Calculate profitability margins.

        Args:
            revenue: TTM revenue
            gross_profit: TTM gross profit
            operating_income: TTM operating income
            net_income: TTM net income
            free_cash_flow: TTM free cash flow

        Returns:
            Dict with gross_margin, operating_margin, net_margin, fcf_margin
        """
        if not revenue or revenue <= 0:
            return {
                "gross_margin": None,
                "operating_margin": None,
                "net_margin": None,
                "fcf_margin": None,
            }

        return {
            "gross_margin": gross_profit / revenue if gross_profit else None,
            "operating_margin": operating_income / revenue if operating_income else None,
            "net_margin": net_income / revenue if net_income else None,
            "fcf_margin": free_cash_flow / revenue if free_cash_flow else None,
        }

    def calculate_rule_of_40(
        self,
        revenue_growth_pct: float,
        fcf_margin_pct: Optional[float] = None,
        operating_margin_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate Rule of 40 score.

        Rule of 40 = Revenue Growth % + Profit Margin %
        (FCF margin preferred, operating margin as fallback)

        Args:
            revenue_growth_pct: Revenue growth percentage (e.g., 20 for 20%)
            fcf_margin_pct: FCF margin percentage (preferred)
            operating_margin_pct: Operating margin percentage (fallback)

        Returns:
            Rule of 40 score
        """
        margin = fcf_margin_pct if fcf_margin_pct is not None else (operating_margin_pct or 0)
        return revenue_growth_pct + margin

    def calculate_roe(
        self,
        net_income: float,
        stockholders_equity: float,
    ) -> Optional[float]:
        """
        Calculate Return on Equity.

        Args:
            net_income: TTM net income
            stockholders_equity: Stockholders' equity

        Returns:
            ROE or None if not calculable
        """
        if stockholders_equity and stockholders_equity > 0:
            return net_income / stockholders_equity
        return None

    def calculate_debt_to_equity(
        self,
        total_debt: Optional[float] = None,
        stockholders_equity: Optional[float] = None,
        long_term_debt: Optional[float] = None,
        short_term_debt: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate Debt-to-Equity ratio.

        Args:
            total_debt: Total debt (if available)
            stockholders_equity: Stockholders' equity
            long_term_debt: Long-term debt (used if total_debt not provided)
            short_term_debt: Short-term debt (used if total_debt not provided)

        Returns:
            D/E ratio or None if not calculable
        """
        if total_debt is None:
            total_debt = (long_term_debt or 0) + (short_term_debt or 0)

        if stockholders_equity and stockholders_equity > 0:
            return total_debt / stockholders_equity

        return None

    def calculate_all_ratios(
        self,
        financials: Dict[str, Any],
        current_price: float,
        shares: float,
        metadata: Optional[Dict[str, Any]] = None,
        revenue_growth_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate all financial ratios from TTM financials.

        Handles both nested format (income_statement/cash_flow/balance_sheet)
        and flat format.

        Args:
            financials: TTM financial data (nested or flat)
            current_price: Current stock price
            shares: Shares outstanding
            metadata: Optional metadata dict
            revenue_growth_pct: Revenue growth % for Rule of 40

        Returns:
            Dict with all calculated ratios
        """
        # Extract values from nested or flat structure
        if "income_statement" in financials:
            # Nested structure
            income = financials.get("income_statement", {})
            cash_flow = financials.get("cash_flow", {})
            balance = financials.get("balance_sheet", {})

            revenue = income.get("total_revenue", 0) or 0
            net_income = income.get("net_income", 0) or 0
            gross_profit = income.get("gross_profit", 0) or 0
            operating_income = income.get("operating_income", 0) or 0
            ebitda = income.get("ebitda", 0) or 0

            fcf = cash_flow.get("free_cash_flow", 0) or 0
            dividends = abs(cash_flow.get("dividends_paid", 0) or 0)

            equity = balance.get("stockholders_equity", 0) or 0
            long_term_debt = balance.get("long_term_debt", 0) or 0
            short_term_debt = balance.get("short_term_debt", 0) or 0
            cash = balance.get("cash_and_equivalents", 0) or 0
        else:
            # Flat structure
            revenue = financials.get("total_revenue", 0) or financials.get("revenue", 0) or 0
            net_income = financials.get("net_income", 0) or 0
            gross_profit = financials.get("gross_profit", 0) or 0
            operating_income = financials.get("operating_income", 0) or 0
            ebitda = financials.get("ebitda", 0) or 0

            fcf = financials.get("free_cash_flow", 0) or 0
            dividends = abs(financials.get("dividends_paid", 0) or 0)

            equity = financials.get("stockholders_equity", 0) or 0
            long_term_debt = financials.get("long_term_debt", 0) or 0
            short_term_debt = financials.get("short_term_debt", 0) or 0
            cash = financials.get("cash_and_equivalents", 0) or 0

        # Calculate derived values
        market_cap = shares * current_price
        total_debt = long_term_debt + short_term_debt
        eps = net_income / shares if shares > 0 else 0
        bvps = equity / shares if shares > 0 else 0

        # Calculate margins
        margins = self.calculate_margins(
            revenue=revenue,
            gross_profit=gross_profit,
            operating_income=operating_income,
            net_income=net_income,
            free_cash_flow=fcf,
        )

        # Calculate Rule of 40
        fcf_margin_pct = (margins["fcf_margin"] or 0) * 100
        rule_of_40 = self.calculate_rule_of_40(
            revenue_growth_pct=revenue_growth_pct,
            fcf_margin_pct=fcf_margin_pct,
        )

        return {
            # Core metrics
            "market_cap": market_cap,
            "shares_outstanding": shares,
            "enterprise_value": market_cap + total_debt - cash,
            # Valuation ratios
            "pe_ratio": self.calculate_pe_ratio(current_price, eps=eps),
            "ps_ratio": self.calculate_ps_ratio(market_cap=market_cap, revenue=revenue),
            "pb_ratio": self.calculate_pb_ratio(current_price, book_value_per_share=bvps),
            "ev_ebitda": self.calculate_ev_ebitda(
                market_cap=market_cap,
                ebitda=ebitda,
                total_debt=total_debt,
                cash=cash,
            ),
            # Dividend
            "payout_ratio": self.calculate_payout_ratio(dividends, net_income),
            # Margins
            "gross_margin": margins["gross_margin"],
            "operating_margin": margins["operating_margin"],
            "net_margin": margins["net_margin"],
            "fcf_margin": margins["fcf_margin"],
            # Returns and leverage
            "roe": self.calculate_roe(net_income, equity),
            "debt_to_equity": self.calculate_debt_to_equity(
                total_debt=total_debt,
                stockholders_equity=equity,
            ),
            # Growth metrics
            "revenue_growth_pct": revenue_growth_pct,
            "rule_of_40_score": rule_of_40,
            # Per-share metrics
            "ttm_eps": eps,
            "book_value_per_share": bvps,
            # Metadata
            "quarters_available": financials.get("quarters_included", 4),
        }

    def validate_ratios(
        self,
        ratios: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate calculated ratios and flag anomalies.

        Args:
            ratios: Dict of calculated ratios

        Returns:
            Dict with validation flags
        """
        warnings = []

        # Check for extreme P/E
        pe = ratios.get("pe_ratio")
        if pe is not None:
            if pe > 100:
                warnings.append(f"High P/E: {pe:.1f}")
            elif pe < 0:
                warnings.append("Negative P/E (loss-making)")

        # Check for negative book value
        pb = ratios.get("pb_ratio")
        if pb is not None and pb < 0:
            warnings.append("Negative book value")

        # Check payout ratio
        payout = ratios.get("payout_ratio")
        if payout is not None and payout > 1.0:
            warnings.append(f"Unsustainable payout: {payout*100:.0f}%")

        # Check debt/equity
        de = ratios.get("debt_to_equity")
        if de is not None and de > 5.0:
            warnings.append(f"High leverage: D/E {de:.1f}")

        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "ratios": ratios,
        }
