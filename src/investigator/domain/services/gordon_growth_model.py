"""
Gordon Growth Model (GGM) - Dividend Discount Model

Calculates intrinsic value for dividend-paying stocks using:
    Fair Value = D₁ / (r - g)

Where:
    D₁ = Expected dividend next year = D₀ × (1 + g)
    r = Required rate of return (Cost of Equity)
    g = Sustainable growth rate

Model constraints:
    - Only applicable to dividend-paying stocks
    - Growth rate (g) must be less than required return (r)
    - Growth rate should be sustainable (typically < 6%)
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GordonGrowthModel:
    """
    Gordon Growth Model for dividend-paying stocks

    Calculates fair value using dividend discount model with constant growth assumption.
    """

    def __init__(self, symbol: str, quarterly_metrics: List[Dict], multi_year_data: List[Dict], db_manager):
        """
        Initialize Gordon Growth Model

        Args:
            symbol: Stock ticker symbol
            quarterly_metrics: List of quarterly financial metrics
            multi_year_data: Multi-year historical data (2+ years for growth calc)
            db_manager: Database manager for queries
        """
        self.symbol = symbol
        self.quarterly_metrics = quarterly_metrics
        self.multi_year_data = multi_year_data
        self.db_manager = db_manager

    def calculate_ggm_valuation(self, cost_of_equity: float) -> Dict:
        """
        Calculate Gordon Growth Model valuation

        Args:
            cost_of_equity: Required rate of return (from CAPM)

        Returns:
            Dictionary with fair value, assumptions, and validation info
        """
        try:
            # Step 1: Check if company pays dividends
            latest_dps = self._get_latest_dps()
            if latest_dps <= 0:
                logger.info(f"{self.symbol} - No dividends paid, GGM not applicable")
                return {"applicable": False, "reason": "No dividends paid", "fair_value_per_share": 0}

            logger.info(f"{self.symbol} - Latest DPS: ${latest_dps:.4f}")
            logger.info(f"{self.symbol} - GGM received cost_of_equity: {cost_of_equity*100:.2f}%")

            # Step 2: Calculate sustainable growth rate
            growth_rate = self._calculate_sustainable_growth()
            logger.info(f"{self.symbol} - Sustainable growth rate: {growth_rate*100:.2f}%")

            # Step 3: Validate model constraints
            validation = self._validate_model_constraints(growth_rate, cost_of_equity)
            if not validation["valid"]:
                logger.warning(f"{self.symbol} - GGM validation failed: {validation['reason']}")
                return {
                    "applicable": False,
                    "reason": validation["reason"],
                    "fair_value_per_share": 0,
                    "warnings": validation.get("warnings", []),
                }

            # Step 4: Calculate next year's expected dividend (D₁)
            d1 = latest_dps * (1 + growth_rate)

            # Step 5: Apply GGM formula: Fair Value = D₁ / (r - g)
            fair_value = d1 / (cost_of_equity - growth_rate)

            # Step 6: Compare to current price
            current_price = self._get_current_price()
            upside_downside = ((fair_value / current_price) - 1) * 100 if current_price > 0 else 0

            logger.info(
                f"{self.symbol} - GGM Fair Value: ${fair_value:.2f}, "
                f"Current: ${current_price:.2f}, Upside: {upside_downside:+.1f}%"
            )

            return {
                "applicable": True,
                "model": "ggm",  # Must match weight dict key (not "Gordon Growth Model")
                "methodology": "Gordon Growth Model",  # Human-readable name
                "fair_value_per_share": round(fair_value, 2),
                "current_price": round(current_price, 2),
                "upside_downside_pct": round(upside_downside, 1),
                "valuation_assessment": self._get_valuation_assessment(upside_downside),
                "assumptions": {
                    "current_dps": round(latest_dps, 4),
                    "expected_dps_next_year": round(d1, 4),
                    "growth_rate": round(growth_rate * 100, 2),  # As percentage
                    "required_return": round(cost_of_equity * 100, 2),  # As percentage
                    "dividend_yield": round((latest_dps / current_price * 100), 2) if current_price > 0 else 0,
                },
                "validation": validation,
            }

        except Exception as e:
            logger.error(f"Error calculating GGM for {self.symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {"applicable": False, "reason": f"Calculation error: {str(e)}", "fair_value_per_share": 0}

    def _get_latest_dps(self) -> float:
        """
        Get latest Dividends Per Share (DPS)

        Uses TTM (Trailing Twelve Months) dividends

        Returns:
            DPS in dollars
        """
        if not self.quarterly_metrics:
            return 0

        # Get last 4 quarters for TTM
        last_4q = self.quarterly_metrics[-4:] if len(self.quarterly_metrics) >= 4 else self.quarterly_metrics

        # Sum dividends from last 4 quarters
        # Check multiple fields for different SEC reporting formats and nested structures
        ttm_dividends = 0
        for q in last_4q:
            # Try nested cash_flow structure first (from QuarterlyData.to_dict())
            cash_flow = q.get("cash_flow", {})
            if isinstance(cash_flow, dict):
                nested_div = self._to_float(cash_flow.get("dividends_paid", 0) or 0)
            else:
                nested_div = 0

            # Try flat structure and various SEC tag names
            div = (
                nested_div  # Nested structure from QuarterlyData.to_dict()
                or self._to_float(q.get("dividends_paid", 0) or 0)  # Flat structure
                or self._to_float(q.get("PaymentsOfOrdinaryDividends", 0) or 0)  # JNJ uses this
                or self._to_float(q.get("DividendsCommonStockCash", 0) or 0)  # Alternative tag
                or self._to_float(q.get("PaymentsOfDividends", 0) or 0)  # AAPL uses this
                or self._to_float(q.get("PaymentsOfDividendsCommonStock", 0) or 0)  # Alternative
                or self._to_float(q.get("common_stock_dividends", 0) or 0)  # Canonical fallback
            )
            ttm_dividends += abs(div)

        # Get shares outstanding
        shares = self._get_shares_outstanding()

        if shares <= 0:
            return 0

        dps = abs(ttm_dividends) / shares if shares > 0 else 0
        return dps

    def _calculate_sustainable_growth(self) -> float:
        """
        Calculate sustainable growth rate using two methods:

        Method 1: Historical dividend growth (CAGR of dividends over 2+ years)
        Method 2: ROE × (1 - Payout Ratio) [Sustainable growth rate formula]

        Returns the more conservative (lower) of the two estimates

        Returns:
            Growth rate as decimal (e.g., 0.05 for 5%)
        """
        # Method 1: Historical dividend growth
        historical_growth = self._calculate_historical_dividend_growth()

        # Method 2: Sustainable growth from fundamentals
        fundamental_growth = self._calculate_fundamental_growth()

        # Use the more conservative estimate
        if historical_growth > 0 and fundamental_growth > 0:
            growth_rate = min(historical_growth, fundamental_growth)
            logger.info(
                f"{self.symbol} - Growth rates: Historical={historical_growth*100:.2f}%, "
                f"Fundamental={fundamental_growth*100:.2f}%, Using={growth_rate*100:.2f}%"
            )
        elif historical_growth > 0:
            growth_rate = historical_growth
            logger.info(f"{self.symbol} - Using historical growth: {growth_rate*100:.2f}%")
        elif fundamental_growth > 0:
            growth_rate = fundamental_growth
            logger.info(f"{self.symbol} - Using fundamental growth: {growth_rate*100:.2f}%")
        else:
            growth_rate = 0.03  # Default: 3% (GDP growth)
            logger.warning(f"{self.symbol} - Insufficient data, using default growth: {growth_rate*100:.2f}%")

        # Cap growth at reasonable level (6% max for dividend growth)
        growth_rate = min(growth_rate, 0.06)

        return growth_rate

    def _calculate_historical_dividend_growth(self) -> float:
        """
        Calculate CAGR of dividends over available history (2+ years ideal)

        Returns:
            Annual growth rate as decimal
        """
        if not self.multi_year_data or len(self.multi_year_data) < 2:
            return 0

        # Extract dividend values (only positive values)
        dividend_values = [
            abs(self._to_float(y.get("dividends_paid", 0)))
            for y in self.multi_year_data
            if self._to_float(y.get("dividends_paid", 0)) != 0
        ]

        if len(dividend_values) < 2:
            return 0

        # Calculate CAGR
        first_div = dividend_values[0]
        last_div = dividend_values[-1]
        years = len(dividend_values) - 1

        if first_div > 0 and last_div > first_div:
            cagr = ((last_div / first_div) ** (1 / years)) - 1
            # Cap at 15% for dividend growth (rarely sustainable above this)
            return max(0, min(cagr, 0.15))

        return 0

    def _calculate_fundamental_growth(self) -> float:
        """
        Calculate sustainable growth: g = ROE × (1 - Payout Ratio)

        Returns:
            Growth rate as decimal
        """
        if not self.quarterly_metrics:
            return 0

        latest = self.quarterly_metrics[-1]

        # Get ROE (Return on Equity)
        roe = self._to_float(latest.get("roe", 0) or 0)

        # Calculate payout ratio = Dividends / Net Income
        net_income = self._to_float(latest.get("net_income", 0) or 0)
        dividends = abs(self._to_float(latest.get("dividends_paid", 0) or 0))

        if net_income <= 0 or roe <= 0:
            return 0

        payout_ratio = dividends / abs(net_income) if net_income != 0 else 0
        payout_ratio = min(payout_ratio, 1.0)  # Cap at 100%

        # Sustainable growth = ROE × (1 - Payout Ratio)
        # Note: ROE may be stored as percentage (e.g., 15.5) or decimal (0.155)
        # Normalize to decimal
        if roe > 1:
            roe = roe / 100  # Convert percentage to decimal

        sustainable_growth = roe * (1 - payout_ratio)

        # Sanity check: cap at 20%
        return max(0, min(sustainable_growth, 0.20))

    def _validate_model_constraints(self, growth_rate: float, cost_of_equity: float) -> Dict:
        """
        Validate GGM model constraints

        Constraints:
        1. Growth rate must be less than required return (g < r)
        2. Growth rate should be reasonable (typically < 6% for dividends)
        3. Company should have consistent dividend history

        Returns:
            Dict with 'valid' flag and 'reason' if invalid
        """
        warnings = []

        # Constraint 1: g < r (critical)
        if growth_rate >= cost_of_equity:
            return {
                "valid": False,
                "reason": f"Growth rate ({growth_rate*100:.2f}%) >= required return ({cost_of_equity*100:.2f}%)",
                "warnings": warnings,
            }

        # Constraint 2: Growth rate reasonableness
        if growth_rate > 0.06:
            warnings.append(f"High growth rate ({growth_rate*100:.2f}%) may not be sustainable for dividends")

        # Constraint 3: Check dividend consistency (at least 2 years of dividends)
        if self.multi_year_data and len(self.multi_year_data) >= 2:
            dividend_years = sum(1 for y in self.multi_year_data if abs(self._to_float(y.get("dividends_paid", 0))) > 0)
            if dividend_years < 2:
                warnings.append("Limited dividend history (< 2 years), valuation may be less reliable")

        return {"valid": True, "warnings": warnings}

    def _get_current_price(self) -> float:
        """
        Get current stock price

        Returns:
            Current price in dollars
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(self.symbol)
            current_price = ticker.info.get("currentPrice", 0)
            if current_price > 0:
                return float(current_price)
        except Exception as e:
            logger.warning(f"Could not fetch current price for {self.symbol}: {e}")

        return 100.0  # Fallback value

    def _get_shares_outstanding(self) -> float:
        """
        Get shares outstanding

        Returns:
            Number of shares
        """
        if self.quarterly_metrics:
            latest = self.quarterly_metrics[-1]
            shares = self._to_float(latest.get("shares_outstanding", 0))
            if shares > 0:
                return shares

        try:
            import yfinance as yf

            ticker = yf.Ticker(self.symbol)
            shares = ticker.info.get("sharesOutstanding", 0)
            if shares > 0:
                return float(shares)
        except Exception as e:
            logger.warning(f"Could not fetch shares outstanding for {self.symbol}: {e}")

        return 1000000000  # Fallback: 1B shares

    def _get_valuation_assessment(self, upside_downside_pct: float) -> str:
        """
        Assess valuation based on upside/downside percentage

        Args:
            upside_downside_pct: Percentage difference from current price

        Returns:
            Valuation assessment string
        """
        if upside_downside_pct > 30:
            return "Significantly Undervalued"
        elif upside_downside_pct > 15:
            return "Undervalued"
        elif upside_downside_pct > -10:
            return "Fairly Valued"
        elif upside_downside_pct > -25:
            return "Overvalued"
        else:
            return "Significantly Overvalued"

    def _to_float(self, value) -> float:
        """
        Convert value to float, handling Decimal type from bulk tables

        Args:
            value: Value to convert (float, int, Decimal, or None)

        Returns:
            Float value, or 0 if None
        """
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        return float(value)
