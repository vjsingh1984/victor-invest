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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from decimal import Decimal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from utils.valuation.company_profile import CompanyProfile


class GordonGrowthModel:
    """
    Gordon Growth Model for dividend-paying stocks

    Calculates fair value using dividend discount model with constant growth assumption.
    """

    def __init__(
        self,
        symbol: str,
        quarterly_metrics: List[Dict],
        multi_year_data: List[Dict],
        db_manager,
        company_profile: Optional["CompanyProfile"] = None,
    ):
        """
        Initialize Gordon Growth Model

        Args:
            symbol: Stock ticker symbol
            quarterly_metrics: List of quarterly financial metrics
            multi_year_data: Multi-year historical data (2+ years for growth calc)
            db_manager: Database manager for queries
            company_profile: Optional profile snapshot with cached fundamentals
        """
        self.symbol = symbol
        self.quarterly_metrics = quarterly_metrics
        self.multi_year_data = multi_year_data
        self.db_manager = db_manager
        self.company_profile = company_profile

    def calculate_ggm_valuation(self, cost_of_equity: float, terminal_growth_rate: Optional[float] = None) -> Dict:
        """
        Calculate Gordon Growth Model valuation

        Args:
            cost_of_equity: Required rate of return (from CAPM)
            terminal_growth_rate: Optional pre-calculated terminal growth rate (decimal, e.g., 0.035 for 3.5%).
                                 If provided, uses this unified rate from TerminalGrowthCalculator.
                                 If None, calculates internally using sustainable growth logic (backward compatible).

        Returns:
            Dictionary with fair value, assumptions, and validation info
        """
        try:
            logger.info(f"🔍 [GGM_START] {self.symbol} - Initializing Gordon Growth Model valuation")
            logger.info(f"🔍 [GGM_INPUTS] {self.symbol} - Cost of Equity (r): {cost_of_equity*100:.2f}%")

            # Step 1: Check if company pays dividends
            logger.info(f"🔍 [GGM_STAGE_1] {self.symbol} - Checking dividend eligibility...")
            latest_dps = self._get_latest_dps()
            if latest_dps <= 0:
                logger.info(f"🔍 [GGM_INELIGIBLE] {self.symbol} - No dividends paid (DPS=${latest_dps:.4f}), GGM not applicable")
                return {
                    'applicable': False,
                    'reason': 'No dividends paid',
                    'fair_value_per_share': 0
                }

            logger.info(f"🔍 [GGM_DIVIDEND] {self.symbol} - ✅ Dividend-paying stock: TTM DPS = ${latest_dps:.4f}")

            # Step 2: Determine growth rate (unified or internal calculation)
            if terminal_growth_rate is not None:
                # Use pre-calculated unified terminal growth rate (from TerminalGrowthCalculator)
                growth_rate = terminal_growth_rate
                logger.info(
                    f"🔍 [GGM_GROWTH] {self.symbol} - Terminal Growth (unified): {growth_rate*100:.2f}% "
                    f"[Using pre-calculated rate from TerminalGrowthCalculator]"
                )
            else:
                # Fall back to internal calculation (backward compatible)
                logger.info(f"🔍 [GGM_STAGE_2] {self.symbol} - Calculating sustainable growth rate...")
                growth_rate = self._calculate_sustainable_growth()
                logger.info(f"🔍 [GGM_GROWTH] {self.symbol} - Sustainable growth rate (g) [internal]: {growth_rate*100:.2f}%")

            # Step 3: Validate model constraints
            logger.info(f"🔍 [GGM_STAGE_3] {self.symbol} - Validating model constraints (g < r)...")
            validation = self._validate_model_constraints(growth_rate, cost_of_equity)
            if not validation['valid']:
                logger.warning(f"🔍 [GGM_CONSTRAINT_FAIL] {self.symbol} - Validation failed: {validation['reason']}")
                return {
                    'applicable': False,
                    'reason': validation['reason'],
                    'fair_value_per_share': 0,
                    'warnings': validation.get('warnings', [])
                }

            if validation.get('warnings'):
                for warning in validation['warnings']:
                    logger.warning(f"🔍 [GGM_WARNING] {self.symbol} - {warning}")

            logger.info(f"🔍 [GGM_CONSTRAINT_PASS] {self.symbol} - ✅ Model constraints satisfied (g={growth_rate*100:.2f}% < r={cost_of_equity*100:.2f}%)")

            # Step 4: Calculate next year's expected dividend (D₁)
            logger.info(f"🔍 [GGM_STAGE_4] {self.symbol} - Calculating expected dividend D₁ = D₀ × (1 + g)")
            d1 = latest_dps * (1 + growth_rate)
            logger.info(
                f"🔍 [GGM_D1_CALC] {self.symbol} - "
                f"D₁ = ${latest_dps:.4f} × (1 + {growth_rate*100:.2f}%) = ${d1:.4f}"
            )

            # Step 5: Apply GGM formula: Fair Value = D₁ / (r - g)
            logger.info(f"🔍 [GGM_STAGE_5] {self.symbol} - Applying GGM formula: Fair Value = D₁ / (r - g)")
            denominator = cost_of_equity - growth_rate
            fair_value = d1 / denominator
            logger.info(
                f"🔍 [GGM_FORMULA] {self.symbol} - "
                f"Fair Value = ${d1:.4f} / ({cost_of_equity*100:.2f}% - {growth_rate*100:.2f}%) = "
                f"${d1:.4f} / {denominator*100:.4f}% = ${fair_value:.2f}"
            )

            # Step 6: Compare to current price
            logger.info(f"🔍 [GGM_STAGE_6] {self.symbol} - Comparing to current market price...")
            current_price = self._get_current_price()
            upside_downside = ((fair_value / current_price) - 1) * 100 if current_price > 0 else 0

            assessment = self._get_valuation_assessment(upside_downside)

            dividend_yield = (latest_dps / current_price * 100) if current_price > 0 else 0

            logger.info(
                f"🔍 [GGM_RESULT] {self.symbol} - "
                f"Fair Value: ${fair_value:.2f} vs Current: ${current_price:.2f} | "
                f"Upside: {upside_downside:+.1f}% | Assessment: {assessment}"
            )
            logger.info(
                f"🔍 [GGM_METRICS] {self.symbol} - "
                f"Dividend Yield: {dividend_yield:.2f}% | "
                f"Growth Rate: {growth_rate*100:.2f}% | "
                f"Required Return: {cost_of_equity*100:.2f}%"
            )

            return {
                'applicable': True,
                'model': 'Gordon Growth Model',
                'fair_value_per_share': round(fair_value, 2),
                'current_price': round(current_price, 2),
                'upside_downside_pct': round(upside_downside, 1),
                'valuation_assessment': assessment,
                'assumptions': {
                    'current_dps': round(latest_dps, 4),
                    'expected_dps_next_year': round(d1, 4),
                    'growth_rate': round(growth_rate * 100, 2),  # As percentage
                    'required_return': round(cost_of_equity * 100, 2),  # As percentage
                    'dividend_yield': round(dividend_yield, 2)
                },
                'validation': validation
            }

        except Exception as e:
            logger.error(f"🔍 [GGM_ERROR] {self.symbol} - Calculation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'applicable': False,
                'reason': f'Calculation error: {str(e)}',
                'fair_value_per_share': 0
            }

    def _get_latest_dps(self) -> float:
        """
        Get latest Dividends Per Share (DPS)

        Uses TTM (Trailing Twelve Months) dividends with rolling quarters

        Strategy:
        - Get 4 most recent quarters (with computed Q4 if needed)
        - Ensures current TTM, not stale FY data

        Returns:
            DPS in dollars
        """
        if not self.quarterly_metrics:
            return 0

        # Import quarterly calculator
        from utils.quarterly_calculator import get_rolling_ttm_periods

        # Get 4 most recent quarters (with Q4 computed if needed)
        ttm_periods = get_rolling_ttm_periods(self.quarterly_metrics, compute_missing=True, num_quarters=4)

        if not ttm_periods:
            logger.warning(f"{self.symbol} - No quarterly data available for TTM DPS calculation")
            return 0

        ttm_dividends = 0.0
        for period in ttm_periods:
            cash_flow = period.get('cash_flow', {})
            if cash_flow.get('is_ytd'):
                logger.error(f"{self.symbol} - YTD data detected in TTM DPS calculation!")
                raise ValueError(f"YTD data not allowed for TTM calculation. Period: {period.get('fiscal_period')}")

            dividend_value = self._extract_dividends(period)
            if dividend_value is not None:
                ttm_dividends += abs(dividend_value)

        if ttm_dividends <= 0:
            logger.info(
                f"{self.symbol} - TTM dividends sum to $0.0M across {len(ttm_periods)} quarters "
                f"(dividends not reported in processed data)"
            )
            return 0

        # Get shares outstanding
        shares = self._get_shares_outstanding()

        if shares <= 0:
            logger.warning(f"{self.symbol} - Unable to determine shares outstanding for DPS calculation")
            return 0

        logger.info(
            f"{self.symbol} - TTM dividends: ${ttm_dividends / 1e6:.1f}M across {len(ttm_periods)} quarters "
            f"(shares used: {shares:,.0f})"
        )

        return ttm_dividends / shares

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
        Calculate dividend growth rate using quarterly YoY patterns

        Strategy:
        1. Get 8 quarters (2 years) with computed Q4s
        2. Analyze YoY dividend growth trends
        3. Use average YoY growth (more accurate than annual CAGR)
        4. Accounts for seasonality in dividend payments

        Returns:
            Annual growth rate as decimal
        """
        # Try quarterly-based dividend growth analysis first
        if self.quarterly_metrics and len(self.quarterly_metrics) >= 6:
            from utils.quarterly_calculator import get_rolling_ttm_periods, analyze_quarterly_patterns

            # Get 8 quarters for 2-year trend analysis
            quarters_8 = get_rolling_ttm_periods(
                self.quarterly_metrics,
                compute_missing=True,
                num_quarters=8
            )

            if len(quarters_8) >= 6:  # Need at least 6 quarters for meaningful analysis
                # Analyze dividend patterns
                patterns = analyze_quarterly_patterns(quarters_8, 'dividends_paid')

                if patterns and 'avg_yoy_growth' in patterns:
                    yoy_growth = patterns['avg_yoy_growth'] / 100  # Convert % to decimal

                    logger.info(
                        f"{self.symbol} - Quarterly dividend analysis: "
                        f"YoY growth {patterns['avg_yoy_growth']:.1f}%, "
                        f"Trend: {patterns.get('trend', 'N/A')}, "
                        f"Seasonality variance: {patterns.get('seasonality', {}).get('variance_pct', 0):.1f}%"
                    )

                    # Cap at 15% for dividend growth (rarely sustainable above this)
                    return max(0, min(yoy_growth, 0.15))

        # Fallback: Use multi-year annual data
        if not self.multi_year_data or len(self.multi_year_data) < 2:
            return 0

        # Extract dividend values (only positive values)
        dividend_values = [
            abs(self._to_float(y.get('dividends_paid', 0)))
            for y in self.multi_year_data
            if self._to_float(y.get('dividends_paid', 0)) != 0
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
        roe = self._to_float(latest.get('roe', 0) or 0)

        # Calculate payout ratio = Dividends / Net Income
        net_income = self._to_float(latest.get('net_income', 0) or 0)
        dividends = abs(self._to_float(latest.get('dividends_paid', 0) or 0))

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
                'valid': False,
                'reason': f'Growth rate ({growth_rate*100:.2f}%) >= required return ({cost_of_equity*100:.2f}%)',
                'warnings': warnings
            }

        # Constraint 2: Growth rate reasonableness
        if growth_rate > 0.06:
            warnings.append(f'High growth rate ({growth_rate*100:.2f}%) may not be sustainable for dividends')

        # Constraint 3: Check dividend consistency (at least 2 years of dividends)
        if self.multi_year_data and len(self.multi_year_data) >= 2:
            dividend_years = sum(
                1 for y in self.multi_year_data
                if abs(self._to_float(y.get('dividends_paid', 0))) > 0
            )
            if dividend_years < 2:
                warnings.append('Limited dividend history (< 2 years), valuation may be less reliable')

        return {
            'valid': True,
            'warnings': warnings
        }

    def _get_current_price(self) -> float:
        """
        Get current stock price

        Returns:
            Current price in dollars
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            current_price = ticker.info.get('currentPrice', 0)
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
        if self.company_profile:
            profile_shares = getattr(self.company_profile, 'shares_outstanding', None)
            if profile_shares:
                logger.info(f"{self.symbol} - GGM using shares from company profile: {profile_shares:,.0f}")
                return float(profile_shares)

        if self.quarterly_metrics:
            for period in reversed(self.quarterly_metrics):
                shares = self._extract_shares(period, prefer_diluted=True)
                if shares:
                    logger.info(f"{self.symbol} - GGM using shares from quarterly metrics: {shares:,.0f}")
                    return shares

        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            shares = ticker.info.get('sharesOutstanding', 0)
            if shares > 0:
                return float(shares)
        except Exception as e:
            logger.warning(f"Could not fetch shares outstanding for {self.symbol}: {e}")

        logger.warning(f"{self.symbol} - Falling back to default share count for valuation")
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
            return 'Significantly Undervalued'
        elif upside_downside_pct > 15:
            return 'Undervalued'
        elif upside_downside_pct > -10:
            return 'Fairly Valued'
        elif upside_downside_pct > -25:
            return 'Overvalued'
        else:
            return 'Significantly Overvalued'

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

    def _extract_dividends(self, period: Dict) -> Optional[float]:
        """Extract dividends_paid from a quarterly period dict."""
        if not isinstance(period, dict):
            return None

        cash_flow = period.get('cash_flow')
        if isinstance(cash_flow, dict):
            dividends = cash_flow.get('dividends_paid')
            if dividends is not None:
                return self._to_float(dividends)

        financial_data = period.get('financial_data')
        if isinstance(financial_data, dict):
            dividends = financial_data.get('dividends_paid')
            if dividends is not None:
                return self._to_float(dividends)

        dividends = period.get('dividends_paid')
        if dividends is not None:
            return self._to_float(dividends)

        return None

    def _extract_shares(self, period: Dict, prefer_diluted: bool = False) -> Optional[float]:
        """Extract share count from a quarterly period dict."""
        if not isinstance(period, dict):
            return None

        preferred_keys = [
            'weighted_average_diluted_shares_outstanding',
            'shares_outstanding',
            'common_stock_shares_outstanding',
            'total_shares_outstanding',
        ]
        if not prefer_diluted:
            preferred_keys = preferred_keys[1:] + preferred_keys[:1]

        for key in preferred_keys:
            value = period.get(key)
            if value:
                return self._to_float(value)

        financial_data = period.get('financial_data')
        if isinstance(financial_data, dict):
            for key in preferred_keys:
                value = financial_data.get(key)
                if value:
                    return self._to_float(value)

        return None
