"""
DCF (Discounted Cash Flow) Valuation Model

Implements professional-grade DCF analysis with:
- Free Cash Flow calculations
- WACC (Weighted Average Cost of Capital)
- Terminal value using perpetuity growth model
- Sensitivity analysis
- Fair value per share calculations
- Sector-based assumptions for terminal growth and projection horizon
"""
import logging
from typing import Dict, List, Optional
import numpy as np

from investigator.config import get_config

# Pre-profitable company industry-specific growth assumptions
from investigator.domain.services.pre_profitable_config import (
    get_growth_assumptions,
    should_use_industry_defaults,
    get_terminal_growth_rate,
    format_assumptions_log
)

logger = logging.getLogger(__name__)


class DCFValuation:
    """
    Discounted Cash Flow Valuation Model

    Calculates intrinsic value of a company by:
    1. Projecting future free cash flows
    2. Discounting them to present value using WACC
    3. Adding terminal value
    4. Converting to per-share value
    """

    def __init__(self, symbol: str, quarterly_metrics: List[Dict],
                 multi_year_data: List[Dict], db_manager):
        """
        Initialize DCF valuation

        Args:
            symbol: Stock symbol
            quarterly_metrics: List of quarterly financial metrics
            multi_year_data: Multi-year historical data
            db_manager: Database manager for queries
        """
        self.symbol = symbol
        self.quarterly_metrics = quarterly_metrics
        self.multi_year_data = multi_year_data
        self.db_manager = db_manager

        # Load DCF config
        self.dcf_config = self._load_dcf_config()
        self.wacc_config = self.dcf_config.get('wacc_parameters', {})

        # Get company sector and sector-based parameters
        self.sector = self._get_company_sector()
        self.sector_params = self._get_sector_parameters()

        # CRITICAL FIX: Cache for get_rolling_ttm_periods() results to avoid redundant Q4 computations
        # Key: (num_quarters, compute_missing) -> Value: List of periods
        # This prevents duplicate YTD conversion, Q4 computation, and fiscal year grouping
        self._ttm_cache: Dict[tuple, List[Dict]] = {}

        # Cache for expensive lookups to avoid repeated queries/warnings
        # Initialized to None (not yet computed) vs 0 (computed but unavailable)
        self._shares_outstanding_cache: Optional[float] = None
        self._current_price_cache: Optional[float] = None

        logger.info(
            f"{self.symbol} - Using sector-based DCF parameters: "
            f"Sector={self.sector}, Terminal Growth={self.sector_params['terminal_growth_rate']*100:.1f}%, "
            f"Projection Years={self.sector_params['projection_years']}"
        )

    def _get_cached_ttm_periods(self, num_quarters: int = 4, compute_missing: bool = True) -> List[Dict]:
        """
        Get TTM periods with caching to avoid redundant Q4 computations

        CRITICAL FIX: Multiple methods call get_rolling_ttm_periods() independently,
        each triggering YTD conversion, Q4 computation, and fiscal year grouping.
        This cache ensures these expensive operations happen only once per DCF execution.

        Args:
            num_quarters: Number of quarters to return (4 for TTM, 12 for trend analysis)
            compute_missing: Whether to compute missing Q4 periods

        Returns:
            List of quarterly periods (cached after first call)
        """
        cache_key = (num_quarters, compute_missing)

        if cache_key not in self._ttm_cache:
            from utils.quarterly_calculator import get_rolling_ttm_periods

            self._ttm_cache[cache_key] = get_rolling_ttm_periods(
                self.quarterly_metrics,
                compute_missing=compute_missing,
                num_quarters=num_quarters
            )

        return self._ttm_cache[cache_key]

    def _is_sec_format(self) -> bool:
        """
        Detect if quarterly_metrics is in SEC filing tool format.

        SEC format: Single item list with FY snapshot containing nested statements
        [{"fiscal_period": "FY", "income_statement": {...}, "cash_flow": {...}, ...}]

        Returns:
            True if SEC format detected, False otherwise
        """
        if not self.quarterly_metrics or len(self.quarterly_metrics) != 1:
            return False

        first_item = self.quarterly_metrics[0]
        return (
            isinstance(first_item, dict) and
            ('income_statement' in first_item or 'cash_flow' in first_item) and
            first_item.get('fiscal_period') == 'FY'
        )

    def _get_sec_data(self) -> Dict:
        """
        Get SEC format data from quarterly_metrics.

        Returns:
            The SEC data dict or empty dict if not SEC format
        """
        if self._is_sec_format():
            return self.quarterly_metrics[0]
        return {}

    def _get_ttm_metrics(self) -> Dict:
        """
        Get TTM metrics for pre-profitable detection.

        Returns:
            Dict with ttm_net_income, ttm_ebitda, sector, industry
        """
        try:
            ttm_periods = self._get_cached_ttm_periods(num_quarters=4)
            if ttm_periods and len(ttm_periods) == 4:
                # Aggregate TTM (sum of 4 quarters)
                ttm_net_income = sum(
                    q.get('income_statement', {}).get('net_income', 0)
                    for q in ttm_periods
                )
                ttm_ebitda = sum(
                    q.get('income_statement', {}).get('ebitda', 0)
                    for q in ttm_periods
                )

                # Use self.sector which is already available (from _get_company_sector())
                # Industry is not easily accessible, so pass None to use sector-level defaults
                return {
                    'ttm_net_income': ttm_net_income,
                    'ttm_ebitda': ttm_ebitda,
                    'sector': self.sector,
                    'industry': None  # Will fall back to sector-level defaults in pre_profitable_config
                }
        except Exception as e:
            logger.warning(f"{self.symbol} - Error getting TTM metrics: {e}")

        return {
            'ttm_net_income': None,
            'ttm_ebitda': None,
            'sector': None,
            'industry': None
        }

    def _load_dcf_config(self) -> Dict:
        """Load DCF configuration from config.yaml via Config class."""
        try:
            config = get_config()
            dcf_config = config.get_raw_section('dcf_valuation', {})
            if dcf_config:
                return dcf_config
            else:
                logger.debug("No dcf_valuation section in config.yaml, using defaults.")
                return self._get_default_dcf_config()
        except Exception as e:
            logger.warning(f"Could not load DCF config: {e}. Using defaults.")
            return self._get_default_dcf_config()

    def _get_default_dcf_config(self) -> Dict:
        """Return default DCF configuration when config.yaml is unavailable."""
        return {
            'default_parameters': {
                'terminal_growth_rate': 0.030,
                'projection_years': 5
            }
        }

    def _get_company_sector(self) -> str:
        """
        Get company sector from symbol table via DatabaseMarketDataFetcher.
        Database query uses COALESCE(sec_sector, Sector) for automatic fallback.
        Checks sector_override config first to correct misclassified companies.
        """
        # Check for sector override in config first
        sector_overrides = self.dcf_config.get('sector_override', {})
        if self.symbol in sector_overrides:
            override_sector = sector_overrides[self.symbol]
            logger.info(f"{self.symbol} - Sector (OVERRIDE): {override_sector} (correcting database misclassification)")
            return override_sector

        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher
            from investigator.config import get_config

            config = get_config()
            fetcher = get_market_data_fetcher(config)
            info = fetcher.get_stock_info(self.symbol)

            # Database query already handles fallback from sec_sector to Sector
            sector = info.get('sector')
            if sector:
                logger.info(f"{self.symbol} - Sector: {sector}")
                return sector

            logger.warning(f"No sector available for {self.symbol}, using Default")
            return "Default"
        except Exception as e:
            logger.warning(f"Could not fetch sector for {self.symbol}: {e}. Using Default.")
            return "Default"

    def _get_sector_parameters(self) -> Dict:
        """Get sector-specific DCF parameters"""
        sector_params = self.dcf_config.get('sector_based_parameters', {})
        default_params = self.dcf_config.get('default_parameters', {
            'terminal_growth_rate': 0.030,
            'projection_years': 5
        })

        # Get parameters for this sector, or use default
        params = sector_params.get(self.sector, sector_params.get('Default', default_params))

        return {
            'terminal_growth_rate': params.get('terminal_growth_rate', default_params['terminal_growth_rate']),
            'projection_years': params.get('projection_years', default_params['projection_years']),
            'rationale': params.get('rationale', 'Default assumptions')
        }

    def calculate_dcf_valuation(self, terminal_growth_rate: Optional[float] = None) -> Dict:
        """
        Calculate full DCF valuation with Rule of 40 integration

        Flow:
        1. Calculate Rule of 40 (efficiency metric)
        2. Detect which valuation method to use (DCF vs P/S)
        3. Route to appropriate valuation calculation
        4. Apply Rule of 40 adjustments to terminal growth if using DCF
           OR use pre-calculated unified terminal growth rate if provided

        Args:
            terminal_growth_rate: Optional pre-calculated terminal growth rate (decimal, e.g., 0.040 for 4.0%).
                                 If provided, uses this unified rate from TerminalGrowthCalculator.
                                 If None, calculates internally using Rule of 40 logic (backward compatible).

        Returns:
            Dictionary with fair value, upside/downside, and assumptions
        """
        try:
            # Step 0: Calculate Rule of 40 to determine valuation strategy
            rule_of_40_result = self._calculate_rule_of_40()
            self.rule_of_40_result = rule_of_40_result  # Store for later use in _project_fcf()
            logger.info(
                f"{self.symbol} - Rule of 40: {rule_of_40_result['score']:.1f}% "
                f"(Revenue Growth: {rule_of_40_result['revenue_growth_pct']:.1f}%, "
                f"Profit Margin: {rule_of_40_result['profit_margin_pct']:.1f}%) "
                f"→ Classification: {rule_of_40_result['classification'].upper()}"
            )

            # Step 1: Calculate Free Cash Flow (FCF)
            latest_fcf = self._calculate_latest_fcf()
            logger.info(f"{self.symbol} - Latest TTM FCF: ${latest_fcf/1e6:.1f}M")

            # Step 2: Project FCF for sector-specific number of years
            projection_years = self.sector_params['projection_years']
            fcf_projections = self._project_fcf(latest_fcf, years=projection_years)
            logger.info(f"{self.symbol} - Projected Year {projection_years} FCF: ${fcf_projections[-1]/1e6:.1f}M")

            # Step 3: Calculate WACC (Weighted Average Cost of Capital)
            wacc = self._calculate_wacc()
            logger.info(f"{self.symbol} - WACC: {wacc*100:.2f}%")

            # Step 4: Determine terminal growth rate (unified or internal calculation)
            if terminal_growth_rate is not None:
                # Use pre-calculated unified terminal growth rate (from TerminalGrowthCalculator)
                logger.info(
                    f"{self.symbol} - Terminal Growth (unified): {terminal_growth_rate*100:.2f}% "
                    f"[Using pre-calculated rate from TerminalGrowthCalculator]"
                )
            else:
                # Fall back to internal calculation (backward compatible)
                base_terminal_growth = self.sector_params['terminal_growth_rate']

                # NEW LOGIC: Skip Rule of 40 adjustment if base is already at 3.5% ceiling
                # This ensures Rule of 40 recognition only applies when base is 2-3%
                if base_terminal_growth >= 0.035:
                    # Base is already at ceiling, no adjustment allowed
                    terminal_growth_rate = base_terminal_growth
                    logger.info(
                        f"{self.symbol} - Terminal Growth (internal): {base_terminal_growth*100:.2f}% (base at ceiling, "
                        f"Rule of 40 adjustment skipped. Rule of 40: {rule_of_40_result['classification']})"
                    )
                else:
                    # Base is below ceiling, apply Rule of 40 adjustment
                    terminal_growth_adjustment = self._get_terminal_growth_adjustment(rule_of_40_result['classification'])
                    raw_terminal_growth = base_terminal_growth + terminal_growth_adjustment

                    # Ensure terminal growth stays within bounds
                    # Allow 3.5% ceiling for companies earning positive quality adjustment
                    min_terminal = self.dcf_config.get('default_parameters', {}).get('min_terminal_growth_rate', 0.020)
                    max_terminal_base = self.dcf_config.get('default_parameters', {}).get('max_terminal_growth_rate', 0.030)

                    # If company earned positive quality adjustment, allow 3.5% ceiling (vs 3.0% standard)
                    if terminal_growth_adjustment > 0:
                        max_terminal = 0.035  # Allow 3.5% for exceptional companies
                        ceiling_reason = "exceptional quality (Rule of 40 excellent)"
                    else:
                        max_terminal = max_terminal_base  # Use config ceiling (3.0%)
                        ceiling_reason = "standard"

                    terminal_growth_rate = max(min(raw_terminal_growth, max_terminal), min_terminal)

                    # Log with proper math showing calculation and clamping
                    if raw_terminal_growth != terminal_growth_rate:
                        logger.info(
                            f"{self.symbol} - Terminal Growth (internal): "
                            f"base={base_terminal_growth*100:.2f}% + Rule of 40 adj={terminal_growth_adjustment*100:+.2f}% "
                            f"({rule_of_40_result['classification']}) = {raw_terminal_growth*100:.2f}% → "
                            f"clamped to {terminal_growth_rate*100:.2f}% (ceiling: {max_terminal*100:.1f}% - {ceiling_reason})"
                        )
                    else:
                        logger.info(
                            f"{self.symbol} - Terminal Growth (internal): "
                            f"base={base_terminal_growth*100:.2f}% + Rule of 40 adj={terminal_growth_adjustment*100:+.2f}% "
                            f"({rule_of_40_result['classification']}) = {terminal_growth_rate*100:.2f}% "
                            f"(within ceiling: {max_terminal*100:.1f}%)"
                        )

            # Step 5: Calculate terminal value
            terminal_value = self._calculate_terminal_value(fcf_projections[-1], terminal_growth_rate)
            logger.info(f"{self.symbol} - Terminal Value: ${terminal_value/1e9:.2f}B (growth rate: {terminal_growth_rate*100:.1f}%)")

            # Step 5: Discount cash flows to present value
            pv_fcf = self._discount_cash_flows(fcf_projections, wacc)
            pv_terminal = terminal_value / ((1 + wacc) ** len(fcf_projections))
            logger.info(f"{self.symbol} - PV of FCF: ${pv_fcf/1e9:.2f}B, PV of Terminal: ${pv_terminal/1e9:.2f}B")
            logger.info(
                "%s - Valuation sum: PV_FCF $%.2fB + PV_Terminal $%.2fB = Enterprise Value $%.2fB",
                self.symbol,
                pv_fcf / 1e9,
                pv_terminal / 1e9,
                (pv_fcf + pv_terminal) / 1e9,
            )

            # Step 6: Calculate enterprise value and equity value
            enterprise_value = pv_fcf + pv_terminal
            equity_value = self._calculate_equity_value(enterprise_value)
            logger.info(f"{self.symbol} - Enterprise Value: ${enterprise_value/1e9:.2f}B, Equity Value: ${equity_value/1e9:.2f}B")

            # Step 7: Get shares outstanding and PROJECT DILUTION (Fix 6)
            basic_shares = self._get_shares_outstanding()

            # CRITICAL FIX #6: Project SBC dilution into terminal year
            # High-SBC companies (ZS: 25% SBC/revenue) dilute shareholders every year
            # We project this dilution forward to get realistic terminal share count
            projected_terminal_shares = basic_shares
            annual_dilution_rate = 0.0

            try:
                # Get annual SBC and current market data
                sbc_annual = self._get_latest_sbc()
                current_price = self._get_current_price()

                if sbc_annual and basic_shares > 0 and current_price > 0:
                    # Calculate SBC dilution rate: SBC_dollars / (shares * price)
                    # This is the % of market cap being diluted per year
                    market_cap = basic_shares * current_price
                    annual_dilution_rate = sbc_annual / market_cap

                    # Only apply dilution adjustment if meaningful (>2% per year)
                    if annual_dilution_rate > 0.02:
                        # Project shares forward to terminal year (end of projection period)
                        projection_years = self.sector_params['projection_years']
                        projected_terminal_shares = basic_shares * ((1 + annual_dilution_rate) ** projection_years)

                        dilution_pct = ((projected_terminal_shares - basic_shares) / basic_shares * 100)

                        logger.info(
                            f"🔍 [SBC_DILUTION] {self.symbol} - Projecting dilution impact\n"
                            f"  Annual SBC: ${sbc_annual/1e9:.2f}B\n"
                            f"  Market cap: ${market_cap/1e9:.2f}B\n"
                            f"  Annual dilution rate: {annual_dilution_rate*100:.2f}%/year\n"
                            f"  Basic shares today: {basic_shares/1e6:.1f}M\n"
                            f"  Projected shares (Year {projection_years}): {projected_terminal_shares/1e6:.1f}M\n"
                            f"  Total dilution: {dilution_pct:+.1f}% over {projection_years} years\n"
                            f"  Per-share impact: {-dilution_pct/(1 + dilution_pct/100)*100:.1f}% (value spread over more shares)"
                        )
                    else:
                        logger.info(
                            f"🔍 [SBC_DILUTION] {self.symbol} - Low SBC dilution ({annual_dilution_rate*100:.2f}%/year), "
                            f"using basic shares without adjustment"
                        )
                else:
                    logger.debug(f"{self.symbol} - Insufficient data for SBC dilution projection, using basic shares")

            except Exception as e:
                logger.debug(f"{self.symbol} - Could not calculate SBC dilution: {e}")

            # Use projected diluted shares for per-share valuation
            dcf_fair_value_per_share = equity_value / projected_terminal_shares if projected_terminal_shares > 0 else 0

            if projected_terminal_shares != basic_shares:
                logger.info(
                    f"{self.symbol} - DCF Fair Share Price: ${dcf_fair_value_per_share:.2f} "
                    f"(Shares: {basic_shares/1e6:.1f}M basic → {projected_terminal_shares/1e6:.1f}M projected terminal)"
                )
            else:
                logger.info(
                    f"{self.symbol} - DCF Fair Share Price: ${dcf_fair_value_per_share:.2f} "
                    f"(Shares Outstanding: {basic_shares:,.0f})"
                )

            # Step 8: Compare to current price
            current_price = self._get_current_price()

            # Optional: Rule-of-40-based P/S valuation
            ps_valuation = None
            if basic_shares > 0:
                ps_valuation = self._calculate_ps_valuation(
                    rule_of_40_result=rule_of_40_result,
                    shares_outstanding=basic_shares,
                    current_price=current_price,
                )

            valuation_breakdown = []
            dcf_weight = 1.0
            ps_weight = 0.0

            rule40_config = self.dcf_config.get('rule_of_40', {})
            ps_config = rule40_config.get('ps_integration', {})
            dcf_weight_cfg = ps_config.get('weights', {}).get('dcf', 0.6)
            ps_weight_cfg = ps_config.get('weights', {}).get('ps', 0.4)

            # Decide whether we blend P/S with DCF
            if ps_valuation:
                dcf_weight = max(dcf_weight_cfg, 0.0)
                ps_weight = max(ps_weight_cfg, 0.0)
                if dcf_weight + ps_weight == 0:
                    dcf_weight, ps_weight = 1.0, 0.0  # fallback

            total_weight = dcf_weight + ps_weight
            dcf_weight_normalised = dcf_weight / total_weight
            ps_weight_normalised = ps_weight / total_weight if ps_valuation else 0.0

            final_fair_value = dcf_fair_value_per_share
            valuation_breakdown.append({
                'method': 'DCF',
                'value': round(dcf_fair_value_per_share, 2),
                'weight': round(dcf_weight_normalised, 3),
            })

            if ps_valuation:
                final_fair_value = (
                    dcf_fair_value_per_share * dcf_weight_normalised +
                    ps_valuation['fair_value_per_share'] * ps_weight_normalised
                )
                valuation_breakdown.append({
                    'method': 'P/S',
                    'value': round(ps_valuation['fair_value_per_share'], 2),
                    'weight': round(ps_weight_normalised, 3),
                    'details': {
                        'applied_ps_multiple': ps_valuation['applied_ps_multiple'],
                        'multiple_range': ps_valuation['multiple_range'],
                        'ttm_revenue_per_share': ps_valuation['ttm_revenue_per_share'],
                        'current_ps_multiple': ps_valuation['current_ps_multiple'],
                        'qualification': ps_valuation['qualification'],
                    },
                })

            upside_downside = ((final_fair_value / current_price) - 1) * 100 if current_price > 0 else 0

            logger.info(
                f"{self.symbol} - Fair Value: ${final_fair_value:.2f}, "
                f"Current: ${current_price:.2f}, Upside: {upside_downside:+.1f}%"
            )

            # Step 9: Sensitivity analysis - REMOVED
            # Both WACC and terminal growth are determined by formulas/config:
            # - WACC: Calculated from risk-free rate, beta, debt/equity (all observable)
            # - Terminal Growth: Determined by sector config + Rule of 40 adjustment
            # No need for arbitrary sensitivity variations
            sensitivity = None

            ps_payload = None
            if ps_valuation:
                ps_payload = {
                    'fair_value_per_share': round(ps_valuation['fair_value_per_share'], 2),
                    'applied_ps_multiple': round(ps_valuation['applied_ps_multiple'], 2),
                    'ttm_revenue_per_share': round(ps_valuation['ttm_revenue_per_share'], 2),
                    'ttm_revenue': round(ps_valuation['ttm_revenue'] / 1e9, 2),  # billions
                    'current_ps_multiple': round(ps_valuation['current_ps_multiple'], 2) if ps_valuation['current_ps_multiple'] is not None else None,
                    'multiple_range': ps_valuation['multiple_range'],
                    'qualification': ps_valuation['qualification'],
                }

            return {
                'fair_value_per_share': round(final_fair_value, 2),
                'current_price': round(current_price, 2),
                'upside_downside_pct': round(upside_downside, 1),
                'valuation_assessment': self._get_valuation_assessment(upside_downside),
                'rule_of_40': {
                    'score': round(rule_of_40_result['score'], 1),
                    'revenue_growth_pct': round(rule_of_40_result['revenue_growth_pct'], 1),
                    'profit_margin_pct': round(rule_of_40_result['profit_margin_pct'], 1),
                    'classification': rule_of_40_result['classification'],
                    'terminal_growth_adjustment': round(terminal_growth_adjustment * 100, 2)
                },
                'assumptions': {
                    'wacc': round(wacc * 100, 2),
                    'base_terminal_growth_rate': round(base_terminal_growth * 100, 2),
                    'terminal_growth_rate': round(terminal_growth_rate * 100, 2),
                    'projection_years': self.sector_params['projection_years'],
                    'sector': self.sector,
                    'sector_rationale': self.sector_params['rationale'],
                    'latest_fcf': round(latest_fcf / 1e6, 1),  # In millions
                    'dcf_mode': getattr(self, 'dcf_mode', 'UNKNOWN'),  # FADING_DCF or STANDARD_DCF
                },
                'valuation_breakdown': valuation_breakdown,
                'ps_valuation': ps_payload,
                'sensitivity_analysis': sensitivity,
                'sensitivity': sensitivity,
                'enterprise_value': round(enterprise_value / 1e9, 2),  # In billions
                'equity_value': round(equity_value / 1e9, 2)
            }

        except Exception as e:
            logger.error(f"Error calculating DCF for {self.symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _calculate_latest_fcf(self) -> float:
        """
        Calculate Free Cash Flow with MEDIAN SMOOTHING (Fix 5)

        Uses median of 12 quarters (3 years) to smooth:
        - Q4 seasonality in enterprise software (massive Q4 in SaaS/enterprise)
        - One-time items (large deals, legal settlements, acquisitions)
        - Quarterly lumpiness (especially for high-growth companies)

        Methodology:
        1. Get 12 quarters of data (3 years)
        2. Calculate quarterly FCF = OCF - CapEx for each quarter
        3. Use MEDIAN (not mean) to avoid outliers
        4. Annualize: smoothed_annual_fcf = median_quarterly_fcf * 4
        5. Fall back to TTM sum if insufficient data (<8 quarters)

        Why median instead of mean?
        - Robust to outliers (one lumpy quarter doesn't skew result)
        - Handles Q4 seasonality naturally (Q4 gets dampened, not amplified)
        - Better for businesses with irregular cash flow patterns

        Returns:
            Smoothed annualized free cash flow in dollars
        """
        # DEBUG: Log what data we received
        logger.info(f"🔍 {self.symbol} - DCF FCF CALCULATION START (with median smoothing)")
        logger.info(f"🔍 {self.symbol} - quarterly_metrics type: {type(self.quarterly_metrics)}")
        logger.info(f"🔍 {self.symbol} - quarterly_metrics length: {len(self.quarterly_metrics) if self.quarterly_metrics else 0}")

        if self.quarterly_metrics and len(self.quarterly_metrics) > 0:
            logger.info(f"🔍 {self.symbol} - Sample period keys: {list(self.quarterly_metrics[0].keys())[:10] if isinstance(self.quarterly_metrics[0], dict) else 'Not a dict'}")

        if not self.quarterly_metrics:
            logger.warning(f"🔍 {self.symbol} - quarterly_metrics is empty!")
            return 0

        # SEC FILING TOOL FORMAT DETECTION
        # SEC filing tool returns data in format: [{"fiscal_period": "FY", "income_statement": {...}, "cash_flow": {...}, ...}]
        # This is a single annual snapshot (FY), not multi-quarter data expected by get_rolling_ttm_periods
        first_item = self.quarterly_metrics[0]
        is_sec_format = (
            len(self.quarterly_metrics) == 1 and
            isinstance(first_item, dict) and
            ('income_statement' in first_item or 'cash_flow' in first_item) and
            first_item.get('fiscal_period') == 'FY'
        )

        if is_sec_format:
            logger.info(f"🔍 {self.symbol} - Detected SEC filing tool format (single FY snapshot)")

            # Extract FCF directly from SEC data
            cash_flow = first_item.get('cash_flow', {})
            fcf = cash_flow.get('free_cash_flow')

            if fcf is not None and fcf != 0:
                logger.info(
                    f"🔍 {self.symbol} - Using SEC direct FCF: ${fcf/1e9:.2f}B "
                    f"(OCF: ${cash_flow.get('operating_cash_flow', 0)/1e9:.2f}B, "
                    f"CapEx: ${cash_flow.get('capital_expenditures', 0)/1e9:.2f}B)"
                )
                return fcf

            # Calculate FCF if not directly available
            ocf = cash_flow.get('operating_cash_flow', 0) or 0
            capex = abs(cash_flow.get('capital_expenditures', 0) or 0)
            fcf = ocf - capex

            if fcf != 0:
                logger.info(
                    f"🔍 {self.symbol} - Calculated SEC FCF: ${fcf/1e9:.2f}B "
                    f"(OCF: ${ocf/1e9:.2f}B - CapEx: ${capex/1e9:.2f}B)"
                )
                return fcf

            logger.warning(f"🔍 {self.symbol} - SEC format detected but no FCF data available")
            return 0

        # DEBUG: Check what fiscal periods are in quarterly_metrics BEFORE passing to get_cached_ttm_periods
        fiscal_periods_before = [p.get('fiscal_period') for p in self.quarterly_metrics]
        logger.info(f"[DCF_INPUT_DEBUG] {self.symbol} - Fiscal periods in quarterly_metrics BEFORE get_cached_ttm_periods: {fiscal_periods_before}")
        fy_count = fiscal_periods_before.count('FY')
        q_count = sum(1 for fp in fiscal_periods_before if fp and str(fp).startswith('Q'))
        logger.info(f"[DCF_INPUT_DEBUG] {self.symbol} - Count: {fy_count} FY periods, {q_count} Q periods")

        # CRITICAL FIX #5: Try to get 12 quarters for median smoothing
        quarters_12 = self._get_cached_ttm_periods(num_quarters=12, compute_missing=True)

        if quarters_12 and len(quarters_12) >= 8:  # Need at least 2 years for smoothing
            # Extract quarterly FCF values
            fcf_quarters = []
            period_labels = []

            for period in quarters_12[:12]:  # Use up to 12 quarters
                cash_flow = period.get('cash_flow', {})

                # CRITICAL FIX: Only skip if YTD conversion FAILED
                # is_ytd=True just means original source was YTD (should be converted by quarterly_calculator)
                # ytd_conversion_failed=True means conversion failed and data is still YTD (must skip)
                if cash_flow.get('ytd_conversion_failed'):
                    fy = period.get('fiscal_year')
                    fp = period.get('fiscal_period')
                    logger.warning(
                        f"{self.symbol} - Skipping {fp}-{fy}: YTD conversion failed "
                        f"(likely missing Q1 for Q2/Q3 conversion)"
                    )
                    continue

                ocf = cash_flow.get('operating_cash_flow', 0) or 0
                capex = abs(cash_flow.get('capital_expenditures', 0) or 0)
                fcf_quarter = ocf - capex

                # Only include quarters with non-zero FCF (avoid polluting median with zero data gaps)
                if fcf_quarter != 0:
                    fcf_quarters.append(fcf_quarter)

                    # Track periods for logging
                    fy = period.get('fiscal_year')
                    fp = period.get('fiscal_period')
                    computed = period.get('computed', False)
                    ytd_converted = ' (YTD→Q)' if cash_flow.get('is_ytd') and not cash_flow.get('ytd_conversion_failed') else ''
                    period_labels.append(f"{fp}-{fy}{'*' if computed else ''}{ytd_converted}")

            if len(fcf_quarters) >= 8:
                # Use MEDIAN to smooth (robust to outliers)
                import numpy as np
                smoothed_quarterly_fcf = np.median(fcf_quarters)
                smoothed_annual_fcf = smoothed_quarterly_fcf * 4

                # Calculate TTM for comparison (traditional method)
                ttm_periods = quarters_12[:4]
                ttm_fcf = sum(
                    (period.get('cash_flow', {}).get('operating_cash_flow', 0) or 0) -
                    abs(period.get('cash_flow', {}).get('capital_expenditures', 0) or 0)
                    for period in ttm_periods
                    if not period.get('cash_flow', {}).get('ytd_conversion_failed')  # Only skip if conversion failed
                )

                logger.info(
                    f"🔍 [FCF_SMOOTHING] {self.symbol} - Using median of {len(fcf_quarters)} quarters: "
                    f"${smoothed_quarterly_fcf/1e9:.2f}B/qtr → ${smoothed_annual_fcf/1e9:.2f}B/yr annualized\n"
                    f"  Periods: {period_labels}\n"
                    f"  Min: ${min(fcf_quarters)/1e9:.2f}B/qtr, Max: ${max(fcf_quarters)/1e9:.2f}B/qtr, "
                    f"Median: ${smoothed_quarterly_fcf/1e9:.2f}B/qtr\n"
                    f"  TTM (traditional): ${ttm_fcf/1e9:.2f}B/yr\n"
                    f"  Smoothed vs TTM: {((smoothed_annual_fcf - ttm_fcf) / ttm_fcf * 100) if ttm_fcf != 0 else 0:.1f}% difference"
                )

                return smoothed_annual_fcf

        # FALLBACK: Use TTM (4 quarters sum) if insufficient data
        logger.warning(f"{self.symbol} - Insufficient data for median smoothing (need 8+ quarters), falling back to TTM")

        ttm_periods = self._get_cached_ttm_periods(num_quarters=4, compute_missing=True)

        logger.info(f"🔍 {self.symbol} - Got {len(ttm_periods) if ttm_periods else 0} TTM periods")

        if not ttm_periods:
            logger.warning(f"{self.symbol} - No quarterly data available for TTM FCF calculation")
            return 0

        # Sum FCF across TTM periods
        ttm_ocf = 0
        ttm_capex = 0
        period_labels = []

        for period in ttm_periods:
            # CLEAN ARCHITECTURE: Statement-level structure
            # Cash flow metrics live in cash_flow statement
            cash_flow = period.get('cash_flow', {})

            # CRITICAL FIX: Only skip if YTD conversion FAILED
            if cash_flow.get('ytd_conversion_failed'):
                fy = period.get('fiscal_year')
                fp = period.get('fiscal_period')
                logger.warning(
                    f"{self.symbol} - Skipping {fp}-{fy} in TTM FCF: YTD conversion failed. "
                    f"This may reduce accuracy of TTM calculation."
                )
                continue  # Skip this quarter, use remaining quarters

            ocf = cash_flow.get('operating_cash_flow', 0) or 0
            capex = abs(cash_flow.get('capital_expenditures', 0) or 0)

            ttm_ocf += ocf
            ttm_capex += capex

            # Track periods for logging
            fy = period.get('fiscal_year')
            fp = period.get('fiscal_period')
            computed = period.get('computed', False)
            period_labels.append(f"{fp}-{fy}{'*' if computed else ''}")

            # DEBUG: Show extraction results for each period
            logger.info(f"🔍 {self.symbol} - {fp}-{fy}: OCF=${ocf/1e6:.1f}M, CapEx=${capex/1e6:.1f}M")

        fcf = ttm_ocf - ttm_capex

        logger.info(
            f"{self.symbol} - TTM FCF (fallback) from {len(ttm_periods)} quarters {period_labels}: "
            f"OCF=${ttm_ocf/1e6:.1f}M, CapEx=${ttm_capex/1e6:.1f}M, FCF=${fcf/1e6:.1f}M "
            f"(* = computed from FY)"
        )

        return fcf

    def _project_fcf(self, latest_fcf: float, years: int = 10) -> List[float]:
        """
        Project FCF for next N years based on historical growth

        Uses conservative tapering: starts at historical growth, tapers to terminal rate

        Args:
            latest_fcf: Most recent FCF
            years: Number of years to project

        Returns:
            List of projected FCF values
        """
        # Debug: Show data availability
        logger.info(
            f"🔍 [FCF_GROWTH] {self.symbol} - Data check: "
            f"multi_year_data={len(self.multi_year_data)} years (need >=3 OR >=8 quarters), "
            f"quarterly_metrics={len(self.quarterly_metrics) if self.quarterly_metrics else 0} quarters"
        )

        # Calculate historical FCF growth rate
        # CRITICAL: Check pre-profitable status FIRST before deciding on data sufficiency
        num_quarters = len(self.quarterly_metrics) if self.quarterly_metrics else 0
        ttm_metrics = self._get_ttm_metrics()

        # Check if we should use industry defaults for pre-profitable companies
        use_industry_defaults = should_use_industry_defaults(
            num_quarters=num_quarters,
            net_income=ttm_metrics['ttm_net_income'],
            ebitda=ttm_metrics['ttm_ebitda'],
            sector=ttm_metrics['sector'],
            industry=ttm_metrics['industry']
        )

        if use_industry_defaults:
            # PRE-PROFITABLE: Use industry-specific growth assumptions
            assumptions = get_growth_assumptions(ttm_metrics['sector'], ttm_metrics['industry'])
            historical_growth = assumptions['default_revenue_growth']

            logger.info(format_assumptions_log(assumptions, ttm_metrics['sector'], ttm_metrics['industry']))
            logger.info(
                f"🔍 [FCF_GROWTH] {self.symbol} - PRE-PROFITABLE: Using industry default growth: "
                f"{historical_growth*100:.1f}% (insufficient historical data: {num_quarters} quarters, "
                f"Net Income: ${ttm_metrics['ttm_net_income']:,.0f})"
            )
        else:
            # PROFITABLE or sufficient data: Use historical growth if available
            has_sufficient_annual_data = len(self.multi_year_data) >= 3
            has_sufficient_quarterly_data = self.quarterly_metrics and len(self.quarterly_metrics) >= 8

            if has_sufficient_annual_data or has_sufficient_quarterly_data:
                if has_sufficient_annual_data:
                    logger.info(f"🔍 [FCF_GROWTH] {self.symbol} - Using annual data ({len(self.multi_year_data)} years)")
                else:
                    logger.info(f"🔍 [FCF_GROWTH] {self.symbol} - Using quarterly data ({len(self.quarterly_metrics)} quarters)")
                historical_growth = self._calculate_historical_fcf_growth()
            else:
                # Profitable company with insufficient data: use conservative 5% default
                historical_growth = 0.05
                logger.info(
                    f"🔍 [FCF_GROWTH] {self.symbol} - Insufficient data: "
                    f"multi_year_data={len(self.multi_year_data)} < 3 years, "
                    f"quarterly_metrics={num_quarters} < 8 quarters. "
                    f"Using fallback growth: {historical_growth*100:.1f}%"
                )

        # REMOVED: Blanket 25% cap that was overriding _calculate_historical_fcf_growth()'s intelligent capping
        # _calculate_historical_fcf_growth() already applies:
        # - No cap for geometric mean (12-quarter, already stable)
        # - Sector-specific caps for simple TTM growth
        # - Only sanity check for negative growth
        # Floor at -5% for negative growth rates (prevents extreme negative projections)
        historical_growth = max(historical_growth, -0.05)   # Floor at -5%

        logger.info(f"{self.symbol} - Historical FCF growth rate (final): {historical_growth*100:.1f}%")

        # Get sector-specific parameters for growth constraints
        sector_caps = self._get_sector_growth_caps()
        sector_params = self._get_sector_parameters()

        # Get terminal growth rate (for perpetuity after projection period)
        base_terminal_growth = sector_params.get('terminal_growth_rate', 0.03)

        # CRITICAL FIX: Fading growth should fade TOWARD terminal, not toward industry median
        # Old logic: Fade FROM company_growth TO industry_median (backwards - caused inflation)
        # New logic: Fade FROM company_growth TOWARD terminal (realistic deceleration)
        #
        # Fade target = terminal growth * multiplier (typically 1.5-2.5x terminal)
        # Example: If terminal = 3%, fade_target = 4.5-7.5% (not 20-25%!)
        #
        # This multiplier accounts for:
        # - Final projection year should be closer to terminal than to peak growth
        # - Allows gradual deceleration rather than cliff drop
        # - More realistic than assuming acceleration to industry median

        fade_target_multiplier = 2.0  # Default: 2x terminal growth for Year 5

        # Adjust multiplier based on company quality (Rule of 40)
        rule_of_40_score = self.rule_of_40_result.get('score', 0)
        if rule_of_40_score > 40:
            # Exceptional quality: Can sustain slightly higher growth
            fade_target_multiplier = 2.5
        elif rule_of_40_score < 20:
            # Poor quality: Faster deceleration toward terminal
            # Changed from 1.5 to 2.0 to be less harsh on companies with real track records
            fade_target_multiplier = 2.0

        # Calculate base fade target (what Year 5 growth should approach)
        base_fade_target = base_terminal_growth * fade_target_multiplier

        # CRITICAL: Apply historical floor to prevent over-penalizing companies with proven track records
        # If a company has sustained 11.9% growth, we shouldn't fade to 4.5%, should respect that performance
        # Floor = 60% of historical growth (company has EARNED the right to higher projections)
        historical_floor = abs(historical_growth) * 0.60
        fade_target_growth = max(base_fade_target, historical_floor)

        # Cap fade target based on company maturity (market cap based constraints)
        market_cap = self._get_market_cap()
        if market_cap > 100e9:  # Mega-cap (>$100B)
            # Large mature companies cannot sustain high growth
            max_fade_target = 0.10  # Cap at 10%
            fade_target_growth = min(fade_target_growth, max_fade_target)
        elif market_cap > 10e9:  # Large-cap ($10B-$100B)
            max_fade_target = 0.15  # Cap at 15%
            fade_target_growth = min(fade_target_growth, max_fade_target)
        # Mid/small cap: No additional cap (fade_target_multiplier handles it)

        # Also cap fade target at historical growth (can't fade UP)
        # If company growing at 8%, Year 5 shouldn't exceed 8%
        fade_target_growth = min(fade_target_growth, abs(historical_growth) * 1.1)  # Allow 10% overshoot

        # Get industry_median_growth for reference (ceiling check only, not a target)
        industry_median_growth = sector_caps.get('industry_median_growth', sector_caps.get('max_growth', 0.15))

        logger.info(
            f"{self.symbol} - Fading Growth Parameters:\n"
            f"  Terminal Growth (perpetuity): {base_terminal_growth*100:.1f}%\n"
            f"  Base Fade Target ({fade_target_multiplier:.1f}x terminal): {base_fade_target*100:.1f}%\n"
            f"  Historical Floor (60% of historical): {historical_floor*100:.1f}%\n"
            f"  Final Fade Target (Year {sector_params.get('projection_years', 5)}): {fade_target_growth*100:.1f}% "
            f"(max of base target and historical floor, before maturity caps)\n"
            f"  Historical FCF Growth: {historical_growth*100:.1f}%\n"
            f"  Industry Median (ceiling only): {industry_median_growth*100:.1f}%\n"
            f"  Market Cap: ${market_cap/1e9:.1f}B\n"
            f"  Rule of 40: {rule_of_40_score:.1f}"
        )

        # CRITICAL DECISION: Choose DCF mode based on multi-factor underperformer detection
        # This prevents rewarding truly struggling companies while NOT penalizing strategic investors
        #
        # Mode 1: FADING DCF (Optimistic) - Default for most companies
        #   - Year 1: Company-specific historical growth
        #   - Final year: Industry median growth
        #   - Linear interpolation between
        #   - Rationale: Most companies deserve benefit of doubt, especially if investing in growth
        #
        # Mode 2: STANDARD DCF (Conservative) - Only for TRUE underperformers
        #   - All years: Company-specific historical growth (constant)
        #   - No fading to industry median
        #   - Rationale: Don't reward companies showing STRUCTURAL DECLINE
        #
        # Multi-Factor Detection: A company is a TRUE underperformer if:
        #   1. Declining revenue (negative revenue growth) - CRITICAL SIGNAL
        #   OR
        #   2. ALL of the following (combination indicates structural problems):
        #      - Revenue growth significantly below industry median (<50% of median)
        #      - Negative FCF growth (not just due to strategic CapEx)
        #      - Weak Rule of 40 score (<20, indicating poor efficiency)
        #
        # Strategic investors (high CapEx, growing revenue) are NOT penalized

        # Get revenue growth from Rule of 40 calculation (already computed)
        revenue_growth = self.rule_of_40_result.get('revenue_growth_pct', 0) / 100  # Convert to decimal
        rule_of_40_score = self.rule_of_40_result.get('score', 0)

        # CAPEX INTENSITY & TURNAROUND DETECTION
        # Detect high-CAPEX growth investment OR turnaround companies
        # where FCF is distorted but business fundamentals are improving
        capex_intensity = 0.0
        is_high_capex_phase = False
        is_turnaround = False
        sequential_revenue_growth = 0.0
        earnings_momentum = 0.0
        growth_rate_to_use = historical_growth  # Default to FCF growth

        if self.quarterly_metrics and len(self.quarterly_metrics) >= 4:
            try:
                # Calculate TTM CAPEX intensity from last 4 quarters
                recent_quarters = self.quarterly_metrics[-4:]
                ttm_opcf = sum(q.get('cash_flow', {}).get('operating_cash_flow', 0) for q in recent_quarters)
                ttm_capex = sum(q.get('cash_flow', {}).get('capital_expenditures', 0) for q in recent_quarters)

                if ttm_opcf > 0:
                    capex_intensity = abs(ttm_capex) / ttm_opcf  # CapEx is usually negative, take abs

                # DETECTION #1: High-CAPEX growth investment (lowered threshold to 40%)
                # CAPEX >40% of OpCF AND revenue growing >10%
                is_high_capex_phase = (capex_intensity > 0.40) and (revenue_growth > 0.10)

                # DETECTION #2: Turnaround / Sequential momentum (for negative TTM YoY)
                # Even if TTM YoY is negative, check if recent quarters show strong acceleration
                if revenue_growth < 0 and len(self.quarterly_metrics) >= 3:
                    # Calculate sequential (QoQ) growth for last 2 quarters
                    q1_revenue = recent_quarters[-1].get('income_statement', {}).get('total_revenue', 0)
                    q2_revenue = recent_quarters[-2].get('income_statement', {}).get('total_revenue', 0)
                    q3_revenue = recent_quarters[-3].get('income_statement', {}).get('total_revenue', 0)

                    if q2_revenue > 0 and q3_revenue > 0:
                        seq_growth_q1 = ((q1_revenue - q2_revenue) / q2_revenue) if q2_revenue > 0 else 0
                        seq_growth_q2 = ((q2_revenue - q3_revenue) / q3_revenue) if q3_revenue > 0 else 0
                        sequential_revenue_growth = (seq_growth_q1 + seq_growth_q2) / 2  # 2-quarter average

                        # Turnaround = avg sequential growth >15% despite negative TTM YoY
                        is_turnaround = sequential_revenue_growth > 0.15

                # DETECTION #3: Earnings momentum (profitability surge)
                # If net income surging despite revenue challenges, likely recovering from trough
                if len(self.quarterly_metrics) >= 2:
                    latest_ni = recent_quarters[-1].get('income_statement', {}).get('net_income', 0)
                    prior_ni = recent_quarters[-2].get('income_statement', {}).get('net_income', 0)

                    # Calculate earnings momentum (avoid div by zero or negative base)
                    if prior_ni > 0:
                        earnings_momentum = (latest_ni - prior_ni) / abs(prior_ni)
                    elif prior_ni < 0 and latest_ni > 0:
                        # Turnaround from loss to profit = 100% momentum
                        earnings_momentum = 1.0

                # COMBINED LOGIC: Use revenue growth if ANY detection triggers
                if is_high_capex_phase or is_turnaround or earnings_momentum > 0.50:
                    # Use revenue growth instead of FCF growth for projection
                    growth_rate_to_use = revenue_growth if revenue_growth > 0 else 0.05  # Minimum 5% for turnarounds

                    detection_reason = []
                    if is_high_capex_phase:
                        detection_reason.append(f"High CAPEX {capex_intensity*100:.1f}% + Revenue {revenue_growth*100:+.1f}%")
                    if is_turnaround:
                        detection_reason.append(f"Turnaround (Sequential {sequential_revenue_growth*100:+.1f}%)")
                    if earnings_momentum > 0.50:
                        detection_reason.append(f"Earnings Surge {earnings_momentum*100:+.1f}%")

                    logger.info(
                        f"🏗️ [GROWTH_INVESTMENT] {self.symbol} - Strategic growth/turnaround detected\n"
                        f"  Detection: {' | '.join(detection_reason)}\n"
                        f"  CAPEX Intensity: {capex_intensity*100:.1f}%\n"
                        f"  TTM Revenue Growth: {revenue_growth*100:+.1f}%\n"
                        f"  Sequential Revenue Growth (2Q avg): {sequential_revenue_growth*100:+.1f}%\n"
                        f"  Earnings Momentum (QoQ): {earnings_momentum*100:+.1f}%\n"
                        f"  FCF Growth: {historical_growth*100:+.1f}% (distorted)\n"
                        f"  🔧 FALLBACK: Using revenue growth ({growth_rate_to_use*100:.1f}%) for projections"
                    )
            except (KeyError, TypeError, ZeroDivisionError) as e:
                logger.debug(f"{self.symbol} - Could not calculate CAPEX intensity: {e}")

        # Detect TRUE underperformers (structural decline, not strategic investment)
        is_declining_revenue = revenue_growth < 0
        is_significantly_lagging = revenue_growth < (industry_median_growth * 0.5)  # <50% of industry median
        is_poor_fcf_growth = growth_rate_to_use < 0  # Use adjusted growth rate
        is_weak_efficiency = rule_of_40_score < 20  # Poor combined growth + profitability

        # TRUE underperformer = declining revenue OR (lagging peers AND poor FCF AND weak efficiency)
        # BUT NOT if ANY of: high-CAPEX growth, turnaround, or strong earnings momentum
        is_true_underperformer = (
            (is_declining_revenue or
             (is_significantly_lagging and is_poor_fcf_growth and is_weak_efficiency))
            and not (is_high_capex_phase or is_turnaround or earnings_momentum > 0.50)
        )

        # DEFAULT to Fading DCF UNLESS company shows clear structural decline
        use_fading_dcf = not is_true_underperformer
        dcf_mode = "FADING_DCF" if use_fading_dcf else "STANDARD_DCF"

        # Store mode and diagnostic info for inclusion in results
        self.dcf_mode = dcf_mode
        self.dcf_mode_diagnostics = {
            'revenue_growth_pct': round(revenue_growth * 100, 1),
            'fcf_growth_pct': round(historical_growth * 100, 1),
            'rule_of_40_score': round(rule_of_40_score, 1),
            'is_declining_revenue': is_declining_revenue,
            'is_significantly_lagging': is_significantly_lagging,
            'is_poor_fcf_growth': is_poor_fcf_growth,
            'is_weak_efficiency': is_weak_efficiency,
            'is_true_underperformer': is_true_underperformer
        }

        projections = []

        if use_fading_dcf:
            # Strategic investor or healthy company: Use Fading DCF (default, optimistic)
            # CRITICAL FIX: Now fades TOWARD terminal (via fade_target), not toward industry median
            logger.info(
                f"📊 [FADING DCF] {self.symbol} - Strategic investor / Healthy company\n"
                f"  Mode: Fading DCF (realistic deceleration)\n"
                f"  Revenue Growth: {revenue_growth*100:+.1f}% | FCF Growth: {historical_growth*100:+.1f}% | Rule of 40: {rule_of_40_score:.1f}\n"
                f"  Year 1: {growth_rate_to_use*100:.1f}% (company-specific, adjusted for turnarounds)\n"
                f"  Year {years}: {fade_target_growth*100:.1f}% (fading toward terminal, quality & maturity adjusted)\n"
                f"  Terminal growth (perpetuity): {base_terminal_growth*100:.1f}% (sector-specific + Rule of 40 adjustment)\n"
                f"  Method: Linear interpolation from company-specific TOWARD terminal over {years} years\n"
                f"  Maturity Cap: ${market_cap/1e9:.1f}B market cap → max Year {years} growth = {fade_target_growth*100:.1f}%\n"
                f"  Rationale: Realistic deceleration toward sustainable terminal growth rate"
            )
        else:
            # TRUE underperformer (structural decline): Use Standard DCF (conservative)
            declining_reason = "❌ Declining revenue" if is_declining_revenue else \
                              f"❌ Lagging peers (revenue growth {revenue_growth*100:.1f}% < {industry_median_growth*0.5*100:.1f}%) + Poor FCF + Weak efficiency"
            logger.info(
                f"📊 [STANDARD DCF] {self.symbol} - TRUE underperformer (structural decline detected)\n"
                f"  Mode: Standard DCF (conservative)\n"
                f"  Revenue Growth: {revenue_growth*100:+.1f}% | FCF Growth: {historical_growth*100:+.1f}% | Rule of 40: {rule_of_40_score:.1f}\n"
                f"  Reason: {declining_reason}\n"
                f"  Growth rate: {historical_growth*100:.1f}% (constant across all projection years)\n"
                f"  Terminal growth (perpetuity): {base_terminal_growth*100:.1f}% (sector-specific + Rule of 40 adjustment)\n"
                f"  Rationale: Structural decline (not strategic investment) - no reward for poor performance"
            )

        for year in range(1, years + 1):
            if use_fading_dcf:
                # Fading DCF: Linear interpolation from company-specific TOWARD terminal
                # CRITICAL FIX: Now fades toward fade_target_growth (derived from terminal),
                # not toward industry_median_growth (which was causing backwards acceleration)
                #
                # Formula: growth(t) = company_growth × (1 - t/n) + fade_target × (t/n)
                # where t = current year, n = total projection years
                # Use growth_rate_to_use (adjusted for turnarounds) instead of historical_growth
                fade_weight = (year - 1) / (years - 1) if years > 1 else 0  # 0 in Year 1 → 1 in final year
                growth_rate = (growth_rate_to_use * (1 - fade_weight)) + (fade_target_growth * fade_weight)
            else:
                # Standard DCF: Constant historical growth rate (conservative)
                growth_rate = historical_growth

            # Compound from previous year's FCF (not power of base FCF)
            if year == 1:
                fcf = latest_fcf * (1 + growth_rate)
            else:
                fcf = projections[-1] * (1 + growth_rate)
            projections.append(fcf)

            logger.info(
                "%s - Projected FCF Year %d: $%.2fB (growth %.2f%%)",
                self.symbol,
                year,
                fcf / 1e9,
                growth_rate * 100,
            )

        # CRITICAL FIX #4: Terminal FCF Margin Validation (CONFIG-DRIVEN)
        # Ensure final year FCF margin is realistic based on sector, size, and stage
        try:
            # Get latest TTM revenue, market cap, and metrics for classification
            ttm_periods = self._get_cached_ttm_periods(num_quarters=4, compute_missing=True)
            if ttm_periods and len(projections) > 0:
                ttm_revenue = sum(
                    period.get('income_statement', {}).get('total_revenue', 0) or 0
                    for period in ttm_periods[:4]
                )

                if ttm_revenue > 0:
                    # Get market cap and current FCF margin
                    market_cap = self._get_market_cap()
                    current_fcf_margin = latest_fcf / ttm_revenue if ttm_revenue > 0 else 0

                    # Get Rule of 40 and revenue growth for stage classification
                    rule_of_40_result = self._calculate_rule_of_40()
                    rule_of_40_score = rule_of_40_result.get('score', 0)

                    # Calculate revenue growth using geometric mean over 12 quarters (3 years)
                    # Geometric mean is more robust than YoY and smooths volatility
                    if len(ttm_periods) >= 12:  # Need 3 years for geometric mean
                        # Get TTM revenue for last 4 quarters (current)
                        current_ttm = ttm_revenue

                        # Get TTM revenue for quarters 5-8 (1 year ago)
                        ttm_1y_ago = sum(
                            period.get('income_statement', {}).get('total_revenue', 0) or 0
                            for period in ttm_periods[4:8]
                        )

                        # Get TTM revenue for quarters 9-12 (2 years ago)
                        ttm_2y_ago = sum(
                            period.get('income_statement', {}).get('total_revenue', 0) or 0
                            for period in ttm_periods[8:12]
                        )

                        # Calculate geometric mean: (current / 2y_ago) ^ (1/2) - 1
                        # This gives CAGR over 2 years using 3 data points (current, 1y ago, 2y ago)
                        if ttm_2y_ago > 0:
                            ratio = current_ttm / ttm_2y_ago
                            # Protect against negative ratio (would produce complex number)
                            if ratio <= 0:
                                logger.warning(f"⚠️ {self.symbol} - Cannot compute geometric mean: negative ratio ({ratio:.3f}). Using 0.")
                                revenue_growth = 0.0
                            else:
                                revenue_growth = ratio ** (1.0 / 2.0) - 1.0

                            # Validate against 1-year YoY for sanity check
                            yoy_growth = ((current_ttm - ttm_1y_ago) / ttm_1y_ago if ttm_1y_ago > 0 else 0)

                            logger.info(
                                f"🔍 [REVENUE_GROWTH] {self.symbol} - Calculated over 12 quarters\n"
                                f"  Current TTM: ${current_ttm/1e9:.2f}B\n"
                                f"  1Y ago TTM: ${ttm_1y_ago/1e9:.2f}B (YoY: {yoy_growth*100:+.1f}%)\n"
                                f"  2Y ago TTM: ${ttm_2y_ago/1e9:.2f}B\n"
                                f"  Geometric mean (2Y CAGR): {revenue_growth*100:+.1f}%"
                            )
                        else:
                            revenue_growth = 0.05  # Default if 2Y ago is zero
                    elif len(ttm_periods) >= 8:  # Fallback to 2-year YoY if <12 quarters
                        current_ttm_revenue = ttm_revenue
                        prior_ttm_revenue = sum(
                            period.get('income_statement', {}).get('total_revenue', 0) or 0
                            for period in ttm_periods[4:8]
                        )
                        revenue_growth = ((current_ttm_revenue - prior_ttm_revenue) / prior_ttm_revenue
                                         if prior_ttm_revenue > 0 else 0)
                        logger.info(f"🔍 [REVENUE_GROWTH] {self.symbol} - Using 2Y YoY (insufficient data for geometric mean): {revenue_growth*100:+.1f}%")
                    else:
                        revenue_growth = 0.05  # Default 5% if insufficient data
                        logger.warning(f"🔍 [REVENUE_GROWTH] {self.symbol} - Using default 5% (insufficient data, need 8+ quarters)")

                    # If revenue growth is negative, use default positive growth for terminal margin classification
                    # Negative growth companies get classified conservatively anyway
                    if revenue_growth < 0:
                        logger.info(
                            f"🔍 [REVENUE_GROWTH] {self.symbol} - Negative growth ({revenue_growth*100:.1f}%), "
                            f"using default 5% for terminal margin classification"
                        )
                        revenue_growth = 0.05  # Use positive default for stage classification

                    # Classify company stage and size
                    if market_cap and market_cap > 0:
                        size_tier = self._classify_company_size(market_cap)
                        stage = self._classify_company_stage(rule_of_40_score, current_fcf_margin, revenue_growth)

                        # Get terminal margin from config (granular by sector/size/stage)
                        max_terminal_margin = self._get_terminal_margin_from_config(
                            sector=self.sector,
                            size_tier=size_tier,
                            stage=stage,
                            current_fcf_margin=current_fcf_margin
                        )
                    else:
                        # Fallback if no market cap available
                        max_terminal_margin = 0.20  # Conservative default
                        logger.warning(
                            f"{self.symbol} - Market cap not available for terminal margin classification, "
                            f"using conservative default {max_terminal_margin*100:.1f}%"
                        )

                    # Project terminal revenue
                    terminal_revenue = ttm_revenue * ((1 + growth_rate_to_use) ** years)

                    # Calculate implied terminal FCF margin
                    final_fcf = projections[-1]
                    implied_margin = final_fcf / terminal_revenue if terminal_revenue > 0 else 0

                    # Cap final year FCF if margin exceeds realistic threshold
                    if implied_margin > max_terminal_margin:
                        adjusted_final_fcf = terminal_revenue * max_terminal_margin
                        logger.info(
                            f"🔍 [TERMINAL_MARGIN] {self.symbol} - Capping terminal FCF margin\n"
                            f"  Implied terminal margin: {implied_margin*100:.1f}% (too high)\n"
                            f"  Max realistic margin: {max_terminal_margin*100:.1f}% ({size_tier}/{stage})\n"
                            f"  Terminal revenue projection: ${terminal_revenue/1e9:.2f}B\n"
                            f"  Final year FCF: ${final_fcf/1e9:.2f}B → ${adjusted_final_fcf/1e9:.2f}B\n"
                            f"  Impact: {((adjusted_final_fcf - final_fcf) / final_fcf * 100):.1f}% reduction"
                        )
                        projections[-1] = adjusted_final_fcf
                    else:
                        logger.info(
                            f"🔍 [TERMINAL_MARGIN] {self.symbol} - Terminal FCF margin check passed\n"
                            f"  Implied terminal margin: {implied_margin*100:.1f}%\n"
                            f"  Max realistic margin: {max_terminal_margin*100:.1f}% ({size_tier}/{stage})\n"
                            f"  Terminal revenue projection: ${terminal_revenue/1e9:.2f}B\n"
                            f"  Final year FCF: ${final_fcf/1e9:.2f}B (no adjustment needed)"
                        )
        except Exception as e:
            logger.debug(f"{self.symbol} - Could not validate terminal FCF margin: {e}")

        return projections

    def _calculate_historical_fcf_growth(self) -> float:
        """
        Calculate FCF growth rate based on revenue growth + sector premium.

        Formula: FCF_GROWTH_RATE = TTM_REVENUE_GEOMETRIC_GROWTH + SECTOR_PREMIUM

        Returns:
            Annual growth rate as decimal (e.g., 0.316 for 31.6%)
        """
        # Get TTM revenue growth (returns percentage, e.g., 28.6 for 28.6%)
        revenue_growth_pct = self._get_ttm_revenue_growth()
        revenue_growth = revenue_growth_pct / 100.0  # Convert to decimal

        # Get validation bounds from config
        validation = self.dcf_config.get('fcf_growth_validation', {})
        revenue_min = validation.get('revenue_growth_min', -0.05)
        revenue_max = validation.get('revenue_growth_max', 0.50)
        fcf_min = validation.get('fcf_growth_min', -0.05)
        fcf_max = validation.get('fcf_growth_max', 0.35)

        # Clamp revenue growth
        revenue_growth_clamped = max(min(revenue_growth, revenue_max), revenue_min)

        # Get sector premium from config
        sector_premium_map = self.dcf_config.get('fcf_sector_premium', {})
        default_premium = sector_premium_map.get('default', 0.02)
        sector_lower = self.sector.lower().replace(" ", "_")
        sector_premium = sector_premium_map.get(sector_lower, default_premium)

        # Calculate FCF growth
        fcf_growth = revenue_growth_clamped + sector_premium

        # Clamp FCF growth
        fcf_growth_clamped = max(min(fcf_growth, fcf_max), fcf_min)

        # Log with required format
        logger.info(
            f"[FCF_GROWTH_FINAL] {self.symbol}:\n"
            f"  TTM Revenue Geometric Growth = {revenue_growth*100:.1f}%"
            + (f" (clamped from {revenue_growth*100:.1f}% to {revenue_growth_clamped*100:.1f}%)" if revenue_growth != revenue_growth_clamped else "") +
            f"\n  Sector Premium ({self.sector}) = {sector_premium*100:.1f}%\n"
            f"  → Year 1 FCF Growth = {fcf_growth*100:.1f}%"
            + (f" (clamped to {fcf_growth_clamped*100:.1f}%)" if fcf_growth != fcf_growth_clamped else "")
        )

        return fcf_growth_clamped

    def _calculate_terminal_value(self, final_year_fcf: float,
                                  terminal_growth_rate: float = 0.03) -> float:
        """
        Calculate terminal value using perpetuity growth model

        Terminal Value = FCF(n+1) / (WACC - g)
        where FCF(n+1) = Final year FCF * (1 + g)

        Args:
            final_year_fcf: FCF in final projection year
            terminal_growth_rate: Perpetual growth rate (default 3%)

        Returns:
            Terminal value in dollars
        """
        wacc = self._calculate_wacc()

        # Terminal value formula
        terminal_value = (final_year_fcf * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)

        return terminal_value

    def _calculate_wacc(self) -> float:
        """
        Calculate Weighted Average Cost of Capital

        WACC = (E/V * Re) + (D/V * Rd * (1-T))

        Where:
        - E = Market value of equity
        - D = Market value of debt
        - V = E + D
        - Re = Cost of equity (CAPM)
        - Rd = Cost of debt
        - T = Tax rate

        Returns:
            WACC as decimal
        """
        if not self.quarterly_metrics:
            return 0.10  # Default 10%

        latest = self.quarterly_metrics[-1]

        # Get market cap (equity value)
        current_price = self._get_current_price()
        shares = self._get_shares_outstanding()
        market_cap = current_price * shares

        # Get total debt from balance_sheet structure
        # CRITICAL FIX: Balance sheet items (debt, equity) should prefer FY data over quarterly
        # Reason: Quarterly 10-Q often omits long_term_debt (only shows short_term_debt)
        # Example: ORCL Q1 2026 shows $9B debt (only short_term), but FY 2025 shows $92.5B (complete)
        balance_sheet = latest.get('balance_sheet', {})
        total_debt = balance_sheet.get('total_debt', 0) or 0

        # Extract debt components (needed for FY fallback logic)
        long_term_debt = balance_sheet.get('long_term_debt', 0) or 0
        short_term_debt = balance_sheet.get('short_term_debt', 0) or balance_sheet.get('debt_current', 0) or 0

        # Fallback: Calculate from components if total_debt not available
        if not total_debt:
            total_debt = long_term_debt + short_term_debt

        # CRITICAL FIX: If quarterly data has incomplete debt (missing long_term_debt),
        # use most recent FY data for structural balance sheet items
        if not long_term_debt:
            # Find most recent FY period
            fy_period = None
            for period in reversed(self.quarterly_metrics):
                if period.get('fiscal_period') == 'FY':
                    fy_period = period
                    break

            if fy_period:
                fy_balance_sheet = fy_period.get('balance_sheet', {})
                fy_long_term_debt = fy_balance_sheet.get('long_term_debt', 0) or 0

                if fy_long_term_debt > 0:
                    # Use FY long_term_debt + Q short_term_debt (short-term can change quarterly)
                    total_debt = fy_long_term_debt + short_term_debt
                    logger.info(
                        f"⚠️  {self.symbol} - Q period missing long_term_debt. "
                        f"Using FY long_term_debt (${fy_long_term_debt/1e9:.2f}B) + "
                        f"Q short_term_debt (${short_term_debt/1e9:.2f}B) = "
                        f"Total debt ${total_debt/1e9:.2f}B"
                    )

        # Total value
        total_value = market_cap + total_debt

        # Weights
        weight_equity = market_cap / total_value if total_value > 0 else 0.8
        weight_debt = total_debt / total_value if total_value > 0 else 0.2

        # Cost of equity (using CAPM): Re = Rf + Beta * (Rm - Rf)
        # Get unlevered beta from market data, then lever it with actual D/E ratio
        beta_unlevered = self._get_unlevered_beta()
        stockholders_equity = balance_sheet.get('stockholders_equity', 0) or 0
        beta = self._calculate_levered_beta(beta_unlevered, total_debt, stockholders_equity, market_cap)
        risk_free_rate = self._get_risk_free_rate()  # 10-year Treasury yield

        # Dynamic Equity Risk Premium based on CFA Institute guidance
        # Per CFA guidance, ERP varies with risk-free rate environment:
        # - Rf < 3%: ERP = 5.5% (midpoint of 5.0-6.0%)
        # - 3% ≤ Rf < 5%: ERP = 6.0% (midpoint of 5.5-6.5%)
        # - Rf ≥ 5%: ERP = 6.5% (midpoint of 6.0-7.0%)
        if risk_free_rate < 0.03:
            market_risk_premium = 0.055  # Low rate environment: 5.5%
            erp_rationale = "Low Rf < 3%: ERP=5.5%"
        elif risk_free_rate < 0.05:
            market_risk_premium = 0.060  # Moderate rate environment: 6.0%
            erp_rationale = "Moderate 3% ≤ Rf < 5%: ERP=6.0%"
        else:
            market_risk_premium = 0.065  # High rate environment: 6.5%
            erp_rationale = "High Rf ≥ 5%: ERP=6.5%"

        cost_of_equity = risk_free_rate + beta * market_risk_premium

        # Cost of debt
        # CRITICAL FIX: Get interest_expense from BOTH nested and flat structures (backward compatible)
        # Try nested structure first (income_statement subdictionary)
        income_statement = latest.get('income_statement', {})
        interest_expense_nested = income_statement.get('interest_expense', 0) if income_statement else 0
        # Fallback to flat structure (sec_companyfacts_processed table)
        interest_expense_flat = latest.get('interest_expense', 0)
        # Use whichever is non-zero (prefer nested if both exist)
        interest_expense = abs(interest_expense_nested or interest_expense_flat or 0)

        # Market-implied default cost of debt (conservative estimate)
        default_cost_of_debt = self.wacc_config.get('default_cost_of_debt', 0.055)
        min_cost_of_debt = 0.02  # Minimum realistic cost of debt (2%)

        # Calculate cost of debt with fallback logic
        if total_debt > 0 and interest_expense > 0:
            # Direct calculation from reported interest
            calculated_cost = interest_expense / total_debt
            if calculated_cost < min_cost_of_debt:
                # Unrealistically low - use default
                cost_of_debt = default_cost_of_debt
                cost_of_debt_source = f"default (calculated {calculated_cost:.2%} < {min_cost_of_debt:.0%} floor)"
            else:
                cost_of_debt = min(calculated_cost, 0.10)  # Cap at 10%
                cost_of_debt_source = "calculated from interest expense"
        elif total_debt > 0:
            # Has debt but no interest expense reported - use market default
            cost_of_debt = default_cost_of_debt
            cost_of_debt_source = "market-implied default (interest expense not reported)"
        else:
            # No debt
            cost_of_debt = default_cost_of_debt
            cost_of_debt_source = "default (no debt)"

        # 🔍 DEBUG TRACE: Log interest expense retrieval with source explanation
        logger.info(
            f"🔍 [DCF_INTEREST] {self.symbol} - Interest expense: ${interest_expense/1e9:.2f}B "
            f"(nested: ${interest_expense_nested/1e9:.2f}B, flat: ${interest_expense_flat/1e9:.2f}B), "
            f"Total debt: ${total_debt/1e9:.2f}B, Cost of debt: {cost_of_debt:.2%} ({cost_of_debt_source})"
        )

        # Tax rate
        tax_rate = 0.21  # Federal corporate tax rate

        logger.info(
            "%s - WACC inputs: Market Cap $%.2fB, Debt $%.2fB (weights E=%.2f, D=%.2f)",
            self.symbol,
            market_cap / 1e9,
            total_debt / 1e9,
            weight_equity,
            weight_debt,
        )
        logger.info(
            "%s - Cost of equity: %.2f%% (Rf %.2f%% + beta %.2f × ERP %.2f%%)",
            self.symbol,
            cost_of_equity * 100,
            risk_free_rate * 100,
            beta,
            market_risk_premium * 100,
        )
        logger.info(
            "%s - Cost of debt: %.2f%% (interest $%.2fB)",
            self.symbol,
            cost_of_debt * 100,
            interest_expense / 1e9,
        )

        # WACC calculation
        wacc_raw = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))

        # Ensure reasonable bounds (7% minimum, 20% maximum)
        wacc = max(0.07, min(wacc_raw, 0.20))

        # Log warning if WACC was clipped
        if wacc_raw < 0.07:
            logger.warning(
                "%s - ⚠️  WACC %.2f%% is below minimum 7%%, clipping to 7%%. "
                "This indicates potentially bad beta data (beta=%.3f, R²=check database). "
                "Recommend re-running beta calculator with fresh data.",
                self.symbol,
                wacc_raw * 100,
                beta
            )
        elif wacc_raw > 0.20:
            logger.warning(
                "%s - ⚠️  WACC %.2f%% is above maximum 20%%, clipping to 20%%. "
                "This indicates unusually high risk parameters.",
                self.symbol,
                wacc_raw * 100
            )

        logger.info(
            "%s - WACC: %.2f%% (raw: %.2f%%, bounded 7-20%%, tax rate %.0f%%)",
            self.symbol,
            wacc * 100,
            wacc_raw * 100,
            tax_rate * 100,
        )

        return wacc

    def _get_unlevered_beta(self) -> float:
        """
        Get unlevered beta from symbol table via DatabaseMarketDataFetcher

        Beta from market data (regression analysis) represents the asset's systematic risk.
        Prefers: b_12_month > b_24_month > b_36_month > b_60_month

        Returns:
            Unlevered beta value, defaults to 1.0 if not available
        """
        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher
            from investigator.config import get_config

            config = get_config()
            fetcher = get_market_data_fetcher(config)
            info = fetcher.get_stock_info(self.symbol)

            beta = info.get('beta')  # Already prioritizes b_12_month > b_24_month > b_36_month > b_60_month
            if beta and beta > 0:
                logger.info(f"{self.symbol} - Using unlevered beta from market data: {beta:.2f}")
                return float(beta)
            else:
                logger.warning(f"{self.symbol} - Beta not available, using default 1.0")
                return 1.0
        except Exception as e:
            logger.warning(f"{self.symbol} - Error fetching beta: {e}, using default 1.0")
            return 1.0

    def _get_sector_median_betas(self) -> Dict[str, float]:
        """
        Return sector median betas for fallback when individual beta is unreliable

        These values are based on academic research and industry portfolio analysis
        """
        return {
            'Healthcare': 0.70,          # Defensive, regulated
            'Health Care': 0.70,         # Alternative naming
            'Consumer Staples': 0.65,    # Very defensive
            'Utilities': 0.60,           # Regulated, stable
            'Financials': 1.10,          # Moderate cyclical
            'Financial Services': 1.10,  # Alternative naming
            'Banks': 1.15,               # Slightly more cyclical
            'Insurance': 0.90,           # Less cyclical than banks
            'Real Estate': 0.85,         # Interest rate sensitive
            'REITs': 0.85,               # Same as real estate
            'Energy': 1.20,              # Commodity cyclical
            'Materials': 1.15,           # Commodity cyclical
            'Basic Materials': 1.15,     # Alternative naming
            'Industrials': 1.10,         # Moderate cyclical
            'Consumer Discretionary': 1.25,  # Highly cyclical
            'Consumer Cyclical': 1.25,   # Alternative naming
            'Technology': 1.30,          # High growth, volatile
            'Information Technology': 1.30,  # Alternative naming
            'Communication Services': 1.05,  # Moderate
            'Telecommunications': 0.80,  # Defensive utilities-like
            'Default': 1.00,             # Market beta
        }

    def _get_beta_r_squared(self) -> float:
        """Get R² for 12-month beta from symbol table"""
        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher
            from investigator.config import get_config

            config = get_config()
            fetcher = get_market_data_fetcher(config)
            info = fetcher.get_stock_info(self.symbol)

            r2 = info.get('r2_12_month', 0)
            return float(r2) if r2 else 0.0
        except Exception as e:
            logger.debug(f"{self.symbol} - Unable to get R² from symbol table: {e}")
            return 0.0

    def _get_sector(self) -> str:
        """Get sector from symbol table"""
        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher
            from investigator.config import get_config

            config = get_config()
            fetcher = get_market_data_fetcher(config)
            info = fetcher.get_stock_info(self.symbol)

            return info.get('Sector', 'Default')
        except Exception as e:
            logger.debug(f"{self.symbol} - Unable to get sector from symbol table: {e}")
            return 'Default'

    def _determine_beta_treatment(
        self,
        beta_unlevered: float,
        r_squared: float,
        total_debt: float,
        equity: float,
        market_cap: float,
        sector: str,
        net_income: Optional[float] = None,
        total_revenue: Optional[float] = None
    ) -> tuple:
        """
        Determine appropriate beta treatment based on company characteristics

        This function implements a decision tree for selecting the right beta:
        1. Extreme buyback companies (AAPL): Use unlevered beta
        2. Financial sector (banks/REITs): Use unlevered beta (debt is inventory)
        3. Low R² beta (JNJ): Use sector median beta
        4. Negative equity: Use unlevered beta
        5. Extreme leverage (D/E > 10): Use unlevered beta (distress risk)
        6. Normal companies: Standard Hamada relevering

        Args:
            beta_unlevered: Unlevered beta from market data
            r_squared: R² from beta regression (0-1)
            total_debt: Total debt from balance sheet
            equity: Stockholders equity from balance sheet
            market_cap: Market capitalization (price × shares)
            sector: Company sector
            net_income: Net income (optional, for SBC detection)
            total_revenue: Total revenue (optional, for SBC detection)

        Returns:
            Tuple of (final_beta, treatment_method, rationale)
        """
        tax_rate = 0.21

        # Step 1: Extreme buyback detection (Equity/MarketCap < 10%)
        # CRITICAL FIX: Exclude high-SBC companies (these are DILUTING, not buying back)
        if market_cap and market_cap > 0 and equity > 0:
            equity_to_mktcap = equity / market_cap

            if equity_to_mktcap < 0.10:
                # Check if this is actually a high-SBC diluting company
                # rather than a true buyback company like AAPL
                is_high_sbc_company = False

                # Get SBC from latest period
                sbc_amount = self._get_latest_sbc()

                if sbc_amount and total_revenue and total_revenue > 0:
                    sbc_pct_of_revenue = sbc_amount / total_revenue

                    # High SBC = dilution, not buybacks
                    if sbc_pct_of_revenue > 0.10:  # SBC > 10% of revenue
                        is_high_sbc_company = True
                        logger.info(
                            f"🔍 [BETA] {self.symbol} - Low Equity/MarketCap={equity_to_mktcap:.2%} but "
                            f"high SBC={sbc_pct_of_revenue:.1%} of revenue indicates DILUTION, not buybacks. "
                            f"Using standard levered beta."
                        )

                # Also check if company is unprofitable (growth-stage, not buyback-stage)
                if not is_high_sbc_company and net_income is not None and net_income < 0:
                    is_high_sbc_company = True
                    logger.info(
                        f"🔍 [BETA] {self.symbol} - Low Equity/MarketCap={equity_to_mktcap:.2%} but "
                        f"negative net income=${net_income/1e9:.2f}B indicates growth-stage company, not buyback company. "
                        f"Using standard levered beta."
                    )

                # If not high-SBC, treat as true extreme buyback
                if not is_high_sbc_company:
                    return (
                        beta_unlevered,
                        "unlevered_extreme_buyback",
                        f"Extreme buyback structure (Equity/MarketCap={equity_to_mktcap:.2%})"
                    )
                # Otherwise, continue to standard levering logic below

        # Step 2: Financial sector (debt is inventory, not leverage)
        financial_sectors = [
            'Financials', 'Financial Services', 'Banks', 'Insurance',
            'Real Estate', 'REITs'
        ]
        if sector in financial_sectors:
            return (
                beta_unlevered,
                "unlevered_financial_sector",
                f"Financial sector ({sector}): debt is inventory, not leverage"
            )

        # Step 3: Low beta quality (R² < 10%)
        if r_squared < 0.10:
            sector_betas = self._get_sector_median_betas()
            fallback_beta = sector_betas.get(sector, 1.0)

            return (
                fallback_beta,
                "sector_median_low_r2",
                f"Low R²={r_squared:.2%}, using {sector} median beta={fallback_beta:.2f}"
            )

        # Step 4: Negative or zero equity (cannot lever)
        if equity <= 0:
            return (
                beta_unlevered,
                "unlevered_negative_equity",
                f"Negative equity ({equity/1e9:.1f}B), cannot lever beta"
            )

        # Step 5: Check for extreme leverage (distress risk)
        debt_to_equity = total_debt / equity
        if debt_to_equity > 10.0:
            return (
                beta_unlevered,
                "unlevered_distress_risk",
                f"Extreme D/E ratio ({debt_to_equity:.1f}), using unlevered to avoid distress artifacts"
            )

        # Step 6: Standard Hamada relevering
        beta_levered = beta_unlevered * (1 + (1 - tax_rate) * debt_to_equity)

        return (
            beta_levered,
            "standard_hamada",
            f"β={beta_unlevered:.2f} × [1+(1-{tax_rate:.0%})×D/E={debt_to_equity:.2f}] = {beta_levered:.2f}"
        )

    def _get_latest_sbc(self) -> Optional[float]:
        """
        Get latest Stock-Based Compensation from TTM periods

        Returns:
            Annual SBC amount or None if not available
        """
        try:
            # Get TTM periods (cached)
            ttm_periods = self._get_cached_ttm_periods(num_quarters=4, compute_missing=True)

            if not ttm_periods:
                return None

            # Sum SBC from last 4 quarters
            sbc_ttm = 0
            quarters_with_sbc = 0

            for period in ttm_periods[:4]:
                income_statement = period.get('income_statement', {})
                # SBC can be in different fields depending on data source
                sbc = (income_statement.get('stock_based_compensation') or
                       income_statement.get('stock_based_compensation_expense') or
                       0)

                if sbc and sbc > 0:
                    sbc_ttm += sbc
                    quarters_with_sbc += 1

            if quarters_with_sbc >= 2:  # Need at least 2 quarters
                return sbc_ttm

            return None

        except Exception as e:
            logger.debug(f"{self.symbol} - Unable to extract SBC: {e}")
            return None

    def _classify_company_stage(self, rule_of_40_score: float, fcf_margin: float, revenue_growth: float) -> str:
        """
        Classify company stage based on Rule of 40, FCF margin, and revenue growth

        Args:
            rule_of_40_score: Rule of 40 score (growth% + margin%)
            fcf_margin: FCF margin (FCF / Revenue)
            revenue_growth: Revenue growth rate

        Returns:
            'hyper_growth', 'growth', or 'mature'
        """
        stage_config = self.dcf_config.get('terminal_fcf_margin', {}).get('stage_classification', {})

        # Hyper-growth: Rule of 40 >50, FCF margin <15%, revenue growth >30%
        hyper_config = stage_config.get('hyper_growth', {})
        if (rule_of_40_score >= hyper_config.get('rule_of_40_min', 50) and
            fcf_margin < hyper_config.get('fcf_margin_max', 0.15) and
            revenue_growth >= hyper_config.get('revenue_growth_min', 0.30)):
            return 'hyper_growth'

        # Mature: Rule of 40 >40, FCF margin >25%, revenue growth <20%
        mature_config = stage_config.get('mature', {})
        if (rule_of_40_score >= mature_config.get('rule_of_40_min', 40) and
            fcf_margin >= mature_config.get('fcf_margin_min', 0.25) and
            revenue_growth <= mature_config.get('revenue_growth_max', 0.20)):
            return 'mature'

        # Growth: Everything in between
        # Rule of 40 40-50, FCF margin 15-25%, revenue growth 20-40%
        return 'growth'

    def _classify_company_size(self, market_cap: float) -> str:
        """
        Classify company size based on market cap

        Args:
            market_cap: Market capitalization in dollars

        Returns:
            'mega_cap', 'large_cap', 'mid_cap', or 'small_cap'
        """
        size_thresholds = self.dcf_config.get('terminal_fcf_margin', {}).get('size_thresholds', {})

        # Convert to billions for comparison
        market_cap_billions = market_cap / 1e9

        if market_cap_billions >= size_thresholds.get('mega_cap', 500):
            return 'mega_cap'
        elif market_cap_billions >= size_thresholds.get('large_cap', 50):
            return 'large_cap'
        elif market_cap_billions >= size_thresholds.get('mid_cap', 10):
            return 'mid_cap'
        else:
            return 'small_cap'

    def _get_terminal_margin_from_config(
        self,
        sector: str,
        size_tier: str,
        stage: str,
        current_fcf_margin: float
    ) -> float:
        """
        Get terminal FCF margin from config based on sector, size, and stage

        Args:
            sector: Sector name (e.g., 'Technology', 'Healthcare')
            size_tier: Size tier ('mega_cap', 'large_cap', 'mid_cap', 'small_cap')
            stage: Company stage ('mature', 'growth', 'hyper_growth')
            current_fcf_margin: Current FCF margin (used as floor to prevent contraction)

        Returns:
            Terminal FCF margin (0.0-1.0)
        """
        terminal_margin_config = self.dcf_config.get('terminal_fcf_margin', {})

        # Try to get sector-specific margin
        sector_config = terminal_margin_config.get(sector, {})

        # Try size-specific configuration
        size_config = sector_config.get(size_tier, {})

        # Get stage-specific margin
        config_margin = size_config.get(stage)

        if config_margin is None:
            # Fall back to sector default
            config_margin = sector_config.get('default')

        if config_margin is None:
            # Fall back to global default
            config_margin = terminal_margin_config.get('default', 0.20)

        # CRITICAL: Never force margin contraction
        # If company already has higher margins, allow them to maintain it
        terminal_margin = max(config_margin, current_fcf_margin * 0.95)  # Allow 5% modest decline

        logger.info(
            f"🔍 [TERMINAL_MARGIN_CONFIG] {self.symbol} - Terminal margin determination\n"
            f"  Sector: {sector}, Size: {size_tier}, Stage: {stage}\n"
            f"  Current FCF margin: {current_fcf_margin*100:.1f}%\n"
            f"  Config target margin: {config_margin*100:.1f}%\n"
            f"  Terminal margin (max): {terminal_margin*100:.1f}%\n"
            f"  Rationale: {'Maintaining current margins' if terminal_margin > config_margin else 'Using config target'}"
        )

        return terminal_margin

    def _calculate_levered_beta(self, beta_unlevered: float, total_debt: float, equity: float, market_cap: float = None) -> float:
        """
        Calculate levered beta using improved beta selection logic

        This method now delegates to _determine_beta_treatment() which implements
        a comprehensive decision tree for beta selection:
        - Extreme buyback companies (Equity/MarketCap < 10%): Use unlevered
        - Financial sector: Use unlevered (debt is inventory)
        - Low R² (< 10%): Use sector median beta
        - Extreme leverage (D/E > 10): Use unlevered (distress)
        - Normal companies: Standard Hamada relevering

        Args:
            beta_unlevered: Unlevered beta from market data
            total_debt: Total debt from balance sheet
            equity: Stockholders equity from balance sheet
            market_cap: Market capitalization (price × shares), optional

        Returns:
            Levered beta incorporating financial leverage (or appropriate alternative)
        """
        # Get R² and sector for smart beta selection
        r_squared = self._get_beta_r_squared()
        sector = self._get_sector()

        # Get net income and revenue for SBC detection (Fix 2)
        net_income = None
        total_revenue = None

        try:
            latest_ttm = self._get_cached_ttm_periods(num_quarters=4, compute_missing=True)
            if latest_ttm:
                latest = latest_ttm[0]
                income_statement = latest.get('income_statement', {})
                net_income = income_statement.get('net_income')
                total_revenue = income_statement.get('total_revenue') or income_statement.get('revenue')
        except Exception as e:
            logger.debug(f"{self.symbol} - Unable to extract net income/revenue for beta treatment: {e}")

        # Use improved beta treatment logic
        beta, treatment, rationale = self._determine_beta_treatment(
            beta_unlevered, r_squared, total_debt, equity, market_cap or 0, sector,
            net_income, total_revenue
        )

        logger.info(f"{self.symbol} - Beta treatment: {treatment} - {rationale}")

        return beta

    def _get_risk_free_rate(self) -> float:
        """
        Get 10-year Treasury yield (risk-free rate) from FRED macro indicators

        Fetches DGS10 (10-Year Treasury Constant Maturity Rate) from the macro indicators
        database, which is extracted from FRED API.

        Returns:
            Risk-free rate as decimal (e.g., 0.045 for 4.5%)
        """
        try:
            from investigator.infrastructure.external.fred import MacroIndicatorsFetcher

            fetcher = MacroIndicatorsFetcher()
            indicators = fetcher.get_latest_indicators(['DGS10'])  # 10-Year Treasury Rate

            if 'DGS10' in indicators and indicators['DGS10']['value'] is not None:
                # DGS10 is already in percentage (e.g., 4.5), convert to decimal (0.045)
                treasury_rate = indicators['DGS10']['value'] / 100
                logger.info(
                    f"{self.symbol} - Using 10Y Treasury rate from FRED: {treasury_rate*100:.2f}% "
                    f"(as of {indicators['DGS10']['date']})"
                )
                return treasury_rate
            else:
                logger.warning(f"{self.symbol} - DGS10 not available, using default risk-free rate: 4.5%")
                return 0.045

        except Exception as e:
            logger.warning(f"{self.symbol} - Error fetching Treasury rate: {e}, using default 4.5%")
            return 0.045

    def _discount_cash_flows(self, fcf_projections: List[float], wacc: float, log_details: bool = True) -> float:
        """
        Discount future cash flows to present value

        PV = FCF(t) / (1 + WACC)^t

        Args:
            fcf_projections: List of future FCF values
            wacc: Discount rate
            log_details: If True, log each year's calculation (default True, False for sensitivity)

        Returns:
            Total present value of cash flows
        """
        pv_total = 0
        for year, fcf in enumerate(fcf_projections, start=1):
            discount_factor = (1 + wacc) ** year
            pv = fcf / discount_factor
            pv_total += pv
            if log_details:
                logger.info(
                    "%s - Discount Year %d: FCF $%.2fB / (1+WACC)^%d = $%.2fB",
                    self.symbol,
                    year,
                    fcf / 1e9,
                    year,
                    pv / 1e9,
                )

        return pv_total

    def _calculate_equity_value(self, enterprise_value: float) -> float:
        """
        Convert enterprise value to equity value

        Equity Value = Enterprise Value - Net Debt + Cash

        Args:
            enterprise_value: Enterprise value in dollars

        Returns:
            Equity value in dollars
        """
        if not self.quarterly_metrics:
            return enterprise_value

        latest = self.quarterly_metrics[-1]

        # Equity Value = Enterprise Value - Net Debt + Cash
        balance_sheet = latest.get('balance_sheet', {})
        total_debt = balance_sheet.get('total_debt', 0) or 0
        # Cash is typically a top-level metric or in balance sheet
        cash = latest.get('cash_and_equivalents', 0) or balance_sheet.get('cash_and_equivalents', 0) or 0

        equity_value = enterprise_value - total_debt + cash

        return equity_value

    def _run_sensitivity_analysis(self, fcf_projections: List[float], base_wacc: float) -> Dict:
        """
        Run sensitivity analysis varying terminal growth rate only

        WACC is determined by formula (risk-free rate, beta, debt/equity), not varied.
        Only terminal growth rate is a legitimate assumption to vary (2-4%).

        Creates a 1D table showing fair value under different terminal growth scenarios

        Args:
            fcf_projections: Projected FCF values
            base_wacc: Calculated WACC from fundamentals

        Returns:
            Dictionary with sensitivity table
        """
        terminal_growth_rates = [0.02, 0.025, 0.03, 0.035, 0.04]

        sensitivity_table = []
        shares = self._get_shares_outstanding()

        for tgr in terminal_growth_rates:
            try:
                # Recalculate with different terminal growth assumptions
                terminal_value = (fcf_projections[-1] * (1 + tgr)) / (base_wacc - tgr)
                # Disable logging for sensitivity calculations (log_details=False)
                pv_fcf = self._discount_cash_flows(fcf_projections, base_wacc, log_details=False)
                pv_terminal = terminal_value / ((1 + base_wacc) ** len(fcf_projections))

                ev = pv_fcf + pv_terminal
                equity_value = self._calculate_equity_value(ev)
                fair_value = equity_value / shares if shares > 0 else 0

                sensitivity_table.append(round(fair_value, 2))
            except:
                sensitivity_table.append(0.0)

        return {
            'terminal_growth_rates': [f"{tgr*100:.1f}%" for tgr in terminal_growth_rates],
            'wacc': f"{base_wacc*100:.2f}%",
            'fair_values': sensitivity_table
        }

    def _get_current_price(self) -> float:
        """
        Get current stock price from symbol table via DatabaseMarketDataFetcher.

        Uses instance-level caching to avoid repeated lookups.

        Returns:
            Current price in dollars
        """
        # Return cached value if already computed
        if self._current_price_cache is not None:
            return self._current_price_cache

        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher
            from investigator.config import get_config

            config = get_config()
            fetcher = get_market_data_fetcher(config)
            info = fetcher.get_stock_info(self.symbol)

            current_price = info.get('current_price', 0)
            if current_price and current_price > 0:
                self._current_price_cache = float(current_price)
                return self._current_price_cache
            else:
                logger.warning(f"Current price not available for {self.symbol}")
                self._current_price_cache = 100.0  # Fallback value for testing
                return self._current_price_cache
        except Exception as e:
            logger.warning(f"Could not fetch current price for {self.symbol}: {e}")
            self._current_price_cache = 100.0  # Fallback value for testing
            return self._current_price_cache

    def _get_shares_outstanding(self) -> float:
        """
        Get shares outstanding from quarterly metrics (already available from SEC/fundamental step).
        Checks multiple nested locations for data availability.

        Uses instance-level caching to avoid repeated lookups and log spam.

        Preserves original SBC dilution logic that is applied later in terminal value projections.

        Returns:
            Number of shares
        """
        # Return cached value if already computed
        if self._shares_outstanding_cache is not None:
            return self._shares_outstanding_cache

        # Use shares_outstanding directly from latest quarterly data
        if self.quarterly_metrics:
            latest = self.quarterly_metrics[-1]

            # Debug: show what values exist at each location
            top_level_value = latest.get('shares_outstanding')
            ratios_dict = latest.get('ratios', {})
            ratios_value = ratios_dict.get('shares_outstanding') if ratios_dict else None
            balance_sheet_dict = latest.get('balance_sheet', {})
            balance_sheet_value = balance_sheet_dict.get('shares_outstanding') if balance_sheet_dict else None

            logger.info(f"🔍 {self.symbol} - shares_outstanding debug: top_level={top_level_value}, ratios={ratios_value}, balance_sheet={balance_sheet_value}")

            # Try multiple locations (priority order):
            # 1. Top level (backward compatibility)
            shares = top_level_value
            location = "top-level"

            # 2. Nested in ratios subdictionary
            if not shares or shares <= 0:
                if ratios_value and ratios_value > 0:
                    shares = ratios_value
                    location = "ratios subdictionary"

            # 3. Nested in balance_sheet subdictionary
            if not shares or shares <= 0:
                if balance_sheet_value and balance_sheet_value > 0:
                    shares = balance_sheet_value
                    location = "balance_sheet subdictionary"

            if shares and shares > 0:
                logger.info(f"{self.symbol} - Using shares outstanding from quarterly metrics ({location}): {shares:,.0f}")
                self._shares_outstanding_cache = float(shares)
                return self._shares_outstanding_cache
            else:
                logger.warning(f"{self.symbol} - shares_outstanding not found in quarterly_metrics (top-level={top_level_value}, ratios={ratios_value}, balance_sheet={balance_sheet_value})")
        else:
            logger.warning(f"{self.symbol} - quarterly_metrics not available")

        # Fallback: Try to get from database/market data
        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher
            from investigator.config import get_config

            config = get_config()
            fetcher = get_market_data_fetcher(config)
            market_data = fetcher.get_stock_info(self.symbol)

            shares_from_db = market_data.get('shares_outstanding')
            if shares_from_db and shares_from_db > 0:
                logger.info(f"{self.symbol} - Using shares outstanding from database/market data: {shares_from_db:,.0f}")
                self._shares_outstanding_cache = float(shares_from_db)
                return self._shares_outstanding_cache
        except Exception as e:
            logger.warning(f"{self.symbol} - Could not fetch shares outstanding from database: {e}")

        # Final fallback with clear warning
        logger.error(f"{self.symbol} - Using fallback shares outstanding (1B) - THIS IS LIKELY INCORRECT!")
        self._shares_outstanding_cache = 1000000000  # Fallback: 1B shares
        return self._shares_outstanding_cache

    def _get_market_cap(self) -> float:
        """
        Get market capitalization (current_price × shares_outstanding)

        Used for applying maturity-based constraints to growth assumptions:
        - Mega-cap (>$100B): Limited growth potential due to size
        - Large-cap ($10B-$100B): Moderate growth potential
        - Mid/Small-cap (<$10B): Higher growth potential

        Returns:
            Market cap in dollars
        """
        try:
            current_price = self._get_current_price()
            shares = self._get_shares_outstanding()
            market_cap = current_price * shares

            logger.debug(f"{self.symbol} - Market cap: ${market_cap/1e9:.1f}B (price=${current_price:.2f}, shares={shares:,.0f})")
            return market_cap
        except Exception as e:
            logger.warning(f"{self.symbol} - Could not calculate market cap: {e}")
            # Return a mid-cap fallback (no extreme constraints)
            return 50e9  # $50B fallback

    def _calculate_rule_of_40(self) -> Dict:
        """
        Calculate Rule of 40: Revenue Growth % + Profit Margin %

        This efficiency metric helps determine:
        1. Which valuation method to use (DCF vs P/S)
        2. Adjustments to terminal growth rate
        3. Appropriate P/S multiple ranges

        Uses TTM (Trailing 12 Months) for current state assessment:
        - Last 4 quarters for revenue growth (YoY)
        - Last 4 quarters for profit margin (FCF margin preferred)

        Returns:
            Dictionary with score, components, and classification
        """
        try:
            # Get TTM revenue growth (YoY)
            revenue_growth_pct = self._get_ttm_revenue_growth()

            # Get TTM profit margin (FCF margin preferred, operating margin fallback)
            profit_margin_pct = self._get_ttm_profit_margin()

            # Calculate Rule of 40 score
            rule_of_40_score = revenue_growth_pct + profit_margin_pct

            # Classify efficiency tier
            classification = self._classify_rule_of_40(rule_of_40_score)

            # Get thresholds from config
            rule_40_config = self.dcf_config.get('rule_of_40', {})
            thresholds = rule_40_config.get('thresholds', {})

            return {
                'score': rule_of_40_score,
                'revenue_growth_pct': revenue_growth_pct,
                'profit_margin_pct': profit_margin_pct,
                'classification': classification,
                'thresholds': thresholds
            }

        except Exception as e:
            logger.warning(f"{self.symbol} - Error calculating Rule of 40: {e}. Using default.")
            return {
                'score': 0.0,
                'revenue_growth_pct': 0.0,
                'profit_margin_pct': 0.0,
                'classification': 'poor',
                'thresholds': {}
            }

    def _qualifies_for_ps_integration(self, classification: str) -> bool:
        """
        Determine if the Rule of 40 classification clears the bar for P/S blending.
        """
        rule_40_config = self.dcf_config.get('rule_of_40', {})
        ps_config = rule_40_config.get('ps_integration', {})
        min_class = (ps_config.get('min_classification') or 'good').lower()

        tiers = ['poor', 'acceptable', 'good', 'excellent']
        tier_index = {tier: idx for idx, tier in enumerate(tiers)}

        current_rank = tier_index.get(classification.lower(), 0)
        min_rank = tier_index.get(min_class, tier_index['good'])

        return current_rank >= min_rank

    def _get_ttm_revenue_amount(self) -> float:
        """
        Return trailing-twelve-month revenue using the four most recent quarters.
        """
        if not self.quarterly_metrics:
            return 0.0

        # SEC FORMAT HANDLING
        if self._is_sec_format():
            sec_data = self._get_sec_data()
            income_stmt = sec_data.get('income_statement', {})
            revenue = income_stmt.get('total_revenue', 0) or income_stmt.get('revenue', 0) or 0
            return float(revenue)

        try:
            # Use cached TTM periods to avoid redundant Q4 computations
            ttm_periods = self._get_cached_ttm_periods(num_quarters=4, compute_missing=True)
            if not ttm_periods:
                return 0.0

            ttm_revenue = sum(
                (period.get('income_statement', {}) or {}).get('total_revenue', 0) or 0
                for period in ttm_periods
            )
            return float(ttm_revenue)
        except Exception as exc:
            logger.debug("%s - Unable to compute TTM revenue: %s", self.symbol, exc)
            return 0.0

    def _calculate_ps_valuation(
        self,
        rule_of_40_result: Dict,
        shares_outstanding: float,
        current_price: float,
    ) -> Optional[Dict]:
        """
        Derive a P/S-based fair value when growth efficiency warrants multiple expansion.
        """
        classification = rule_of_40_result.get('classification', 'poor')
        if not self._qualifies_for_ps_integration(classification):
            return None

        ttm_revenue = self._get_ttm_revenue_amount()
        if ttm_revenue <= 0 or shares_outstanding <= 0:
            return None

        revenue_per_share = ttm_revenue / shares_outstanding
        market_cap = current_price * shares_outstanding if current_price > 0 else None
        current_ps_multiple = (market_cap / ttm_revenue) if market_cap and ttm_revenue > 0 else None

        rule_40_config = self.dcf_config.get('rule_of_40', {})
        multiple_ranges = rule_40_config.get('multiple_ranges', {})
        classification_key = classification.lower()
        range_config = multiple_ranges.get(classification_key) or multiple_ranges.get('default')

        if not range_config:
            return None

        min_ps = range_config.get('min_ps')
        max_ps = range_config.get('max_ps')
        if not min_ps or not max_ps:
            return None

        midpoint = (min_ps + max_ps) / 2
        applied_multiple = midpoint

        if current_ps_multiple is not None:
            if current_ps_multiple < min_ps:
                applied_multiple = min_ps
            elif current_ps_multiple > max_ps:
                applied_multiple = max_ps
            else:
                if classification_key == 'excellent':
                    applied_multiple = (current_ps_multiple + max_ps) / 2
                elif classification_key == 'good':
                    applied_multiple = (current_ps_multiple + midpoint) / 2
                else:
                    applied_multiple = current_ps_multiple
        else:
            applied_multiple = max_ps if classification_key == 'excellent' else midpoint

        ps_fair_value = revenue_per_share * applied_multiple

        return {
            'fair_value_per_share': ps_fair_value,
            'applied_ps_multiple': applied_multiple,
            'multiple_range': {
                'min': min_ps,
                'max': max_ps,
                'description': range_config.get('description'),
            },
            'ttm_revenue': ttm_revenue,
            'ttm_revenue_per_share': revenue_per_share,
            'current_ps_multiple': current_ps_multiple,
            'qualification': classification,
        }

    def _get_ttm_revenue_growth(self) -> float:
        """
        Calculate TTM (Trailing 12 Months) revenue growth rate using geometric mean

        Strategy (Stable Growth with Geometric Mean):
        1. Get 12 quarters for 3 TTM periods
        2. Calculate Current TTM (Q0-Q3), Prior1 TTM (Q4-Q7), Prior2 TTM (Q8-Q11)
        3. Calculate growth: Current vs Prior1, Prior1 vs Prior2
        4. Use geometric mean: sqrt((1+g1) × (1+g2)) - 1
        5. Edge case: If only 10-11 quarters, use simple growth to avoid inflated rates

        Returns:
            Revenue growth rate as percentage (e.g., 15.5 for 15.5%)
        """
        # SEC FORMAT HANDLING
        # SEC filing tool returns single FY snapshot - cannot calculate YoY growth
        # Fall back to sector-based growth estimate from config
        if self._is_sec_format():
            # Get sector growth defaults from config.yaml
            sec_defaults = self.dcf_config.get('sec_format_defaults', {})
            sector_growth_config = sec_defaults.get('sector_revenue_growth', {})
            default_growth = sector_growth_config.get(self.sector, sector_growth_config.get('Default', 5.0))
            logger.info(
                f"🔍 {self.symbol} - SEC format: Using sector-based revenue growth estimate "
                f"({self.sector}: {default_growth:.1f}%) - single FY snapshot cannot calculate YoY"
            )
            return default_growth

        if not self.quarterly_metrics or len(self.quarterly_metrics) < 8:
            logger.warning(f"{self.symbol} - Insufficient quarterly data for TTM revenue growth. Need 8 quarters, have {len(self.quarterly_metrics) if self.quarterly_metrics else 0}")
            return 0.0

        try:
            # Use cached TTM periods (12 quarters) to avoid redundant Q4 computations
            quarters_12 = self._get_cached_ttm_periods(num_quarters=12, compute_missing=True)

            if not quarters_12 or len(quarters_12) < 8:
                logger.warning(f"{self.symbol} - Need at least 8 quarters for revenue growth, got {len(quarters_12) if quarters_12 else 0}")
                return 0.0

            # Filter out FY (full year) periods - only use quarterly data (Q1-Q4)
            quarterly_only = [q for q in quarters_12 if q.get('fiscal_period') in ['Q1', 'Q2', 'Q3', 'Q4']]

            if len(quarterly_only) < 8:
                logger.warning(f"{self.symbol} - After filtering FY periods, only {len(quarterly_only)} quarters available")
                return 0.0

            # Extract revenue values
            revenue_values = []
            for q in quarterly_only[:12]:  # Use up to 12 quarters
                income_stmt = q.get('income_statement', {})
                revenue = income_stmt.get('total_revenue', 0) or 0
                revenue_values.append(revenue)

            # Calculate 3 TTM periods
            current_ttm = sum(revenue_values[0:4]) if len(revenue_values) >= 4 else 0
            prior1_ttm = sum(revenue_values[4:8]) if len(revenue_values) >= 8 else 0
            prior2_ttm = sum(revenue_values[8:12]) if len(revenue_values) >= 12 else 0

            # Calculate growth rates
            growth_current_vs_prior1 = ((current_ttm / prior1_ttm) - 1) * 100 if prior1_ttm > 0 else 0
            growth_prior1_vs_prior2 = ((prior1_ttm / prior2_ttm) - 1) * 100 if prior2_ttm > 0 else 0

            # Geometric mean calculation (only if exactly 12 quarters to avoid inflated rates)
            if len(revenue_values) == 12 and prior1_ttm > 0 and prior2_ttm > 0:
                # Full 12 quarters available - use geometric mean for stability
                ratio1 = 1 + (growth_current_vs_prior1 / 100)
                ratio2 = 1 + (growth_prior1_vs_prior2 / 100)
                # Protect against negative product (would produce complex number)
                ratio_product = ratio1 * ratio2
                if ratio_product <= 0:
                    # Fallback to simple growth if product is negative (severe decline)
                    logger.warning(
                        f"⚠️ {self.symbol} - Cannot compute geometric mean: negative ratio product "
                        f"(ratio1={ratio1:.3f}, ratio2={ratio2:.3f}). Using simple growth."
                    )
                    yoy_growth_pct = growth_current_vs_prior1
                else:
                    geometric_mean_ratio = ratio_product ** 0.5
                    yoy_growth_pct = (geometric_mean_ratio - 1) * 100

                logger.info(
                    f"🔍 [REVENUE_GROWTH] {self.symbol} - TTM Revenue Growth (Geometric Mean - 12 quarters):\n"
                    f"  Current TTM: ${current_ttm/1e9:.2f}B (growth: {growth_current_vs_prior1:+.1f}%)\n"
                    f"  Prior1 TTM:  ${prior1_ttm/1e9:.2f}B (growth: {growth_prior1_vs_prior2:+.1f}%)\n"
                    f"  Prior2 TTM:  ${prior2_ttm/1e9:.2f}B\n"
                    f"  Geometric Mean Growth: {yoy_growth_pct:.1f}%"
                )
            elif len(revenue_values) >= 8 and prior1_ttm > 0:
                # Simple growth if we don't have exactly 12 quarters
                yoy_growth_pct = growth_current_vs_prior1
                logger.info(
                    f"🔍 [REVENUE_GROWTH] {self.symbol} - TTM Revenue Growth (Simple - {len(revenue_values)} quarters):\n"
                    f"  Current TTM: ${current_ttm/1e9:.2f}B\n"
                    f"  Prior1 TTM:  ${prior1_ttm/1e9:.2f}B\n"
                    f"  Growth: {yoy_growth_pct:.1f}%"
                )
                if len(revenue_values) >= 10:
                    logger.warning(
                        f"⚠️ {self.symbol} - Only {len(revenue_values)} quarters available for revenue (need exactly 12 for geometric mean). "
                        f"Using simple TTM growth to avoid inflated rates from incomplete prior2 period."
                    )
            else:
                logger.warning(f"{self.symbol} - Insufficient data for revenue growth calculation")
                yoy_growth_pct = 0.0

            return yoy_growth_pct

        except Exception as e:
            logger.warning(f"{self.symbol} - Error calculating TTM revenue growth: {e}")
            return 0.0

    def _get_ttm_profit_margin(self) -> float:
        """
        Calculate TTM profit margin

        Priority:
        1. FCF Margin = (TTM FCF / TTM Revenue) * 100
        2. Operating Margin = (TTM Operating Income / TTM Revenue) * 100 (if FCF unreliable)

        Returns:
            Profit margin as percentage (e.g., 25.0 for 25%)
        """
        # SEC FORMAT HANDLING
        # SEC filing tool returns annual (FY) data with nested statements
        if self._is_sec_format():
            sec_data = self._get_sec_data()
            cash_flow = sec_data.get('cash_flow', {})
            income_stmt = sec_data.get('income_statement', {})

            # Get FCF and revenue from SEC data
            fcf = cash_flow.get('free_cash_flow', 0) or 0
            if fcf == 0:
                ocf = cash_flow.get('operating_cash_flow', 0) or 0
                capex = abs(cash_flow.get('capital_expenditures', 0) or 0)
                fcf = ocf - capex

            revenue = income_stmt.get('total_revenue', 0) or income_stmt.get('revenue', 0) or 0

            if revenue <= 0:
                logger.warning(f"{self.symbol} - SEC format: revenue is {revenue}, cannot calculate margin")
                return 0.0

            fcf_margin_pct = (fcf / revenue) * 100

            # Fallback to operating margin if FCF margin is too negative
            if fcf_margin_pct < -20:
                operating_income = income_stmt.get('operating_income', 0) or 0
                operating_margin_pct = (operating_income / revenue) * 100
                logger.info(
                    f"🔍 {self.symbol} - SEC format profit margin (Operating): {operating_margin_pct:.1f}% "
                    f"(FCF margin {fcf_margin_pct:.1f}% too negative)"
                )
                return operating_margin_pct

            logger.info(
                f"🔍 {self.symbol} - SEC format profit margin (FCF): {fcf_margin_pct:.1f}% "
                f"(FCF: ${fcf/1e9:.2f}B, Revenue: ${revenue/1e9:.2f}B)"
            )
            return fcf_margin_pct

        if not self.quarterly_metrics or len(self.quarterly_metrics) < 4:
            logger.warning(f"{self.symbol} - Insufficient quarterly data for TTM profit margin")
            return 0.0

        try:
            # Use cached TTM periods (4 quarters) to avoid redundant Q4 computations
            ttm_periods = self._get_cached_ttm_periods(num_quarters=4, compute_missing=True)

            if not ttm_periods:
                return 0.0

            # Calculate TTM FCF and Revenue
            ttm_fcf = 0
            ttm_revenue = 0

            for period in ttm_periods:
                # FCF from cash flow statement
                cash_flow = period.get('cash_flow', {})
                ocf = cash_flow.get('operating_cash_flow', 0) or 0
                capex = abs(cash_flow.get('capital_expenditures', 0) or 0)
                fcf = ocf - capex
                ttm_fcf += fcf

                # Revenue from income statement
                income_stmt = period.get('income_statement', {})
                revenue = income_stmt.get('total_revenue', 0) or 0
                ttm_revenue += revenue

            if ttm_revenue <= 0:
                logger.warning(f"{self.symbol} - TTM revenue is {ttm_revenue}, cannot calculate margin")
                return 0.0

            # Try FCF margin first
            fcf_margin_pct = (ttm_fcf / ttm_revenue) * 100

            # If FCF is negative or very low, use operating margin as fallback
            if fcf_margin_pct < -20:  # Allow some negative FCF for growth companies
                logger.info(f"{self.symbol} - FCF margin is {fcf_margin_pct:.1f}% (negative), using operating margin fallback")

                # Calculate operating margin
                ttm_operating_income = sum(
                    period.get('income_statement', {}).get('operating_income', 0) or 0
                    for period in ttm_periods
                )

                operating_margin_pct = (ttm_operating_income / ttm_revenue) * 100

                logger.info(
                    f"{self.symbol} - Using Operating Margin: {operating_margin_pct:.1f}% "
                    f"(TTM Operating Income: ${ttm_operating_income/1e9:.2f}B)"
                )

                return operating_margin_pct
            else:
                logger.info(
                    f"{self.symbol} - Using FCF Margin: {fcf_margin_pct:.1f}% "
                    f"(TTM FCF: ${ttm_fcf/1e9:.2f}B, TTM Revenue: ${ttm_revenue/1e9:.2f}B)"
                )

                return fcf_margin_pct

        except Exception as e:
            logger.warning(f"{self.symbol} - Error calculating TTM profit margin: {e}")
            return 0.0

    def _classify_rule_of_40(self, score: float) -> str:
        """
        Classify Rule of 40 score into efficiency tier

        Thresholds (from config):
        - Excellent: ≥ 50%
        - Good: 40-50%
        - Acceptable: 30-40%
        - Poor: < 30%

        Args:
            score: Rule of 40 score

        Returns:
            Classification string
        """
        rule_140_config = self.dcf_config.get('rule_of_40', {})
        thresholds = rule_140_config.get('thresholds', {
            'excellent': 50.0,
            'good': 40.0,
            'acceptable': 30.0,
            'poor': 20.0
        })

        if score >= thresholds.get('excellent', 50.0):
            return 'excellent'
        elif score >= thresholds.get('good', 40.0):
            return 'good'
        elif score >= thresholds.get('acceptable', 30.0):
            return 'acceptable'
        else:
            return 'poor'

    def _get_sector_growth_caps(self) -> Dict:
        """
        Get sector-specific FCF growth rate caps

        Returns min/max growth bounds based on industry characteristics

        Returns:
            Dictionary with min_growth and max_growth
        """
        growth_caps = self.dcf_config.get('fcf_growth_caps_by_sector', {})
        default_caps = growth_caps.get('Default', {
            'min_growth': -0.10,
            'max_growth': 0.25,
            'rationale': 'Balanced defaults'
        })

        sector_caps = growth_caps.get(self.sector, default_caps)

        logger.info(
            f"{self.symbol} - Using {self.sector} growth caps: "
            f"min={sector_caps['min_growth']*100:.0f}%, max={sector_caps['max_growth']*100:.0f}%"
        )

        return sector_caps

    def _get_terminal_growth_adjustment(self, classification: str) -> float:
        """
        Get terminal growth rate adjustment based on company maturity and efficiency

        IMPROVED LOGIC (per user feedback):
        - High FCF margin (>25%) + positive revenue growth → 3.5% (mature, efficient)
        - Rule of 40 >40 → 4.0% (high growth)
        - Otherwise → 3.0% (declining/inefficient)

        This logic recognizes that mature companies with strong FCF margins (like AAPL)
        shouldn't be penalized by low Rule of 40 scores (which reward revenue growth over efficiency).

        Args:
            classification: Rule of 40 tier (excellent/good/acceptable/poor)

        Returns:
            Adjustment to add to base terminal growth rate (as decimal)
        """
        try:
            # Get current Rule of 40 score and components
            rule_of_40_result = self._calculate_rule_of_40()
            rule_of_40_score = rule_of_40_result.get('score', 0)
            revenue_growth_pct = rule_of_40_result.get('revenue_growth_pct', 0)
            fcf_margin_pct = rule_of_40_result.get('profit_margin_pct', 0)  # This is FCF margin

            # IMPROVED LOGIC: Reward FCF efficiency for mature companies
            if fcf_margin_pct > 25.0 and revenue_growth_pct > 0:
                # Mature, efficient company (like AAPL with 27.8% FCF margin)
                adjustment = 0.005  # 3.0% base + 0.5% = 3.5% terminal growth
                reason = f"Mature, efficient company (FCF margin {fcf_margin_pct:.1f}% >25%, revenue growth {revenue_growth_pct:.1f}% >0)"
            elif rule_of_40_score > 40:
                # High growth company
                adjustment = 0.010  # 3.0% base + 1.0% = 4.0% terminal growth
                reason = f"High growth company (Rule of 40: {rule_of_40_score:.1f}% >40)"
            else:
                # Declining or inefficient company
                adjustment = 0.0  # 3.0% base (no adjustment)
                reason = f"Declining/inefficient (Rule of 40: {rule_of_40_score:.1f}% <40, FCF margin {fcf_margin_pct:.1f}%)"

            logger.info(
                f"{self.symbol} - Terminal growth adjustment: {adjustment*100:+.2f}% → {reason}"
            )

            return adjustment

        except Exception as e:
            logger.warning(f"{self.symbol} - Error calculating terminal growth adjustment: {e}. Using 0.0%")
            return 0.0

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
