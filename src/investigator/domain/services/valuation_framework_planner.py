"""
Valuation Framework Planner

Determines which valuation frameworks to execute based on sector, industry, and
company financial profile. Ensures optimal framework selection for blended valuation.

This is the planning layer that runs BEFORE framework execution, allowing all
frameworks to be executed in parallel.

Created: 2025-11-12
Author: Claude Code
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FrameworkConfig:
    """Configuration for a single valuation framework"""
    type: str              # Framework type (e.g., 'dcf_growth', 'pe_ratio', 'ev_ebitda')
    priority: int          # Execution priority (1 = highest)
    weight: float          # Weight in blended valuation (0.0-1.0)
    params: Dict[str, Any] # Framework-specific parameters
    reason: str            # Why this framework was selected


class ValuationFrameworkPlanner:
    """
    Determines which valuation frameworks to execute based on company characteristics

    Replaces the sequential sector routing approach with upfront planning,
    enabling parallel execution of all frameworks.

    Framework Selection Logic:
    1. DCF (Growth) - Always included for companies with positive cash flow
    2. DCF (Fading) - Mature/declining companies (revenue growth <5%)
    3. P/E Ratio - Companies with positive earnings
    4. EV/EBITDA - Companies with positive EBITDA
    5. P/S Ratio - Growth companies, negative earnings
    6. PEG Ratio - High growth companies (growth >15%)
    7. Gordon Growth Model - Dividend-paying companies (payout ratio >20%)

    Example:
        >>> planner = ValuationFrameworkPlanner(
        ...     symbol='ZS',
        ...     sector='Technology',
        ...     industry='Software - Infrastructure'
        ... )
        >>> frameworks = planner.plan_frameworks(
        ...     has_positive_earnings=True,
        ...     has_positive_ebitda=True,
        ...     has_positive_fcf=True,
        ...     revenue_growth_pct=28.6,
        ...     payout_ratio=0.0
        ... )
        >>> len(frameworks)
        4  # DCF Growth, P/E, EV/EBITDA, PEG
    """

    # Framework type constants
    FRAMEWORK_DCF_GROWTH = 'dcf_growth'
    FRAMEWORK_DCF_FADING = 'dcf_fading'
    FRAMEWORK_PE_RATIO = 'pe_ratio'
    FRAMEWORK_EV_EBITDA = 'ev_ebitda'
    FRAMEWORK_PS_RATIO = 'ps_ratio'
    FRAMEWORK_PEG_RATIO = 'peg_ratio'
    FRAMEWORK_GORDON_GROWTH = 'gordon_growth_model'

    # Thresholds
    THRESHOLD_HIGH_GROWTH = 15.0        # Revenue growth for PEG ratio
    THRESHOLD_MATURE_GROWTH = 5.0       # Revenue growth for fading DCF
    THRESHOLD_DIVIDEND_PAYOUT = 20.0    # Payout ratio for GGM

    # Company size thresholds (market cap in billions)
    SIZE_MEGA_CAP = 200.0       # $200B+ (AAPL, MSFT, GOOGL, META, AMZN)
    SIZE_LARGE_CAP = 10.0       # $10B - $200B
    SIZE_MID_CAP = 2.0          # $2B - $10B
    SIZE_SMALL_CAP = 0.3        # $300M - $2B
    # Below $300M = micro-cap

    # Projection years by company type
    PROJECTION_YEARS = {
        'tech_light_asset': 5,      # SaaS, software (standard)
        'tech_heavy_asset': 7,      # Semiconductors, hardware
        'mature_stable': 10,        # Utilities, consumer staples
        'high_growth': 5,           # Early-stage, high growth
        'default': 5                # Conservative default
    }

    # Fading growth assumptions by company size and stage
    # IMPORTANT: fcf_growth_ceiling is a MAXIMUM CAP, not a fixed value
    # Actual initial growth = min(historical_fcf_growth, ceiling)
    # This prevents deflating high-growth stocks with arbitrary fixed values
    GROWTH_ASSUMPTIONS = {
        'early_stage_saas': {
            'fcf_growth_ceiling': 0.30,     # 30% CEILING (cap historical growth)
            'fcf_growth_fade_to': 0.10,     # 10% by year 5 (fade target)
            'terminal_growth': 0.035,       # 3.5% perpetuity
            'projection_years': 5,
            'rationale': 'High-growth SaaS, 30% ceiling prevents unrealistic projections'
        },
        'mid_stage_tech': {
            'fcf_growth_ceiling': 0.20,     # 20% CEILING
            'fcf_growth_fade_to': 0.09,     # 9% by year 5
            'terminal_growth': 0.035,
            'projection_years': 5,
            'rationale': 'Mid-stage tech, 20% ceiling with gentle fade'
        },
        'mature_platform': {  # DoorDash, Uber, Airbnb
            'fcf_growth_ceiling': 0.15,     # 15% CEILING
            'fcf_growth_fade_to': 0.06,     # 6% by year 5
            'terminal_growth': 0.030,
            'projection_years': 5,
            'rationale': 'Mature platform, 15% ceiling reflects slowing growth'
        },
        'mega_cap_tech': {  # AAPL, MSFT, GOOGL, META
            'fcf_growth_ceiling': 0.10,     # 10% CEILING
            'fcf_growth_fade_to': 0.04,     # 4% by year 5
            'terminal_growth': 0.030,
            'projection_years': 5,
            'rationale': 'Mega-cap, 10% ceiling - hard to sustain growth at scale'
        },
        'stable_mature': {
            'fcf_growth_ceiling': 0.06,     # 6% CEILING
            'fcf_growth_fade_to': 0.03,     # 3% by year 5
            'terminal_growth': 0.025,
            'projection_years': 7,
            'rationale': 'Stable mature, 6% ceiling for utilities/consumer staples'
        }
    }

    # Sector-specific weights
    SECTOR_WEIGHTS = {
        'Technology': {
            FRAMEWORK_DCF_GROWTH: 0.35,
            FRAMEWORK_PE_RATIO: 0.25,
            FRAMEWORK_EV_EBITDA: 0.20,
            FRAMEWORK_PEG_RATIO: 0.20,
        },
        'Healthcare': {
            FRAMEWORK_DCF_GROWTH: 0.30,
            FRAMEWORK_PE_RATIO: 0.25,
            FRAMEWORK_EV_EBITDA: 0.25,
            FRAMEWORK_PEG_RATIO: 0.20,
        },
        'Financials': {
            FRAMEWORK_PE_RATIO: 0.40,
            FRAMEWORK_PS_RATIO: 0.30,
            FRAMEWORK_GORDON_GROWTH: 0.30,
        },
        'Real Estate': {
            FRAMEWORK_DCF_GROWTH: 0.40,
            FRAMEWORK_GORDON_GROWTH: 0.40,
            FRAMEWORK_PS_RATIO: 0.20,
        },
        'Utilities': {
            FRAMEWORK_DCF_FADING: 0.40,
            FRAMEWORK_GORDON_GROWTH: 0.40,
            FRAMEWORK_PE_RATIO: 0.20,
        },
        'Consumer Defensive': {
            FRAMEWORK_DCF_FADING: 0.30,
            FRAMEWORK_PE_RATIO: 0.30,
            FRAMEWORK_EV_EBITDA: 0.25,
            FRAMEWORK_GORDON_GROWTH: 0.15,
        },
        # Default weights for other sectors
        'default': {
            FRAMEWORK_DCF_GROWTH: 0.35,
            FRAMEWORK_PE_RATIO: 0.30,
            FRAMEWORK_EV_EBITDA: 0.25,
            FRAMEWORK_PS_RATIO: 0.10,
        }
    }

    def __init__(
        self,
        symbol: str,
        sector: str,
        industry: str,
        market_cap_billions: float = 0.0,
        base_terminal_growth: float = 0.035
    ):
        """
        Initialize framework planner

        Args:
            symbol: Stock symbol
            sector: Company sector (e.g., 'Technology', 'Healthcare')
            industry: Company industry (e.g., 'Software - Infrastructure')
            market_cap_billions: Market capitalization in billions (for size classification)
            base_terminal_growth: Base terminal growth rate for DCF (default: 3.5%)
        """
        self.symbol = symbol
        self.sector = sector
        self.industry = industry
        self.market_cap_billions = market_cap_billions
        self.base_terminal_growth = base_terminal_growth
        self.last_plan: Optional[List[FrameworkConfig]] = None
        self.company_stage: Optional[str] = None  # Classify on first use

    def plan_frameworks(
        self,
        has_positive_earnings: bool,
        has_positive_ebitda: bool,
        has_positive_fcf: bool,
        has_revenue: bool = True,
        revenue_growth_pct: float = 0.0,
        payout_ratio: float = 0.0,
        is_declining: bool = False
    ) -> List[FrameworkConfig]:
        """
        Plan which valuation frameworks to execute

        This is the SINGLE PLANNING STEP that happens before parallel execution.
        All frameworks determined here will be executed concurrently.

        Args:
            has_positive_earnings: Company has positive net income
            has_positive_ebitda: Company has positive EBITDA
            has_positive_fcf: Company has positive free cash flow
            has_revenue: Company has revenue (almost always true)
            revenue_growth_pct: Revenue growth percentage (e.g., 28.6 for 28.6%)
            payout_ratio: Dividend payout ratio (e.g., 35.0 for 35%)
            is_declining: Company is in decline (revenue/margins contracting)

        Returns:
            List of FrameworkConfig objects, ordered by priority

        Example:
            >>> frameworks = planner.plan_frameworks(
            ...     has_positive_earnings=True,
            ...     has_positive_ebitda=True,
            ...     has_positive_fcf=True,
            ...     revenue_growth_pct=28.6,
            ...     payout_ratio=0.0
            ... )
            >>> [f.type for f in frameworks]
            ['dcf_growth', 'pe_ratio', 'ev_ebitda', 'peg_ratio']
        """
        frameworks = []
        priority = 1

        # Get sector-specific weights
        sector_weights = self.SECTOR_WEIGHTS.get(
            self.sector,
            self.SECTOR_WEIGHTS['default']
        )

        # 1. DCF Frameworks (always highest priority if cash flow positive)
        if has_positive_fcf:
            if is_declining or revenue_growth_pct < self.THRESHOLD_MATURE_GROWTH:
                # Fading DCF for mature/declining companies
                frameworks.append(FrameworkConfig(
                    type=self.FRAMEWORK_DCF_FADING,
                    priority=priority,
                    weight=sector_weights.get(self.FRAMEWORK_DCF_FADING, 0.35),
                    params={
                        'base_terminal_growth': self.base_terminal_growth,
                        'fading_period_years': 5,
                    },
                    reason=f"Mature/declining (revenue growth {revenue_growth_pct:.1f}% <{self.THRESHOLD_MATURE_GROWTH}%)"
                ))
                priority += 1
            else:
                # Growth DCF for growing companies
                frameworks.append(FrameworkConfig(
                    type=self.FRAMEWORK_DCF_GROWTH,
                    priority=priority,
                    weight=sector_weights.get(self.FRAMEWORK_DCF_GROWTH, 0.35),
                    params={
                        'base_terminal_growth': self.base_terminal_growth,
                        'projection_years': 10,
                    },
                    reason=f"Growing company (revenue growth {revenue_growth_pct:.1f}% >{self.THRESHOLD_MATURE_GROWTH}%)"
                ))
                priority += 1

        # 2. P/E Ratio (if positive earnings)
        if has_positive_earnings:
            frameworks.append(FrameworkConfig(
                type=self.FRAMEWORK_PE_RATIO,
                priority=priority,
                weight=sector_weights.get(self.FRAMEWORK_PE_RATIO, 0.30),
                params={
                    'use_forward_pe': True,  # Use forward P/E if available
                },
                reason="Positive earnings available"
            ))
            priority += 1

        # 3. EV/EBITDA (if positive EBITDA)
        if has_positive_ebitda:
            frameworks.append(FrameworkConfig(
                type=self.FRAMEWORK_EV_EBITDA,
                priority=priority,
                weight=sector_weights.get(self.FRAMEWORK_EV_EBITDA, 0.25),
                params={},
                reason="Positive EBITDA available"
            ))
            priority += 1

        # 4. PEG Ratio (high growth companies)
        if has_positive_earnings and revenue_growth_pct > self.THRESHOLD_HIGH_GROWTH:
            frameworks.append(FrameworkConfig(
                type=self.FRAMEWORK_PEG_RATIO,
                priority=priority,
                weight=sector_weights.get(self.FRAMEWORK_PEG_RATIO, 0.20),
                params={
                    'expected_growth': revenue_growth_pct,
                },
                reason=f"High growth (revenue growth {revenue_growth_pct:.1f}% >{self.THRESHOLD_HIGH_GROWTH}%)"
            ))
            priority += 1

        # 5. Gordon Growth Model (dividend-paying companies)
        if payout_ratio > self.THRESHOLD_DIVIDEND_PAYOUT:
            frameworks.append(FrameworkConfig(
                type=self.FRAMEWORK_GORDON_GROWTH,
                priority=priority,
                weight=sector_weights.get(self.FRAMEWORK_GORDON_GROWTH, 0.15),
                params={
                    'base_terminal_growth': self.base_terminal_growth,
                },
                reason=f"Dividend-paying (payout ratio {payout_ratio:.1f}% >{self.THRESHOLD_DIVIDEND_PAYOUT}%)"
            ))
            priority += 1

        # 6. P/S Ratio (fallback for negative earnings growth companies)
        if not has_positive_earnings and has_revenue:
            frameworks.append(FrameworkConfig(
                type=self.FRAMEWORK_PS_RATIO,
                priority=priority,
                weight=sector_weights.get(self.FRAMEWORK_PS_RATIO, 0.30),
                params={},
                reason="Negative earnings, using revenue multiple"
            ))
            priority += 1

        # Normalize weights to sum to 1.0
        total_weight = sum(f.weight for f in frameworks)
        if total_weight > 0:
            for framework in frameworks:
                framework.weight = framework.weight / total_weight

        # Store for reference
        self.last_plan = frameworks

        # Log the plan
        logger.info(
            f"{self.symbol} - Framework Plan: {len(frameworks)} frameworks selected"
        )
        for f in frameworks:
            logger.info(
                f"  [{f.priority}] {f.type}: {f.weight*100:.1f}% weight | {f.reason}"
            )

        return frameworks

    def get_last_plan(self) -> Optional[List[FrameworkConfig]]:
        """
        Get the most recent framework plan

        Returns:
            Last plan, or None if no plans generated
        """
        return self.last_plan

    def get_framework_count(self) -> int:
        """
        Get the number of frameworks in the last plan

        Returns:
            Number of frameworks, or 0 if no plan exists
        """
        return len(self.last_plan) if self.last_plan else 0

    def get_total_weight(self) -> float:
        """
        Get the total weight of all frameworks (should always be 1.0)

        Returns:
            Total weight
        """
        if not self.last_plan:
            return 0.0
        return sum(f.weight for f in self.last_plan)

    def calculate_fading_growth_rates(
        self,
        historical_fcf_growth: float,
        company_stage: str,
        projection_years: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate fading growth rates using historical growth + YAML guardrails

        CRITICAL: Uses historical geometric mean FCF growth as starting point,
        capped at YAML ceiling. This prevents deflating high-growth stocks.

        Args:
            historical_fcf_growth: Geometric mean of historical FCF growth (decimal, e.g., 0.35 for 35%)
            company_stage: Company stage key (e.g., 'early_stage_saas', 'mid_stage_tech')
            projection_years: Number of years to project (default: from YAML)

        Returns:
            Dictionary with:
                - growth_rates: List of growth rates for each projection year
                - initial_growth: Starting growth rate (capped historical)
                - fade_to_growth: Target growth rate by final year
                - ceiling_applied: Whether historical was capped
                - projection_years: Number of projection years

        Example:
            >>> planner.calculate_fading_growth_rates(
            ...     historical_fcf_growth=0.35,  # 35% historical
            ...     company_stage='early_stage_saas'
            ... )
            {
                'growth_rates': [0.30, 0.25, 0.20, 0.15, 0.10],
                'initial_growth': 0.30,  # Capped from 35%
                'fade_to_growth': 0.10,
                'ceiling_applied': True,  # 35% > 30% ceiling
                'projection_years': 5
            }
        """
        assumptions = self.GROWTH_ASSUMPTIONS.get(company_stage)
        if not assumptions:
            logger.warning(
                f"{self.symbol} - Unknown company stage: {company_stage}, using 'mid_stage_tech'"
            )
            assumptions = self.GROWTH_ASSUMPTIONS['mid_stage_tech']

        # Step 1: Get YAML parameters
        ceiling = assumptions['fcf_growth_ceiling']
        fade_to = assumptions['fcf_growth_fade_to']
        years = projection_years or assumptions['projection_years']

        # Step 2: Cap initial growth at ceiling (prevents unrealistic projections)
        initial_growth = min(historical_fcf_growth, ceiling)
        ceiling_applied = historical_fcf_growth > ceiling

        # Step 3: Linear fade from initial to fade_to
        growth_rates = []
        for year in range(years):
            if years == 1:
                rate = initial_growth
            else:
                progress = year / (years - 1)
                rate = initial_growth - (initial_growth - fade_to) * progress
            growth_rates.append(rate)

        # Log the calculation
        if ceiling_applied:
            logger.info(
                f"{self.symbol} - Fading Growth: {historical_fcf_growth*100:.1f}% (historical) "
                f"capped at {ceiling*100:.1f}% → fades to {fade_to*100:.1f}% (Y{years})"
            )
        else:
            logger.info(
                f"{self.symbol} - Fading Growth: {initial_growth*100:.1f}% (historical) "
                f"→ fades to {fade_to*100:.1f}% (Y{years}) [no ceiling applied]"
            )

        return {
            'growth_rates': growth_rates,
            'initial_growth': initial_growth,
            'fade_to_growth': fade_to,
            'ceiling_applied': ceiling_applied,
            'historical_growth': historical_fcf_growth,
            'ceiling': ceiling,
            'projection_years': years,
            'terminal_growth': assumptions['terminal_growth']
        }

    def classify_company_stage(
        self,
        revenue_growth_pct: float,
        fcf_margin_pct: float
    ) -> str:
        """
        Classify company into growth stage for fading DCF assumptions

        Uses market cap, sector, revenue growth, and FCF margin to determine
        which fading growth profile to apply.

        Args:
            revenue_growth_pct: Revenue growth percentage (e.g., 28.6 for 28.6%)
            fcf_margin_pct: FCF margin percentage (e.g., 30.2 for 30.2%)

        Returns:
            Company stage key (e.g., 'mid_stage_tech', 'mega_cap_tech', 'mature_platform')

        Classification Logic:
            1. Mega-cap tech (>$200B): 'mega_cap_tech'
            2. High-growth SaaS (FCF margin >20%, revenue growth >30%): 'early_stage_saas'
            3. Mid-stage tech (FCF margin 10-25%, revenue growth 15-30%): 'mid_stage_tech'
            4. Mature platform (large-cap, FCF margin >15%, revenue growth 10-20%): 'mature_platform'
            5. Default: 'mature_platform'

        Example:
            >>> planner.classify_company_stage(revenue_growth_pct=28.6, fcf_margin_pct=30.2)
            'mid_stage_tech'  # ZS: $32B market cap, 28.6% growth, 30.2% FCF margin
        """
        # Priority 1: Mega-cap tech (>$200B)
        if self.market_cap_billions > self.SIZE_MEGA_CAP:
            logger.info(
                f"{self.symbol} - Classified as 'mega_cap_tech' "
                f"(market cap: ${self.market_cap_billions:.1f}B > $200B)"
            )
            return 'mega_cap_tech'

        # Priority 2: Early-stage SaaS (high growth, high margins, tech sector)
        if (
            self.sector in ['Technology', 'Communication Services'] and
            revenue_growth_pct > 30.0 and
            fcf_margin_pct > 20.0 and
            self.market_cap_billions < self.SIZE_LARGE_CAP  # < $10B
        ):
            logger.info(
                f"{self.symbol} - Classified as 'early_stage_saas' "
                f"(revenue growth {revenue_growth_pct:.1f}% >30%, FCF margin {fcf_margin_pct:.1f}% >20%)"
            )
            return 'early_stage_saas'

        # Priority 3: Mid-stage tech (moderate growth, good margins)
        if (
            self.sector in ['Technology', 'Communication Services'] and
            15.0 <= revenue_growth_pct <= 40.0 and
            10.0 <= fcf_margin_pct <= 35.0
        ):
            logger.info(
                f"{self.symbol} - Classified as 'mid_stage_tech' "
                f"(revenue growth {revenue_growth_pct:.1f}%, FCF margin {fcf_margin_pct:.1f}%)"
            )
            return 'mid_stage_tech'

        # Priority 4: Mature platform (large-cap, stable growth)
        if (
            self.market_cap_billions > self.SIZE_LARGE_CAP and  # >$10B
            10.0 <= revenue_growth_pct <= 25.0 and
            fcf_margin_pct > 10.0
        ):
            logger.info(
                f"{self.symbol} - Classified as 'mature_platform' "
                f"(market cap ${self.market_cap_billions:.1f}B, "
                f"revenue growth {revenue_growth_pct:.1f}%, FCF margin {fcf_margin_pct:.1f}%)"
            )
            return 'mature_platform'

        # Default: Mature platform (conservative assumptions)
        logger.info(
            f"{self.symbol} - Classified as 'mature_platform' (default) "
            f"(market cap ${self.market_cap_billions:.1f}B, sector {self.sector})"
        )
        return 'mature_platform'

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ValuationFrameworkPlanner(symbol='{self.symbol}', "
            f"sector='{self.sector}', "
            f"industry='{self.industry}', "
            f"market_cap=${self.market_cap_billions:.1f}B, "
            f"base_rate={self.base_terminal_growth*100:.2f}%)"
        )
