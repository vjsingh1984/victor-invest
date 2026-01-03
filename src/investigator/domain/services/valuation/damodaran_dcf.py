"""
Damodaran 3-Stage DCF Model - Comprehensive DCF with growth transitions.

Implements Aswath Damodaran's 3-stage DCF framework:
1. High Growth Phase (5 years): Company-specific high growth
2. Transition Phase (5 years): Linear decay to terminal growth
3. Terminal Phase: Stable growth at GDP rate

Features:
- Industry-specific cost of capital
- Monte Carlo sensitivity analysis
- Revenue bridge for negative FCF companies
- Confidence scoring based on data quality

Usage:
    from investigator.domain.services.valuation.damodaran_dcf import DamodaranDCFModel

    dcf = DamodaranDCFModel(company_profile)
    result = dcf.calculate(
        current_fcf=5e9,
        revenue_growth=0.15,
        fcf_margin=0.18,
        current_revenue=30e9,
        shares_outstanding=1e9
    )
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from investigator.domain.services.valuation.models.base import (
    BaseValuationModel,
    ModelDiagnostics,
    ModelNotApplicable,
    ValuationModelResult,
    ValuationOutput,
)
from investigator.domain.services.valuation.models.company_profile import CompanyProfile
from investigator.domain.services.valuation.cost_of_capital import (
    IndustryCostOfCapital,
    get_industry_cost_of_capital,
)

logger = logging.getLogger(__name__)


@dataclass
class DCFPhase:
    """Configuration for a DCF phase."""
    name: str
    years: int
    growth_rate_start: float
    growth_rate_end: float
    fcf_margin: float
    discount_rate: float


@dataclass
class DCFProjection:
    """Single year projection in DCF."""
    year: int
    phase: str
    revenue: float
    growth_rate: float
    fcf: float
    fcf_margin: float
    discount_factor: float
    present_value: float


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo sensitivity analysis."""
    mean_fair_value: float
    median_fair_value: float
    std_dev: float
    percentile_10: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    iterations: int


class DamodaranDCFModel(BaseValuationModel):
    """
    Damodaran 3-Stage DCF Model.

    Three phases:
    1. High Growth (5 years): Maintains current growth rate
    2. Transition (5 years): Linear decay to terminal growth
    3. Terminal: Perpetual growth at GDP rate

    For negative FCF companies, uses revenue bridge approach:
    - Projects revenue and target FCF margin
    - Estimates when company will reach profitability
    """

    model_name = "damodaran_dcf"
    methodology = "3-Stage DCF (Damodaran)"

    # Default phase configuration
    DEFAULT_HIGH_GROWTH_YEARS = 5
    DEFAULT_TRANSITION_YEARS = 5
    DEFAULT_TERMINAL_GROWTH = 0.025  # 2.5% (long-term GDP)

    def __init__(
        self,
        company_profile: CompanyProfile,
        cost_of_capital: Optional[IndustryCostOfCapital] = None
    ):
        """
        Initialize DCF model.

        Args:
            company_profile: Company profile with sector/industry info
            cost_of_capital: Optional cost of capital calculator
        """
        super().__init__(company_profile)
        self.coc = cost_of_capital or get_industry_cost_of_capital()

    def calculate(
        self,
        current_fcf: Optional[float] = None,
        revenue_growth: Optional[float] = None,
        fcf_margin: Optional[float] = None,
        current_revenue: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        debt_to_equity: float = 0.0,
        tax_rate: float = 0.21,
        high_growth_years: int = DEFAULT_HIGH_GROWTH_YEARS,
        transition_years: int = DEFAULT_TRANSITION_YEARS,
        terminal_growth: Optional[float] = None,
        run_monte_carlo: bool = False,
        monte_carlo_iterations: int = 1000,
        current_price: Optional[float] = None,
        **kwargs: Any
    ) -> ValuationOutput:
        """
        Calculate intrinsic value using 3-stage DCF.

        Args:
            current_fcf: Current TTM free cash flow
            revenue_growth: Expected revenue growth rate (as decimal)
            fcf_margin: FCF margin (as decimal)
            current_revenue: Current TTM revenue
            shares_outstanding: Shares outstanding
            debt_to_equity: Debt-to-equity ratio
            tax_rate: Corporate tax rate
            high_growth_years: Years in high growth phase
            transition_years: Years in transition phase
            terminal_growth: Terminal growth rate (defaults to industry-specific)
            run_monte_carlo: Whether to run Monte Carlo analysis
            monte_carlo_iterations: Number of Monte Carlo iterations
            current_price: Optional current price for context

        Returns:
            ValuationModelResult or ModelNotApplicable
        """
        # Validate inputs
        if shares_outstanding is None or shares_outstanding <= 0:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="Shares outstanding not available",
                diagnostics=ModelDiagnostics(flags=["missing_shares"])
            )

        # Determine if we can use FCF directly or need revenue bridge
        use_revenue_bridge = False
        if current_fcf is None or current_fcf <= 0:
            if current_revenue is None or current_revenue <= 0:
                return ModelNotApplicable(
                    model_name=self.model_name,
                    reason="Neither positive FCF nor revenue available",
                    diagnostics=ModelDiagnostics(flags=["missing_fcf_and_revenue"])
                )
            use_revenue_bridge = True
            logger.info(
                f"[{self.company_profile.symbol}] Using revenue bridge approach "
                f"(FCF={current_fcf}, Revenue={current_revenue/1e9:.1f}B)"
            )

        # Get cost of capital
        industry = self.company_profile.industry or self.company_profile.sector or 'default'
        coc_result = self.coc.calculate_wacc(
            industry=industry,
            debt_to_equity=debt_to_equity,
            tax_rate=tax_rate
        )

        # Get terminal growth rate
        if terminal_growth is None:
            country = kwargs.get('country', 'US')
            terminal_growth = self.coc.get_terminal_growth_rate(industry, country)

        # Normalize growth rate
        if revenue_growth is not None:
            growth_rate = revenue_growth if abs(revenue_growth) < 5 else revenue_growth / 100
        else:
            # Default to moderate growth
            growth_rate = 0.10

        # Normalize FCF margin
        if fcf_margin is not None:
            margin = fcf_margin if abs(fcf_margin) < 1 else fcf_margin / 100
        else:
            margin = 0.15  # Default target margin

        # Calculate DCF
        if use_revenue_bridge:
            result = self._calculate_revenue_bridge_dcf(
                current_revenue=current_revenue,
                growth_rate=growth_rate,
                target_fcf_margin=margin,
                wacc=coc_result.wacc,
                terminal_growth=terminal_growth,
                high_growth_years=high_growth_years,
                transition_years=transition_years,
                shares_outstanding=shares_outstanding
            )
        else:
            result = self._calculate_fcf_dcf(
                current_fcf=current_fcf,
                growth_rate=growth_rate,
                fcf_margin=margin,
                wacc=coc_result.wacc,
                terminal_growth=terminal_growth,
                high_growth_years=high_growth_years,
                transition_years=transition_years,
                shares_outstanding=shares_outstanding
            )

        if result is None:
            return ModelNotApplicable(
                model_name=self.model_name,
                reason="DCF calculation failed",
                diagnostics=ModelDiagnostics(flags=["calculation_error"])
            )

        fair_value, projections, enterprise_value, terminal_value = result

        # Run Monte Carlo if requested
        monte_carlo = None
        if run_monte_carlo:
            monte_carlo = self._run_monte_carlo(
                base_fair_value=fair_value,
                growth_rate=growth_rate,
                fcf_margin=margin,
                wacc=coc_result.wacc,
                current_fcf=current_fcf,
                current_revenue=current_revenue,
                shares_outstanding=shares_outstanding,
                high_growth_years=high_growth_years,
                transition_years=transition_years,
                terminal_growth=terminal_growth,
                iterations=monte_carlo_iterations,
                use_revenue_bridge=use_revenue_bridge
            )

        # Calculate upside/downside
        upside_potential = None
        if current_price and current_price > 0:
            upside_potential = (fair_value / current_price - 1) * 100

        # Estimate confidence
        confidence = self._calculate_confidence(
            use_revenue_bridge=use_revenue_bridge,
            growth_rate=growth_rate,
            fcf_margin=margin,
            monte_carlo=monte_carlo
        )

        # Build assumptions
        assumptions = {
            'wacc': coc_result.wacc,
            'cost_of_equity': coc_result.cost_of_equity,
            'levered_beta': coc_result.levered_beta,
            'unlevered_beta': coc_result.unlevered_beta,
            'risk_free_rate': coc_result.risk_free_rate,
            'high_growth_years': high_growth_years,
            'transition_years': transition_years,
            'terminal_growth': terminal_growth,
            'high_growth_rate': growth_rate,
            'fcf_margin': margin,
            'use_revenue_bridge': use_revenue_bridge,
        }

        # Build metadata
        metadata = {
            'enterprise_value': enterprise_value,
            'terminal_value': terminal_value,
            'terminal_value_pct_of_ev': terminal_value / enterprise_value * 100 if enterprise_value > 0 else 0,
            'projection_summary': self._summarize_projections(projections),
        }

        if monte_carlo:
            metadata['monte_carlo'] = {
                'mean': monte_carlo.mean_fair_value,
                'median': monte_carlo.median_fair_value,
                'std_dev': monte_carlo.std_dev,
                'range_10_90': (monte_carlo.percentile_10, monte_carlo.percentile_90),
                'iterations': monte_carlo.iterations,
            }

        if upside_potential is not None:
            metadata['upside_potential_pct'] = upside_potential
            metadata['current_price'] = current_price

        logger.info(
            f"[{self.company_profile.symbol}] Damodaran DCF: "
            f"Fair Value=${fair_value:.2f}, EV=${enterprise_value/1e9:.1f}B, "
            f"WACC={coc_result.wacc:.1%}, Terminal={terminal_growth:.1%}"
        )

        return ValuationModelResult(
            model_name=self.model_name,
            fair_value=fair_value,
            confidence_score=confidence,
            methodology=self.methodology,
            assumptions=assumptions,
            diagnostics=ModelDiagnostics(
                data_quality_score=0.8 if not use_revenue_bridge else 0.6,
                flags=["revenue_bridge"] if use_revenue_bridge else []
            ),
            metadata=metadata
        )

    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        """Estimate confidence for DCF model."""
        return self._calculate_confidence(
            use_revenue_bridge=raw_output.get('use_revenue_bridge', False),
            growth_rate=raw_output.get('high_growth_rate', 0.10),
            fcf_margin=raw_output.get('fcf_margin', 0.15),
            monte_carlo=None
        )

    def _calculate_fcf_dcf(
        self,
        current_fcf: float,
        growth_rate: float,
        fcf_margin: float,
        wacc: float,
        terminal_growth: float,
        high_growth_years: int,
        transition_years: int,
        shares_outstanding: float
    ) -> Optional[Tuple[float, List[DCFProjection], float, float]]:
        """Calculate DCF using FCF projections."""
        projections = []
        pv_sum = 0
        fcf = current_fcf
        year = 0

        # Phase 1: High Growth
        for i in range(high_growth_years):
            year += 1
            fcf = fcf * (1 + growth_rate)
            discount_factor = 1 / ((1 + wacc) ** year)
            pv = fcf * discount_factor
            pv_sum += pv

            projections.append(DCFProjection(
                year=year,
                phase='high_growth',
                revenue=0,  # Not tracked in FCF approach
                growth_rate=growth_rate,
                fcf=fcf,
                fcf_margin=fcf_margin,
                discount_factor=discount_factor,
                present_value=pv
            ))

        # Phase 2: Transition
        for i in range(transition_years):
            year += 1
            # Linear decay from high growth to terminal growth
            blend = (i + 1) / transition_years
            current_growth = growth_rate * (1 - blend) + terminal_growth * blend

            fcf = fcf * (1 + current_growth)
            discount_factor = 1 / ((1 + wacc) ** year)
            pv = fcf * discount_factor
            pv_sum += pv

            projections.append(DCFProjection(
                year=year,
                phase='transition',
                revenue=0,
                growth_rate=current_growth,
                fcf=fcf,
                fcf_margin=fcf_margin,
                discount_factor=discount_factor,
                present_value=pv
            ))

        # Terminal Value (Gordon Growth Model)
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        terminal_pv = terminal_value / ((1 + wacc) ** year)

        # Enterprise Value
        enterprise_value = pv_sum + terminal_pv

        # Fair Value per share
        fair_value = enterprise_value / shares_outstanding

        return (fair_value, projections, enterprise_value, terminal_value)

    def _calculate_revenue_bridge_dcf(
        self,
        current_revenue: float,
        growth_rate: float,
        target_fcf_margin: float,
        wacc: float,
        terminal_growth: float,
        high_growth_years: int,
        transition_years: int,
        shares_outstanding: float
    ) -> Optional[Tuple[float, List[DCFProjection], float, float]]:
        """
        Calculate DCF using revenue bridge for negative FCF companies.

        Assumes company reaches target FCF margin by end of high growth phase.
        """
        projections = []
        pv_sum = 0
        revenue = current_revenue
        year = 0

        # Start with negative/low margin, grow to target
        starting_margin = -0.05  # -5% starting margin
        margin = starting_margin

        # Phase 1: High Growth (margin improvement)
        for i in range(high_growth_years):
            year += 1
            revenue = revenue * (1 + growth_rate)

            # Linear margin improvement
            blend = (i + 1) / high_growth_years
            margin = starting_margin * (1 - blend) + target_fcf_margin * blend

            fcf = revenue * margin
            discount_factor = 1 / ((1 + wacc) ** year)
            pv = fcf * discount_factor
            pv_sum += pv

            projections.append(DCFProjection(
                year=year,
                phase='high_growth',
                revenue=revenue,
                growth_rate=growth_rate,
                fcf=fcf,
                fcf_margin=margin,
                discount_factor=discount_factor,
                present_value=pv
            ))

        # Phase 2: Transition
        for i in range(transition_years):
            year += 1
            blend = (i + 1) / transition_years
            current_growth = growth_rate * (1 - blend) + terminal_growth * blend

            revenue = revenue * (1 + current_growth)
            fcf = revenue * target_fcf_margin  # Maintain target margin

            discount_factor = 1 / ((1 + wacc) ** year)
            pv = fcf * discount_factor
            pv_sum += pv

            projections.append(DCFProjection(
                year=year,
                phase='transition',
                revenue=revenue,
                growth_rate=current_growth,
                fcf=fcf,
                fcf_margin=target_fcf_margin,
                discount_factor=discount_factor,
                present_value=pv
            ))

        # Terminal Value
        terminal_revenue = revenue * (1 + terminal_growth)
        terminal_fcf = terminal_revenue * target_fcf_margin
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        terminal_pv = terminal_value / ((1 + wacc) ** year)

        # Enterprise Value
        enterprise_value = pv_sum + terminal_pv

        # Fair Value per share
        fair_value = max(0, enterprise_value / shares_outstanding)

        return (fair_value, projections, enterprise_value, terminal_value)

    def _run_monte_carlo(
        self,
        base_fair_value: float,
        growth_rate: float,
        fcf_margin: float,
        wacc: float,
        current_fcf: Optional[float],
        current_revenue: Optional[float],
        shares_outstanding: float,
        high_growth_years: int,
        transition_years: int,
        terminal_growth: float,
        iterations: int,
        use_revenue_bridge: bool
    ) -> MonteCarloResult:
        """Run Monte Carlo sensitivity analysis."""
        fair_values = []

        for _ in range(iterations):
            # Random variation in key inputs (normal distribution)
            growth_var = random.gauss(0, 0.03)  # ±3% std dev
            margin_var = random.gauss(0, 0.02)  # ±2% std dev
            wacc_var = random.gauss(0, 0.01)    # ±1% std dev

            sim_growth = max(0, growth_rate + growth_var)
            sim_margin = max(0.01, fcf_margin + margin_var)
            sim_wacc = max(0.05, wacc + wacc_var)

            if use_revenue_bridge:
                result = self._calculate_revenue_bridge_dcf(
                    current_revenue=current_revenue,
                    growth_rate=sim_growth,
                    target_fcf_margin=sim_margin,
                    wacc=sim_wacc,
                    terminal_growth=terminal_growth,
                    high_growth_years=high_growth_years,
                    transition_years=transition_years,
                    shares_outstanding=shares_outstanding
                )
            else:
                result = self._calculate_fcf_dcf(
                    current_fcf=current_fcf,
                    growth_rate=sim_growth,
                    fcf_margin=sim_margin,
                    wacc=sim_wacc,
                    terminal_growth=terminal_growth,
                    high_growth_years=high_growth_years,
                    transition_years=transition_years,
                    shares_outstanding=shares_outstanding
                )

            if result:
                fair_values.append(result[0])

        if not fair_values:
            return MonteCarloResult(
                mean_fair_value=base_fair_value,
                median_fair_value=base_fair_value,
                std_dev=0,
                percentile_10=base_fair_value,
                percentile_25=base_fair_value,
                percentile_75=base_fair_value,
                percentile_90=base_fair_value,
                iterations=0
            )

        fair_values.sort()
        n = len(fair_values)

        mean_fv = sum(fair_values) / n
        median_fv = fair_values[n // 2]
        variance = sum((v - mean_fv) ** 2 for v in fair_values) / n
        std_dev = variance ** 0.5

        return MonteCarloResult(
            mean_fair_value=mean_fv,
            median_fair_value=median_fv,
            std_dev=std_dev,
            percentile_10=fair_values[int(n * 0.10)],
            percentile_25=fair_values[int(n * 0.25)],
            percentile_75=fair_values[int(n * 0.75)],
            percentile_90=fair_values[int(n * 0.90)],
            iterations=n
        )

    def _summarize_projections(
        self,
        projections: List[DCFProjection]
    ) -> Dict[str, Any]:
        """Summarize projections for metadata."""
        if not projections:
            return {}

        high_growth = [p for p in projections if p.phase == 'high_growth']
        transition = [p for p in projections if p.phase == 'transition']

        return {
            'total_years': len(projections),
            'high_growth_years': len(high_growth),
            'transition_years': len(transition),
            'final_fcf': projections[-1].fcf if projections else 0,
            'total_pv_of_fcf': sum(p.present_value for p in projections),
        }

    def _calculate_confidence(
        self,
        use_revenue_bridge: bool,
        growth_rate: float,
        fcf_margin: float,
        monte_carlo: Optional[MonteCarloResult]
    ) -> float:
        """Calculate confidence score for DCF."""
        base = 0.70

        # Revenue bridge is less reliable
        if use_revenue_bridge:
            base -= 0.15

        # Extreme growth rates reduce confidence
        if growth_rate > 0.40:
            base -= 0.10
        elif growth_rate < 0.05:
            base -= 0.05

        # Very high margins are harder to sustain
        if fcf_margin > 0.30:
            base -= 0.05

        # Monte Carlo dispersion affects confidence
        if monte_carlo and monte_carlo.std_dev > 0:
            cv = monte_carlo.std_dev / monte_carlo.mean_fair_value
            if cv > 0.30:
                base -= 0.10
            elif cv < 0.15:
                base += 0.05

        return max(0.30, min(0.90, base))
