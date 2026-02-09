"""
Parallel Valuation Orchestrator

Executes multiple valuation frameworks concurrently, sharing unified terminal growth
calculations and computing a blended fair value.

This replaces the sequential SectorValuationRouter approach with parallel execution,
ensuring all frameworks use identical assumptions and reducing latency.

Created: 2025-11-12
Author: Claude Code
Updated: 2025-02-09 - Integrated with existing valuation models
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from investigator.domain.services.terminal_growth_calculator import TerminalGrowthCalculator
from investigator.domain.services.valuation_framework_planner import FrameworkConfig, ValuationFrameworkPlanner

logger = logging.getLogger(__name__)


@dataclass
class FrameworkResult:
    """Result from a single valuation framework execution"""

    framework_type: str
    fair_value: Optional[float]
    confidence: float  # 0.0-1.0
    metrics: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class BlendedValuationResult:
    """Blended result from multiple valuation frameworks"""

    blended_fair_value: float
    current_price: float
    upside_pct: float
    framework_results: List[FrameworkResult]
    weights_used: Dict[str, float]
    terminal_growth_info: Dict[str, Any]
    execution_summary: Dict[str, Any]


class ParallelValuationOrchestrator:
    """
    Orchestrates parallel execution of multiple valuation frameworks

    Key Features:
    1. Executes all frameworks concurrently using asyncio.gather()
    2. Shares unified terminal growth across all DCF calculations
    3. Computes weighted blended fair value
    4. Handles framework failures gracefully (reweights remaining frameworks)
    5. Tracks execution metrics for performance monitoring

    Example:
        >>> orchestrator = ParallelValuationOrchestrator(
        ...     symbol='ZS',
        ...     sector='Technology',
        ...     current_price=317.08
        ... )
        >>> result = await orchestrator.execute_valuation(
        ...     frameworks=[...],  # From ValuationFrameworkPlanner
        ...     rule_of_40_score=58.8,
        ...     revenue_growth_pct=28.6,
        ...     fcf_margin_pct=30.2,
        ...     financials={...}
        ... )
        >>> result.blended_fair_value
        291.36
    """

    def __init__(self, symbol: str, sector: str, current_price: float, base_terminal_growth: float = 0.035):
        """
        Initialize parallel valuation orchestrator

        Args:
            symbol: Stock symbol
            sector: Company sector
            current_price: Current stock price
            base_terminal_growth: Base terminal growth rate (default: 3.5%)
        """
        self.symbol = symbol
        self.sector = sector
        self.current_price = current_price
        self.base_terminal_growth = base_terminal_growth

        # Create unified terminal growth calculator
        self.terminal_growth_calc = TerminalGrowthCalculator(
            symbol=symbol, sector=sector, base_terminal_growth=base_terminal_growth
        )

    async def execute_valuation(
        self,
        frameworks: List[FrameworkConfig],
        rule_of_40_score: float,
        revenue_growth_pct: float,
        fcf_margin_pct: float,
        financials: Dict[str, Any],
        dcf_calculator: Any = None,  # DCFValuation instance
        ggm_calculator: Any = None,  # GordonGrowthModel instance
    ) -> BlendedValuationResult:
        """
        Execute all valuation frameworks in parallel

        This is the SINGLE EXECUTION STEP that runs after planning.
        All frameworks execute concurrently with shared terminal growth.

        Args:
            frameworks: List of framework configurations from planner
            rule_of_40_score: Rule of 40 score for terminal growth calculation
            revenue_growth_pct: Revenue growth percentage
            fcf_margin_pct: FCF margin percentage
            financials: Company financials dictionary
            dcf_calculator: DCFValuation instance (if DCF frameworks included)
            ggm_calculator: GordonGrowthModel instance (if GGM framework included)

        Returns:
            BlendedValuationResult with fair value and execution details
        """
        start_time = datetime.now()

        # Step 1: Calculate unified terminal growth (SINGLE SOURCE OF TRUTH)
        terminal_growth_result = self.terminal_growth_calc.calculate_terminal_growth(
            rule_of_40_score=rule_of_40_score, revenue_growth_pct=revenue_growth_pct, fcf_margin_pct=fcf_margin_pct
        )

        logger.info(
            f"{self.symbol} - Terminal Growth (unified): "
            f"{terminal_growth_result['terminal_growth_rate']*100:.2f}% "
            f"[{terminal_growth_result['tier']}]"
        )

        # Step 2: Execute all frameworks in parallel
        tasks = []
        for framework in frameworks:
            task = self._execute_framework(
                framework=framework,
                terminal_growth_rate=terminal_growth_result["terminal_growth_rate"],
                financials=financials,
                dcf_calculator=dcf_calculator,
                ggm_calculator=ggm_calculator,
            )
            tasks.append(task)

        # Run all frameworks concurrently
        logger.info(f"{self.symbol} - Executing {len(frameworks)} frameworks in parallel...")
        framework_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Process results and handle failures
        valid_results = []
        failed_frameworks = []

        for i, result in enumerate(framework_results):
            if isinstance(result, Exception):
                # Framework execution raised exception
                logger.warning(f"{self.symbol} - {frameworks[i].type} failed: {result}")
                failed_frameworks.append(frameworks[i].type)
            elif result.error:
                # Framework returned error
                logger.warning(f"{self.symbol} - {result.framework_type} error: {result.error}")
                failed_frameworks.append(result.framework_type)
            elif result.fair_value is None or result.fair_value <= 0:
                # Framework returned invalid value
                logger.warning(f"{self.symbol} - {result.framework_type} invalid value: {result.fair_value}")
                failed_frameworks.append(result.framework_type)
            else:
                # Framework succeeded
                valid_results.append((frameworks[i], result))

        # Step 4: Reweight remaining frameworks
        if not valid_results:
            raise ValueError(f"{self.symbol} - All {len(frameworks)} frameworks failed: {failed_frameworks}")

        total_weight = sum(f.weight for f, _ in valid_results)
        normalized_weights = {f.type: f.weight / total_weight for f, _ in valid_results}

        # Step 5: Compute weighted blended fair value
        blended_fair_value = sum(
            result.fair_value * normalized_weights[framework.type] for framework, result in valid_results
        )

        upside_pct = ((blended_fair_value - self.current_price) / self.current_price) * 100

        # Step 6: Create execution summary
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        execution_summary = {
            "total_frameworks": len(frameworks),
            "successful_frameworks": len(valid_results),
            "failed_frameworks": len(failed_frameworks),
            "failed_framework_names": failed_frameworks,
            "execution_time_ms": execution_time_ms,
            "parallel_execution": True,
        }

        # Log summary
        logger.info(
            f"{self.symbol} - Blended Fair Value: ${blended_fair_value:.2f} "
            f"(Current: ${self.current_price:.2f}, Upside: {upside_pct:+.1f}%)"
        )
        logger.info(
            f"{self.symbol} - Execution: {len(valid_results)}/{len(frameworks)} "
            f"frameworks succeeded in {execution_time_ms:.0f}ms"
        )

        # Return comprehensive result
        return BlendedValuationResult(
            blended_fair_value=blended_fair_value,
            current_price=self.current_price,
            upside_pct=upside_pct,
            framework_results=[r for _, r in valid_results],
            weights_used=normalized_weights,
            terminal_growth_info=terminal_growth_result,
            execution_summary=execution_summary,
        )

    def _calculate_framework_confidence(
        self,
        framework_type: str,
        fair_value: float,
        financials: Dict[str, Any],
        terminal_growth_rate: float = None
    ) -> float:
        """
        Calculate confidence score for a valuation framework (0.0 to 1.0)

        Confidence factors:
        - Data quality (completeness of financials)
        - Valuation spread (fair_value vs current price reasonableness)
        - Framework applicability (sector-specific suitability)
        - Terminal growth tier (higher tier = higher confidence)

        Args:
            framework_type: Type of valuation framework
            fair_value: Calculated fair value
            financials: Company financials
            terminal_growth_rate: Terminal growth rate (if applicable)

        Returns:
            Confidence score from 0.0 (low) to 1.0 (high)
        """
        confidence_factors = []

        # Factor 1: Data completeness (30% weight)
        required_fields = ['revenue', 'net_income', 'free_cash_flow', 'total_assets']
        available_fields = sum(1 for field in required_fields if financials.get(field) is not None)
        data_completeness = available_fields / len(required_fields)
        confidence_factors.append(0.3 * data_completeness)

        # Factor 2: Valuation reasonableness (25% weight)
        # Fair value within 10% to 300% of current price is reasonable
        price_ratio = fair_value / self.current_price
        if 0.1 <= price_ratio <= 3.0:
            reasonableness = 1.0
        elif price_ratio < 0.01 or price_ratio > 10.0:
            reasonableness = 0.3
        else:
            reasonableness = 0.7
        confidence_factors.append(0.25 * reasonableness)

        # Factor 3: Framework applicability (25% weight)
        # DCF and GGM are more reliable for dividend-paying or cash-positive companies
        if framework_type in [ValuationFrameworkPlanner.FRAMEWORK_DCF_GROWTH,
                             ValuationFrameworkPlanner.FRAMEWORK_DCF_FADING]:
            # DCF works better for companies with positive FCF
            if financials.get('free_cash_flow', 0) > 0:
                applicability = 0.9
            else:
                applicability = 0.5
        elif framework_type == ValuationFrameworkPlanner.FRAMEWORK_GORDON_GROWTH:
            # GGM requires dividends
            applicability = 0.8 if financials.get('dividend_per_share', 0) > 0 else 0.3
        else:
            # P/E, PEG, P/S, EV/EBITDA are generally applicable
            applicability = 0.8
        confidence_factors.append(0.25 * applicability)

        # Factor 4: Terminal growth tier (20% weight) - if applicable
        if terminal_growth_rate is not None:
            # Higher terminal growth tiers generally mean better-understood companies
            if terminal_growth_rate >= 0.04:
                tier_confidence = 0.9
            elif terminal_growth_rate >= 0.03:
                tier_confidence = 0.8
            elif terminal_growth_rate >= 0.02:
                tier_confidence = 0.7
            else:
                tier_confidence = 0.6
            confidence_factors.append(0.2 * tier_confidence)
        else:
            # No terminal growth factor for non-DCF frameworks
            confidence_factors.append(0.2 * 0.7)  # Average confidence

        return min(1.0, max(0.0, sum(confidence_factors)))

    async def _execute_framework(
        self,
        framework: FrameworkConfig,
        terminal_growth_rate: float,
        financials: Dict[str, Any],
        dcf_calculator: Any = None,
        ggm_calculator: Any = None,
    ) -> FrameworkResult:
        """
        Execute a single valuation framework

        Args:
            framework: Framework configuration
            terminal_growth_rate: Unified terminal growth rate (from calculator)
            financials: Company financials
            dcf_calculator: DCF calculator instance
            ggm_calculator: GGM calculator instance

        Returns:
            FrameworkResult with fair value and metrics
        """
        start_time = datetime.now()

        try:
            # Route to appropriate calculator based on framework type
            if framework.type == ValuationFrameworkPlanner.FRAMEWORK_DCF_GROWTH:
                fair_value, metrics = await self._execute_dcf_growth(
                    terminal_growth_rate=terminal_growth_rate,
                    financials=financials,
                    params=framework.params,
                    dcf_calculator=dcf_calculator,
                )

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_DCF_FADING:
                fair_value, metrics = await self._execute_dcf_fading(
                    terminal_growth_rate=terminal_growth_rate,
                    financials=financials,
                    params=framework.params,
                    dcf_calculator=dcf_calculator,
                )

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_PE_RATIO:
                fair_value, metrics = await self._execute_pe_ratio(
                    financials=financials, params=framework.params
                )

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_EV_EBITDA:
                fair_value, metrics = await self._execute_ev_ebitda(financials=financials, params=framework.params)

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_PS_RATIO:
                fair_value, metrics = await self._execute_ps_ratio(financials=financials, params=framework.params)

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_PEG_RATIO:
                fair_value, metrics = await self._execute_peg_ratio(
                    financials=financials, params=framework.params
                )

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_GORDON_GROWTH:
                fair_value, metrics = await self._execute_gordon_growth(
                    terminal_growth_rate=terminal_growth_rate,
                    financials=financials,
                    params=framework.params,
                    ggm_calculator=ggm_calculator,
                )

            else:
                raise ValueError(f"Unknown framework type: {framework.type}")

            # Calculate confidence score
            confidence = self._calculate_framework_confidence(
                framework_type=framework.type,
                fair_value=fair_value,
                financials=financials,
                terminal_growth_rate=terminal_growth_rate if framework.type in [
                    ValuationFrameworkPlanner.FRAMEWORK_DCF_GROWTH,
                    ValuationFrameworkPlanner.FRAMEWORK_DCF_FADING,
                    ValuationFrameworkPlanner.FRAMEWORK_GORDON_GROWTH,
                ] else None
            )

            # Calculate execution time
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            logger.info(
                f"{self.symbol} - {framework.type}: ${fair_value:.2f} "
                f"(weight: {framework.weight*100:.1f}%, confidence: {confidence:.2f}, {execution_time_ms:.0f}ms)"
            )

            return FrameworkResult(
                framework_type=framework.type,
                fair_value=fair_value,
                confidence=confidence,
                metrics=metrics,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            logger.error(f"{self.symbol} - {framework.type} execution failed: {e}", exc_info=True)
            return FrameworkResult(
                framework_type=framework.type, fair_value=None, confidence=0.0, metrics={}, error=str(e)
            )

    # Framework execution methods - Integrated with existing valuation models

    async def _execute_dcf_growth(
        self, terminal_growth_rate: float, financials: Dict[str, Any], params: Dict[str, Any], dcf_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """
        Execute DCF with growth assumptions

        Integrated with existing DCFValuation.calculate_dcf_valuation()
        """
        if dcf_calculator is None:
            logger.warning(f"{self.symbol} - DCF calculator not provided, using placeholder")
            # Fallback: Simple DCF approximation
            fcf = financials.get('free_cash_flow', 0)
            if fcf <= 0:
                return 0.0, {"method": "dcf_growth", "error": "Negative FCF", "terminal_growth": terminal_growth_rate}

            # Simple 5-year projection with terminal value
            discount_rate = financials.get('wacc', 0.10)
            growth_rate = financials.get('revenue_growth', 0.10)  # Use historical growth

            pv_fcf = sum(fcf * ((1 + growth_rate) ** i) / ((1 + discount_rate) ** (i + 1)) for i in range(5))
            terminal_value = (fcf * (1 + growth_rate) ** 5) * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 5)

            fair_value = (pv_fcf + pv_terminal) / financials.get('shares_outstanding', 1)
            return fair_value, {"method": "dcf_growth", "terminal_growth": terminal_growth_rate, "fallback": True}

        # Use actual DCF calculator
        try:
            result = dcf_calculator.calculate_dcf_valuation(terminal_growth_rate=terminal_growth_rate)
            fair_value = result.get('fair_value_per_share', 0)
            return fair_value, {
                "method": "dcf_growth",
                "terminal_growth": terminal_growth_rate,
                "wacc": result.get('wacc', 0),
                "projection_years": result.get('projection_years', 5)
            }
        except Exception as e:
            logger.error(f"{self.symbol} - DCF calculation failed: {e}")
            return 0.0, {"method": "dcf_growth", "error": str(e), "terminal_growth": terminal_growth_rate}

    async def _execute_dcf_fading(
        self, terminal_growth_rate: float, financials: Dict[str, Any], params: Dict[str, Any], dcf_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """
        Execute DCF with fading growth assumptions

        Uses existing DCF calculator with adjusted terminal growth for fading
        """
        if dcf_calculator is None:
            logger.warning(f"{self.symbol} - DCF calculator not provided, using placeholder")
            # Fallback: Lower growth assumption for fading model
            fading_growth = terminal_growth_rate * 0.7  # 30% lower than base
            fcf = financials.get('free_cash_flow', 0)

            if fcf <= 0:
                return 0.0, {"method": "dcf_fading", "error": "Negative FCF", "terminal_growth": fading_growth}

            # Apply fading growth (declining from current to terminal)
            discount_rate = financials.get('wacc', 0.10)
            current_growth = financials.get('revenue_growth', 0.15)

            pv_fcf = 0
            for i in range(5):
                # Fade from current growth to terminal growth
                year_growth = current_growth - (current_growth - fading_growth) * (i / 4)
                pv_fcf += fcf * ((1 + year_growth) ** i) / ((1 + discount_rate) ** (i + 1))

            terminal_value = (fcf * (1 + fading_growth) ** 5) * (1 + fading_growth) / (discount_rate - fading_growth)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 5)

            fair_value = (pv_fcf + pv_terminal) / financials.get('shares_outstanding', 1)
            return fair_value, {"method": "dcf_fading", "terminal_growth": fading_growth, "fading_applied": True}

        # Use DCF calculator with lower terminal growth for fading model
        try:
            fading_growth = terminal_growth_rate * 0.7  # Conservative estimate
            result = dcf_calculator.calculate_dcf_valuation(terminal_growth_rate=fading_growth)
            fair_value = result.get('fair_value_per_share', 0)
            return fair_value, {
                "method": "dcf_fading",
                "terminal_growth": fading_growth,
                "fading_applied": True,
                "wacc": result.get('wacc', 0)
            }
        except Exception as e:
            logger.error(f"{self.symbol} - DCF fading calculation failed: {e}")
            return 0.0, {"method": "dcf_fading", "error": str(e)}

    async def _execute_pe_ratio(
        self, financials: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """Execute P/E ratio valuation"""
        eps = financials.get('earnings_per_share', 0)
        sector = params.get('sector', self.sector)

        if eps <= 0:
            return 0.0, {"method": "pe_ratio", "error": "Negative or zero EPS"}

        # Get sector-specific P/E multiple
        sector_pe_multiples = {
            'Technology': 25.0,
            'Healthcare': 22.0,
            'Consumer Cyclical': 20.0,
            'Consumer Defensive': 18.0,
            'Industrials': 18.0,
            'Financials': 12.0,
            'Energy': 12.0,
            'Utilities': 15.0,
            'Real Estate': 16.0,
            'Communication Services': 20.0,
        }

        pe_multiple = sector_pe_multiples.get(sector, 18.0)

        # Adjust for growth (higher growth = higher P/E)
        growth_rate = financials.get('earnings_growth', 0.10)
        if growth_rate > 0.20:
            pe_multiple *= 1.3
        elif growth_rate > 0.10:
            pe_multiple *= 1.1

        fair_value = eps * pe_multiple
        return fair_value, {
            "method": "pe_ratio",
            "pe_multiple": pe_multiple,
            "eps": eps,
            "sector": sector
        }

    async def _execute_ev_ebitda(
        self, financials: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """Execute EV/EBITDA valuation"""
        ebitda = financials.get('ebitda', 0)
        net_debt = financials.get('net_debt', 0)
        sector = params.get('sector', self.sector)

        if ebitda <= 0:
            return 0.0, {"method": "ev_ebitda", "error": "Negative or zero EBITDA"}

        # Get sector-specific EV/EBITDA multiple
        sector_ev_multiples = {
            'Technology': 18.0,
            'Healthcare': 14.0,
            'Consumer Cyclical': 12.0,
            'Consumer Defensive': 10.0,
            'Industrials': 11.0,
            'Financials': 0,  # Not applicable for banks
            'Energy': 8.0,
            'Utilities': 10.0,
            'Real Estate': 0,  # Use FFO instead
            'Communication Services': 10.0,
        }

        ev_multiple = sector_ev_multiples.get(sector, 12.0)

        if ev_multiple <= 0:
            return 0.0, {"method": "ev_ebitda", "error": f"EV/EBITDA not applicable for {sector}"}

        enterprise_value = ebitda * ev_multiple
        equity_value = enterprise_value - net_debt
        fair_value = equity_value / financials.get('shares_outstanding', 1)

        return fair_value, {
            "method": "ev_ebitda",
            "ev_multiple": ev_multiple,
            "ebitda": ebitda,
            "enterprise_value": enterprise_value,
            "sector": sector
        }

    async def _execute_ps_ratio(
        self, financials: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """Execute P/S ratio valuation"""
        revenue_per_share = financials.get('revenue_per_share', 0)
        sector = params.get('sector', self.sector)

        if revenue_per_share <= 0:
            # Fallback to revenue / shares
            total_revenue = financials.get('revenue', 0)
            shares = financials.get('shares_outstanding', 1)
            if total_revenue <= 0 or shares <= 0:
                return 0.0, {"method": "ps_ratio", "error": "No revenue data"}
            revenue_per_share = total_revenue / shares

        # Get sector-specific P/S multiple
        sector_ps_multiples = {
            'Technology': 8.0,
            'Healthcare': 5.0,
            'Consumer Cyclical': 2.0,
            'Consumer Defensive': 2.0,
            'Industrials': 2.0,
            'Financials': 1.5,
            'Energy': 1.5,
            'Utilities': 2.0,
            'Real Estate': 5.0,
            'Communication Services': 2.5,
        }

        ps_multiple = sector_ps_multiples.get(sector, 2.5)

        # Adjust for growth
        growth_rate = financials.get('revenue_growth', 0.10)
        if growth_rate > 0.20:
            ps_multiple *= 1.5
        elif growth_rate > 0.10:
            ps_multiple *= 1.2

        fair_value = revenue_per_share * ps_multiple
        return fair_value, {
            "method": "ps_ratio",
            "ps_multiple": ps_multiple,
            "revenue_per_share": revenue_per_share,
            "sector": sector
        }

    async def _execute_peg_ratio(
        self, financials: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """Execute PEG ratio valuation"""
        eps = financials.get('earnings_per_share', 0)
        growth_rate = financials.get('earnings_growth', financials.get('revenue_growth', 0.10))

        if eps <= 0 or growth_rate <= 0:
            return 0.0, {"method": "peg_ratio", "error": "Need positive EPS and growth rate"}

        # PEG ratio = P/E / Growth Rate
        # Fair value = EPS * PEG * Growth Rate = EPS * PEG_multiple
        # Industry-standard PEG = 1.0 means fairly valued

        # Get sector-specific PEG multiple (typically 0.8 - 1.5)
        sector_peg_multiples = {
            'Technology': 1.5,
            'Healthcare': 1.3,
            'Consumer Cyclical': 1.1,
            'Consumer Defensive': 1.0,
            'Industrials': 1.0,
            'Financials': 0.9,
            'Energy': 0.9,
            'Utilities': 1.0,
            'Real Estate': 1.1,
            'Communication Services': 1.2,
        }

        peg_multiple = sector_peg_multiples.get(params.get('sector', self.sector), 1.0)

        # Adjust for quality (Rule of 40 companies can command higher PEG)
        rule_of_40 = financials.get('rule_of_40', 0)
        if rule_of_40 > 60:
            peg_multiple *= 1.3
        elif rule_of_40 > 40:
            peg_multiple *= 1.1

        fair_value = eps * peg_multiple * growth_rate

        return fair_value, {
            "method": "peg_ratio",
            "peg_multiple": peg_multiple,
            "eps": eps,
            "growth_rate": growth_rate,
            "peg_ratio": (eps * peg_multiple * growth_rate) / eps if eps > 0 else 0
        }

    async def _execute_gordon_growth(
        self, terminal_growth_rate: float, financials: Dict[str, Any], params: Dict[str, Any], ggm_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """Execute Gordon Growth Model valuation"""
        if ggm_calculator is None:
            logger.warning(f"{self.symbol} - GGM calculator not provided, using simplified calculation")
            # Simplified GGM: P = D1 / (r - g)
            dividend_per_share = financials.get('dividend_per_share', 0)
            cost_of_equity = financials.get('cost_of_equity', financials.get('wacc', 0.10))

            if dividend_per_share <= 0:
                return 0.0, {"method": "gordon_growth", "error": "No dividends", "terminal_growth": terminal_growth_rate}

            if cost_of_equity <= terminal_growth_rate:
                return 0.0, {"method": "gordon_growth", "error": f"Cost of equity ({cost_of_equity:.2f}) <= terminal growth ({terminal_growth_rate:.2f})", "terminal_growth": terminal_growth_rate}

            # Assume 3% dividend growth for next year
            dividend_next_year = dividend_per_share * 1.03
            fair_value = dividend_next_year / (cost_of_equity - terminal_growth_rate)

            return fair_value, {
                "method": "gordon_growth",
                "terminal_growth": terminal_growth_rate,
                "dividend_next_year": dividend_next_year,
                "cost_of_equity": cost_of_equity,
                "fallback": True
            }

        # Use actual GGM calculator
        try:
            cost_of_equity = financials.get('cost_of_equity', financials.get('wacc', 0.10))
            result = ggm_calculator.calculate_ggm_valuation(
                cost_of_equity=cost_of_equity,
                terminal_growth_rate=terminal_growth_rate
            )

            if not result.get('applicable', False):
                return 0.0, {"method": "gordon_growth", "error": result.get('reason', 'Not applicable'), "terminal_growth": terminal_growth_rate}

            fair_value = result.get('fair_value_per_share', 0)
            return fair_value, {
                "method": "gordon_growth",
                "terminal_growth": terminal_growth_rate,
                "dividend_growth": result.get('growth_rate', 0),
                "cost_of_equity": cost_of_equity
            }
        except Exception as e:
            logger.error(f"{self.symbol} - GGM calculation failed: {e}")
            return 0.0, {"method": "gordon_growth", "error": str(e), "terminal_growth": terminal_growth_rate}

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ParallelValuationOrchestrator(symbol='{self.symbol}', "
            f"sector='{self.sector}', "
            f"current_price=${self.current_price:.2f})"
        )
