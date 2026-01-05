"""
Parallel Valuation Orchestrator

Executes multiple valuation frameworks concurrently, sharing unified terminal growth
calculations and computing a blended fair value.

This replaces the sequential SectorValuationRouter approach with parallel execution,
ensuring all frameworks use identical assumptions and reducing latency.

Created: 2025-11-12
Author: Claude Code
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
        pe_calculator: Any = None,  # PERatioValuation instance
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
            pe_calculator: PERatioValuation instance (if P/E framework included)
            ggm_calculator: GordonGrowthModel instance (if GGM framework included)

        Returns:
            BlendedValuationResult with fair value and execution details

        Example:
            >>> result = await orchestrator.execute_valuation(
            ...     frameworks=[dcf_config, pe_config, ev_config],
            ...     rule_of_40_score=58.8,
            ...     revenue_growth_pct=28.6,
            ...     fcf_margin_pct=30.2,
            ...     financials={...},
            ...     dcf_calculator=dcf_calc
            ... )
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
                pe_calculator=pe_calculator,
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

    async def _execute_framework(
        self,
        framework: FrameworkConfig,
        terminal_growth_rate: float,
        financials: Dict[str, Any],
        dcf_calculator: Any = None,
        pe_calculator: Any = None,
        ggm_calculator: Any = None,
    ) -> FrameworkResult:
        """
        Execute a single valuation framework

        Args:
            framework: Framework configuration
            terminal_growth_rate: Unified terminal growth rate (from calculator)
            financials: Company financials
            dcf_calculator: DCF calculator instance
            pe_calculator: P/E calculator instance
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
                    financials=financials, params=framework.params, pe_calculator=pe_calculator
                )

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_EV_EBITDA:
                fair_value, metrics = await self._execute_ev_ebitda(financials=financials, params=framework.params)

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_PS_RATIO:
                fair_value, metrics = await self._execute_ps_ratio(financials=financials, params=framework.params)

            elif framework.type == ValuationFrameworkPlanner.FRAMEWORK_PEG_RATIO:
                fair_value, metrics = await self._execute_peg_ratio(
                    financials=financials, params=framework.params, pe_calculator=pe_calculator
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

            # Calculate execution time
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            logger.info(
                f"{self.symbol} - {framework.type}: ${fair_value:.2f} "
                f"(weight: {framework.weight*100:.1f}%, {execution_time_ms:.0f}ms)"
            )

            return FrameworkResult(
                framework_type=framework.type,
                fair_value=fair_value,
                confidence=1.0,  # TODO: Implement confidence scoring
                metrics=metrics,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            logger.error(f"{self.symbol} - {framework.type} execution failed: {e}", exc_info=True)
            return FrameworkResult(
                framework_type=framework.type, fair_value=None, confidence=0.0, metrics={}, error=str(e)
            )

    # Framework execution methods (placeholders - will integrate with existing calculators)

    async def _execute_dcf_growth(
        self, terminal_growth_rate: float, financials: Dict[str, Any], params: Dict[str, Any], dcf_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """Execute DCF with growth assumptions"""
        # TODO: Call existing DCFValuation with terminal_growth_rate parameter
        # For now, return placeholder
        return 291.36, {"method": "dcf_growth", "terminal_growth": terminal_growth_rate}

    async def _execute_dcf_fading(
        self, terminal_growth_rate: float, financials: Dict[str, Any], params: Dict[str, Any], dcf_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """Execute DCF with fading growth assumptions"""
        # TODO: Call existing DCFValuation with fading growth logic
        return 280.00, {"method": "dcf_fading", "terminal_growth": terminal_growth_rate}

    async def _execute_pe_ratio(
        self, financials: Dict[str, Any], params: Dict[str, Any], pe_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """Execute P/E ratio valuation"""
        # TODO: Call existing P/E calculator
        return 305.00, {"method": "pe_ratio"}

    async def _execute_ev_ebitda(
        self, financials: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """Execute EV/EBITDA valuation"""
        # TODO: Implement EV/EBITDA calculation
        return 295.00, {"method": "ev_ebitda"}

    async def _execute_ps_ratio(
        self, financials: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """Execute P/S ratio valuation"""
        # TODO: Implement P/S ratio calculation
        return 310.00, {"method": "ps_ratio"}

    async def _execute_peg_ratio(
        self, financials: Dict[str, Any], params: Dict[str, Any], pe_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """Execute PEG ratio valuation"""
        # TODO: Implement PEG ratio calculation
        return 300.00, {"method": "peg_ratio"}

    async def _execute_gordon_growth(
        self, terminal_growth_rate: float, financials: Dict[str, Any], params: Dict[str, Any], ggm_calculator: Any
    ) -> tuple[float, Dict[str, Any]]:
        """Execute Gordon Growth Model valuation"""
        # TODO: Call existing GordonGrowthModel with terminal_growth_rate
        return 285.00, {"method": "gordon_growth", "terminal_growth": terminal_growth_rate}

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ParallelValuationOrchestrator(symbol='{self.symbol}', "
            f"sector='{self.sector}', "
            f"current_price=${self.current_price:.2f})"
        )
