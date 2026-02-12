#!/usr/bin/env python3
"""
Batch Analysis Runner - Process large stock lists with Victor Framework

Runs full analysis for Russell 1000 or top N stocks by market cap,
processing them in configurable batch sizes with proper throttling.

Features:
- Async parallel execution within batches using asyncio.gather
- Integration with victor-invest workflows
- Real-time progress tracking
- Checkpoint/resume capability
- Skip already-processed symbols

Usage:
    # Russell 1000 stocks, 5 at a time
    nohup python3 scripts/batch_analysis_runner.py --russell1000 --batch-size 5 > /tmp/batch_analysis.log 2>&1 &

    # Top 1000 by market cap, 3 at a time with 5s delay
    python3 scripts/batch_analysis_runner.py --top 1000 --batch-size 3 --delay 5

    # Resume from a specific symbol
    python3 scripts/batch_analysis_runner.py --russell1000 --resume-from MSFT

Author: Victor-Invest Team
Date: 2025-12-30
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

# Shared symbol repository for consistent ticker fetching
from investigator.infrastructure.database.symbol_repository import SymbolRepository

# Victor-Invest imports - uses BaseYAMLWorkflowProvider pattern
from victor_invest.workflows import (
    AnalysisMode,
    InvestmentWorkflowProvider,
    run_analysis,  # Fallback for backwards compatibility
)

# RL Infrastructure imports for tracking predictions
try:
    from investigator.domain.services.rl.outcome_tracker import OutcomeTracker
    from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    OutcomeTracker = None
    ValuationContext = None

# Shared market data services (used by rl_backtest, batch_analysis_runner, victor_invest)
from investigator.domain.services.market_data import (
    SharesService,
    PriceService,
    DataValidationService,
    SymbolMetadataService,
    TechnicalAnalysisService,
    get_technical_analysis_service,
)

# Shared valuation config services (single source of truth for sector multiples, CAPM, GGM)
from investigator.domain.services.valuation_shared import (
    ValuationConfigService,
    SectorMultiplesService,
)

# Data source facade for economic indicators
from investigator.domain.services.data_sources.facade import get_data_source_facade

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Configure logging to both console and file
log_filename = f"logs/batch_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"

# Suppress ALL loggers except ours before basicConfig
logging.root.setLevel(logging.ERROR)

# Set up our batch runner logger
logger = logging.getLogger("batch_runner")
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear any existing handlers

# Create formatters and handlers
formatter = logging.Formatter("%(asctime)s - %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Aggressively suppress ALL other loggers
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.ERROR)

# Also suppress these common noisy ones explicitly
for noisy in [
    "investigator",
    "victor_invest",
    "sqlalchemy",
    "urllib3",
    "httpx",
    "httpcore",
    "asyncio",
    "root",
    "SEC",
    "FRED",
]:
    logging.getLogger(noisy).setLevel(logging.ERROR)
    logging.getLogger(noisy).propagate = False

logger.info(f"Logging to: {log_filename}")


@dataclass
class SymbolResult:
    """Result of analyzing a single symbol."""

    symbol: str
    success: bool
    fair_value: Optional[float] = None
    current_price: Optional[float] = None
    upside_pct: Optional[float] = None
    tier: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    # Company metadata
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    # RL tracking fields
    rl_record_id: Optional[int] = None
    model_fair_values: Optional[Dict[str, float]] = None
    model_weights: Optional[Dict[str, float]] = None
    # Data quality flags
    data_quality_warnings: Optional[List[str]] = None


@dataclass
class BatchResult:
    """Result of a batch of analyses."""

    batch_num: int
    symbols: List[str]
    results: List[SymbolResult]
    total_duration: float

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.success)


class BatchAnalysisRunner:
    """Runs batch analysis using Victor-Invest workflows."""

    def __init__(
        self,
        batch_size: int = 5,
        delay_between_batches: int = 10,
        mode: str = "standard",
        output_dir: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.mode = AnalysisMode(mode)
        self.output_dir = Path(output_dir) if output_dir else Path("batch_results")

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        self.processed_file = Path("logs/batch_processed_symbols.txt")
        self.error_file = Path("logs/batch_error_symbols.txt")
        self.results_file = self.output_dir / "batch_analysis_results.jsonl"

        # Shared symbol repository for consistent ticker fetching
        self.symbol_repo = SymbolRepository()

        # Keep engine references for methods that need direct DB access
        self.stock_engine = self.symbol_repo.stock_engine
        self.sec_engine = self.symbol_repo.sec_engine

        # RL Outcome Tracker for recording predictions
        self.outcome_tracker = None
        if RL_AVAILABLE:
            try:
                self.outcome_tracker = OutcomeTracker()
                logger.info("RL OutcomeTracker initialized - predictions will be recorded for training")
            except Exception as e:
                logger.warning(f"Failed to initialize OutcomeTracker: {e}")

        # Initialize shared market data services
        # These provide consistent implementations across rl_backtest, batch_analysis_runner, and victor_invest
        self.shares_service = SharesService()
        self.price_service = PriceService()
        self.metadata_service = SymbolMetadataService()
        self.validation_service = DataValidationService()
        self.technical_service = get_technical_analysis_service()
        logger.info("Shared market data services initialized (including TechnicalAnalysisService)")

        # Initialize shared valuation config services
        # Single source of truth for sector multiples, CAPM, GGM defaults
        self.valuation_config_service = ValuationConfigService()
        self.sector_multiples_service = SectorMultiplesService(self.valuation_config_service)
        logger.info("Shared valuation config services initialized")

        # Initialize data source facade for economic indicators
        self.data_source_facade = get_data_source_facade()
        logger.info("DataSourceFacade initialized for economic indicators")

        # Initialize InvestmentWorkflowProvider (BaseYAMLWorkflowProvider pattern)
        # This loads YAML workflows and registers handlers for compute nodes
        self.workflow_provider = InvestmentWorkflowProvider()
        logger.info(f"InvestmentWorkflowProvider initialized with workflows: {self.workflow_provider.get_workflow_names()}")

    def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol metadata (shares, sector, industry) from stock database.

        Delegates to shared SymbolMetadataService for consistent implementation
        across rl_backtest, batch_analysis_runner, and victor_invest.
        """
        metadata = self.metadata_service.get_metadata(symbol)
        if metadata:
            return {
                "sector": metadata.sector,
                "industry": metadata.industry,
                "market_cap": metadata.market_cap,
                "shares_outstanding": metadata.shares_outstanding,
                "beta": metadata.beta or 1.0,
            }
        return {
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": None,
            "shares_outstanding": None,
            "beta": 1.0,
        }

    def get_sec_shares_outstanding(self, symbol: str) -> Optional[float]:
        """
        Get shares outstanding from most recent SEC filing.

        Delegates to shared SharesService for consistent implementation
        across rl_backtest, batch_analysis_runner, and victor_invest.
        """
        return self.shares_service.get_sec_shares(symbol)

    def validate_shares_data(self, symbol: str, current_price: float) -> List[str]:
        """
        Validate shares data for potential issues like unaccounted splits.

        Delegates to shared DataValidationService for consistent implementation
        across rl_backtest, batch_analysis_runner, and victor_invest.

        Returns list of warning messages if issues detected.
        """
        validation_warnings = self.validation_service.validate_shares(symbol, current_price)
        return [str(w) for w in validation_warnings]

    def get_domestic_filers(self) -> Set[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_domestic_filers()

    def get_russell1000_symbols(self) -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_russell1000_symbols()

    def get_all_symbols(self, us_only: bool = True, order_by: str = "stockid") -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_all_symbols(us_only=us_only, order_by=order_by)

    def get_top_n_symbols(self, n: int, us_only: bool = True) -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_top_n_symbols(n, us_only=us_only)

    def get_sp500_symbols(self) -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_sp500_symbols()

    def get_already_processed_symbols(self) -> Set[str]:
        """Get symbols that already have SEC processed data."""
        with self.sec_engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT symbol FROM sec_companyfacts_processed"))
            symbols = {row[0] for row in result.fetchall()}
            logger.info(f"Found {len(symbols)} symbols already in SEC database")
            return symbols

    def load_processed_from_file(self) -> Set[str]:
        """Load previously processed symbols from tracking file."""
        if self.processed_file.exists():
            with open(self.processed_file) as f:
                return {line.strip() for line in f if line.strip()}
        return set()

    def save_processed_symbol(self, symbol: str):
        """Save a processed symbol to tracking file."""
        with open(self.processed_file, "a") as f:
            f.write(f"{symbol}\n")

    def save_error_symbol(self, symbol: str, error: str):
        """Save an error symbol to tracking file."""
        with open(self.error_file, "a") as f:
            f.write(f"{symbol}: {error}\n")

    def save_result(self, result: SymbolResult):
        """Append result to JSONL file."""
        with open(self.results_file, "a") as f:
            result_dict = asdict(result)
            result_dict["timestamp"] = datetime.now().isoformat()
            f.write(json.dumps(result_dict) + "\n")

    def _fetch_economic_indicators(self) -> Dict[str, Any]:
        """Fetch economic indicators (Regional Fed + CBOE) from DataSourceFacade.

        Returns a flattened dict of key economic metrics for RL features:
        - gdpnow, cfnai, nfci, kcfsi (economic activity/financial conditions)
        - inflation_expectations, recession_probability (macro outlook)
        - vix, skew, vix_term_structure (volatility/risk sentiment)
        """
        try:
            analysis_data = self.data_source_facade.get_historical_data_sync(
                symbol="_MACRO",
                as_of_date=date.today(),
            )

            # Extract regional Fed summary
            regional_fed = analysis_data.regional_fed_indicators or {}
            fed_summary = regional_fed.get("summary", {}) if isinstance(regional_fed, dict) else {}

            # Extract CBOE data
            cboe = analysis_data.cboe_data or {}
            vix = cboe.get("vix")
            vix3m = cboe.get("vix3m")

            # Classify volatility regime to int
            regime_map = {
                "very_low": 0, "low": 1, "normal": 2,
                "elevated": 3, "high": 4, "extreme": 5,
            }
            vol_regime = regime_map.get(cboe.get("volatility_regime", "normal"), 2)

            return {
                # Regional Fed indicators
                "gdpnow": fed_summary.get("gdpnow"),
                "cfnai": fed_summary.get("cfnai"),
                "nfci": fed_summary.get("nfci"),
                "kcfsi": fed_summary.get("kcfsi"),
                "inflation_expectations": fed_summary.get("inflation_expectations"),
                "recession_probability": fed_summary.get("recession_probability"),
                "empire_state_mfg": fed_summary.get("empire_state_mfg"),
                # CBOE data
                "vix": vix,
                "vix_term_structure": (vix3m / vix) if vix and vix3m and vix > 0 else 1.0,
                "skew": cboe.get("skew"),
                "volatility_regime": vol_regime,
                "is_backwardation": cboe.get("is_backwardation", False),
            }
        except Exception as e:
            logger.debug(f"Failed to fetch economic indicators: {e}")
            return {}

    def _convert_workflow_result(self, symbol: str, workflow_result: Any) -> Any:
        """Convert YAML workflow result to AnalysisWorkflowState format.

        Bridges the gap between YAML workflow execution output and the
        expected AnalysisWorkflowState for RL tracking and result processing.
        """
        from victor_invest.workflows import AnalysisWorkflowState

        # Extract outputs from workflow context
        context_data = {}
        if hasattr(workflow_result, 'context'):
            ctx = workflow_result.context
            # Get outputs from context - either dict or WorkflowContext
            if hasattr(ctx, 'get'):
                context_data = {
                    "symbol": ctx.get("symbol", symbol),
                    "mode": ctx.get("mode", self.mode.value),
                    "sec_data": ctx.get("sec_data"),
                    "market_data": ctx.get("market_data"),
                    "fundamental_analysis": ctx.get("fundamental_analysis"),
                    "technical_analysis": ctx.get("technical_analysis"),
                    "market_context": ctx.get("market_context"),
                    "synthesis": ctx.get("synthesis"),
                    "recommendation": ctx.get("recommendation") or ctx.get("synthesis", {}).get("recommendation"),
                }
            elif isinstance(ctx, dict):
                context_data = ctx
        elif isinstance(workflow_result, dict):
            context_data = workflow_result

        # Create AnalysisWorkflowState from context data
        state = AnalysisWorkflowState(
            symbol=symbol.upper(),
            mode=self.mode,
        )

        # Populate state fields
        if context_data.get("sec_data"):
            state.sec_data = context_data["sec_data"]
        if context_data.get("market_data"):
            state.market_data = context_data["market_data"]
        if context_data.get("fundamental_analysis"):
            state.fundamental_analysis = context_data["fundamental_analysis"]
        if context_data.get("technical_analysis"):
            state.technical_analysis = context_data["technical_analysis"]
        if context_data.get("market_context"):
            state.market_context = context_data["market_context"]
        if context_data.get("synthesis"):
            state.synthesis = context_data["synthesis"]
        if context_data.get("recommendation"):
            state.recommendation = context_data["recommendation"]

        return state

    def _build_valuation_context(
        self, symbol: str, result: Any, current_price: Optional[float] = None, fair_value: Optional[float] = None
    ) -> Any:
        """Build ValuationContext from analysis result for RL tracking.

        Includes technical indicators from shared TechnicalAnalysisService and
        economic indicators from DataSourceFacade for consistent feature extraction
        across rl_backtest and batch_analysis_runner.
        """
        if not RL_AVAILABLE or ValuationContext is None:
            return {}

        # Fetch economic indicators (Regional Fed + CBOE)
        economic_data = self._fetch_economic_indicators()

        synthesis = result.synthesis or {}
        fundamental = result.fundamental_analysis or {}

        # Extract metrics from fundamental analysis
        profitability = fundamental.get("profitability", {})
        growth = fundamental.get("growth", {})
        valuation = fundamental.get("valuation", {})

        # Classify growth stage
        net_income = fundamental.get("net_income", 0) or 0
        payout_ratio = profitability.get("payout_ratio", 0) or 0
        revenue_growth = growth.get("revenue_growth", 0) or 0

        if net_income < 0:
            growth_stage = GrowthStage.PRE_PROFIT
        elif payout_ratio > 0.30:
            growth_stage = GrowthStage.DIVIDEND_PAYING
        elif revenue_growth > 0.25:
            growth_stage = GrowthStage.HIGH_GROWTH
        else:
            growth_stage = GrowthStage.MATURE

        # Classify company size by market cap
        market_cap = fundamental.get("market_cap", 0) or 0
        if market_cap > 200e9:
            company_size = CompanySize.MEGA_CAP
        elif market_cap > 10e9:
            company_size = CompanySize.LARGE_CAP
        elif market_cap > 2e9:
            company_size = CompanySize.MID_CAP
        elif market_cap > 300e6:
            company_size = CompanySize.SMALL_CAP
        else:
            company_size = CompanySize.MICRO_CAP

        # Get technical features from shared TechnicalAnalysisService
        # This ensures consistency with rl_backtest.py feature extraction
        price = current_price or synthesis.get("current_price")
        fv = fair_value or synthesis.get("fair_value")
        tech_features = self.technical_service.get_technical_features(
            symbol=symbol,
            current_price=price,
            fair_value=fv,
        )

        return ValuationContext(
            symbol=symbol,
            analysis_date=date.today(),
            sector=fundamental.get("sector", "Unknown"),
            industry=fundamental.get("industry", "Unknown"),
            growth_stage=growth_stage,
            company_size=company_size,
            profitability_score=min(1.0, max(0, (profitability.get("net_margin", 0) or 0) + 0.1) / 0.3),
            pe_level=min(1.0, (valuation.get("pe_ratio", 20) or 20) / 50),
            revenue_growth=revenue_growth,
            fcf_margin=profitability.get("fcf_margin", 0) or 0,
            rule_of_40_score=growth.get("rule_of_40", 0) or 0,
            payout_ratio=payout_ratio,
            debt_to_equity=min(3.0, profitability.get("debt_to_equity", 0) or 0),
            gross_margin=profitability.get("gross_margin", 0) or 0,
            operating_margin=profitability.get("operating_margin", 0) or 0,
            data_quality_score=fundamental.get("data_quality_score", 75.0) or 75.0,
            quarters_available=fundamental.get("quarters_available", 4) or 4,
            current_price=price,
            # Technical indicators (from shared TechnicalAnalysisService)
            rsi_14=tech_features.rsi_14,
            macd_histogram=tech_features.macd_histogram,
            obv_trend=tech_features.obv_trend,
            adx_14=tech_features.adx_14,
            stoch_k=tech_features.stoch_k,
            mfi_14=tech_features.mfi_14,
            # Entry/Exit signal features
            entry_signal_strength=tech_features.entry_signal_strength,
            exit_signal_strength=tech_features.exit_signal_strength,
            signal_confluence=tech_features.signal_confluence,
            days_from_support=tech_features.days_from_support,
            risk_reward_ratio=tech_features.risk_reward_ratio,
            # Economic indicators (Regional Fed)
            gdpnow=economic_data.get("gdpnow") or 2.0,
            cfnai=economic_data.get("cfnai") or 0.0,
            nfci=economic_data.get("nfci") or 0.0,
            kcfsi=economic_data.get("kcfsi") or 0.0,
            inflation_expectations=economic_data.get("inflation_expectations") or 2.5,
            recession_probability=economic_data.get("recession_probability") or 0.15,
            empire_state_mfg=economic_data.get("empire_state_mfg") or 0.0,
            # CBOE volatility data
            vix=economic_data.get("vix") or 18.0,
            vix_term_structure=economic_data.get("vix_term_structure") or 1.0,
            skew=economic_data.get("skew") or 120.0,
            volatility_regime=economic_data.get("volatility_regime") or 2,
            is_backwardation=economic_data.get("is_backwardation", False),
        )

    async def analyze_symbol(self, symbol: str) -> SymbolResult:
        """Run analysis for a single symbol using InvestmentWorkflowProvider.

        Uses BaseYAMLWorkflowProvider pattern with YAML-defined workflows and
        shared handlers for compute nodes. This ensures consistent execution
        across CLI, batch runner, and RL backtest.
        """
        start_time = time.time()

        try:
            # Get symbol metadata first (sector, shares, etc.)
            metadata = self.get_symbol_metadata(symbol)
            sector = metadata.get("sector", "Unknown")
            industry = metadata.get("industry", "Unknown")

            logger.info(f"  Starting: {symbol} ({sector})")

            # Map mode to workflow name
            workflow_name = self.workflow_provider.get_workflow_for_task_type(self.mode.value) or self.mode.value

            # Get the workflow definition
            workflow = self.workflow_provider.get_workflow(workflow_name)

            if workflow:
                # Execute via YAML workflow with shared handlers
                from victor.workflows.executor import WorkflowExecutor, WorkflowContext

                # Create execution context with symbol
                context = WorkflowContext({"symbol": symbol.upper(), "mode": self.mode.value})

                # Create executor (no orchestrator needed for pure compute workflows)
                # All our handlers have llm_allowed: false
                executor = WorkflowExecutor(orchestrator=None)

                # Execute the workflow
                workflow_result = await executor.execute(workflow, context)

                # Convert workflow result to expected format
                result = self._convert_workflow_result(symbol, workflow_result)
            else:
                # Fallback to Python-based execution if workflow not found
                logger.debug(f"  Workflow '{workflow_name}' not found, using Python fallback")
                result = await run_analysis(symbol.upper(), self.mode)

            duration = time.time() - start_time

            # Extract key metrics from result
            fair_value = None
            current_price = None
            upside_pct = None
            tier = None
            model_fair_values = {}
            model_weights = {}
            rl_record_id = None
            shares_outstanding = metadata.get("shares_outstanding")
            market_cap = metadata.get("market_cap")

            if result.synthesis:
                fair_value = result.synthesis.get("fair_value")
                current_price = result.synthesis.get("current_price")
                tier = result.synthesis.get("tier")
                model_fair_values = result.synthesis.get("model_fair_values", {})
                model_weights = result.synthesis.get("model_weights", {})

                # Calculate fair_value from model weights if not directly provided
                if fair_value is None and model_fair_values and model_weights:
                    total_weight = sum(model_weights.values())
                    if total_weight > 0:
                        weighted_sum = sum(
                            fv * model_weights.get(model, 0)
                            for model, fv in model_fair_values.items()
                            if fv is not None
                        )
                        fair_value = weighted_sum / total_weight
                        logger.debug(f"  Calculated fair_value from model weights: ${fair_value:.2f}")

                if fair_value and current_price and current_price > 0:
                    upside_pct = ((fair_value / current_price) - 1) * 100

                # Update market cap from current price if available
                if current_price and shares_outstanding:
                    market_cap = shares_outstanding * current_price

            # Validate shares data for potential split issues
            data_warnings = []
            if current_price:
                data_warnings = self.validate_shares_data(symbol, current_price)
                for warning in data_warnings:
                    logger.warning(f"  {symbol}: {warning}")

            # Record prediction for RL training
            if self.outcome_tracker and fair_value and current_price:
                try:
                    # Build context features from analysis result (includes technical indicators)
                    context = self._build_valuation_context(
                        symbol, result, current_price=current_price, fair_value=fair_value
                    )

                    rl_record_id = self.outcome_tracker.record_prediction(
                        symbol=symbol,
                        analysis_date=date.today(),
                        blended_fair_value=fair_value,
                        current_price=current_price,
                        model_fair_values=model_fair_values,
                        model_weights=model_weights,
                        tier_classification=tier or "unknown",
                        context_features=context,
                        fiscal_period=result.synthesis.get("fiscal_period") if result.synthesis else None,
                    )
                    if rl_record_id:
                        logger.debug(f"  RL prediction recorded: {symbol} -> record_id={rl_record_id}")
                except Exception as e:
                    logger.warning(f"  Failed to record RL prediction for {symbol}: {e}")

            # Enhanced logging with shares and market cap
            fv_str = f"${fair_value:.2f}" if fair_value else "N/A"
            price_str = f"${current_price:.2f}" if current_price else "N/A"
            upside_str = f"{upside_pct:.1f}%" if upside_pct is not None else "N/A"
            shares_str = f"{shares_outstanding/1e9:.2f}B" if shares_outstanding else "N/A"
            mktcap_str = f"${market_cap/1e9:.0f}B" if market_cap else "N/A"

            logger.info(
                f"  Done: {symbol} [{tier or 'unknown'}] in {duration:.1f}s | "
                f"FV={fv_str}, Price={price_str}, "
                f"Upside={upside_str}, Shares={shares_str}, MktCap={mktcap_str}"
            )

            return SymbolResult(
                symbol=symbol,
                success=True,
                fair_value=fair_value,
                current_price=current_price,
                upside_pct=upside_pct,
                tier=tier,
                duration_seconds=duration,
                shares_outstanding=shares_outstanding,
                market_cap=market_cap,
                sector=sector,
                industry=industry,
                rl_record_id=rl_record_id,
                model_fair_values=model_fair_values,
                model_weights=model_weights,
                data_quality_warnings=data_warnings if data_warnings else None,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"  Failed: {symbol} - {e}")
            return SymbolResult(
                symbol=symbol,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    async def run_batch(self, batch_num: int, symbols: List[str]) -> BatchResult:
        """Run analysis for a batch of symbols in parallel."""
        if not symbols:
            return BatchResult(
                batch_num=batch_num,
                symbols=[],
                results=[],
                total_duration=0,
            )

        logger.info(f"Running batch {batch_num}: {', '.join(symbols)}")
        start_time = time.time()

        # Execute all symbols in parallel using asyncio.gather
        tasks = [self.analyze_symbol(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to SymbolResult
        processed_results = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                processed_results.append(
                    SymbolResult(
                        symbol=symbol,
                        success=False,
                        error=str(result),
                        duration_seconds=0,
                    )
                )
            else:
                processed_results.append(result)

        total_duration = time.time() - start_time

        return BatchResult(
            batch_num=batch_num,
            symbols=symbols,
            results=processed_results,
            total_duration=total_duration,
        )

    async def run(
        self,
        symbols: List[str],
        skip_processed: bool = True,
        resume_from: Optional[str] = None,
        skip_domestic_filter: bool = False,
    ):
        """Run batch analysis for all symbols."""
        print(f"  Starting batch run with {len(symbols)} symbols...", flush=True)
        start_time = datetime.now()

        # Filter out already processed symbols
        if skip_processed:
            print("  Checking for already processed symbols...", flush=True)
            already_processed = self.get_already_processed_symbols()
            file_processed = self.load_processed_from_file()
            all_processed = already_processed | file_processed

            original_count = len(symbols)
            symbols = [s for s in symbols if s not in all_processed]
            print(
                f"  Filtered: {original_count} -> {len(symbols)} symbols "
                f"(skipping {len(all_processed)} already processed)", flush=True
            )
            logger.info(
                f"Filtered from {original_count} to {len(symbols)} symbols "
                f"(skipping {len(all_processed)} already processed)"
            )

        # Filter out foreign filers (20-F/6-K) - they lack quarterly data for proper valuation
        if not skip_domestic_filter:
            print("  Checking for domestic filers...", flush=True)
            domestic_filers = self.get_domestic_filers()
            foreign_count = len([s for s in symbols if s not in domestic_filers])
            if foreign_count > 0:
                symbols = [s for s in symbols if s in domestic_filers]
                print(f"  Filtered out {foreign_count} foreign filers -> {len(symbols)} remaining", flush=True)
                logger.info(
                    f"Filtered out {foreign_count} foreign filers (20-F/6-K) - " f"{len(symbols)} domestic filers remaining"
                )
        else:
            print("  Skipping domestic filer filter (--skip-domestic-filter)", flush=True)

        # Resume from specific symbol
        if resume_from:
            try:
                idx = symbols.index(resume_from)
                symbols = symbols[idx:]
                print(f"  Resuming from {resume_from}, {len(symbols)} remaining", flush=True)
                logger.info(f"Resuming from {resume_from}, {len(symbols)} symbols remaining")
            except ValueError:
                logger.warning(f"Resume symbol {resume_from} not found in list")

        if not symbols:
            print("  No symbols to process after filtering!", flush=True)
            logger.info("No symbols to process!")
            return

        total_symbols = len(symbols)
        total_batches = (total_symbols + self.batch_size - 1) // self.batch_size

        print("=" * 60, flush=True)
        print("VICTOR-INVEST BATCH ANALYSIS RUNNER", flush=True)
        print("=" * 60, flush=True)
        print(f"  Total symbols: {total_symbols}", flush=True)
        print(f"  Batch size: {self.batch_size}", flush=True)
        print(f"  Total batches: {total_batches}", flush=True)
        print(f"  Analysis mode: {self.mode.value}", flush=True)
        print(f"  RL Policy: Using dual policy (technical + fundamental)", flush=True)
        print("=" * 60, flush=True)

        logger.info("=" * 60)
        logger.info("VICTOR-INVEST BATCH ANALYSIS RUNNER")
        logger.info("=" * 60)
        logger.info(f"Total symbols: {total_symbols}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Analysis mode: {self.mode.value}")
        logger.info(f"Delay between batches: {self.delay_between_batches}s")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

        success_count = 0
        error_count = 0
        all_results: List[SymbolResult] = []

        for batch_num in range(total_batches):
            batch_start = batch_num * self.batch_size
            batch_end = min(batch_start + self.batch_size, total_symbols)
            batch_symbols = symbols[batch_start:batch_end]

            print(f"\nBATCH {batch_num + 1}/{total_batches}: {', '.join(batch_symbols)}", flush=True)
            logger.info(f"\n{'='*40}")
            logger.info(f"BATCH {batch_num + 1}/{total_batches} " f"({batch_start + 1}-{batch_end} of {total_symbols})")
            logger.info(f"Symbols: {', '.join(batch_symbols)}")
            logger.info(f"{'='*40}")

            # Run batch
            batch_result = await self.run_batch(batch_num + 1, batch_symbols)

            # Track results
            for result in batch_result.results:
                all_results.append(result)
                self.save_result(result)

                if result.success:
                    self.save_processed_symbol(result.symbol)
                    success_count += 1
                    upside_str = f"{result.upside_pct:+.1f}%" if result.upside_pct else "N/A"
                    print(f"  ✓ {result.symbol}: {upside_str} ({result.duration_seconds:.1f}s)", flush=True)
                else:
                    self.save_error_symbol(result.symbol, result.error or "unknown")
                    error_count += 1
                    print(f"  ✗ {result.symbol}: {result.error[:50] if result.error else 'unknown'}", flush=True)

            # Progress update
            elapsed = (datetime.now() - start_time).total_seconds()
            processed = batch_end
            rate = processed / elapsed * 60 if elapsed > 0 else 0
            remaining = total_symbols - processed
            eta_minutes = remaining / rate if rate > 0 else 0

            print(
                f"  Progress: {processed}/{total_symbols} ({processed/total_symbols*100:.1f}%) | "
                f"Success: {success_count} | Errors: {error_count} | ETA: {eta_minutes:.0f}m", flush=True
            )
            logger.info(
                f"\nProgress: {processed}/{total_symbols} ({processed/total_symbols*100:.1f}%) | "
                f"Success: {success_count} | Errors: {error_count} | "
                f"Rate: {rate:.1f}/min | ETA: {eta_minutes:.0f} min"
            )

            # Delay before next batch (unless this is the last batch)
            if batch_num < total_batches - 1:
                logger.info(f"Waiting {self.delay_between_batches}s before next batch...")
                await asyncio.sleep(self.delay_between_batches)

        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()

        # Calculate summary statistics
        successful_results = [r for r in all_results if r.success and r.upside_pct is not None]

        logger.info("\n" + "=" * 60)
        logger.info("BATCH ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Symbols processed: {success_count}")
        logger.info(f"Symbols failed: {error_count}")
        if success_count + error_count > 0:
            logger.info(f"Success rate: {success_count/(success_count+error_count)*100:.1f}%")

        if successful_results:
            avg_upside = sum(r.upside_pct for r in successful_results) / len(successful_results)
            undervalued = [r for r in successful_results if r.upside_pct > 20]
            overvalued = [r for r in successful_results if r.upside_pct < -20]

            logger.info(f"\nValuation Summary:")
            logger.info(f"  Average upside: {avg_upside:.1f}%")
            logger.info(f"  Undervalued (>20% upside): {len(undervalued)}")
            logger.info(f"  Overvalued (<-20% upside): {len(overvalued)}")

            if undervalued:
                top_5 = sorted(undervalued, key=lambda r: r.upside_pct, reverse=True)[:5]
                logger.info(f"\n  Top 5 Undervalued:")
                for r in top_5:
                    logger.info(
                        f"    {r.symbol}: {r.upside_pct:.1f}% upside (${r.current_price:.2f} -> ${r.fair_value:.2f})"
                    )

        # RL Training Data Summary
        rl_recorded = [r for r in all_results if r.rl_record_id is not None]
        if rl_recorded:
            logger.info(f"\nRL Training Data:")
            logger.info(f"  Predictions recorded: {len(rl_recorded)}")
            logger.info(f"  Ready for reward calculation after 30/90/365 days")
            logger.info(f"  Run 'python3 scripts/rl_update_outcomes.py' to update outcomes")
        elif self.outcome_tracker:
            logger.info(f"\nRL Training: No predictions recorded (check OutcomeTracker)")

        # Data Quality Warnings Summary
        results_with_warnings = [r for r in all_results if r.data_quality_warnings]
        if results_with_warnings:
            logger.info(f"\nData Quality Warnings ({len(results_with_warnings)} symbols):")
            for r in results_with_warnings[:10]:  # Show top 10
                for warning in r.data_quality_warnings:
                    logger.info(f"  {r.symbol}: {warning}")
            if len(results_with_warnings) > 10:
                logger.info(f"  ... and {len(results_with_warnings) - 10} more symbols with warnings")
            logger.info(f"  Tip: Review these symbols for potential stock split issues")

        # Sector Distribution
        sector_counts: Dict[str, int] = {}
        for r in successful_results:
            if r.sector:
                sector_counts[r.sector] = sector_counts.get(r.sector, 0) + 1
        if sector_counts:
            logger.info(f"\nSector Distribution:")
            for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1])[:8]:
                logger.info(f"  {sector}: {count}")

        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.results_file}")


def main():
    parser = argparse.ArgumentParser(description="Victor-Invest Batch Analysis Runner")

    # Symbol source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--russell1000", action="store_true", help="Process Russell 1000 stocks")
    source_group.add_argument("--sp500", action="store_true", help="Process S&P 500 stocks")
    source_group.add_argument("--all", action="store_true", help="Process ALL stocks from symbol table")
    source_group.add_argument("--top", type=int, metavar="N", help="Process top N stocks by market cap")
    source_group.add_argument("--file", type=str, metavar="FILE", help="Process symbols from file (one per line)")
    source_group.add_argument("--symbols", type=str, nargs="+", help="Process specific symbols")

    # Processing options
    parser.add_argument("--batch-size", type=int, default=5, help="Number of symbols per batch (default: 5)")
    parser.add_argument("--delay", type=int, default=10, help="Delay between batches in seconds (default: 10)")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Analysis mode (default: standard)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip already processed symbols")
    parser.add_argument("--resume-from", type=str, metavar="SYMBOL", help="Resume from specific symbol")
    parser.add_argument(
        "--include-foreign",
        action="store_true",
        help="Include foreign stocks without SEC filings (default: US only with CIK)",
    )
    parser.add_argument(
        "--skip-domestic-filter",
        action="store_true",
        help="Skip the domestic filer filter (process all stocks even without SEC quarterly data)",
    )
    parser.add_argument(
        "--order-by",
        choices=["stockid", "mktcap", "ticker"],
        default="stockid",
        help="Sort order: stockid (ascending), mktcap (descending), ticker (alphabetical). Default: stockid",
    )

    args = parser.parse_args()

    print(f"Starting batch analysis runner...", flush=True)
    print(f"  Mode: {args.mode}, Batch size: {args.batch_size}, Delay: {args.delay}s", flush=True)

    try:
        runner = BatchAnalysisRunner(
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
            mode=args.mode,
            output_dir=args.output,
        )
        print("  Runner initialized successfully", flush=True)
    except Exception as e:
        print(f"ERROR initializing runner: {e}", flush=True)
        raise

    # Get symbols based on source
    us_only = not args.include_foreign
    if args.russell1000:
        print("  Fetching Russell 1000 symbols...", flush=True)
        symbols = runner.get_russell1000_symbols()
        print(f"  Found {len(symbols)} symbols", flush=True)
    elif args.sp500:
        print("  Fetching S&P 500 symbols...", flush=True)
        symbols = runner.get_sp500_symbols()
        print(f"  Found {len(symbols)} symbols", flush=True)
    elif getattr(args, 'all', False):
        print(f"  Fetching ALL stocks from symbol table (order: {args.order_by})...", flush=True)
        symbols = runner.get_all_symbols(us_only=us_only, order_by=args.order_by)
        print(f"  Found {len(symbols)} symbols", flush=True)
    elif args.top:
        print(f"  Fetching top {args.top} stocks by market cap...", flush=True)
        symbols = runner.get_top_n_symbols(args.top, us_only=us_only)
        print(f"  Found {len(symbols)} symbols", flush=True)
        if us_only:
            logger.info("Filtering to US stocks with SEC CIK only (use --include-foreign for all)")
    elif args.file:
        with open(args.file) as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(symbols)} symbols from {args.file}", flush=True)
    elif args.symbols:
        symbols = args.symbols
        print(f"  Processing {len(symbols)} specified symbols", flush=True)

    # Run the batch analysis
    skip_domestic = getattr(args, 'skip_domestic_filter', False)
    asyncio.run(
        runner.run(
            symbols=symbols,
            skip_processed=not args.no_skip,
            resume_from=args.resume_from,
            skip_domestic_filter=skip_domestic,
        )
    )


if __name__ == "__main__":
    main()
