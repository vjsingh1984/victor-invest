"""
Fundamental Analysis Agent
Specialized agent for fundamental analysis and financial metrics evaluation using Ollama LLMs
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from investigator.domain.agents.base import InvestmentAgent
from investigator.domain.models.analysis import AgentResult, AgentTask, TaskStatus
from investigator.domain.services.company_metadata_service import CompanyMetadataService
from investigator.domain.services.data_normalizer import (  # TODO: Move to infrastructure
    DataNormalizer,
    normalize_financials,
)
from investigator.domain.services.deterministic_competitive_analyzer import analyze_competitive_position

# Deterministic services (replace LLM calls with rule-based computation)
from investigator.domain.services.dynamic_model_weighting import DynamicModelWeightingService
from investigator.domain.services.fiscal_period_service import get_fiscal_period_service
from investigator.domain.services.safe_formatters import format_currency as _fmt_currency
from investigator.domain.services.safe_formatters import format_int_with_commas as _fmt_int_comma
from investigator.domain.services.safe_formatters import format_percentage as _fmt_pct
from investigator.domain.services.safe_formatters import (
    is_valid_number,
    safe_round,
)
from investigator.domain.services.toon_formatter import TOONFormatter, to_toon_quarterly
from investigator.domain.services.valuation import SectorValuationRouter  # Sector-aware valuation routing

# New valuation models (Milestone 7 - Plan implementation)
from investigator.domain.services.valuation.dcf import DCFValuation
from investigator.domain.services.valuation.ggm import GordonGrowthModel

# Clean architecture imports (Phase 6-7 migration)
from investigator.domain.services.valuation.helpers import (
    normalize_model_output,
    serialize_company_profile,
)

# Clean architecture imports (migrated from utils/valuation/framework)
from investigator.domain.services.valuation.models import (
    CompanyArchetype,
    CompanyProfile,
    DataQualityFlag,
)
from investigator.domain.services.valuation.orchestrator import MultiModelValuationOrchestrator
from investigator.infrastructure.cache import CacheManager
from investigator.infrastructure.cache.cache_key_builder import build_cache_key
from investigator.infrastructure.cache.cache_types import CacheType
from investigator.infrastructure.data.sector_multiples_loader import SectorMultiplesLoader
from investigator.infrastructure.database.market_data import get_market_data_fetcher  # Singleton pattern
from investigator.infrastructure.database.ticker_mapper import TickerCIKMapper  # TODO: Move to infrastructure
from investigator.infrastructure.formatters import ValuationTableFormatter
from investigator.infrastructure.sec.canonical_mapper import get_canonical_mapper

from .constants import (
    FALLBACK_CANONICAL_KEYS,
    PROCESSED_ADDITIONAL_FINANCIAL_KEYS,
    PROCESSED_RATIO_KEYS,
)
from .company_fetch import fetch_latest_company_data_from_processed_table
from .data_quality_assessor import get_data_quality_assessor
from .deterministic_analyzer import DeterministicAnalyzer
from .deterministic_payloads import (
    build_deterministic_cache_record,
    build_deterministic_response,
)
from .formatters import safe_fmt_float as _safe_fmt_float
from .formatters import safe_fmt_int_comma as _safe_fmt_int_comma
from .formatters import safe_fmt_pct as _safe_fmt_pct
from .logging_utils import (
    format_trend_context,
    log_data_quality_issues,
    log_individual_model_result,
    log_quarterly_snapshot,
    log_table,
    log_valuation_snapshot,
)
from .models import QuarterlyData
from .quarterly_fetch import (
    build_financials_from_bulk_tables,
    build_financials_from_processed_data,
    fetch_processed_quarter_payload,
    normalize_cached_quarter,
    query_recent_processed_periods,
)
from .summaries import extract_latest_financials as _extract_latest_financials_helper
from .summaries import get_historical_trend as _get_historical_trend_helper
from .summaries import summarize_company_data as _summarize_company_data_helper
from .trend_analyzer import get_trend_analyzer
from .valuation_models import calculate_relative_valuation_models
from .valuation_extensions import calculate_valuation_extensions
from .valuation_orchestrator import (
    dispatch_valuation_synthesis,
    log_multi_model_summary,
    run_multi_model_blending,
)
from .valuation_synthesis import (
    build_models_detail_lines,
    build_valuation_synthesis_prompt,
)
from .valuation_weighting import resolve_fallback_weights


class FundamentalAnalysisAgent(InvestmentAgent):
    """
    Agent specialized in fundamental analysis and company valuation.

    High-level flow (ascii schematic):

        ┌────────────┐        ┌────────────────────┐        ┌────────────────────┐
        │  SEC Agent │ ─────▶ │  Processed Tables  │ ─────▶ │ Fundamental Agent  │
        │ (raw facts)│        │  (sec_companyfacts)│        │  (ratios + models) │
        └────────────┘        └────────┬───────────┘        └─────────┬──────────┘
                                       │                                │
                                       │ company_data                   │ blended outputs
                                       ▼                                ▼
                                deterministic health/growth      Multi-model valuation
                                scoring + DCF/GGM/multiples      → cached for synthesis
    """

    def __init__(self, agent_id: str, ollama_client, event_bus, cache_manager: CacheManager):
        from investigator.config import get_config

        config = get_config()
        self.config = config

        self.primary_model = config.ollama.models.get("fundamental_analysis", "deepseek-r1:32b")
        self.comparison_model = config.ollama.models.get("comparison", self.primary_model)

        # Specialized models for different analysis types
        self.models = {
            "valuation": self.primary_model,
            "quality": self.primary_model,
            "comparison": self.comparison_model,
        }

        super().__init__(agent_id, ollama_client, event_bus, cache_manager)
        self.market_data = get_market_data_fetcher(config)
        self.ticker_mapper = TickerCIKMapper()

        # Initialize canonical key mapper for sector-aware XBRL tag extraction
        self.canonical_mapper = get_canonical_mapper()

        # Initialize CompanyMetadataService for centralized sector/industry lookup with override support
        self.company_metadata_service = CompanyMetadataService()

        # Cache for shares outstanding (avoid redundant DB queries per symbol)
        self._shares_cache = {}

        # Cache for sector information (avoid redundant DB queries per symbol)
        # NOTE: Kept for backward compatibility but now delegating to CompanyMetadataService
        self._sector_cache = {}

        # Sector multiples loader (lazy to allow missing reference file)
        self._sector_multiples_loader: Optional[SectorMultiplesLoader] = None

        valuation_cfg = getattr(config, "valuation", None)
        multiples_path: Optional[str] = None
        freshness_days = 7
        delta_threshold = 0.15
        if isinstance(valuation_cfg, dict):
            multiples_path = valuation_cfg.get("sector_multiples_path")
            freshness_days = valuation_cfg.get("sector_multiples_freshness_days", freshness_days)
            delta_threshold = valuation_cfg.get("sector_multiples_delta_threshold", delta_threshold)
        elif valuation_cfg is not None:
            multiples_path = getattr(valuation_cfg, "sector_multiples_path", None)
            freshness_days = getattr(valuation_cfg, "sector_multiples_freshness_days", freshness_days)
            delta_threshold = getattr(valuation_cfg, "sector_multiples_delta_threshold", delta_threshold)

        if multiples_path:
            reference_path = Path(multiples_path)
        else:
            reference_path = Path("config/sector_multiples.json")

        self._sector_multiples_loader = SectorMultiplesLoader(
            reference_path=reference_path,
            freshness_days=freshness_days,
            delta_threshold=delta_threshold,
        )

        # Load model selection rules
        self._model_selection_rules = self._load_model_selection_rules()

        # Multi-model valuation orchestrator (weights and diagnostics)
        self.multi_model_orchestrator = MultiModelValuationOrchestrator()

        # Dynamic model weighting service (tier-based weight determination)
        # Load valuation config from config.yaml (migrated from config.json)
        config_file = getattr(config, "config_file", "config.yaml")
        with open(config_file, "r") as f:
            raw_config = yaml.safe_load(f)
        valuation_config_dict = raw_config.get("valuation", {})
        self.dynamic_weighting_service = DynamicModelWeightingService(valuation_config_dict)

        # Deterministic processing config (replaces LLM calls with rule-based computation)
        deterministic_config = valuation_config_dict.get("deterministic", {})
        self.use_deterministic = deterministic_config.get("enabled", True)
        self.deterministic_valuation_synthesis = deterministic_config.get("valuation_synthesis", True)
        self.deterministic_competitive_analysis = deterministic_config.get("competitive_analysis", True)

        # Key fundamental metrics to analyze
        self.key_metrics = [
            "pe_ratio",
            "peg_ratio",
            "price_to_book",
            "price_to_sales",
            "debt_to_equity",
            "current_ratio",
            "quick_ratio",
            "roe",
            "roa",
            "roic",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "revenue_growth",
            "earnings_growth",
            "free_cash_flow",
            "fcf_yield",
            "dividend_yield",
        ]

        # Valuation models to apply
        self.valuation_models = ["dcf", "ddm", "relative_valuation", "asset_based", "earnings_power"]

    def _debug_log_prompt(self, label: str, prompt: str) -> None:
        """Emit prompt text when debug logging is enabled."""
        if self.logger.isEnabledFor(logging.DEBUG):
            trimmed = prompt if len(prompt) <= 6000 else f"{prompt[:6000]}\n...[truncated]"
            self.logger.debug("📤 %s PROMPT:\n%s", label, trimmed)

    def _debug_log_response(self, label: str, response: Any) -> None:
        """Emit LLM response when debug logging is enabled."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        if isinstance(response, (dict, list)):
            try:
                payload = json.dumps(response, indent=2, default=str)
            except (TypeError, ValueError):
                payload = str(response)
        else:
            payload = str(response)

        if len(payload) > 6000:
            payload = f"{payload[:6000]}\n...[truncated]"

        self.logger.debug("📥 %s RESPONSE:\n%s", label, payload)

    def _get_current_fiscal_period(
        self, symbol: str, financials: Optional[Dict] = None, cik: Optional[str] = None
    ) -> str:
        """
        Determine current fiscal period using 2-tier strategy.

        TIER 1 (Preferred): Bulk-loaded SEC DERA tables (sec_sub_data)
        TIER 2 (Fallback): Financial data or calendar approximation

        This is CRITICAL for fiscal period-based caching (Phase 2 enhancement).
        Ensures different fiscal quarters don't overwrite each other in cache.

        Args:
            symbol: Stock ticker symbol
            financials: Optional financial data dict (if already loaded)
            cik: Optional CIK for querying bulk tables

        Returns:
            Fiscal period string in format 'YYYY-QN' (e.g., '2025-Q2')

        Examples:
            '2025-Q2' - Second quarter 2025 (from sec_sub_data)
            '2024-FY' - Full year 2024 (from sec_sub_data)
        """
        from datetime import datetime

        # DEBUG: Entry point logging
        self.logger.debug(
            "🔍 [FISCAL_PERIOD_ENTRY] %s - _get_current_fiscal_period() called with cik=%s, financials=%s",
            symbol,
            cik,
            "present" if financials else "None",
        )

        try:
            # TIER 1: Try bulk-loaded tables first (authoritative source)
            if cik:
                try:
                    from investigator.infrastructure.sec.data_strategy import get_fiscal_period_strategy

                    strategy = get_fiscal_period_strategy()  # Uses db_manager internally
                    fy, fp, adsh = strategy.get_latest_fiscal_period(symbol, cik)

                    if fy and fp:
                        self.logger.info(
                            f"Using fiscal period from bulk tables for {symbol}: " f"{fy}-{fp} (ADSH: {adsh})"
                        )
                        return f"{fy}-{fp}"

                except Exception as e:
                    self.logger.debug(f"Bulk table lookup failed for {symbol}: {e}")

            # TIER 1.5: Query sec_companyfacts_processed for latest filing metadata
            if cik:
                self.logger.info(f"🔍 TIER 1.5: Checking processed SEC filings for {symbol}")
                try:
                    from sqlalchemy import text

                    from investigator.infrastructure.database.db import get_db_manager

                    db_manager = get_db_manager()
                    with db_manager.engine.connect() as conn:
                        latest = conn.execute(
                            text(
                                """
                                SELECT fiscal_year, fiscal_period, filed_date
                                FROM sec_companyfacts_processed
                                WHERE symbol = :symbol
                                ORDER BY filed_date DESC NULLS LAST,
                                         period_end_date DESC NULLS LAST
                                LIMIT 1
                                """
                            ),
                            {"symbol": symbol.upper()},
                        ).fetchone()

                    if latest and latest.fiscal_year and latest.fiscal_period:
                        self.logger.info(
                            "✅ Using fiscal period from processed table for %s: %s-%s (filed: %s)",
                            symbol,
                            latest.fiscal_year,
                            latest.fiscal_period,
                            latest.filed_date,
                        )
                        return f"{latest.fiscal_year}-{latest.fiscal_period}"
                except Exception as e:
                    self.logger.warning(f"Processed SEC lookup failed for {symbol}: {e}", exc_info=True)

            # TIER 2A: Check if financials have fiscal period from SEC data
            if financials:
                fiscal_year = financials.get("fiscal_year") or financials.get("fy")
                fiscal_period = financials.get("fiscal_period") or financials.get("fp")

                if fiscal_year and fiscal_period:
                    # Validate it's not a future period (indicates calendar-based)
                    now = datetime.now()
                    current_year = now.year
                    current_quarter = ((now.month - 1) // 3) + 1

                    if isinstance(fiscal_period, str) and fiscal_period.startswith("Q"):
                        quarter_num = int(fiscal_period[1])
                    elif fiscal_period == "FY":
                        quarter_num = 4
                    else:
                        quarter_num = 0

                    # Accept if historical or valid current quarter
                    is_future = (fiscal_year > current_year) or (
                        fiscal_year == current_year and quarter_num >= current_quarter
                    )

                    if not is_future:
                        return f"{fiscal_year}-{fiscal_period}"
                    else:
                        self.logger.warning(
                            f"Fiscal period {fiscal_year}-{fiscal_period} appears to be "
                            f"future/current quarter (not filed yet). Using fallback."
                        )

            # TIER 2B: Calendar-based fallback (last resort)
            now = datetime.now()
            year = now.year
            month = now.month

            # Use PREVIOUS quarter (current quarter not filed yet)
            quarter = ((month - 1) // 3) + 1
            if quarter == 1:
                # If current quarter is Q1, use previous year Q4
                year -= 1
                quarter = 4
            else:
                quarter -= 1

            self.logger.warning(
                f"Using calendar-based PREVIOUS quarter {year}-Q{quarter} for {symbol}. "
                f"This is a fallback - actual fiscal periods should come from bulk tables."
            )

            return f"{year}-Q{quarter}"

        except Exception as e:
            self.logger.warning(
                f"Failed to determine fiscal period for {symbol}: {e}. " f"Using 'unknown' as fallback."
            )
            return "unknown"

    def _require_financials(self, company_data: Dict) -> Dict:
        """Ensure financial data exists and normalize field names, raising a clear error if not."""
        financials = company_data.get("financials") or {}
        if not financials:
            # FIX #4: More helpful error message
            symbol = company_data.get("symbol", "UNKNOWN")
            cik = company_data.get("cik", "unknown")
            raise ValueError(
                f"Financial statement data is unavailable for {symbol}. "
                f"Data sources checked: cache, database, SEC API. "
                f"This may indicate: (1) Invalid ticker symbol, (2) No SEC filings available, "
                f"(3) CIK resolution failure. CIK={cik}. Check logs for details."
            )

        # CRITICAL: Normalize field names to snake_case for internal consistency
        # This ensures all internal Python code uses snake_case, matching CLAUDE.md standards
        # SEC data from extractors should be converted to snake_case at source
        normalized_financials = DataNormalizer.normalize_field_names(financials, to_camel_case=False)

        return normalized_financials

    def _build_company_profile(self, symbol: str, company_data: Dict, ratios: Dict) -> CompanyProfile:
        """
        Assemble a CompanyProfile snapshot from the data already loaded by the agent.

        The profile focuses on universally available metrics so later phases can
        refine archetype detection without re-plumbing the fundamentals workflow.
        """
        self.logger.debug(f"Building company profile for {symbol}")

        financials = self._require_financials(company_data)
        market_data = company_data.get("market_data") or {}
        data_quality = company_data.get("data_quality") or {}

        # CRITICAL: Use _get_sector_for_symbol() first to respect config.yaml sector overrides
        # This ensures CompanyMetadataService priority order is followed (config override → cache → database → fallbacks)
        sector = self._get_sector_for_symbol(symbol)
        industry = market_data.get("industry") or company_data.get("industry")

        profile = CompanyProfile(symbol=symbol, sector=sector or "Unknown", industry=industry)

        # Get TTM metrics as fallback source for key financial values
        ttm_metrics = company_data.get("ttm_metrics", {})

        # Use multiple fallback sources for key financial values
        # The financials dict may not always have these keys, but ttm_metrics often does
        free_cash_flow = (
            financials.get("free_cash_flow")
            or ttm_metrics.get("free_cash_flow")
            or ttm_metrics.get("FreeCashFlow")
            or 0
        )
        revenue = (
            financials.get("revenues")
            or financials.get("total_revenue")
            or ttm_metrics.get("revenues")
            or ttm_metrics.get("total_revenue")
            or 0
        )
        net_income = (
            financials.get("net_income") or ttm_metrics.get("net_income") or ttm_metrics.get("NetIncomeLoss") or 0
        )
        ebitda = (
            financials.get("ebitda")
            or financials.get("operating_income")
            or ttm_metrics.get("ebitda")
            or ttm_metrics.get("operating_income")
            or 0
        )

        self.logger.info(
            f"{symbol} - _build_company_profile extracted values: FCF=${free_cash_flow/1e9:.2f}B, Revenue=${revenue/1e9:.2f}B, NetIncome=${net_income/1e9:.2f}B, EBITDA=${ebitda/1e9:.2f}B"
        )

        profile.has_positive_fcf = (free_cash_flow or 0) > 0
        profile.has_positive_earnings = (net_income or 0) > 0
        profile.has_positive_ebitda = (ebitda or 0) > 0
        profile.ttm_fcf = free_cash_flow
        profile.fcf_margin = (free_cash_flow / revenue) if revenue else None

        # CRITICAL FIX: Set actual values for model applicability checks
        # These are used by DynamicModelWeightingService to determine which models are applicable
        profile.free_cash_flow = free_cash_flow
        profile.ebitda = ebitda
        profile.net_income = net_income
        profile.revenue = revenue

        # Calculate revenue_growth_yoy from ratios or quarterly_data
        revenue_growth_yoy = ratios.get("revenue_growth") or ratios.get("revenue_growth_yoy")

        if revenue_growth_yoy is None:
            quarterly_data = company_data.get("quarterly_data", [])
            if quarterly_data and len(quarterly_data) >= 5:
                try:
                    from investigator.domain.agents.fundamental.models import QuarterlyData

                    revenues = []
                    for q in quarterly_data[:8]:
                        if isinstance(q, QuarterlyData):
                            rev = q.financial_data.get("revenues", 0)
                        elif isinstance(q, dict):
                            rev = q.get("financial_data", {}).get("revenues", 0) or q.get("revenues", 0)
                        else:
                            rev = 0
                        revenues.append(float(rev) if rev else 0)

                    if len(revenues) >= 5 and revenues[4] > 0:
                        revenue_growth_yoy = (revenues[0] - revenues[4]) / revenues[4]
                        self.logger.debug(
                            f"{symbol} - Calculated revenue_growth_yoy from quarterly data: {revenue_growth_yoy*100:.1f}%"
                        )
                except Exception as e:
                    self.logger.warning(f"{symbol} - Failed to calculate revenue_growth_yoy: {e}")

        profile.revenue_growth_yoy = revenue_growth_yoy
        profile.earnings_growth_yoy = ratios.get("earnings_growth") or ratios.get("earnings_growth_yoy")
        profile.revenue_volatility = ratios.get("revenue_volatility")
        profile.gross_margin_trend = ratios.get("gross_margin_trend")
        profile.gross_margin = ratios.get("gross_margin")  # Actual gross margin for P/S quality premium
        profile.net_revenue_retention = ratios.get("net_revenue_retention")  # NRR for SaaS companies
        profile.ebitda_margin_trend = ratios.get("ebitda_margin_trend")
        profile.return_on_equity = ratios.get("return_on_equity") or ratios.get("roe")
        profile.earnings_quality_score = ratios.get("earnings_quality_score")

        total_debt = financials.get("total_debt") or 0
        cash = financials.get("cash") or 0
        net_debt = total_debt - cash if total_debt is not None and cash is not None else None

        profile.net_debt_to_ebitda = (
            (net_debt / ebitda)
            if ebitda not in (None, 0) and net_debt is not None
            else ratios.get("net_debt_to_ebitda")
        )
        profile.interest_coverage = ratios.get("interest_coverage")
        profile.debt_to_equity = ratios.get("debt_to_equity") or ratios.get("debt_to_capital")

        # CRITICAL FIX: Use multiple fallback sources for dividends_paid
        # SEC data stores dividends as negative cash outflow, so we take absolute value
        dividends_paid = abs(
            financials.get("dividends_paid")
            or financials.get("PaymentsOfDividends")
            or ttm_metrics.get("dividends_paid")
            or ttm_metrics.get("PaymentsOfDividends")
            or ttm_metrics.get("payments_of_dividends")
            or 0
        )
        shares_outstanding = financials.get("shares_outstanding") or market_data.get("shares_outstanding")
        profile.pays_dividends = dividends_paid > 0
        profile.dividends_paid = dividends_paid  # CRITICAL FIX: Set actual value for GGM applicability
        self.logger.info(
            f"{symbol} - dividends_paid extracted: ${dividends_paid/1e9:.2f}B, pays_dividends={profile.pays_dividends}"
        )
        profile.dividend_yield = ratios.get("dividend_yield") or market_data.get("dividend_yield")
        profile.dividend_payout_ratio = ratios.get("payout_ratio") or ratios.get("dividend_payout_ratio")
        profile.dividend_growth_rate = ratios.get("dividend_growth_rate")

        profile.book_value_per_share = ratios.get("book_value_per_share")
        profile.shares_outstanding = (
            financials.get("shares_outstanding")
            or financials.get("shares_outstanding_diluted")
            or ratios.get("shares_outstanding")
            or company_data.get("shares_outstanding")
            or market_data.get("shares_outstanding")
        )
        cash_candidates = [
            financials.get("cash"),
            financials.get("cash_and_equivalents"),
            financials.get("cash_and_cash_equivalents"),
        ]
        profile.cash = next((float(c) for c in cash_candidates if c is not None), None)

        debt_candidates = [
            financials.get("total_debt"),
            financials.get("long_term_debt"),
            financials.get("total_liabilities"),
            market_data.get("total_debt"),
        ]
        profile.total_debt = next((float(d) for d in debt_candidates if d is not None), None)
        profile.current_price = (
            market_data.get("price")
            or market_data.get("close")
            or market_data.get("current_price")
            or ratios.get("current_price")
        )
        profile.market_cap = market_data.get("market_cap") or market_data.get("market_capitalization")
        if not profile.market_cap and profile.current_price and profile.shares_outstanding:
            try:
                profile.market_cap = float(profile.current_price) * float(profile.shares_outstanding)
            except (TypeError, ValueError):
                profile.market_cap = None

        profile.beta = market_data.get("beta") or market_data.get("five_year_beta") or ratios.get("beta")

        average_volume = (
            market_data.get("average_daily_volume")
            or market_data.get("avg_daily_volume")
            or market_data.get("three_month_avg_volume")
        )
        if average_volume and profile.current_price:
            profile.daily_liquidity_usd = float(average_volume) * float(profile.current_price)
            if profile.daily_liquidity_usd < 5_000_000:
                profile.add_flag(DataQualityFlag.LOW_LIQUIDITY)

        quarters = company_data.get("quarterly_data") or []
        profile.quarters_available = len(quarters)
        if profile.quarters_available and profile.quarters_available < 8:
            profile.add_flag(DataQualityFlag.MISSING_QUARTERS)

        dq_score = data_quality.get("data_quality_score")
        if isinstance(dq_score, (int, float)):
            profile.data_completeness_score = max(0.0, min(float(dq_score) / 100.0, 1.0))

        if data_quality.get("consistency_issues"):
            profile.add_flag(DataQualityFlag.OUTLIER_DETECTED)

        if data_quality.get("stale_data"):
            profile.add_flag(DataQualityFlag.STALE_REFERENCE_DATA)

        profile.rule_of_40_score = company_data.get("rule_of_40_score")
        profile.rule_of_40_classification = company_data.get("rule_of_40_classification")

        if shares_outstanding:
            profile.dividend_yield = profile.dividend_yield or (
                (dividends_paid / shares_outstanding) / (profile.current_price or 1) if profile.current_price else None
            )

        # CRITICAL FIX (2025-11-20): Detect primary archetype based on company characteristics
        # This archetype is used by P/S and P/B models to apply growth/quality adjustments
        # Without archetype detection, high-growth SaaS companies get no growth premium on P/S multiples
        revenue_growth = profile.revenue_growth_yoy or 0
        rule_of_40 = profile.rule_of_40_score or 0
        payout_ratio = profile.dividend_payout_ratio or 0
        pays_dividends = profile.pays_dividends or False

        if rule_of_40 > 40 or revenue_growth > 0.20:
            # High-growth companies: Rule of 40 > 40% OR revenue growth > 20%
            # Examples: SNOW (Rule of 40 = 46.6%), DASH (revenue growth = 30%+)
            profile.primary_archetype = CompanyArchetype.HIGH_GROWTH
            self.logger.info(
                f"{symbol} - Detected HIGH_GROWTH archetype (Rule of 40: {rule_of_40:.1f}%, Revenue Growth: {revenue_growth*100:.1f}%)"
            )
        elif pays_dividends and payout_ratio >= 40:
            # Dividend aristocrats: Payout ratio >= 40%
            profile.primary_archetype = CompanyArchetype.MATURE_DIVIDEND
            self.logger.info(
                f"{symbol} - Detected MATURE_DIVIDEND archetype (Payout: {payout_ratio:.1f}%)"
                if payout_ratio is not None
                else f"{symbol} - Detected MATURE_DIVIDEND archetype (Payout: N/A)"
            )
        elif sector in ["Banks", "Financial Services", "Insurance", "Financials"]:
            # Financial services companies use different valuation frameworks
            profile.primary_archetype = CompanyArchetype.FINANCIAL
            self.logger.info(f"{symbol} - Detected FINANCIAL archetype (Sector: {sector})")
        elif sector in ["Energy", "Materials", "Industrials"]:
            # Cyclical sectors - commodity-driven, use EV/EBITDA
            profile.primary_archetype = CompanyArchetype.CYCLICAL
            self.logger.info(f"{symbol} - Detected CYCLICAL archetype (Sector: {sector})")
        else:
            # No specific archetype detected
            self.logger.debug(
                f"{symbol} - No primary archetype detected (Rule of 40: {rule_of_40:.1f}%, Revenue Growth: {revenue_growth*100:.1f}%, Sector: {sector})"
            )

        return profile

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """
        Get sector classification for a symbol with caching and config override support.

        This sector information is used by CanonicalKeyMapper for sector-aware
        XBRL tag fallback chains (e.g., Utilities use different revenue tags than Technology).

        Priority order (via CompanyMetadataService):
        0. Config.yaml sector overrides (highest priority - for misclassified companies)
        1. Instance cache (avoid redundant calls)
        2. Database (sec_sector, then Sector column)
        3. Peer group JSON mapping
        4. Sector map text file
        5. Fallback to 'Unknown' (uses global fallback tags only)

        Returns:
            str: Sector name (e.g., 'Technology', 'Utilities'), or 'Unknown' if not available
        """
        # Check instance cache first
        if symbol in self._sector_cache:
            self.logger.debug(f"Using cached sector for {symbol}: {self._sector_cache[symbol]}")
            return self._sector_cache[symbol]

        # Use CompanyMetadataService for centralized sector lookup with override support
        try:
            sector = self.company_metadata_service.get_sector(symbol, use_cache=True)
            self._sector_cache[symbol] = sector
            self.logger.debug(f"CompanyMetadataService returned sector for {symbol}: {sector}")
            return sector
        except Exception as e:
            self.logger.warning(
                f"Error fetching sector via CompanyMetadataService for {symbol}: {e}, using 'Unknown' fallback"
            )
            self._sector_cache[symbol] = "Unknown"
            return "Unknown"

    def _get_shares_outstanding(self, symbol: str, cik: str) -> float:
        """
        Extract shares outstanding from SEC NUM data with caching.

        Priority order:
        1. Instance cache (avoid redundant DB queries)
        2. CommonStockSharesOutstanding (most recent filing)
        3. WeightedAverageNumberOfSharesOutstandingBasic (if above missing)
        4. EntityCommonStockSharesOutstanding (fallback)

        Returns:
            float: Shares outstanding, or 0 if not available
        """
        # Check instance cache first
        cache_key = f"{symbol}:{cik}"
        if cache_key in self._shares_cache:
            self.logger.debug(f"Using cached shares outstanding for {symbol}: {self._shares_cache[cache_key]:,.0f}")
            return self._shares_cache[cache_key]

        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db_manager = get_db_manager()
            with db_manager.get_session() as session:
                # Query for shares outstanding with priority ordering
                # Priority: EntityCommonStockSharesOutstanding (DEI - always actual shares) > CommonStock > WeightedAverage
                query = text(
                    """
                    SELECT n.value, n.ddate, n.tag
                    FROM sec_num_data n
                    JOIN sec_sub_data s ON n.adsh = s.adsh
                    WHERE s.cik = :cik
                      AND n.tag IN (
                          'EntityCommonStockSharesOutstanding',
                          'CommonStockSharesOutstanding',
                          'WeightedAverageNumberOfSharesOutstandingBasic'
                      )
                      AND n.ddate IS NOT NULL
                    ORDER BY n.ddate DESC,
                             CASE n.tag
                                 WHEN 'EntityCommonStockSharesOutstanding' THEN 1
                                 WHEN 'CommonStockSharesOutstanding' THEN 2
                                 WHEN 'WeightedAverageNumberOfSharesOutstandingBasic' THEN 3
                                 ELSE 4
                             END
                    LIMIT 1
                """
                )

                result = session.execute(query, {"cik": cik}).fetchone()

                if result and result[0]:
                    shares = float(result[0])
                    tag_used = result[2]

                    # Scale normalization: Some companies report shares in millions (value < 100,000)
                    # For large-cap companies (market cap > $1B), this is likely in millions
                    # DEI namespace (EntityCommonStockSharesOutstanding) is always in actual shares
                    if tag_used != "EntityCommonStockSharesOutstanding" and shares < 100_000:
                        # Cross-check with market data to detect millions reporting
                        stock_info = self.market_data.get_stock_info(symbol)
                        market_cap = stock_info.get("market_cap", 0)
                        price = stock_info.get("current_price") or stock_info.get("price", 0)

                        if market_cap and market_cap > 1_000_000_000:  # $1B+ market cap
                            # Shares value < 100k but market cap > $1B indicates millions reporting
                            self.logger.warning(
                                f"⚠️  {symbol}: Detected shares in millions ({shares:,.0f}) - "
                                f"normalizing to actual count (×1M). Market cap: ${market_cap/1e9:.1f}B"
                            )
                            shares = shares * 1_000_000
                        elif price and price > 0:
                            # Alternative: estimate expected shares from market_cap/price
                            implied_shares = market_cap / price if market_cap else 0
                            if implied_shares > 1_000_000 and shares < 10_000:
                                self.logger.warning(
                                    f"⚠️  {symbol}: Shares ({shares:,.0f}) seems low vs implied ({implied_shares:,.0f}) - "
                                    f"normalizing to actual count (×1M)"
                                )
                                shares = shares * 1_000_000

                    self._shares_cache[cache_key] = shares
                    self.logger.info(
                        f"Found shares outstanding for {symbol}: {shares:,.0f} shares (tag: {tag_used}, date: {result[1]})"
                    )
                    return shares

            # Fall back to market data service
            stock_info = self.market_data.get_stock_info(symbol)
            fallback_shares = stock_info.get("shares_outstanding")
            if fallback_shares:
                shares = float(fallback_shares)
                self._shares_cache[cache_key] = shares
                self.logger.info(
                    "Using market data fallback for %s shares outstanding: %s",
                    symbol,
                    f"{shares:,.0f}",
                )
                return shares

            self.logger.warning(f"No shares outstanding found for {symbol} (CIK: {cik})")
            self._shares_cache[cache_key] = 0
            return 0

        except Exception as e:
            self.logger.error(f"Error fetching shares outstanding for {symbol}: {e}")
            return 0

    def _get_public_float(self, symbol: str, cik: str) -> float:
        """
        Extract public float (EntityPublicFloat) from SEC DEI namespace.

        Public float is the portion of shares available for trading, excluding
        shares held by insiders and controlling shareholders. This is always
        reported in USD in the DEI namespace.

        Args:
            symbol: Stock ticker
            cik: Company CIK (numeric string)

        Returns:
            float: Public float in USD, or 0 if not available
        """
        # Check instance cache first
        cache_key = f"{symbol}:{cik}:float"
        if hasattr(self, "_float_cache") and cache_key in self._float_cache:
            return self._float_cache[cache_key]

        if not hasattr(self, "_float_cache"):
            self._float_cache = {}

        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db_manager = get_db_manager()
            with db_manager.get_session() as session:
                # EntityPublicFloat is in DEI namespace, reported in USD
                query = text(
                    """
                    SELECT n.value, n.ddate, n.uom
                    FROM sec_num_data n
                    JOIN sec_sub_data s ON n.adsh = s.adsh
                    WHERE s.cik = :cik
                      AND n.tag = 'EntityPublicFloat'
                      AND n.ddate IS NOT NULL
                    ORDER BY n.ddate DESC
                    LIMIT 1
                    """
                )

                result = session.execute(query, {"cik": cik}).fetchone()

                if result and result[0]:
                    public_float = float(result[0])
                    uom = result[2]
                    self._float_cache[cache_key] = public_float
                    self.logger.info(
                        f"Found public float for {symbol}: ${public_float:,.0f} (unit: {uom}, date: {result[1]})"
                    )
                    return public_float

            self.logger.debug(f"No public float found for {symbol} (CIK: {cik})")
            self._float_cache[cache_key] = 0
            return 0

        except Exception as e:
            self.logger.error(f"Error fetching public float for {symbol}: {e}")
            return 0

    def register_capabilities(self) -> List:
        """Register agent capabilities"""
        from investigator.domain.agents.base import AgentCapability, AnalysisType

        return [
            AgentCapability(
                analysis_type=AnalysisType.FUNDAMENTAL_ANALYSIS,
                min_data_required={"symbol": str},
                max_processing_time=360,  # Increased 2x for slower hardware
                required_models=[self.primary_model],
                cache_ttl=3600,
            )
        ]

    async def process(self, task: AgentTask) -> AgentResult:
        """Process fundamental analysis task"""
        symbol = task.context.get("symbol")
        analysis_depth = task.context.get("depth", "comprehensive")
        include_forecast = task.context.get("include_forecast", True)

        # DEBUG: Explicit logging to trace execution
        self.logger.debug("FundamentalAgent.process() START for %s", symbol)
        self.logger.info(f"Performing {analysis_depth} fundamental analysis for {symbol}")

        try:
            # Fetch company facts and financials
            self.logger.debug("Calling _fetch_company_data for %s", symbol)
            company_data = await self._fetch_company_data(symbol)
            self.logger.debug(
                "_fetch_company_data returned %s keys: %s",
                len(company_data),
                list(company_data.keys())[:10],
            )

            # NEW: Multi-quarter historical analysis
            try:
                # CRITICAL: Request 12 quarters (3 years) to ensure we get ≥8 after Q4 computation
                # AVGO showed only 7 quarters when requesting 8 (boundary case - Q4 not yet filed)
                quarterly_data = await self._fetch_historical_quarters(symbol, num_quarters=12)

                # CRITICAL FIX: Add quarterly_data to company_data for valuation methods
                # DCF and GGM need this data for FCF, dividends, and growth rate calculations
                company_data["quarterly_data"] = quarterly_data
                self.logger.info(
                    f"Added {len(quarterly_data)} quarters to company_data for {symbol} (for DCF/GGM valuation)"
                )

                if len(quarterly_data) >= 4:
                    # Analyze trends across quarters
                    revenue_trend = self._analyze_revenue_trend(quarterly_data)
                    margin_trend = self._analyze_margin_trend(quarterly_data)
                    cash_flow_trend = self._analyze_cash_flow_trend(quarterly_data)
                    comparisons = self._calculate_quarterly_comparisons(quarterly_data)
                    cyclical = self._detect_cyclical_patterns(quarterly_data)

                    # Add trend analysis to company_data
                    company_data["trend_analysis"] = {
                        "revenue": revenue_trend,
                        "margins": margin_trend,
                        "cash_flow": cash_flow_trend,
                        "comparisons": comparisons,
                        "cyclical": cyclical,
                        "num_quarters": len(quarterly_data),
                    }

                    self.logger.info(
                        f"Multi-quarter analysis for {symbol}: "
                        f"Revenue={revenue_trend['trend']}, "
                        f"Margins={margin_trend['net_margin_trend']}, "
                        f"Cash Quality={cash_flow_trend['quality_of_earnings']}/100, "
                        f"Cyclical={cyclical['seasonal_pattern']}"
                    )
                else:
                    self.logger.warning(f"Insufficient quarterly data for {symbol}: {len(quarterly_data)} quarters")
                    company_data["trend_analysis"] = None

            except Exception as e:
                self.logger.warning(f"Multi-quarter analysis failed for {symbol}: {e}", exc_info=True)
                company_data["trend_analysis"] = None
                company_data["quarterly_data"] = []  # Ensure empty list if extraction fails

            # Calculate financial ratios
            ratios = await self._calculate_financial_ratios(company_data)

            # CRITICAL FIX: Update company_data with calculated market_cap and shares
            # This ensures LLM prompts receive correct values (not market_cap=0, price=0)
            if "market_cap" in ratios:
                company_data["market_cap"] = ratios["market_cap"]
                self.logger.info(f"Updated company_data market_cap for {symbol}: ${ratios['market_cap']:,.0f}")
            if "shares_outstanding" in ratios:
                company_data["shares_outstanding"] = ratios["shares_outstanding"]
                self.logger.info(f"Updated company_data shares for {symbol}: {ratios['shares_outstanding']:,.0f}")
            if "current_price" in ratios:
                if "market_data" not in company_data:
                    company_data["market_data"] = {}
                company_data["market_data"]["current_price"] = ratios["current_price"]

            # FEATURE #1: Assess data quality (migrated from old solution)
            data_quality = self._assess_data_quality(company_data, ratios)
            company_data["data_quality"] = data_quality
            self.logger.info(
                f"Data quality for {symbol}: {data_quality['quality_grade']} "
                f"({data_quality['data_quality_score']:.1f}% - "
                f"{data_quality['core_metrics_populated']} core metrics)"
            )

            # FEATURE #3: Log quality improvement metrics
            if data_quality.get("quality_improvement", 0) > 0:
                self.logger.info(f"Data enrichment for {symbol}: {data_quality['enhancement_summary']}")

            # FEATURE #2: Calculate confidence level based on data quality
            confidence = self._calculate_confidence_level(data_quality)
            company_data["confidence"] = confidence
            self.logger.info(
                f"Analysis confidence for {symbol}: {confidence['confidence_level']} "
                f"({confidence['confidence_score']}/100) - {confidence['rationale']}"
            )

            # CRITICAL FIX #4: Sanitize data before LLM calls
            company_data, ratios = self._sanitize_for_llm(company_data, ratios, symbol)

            # Analyze financial health (with data quality in prompt)
            health_analysis = await self._analyze_financial_health(company_data, ratios, symbol)

            # Analyze growth metrics
            growth_analysis = await self._analyze_growth(company_data, symbol)

            # Analyze profitability
            profitability = await self._analyze_profitability(company_data, ratios, symbol)

            # Perform valuation analysis
            valuation = await self._perform_valuation(company_data, ratios, symbol)

            # Analyze competitive position
            competitive = await self._analyze_competitive_position(company_data, symbol)

            # Generate earnings forecast if requested
            forecast = None
            if include_forecast:
                forecast = await self._generate_forecast(company_data, growth_analysis, symbol)

            # Calculate quality score
            quality_score = await self._calculate_quality_score(
                health_analysis, growth_analysis, profitability, competitive
            )

            # Synthesize comprehensive report
            report = await self._synthesize_fundamental_report(
                {
                    "symbol": symbol,
                    "company_data": self._summarize_company_data(company_data),
                    "ratios": ratios,
                    "health_analysis": health_analysis,
                    "growth_analysis": growth_analysis,
                    "profitability": profitability,
                    "valuation": valuation,
                    "competitive_analysis": competitive,
                    "forecast": forecast,
                    "quality_score": quality_score,
                    "data_quality": data_quality,  # FEATURE #1: Include data quality in synthesis
                    "confidence": confidence,  # FEATURE #2: Include confidence in synthesis
                    "fiscal_period": company_data.get("fiscal_period"),  # Period for caching
                }
            )

            # Extract multi-model summary from valuation results
            # NOTE: _perform_valuation() returns wrapped response: {"response": {...}, "prompt": ..., "model_info": ..., "metadata": ...}
            # The actual data is in valuation["response"]["valuation_methods"]["multi_model"]
            multi_model_summary = {}
            llm_fair_value_estimate = 0

            # CRITICAL FIX: valuation is wrapped by _wrap_llm_response, so data is in valuation["response"]
            valuation_unwrapped = {}
            if isinstance(valuation, dict):
                response_data = valuation.get("response", {})
                if isinstance(response_data, dict):
                    valuation_unwrapped = response_data
                    # Get valuation_methods from the response data
                    valuation_methods = response_data.get("valuation_methods", {})
                    if isinstance(valuation_methods, dict):
                        multi_model_summary = valuation_methods.get("multi_model", {})

                    # Also get LLM fair value estimate from response data
                    llm_fair_value_estimate = response_data.get("fair_value_estimate", 0)

            blended_fair_value = multi_model_summary.get("blended_fair_value")

            # Use blended fair value as primary (fallback to LLM estimate if unavailable)
            primary_fair_value = blended_fair_value if blended_fair_value else llm_fair_value_estimate

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={
                    "status": "success",
                    "symbol": symbol,
                    "analysis": report,
                    "valuation": valuation_unwrapped,  # CRITICAL FIX: Store unwrapped valuation data
                    "ratios": ratios,  # FIX #2: Include calculated ratios in output
                    "quality_score": quality_score,
                    "data_quality": data_quality,  # FEATURE #1: Include in results
                    "confidence": confidence,  # FEATURE #2: Include in results
                    "investment_grade": report.get("investment_grade", "B"),
                    # PRIMARY FIX: Use blended fair value from multi-model orchestrator
                    "fair_value": primary_fair_value,
                    "llm_fair_value_estimate": llm_fair_value_estimate,  # Keep for reference
                    "multi_model_summary": multi_model_summary,  # Full multi-model data for synthesis
                    "recommendation": report.get("recommendation", "hold"),
                    "fiscal_period": company_data.get("fiscal_period"),  # Include fiscal period in output
                },
                processing_time=0,  # Will be calculated by base class
            )

        except Exception as e:
            self.logger.error(f"Fundamental analysis failed for {symbol}: {e}", exc_info=True)
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result_data={"status": "error", "symbol": symbol, "error": str(e)},
                processing_time=0,
                error=str(e),
            )

    async def recalculate_derived_metrics(self, task: AgentTask, cached_result: Dict) -> Dict:
        """
        HYBRID CACHING FIX (Phase 1):
        Recalculate deterministic metrics (CompanyProfile, ratios) from cached LLM responses.

        This method is called by base.py when cache hits occur. It ensures that:
        1. Expensive LLM responses stay cached (30-60s synthesis)
        2. Deterministic metrics are always fresh (~100ms calculation)
        3. CompanyProfile.revenue_growth_yoy is populated for P/S model

        Args:
            task: Original AgentTask with symbol
            cached_result: Cached LLM response data

        Returns:
            Enriched cached_result with fresh CompanyProfile and derived metrics
        """
        symbol = task.symbol
        self.logger.debug(f"{symbol} - Recalculating derived metrics from cached LLM response")

        try:
            # Step 1: Re-fetch company_data (cheap database queries ~100ms)
            company_data = await self._fetch_company_data(symbol)

            if not company_data or "error" in company_data:
                self.logger.warning(f"{symbol} - Failed to re-fetch company_data for metric recalculation")
                return cached_result  # Return original cached result if fetch fails

            # Step 2: Re-calculate financial ratios (deterministic, ~50ms)
            ratios = await self._calculate_financial_ratios(company_data, symbol)

            # Step 3: Re-build CompanyProfile (ensures revenue_growth_yoy is populated)
            profile = self._build_company_profile(symbol, company_data, ratios)

            # Step 4: Update cached_result with fresh CompanyProfile and ratios
            # Preserve LLM synthesis results but update deterministic metrics
            enriched_result = cached_result.copy()

            # Replace CompanyProfile in valuation section
            if "valuation" in enriched_result and isinstance(enriched_result["valuation"], dict):
                if "company_profile" in enriched_result["valuation"]:
                    # Convert CompanyProfile dataclass to dict for JSON serialization
                    from dataclasses import asdict

                    enriched_result["valuation"]["company_profile"] = asdict(profile)
                    self.logger.debug(f"{symbol} - Updated CompanyProfile in cached valuation data")

            # Update ratios (used by valuation models)
            if "ratios" in enriched_result:
                enriched_result["ratios"].update(ratios)
                self.logger.debug(f"{symbol} - Updated {len(ratios)} ratios in cached data")

            # Update company_data reference (for consistency)
            if "company_data" in enriched_result:
                enriched_result["company_data"] = company_data

            revenue_growth_str = (
                f"{profile.revenue_growth_yoy:.1%}" if profile.revenue_growth_yoy is not None else "None"
            )
            self.logger.info(
                f"{symbol} - Successfully recalculated derived metrics (revenue_growth_yoy: {revenue_growth_str})"
            )

            return enriched_result

        except Exception as e:
            self.logger.warning(f"{symbol} - Failed to recalculate derived metrics: {e}", exc_info=True)
            return cached_result  # Fallback to original cached data on error

    async def _fetch_company_data(self, symbol: str) -> Dict:
        """Fetch comprehensive company financial data.

        Data flow reference:

            ┌────────────┐     canonical mapping + normalization     ┌────────────────────────┐
            │  SEC AGENT │ ────────────────────────────────────────▶ │ sec_companyfacts_proc. │
            └─────┬──────┘                                         └────────────┬───────────┘
                  │   raw filings / company facts                                │ SELECT + safe_float()
                  v                                                              v
            ┌──────────────┐      enrichment + cache      ┌────────────────────────────────────────┐
            │ company_data │ ◀──────────────────────────── │ FundamentalAgent._fetch_company_data() │
            └──────────────┘                               └────────────────┬───────────────────────┘
                                                                            │
                                                                            ├─► company_data['financials']
                                                                            │      (used by ratio + quality logic)
                                                                            └─► quarterly_data → DCF / multiples
        """
        try:
            # FIX #1: Resolve CIK first for proper cache key
            cik = None
            fiscal_period_label = None  # Initialize for period-based caching

            try:
                cik = self.ticker_mapper.resolve_cik(symbol)
                if cik:
                    self.logger.debug(f"Resolved CIK {cik} for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to resolve CIK for {symbol}: {e}")

            # CRITICAL (Phase 2): Build cache key with fiscal_period to prevent overwrites
            # Different fiscal quarters should not overwrite each other in cache
            fiscal_period = self._get_current_fiscal_period(symbol, financials=None, cik=cik)

            cache_key = {"symbol": symbol, "fiscal_period": fiscal_period}  # Phase 2: Prevents Q1/Q2/Q3/Q4 overwrites
            if cik:
                cache_key["cik"] = cik

            self.logger.debug(
                f"Cache key for {symbol}: {cache_key} " f"(fiscal_period ensures quarter-specific caching)"
            )

            # Check cache (24 hour TTL for fundamental data)
            cached = self.cache.get(CacheType.COMPANY_FACTS, cache_key) if self.cache else None
            # FIX #3: Validate cached data has usable financials
            if cached and cached.get("financials"):
                self.logger.info(f"Using cached company data for {symbol}")
                return cached

            # CLEAN ARCHITECTURE (Phase 1.2d): Migration complete - uses processed table exclusively
            # Data source: sec_companyfacts_processed table (populated by SECDataProcessor in SEC Agent)

            try:
                # Fetch company data from processed table
                self.logger.info(f"[CLEAN ARCH] Fetching company data for {symbol} from processed table")
                processed_data = self._fetch_company_data_from_processed_table(symbol)

                if not processed_data:
                    raise ValueError(
                        f"No processed data found for {symbol}. "
                        f"Ensure SEC Agent has run successfully to populate sec_companyfacts_processed table."
                    )

                # Extract data from clean architecture
                financial_metrics = processed_data["financial_metrics"]
                financial_ratios = processed_data["financial_ratios"]
                data_source = "clean_architecture"

                self.logger.info(
                    f"[CLEAN ARCH] ✅ Successfully fetched {symbol} data from processed table "
                    f"(quality: {processed_data.get('data_quality_score', 0):.2f})"
                )

                # Extract fiscal period for period-based caching
                fiscal_year = financial_metrics.get("fiscal_year")
                fiscal_period = financial_metrics.get("fiscal_period")
                fiscal_period_label = f"{fiscal_year}-{fiscal_period}" if fiscal_year and fiscal_period else None

                # Use the extracted metrics as company_facts
                company_facts = financial_metrics

                # Convert ratios to financial_statements format for compatibility
                # IMPORTANT: This dict uses camelCase consistently (SEC CompanyFacts standard)
                # All field mappings are defined in utils/data_normalizer.py
                financial_statements = {
                    # Income Statement (camelCase)
                    "revenues": financial_metrics.get("revenues", 0),
                    "net_income": financial_metrics.get("net_income", 0),
                    "gross_profit": financial_metrics.get("gross_profit", 0),
                    "operating_income": financial_metrics.get("operating_income", 0),
                    # Balance Sheet (snake_case)
                    "total_assets": financial_metrics.get("assets", 0),
                    "stockholders_equity": financial_metrics.get("equity", 0),
                    "current_assets": financial_metrics.get("assets_current", 0),
                    "current_liabilities": financial_metrics.get("liabilities_current", 0),
                    "total_liabilities": financial_metrics.get("liabilities", 0),
                    "total_debt": financial_metrics.get("total_debt", 0),
                    "long_term_debt": financial_metrics.get("long_term_debt", 0),
                    "short_term_debt": (
                        financial_metrics.get("short_term_debt")
                        or financial_metrics.get("debt_current")
                        or self._derive_short_term_debt(financial_metrics)
                        or 0
                    ),
                    # Cash Flow Statement (camelCase) - NEWLY ADDED
                    "operating_cash_flow": financial_metrics.get("operating_cash_flow", 0),
                    "capital_expenditures": financial_metrics.get("capital_expenditures", 0),
                    "free_cash_flow": financial_metrics.get("free_cash_flow", 0),
                    "dividends_paid": financial_metrics.get("dividends_paid", 0),
                    "preferred_stock_dividends": financial_metrics.get("preferred_stock_dividends", 0),
                    # Cash and Cash Equivalents (CRITICAL FIX #2)
                    "cash_and_equivalents": financial_metrics.get("cash_and_equivalents", 0),
                    "cash": financial_metrics.get(
                        "cash",
                        financial_metrics.get("cash_and_equivalents", 0),
                    ),
                    "shares_outstanding": (
                        financial_metrics.get("shares_outstanding")
                        or financial_metrics.get("weighted_average_diluted_shares_outstanding", 0)
                    ),
                    # Pre-calculated ratios (for backwards compatibility)
                    "current_ratio": financial_ratios.get("current_ratio", 0),
                    "quick_ratio": financial_ratios.get("quick_ratio", 0),
                    "debt_to_equity": financial_ratios.get("debt_to_equity", 0),
                    "roe": financial_ratios.get("roe", 0),
                    "roa": financial_ratios.get("roa", 0),
                    "gross_margin": financial_ratios.get("gross_margin", 0),
                    "operating_margin": financial_ratios.get("operating_margin", 0),
                    "net_margin": financial_ratios.get("net_margin", 0),
                    # Data source tracking (for monitoring migration)
                    "data_source": data_source,
                }

            except ValueError as cache_error:
                # If cache miss, SEC Agent hasn't run yet (dependency violation)
                raise ValueError(
                    f"SEC Agent cache miss for {symbol}: {cache_error}. "
                    f"Ensure SEC Agent runs before Fundamental Agent."
                )
            except Exception as api_error:
                self.logger.error(
                    "Failed to hydrate company data for %s from SEC cache pipeline: %s",
                    symbol,
                    api_error,
                    exc_info=True,
                )
                # Explicitly surface migration guidance instead of silently falling back
                raise RuntimeError(
                    f"{symbol} - Clean-architecture cache miss. Please ensure the SEC agent has "
                    "persisted data via sec_companyfacts_processed before running the fundamental agent."
                ) from api_error

            # Fetch current market data
            market_data = await self.market_data.get_quote(symbol)

            company_data = {
                "symbol": symbol,
                "cik": cik,
                "facts": company_facts,
                "financials": financial_statements,
                "market_data": market_data,
                "fiscal_period": fiscal_period_label,  # Period-based caching (initialized at method start)
                "fetched_at": datetime.now().isoformat(),
            }

            # FIX #3: Validate we have usable data before returning
            if not financial_statements:
                raise ValueError(
                    f"No financial data available for {symbol} from cache, database, or SEC API. "
                    f"This may indicate: (1) Invalid ticker symbol, (2) No SEC filings available, "
                    f"(3) CIK resolution failure. CIK={'found' if cik else 'not found'}"
                )

            # Cache the successfully retrieved data
            if self.cache and company_facts:
                try:
                    self.cache.set(CacheType.COMPANY_FACTS, cache_key, company_data)
                    self.logger.debug(f"Cached company data for {symbol} with CIK {cik}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache company data: {e}")

            return company_data

        except ValueError:
            # Re-raise ValueError for missing data (already has good error message)
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch company data for {symbol}: {e}", exc_info=True)
            raise ValueError(f"Failed to fetch company data for {symbol}: {str(e)}")

    @staticmethod
    def _derive_short_term_debt(metrics: Dict[str, Any]) -> Optional[float]:
        """Infer short-term debt when only total vs long-term components are available."""
        try:
            total = metrics.get("total_debt")
            long_term = metrics.get("long_term_debt")
            if total is None or long_term is None:
                return None
            delta = float(total) - float(long_term)
            return delta if delta > 0 else None
        except (TypeError, ValueError):
            return None

    async def _fetch_historical_quarters(self, symbol: str, num_quarters: int = 12) -> List[QuarterlyData]:
        """
        Fetch historical quarterly data using HYBRID 12-quarter strategy

        Phase 9 Implementation: ALWAYS returns 12 quarters for geometric mean calculation
        - 10-12 quarters from bulk tables (sec_sub_data) - fast, ADSH-linked
        - 0-2 quarters from API - fresh, fills gaps

        This ensures consistent multi-quarter trend analysis (QoQ, YoY, 3-year geometric mean)

        Args:
            symbol: Stock ticker symbol
            num_quarters: Number of quarters to fetch (default: 12 = 3 years)

        Returns:
            List[QuarterlyData] with exactly 12 quarters, sorted chronologically (oldest → newest)
            Each quarter includes: fy, fp, adsh, revenues, assets, etc.

        Raises:
            ValueError: If insufficient quarterly data available
        """
        cache_key = build_cache_key(CacheType.QUARTERLY_METRICS, symbol=symbol, num_quarters=num_quarters)
        cached_data = self.cache.get(CacheType.QUARTERLY_METRICS, cache_key) if self.cache else None

        if cached_data:
            if isinstance(cached_data, list) and all(isinstance(q, QuarterlyData) for q in cached_data):
                self.logger.info(f"🔍 CACHE HIT: Fetched historical quarters for {symbol} from cache.")
                return cached_data
            else:
                self.logger.warning(
                    f"⚠️  Cached historical quarters for {symbol} found but is malformed "
                    f"(type: {type(cached_data)}). Invalidate and re-fetching."
                )
                # Invalidate cache entry to prevent recurring issues
                self.cache.delete(CacheType.QUARTERLY_METRICS, cache_key)
                # Proceed to fetch from DB

        self.logger.info(f"Fetching {num_quarters} quarters from processed table for {symbol}")

        try:
            cik = self.ticker_mapper.resolve_cik(symbol)
            if not cik:
                raise ValueError(f"No CIK found for {symbol}")

            from investigator.infrastructure.database.db import get_db_manager

            db_manager = get_db_manager()
            fiscal_period_service = get_fiscal_period_service()
            quarters_data = query_recent_processed_periods(
                symbol=symbol,
                num_quarters=num_quarters,
                db_manager=db_manager,
                fiscal_period_service=fiscal_period_service,
                logger=self.logger,
            )
            if not quarters_data:
                return []

            quarterly_data_list: List[QuarterlyData] = []
            bulk_strategy = None

            for q in reversed(quarters_data):
                quarter_cache_key = build_cache_key(
                    CacheType.QUARTERLY_METRICS,
                    symbol=symbol,
                    fiscal_year=q["fiscal_year"],
                    fiscal_period=q["fiscal_period"],
                    adsh=q["adsh"],
                )
                cached_quarter = (
                    self.cache.get(CacheType.QUARTERLY_METRICS, quarter_cache_key) if self.cache else None
                )
                cached_quarter = normalize_cached_quarter(
                    cached_quarter=cached_quarter,
                    quarterly_data_cls=QuarterlyData,
                    symbol=symbol,
                    fiscal_year=q["fiscal_year"],
                    fiscal_period=q["fiscal_period"],
                    logger=self.logger,
                )
                if cached_quarter and isinstance(cached_quarter, QuarterlyData):
                    quarterly_data_list.append(cached_quarter)
                    continue

                self.logger.debug(
                    "🔍 [FETCH_QUARTERS] Attempting processed table for %s %s-%s ADSH=%s...",
                    symbol,
                    q["fiscal_year"],
                    q["fiscal_period"],
                    q["adsh"][:20],
                )
                processed_data = self._fetch_from_processed_table(
                    symbol, q["fiscal_year"], q["fiscal_period"], q["adsh"]
                )

                use_processed = False
                is_ytd_cashflow = False
                is_ytd_income = False
                if processed_data:
                    processed_payload = build_financials_from_processed_data(
                        processed_data=processed_data,
                        shares_outstanding=q.get("shares_outstanding", 0),
                    )
                    if processed_payload:
                        use_processed = True
                        financial_data = processed_payload["financial_data"]
                        ratios = processed_payload["ratios"]
                        quality = processed_payload["quality"]
                        is_ytd_cashflow = processed_payload["is_ytd_cashflow"]
                        is_ytd_income = processed_payload["is_ytd_income"]
                        revenue = processed_payload["revenue"]
                        self.logger.info(
                            "✅ Using pre-processed data from sec_companyfacts_processed for %s %s-%s "
                            "(Revenue: $%.1fB, Quality: %s%%)",
                            symbol,
                            q["fiscal_year"],
                            q["fiscal_period"],
                            revenue / 1e9,
                            quality,
                        )
                    else:
                        income_statement = processed_data.get("income_statement", {})
                        revenue = income_statement.get("total_revenue", 0)
                        self.logger.warning(
                            "⚠️  Processed data for %s %s-%s has zero/missing revenue (Revenue: $%s), falling back to bulk tables (ADSH: %s)",
                            symbol,
                            q["fiscal_year"],
                            q["fiscal_period"],
                            revenue,
                            q["adsh"],
                        )

                if not use_processed:
                    self.logger.warning(
                        "⚠️  Processed data not found for %s %s-%s, falling back to bulk tables with canonical key extraction (ADSH: %s)",
                        symbol,
                        q["fiscal_year"],
                        q["fiscal_period"],
                        q["adsh"],
                    )
                    sector = self._get_sector_for_symbol(symbol)
                    if bulk_strategy is None:
                        from investigator.infrastructure.sec.data_strategy import get_fiscal_period_strategy

                        bulk_strategy = get_fiscal_period_strategy()

                    financial_data = build_financials_from_bulk_tables(
                        symbol=symbol,
                        fiscal_year=q["fiscal_year"],
                        fiscal_period=q["fiscal_period"],
                        adsh=q["adsh"],
                        sector=sector,
                        canonical_keys_needed=FALLBACK_CANONICAL_KEYS,
                        canonical_mapper=self.canonical_mapper,
                        strategy=bulk_strategy,
                        logger=self.logger,
                    )
                    ratios = self._calculate_quarterly_ratios(financial_data)
                    quality = self._assess_quarter_quality(financial_data)

                qdata = QuarterlyData(
                    fiscal_year=q["fiscal_year"],
                    fiscal_period=q["fiscal_period"],
                    financial_data=financial_data,
                    ratios=ratios,
                    data_quality=quality,
                    filing_date=str(q["filed"]),
                    is_ytd_cashflow=is_ytd_cashflow,
                    is_ytd_income=is_ytd_income,
                )
                qdata.adsh = q["adsh"]
                qdata.period_end_date = str(q["period_end"]) if q["period_end"] else None
                qdata.form = q["form"]
                quarterly_data_list.append(qdata)

                self.logger.debug(
                    "📊 [FETCH_QUARTERS] Created QuarterlyData for %s %s-%s: OCF=$%.2fB, CapEx=$%.2fB, Quality=%s%%",
                    symbol,
                    q["fiscal_year"],
                    q["fiscal_period"],
                    financial_data.get("operating_cash_flow", 0) / 1e9,
                    abs(financial_data.get("capital_expenditures", 0)) / 1e9,
                    quality,
                )
                if self.cache:
                    self.cache.set(CacheType.QUARTERLY_METRICS, quarter_cache_key, qdata)
                    self.logger.debug(
                        "Cached quarter %s %s-%s (ADSH: %s)",
                        symbol,
                        q["fiscal_year"],
                        q["fiscal_period"],
                        q["adsh"],
                    )

            log_quarterly_snapshot(self.logger, symbol, quarterly_data_list)
            self.logger.info(
                "Successfully fetched %s quarters for %s using hybrid strategy: %s → %s",
                len(quarterly_data_list),
                symbol,
                quarterly_data_list[0].period_label,
                quarterly_data_list[-1].period_label,
            )
            return quarterly_data_list

        except ValueError:
            # Re-raise ValueError for no data
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch historical quarters for {symbol}: {e}")
            raise ValueError(f"Failed to fetch historical quarters for {symbol}: {str(e)}")

    def _fetch_company_data_from_processed_table(self, symbol: str) -> Optional[Dict]:
        """
        Fetch latest company-level data from sec_companyfacts_processed table (CLEAN ARCHITECTURE).

        This replaces the old extractor (utils/sec_companyfacts_extractor.py) with a direct
        database query for company-level financial data.

        Migration: Phase 1.2b (Company-Level Data Path)
        - Reads from sec_companyfacts_processed table (populated by SECDataProcessor)
        - Returns same structure as old extractor for downstream compatibility
        - Prefers FY data, falls back to most recent quarter

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with financial_metrics and financial_ratios matching old extractor output,
            or None if no data found
        """
        try:
            from investigator.infrastructure.database.db import get_db_manager

            db_manager = get_db_manager()
            return fetch_latest_company_data_from_processed_table(
                symbol=symbol,
                db_manager=db_manager,
                logger=self.logger,
                processed_additional_financial_keys=PROCESSED_ADDITIONAL_FINANCIAL_KEYS,
                processed_ratio_keys=PROCESSED_RATIO_KEYS,
            )

        except Exception as e:
            self.logger.error(f"[CLEAN ARCH] Failed to fetch company data from processed table for {symbol}: {e}")
            return None

    def _fetch_from_processed_table(
        self, symbol: str, fiscal_year: int, fiscal_period: str, adsh: str
    ) -> Optional[Dict]:
        """
        Fetch pre-processed quarterly data from sec_companyfacts_processed table (3-table architecture)

        This provides fast access to already-extracted financial data and pre-calculated ratios,
        avoiding the need to parse raw us-gaap structure or query bulk tables.

        Args:
            symbol: Stock ticker
            fiscal_year: Fiscal year
            fiscal_period: Fiscal period (Q1, Q2, Q3, Q4, FY)
            adsh: Accession number (unique filing identifier)

        Returns:
            Dictionary with financial_data, ratios, and quality, or None if not found
        """
        try:
            from investigator.infrastructure.database.db import get_db_manager

            engine = get_db_manager().engine
            fiscal_period_service = get_fiscal_period_service()
            return fetch_processed_quarter_payload(
                symbol=symbol,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                adsh=adsh,
                engine=engine,
                fiscal_period_service=fiscal_period_service,
                logger=self.logger,
            )

        except Exception as e:
            self.logger.warning(f"Error fetching from processed table for {symbol}: {e}")
            return None

    def _calculate_quarterly_ratios(self, financial_data: Dict) -> Dict:
        """
        Calculate financial ratios for a single quarter.

        Handles both float and Decimal types (bulk tables return Decimal).

        Args:
            financial_data: Financial metrics for the quarter

        Returns:
            Dictionary of calculated ratios
        """
        from decimal import Decimal

        # Helper to convert Decimal to float
        def to_float(val):
            if isinstance(val, Decimal):
                return float(val)
            return val if val else 0

        ratios = {}

        revenue = to_float(financial_data.get("revenues", 0))
        net_income = to_float(financial_data.get("net_income", 0))
        assets = to_float(financial_data.get("total_assets", 0))
        liabilities = to_float(financial_data.get("total_liabilities", 0))
        equity = to_float(financial_data.get("stockholders_equity", 0))
        ocf = to_float(financial_data.get("operating_cash_flow", 0))
        capex = to_float(financial_data.get("capital_expenditures", 0))
        total_debt = to_float(financial_data.get("total_debt", 0))

        # Profitability ratios
        if revenue > 0:
            ratios["profit_margin"] = (float(net_income) / float(revenue)) * 100
            ratios["revenue_per_asset"] = float(revenue) / float(assets) if assets > 0 else 0

        # Efficiency ratios
        if assets > 0:
            ratios["roa"] = (float(net_income) / float(assets)) * 100

        if equity > 0:
            ratios["roe"] = (float(net_income) / float(equity)) * 100

        # Solvency ratios (FIXED: Use total_debt instead of total_liabilities)
        if assets > 0:
            ratios["debt_to_assets"] = (float(total_debt) / float(assets)) * 100

        if equity > 0:
            ratios["debt_to_equity"] = (float(total_debt) / float(equity)) * 100

        # Cash flow ratios
        if net_income > 0:
            ratios["cash_conversion"] = (float(ocf) / float(net_income)) * 100

        ratios["free_cash_flow"] = float(ocf) - float(capex)

        return ratios

    def _assess_quarter_quality(self, financial_data: Dict) -> Dict:
        """Delegate single-quarter quality checks to DataQualityAssessor."""
        return self._get_data_quality_assessor().assess_quarter_quality(financial_data)

    def _get_data_quality_assessor(self):
        """Lazily resolve data quality assessor to keep agent methods thin and testable."""
        assessor = getattr(self, "_data_quality_assessor", None)
        if assessor is None:
            assessor = get_data_quality_assessor(getattr(self, "logger", None))
            self._data_quality_assessor = assessor
        return assessor

    def _get_trend_analyzer(self):
        """Lazily resolve trend analyzer to keep agent methods thin and testable."""
        analyzer = getattr(self, "_trend_analyzer", None)
        if analyzer is None:
            analyzer = get_trend_analyzer(getattr(self, "logger", None))
            self._trend_analyzer = analyzer
        return analyzer

    def _get_deterministic_analyzer(self) -> DeterministicAnalyzer:
        """Lazily resolve per-agent deterministic analyzer for rule-based sub-analyses."""
        analyzer = getattr(self, "_deterministic_analyzer", None)
        if analyzer is None:
            analyzer = DeterministicAnalyzer(
                agent_id=getattr(self, "agent_id", "fundamental_analysis"),
                logger=getattr(self, "logger", None),
            )
            self._deterministic_analyzer = analyzer
        return analyzer

    def _analyze_revenue_trend(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """Delegate revenue trend analysis to dedicated TrendAnalyzer service."""
        return self._get_trend_analyzer().analyze_revenue_trend(quarterly_data)

    def _analyze_margin_trend(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """Delegate margin trend analysis to dedicated TrendAnalyzer service."""
        return self._get_trend_analyzer().analyze_margin_trend(quarterly_data)

    def _analyze_cash_flow_trend(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """Delegate cash flow trend analysis to dedicated TrendAnalyzer service."""
        return self._get_trend_analyzer().analyze_cash_flow_trend(quarterly_data)

    def _calculate_quarterly_comparisons(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """Delegate quarterly comparisons to dedicated TrendAnalyzer service."""
        return self._get_trend_analyzer().calculate_quarterly_comparisons(quarterly_data)

    def _detect_cyclical_patterns(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """Delegate cyclical pattern detection to dedicated TrendAnalyzer service."""
        return self._get_trend_analyzer().detect_cyclical_patterns(quarterly_data)

    async def _calculate_financial_ratios(self, company_data: Dict) -> Dict:
        """Calculate comprehensive financial ratios"""
        financials = self._require_financials(company_data)
        market_data = company_data["market_data"]

        # Extract symbol and CIK from company_data
        symbol = company_data.get("symbol", "UNKNOWN")
        cik = company_data.get("cik", "")

        ratios = {}

        # DEBUG: Method entry point
        self.logger.info(f"RATIOS_CALC_DEBUG - _calculate_financial_ratios() called for {symbol}")
        self.logger.info(f"RATIOS_CALC_DEBUG - company_data keys: {list(company_data.keys())}")
        quarterly_data_check = company_data.get("quarterly_data", [])
        self.logger.info(
            f"RATIOS_CALC_DEBUG - quarterly_data exists: {quarterly_data_check is not None}, length: {len(quarterly_data_check) if quarterly_data_check else 0}"
        )

        # Valuation ratios
        if market_data and financials:
            # Use current_price from market_data (not 'price')
            price = market_data.get("current_price", market_data.get("price", 0))

            # Extract shares outstanding from SEC database
            # Priority: EntityCommonStockSharesOutstanding (DEI) > CommonStock > WeightedAverage
            shares = self._get_shares_outstanding(symbol, cik)

            # Also extract public float for liquidity analysis
            public_float_usd = self._get_public_float(symbol, cik)
            if public_float_usd > 0:
                ratios["public_float_usd"] = public_float_usd
                # Store current price for float_shares calculation
                ratios["current_price"] = price

            # Fallback strategy if shares not in database
            if shares == 0:
                # Try to estimate from equity and price
                estimated_equity = financials.get("stockholders_equity") or 0
                if price > 0 and estimated_equity > 0:
                    shares = estimated_equity / price
                    self.logger.info(f"Estimated shares for {symbol} from equity/price: {shares:,.0f}")

            # Last resort fallback
            if shares == 0:
                shares = 1
                self.logger.warning(f"Using shares=1 for {symbol} - per-share metrics will be inaccurate")

            # Calculate market cap properly (use price * shares if available)
            if price > 0 and shares > 1:
                market_cap = price * shares
                self.logger.info(
                    f"Calculated market cap for {symbol}: ${market_cap:,.0f} ({price:.2f} × {shares:,.0f})"
                )
            else:
                # Fallback: use total equity as proxy for market cap
                market_cap = financials.get("stockholders_equity") or 0
                self.logger.warning(f"Using total equity as market cap proxy for {symbol}: ${market_cap:,.0f}")

            # P/E Ratio - CRITICAL: Use TTM (Trailing Twelve Months) net income, not quarterly
            # Calculate TTM net income from last 4 quarters (if available)
            quarterly_data = company_data.get("quarterly_data", [])
            ttm_net_income = self._calculate_ttm_net_income(quarterly_data, symbol)

            # Fallback to latest period net income if TTM calculation fails
            earnings = ttm_net_income if ttm_net_income > 0 else (financials.get("net_income") or 0)

            if earnings > 0 and market_cap > 0:
                ratios["pe_ratio"] = float(market_cap) / float(earnings)
                ratios["eps"] = float(earnings) / float(shares) if shares > 0 else 0

                # Debug logging to track TTM vs quarterly EPS
                if ttm_net_income > 0:
                    quarterly_ni = financials.get("net_income") or 0
                    self.logger.info(
                        f"{symbol} - Using TTM net income for EPS: ${ttm_net_income:,.0f} "
                        f"(vs quarterly: ${quarterly_ni:,.0f}) → EPS=${ratios['eps']:.2f}"
                    )
                else:
                    self.logger.warning(f"{symbol} - TTM net income not available, falling back to quarterly for EPS")

            # Price to Book
            book_value = financials.get("stockholders_equity") or 0
            if book_value > 0 and market_cap > 0:
                ratios["price_to_book"] = float(market_cap) / float(book_value)
                ratios["book_value_per_share"] = float(book_value) / float(shares) if shares > 0 else 0

            # Price to Sales
            revenue = financials.get("revenues") or 0
            if revenue > 0 and market_cap > 0:
                ratios["price_to_sales"] = float(market_cap) / float(revenue)
                # Calculate revenue_per_share for P/S valuation model
                ratios["revenue_per_share"] = float(revenue) / float(shares) if shares > 0 else 0

            # PEG Ratio
            growth_rate = self._calculate_growth_rate(financials, "net_income")
            if ratios.get("pe_ratio") and growth_rate > 0:
                ratios["peg_ratio"] = ratios["pe_ratio"] / (growth_rate * 100)

        # Liquidity ratios
        current_assets = financials.get("current_assets") or 0
        current_liabilities = financials.get("current_liabilities") or 0
        inventory = financials.get("inventory") or 0

        ratios["current_ratio"] = current_assets / current_liabilities if current_liabilities > 0 else 0
        ratios["quick_ratio"] = (current_assets - inventory) / current_liabilities if current_liabilities > 0 else 0

        # Leverage ratios
        total_debt = financials.get("total_debt") or 0
        total_equity = financials.get("stockholders_equity") or 0
        total_assets = financials.get("total_assets") or 0

        ratios["debt_to_equity"] = total_debt / total_equity if total_equity > 0 else 0
        ratios["debt_to_assets"] = total_debt / total_assets if total_assets > 0 else 0

        # Profitability ratios
        net_income = financials.get("net_income") or 0
        revenue = financials.get("revenues") or 0
        gross_profit = financials.get("gross_profit") or 0
        operating_income = financials.get("operating_income") or 0

        ratios["roe"] = net_income / total_equity if total_equity > 0 else 0
        ratios["roa"] = net_income / total_assets if total_assets > 0 else 0
        ratios["gross_margin"] = gross_profit / revenue if revenue > 0 else 0
        ratios["operating_margin"] = operating_income / revenue if revenue > 0 else 0
        ratios["net_margin"] = net_income / revenue if revenue > 0 else 0

        # Efficiency ratios
        ratios["asset_turnover"] = revenue / total_assets if total_assets > 0 else 0
        ratios["inventory_turnover"] = (financials.get("cost_of_revenue") or 0) / inventory if inventory > 0 else 0

        # Cash flow ratios
        operating_cash_flow = financials.get("operating_cash_flow") or 0
        capex = financials.get("capital_expenditures") or 0
        free_cash_flow = operating_cash_flow - abs(capex)

        ratios["operating_cash_flow"] = operating_cash_flow
        ratios["free_cash_flow"] = free_cash_flow
        ratios["fcf_yield"] = free_cash_flow / market_cap if market_cap > 0 else 0

        # Dividend metrics (common + preferred)
        common_divs = abs(financials.get("dividends_paid", 0) or 0)
        preferred_divs = abs(financials.get("preferred_stock_dividends", 0) or 0)
        total_dividends = common_divs + preferred_divs
        ratios["dividend_yield"] = total_dividends / price if price > 0 else 0
        # payout_ratio stored as ratio (0.0 to 1.0), not percentage
        ratios["payout_ratio"] = total_dividends / net_income if net_income > 0 else 0

        # CRITICAL FIX: Include market_cap and shares in ratios dict for company_data update
        # This ensures LLM prompts receive correct values (not market_cap=0)
        if market_data and financials:
            ratios["market_cap"] = market_cap
            ratios["shares_outstanding"] = shares
            ratios["current_price"] = price

        # CRITICAL FIX: Calculate revenue_growth_yoy from quarterly data BEFORE _build_company_profile
        # This fixes the P/S granular calculation which requires revenue_growth_yoy in CompanyProfile
        quarterly_data = company_data.get("quarterly_data", [])
        self.logger.info(
            f"REVENUE_GROWTH_DEBUG - quarterly_data exists: {quarterly_data is not None}, length: {len(quarterly_data) if quarterly_data else 0}"
        )
        if quarterly_data and len(quarterly_data) >= 5:
            # Calculate YoY revenue growth (comparing most recent quarter to 4 quarters ago)
            try:
                from investigator.domain.agents.fundamental.models import QuarterlyData

                # CRITICAL FIX: Filter out QFY (Full Year) periods and sort in descending order (newest first)
                quarterly_only = []
                for q in quarterly_data:
                    if isinstance(q, QuarterlyData):
                        period = q.fiscal_period if hasattr(q, "fiscal_period") else None
                        fiscal_year = q.fiscal_year if hasattr(q, "fiscal_year") else 0
                    elif isinstance(q, dict):
                        period = q.get("fiscal_period")
                        fiscal_year = q.get("fiscal_year", 0)
                    else:
                        continue

                    # Filter: Only include quarterly periods (Q1, Q2, Q3, Q4), exclude QFY
                    if period and isinstance(period, str) and period.startswith("Q") and period not in ["QFY", "FY"]:
                        quarterly_only.append((q, fiscal_year, period))

                # Sort by fiscal year (descending) then by quarter (Q4 > Q3 > Q2 > Q1)
                quarter_order = {"Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
                quarterly_sorted = sorted(
                    quarterly_only, key=lambda x: (x[1], quarter_order.get(x[2], 0)), reverse=True  # Newest first
                )

                # Get revenues from most recent quarters
                revenues = []
                periods = []  # Track fiscal periods for diagnostic logging
                for q, fy, period in quarterly_sorted[:8]:  # Use up to 8 quarters for YoY calculation
                    if isinstance(q, QuarterlyData):
                        rev = q.financial_data.get("revenues", 0)
                    elif isinstance(q, dict):
                        rev = q.get("financial_data", {}).get("revenues", 0) or q.get("revenues", 0)
                    else:
                        rev = 0
                    revenues.append(float(rev) if rev else 0)
                    periods.append(f"{fy}-{period}")

                # Calculate YoY growth (current vs 4 quarters ago)
                if len(revenues) >= 5 and revenues[4] > 0:
                    yoy_growth = (revenues[0] - revenues[4]) / revenues[4]
                    ratios["revenue_growth_yoy"] = yoy_growth
                    ratios["revenue_growth"] = yoy_growth  # Also add as "revenue_growth" for compatibility
                    self.logger.info(f"Calculated revenue_growth_yoy from quarterly data: {yoy_growth*100:.1f}%")
            except Exception as e:
                self.logger.warning(f"Failed to calculate revenue_growth_yoy from quarterly data: {e}")

        return ratios

    def _assess_data_quality(self, company_data: Dict, ratios: Dict) -> Dict:
        """Delegate comprehensive data quality scoring to DataQualityAssessor."""
        return self._get_data_quality_assessor().assess_data_quality(company_data, ratios)

    def _calculate_confidence_level(self, data_quality: Dict) -> Dict:
        """Delegate confidence mapping to DataQualityAssessor."""
        return self._get_data_quality_assessor().calculate_confidence_level(data_quality)

    def _sanitize_for_llm(self, company_data: Dict, ratios: Dict, symbol: str) -> tuple:
        """
        Sanitize data before sending to LLM prompts.

        CRITICAL FIX #4: Validates and fixes data quality issues that could lead to
        incorrect analysis (market_cap=0, price=0, ratio inconsistencies).

        Args:
            company_data: Company data dict
            ratios: Financial ratios dict
            symbol: Stock symbol for logging

        Returns:
            Tuple of (sanitized_company_data, sanitized_ratios)
        """
        # Fix $0 market cap (CRITICAL FIX #1)
        if company_data.get("market_cap", 0) == 0 and ratios.get("market_cap", 0) > 0:
            company_data["market_cap"] = ratios["market_cap"]
            self.logger.warning(f"⚠️  {symbol}: Backfilled market_cap from ratios: ${company_data['market_cap']:,.0f}")

        # Fix $0 price
        current_price = company_data.get("market_data", {}).get("current_price", 0)
        if current_price == 0 and ratios.get("current_price", 0) > 0:
            if "market_data" not in company_data:
                company_data["market_data"] = {}
            company_data["market_data"]["current_price"] = ratios["current_price"]
            self.logger.warning(f"⚠️  {symbol}: Backfilled price from ratios: ${ratios['current_price']:.2f}")

        # Fix $0 shares outstanding
        if company_data.get("shares_outstanding", 0) == 0 and ratios.get("shares_outstanding", 0) > 0:
            company_data["shares_outstanding"] = ratios["shares_outstanding"]
            self.logger.warning(f"⚠️  {symbol}: Backfilled shares from ratios: {ratios['shares_outstanding']:,.0f}")

        # Validate and normalize leverage ratios using underlying financial totals
        financials = company_data.get("financials") or {}

        def _extract_numeric(*keys: str) -> Optional[float]:
            for key in keys:
                if key in financials:
                    value = financials.get(key)
                    if value is None:
                        continue
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return None

        total_debt = _extract_numeric("total_debt")
        if total_debt is None:
            long_term_debt = _extract_numeric("long_term_debt")
            short_term_debt = _extract_numeric("short_term_debt", "debt_current")
            if long_term_debt is not None or short_term_debt is not None:
                total_debt = (long_term_debt or 0.0) + (short_term_debt or 0.0)

        total_equity = _extract_numeric("stockholders_equity", "equity")
        total_assets = _extract_numeric("total_assets", "assets")

        leverage_abs_tol = 0.05  # Absolute tolerance for ratio differences
        leverage_rel_tol = 0.05  # Relative tolerance (5%)

        def _normalize_ratio(name: str, recomputed: Optional[float]):
            if recomputed is None or not math.isfinite(recomputed):
                return

            existing_raw = ratios.get(name)
            needs_log = False

            if existing_raw is None:
                needs_log = recomputed != 0
                existing_value = None
            else:
                try:
                    existing_value = float(existing_raw)
                except (TypeError, ValueError):
                    existing_value = None
                    needs_log = True
                else:
                    if not math.isfinite(existing_value):
                        needs_log = True
                    elif abs(existing_value - recomputed) > max(leverage_abs_tol, leverage_rel_tol * abs(recomputed)):
                        needs_log = True

            if needs_log:
                self.logger.warning(
                    f"⚠️  {symbol}: Normalized {name.replace('_', ' ')} "
                    f"from {existing_raw if existing_raw is not None else 'missing'} "
                    f"to {recomputed:.3f} using core financials."
                )

            ratios[name] = recomputed

        if total_debt is not None and total_equity and total_equity != 0:
            _normalize_ratio("debt_to_equity", total_debt / total_equity)

        if total_debt is not None and total_assets and total_assets != 0:
            _normalize_ratio("debt_to_assets", total_debt / total_assets)

        # Validate quick_ratio <= current_ratio
        current_ratio = ratios.get("current_ratio", 0)
        quick_ratio = ratios.get("quick_ratio", 0)

        if quick_ratio > current_ratio and current_ratio > 0:
            self.logger.warning(
                f"⚠️  {symbol}: Invalid ratios - quick_ratio ({quick_ratio:.2f}) > current_ratio ({current_ratio:.2f}). "
                f"Adjusting quick_ratio to equal current_ratio."
            )
            ratios["quick_ratio"] = current_ratio

        # Log data quality issues for monitoring (CRITICAL FIX #6)
        log_data_quality_issues(self.logger, symbol, company_data, ratios)

        return company_data, ratios

    def _log_data_quality_issues(self, symbol: str, company_data: Dict, ratios: Dict):
        """Compatibility shim that delegates to shared logging helpers."""
        log_data_quality_issues(self.logger, symbol, company_data, ratios)

    def _format_trend_context(self, company_data: Dict) -> str:
        return format_trend_context(company_data)

    def _log_quarterly_snapshot(self, symbol: str, quarterly_data: List["QuarterlyData"]) -> None:
        """
        Backwards-compatible hook for legacy callers; prefer log_quarterly_snapshot helper.
        """
        log_quarterly_snapshot(self.logger, symbol, quarterly_data)

    def _log_valuation_snapshot(self, symbol: str, valuation_results: Dict[str, Any]) -> None:
        """Compatibility shim for legacy callers."""
        log_valuation_snapshot(self.logger, symbol, valuation_results)

    def _build_deterministic_response(self, label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structure consistent with _wrap_llm_response for rule-based analyses."""
        return build_deterministic_response(self.agent_id, label, payload)

    def _store_deterministic_analysis(
        self,
        *,
        symbol: str,
        label: str,
        payload: Dict[str, Any],
        period: Optional[str],
    ) -> None:
        """Persist deterministic analyses in the LLM cache for downstream reuse."""
        if not self.cache or not isinstance(payload, dict):
            return

        cache_key, wrapped = build_deterministic_cache_record(
            symbol=symbol,
            agent_id=self.agent_id,
            label=label,
            payload=payload,
            period=period,
        )

        try:
            self.cache.set(CacheType.LLM_RESPONSE, cache_key, wrapped)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to store deterministic %s for %s: %s", label, symbol, exc)

    async def _analyze_financial_health(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """Delegate deterministic financial-health analysis to specialized analyzer."""
        return await self._get_deterministic_analyzer().analyze_financial_health(company_data, ratios, symbol)

    async def _analyze_growth(self, company_data: Dict, symbol: str) -> Dict:
        """Delegate deterministic growth analysis to specialized analyzer."""
        return await self._get_deterministic_analyzer().analyze_growth(company_data, symbol)

    async def _analyze_profitability(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """Delegate deterministic profitability analysis to specialized analyzer."""
        return await self._get_deterministic_analyzer().analyze_profitability(company_data, ratios, symbol)

    def _hydrate_cost_of_capital_inputs(
        self,
        profile: CompanyProfile,
        company_data: Dict[str, Any],
        ratios: Dict[str, Any],
        symbol: str,
    ) -> None:
        """Populate missing beta/debt inputs with readily available data."""
        market_data = company_data.get("market_data", {})
        stock_info: Dict[str, Any] = {}
        try:
            stock_info = self.market_data.get_stock_info(symbol)
        except Exception:
            stock_info = {}

        if profile.beta is None:
            fallback_beta = (
                ratios.get("beta") or market_data.get("beta") or company_data.get("beta") or stock_info.get("beta")
            )
            try:
                if fallback_beta is not None:
                    profile.beta = float(fallback_beta)
            except (TypeError, ValueError):
                pass

        financials = self._require_financials(company_data)
        if profile.total_debt is None:
            total_debt = financials.get("total_debt")
            if total_debt is None:
                lt = financials.get("long_term_debt")
                st = financials.get("short_term_debt")
                if lt is not None or st is not None:
                    total_debt = (lt or 0) + (st or 0)
            if total_debt is not None:
                profile.total_debt = total_debt

        if profile.interest_coverage is None:
            coverage = ratios.get("interest_coverage") or company_data.get("interest_coverage")
            if coverage is not None:
                profile.interest_coverage = coverage

        # Ensure we retain interest expense when coverage is missing
        if financials.get("interest_expense") in (None, 0):
            derived_interest = company_data.get("facts", {}).get("interest_expense")
            if derived_interest:
                financials["interest_expense"] = derived_interest

    def _evaluate_cost_of_capital_inputs(
        self,
        profile: CompanyProfile,
        company_data: Dict[str, Any],
    ) -> List[str]:
        """Identify missing inputs that force the DCF to fall back to defaults."""
        issues: List[str] = []
        if profile.beta is None:
            issues.append("missing_beta")

        financials = self._require_financials(company_data)
        total_debt = profile.total_debt
        if total_debt is None:
            total_debt = financials.get("total_debt")
        interest_expense = financials.get("interest_expense")
        if (total_debt or 0) > 0 and not interest_expense:
            issues.append("missing_interest_expense")
        if (total_debt or 0) > 0 and not profile.interest_coverage:
            issues.append("missing_interest_coverage")
        return issues

    def _apply_cost_of_capital_penalty(
        self,
        valuation_dict: Dict[str, Any],
        issues: List[str],
    ) -> Dict[str, Any]:
        """Reduce confidence when DCF had to assume default WACC inputs."""
        if not issues or not isinstance(valuation_dict, dict):
            return valuation_dict
        if not valuation_dict.get("applicable", True):
            return valuation_dict

        penalty = min(0.15 * len(issues), 0.45)
        current_confidence = valuation_dict.get("confidence_score") or 0.0
        valuation_dict["confidence_score"] = max(0.0, current_confidence - penalty)

        diagnostics = valuation_dict.get("diagnostics") or {}
        flags = diagnostics.get("flags") or []
        for issue in issues:
            flag = f"COST_INPUT_{issue.upper()}"
            if flag not in flags:
                flags.append(flag)
        diagnostics["flags"] = flags
        diag_score = diagnostics.get("data_quality_score")
        if isinstance(diag_score, (int, float)):
            diagnostics["data_quality_score"] = max(0.0, diag_score - (penalty * 100))
        else:
            diagnostics["data_quality_score"] = max(0.0, 100 - (penalty * 100))
        valuation_dict["diagnostics"] = diagnostics

        metadata = valuation_dict.get("metadata") or {}
        metadata["cost_of_capital_issues"] = issues
        valuation_dict["metadata"] = metadata
        return valuation_dict

    def _calculate_cost_of_equity(self, symbol: str) -> float:
        """
        Calculate Cost of Equity using CAPM for Gordon Growth Model

        Formula: Re = Rf + β × (Rm - Rf)

        Args:
            symbol: Stock ticker symbol

        Returns:
            Cost of equity as decimal (e.g., 0.10 for 10%)
        """
        try:
            from investigator.infrastructure.database.market_data import get_market_data_fetcher

            fetcher = get_market_data_fetcher(self.config)
            info = fetcher.get_stock_info(symbol)
            raw_beta = info.get("beta", 1.0) or 1.0

            # CRITICAL FIX: Apply beta bounds to handle data quality issues
            # Very low beta (< 0.3) is likely low R² / statistically insignificant
            # Very high beta (> 2.5) may be distorted by outliers
            BETA_FLOOR = 0.50  # Minimum reasonable beta for cost of equity
            BETA_CAP = 2.50  # Maximum reasonable beta

            beta_adjustment = None
            if raw_beta < BETA_FLOOR:
                beta_adjustment = f"low_beta_floor ({raw_beta:.2f} → {BETA_FLOOR})"
                beta = BETA_FLOOR
            elif raw_beta > BETA_CAP:
                beta_adjustment = f"high_beta_cap ({raw_beta:.2f} → {BETA_CAP})"
                beta = BETA_CAP
            else:
                beta = raw_beta

            # Get risk-free rate from FRED (10Y Treasury)
            from investigator.infrastructure.external.fred import MacroIndicatorsFetcher

            macro_fetcher = MacroIndicatorsFetcher()
            indicators = macro_fetcher.get_latest_indicators(["DGS10"])

            if "DGS10" in indicators and indicators["DGS10"]["value"] is not None:
                risk_free_rate = indicators["DGS10"]["value"] / 100  # Convert % to decimal
            else:
                risk_free_rate = 0.045  # Default 4.5%

            market_risk_premium = 0.07  # 7% historical equity premium

            cost_of_equity = risk_free_rate + beta * market_risk_premium

            log_msg = (
                f"{symbol} - Cost of Equity (CAPM): {cost_of_equity*100:.2f}% "
                f"(Rf={risk_free_rate*100:.2f}%, Beta={beta:.2f}, MRP={market_risk_premium*100:.0f}%)"
            )
            if beta_adjustment:
                log_msg += f" [adjustment: {beta_adjustment}]"
            self.logger.info(log_msg)

            return cost_of_equity
        except Exception as e:
            self.logger.warning(f"{symbol} - Error calculating cost of equity: {e}, using default 10%")
            return 0.10

    async def _calculate_dcf_professional(
        self,
        symbol: str,
        quarterly_data: List[Dict],
        company_profile: CompanyProfile,
    ) -> Dict:
        """
        Calculate DCF valuation using professional DCFValuation module with WACC

        Uses:
        - Free Cash Flow (Operating Cash Flow - CapEx)
        - WACC with levered beta from symbol table
        - 10Y Treasury rate from FRED
        - 3-5 year projections with terminal value

        Args:
            symbol: Stock ticker symbol
            quarterly_data: List of quarterly financial data (8 quarters from hybrid strategy)

        Returns:
            DCF valuation result dict with fair_value_per_share, upside_downside_pct, assumptions
        """
        try:
            from investigator.infrastructure.database.db import get_db_manager

            # Hybrid strategy already provides 8 quarters (2 years) of data
            # DCF module will aggregate and project forward
            # Convert QuarterlyData objects to dicts if needed
            quarterly_metrics = [q.to_dict() if isinstance(q, QuarterlyData) else q for q in quarterly_data]
            multi_year_data = []  # DCF will aggregate from quarterly_metrics

            db_manager = get_db_manager()

            model = DCFValuation(
                symbol=symbol,
                quarterly_metrics=quarterly_metrics,
                multi_year_data=multi_year_data,
                db_manager=db_manager,
            )
            result = model.calculate_dcf_valuation()

            if result.get("applicable", True) and (result.get("fair_value_per_share") or 0) > 0:
                pass
            else:
                self.logger.warning(
                    f"{symbol} - DCF valuation not applicable: {result.get('reason', 'unknown reason')}"
                )
            return result
        except Exception as e:
            self.logger.error(f"{symbol} - DCF calculation error: {e}", exc_info=True)
            return {"fair_value_per_share": 0, "applicable": False, "error": str(e)}

    async def _calculate_ggm(
        self,
        symbol: str,
        cost_of_equity: float,
        quarterly_data: List[Dict],
        company_profile: CompanyProfile,
    ) -> Dict:
        """
        Calculate Gordon Growth Model valuation for dividend-paying stocks

        Formula: Fair Value = D₁ / (r - g)
        Where:
        - D₁ = Next year's expected dividend per share
        - r = Cost of equity (from CAPM)
        - g = Sustainable growth rate = ROE × (1 - Payout Ratio)

        Args:
            symbol: Stock ticker symbol
            cost_of_equity: Required return on equity (from CAPM)
            quarterly_data: List of quarterly financial data (8 quarters from hybrid strategy)

        Returns:
            GGM valuation result dict with fair_value_per_share, upside_downside_pct, assumptions
        """
        try:
            from investigator.infrastructure.database.db import get_db_manager

            # Hybrid strategy already provides 8 quarters (2 years) for growth calculation
            # Convert QuarterlyData objects to dicts if needed
            quarterly_metrics = [q.to_dict() if isinstance(q, QuarterlyData) else q for q in quarterly_data]
            multi_year_data = []  # GGM will aggregate from quarterly_metrics

            db_manager = get_db_manager()

            model = GordonGrowthModel(
                symbol=symbol,
                quarterly_metrics=quarterly_metrics,
                multi_year_data=multi_year_data,
                db_manager=db_manager,
            )
            result = model.calculate_ggm_valuation(cost_of_equity=cost_of_equity)
            # GGM returns a dict directly (not ValuationModelResult), so no normalization needed

            if not result.get("applicable"):
                self.logger.info(f"{symbol} - GGM not applicable: {result.get('reason', 'Unknown')}")
            return result
        except Exception as e:
            self.logger.error(f"{symbol} - GGM calculation error: {e}", exc_info=True)
            return {"applicable": False, "reason": f"Error: {str(e)}", "fair_value_per_share": 0}

    async def _perform_valuation(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """
        Perform comprehensive valuation analysis with DCF and GGM (Gordon Growth Model)

        Valuation Methods:
        1. Professional DCF (all stocks) - uses WACC with levered beta and 10Y Treasury
        2. Gordon Growth Model (dividend stocks only) - uses CAPM cost of equity
        3. Relative Valuation (P/E, P/B, P/S multiples)
        4. Asset-based Valuation (book value)
        5. Earnings Power Value (EPV)
        """
        market_data = company_data["market_data"]
        financials = self._require_financials(company_data)
        data_quality = company_data.get("data_quality", {})
        trend_context = format_trend_context(company_data)

        valuation_results = {}

        company_profile = self._build_company_profile(symbol, company_data, ratios)
        company_profile_payload = serialize_company_profile(company_profile)

        # Get quarterly data from hybrid strategy (8 quarters = 2 years)
        quarterly_data = company_data.get("quarterly_data", [])

        # Hydrate cost-of-capital inputs before kicking off valuation
        self._hydrate_cost_of_capital_inputs(company_profile, company_data, ratios, symbol)
        cost_of_capital_issues = self._evaluate_cost_of_capital_inputs(company_profile, company_data)

        # === SECTOR-AWARE VALUATION ROUTING ===
        # Route to sector-specific valuation methods (Insurance P/BV, Bank ROE, REIT FFO)
        # Falls back to DCF for non-special sectors
        sector_specific_result = None
        if company_profile.sector and company_profile.industry:
            try:
                router = SectorValuationRouter()
                current_price = market_data.get("current_price")

                if current_price:
                    # Get database URL from config
                    from investigator.config import get_config

                    config = get_config()
                    database_url = (
                        f"postgresql://{config.database.username}:{config.database.password}"
                        f"@{config.database.host}:{config.database.port}/{config.database.database}"
                    )

                    valuation_result = router.route_valuation(
                        symbol=symbol,
                        sector=company_profile.sector,
                        industry=company_profile.industry,
                        financials=financials,
                        current_price=current_price,
                        database_url=database_url,
                    )

                    # Router returns None for non-special sectors (use standard DCF)
                    if valuation_result is not None:
                        # Convert ValuationResult to standard format
                        sector_specific_result = {
                            "method": valuation_result.method,
                            "fair_value": valuation_result.fair_value,
                            "current_price": valuation_result.current_price,
                            "upside_percent": valuation_result.upside_percent,
                            "confidence": valuation_result.confidence,
                            "details": valuation_result.details,
                            "warnings": valuation_result.warnings,
                        }

                        # Log sector-specific valuation usage
                        self.logger.info(
                            f"{symbol} - Used sector-specific valuation: {valuation_result.method} "
                            f"(FV=${valuation_result.fair_value:.2f}, Upside={valuation_result.upside_percent:+.1f}%)"
                        )
                        valuation_results["sector_specific"] = sector_specific_result
            except Exception as e:
                self.logger.warning(f"{symbol} - Sector-specific valuation failed: {e}, falling back to DCF")

        # === PROFESSIONAL DCF VALUATION (all stocks) ===
        # Uses WACC with levered beta from symbol table and 10Y Treasury from FRED
        # Note: Always calculate DCF for comparison, even if sector-specific method was used
        dcf_professional = await self._calculate_dcf_professional(symbol, quarterly_data, company_profile)

        # Add required fields for orchestrator blending BEFORE applying penalty
        # (DCF uses raw dict format, not ValuationModelResult)
        if isinstance(dcf_professional, dict) and dcf_professional.get("fair_value_per_share"):
            dcf_professional.setdefault("model", "dcf")
            dcf_professional.setdefault("applicable", True)
            dcf_professional.setdefault("confidence_score", 0.7)  # Default DCF confidence (may be reduced by penalty)
            dcf_professional.setdefault("weight", 0.0)  # Will be calculated by orchestrator
            self.logger.info(
                f"🔧 {symbol} - Added orchestrator fields to DCF: model={dcf_professional.get('model')}, applicable={dcf_professional.get('applicable')}, confidence={dcf_professional.get('confidence_score')}"
            )
        else:
            self.logger.warning(
                f"⚠️ {symbol} - DCF did not get orchestrator fields: isinstance={isinstance(dcf_professional, dict)}, fair_value={dcf_professional.get('fair_value_per_share') if isinstance(dcf_professional, dict) else 'N/A'}"
            )

        # Apply cost of capital penalty (may reduce confidence_score)
        dcf_professional = self._apply_cost_of_capital_penalty(dcf_professional, cost_of_capital_issues)

        valuation_results["dcf_professional"] = dcf_professional

        # Persist deterministic DCF snapshot so downstream synthesis can reuse it without recomputation
        self._store_deterministic_analysis(
            symbol=symbol,
            label="deterministic_dcf",
            payload=dcf_professional,
            period=company_data.get("fiscal_period"),
        )

        # Log DCF result immediately
        log_individual_model_result(self.logger, symbol, "DCF", dcf_professional)

        relative_models = calculate_relative_valuation_models(
            symbol=symbol,
            company_profile=company_profile,
            company_data=company_data,
            ratios=ratios,
            financials=financials,
            market_data=market_data,
            config=self.config,
            sector_specific_result=valuation_results.get("sector_specific"),
            lookup_sector_multiple=self._lookup_sector_multiple,
            calculate_enterprise_value=self._calculate_enterprise_value,
            logger=self.logger,
        )
        normalized_pe = relative_models["pe"]
        normalized_ev_ebitda = relative_models["ev_ebitda"]
        normalized_ps = relative_models["ps"]
        normalized_pb = relative_models["pb"]
        valuation_results["pe"] = normalized_pe
        valuation_results["ev_ebitda"] = normalized_ev_ebitda
        valuation_results["ps"] = normalized_ps
        valuation_results["pb"] = normalized_pb

        # Log relative-model results immediately
        log_individual_model_result(self.logger, symbol, "P/E", normalized_pe)
        log_individual_model_result(self.logger, symbol, "EV/EBITDA", normalized_ev_ebitda)
        log_individual_model_result(self.logger, symbol, "P/S", normalized_ps)

        # Log P/B result immediately
        log_individual_model_result(self.logger, symbol, "P/B", normalized_pb)

        payout_ratio = await calculate_valuation_extensions(
            symbol=symbol,
            valuation_results=valuation_results,
            financials=financials,
            ratios=ratios,
            market_data=market_data,
            company_profile=company_profile,
            quarterly_data=quarterly_data,
            calculate_cost_of_equity=self._calculate_cost_of_equity,
            calculate_ggm=self._calculate_ggm,
            normalize_model_output=normalize_model_output,
            log_model_result=log_individual_model_result,
            logger=self.logger,
        )

        # === MULTI-MODEL BLENDING + SUMMARY LOGGING ===
        multi_model_summary, tier_classification = run_multi_model_blending(
            symbol=symbol,
            valuation_results=valuation_results,
            company_profile=company_profile,
            company_data=company_data,
            ratios=ratios,
            financials=financials,
            dcf_professional=dcf_professional,
            normalized_pe=normalized_pe,
            normalized_ev_ebitda=normalized_ev_ebitda,
            normalized_ps=normalized_ps,
            normalized_pb=normalized_pb,
            select_models_for_company=self._select_models_for_company,
            resolve_fallback_weights=self._resolve_fallback_weights,
            multi_model_orchestrator=self.multi_model_orchestrator,
            logger=self.logger,
        )
        summary_metrics = log_multi_model_summary(
            symbol=symbol,
            valuation_results=valuation_results,
            company_data=company_data,
            tier_classification=tier_classification,
            dcf_professional=dcf_professional,
            normalized_pe=normalized_pe,
            normalized_ev_ebitda=normalized_ev_ebitda,
            normalized_ps=normalized_ps,
            normalized_pb=normalized_pb,
            log_valuation_snapshot=log_valuation_snapshot,
            format_valuation_summary_table=ValuationTableFormatter.format_valuation_summary_table,
            logger=self.logger,
        )
        blended_fair_value = summary_metrics["blended_fair_value"]
        overall_confidence = summary_metrics["overall_confidence"]
        model_agreement_score = summary_metrics["model_agreement_score"]
        divergence_flag = summary_metrics["divergence_flag"]
        applicable_models = summary_metrics["applicable_models"]
        notes = summary_metrics["notes"]

        models_detail_lines = build_models_detail_lines(
            multi_model_summary.get("models", []),
            format_currency=_fmt_currency,
            format_percentage=_fmt_pct,
        )
        archetype_labels = ", ".join(company_profile.archetype_labels()) or "Unclassified"
        prompt = build_valuation_synthesis_prompt(
            data_quality=data_quality,
            trend_context=trend_context,
            sector=company_profile.sector,
            industry=company_profile.industry,
            archetype_labels=archetype_labels,
            data_quality_flags=company_profile_payload.get("data_quality_flags", []),
            current_price=market_data.get("price"),
            market_cap=market_data.get("market_cap", 0),
            payout_ratio=payout_ratio,
            blended_fair_value=blended_fair_value,
            overall_confidence=overall_confidence,
            model_agreement_score=model_agreement_score,
            divergence_flag=divergence_flag,
            applicable_models=applicable_models,
            notes=notes,
            models_detail_lines=models_detail_lines,
            format_currency=_fmt_currency,
            format_int_with_commas=_fmt_int_comma,
            format_percentage=_safe_fmt_pct,
        )

        return await dispatch_valuation_synthesis(
            symbol=symbol,
            prompt=prompt,
            company_data=company_data,
            market_data=market_data,
            valuation_results=valuation_results,
            multi_model_summary=multi_model_summary,
            data_quality=data_quality,
            company_profile_payload=company_profile_payload,
            notes=notes,
            use_deterministic=self.use_deterministic,
            deterministic_valuation_synthesis=self.deterministic_valuation_synthesis,
            build_deterministic_response=self._build_deterministic_response,
            debug_log_prompt=self._debug_log_prompt,
            debug_log_response=self._debug_log_response,
            ollama_client=self.ollama,
            valuation_model=self.models["valuation"],
            cache_llm_response=self._cache_llm_response,
            wrap_llm_response=self._wrap_llm_response,
            logger=self.logger,
        )

    async def _analyze_competitive_position(self, company_data: Dict, symbol: str) -> Dict:
        """Analyze company's competitive position"""
        # Check if deterministic competitive analysis is enabled (saves tokens, faster)
        if self.use_deterministic and self.deterministic_competitive_analysis:
            self.logger.debug(f"{symbol} - Using deterministic competitive analysis (LLM bypass)")

            response_data = analyze_competitive_position(symbol=symbol, company_data=company_data)

            return self._build_deterministic_response("competitive_position", response_data)

        # === LLM Path (fallback when deterministic is disabled) ===
        financials = self._require_financials(company_data)
        data_quality = company_data.get("data_quality", {})
        trend_context = format_trend_context(company_data)

        prompt = f"""
        Analyze the competitive position of {company_data['symbol']}:

        DATA QUALITY ASSESSMENT:
        - Overall Quality: {data_quality.get('quality_grade', 'Unknown')} ({_safe_fmt_pct(data_quality.get('data_quality_score', 0))})
        - {data_quality.get('assessment', 'Data quality information not available')}
        - Core Metrics: {data_quality.get('core_metrics_populated', 'N/A')} populated
        - Consistency Issues: {', '.join(data_quality.get('consistency_issues', [])) or 'None detected'}
        {trend_context}

        Company Metrics:
        Market Cap: ${_safe_fmt_int_comma(company_data.get('market_data', {}).get('market_cap'))}
        Revenue: ${_safe_fmt_int_comma(financials.get('revenues'))}

        Evaluate:
        1. Market position and share
        2. Competitive advantages (moat analysis)
        3. Industry dynamics and trends
        4. Barriers to entry
        5. Supplier and customer power
        6. Threat of substitutes
        7. Competitive risks
        8. Strategic positioning score (0-100)

        Use Porter's Five Forces and moat analysis frameworks.

        IMPORTANT: Consider the data quality assessment when determining confidence levels.
        If data quality is below 75%, flag this in your analysis and adjust confidence accordingly.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object that strictly follows the schema below (values are illustrative):
        {{
          "market_position_and_share": {{
            "assessment": "Leader",
            "commentary": "The company is the market leader with a 40% market share."
          }},
          "competitive_advantages_moat": {{
            "assessment": "Wide",
            "commentary": "The company has a wide economic moat due to its strong brand, network effects, and high switching costs."
          }},
          "industry_dynamics_and_trends": {{
            "assessment": "Favorable",
            "commentary": "The industry is growing at a healthy rate, and the company is well-positioned to benefit from this growth."
          }},
          "barriers_to_entry": {{
            "assessment": "High",
            "commentary": "The industry has high barriers to entry, which limits the threat of new entrants."
          }},
          "supplier_and_customer_power": {{
            "assessment": "Low",
            "commentary": "The company has a diversified supplier base and a large, fragmented customer base, which limits the power of suppliers and customers."
          }},
          "threat_of_substitutes": {{
            "assessment": "Low",
            "commentary": "There are few substitutes for the company's products."
          }},
          "competitive_risks": [
            "Intensifying competition from existing players",
            "Technological disruption"
          ],
          "strategic_positioning_score": 85
        }}
        """

        # Save prompt to cache for auditing

        prompt_name = "_analyze_competitive_position_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        response = await self.ollama.generate(
            model=self.models["quality"],
            prompt=prompt,
            system="Analyze competitive position and strategic advantages.",
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
            prompt_name=prompt_name,
        )

        self._debug_log_response(prompt_name, response)

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["quality"],
            symbol=symbol,
            llm_type="fundamental_competitive_position",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
        )

        return self._wrap_llm_response(
            response=response,
            model=self.models["quality"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
        )

    def _lookup_sector_multiple(self, sector: Optional[str], multiple: str) -> Optional[float]:
        """Fetch sector-level reference multiples from configuration if available."""
        if not sector:
            self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] sector is None, returning None")
            return None

        try:
            self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Looking up {sector}/{multiple}")
            self.logger.warning(
                f"[SECTOR_LOOKUP_DEBUG] _sector_multiples_loader exists: {self._sector_multiples_loader is not None}"
            )

            if self._sector_multiples_loader:
                record = self._sector_multiples_loader.get(sector)
                self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Loader record for {sector}: {record}")
                if record:
                    value = getattr(record, multiple, None)
                    self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Record.{multiple} = {value}")
                    if value is not None:
                        self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Returning value from loader: {value}")
                        return float(value)

            valuation_settings = getattr(self.config, "valuation", None)
            self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] valuation_settings exists: {valuation_settings is not None}")
            if isinstance(valuation_settings, dict):
                multiples = valuation_settings.get("sector_multiples", {}) or {}
            elif valuation_settings is not None:
                multiples = getattr(valuation_settings, "sector_multiples", {}) or {}
            else:
                self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] No valuation_settings, returning None")
                return None

            self.logger.warning(
                f"[SECTOR_LOOKUP_DEBUG] Config multiples keys: {list(multiples.keys()) if multiples else 'empty'}"
            )
            sector_key = sector.lower()
            for key, values in multiples.items():
                if key.lower() == sector_key:
                    value = values.get(multiple)
                    self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Found {key} matching {sector_key}, {multiple}={value}")
                    if value is not None:
                        self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Returning value from config: {value}")
                        return float(value)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] Exception in lookup: {exc}")
            self.logger.debug(f"Failed to load sector multiple for {sector}/{multiple}: {exc}")

        self.logger.warning(f"[SECTOR_LOOKUP_DEBUG] No value found, returning None")
        return None

    @staticmethod
    def _calculate_enterprise_value(market_data: Dict, financials: Dict) -> Optional[float]:
        ev_candidates = [
            market_data.get("enterprise_value"),
            market_data.get("enterpriseValue"),
            market_data.get("enterprise_value_real_time"),
        ]
        for ev in ev_candidates:
            if ev is not None:
                try:
                    return float(ev)
                except (TypeError, ValueError):
                    continue

        market_cap = market_data.get("market_cap") or market_data.get("marketCap")
        if market_cap is None:
            return None

        total_debt = financials.get("total_debt") or financials.get("long_term_debt") or market_data.get("total_debt")
        cash = financials.get("cash") or financials.get("cash_and_equivalents") or market_data.get("cash")

        try:
            market_cap_val = float(market_cap)
            debt_val = float(total_debt) if total_debt is not None else 0.0
            cash_val = float(cash) if cash is not None else 0.0
            return market_cap_val + debt_val - cash_val
        except (TypeError, ValueError):
            return None

    def _resolve_fallback_weights(
        self,
        company_profile: CompanyProfile,
        models_for_blending: List[Dict[str, Any]],
        financials: Optional[Dict[str, Any]] = None,
        ratios: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        """Delegate dynamic/static fallback weighting logic to shared helper."""
        return resolve_fallback_weights(
            company_profile=company_profile,
            models_for_blending=models_for_blending,
            financials=financials,
            ratios=ratios,
            dynamic_weighting_service=self.dynamic_weighting_service,
            config=self.config,
            logger=self.logger,
        )

    def _load_model_selection_rules(self) -> Dict[str, Any]:
        rules_path = Path("config/model_selection.yaml")
        if not rules_path.exists():
            return {}
        try:
            with rules_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception as exc:
            self.logger.warning(f"Failed to load model selection rules: {exc}")
            return {}

    def _select_models_for_company(self, profile: CompanyProfile) -> Optional[List[str]]:
        if not self._model_selection_rules:
            return None

        rules = self._model_selection_rules if isinstance(self._model_selection_rules, dict) else {}
        defaults = rules.get("defaults", {}) if isinstance(rules.get("defaults"), dict) else {}

        include = set(defaults.get("include", []))
        exclude = set(defaults.get("exclude", []))
        blocking_flags: Dict[str, List[str]] = {}

        def _merge_blocking(rule_blocking: Optional[Dict[str, Any]]) -> None:
            if not isinstance(rule_blocking, dict):
                return
            for flag, models in rule_blocking.items():
                if not isinstance(models, (list, tuple)):
                    continue
                existing = blocking_flags.setdefault(flag.upper(), [])
                existing.extend(str(model) for model in models)

        _merge_blocking(defaults.get("blocking_flags"))

        archetype_rules = rules.get("archetypes", {}) if isinstance(rules.get("archetypes"), dict) else {}
        primary = profile.primary_archetype.name.lower() if profile.primary_archetype else None
        if primary and archetype_rules.get(primary):
            rule = archetype_rules[primary] or {}
            include.update(rule.get("include", []))
            exclude.update(rule.get("exclude", []))
            _merge_blocking(rule.get("blocking_flags"))

            secondary_rules = rule.get("secondary") if isinstance(rule.get("secondary"), dict) else {}
            secondary = profile.secondary_archetype.name.lower() if profile.secondary_archetype else None
            if secondary and secondary in secondary_rules:
                sec_rule = secondary_rules[secondary] or {}
                include.update(sec_rule.get("include", []))
                exclude.update(sec_rule.get("exclude", []))
                _merge_blocking(sec_rule.get("blocking_flags"))

        allowed = [model for model in include if model not in exclude]
        if blocking_flags and profile.data_quality_flags:
            active_flags = {flag.name.upper() for flag in profile.data_quality_flags}
            for flag in active_flags:
                blocked = blocking_flags.get(flag)
                if not blocked:
                    continue
                allowed = [model for model in allowed if model not in blocked]

        min_models = defaults.get("min_models")
        if isinstance(min_models, int) and min_models > 0 and len(allowed) < min_models:
            return None

        return allowed if allowed else None

    async def _generate_forecast(self, company_data: Dict, growth_analysis: Dict, symbol: str) -> Dict:
        """Generate earnings and revenue forecast"""
        financials = self._require_financials(company_data)
        data_quality = company_data.get("data_quality", {})
        trend_context = format_trend_context(company_data)

        prompt = f"""
        Generate financial forecasts based on historical data and growth analysis:

        DATA QUALITY ASSESSMENT:
        - Overall Quality: {data_quality.get('quality_grade', 'Unknown')} ({_safe_fmt_pct(data_quality.get('data_quality_score', 0))})
        - {data_quality.get('assessment', 'Data quality information not available')}
        - Core Metrics: {data_quality.get('core_metrics_populated', 'N/A')} populated
        - Consistency Issues: {', '.join(data_quality.get('consistency_issues', [])) or 'None detected'}
        {trend_context}

        Historical Financials:
        {json.dumps(self._get_historical_trend(financials), indent=2)}

        Growth Analysis:
        {json.dumps(growth_analysis, indent=2)}

        Provide forecasts for next 3 years:
        1. Revenue forecast (with growth rates)
        2. Earnings forecast
        3. Free cash flow forecast
        4. Margin projections
        5. Key assumptions
        6. Scenario analysis (base/bull/bear)
        7. Confidence intervals

        Be realistic and consider industry trends.

        IMPORTANT: Consider the data quality assessment when determining confidence levels.
        If data quality is below 75%, flag this in your analysis and adjust confidence accordingly.
        Lower confidence should result in wider confidence intervals.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object that strictly follows the schema below (values are illustrative):
        {{
          "revenue_forecast": [
            {{ "year": 2026, "revenue": 110, "growth_rate": 0.10 }},
            {{ "year": 2027, "revenue": 121, "growth_rate": 0.10 }},
            {{ "year": 2028, "revenue": 133, "growth_rate": 0.10 }}
          ],
          "earnings_forecast": [
            {{ "year": 2026, "eps": 5.50 }},
            {{ "year": 2027, "eps": 6.05 }},
            {{ "year": 2028, "eps": 6.65 }}
          ],
          "free_cash_flow_forecast": [
            {{ "year": 2026, "fcf": 15 }},
            {{ "year": 2027, "fcf": 18 }},
            {{ "year": 2028, "fcf": 21 }}
          ],
          "margin_projections": {{
            "gross_margin": 0.45,
            "operating_margin": 0.25,
            "net_margin": 0.15
          }},
          "key_assumptions": [
            "Market growth of 5% per year",
            "Stable competitive landscape",
            "No major economic downturns"
          ],
          "scenario_analysis": {{
            "base_case": {{ "revenue_growth": 0.10, "eps": 6.65 }},
            "bull_case": {{ "revenue_growth": 0.15, "eps": 7.50 }},
            "bear_case": {{ "revenue_growth": 0.05, "eps": 5.80 }}
          }},
          "confidence_intervals": {{
            "revenue_2028": [125, 140],
            "eps_2028": [6.50, 7.00]
          }}
        }}
        """

        # Save prompt to cache for auditing

        prompt_name = "_generate_forecast_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        response = await self.ollama.generate(
            model=self.models["valuation"],
            prompt=prompt,
            system="Generate realistic financial forecasts with clear assumptions.",
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
            prompt_name=prompt_name,
        )

        self._debug_log_response(prompt_name, response)

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["valuation"],
            symbol=symbol,
            llm_type="fundamental_forecast",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
        )

        return self._wrap_llm_response(
            response=response,
            model=self.models["valuation"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
        )

    async def _calculate_quality_score(
        self, health: Dict, growth: Dict, profitability: Dict, competitive: Dict
    ) -> float:
        """Calculate overall company quality score"""
        scores = []
        weights = []

        # Financial health score (30% weight)
        if "overall_health_score" in health:
            scores.append(health["overall_health_score"])
            weights.append(0.30)

        # Growth score (25% weight)
        if "growth_score" in growth:
            scores.append(growth["growth_score"])
            weights.append(0.25)

        # Profitability score (25% weight)
        if "profitability_score" in profitability:
            scores.append(profitability["profitability_score"])
            weights.append(0.25)

        # Competitive position score (20% weight)
        if "strategic_positioning_score" in competitive:
            scores.append(competitive["strategic_positioning_score"])
            weights.append(0.20)

        # Calculate weighted average
        if scores and weights:
            quality_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return float(quality_score)

        return 50.0  # Default middle score

    async def _synthesize_fundamental_report(self, analysis_data: Dict) -> Dict:
        """Synthesize comprehensive fundamental analysis report"""
        # Extract symbol, data quality, confidence, and period for caching
        symbol = analysis_data.get("symbol", "UNKNOWN")
        data_quality = analysis_data.get("data_quality", {})
        confidence = analysis_data.get("confidence", {})
        fiscal_period = analysis_data.get("fiscal_period")  # Extract period from analysis_data

        # Check if TOON format is enabled
        use_toon = getattr(self.config.ollama, "use_toon_format", False) and getattr(
            self.config.ollama, "toon_agents", {}
        ).get("fundamental_analysis", False)

        # Format data section (TOON or JSON)
        if use_toon:
            # Extract quarterly data for TOON formatting (63% token savings)
            quarterly_data = analysis_data.get("quarterly_data", [])

            if quarterly_data and isinstance(quarterly_data, list) and len(quarterly_data) > 0:
                try:
                    # Convert QuarterlyData objects to dicts if needed
                    quarterly_dicts = []
                    for q in quarterly_data:
                        if hasattr(q, "__dict__"):
                            quarterly_dicts.append(vars(q))
                        elif isinstance(q, dict):
                            quarterly_dicts.append(q)

                    if quarterly_dicts:
                        # Convert to TOON format
                        toon_quarterly = to_toon_quarterly(quarterly_dicts)

                        # Remove quarterly_data from analysis_data to avoid duplication
                        remaining_data = {k: v for k, v in analysis_data.items() if k != "quarterly_data"}

                        # Build data section with TOON quarterly + JSON for other data
                        data_section = (
                            f"{toon_quarterly}\n\nAdditional Analysis:\n{json.dumps(remaining_data, indent=2)[:8000]}"
                        )
                    else:
                        # No valid quarterly data, fall back to JSON
                        data_section = json.dumps(analysis_data, indent=2)[:10000]
                except Exception as e:
                    self.logger.warning(f"Failed to convert quarterly data to TOON for {symbol}: {e}")
                    data_section = json.dumps(analysis_data, indent=2)[:10000]
            else:
                # No quarterly data, use JSON
                data_section = json.dumps(analysis_data, indent=2)[:10000]
        else:
            # TOON disabled, use JSON (current behavior)
            data_section = json.dumps(analysis_data, indent=2)[:10000]

        prompt = f"""
        Synthesize a comprehensive fundamental analysis report:

        DATA QUALITY ASSESSMENT:
        - Overall Quality: {data_quality.get('quality_grade', 'Unknown')} ({_safe_fmt_pct(data_quality.get('data_quality_score', 0))})
        - {data_quality.get('assessment', 'Data quality information not available')}
        - Core Metrics: {data_quality.get('core_metrics_populated', 'N/A')} populated
        - Market Data: {data_quality.get('market_metrics_populated', 'N/A')} populated
        - Ratio Metrics: {data_quality.get('ratio_metrics_populated', 'N/A')} populated
        - Consistency Issues: {', '.join(data_quality.get('consistency_issues', [])) or 'None detected'}

        DATA ENRICHMENT IMPACT (FEATURE #3):
        - Raw Extraction Quality: {_safe_fmt_pct(data_quality.get('extraction_quality', 0))}
        - Enhanced Quality (after enrichment): {_safe_fmt_pct(data_quality.get('data_quality_score', 0))}
        - Quality Improvement: +{_safe_fmt_float(data_quality.get('quality_improvement', 0), 1)} points
        - Enhancement Summary: {data_quality.get('enhancement_summary', 'N/A')}

        ANALYSIS CONFIDENCE LEVEL:
        - Confidence: {confidence.get('confidence_level', 'UNKNOWN')} ({confidence.get('confidence_score', 0)}/100)
        - Rationale: {confidence.get('rationale', 'No confidence assessment available')}
        - Based on Data Quality: {confidence.get('quality_grade', 'Unknown')} quality data

        {data_section}

        Create a structured investment report with:
        1. Executive Summary
        2. Investment Thesis
        3. Financial Analysis Summary
        4. Valuation Assessment
        5. Growth Prospects
        6. Risk Analysis
        7. Competitive Position
        8. Investment Grade (AAA to D)
        9. Price Target (12-month)
        10. Investment Recommendation (strong buy/buy/hold/sell/strong sell)
        11. Key Catalysts
        12. Key Risks

        Provide clear, actionable insights for investors.

        IMPORTANT: The data quality assessment above should influence your confidence levels.
        - If data quality is Excellent/Good (≥75%): High confidence in analysis
        - If data quality is Fair (60-75%): Moderate confidence, note data limitations
        - If data quality is Poor/Very Poor (<60%): Low confidence, significant data concerns

        Adjust your investment recommendation strength and price target confidence based on data quality.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object that strictly follows the schema below (values are illustrative):
        {{
          "executive_summary": "The company is a market leader with strong growth prospects and a wide economic moat. The stock is currently undervalued and offers an attractive risk/reward profile.",
          "investment_thesis": "The company is well-positioned to benefit from the secular growth in its industry. Its strong brand, network effects, and high switching costs provide a sustainable competitive advantage.",
          "financial_analysis_summary": "The company has a strong financial profile, with a history of consistent revenue growth, expanding margins, and strong cash flow generation.",
          "valuation_assessment": "The stock is currently trading at a discount to its intrinsic value, with a potential upside of 20% to our fair value estimate of $150.",
          "growth_prospects": "The company has multiple growth drivers, including new product launches, expansion into new markets, and strategic acquisitions.",
          "risk_analysis": "The main risks to our thesis are increased competition, regulatory changes, and a slowdown in the overall economy.",
          "competitive_position": "The company has a strong competitive position, with a dominant market share and a wide economic moat.",
          "investment_grade": "A",
          "price_target": 150.00,
          "investment_recommendation": "buy",
          "key_catalysts": [
            "Successful launch of new products",
            "Expansion into new geographic markets"
          ],
          "key_risks": [
            "Increased competition",
            "Regulatory changes"
          ]
        }}

        """

        prompt_name = "_synthesize_fundamental_report_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        # Build system prompt with optional TOON explanation
        system_prompt = "You are a senior equity analyst providing investment recommendations."
        if use_toon and quarterly_data:
            system_prompt += "\n\n" + TOONFormatter.get_format_explanation()

        response = await self.ollama.generate(
            model=self.models["quality"],
            prompt=prompt,
            system=system_prompt,
            format="json",
            period=fiscal_period,  # Period-based caching
            prompt_name=prompt_name,
        )

        self._debug_log_response(prompt_name, response)

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["quality"],
            symbol=symbol,
            llm_type="fundamental_investment_thesis",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=fiscal_period,  # Period-based caching
        )

        return self._wrap_llm_response(
            response=response,
            model=self.models["quality"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=fiscal_period,  # Period-based caching
        )

    def _calculate_growth_rate(self, financials: Dict, metric: str) -> float:
        """Calculate compound annual growth rate for a metric"""
        # Simplified CAGR calculation (would use historical data in production)
        # This is a placeholder that would access historical data
        return 0.10  # 10% placeholder growth rate

    def _calculate_ttm_net_income(self, quarterly_data: List, symbol: str) -> float:
        """
        Calculate Trailing Twelve Months (TTM) net income from quarterly data.

        This is CRITICAL for accurate P/E ratio calculations. Market P/E ratios
        should ALWAYS use TTM earnings, not quarterly earnings.

        Args:
            quarterly_data: List of quarterly data objects (last 12 quarters)
            symbol: Stock symbol for logging

        Returns:
            TTM net income (sum of last 4 quarters), or 0 if insufficient data
        """
        from investigator.domain.agents.fundamental.models import QuarterlyData

        # Debug: Log quarterly data structure
        self.logger.info(
            f"🔍 [TTM_DEBUG] {symbol} - Quarterly data received: "
            f"{len(quarterly_data) if quarterly_data else 0} periods"
        )

        # ENHANCED DEBUG: Show ALL quarters BEFORE filtering (CRITICAL for ORCL Q2-2025 investigation)
        if quarterly_data:
            self.logger.info(f"🔍 [QUARTER_FILTER] {symbol} - ALL QUARTERS BEFORE FY FILTERING:")
            for idx, q in enumerate(quarterly_data):
                if isinstance(q, QuarterlyData):
                    fp = getattr(q, "fiscal_period", "UNKNOWN")
                    fy = getattr(q, "fiscal_year", "UNKNOWN")
                    period_end = getattr(q, "period_end_date", "UNKNOWN")
                elif isinstance(q, dict):
                    fp = q.get("fiscal_period", "UNKNOWN")
                    fy = q.get("fiscal_year", "UNKNOWN")
                    period_end = q.get("period_end_date", "UNKNOWN")
                else:
                    fp = fy = period_end = "UNKNOWN_TYPE"

                self.logger.info(f"  [{idx}] fiscal_year={fy}, fiscal_period={fp}, period_end={period_end}")

        if not quarterly_data or len(quarterly_data) < 4:
            self.logger.warning(
                f"{symbol} - Insufficient quarterly data for TTM calculation "
                f"({len(quarterly_data) if quarterly_data else 0} quarters < 4 required)"
            )
            return 0

        # CRITICAL: Filter out FY (Full Year) periods to avoid double-counting
        # FY periods contain the full year's net income (Q1+Q2+Q3+Q4 already summed)
        # If we include FY in our TTM sum, we'd double-count net income
        actual_quarters = []
        filtered_out = []
        for q in quarterly_data:
            # Extract fiscal_period from QuarterlyData or dict
            if isinstance(q, QuarterlyData):
                fiscal_period = getattr(q, "fiscal_period", "").upper()
                fiscal_year = getattr(q, "fiscal_year", "UNKNOWN")
                period_end = getattr(q, "period_end_date", "UNKNOWN")
            elif isinstance(q, dict):
                fiscal_period = q.get("fiscal_period", "").upper()
                fiscal_year = q.get("fiscal_year", "UNKNOWN")
                period_end = q.get("period_end_date", "UNKNOWN")
            else:
                continue

            # Only include actual quarters (Q1, Q2, Q3, Q4), exclude FY
            if fiscal_period in ["Q1", "Q2", "Q3", "Q4"]:
                actual_quarters.append(q)
            else:
                # Track what got filtered out
                filtered_out.append(f"{fiscal_year}-{fiscal_period} (period_end={period_end})")

        self.logger.info(
            f"🔍 [TTM_DEBUG] {symbol} - Filtered to {len(actual_quarters)} actual quarters "
            f"(excluded FY periods from {len(quarterly_data)} total periods)"
        )

        # ENHANCED DEBUG: Show what was filtered out
        if filtered_out:
            self.logger.info(f"🔍 [QUARTER_FILTER] {symbol} - FILTERED OUT (FY periods): {', '.join(filtered_out)}")

        # ENHANCED DEBUG: Show quarters AFTER filtering (what we'll use for TTM)
        if actual_quarters:
            self.logger.info(f"🔍 [QUARTER_FILTER] {symbol} - QUARTERS AFTER FY FILTERING (will use first 4 for TTM):")
            for idx, q in enumerate(actual_quarters[:8]):  # Show first 8
                if isinstance(q, QuarterlyData):
                    fp = getattr(q, "fiscal_period", "UNKNOWN")
                    fy = getattr(q, "fiscal_year", "UNKNOWN")
                    period_end = getattr(q, "period_end_date", "UNKNOWN")
                elif isinstance(q, dict):
                    fp = q.get("fiscal_period", "UNKNOWN")
                    fy = q.get("fiscal_year", "UNKNOWN")
                    period_end = q.get("period_end_date", "UNKNOWN")
                else:
                    fp = fy = period_end = "UNKNOWN_TYPE"

                prefix = "**WILL USE**" if idx < 4 else "available"
                self.logger.info(f"  [{idx}] {prefix}: {fy}-{fp} (period_end={period_end})")

        if len(actual_quarters) < 4:
            self.logger.warning(
                f"{symbol} - Insufficient actual quarterly data for TTM calculation "
                f"({len(actual_quarters)} quarters < 4 required after filtering out FY periods)"
            )
            return 0

        # Take the last 4 quarters (most recent = index 0)
        last_4_quarters = actual_quarters[:4]

        ttm_net_income = 0
        quarter_details = []

        for idx, q in enumerate(last_4_quarters):
            # Debug: Log quarter structure
            self.logger.info(f"🔍 [TTM_DEBUG] {symbol} - Q{idx+1} type: {type(q).__name__}")

            # Handle both dict and QuarterlyData object
            if isinstance(q, QuarterlyData):
                # Debug: Log QuarterlyData attributes
                self.logger.info(
                    f"🔍 [TTM_DEBUG] {symbol} - Q{idx+1} QuarterlyData attributes: "
                    f"{list(q.__dict__.keys()) if hasattr(q, '__dict__') else 'no __dict__'}"
                )

                ni = q.financial_data.get("net_income", 0) if hasattr(q, "financial_data") else 0
                period = f"{q.fiscal_year}-{q.fiscal_period}" if hasattr(q, "fiscal_year") else "Unknown"

                # Debug: Log extracted values
                self.logger.info(
                    f"🔍 [TTM_DEBUG] {symbol} - Q{idx+1} {period}: "
                    f"net_income=${ni:,.0f} (from QuarterlyData.financial_data)"
                )

            elif isinstance(q, dict):
                # Debug: Log dict keys
                self.logger.info(f"🔍 [TTM_DEBUG] {symbol} - Q{idx+1} dict keys: {list(q.keys())[:10]}...")

                # Try multiple paths to find net_income
                ni = (
                    q.get("financial_data", {}).get("net_income")
                    or q.get("net_income")
                    or q.get("financials", {}).get("net_income")
                    or 0
                )

                period = f"{q.get('fiscal_year', 'Unknown')}-{q.get('fiscal_period', 'Q?')}"

                # Debug: Log all attempted paths
                self.logger.info(
                    f"🔍 [TTM_DEBUG] {symbol} - Q{idx+1} {period}: "
                    f"financial_data.net_income={q.get('financial_data', {}).get('net_income')}, "
                    f"net_income={q.get('net_income')}, "
                    f"financials.net_income={q.get('financials', {}).get('net_income')}, "
                    f"→ using ni=${ni:,.0f}"
                )
            else:
                self.logger.warning(f"{symbol} - Unexpected quarterly data type: {type(q)}")
                continue

            ttm_net_income += ni
            quarter_details.append(f"{period}=${ni:,.0f}")

        if ttm_net_income > 0:
            self.logger.info(
                f"✅ {symbol} - TTM Net Income: ${ttm_net_income:,.0f} " f"(sum of {', '.join(quarter_details)})"
            )
        else:
            self.logger.warning(
                f"❌ {symbol} - TTM Net Income is zero or negative: ${ttm_net_income:,.0f} "
                f"(quarters: {', '.join(quarter_details)})"
            )

        return ttm_net_income

    def _get_historical_trend(self, financials: Dict) -> Dict:
        """Get historical financial trends"""
        return _get_historical_trend_helper(financials)

    def _summarize_company_data(self, company_data: Dict) -> Dict:
        """Create summary of company data for report"""
        return _summarize_company_data_helper(company_data)

    def _extract_latest_financials(self, quarterly_data: List) -> Dict:
        """Extract latest financial statement from quarterly data (supports both Dict and QuarterlyData objects)"""
        return _extract_latest_financials_helper(quarterly_data)
