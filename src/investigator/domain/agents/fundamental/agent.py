"""
Fundamental Analysis Agent
Specialized agent for fundamental analysis and financial metrics evaluation using Ollama LLMs
"""

import asyncio
import json
import logging
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from investigator.domain.agents.base import InvestmentAgent
from investigator.domain.models.analysis import AgentResult, AgentTask, TaskStatus
from investigator.domain.services.company_metadata_service import CompanyMetadataService
from investigator.domain.services.data_normalizer import (  # TODO: Move to infrastructure
    DataNormalizer,
    normalize_financials,
    round_for_prompt,
)
from investigator.domain.services.deterministic_competitive_analyzer import analyze_competitive_position

# Deterministic services (replace LLM calls with rule-based computation)
from investigator.domain.services.deterministic_valuation_synthesizer import synthesize_valuation
from investigator.domain.services.dynamic_model_weighting import DynamicModelWeightingService
from investigator.domain.services.fiscal_period_service import get_fiscal_period_service
from investigator.domain.services.safe_formatters import format_currency as _fmt_currency
from investigator.domain.services.safe_formatters import format_int_with_commas as _fmt_int_comma
from investigator.domain.services.safe_formatters import (
    format_number,
)
from investigator.domain.services.safe_formatters import format_percentage as _fmt_pct
from investigator.domain.services.safe_formatters import (
    is_valid_number,
    safe_round,
)
from investigator.domain.services.toon_formatter import TOONFormatter, to_toon_quarterly
from investigator.domain.services.valuation import SectorValuationRouter  # Sector-aware valuation routing

# New valuation models (Milestone 7 - Plan implementation)
from investigator.domain.services.valuation.damodaran_dcf import DamodaranDCFModel
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
    EVEBITDAModel,
    PBMultipleModel,
    PEMultipleModel,
    PSMultipleModel,
)
from investigator.domain.services.valuation.models.common import clamp
from investigator.domain.services.valuation.models.saas_valuation import SaaSValuationModel
from investigator.domain.services.valuation.orchestrator import MultiModelValuationOrchestrator
from investigator.domain.services.valuation.rule_of_40_valuation import RuleOf40Valuation
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
from .logging_utils import (
    format_trend_context,
    log_data_quality_issues,
    log_individual_model_result,
    log_quarterly_snapshot,
    log_table,
    log_valuation_snapshot,
)
from .models import QuarterlyData


# Module-level safe formatting functions for use in prompts
def _safe_fmt_pct(value: Any, decimals: int = 1) -> str:
    """Safe percentage formatting with None/NaN handling."""
    if value is None:
        return "N/A"
    rounded = round_for_prompt(value, decimals)
    if rounded is None:
        return "N/A"
    return f"{rounded:.{decimals}f}%"


def _safe_fmt_float(value: Any, decimals: int = 2) -> str:
    """Safe float formatting with None/NaN handling."""
    if value is None:
        return "N/A"
    rounded = round_for_prompt(value, decimals)
    if rounded is None:
        return "N/A"
    return f"{rounded:.{decimals}f}"


def _safe_fmt_int_comma(value: Any) -> str:
    """Safe integer formatting with comma separators."""
    if value is None:
        return "N/A"
    rounded = round_for_prompt(value, 0)
    if rounded is None:
        return "N/A"
    return f"{rounded:,.0f}"


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
            # Get CIK for symbol
            cik = self.ticker_mapper.resolve_cik(symbol)
            if not cik:
                raise ValueError(f"No CIK found for {symbol}")

            # CLEAN ARCHITECTURE: Query sec_companyfacts_processed table directly
            # This table has ALL historical quarters (e.g., 66 quarters for AAPL from 2009-2025)
            # No need for hybrid strategy - just get the most recent num_quarters
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db_manager = get_db_manager()

            query = text(
                """
                SELECT
                    symbol, fiscal_year, fiscal_period, adsh,
                    filed_date as filed,
                    period_end_date as period_end,
                    form_type as form,
                    total_revenue,
                    net_income,
                    gross_profit,
                    operating_income,
                    interest_expense,
                    income_tax_expense,
                    cost_of_revenue,
                    total_assets,
                    total_liabilities,
                    stockholders_equity,
                    current_assets,
                    current_liabilities,
                    accounts_receivable,
                    inventory,
                    cash_and_equivalents,
                    long_term_debt,
                    short_term_debt,
                    total_debt,
                    operating_cash_flow,
                    capital_expenditures,
                    free_cash_flow,
                    dividends_paid,
                    cash_flow_statement_qtrs,
                    income_statement_qtrs,
                    property_plant_equipment_net,
                    weighted_average_diluted_shares_outstanding as shares_outstanding
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                ORDER BY
                    fiscal_year DESC,
                    CASE fiscal_period
                        WHEN 'FY' THEN 4
                        WHEN 'Q3' THEN 3
                        WHEN 'Q2' THEN 2
                        WHEN 'Q1' THEN 1
                        ELSE 0
                    END DESC
                LIMIT :sql_limit
            """
            )

            # Calculate SQL LIMIT: For Q4 computation, need extra rows to fetch Q1-Q3 for all years
            # Formula: requested_quarters + 3 ensures we always have enough data
            # Example: 12 quarters requested → LIMIT 15 (handles both FY-aligned and mid-quarter cases)
            sql_limit = num_quarters + 3

            with db_manager.get_session() as session:
                result = session.execute(query, {"symbol": symbol, "sql_limit": sql_limit})
                rows = result.fetchall()

            if not rows:
                self.logger.warning(f"No quarterly data in processed table for {symbol}")
                return []

            # Convert rows to dict format (same as hybrid strategy output)
            quarters_data = []

            # Get FiscalPeriodService for centralized YTD detection
            fiscal_period_service = get_fiscal_period_service()

            for idx, row in enumerate(rows):
                # Convert numeric YTD indicators to boolean flags using centralized service
                # cash_flow_statement_qtrs/income_statement_qtrs indicate how many quarters data is cumulative
                # qtrs=1 means individual quarter, qtrs>=2 means YTD cumulative
                cf_qtrs = int(row.cash_flow_statement_qtrs) if row.cash_flow_statement_qtrs else 1
                inc_qtrs = int(row.income_statement_qtrs) if row.income_statement_qtrs else 1

                quarters_data.append(
                    {
                        "symbol": row.symbol,
                        "fiscal_year": row.fiscal_year,
                        "fiscal_period": row.fiscal_period,
                        "adsh": row.adsh,
                        "filed": str(row.filed) if row.filed else None,
                        "period_end": str(row.period_end) if row.period_end else None,
                        "form": row.form,
                        # Income Statement
                        "total_revenue": float(row.total_revenue) if row.total_revenue else 0,
                        "net_income": float(row.net_income) if row.net_income else 0,
                        "gross_profit": float(row.gross_profit) if row.gross_profit else 0,
                        "operating_income": float(row.operating_income) if row.operating_income else 0,
                        "interest_expense": float(row.interest_expense) if row.interest_expense else 0,
                        "income_tax_expense": float(row.income_tax_expense) if row.income_tax_expense else 0,
                        "cost_of_revenue": float(row.cost_of_revenue) if row.cost_of_revenue else 0,
                        # Balance Sheet
                        "total_assets": float(row.total_assets) if row.total_assets else 0,
                        "total_liabilities": float(row.total_liabilities) if row.total_liabilities else 0,
                        "stockholders_equity": float(row.stockholders_equity) if row.stockholders_equity else 0,
                        "current_assets": float(row.current_assets) if row.current_assets else 0,
                        "current_liabilities": float(row.current_liabilities) if row.current_liabilities else 0,
                        "accounts_receivable": float(row.accounts_receivable) if row.accounts_receivable else 0,
                        "inventory": float(row.inventory) if row.inventory else 0,
                        "cash_and_equivalents": float(row.cash_and_equivalents) if row.cash_and_equivalents else 0,
                        "long_term_debt": float(row.long_term_debt) if row.long_term_debt else 0,
                        "short_term_debt": float(row.short_term_debt) if row.short_term_debt else 0,
                        "total_debt": float(row.total_debt) if row.total_debt else 0,
                        # Cash Flow
                        "operating_cash_flow": float(row.operating_cash_flow) if row.operating_cash_flow else 0,
                        "capital_expenditures": float(row.capital_expenditures) if row.capital_expenditures else 0,
                        "free_cash_flow": float(row.free_cash_flow) if row.free_cash_flow else 0,
                        "dividends_paid": float(row.dividends_paid) if row.dividends_paid else 0,
                        # Other
                        "property_plant_equipment_net": (
                            float(row.property_plant_equipment_net) if row.property_plant_equipment_net else 0
                        ),
                        "shares_outstanding": float(row.shares_outstanding) if row.shares_outstanding else 0,
                        # Keep numeric values for data processor
                        "cash_flow_statement_qtrs": cf_qtrs,
                        "income_statement_qtrs": inc_qtrs,
                        # Add boolean flags for QuarterlyData constructor using centralized service
                        "is_ytd_cashflow": fiscal_period_service.is_ytd(cf_qtrs),
                        "is_ytd_income": fiscal_period_service.is_ytd(inc_qtrs),
                    }
                )

            # ═══════════════════════════════════════════════════════════════════════════
            # FY Period Handling - CORRECT Design
            # ═══════════════════════════════════════════════════════════════════════════
            # FY periods MUST be included in the input to quarterly_processor
            #
            # Data Flow:
            #   1. quarters_data contains BOTH FY and Q periods (from database)
            #   2. get_rolling_ttm_periods() will:
            #      - Use FY periods to compute missing Q4 (Q4 = FY - Q1 - Q2 - Q3)
            #      - Convert YTD Q2/Q3 to individual quarters
            #      - Return ONLY Q periods (Q1, Q2, Q3, Q4) - FY periods filtered internally
            #   3. DCF receives clean quarterly data (no FY periods in output)
            #
            # DO NOT filter FY periods here - they are needed for Q4 computation!
            # ═══════════════════════════════════════════════════════════════════════════

            fy_count = sum(1 for q in quarters_data if q.get("fiscal_period") == "FY")
            q_count = sum(1 for q in quarters_data if q.get("fiscal_period", "").startswith("Q"))

            self.logger.info(
                f"✅ Retrieved {len(quarters_data)} periods from processed table for {symbol} "
                f"({q_count} Q periods, {fy_count} FY periods - FY needed for Q4 computation)"
            )

            if len(quarters_data) < num_quarters:
                self.logger.warning(
                    f"Only {len(quarters_data)} quarters available for {symbol} (target: {num_quarters}). "
                    f"Company may be newly public or have incomplete filing history."
                )

            # Convert quarters_data to QuarterlyData objects
            quarterly_data_list = []

            for q in reversed(quarters_data):  # Reverse to chronological order (oldest first)
                # Phase 10: Check cache for this specific quarter (ADSH-based)
                quarter_cache_key = build_cache_key(
                    CacheType.QUARTERLY_METRICS,
                    symbol=symbol,
                    fiscal_year=q["fiscal_year"],
                    fiscal_period=q["fiscal_period"],
                    adsh=q["adsh"],
                )

                cached_quarter = self.cache.get(CacheType.QUARTERLY_METRICS, quarter_cache_key) if self.cache else None

                # Fix for Issue #1: Ensure cached data is QuarterlyData object, not dict/string
                if cached_quarter:
                    if isinstance(cached_quarter, dict):
                        # Cache returned dict, convert to QuarterlyData
                        try:
                            cached_quarter = QuarterlyData.from_dict(cached_quarter)
                            self.logger.debug(
                                f"Cache HIT (dict→QuarterlyData) for {symbol} {q['fiscal_year']}-{q['fiscal_period']}"
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to deserialize cached quarter for {symbol}: {e}, re-fetching")
                            cached_quarter = None
                    elif not isinstance(cached_quarter, QuarterlyData):
                        # Unexpected type, re-fetch
                        self.logger.warning(
                            f"Invalid cached quarter type for {symbol} {q['fiscal_year']}-{q['fiscal_period']}: "
                            f"{type(cached_quarter)}, re-fetching"
                        )
                        cached_quarter = None
                    else:
                        self.logger.debug(
                            f"Cache HIT for {symbol} {q['fiscal_year']}-{q['fiscal_period']} (ADSH: {q['adsh']})"
                        )

                if cached_quarter and isinstance(cached_quarter, QuarterlyData):
                    quarterly_data_list.append(cached_quarter)
                    continue

                # NEW: Try fetching from sec_companyfacts_processed table (3-table architecture)
                self.logger.debug(
                    f"🔍 [FETCH_QUARTERS] Attempting processed table for {symbol} {q['fiscal_year']}-{q['fiscal_period']} ADSH={q['adsh'][:20]}..."
                )
                processed_data = self._fetch_from_processed_table(
                    symbol, q["fiscal_year"], q["fiscal_period"], q["adsh"]
                )

                # Validate processed data quality before using it
                use_processed = False
                if processed_data:
                    # CLEAN ARCHITECTURE: Statement-level structure
                    # Revenue is in income_statement now, not financial_data
                    income_statement = processed_data.get("income_statement", {})
                    revenue = income_statement.get("total_revenue", 0)

                    # Check if processed data has meaningful values (not all zeros)
                    if revenue and revenue > 0:
                        ratios = processed_data.get("ratios", {})
                        quality = processed_data.get("data_quality_score", 0)
                        use_processed = True

                        # Extract and flatten statement-level structure for QuarterlyData
                        cash_flow = processed_data.get("cash_flow", {})
                        balance_sheet = processed_data.get("balance_sheet", {})

                        # CRITICAL: Extract is_ytd flags BEFORE flattening
                        # These flags indicate if Q2/Q3 values are YTD cumulative (from 10-Q filings)
                        # SEC doesn't provide is_ytd in raw JSON - we infer from fiscal_period in _fetch_from_processed_table()
                        is_ytd_cashflow = cash_flow.get("is_ytd", False)
                        is_ytd_income = income_statement.get("is_ytd", False)

                        # Create financial_data dict from statement-level structure
                        financial_data = {
                            # Income Statement (7 fields)
                            "revenues": income_statement.get("total_revenue", 0),
                            "net_income": income_statement.get("net_income", 0),
                            "gross_profit": income_statement.get("gross_profit", 0),
                            "operating_income": income_statement.get("operating_income", 0),
                            "interest_expense": income_statement.get("interest_expense", 0),
                            "income_tax_expense": income_statement.get("income_tax_expense", 0),
                            # Balance Sheet (10 fields)
                            "total_assets": balance_sheet.get("total_assets", 0),
                            "total_liabilities": balance_sheet.get("total_liabilities", 0),
                            "stockholders_equity": balance_sheet.get("stockholders_equity", 0),
                            "current_assets": balance_sheet.get("current_assets", 0),
                            "current_liabilities": balance_sheet.get("current_liabilities", 0),
                            # CRITICAL: Include debt fields for DCF/WACC calculation
                            "total_debt": balance_sheet.get("total_debt", 0),
                            "long_term_debt": balance_sheet.get("long_term_debt", 0),
                            "short_term_debt": balance_sheet.get("short_term_debt", 0),
                            "cash_and_equivalents": balance_sheet.get("cash_and_equivalents", 0),
                            "net_debt": balance_sheet.get("net_debt", 0),
                            # Cash Flow (3 fields)
                            "operating_cash_flow": cash_flow.get("operating_cash_flow", 0),
                            "capital_expenditures": cash_flow.get("capital_expenditures", 0),
                            "dividends_paid": cash_flow.get("dividends_paid", 0),
                            # Extract shares_outstanding for DCF/symbol_update
                            "weighted_average_diluted_shares_outstanding": q.get("shares_outstanding", 0),
                        }

                        self.logger.info(
                            f"✅ Using pre-processed data from sec_companyfacts_processed for "
                            f"{symbol} {q['fiscal_year']}-{q['fiscal_period']} (Revenue: ${revenue/1e9:.1f}B, Quality: {quality}%)"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️  Processed data for {symbol} {q['fiscal_year']}-{q['fiscal_period']} has zero/missing revenue "
                            f"(Revenue: ${revenue}), falling back to bulk tables (ADSH: {q['adsh']})"
                        )

                if not use_processed:
                    # FALLBACK: Extract from bulk tables using CanonicalKeyMapper
                    self.logger.warning(
                        f"⚠️  Processed data not found for {symbol} {q['fiscal_year']}-{q['fiscal_period']}, "
                        f"falling back to bulk tables with canonical key extraction (ADSH: {q['adsh']})"
                    )

                    # Get sector for canonical key extraction (with fallback to 'Unknown')
                    sector = self._get_sector_for_symbol(symbol)

                    # Define canonical keys needed for quarterly data extraction
                    canonical_keys_needed = FALLBACK_CANONICAL_KEYS

                    # Collect all XBRL tags from canonical keys (sector-aware)
                    all_tags = set()
                    for canonical_key in canonical_keys_needed:
                        # Get sector-specific + global fallback tags
                        tags = self.canonical_mapper.get_tags(canonical_key, sector)
                        all_tags.update(tags)

                    # Initialize strategy for bulk table access
                    from investigator.infrastructure.sec.data_strategy import get_fiscal_period_strategy

                    strategy = get_fiscal_period_strategy()

                    # Extract data from bulk tables using collected tag list
                    tag_values = strategy.get_num_data_for_adsh(q["adsh"], tags=list(all_tags))

                    # Helper to convert Decimal to float (bulk tables return Decimal type)
                    from decimal import Decimal

                    def to_float(val):
                        if isinstance(val, Decimal):
                            return float(val)
                        return val if val is not None else 0

                    # Map using canonical keys with sector-aware fallback priority
                    def extract_canonical_value(canonical_key: str) -> float:
                        """Extract value using canonical key fallback chain"""
                        tags_priority = self.canonical_mapper.get_tags(canonical_key, sector)
                        for tag in tags_priority:
                            if tag in tag_values and tag_values[tag] is not None:
                                return to_float(tag_values[tag])
                        return 0

                    ocf_value = extract_canonical_value("operating_cash_flow")
                    capex_value = extract_canonical_value("capital_expenditures")
                    fcf_value = extract_canonical_value("free_cash_flow")

                    # Derive free cash flow if missing/zero (ocf - |capex|)
                    if ocf_value is not None or capex_value is not None:
                        ocf_float = float(ocf_value or 0.0)
                        capex_float = float(capex_value or 0.0)
                        derived_fcf = ocf_float - abs(capex_float)
                        # Prefer derived value when canonical lookup returned None/0
                        if fcf_value is None or abs(float(fcf_value)) < 1e-6:
                            fcf_value = derived_fcf
                            if abs(derived_fcf) > 1e-6:
                                self.logger.debug(
                                    "🔄 [FALLBACK] Derived FCF for %s %s-%s via OCF %.2f - |CapEx| %.2f = %.2f",
                                    symbol,
                                    q["fiscal_year"],
                                    q["fiscal_period"],
                                    ocf_float,
                                    capex_float,
                                    derived_fcf,
                                )

                    financial_data = {
                        "revenues": extract_canonical_value("total_revenue"),
                        "net_income": extract_canonical_value("net_income"),
                        "total_assets": extract_canonical_value("total_assets"),
                        "total_liabilities": extract_canonical_value("total_liabilities"),
                        "stockholders_equity": extract_canonical_value("stockholders_equity"),
                        "current_assets": extract_canonical_value("current_assets"),
                        "current_liabilities": extract_canonical_value("current_liabilities"),
                        "long_term_debt": extract_canonical_value("long_term_debt"),
                        "short_term_debt": extract_canonical_value("short_term_debt"),
                        "total_debt": extract_canonical_value("total_debt"),
                        "operating_cash_flow": ocf_value,
                        "capital_expenditures": capex_value,
                        "free_cash_flow": fcf_value if fcf_value is not None else 0,
                        "dividends_paid": extract_canonical_value("dividends_paid"),
                        "weighted_average_diluted_shares_outstanding": extract_canonical_value(
                            "weighted_average_diluted_shares_outstanding"
                        ),
                    }

                    # Calculate ratios
                    ratios = self._calculate_quarterly_ratios(financial_data)

                    # Assess quality
                    quality = self._assess_quarter_quality(financial_data)

                # Create QuarterlyData with ADSH threading
                qdata = QuarterlyData(
                    fiscal_year=q["fiscal_year"],
                    fiscal_period=q["fiscal_period"],
                    financial_data=financial_data,
                    ratios=ratios,
                    data_quality=quality,
                    filing_date=str(q["filed"]),
                    is_ytd_cashflow=is_ytd_cashflow if use_processed else False,  # Pass YTD flags
                    is_ytd_income=is_ytd_income if use_processed else False,
                )

                # Thread ADSH through (Phase 10)
                qdata.adsh = q["adsh"]
                qdata.period_end_date = str(q["period_end"]) if q["period_end"] else None
                qdata.form = q["form"]

                quarterly_data_list.append(qdata)

                # Log quarter details for debugging
                self.logger.debug(
                    f"📊 [FETCH_QUARTERS] Created QuarterlyData for {symbol} {q['fiscal_year']}-{q['fiscal_period']}: "
                    f"OCF=${financial_data.get('operating_cash_flow', 0)/1e9:.2f}B, "
                    f"CapEx=${abs(financial_data.get('capital_expenditures', 0))/1e9:.2f}B, "
                    f"Quality={quality}%"
                )

                # Phase 10: Cache this quarter separately (ADSH-based caching)
                if self.cache:
                    self.cache.set(CacheType.QUARTERLY_METRICS, quarter_cache_key, qdata)
                    self.logger.debug(
                        f"Cached quarter {symbol} {q['fiscal_year']}-{q['fiscal_period']} " f"(ADSH: {q['adsh']})"
                    )

            log_quarterly_snapshot(self.logger, symbol, quarterly_data_list)
            self.logger.info(
                f"Successfully fetched {len(quarterly_data_list)} quarters for {symbol} using hybrid strategy: "
                f"{quarterly_data_list[0].period_label} → {quarterly_data_list[-1].period_label}"
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
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db_manager = get_db_manager()
            with db_manager.engine.connect() as conn:
                # Query for most recent period (prefer FY, then latest Q)
                result = conn.execute(
                    text(
                        """
                        SELECT *
                        FROM sec_companyfacts_processed
                        WHERE symbol = :symbol
                        ORDER BY
                            fiscal_year DESC,
                            CASE fiscal_period
                                WHEN 'FY' THEN 4
                                WHEN 'Q4' THEN 3
                                WHEN 'Q3' THEN 2
                                WHEN 'Q2' THEN 1
                                WHEN 'Q1' THEN 0
                            END DESC
                        LIMIT 1
                    """
                    ),
                    {"symbol": symbol},
                ).fetchone()

                if not result:
                    self.logger.warning(
                        f"[CLEAN ARCH] No processed data found for {symbol} in sec_companyfacts_processed"
                    )
                    return None

                row = dict(result._mapping)

                def safe_float(key: str) -> float:
                    value = row.get(key)
                    if value is None:
                        return 0.0
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return 0.0

                # CRITICAL DATA QUALITY CHECK: Detect corrupt data from failed YTD conversions
                # Root cause: Missing quarters → Failed YTD conversion → Negative revenue stored
                # Examples: ORCL Q2 (-$1,085M), ZS Q1 (NULL), META Q2 (-$178M)
                revenue = safe_float("total_revenue")
                operating_income = safe_float("operating_income")
                fiscal_year = row.get("fiscal_year")
                fiscal_period = row.get("fiscal_period")

                # Check 1: Negative revenue (physically impossible for most companies)
                if revenue < 0:
                    self.logger.error(
                        f"❌ CORRUPT DATA DETECTED: {symbol} {fiscal_year}-{fiscal_period} has NEGATIVE revenue: ${revenue:,.0f}. "
                        f"This indicates failed YTD conversion. DELETING corrupt record and forcing re-fetch."
                    )
                    # Delete corrupt record from database
                    conn.execute(
                        text(
                            """
                            DELETE FROM sec_companyfacts_processed
                            WHERE symbol = :symbol
                              AND fiscal_year = :fiscal_year
                              AND fiscal_period = :fiscal_period
                        """
                        ),
                        {"symbol": symbol, "fiscal_year": fiscal_year, "fiscal_period": fiscal_period},
                    )
                    conn.commit()
                    self.logger.warning(
                        f"⚠️  Deleted corrupt record for {symbol} {fiscal_year}-{fiscal_period}. "
                        f"SEC Agent should re-fetch and reprocess this period in next run."
                    )
                    return None  # Force fresh fetch from SEC

                # Check 2: Zero revenue when company should have revenue (likely incomplete data)
                if revenue == 0 and fiscal_period != "Q1":
                    self.logger.warning(
                        f"⚠️  {symbol} {fiscal_year}-{fiscal_period} has ZERO revenue. "
                        f"May indicate incomplete data or failed YTD conversion."
                    )

                # Log which period we're using
                self.logger.info(
                    f"[CLEAN ARCH] Fetched company data for {symbol} from processed table: "
                    f"{fiscal_year}-{fiscal_period} (filed: {row.get('filed_date')}) | Revenue: ${revenue:,.0f}"
                )

                # Map database columns to old extractor format for compatibility
                financial_metrics = {
                    # Income Statement
                    "revenues": safe_float("total_revenue"),
                    "net_income": safe_float("net_income"),
                    "gross_profit": safe_float("gross_profit"),
                    "operating_income": safe_float("operating_income"),
                    "cost_of_revenue": safe_float("cost_of_revenue"),
                    # Balance Sheet (NOTE: Field name mappings for compatibility)
                    "assets": safe_float("total_assets"),  # total_assets → assets
                    "equity": safe_float("stockholders_equity"),  # stockholders_equity → equity
                    "assets_current": safe_float("current_assets"),  # current_assets → assets_current
                    "liabilities_current": safe_float("current_liabilities"),
                    "liabilities": safe_float("total_liabilities"),  # total_liabilities → liabilities
                    "total_debt": safe_float("total_debt"),
                    "long_term_debt": safe_float("long_term_debt"),
                    "debt_current": safe_float("short_term_debt"),  # short_term_debt → debt_current
                    "inventory": safe_float("inventory"),
                    "cash_and_equivalents": safe_float("cash_and_equivalents"),
                    # Cash Flow
                    "operating_cash_flow": safe_float("operating_cash_flow"),
                    "capital_expenditures": safe_float("capital_expenditures"),
                    "free_cash_flow": safe_float("free_cash_flow"),
                    "shares_outstanding": safe_float("shares_outstanding"),
                    # Metadata
                    "fiscal_year": row.get("fiscal_year"),
                    "fiscal_period": row.get("fiscal_period"),
                    "symbol": row.get("symbol"),
                    "data_date": row.get("filed_date").isoformat() if row.get("filed_date") else None,
                    "weighted_average_diluted_shares_outstanding": safe_float(
                        "weighted_average_diluted_shares_outstanding"
                    ),
                    "cash_and_equivalents": safe_float("cash_and_equivalents"),
                }

                for key in PROCESSED_ADDITIONAL_FINANCIAL_KEYS:
                    financial_metrics[key] = safe_float(key)

                # Derive critical fields if the canonical tags were missing or zeroed in the warehouse.
                long_term_debt = financial_metrics.get("long_term_debt") or 0.0
                short_term_debt = financial_metrics.get("debt_current") or 0.0
                if not financial_metrics.get("total_debt") and (long_term_debt or short_term_debt):
                    financial_metrics["total_debt"] = long_term_debt + short_term_debt

                if not financial_metrics.get("cash"):
                    cash_guess = financial_metrics.get("cash_and_equivalents") or safe_float("cash")
                    financial_metrics["cash"] = cash_guess

                if not financial_metrics.get("shares_outstanding"):
                    shares_guess = financial_metrics.get("weighted_average_diluted_shares_outstanding") or safe_float(
                        "shares_outstanding"
                    )
                    financial_metrics["shares_outstanding"] = shares_guess

                financial_ratios = {
                    # Ratios (already calculated in database)
                    "current_ratio": safe_float("current_ratio"),
                    "quick_ratio": safe_float("quick_ratio"),
                    "debt_to_equity": safe_float("debt_to_equity"),
                    "roe": safe_float("roe"),
                    "roa": safe_float("roa"),
                    "gross_margin": safe_float("gross_margin"),
                    "operating_margin": safe_float("operating_margin"),
                    "net_margin": safe_float("net_margin"),
                    # Metadata
                    "symbol": row.get("symbol"),
                    "data_date": row.get("filed_date").isoformat() if row.get("filed_date") else None,
                    "raw_metrics": financial_metrics,  # Include raw metrics for reference
                }

                for key in PROCESSED_RATIO_KEYS:
                    financial_ratios[key] = safe_float(key)

                return {
                    "financial_metrics": financial_metrics,
                    "financial_ratios": financial_ratios,
                    "data_quality_score": safe_float("data_quality_score"),
                    "source": "clean_architecture",  # Mark as clean architecture source
                }

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
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            engine = get_db_manager().engine

            # DEBUG: Log query parameters
            self.logger.info(
                f"🔍 [PROCESSED_TABLE] Querying for {symbol} {fiscal_year}-{fiscal_period} ADSH={adsh[:20]}..."
            )

            query = text(
                """
                SELECT *
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND fiscal_year = :fiscal_year
                  AND fiscal_period = :fiscal_period
                  AND adsh = :adsh
                LIMIT 1
            """
            )

            with engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "symbol": symbol.upper(),
                        "fiscal_year": fiscal_year,
                        "fiscal_period": fiscal_period,
                        "adsh": adsh,
                    },
                ).fetchone()

            if not result:
                self.logger.warning(
                    f"❌ [PROCESSED_TABLE] No data found for {symbol} {fiscal_year}-{fiscal_period} ADSH={adsh[:20]}"
                )
                return None

            row = dict(result._mapping)

            # Helper to convert Decimal to float (database returns Decimal type)
            from decimal import Decimal

            def to_float(val):
                if val is None:
                    return 0.0
                if isinstance(val, Decimal):
                    return float(val)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0

            # CLEAN ARCHITECTURE: Statement-level structure matching GAAP
            # Organized by financial statement type with YTD tracking

            # Get FiscalPeriodService for centralized YTD detection
            fiscal_period_service = get_fiscal_period_service()

            # Determine if this is YTD data based on statement-specific qtrs values
            # qtrs=1 means point-in-time, qtrs>=2 means YTD cumulative
            income_qtrs_val = row.get("income_statement_qtrs")
            cashflow_qtrs_val = row.get("cash_flow_statement_qtrs")
            is_ytd_income = fiscal_period_service.is_ytd(income_qtrs_val) if income_qtrs_val else False
            is_ytd_cashflow = fiscal_period_service.is_ytd(cashflow_qtrs_val) if cashflow_qtrs_val else False

            ocf_val = to_float(row.get("operating_cash_flow"))
            capex_val = to_float(row.get("capital_expenditures"))
            raw_fcf = row.get("free_cash_flow")  # preserve original for diagnostics
            free_cash_flow_val = to_float(row.get("free_cash_flow"))

            # Derive FCF when missing or zero (common gap in historical records)
            if (raw_fcf is None or abs(free_cash_flow_val) < 1e-6) and (ocf_val is not None and capex_val is not None):
                derived_fcf = float(ocf_val) - abs(float(capex_val))
                free_cash_flow_val = derived_fcf
                if abs(derived_fcf) > 1e-6:
                    self.logger.debug(
                        "🔄 [PROCESSED_TABLE] Derived FCF for %s %s-%s via OCF %.2f - |CapEx| %.2f = %.2f",
                        symbol,
                        fiscal_year,
                        fiscal_period,
                        ocf_val,
                        capex_val,
                        derived_fcf,
                    )

            data = {
                # Metadata
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "adsh": adsh,
                # Income Statement (YTD based on income_statement_qtrs)
                "income_statement": {
                    "total_revenue": to_float(row.get("total_revenue")),
                    "net_income": to_float(row.get("net_income")),
                    "gross_profit": to_float(row.get("gross_profit")),
                    "operating_income": to_float(row.get("operating_income")),
                    "cost_of_revenue": to_float(row.get("cost_of_revenue")),
                    "research_and_development_expense": to_float(row.get("research_and_development_expense")),
                    "selling_general_administrative_expense": to_float(
                        row.get("selling_general_administrative_expense")
                    ),
                    "operating_expenses": to_float(row.get("operating_expenses")),
                    "interest_expense": to_float(row.get("interest_expense")),
                    "income_tax_expense": to_float(row.get("income_tax_expense")),
                    "earnings_per_share": to_float(row.get("earnings_per_share")),
                    "earnings_per_share_diluted": to_float(row.get("earnings_per_share_diluted")),
                    "preferred_stock_dividends": to_float(row.get("preferred_stock_dividends")),
                    "common_stock_dividends": to_float(row.get("common_stock_dividends")),
                    "weighted_average_diluted_shares_outstanding": to_float(
                        row.get("weighted_average_diluted_shares_outstanding")
                    ),
                    "is_ytd": is_ytd_income,
                },
                # Cash Flow Statement (YTD based on cash_flow_statement_qtrs)
                "cash_flow": {
                    "operating_cash_flow": ocf_val,
                    "capital_expenditures": capex_val,
                    "free_cash_flow": free_cash_flow_val,
                    "dividends_paid": to_float(row.get("dividends_paid")),
                    "investing_cash_flow": to_float(row.get("investing_cash_flow")),
                    "financing_cash_flow": to_float(row.get("financing_cash_flow")),
                    "depreciation_amortization": to_float(row.get("depreciation_amortization")),
                    "stock_based_compensation": to_float(row.get("stock_based_compensation")),
                    "preferred_stock_dividends": to_float(row.get("preferred_stock_dividends")),
                    "common_stock_dividends": to_float(row.get("common_stock_dividends")),
                    "is_ytd": is_ytd_cashflow,
                },
                # Balance Sheet (ALWAYS point-in-time snapshot)
                "balance_sheet": {
                    "total_assets": to_float(row.get("total_assets")),
                    "total_liabilities": to_float(row.get("total_liabilities")),
                    "stockholders_equity": to_float(row.get("stockholders_equity")),
                    "current_assets": to_float(row.get("current_assets")),
                    "current_liabilities": to_float(row.get("current_liabilities")),
                    "retained_earnings": to_float(row.get("retained_earnings")),
                    "accounts_payable": to_float(row.get("accounts_payable")),
                    "accrued_liabilities": to_float(row.get("accrued_liabilities")),
                    "long_term_debt": to_float(row.get("long_term_debt")),
                    "short_term_debt": to_float(row.get("short_term_debt")),
                    "total_debt": to_float(row.get("total_debt")),
                    "net_debt": to_float(row.get("net_debt")),
                    "cash_and_equivalents": to_float(row.get("cash_and_equivalents")),
                    "accounts_receivable": to_float(row.get("accounts_receivable")),
                    "inventory": to_float(row.get("inventory")),
                    "property_plant_equipment": to_float(row.get("property_plant_equipment")),
                    "accumulated_depreciation": to_float(row.get("accumulated_depreciation")),
                    "property_plant_equipment_net": to_float(row.get("property_plant_equipment_net")),
                    "goodwill": to_float(row.get("goodwill")),
                    "intangible_assets": to_float(row.get("intangible_assets")),
                    "deferred_revenue": to_float(row.get("deferred_revenue")),
                    "treasury_stock": to_float(row.get("treasury_stock")),
                    "other_comprehensive_income": to_float(row.get("other_comprehensive_income")),
                    "book_value": to_float(row.get("book_value")),
                    "book_value_per_share": to_float(row.get("book_value_per_share")),
                    "working_capital": to_float(row.get("working_capital")),
                },
                "market_metrics": {
                    "market_cap": to_float(row.get("market_cap")),
                    "enterprise_value": to_float(row.get("enterprise_value")),
                    "shares_outstanding": to_float(row.get("shares_outstanding")),
                },
                # Financial Ratios (organized by category)
                "ratios": {
                    "liquidity": {
                        "current_ratio": to_float(row.get("current_ratio")),
                        "quick_ratio": to_float(row.get("quick_ratio")),
                    },
                    "leverage": {
                        "debt_to_equity": to_float(row.get("debt_to_equity")),
                        "interest_coverage": to_float(row.get("interest_coverage")),
                    },
                    "profitability": {
                        "roa": to_float(row.get("roa")),
                        "roe": to_float(row.get("roe")),
                        "gross_margin": to_float(row.get("gross_margin")),
                        "operating_margin": to_float(row.get("operating_margin")),
                        "net_margin": to_float(row.get("net_margin")),
                        # return_on_assets and return_on_equity columns dropped (duplicates of roa/roe)
                    },
                    "efficiency": {
                        "asset_turnover": to_float(row.get("asset_turnover")),
                    },
                    "distribution": {
                        "dividend_payout_ratio": to_float(row.get("dividend_payout_ratio")),
                        "dividend_yield": to_float(row.get("dividend_yield")),
                    },
                    "tax": {
                        "effective_tax_rate": to_float(row.get("effective_tax_rate")),
                    },
                },
                # Data Quality
                "data_quality_score": to_float(row.get("data_quality_score")),
            }

            # 🔍 DEBUG TRACE: Log interest_expense after income_statement extraction from database
            interest_expense = data["income_statement"].get("interest_expense", 0)
            income_tax_expense = data["income_statement"].get("income_tax_expense", 0)
            self.logger.info(
                f"🔍 [FUNDAMENTAL_FETCH] {symbol} {fiscal_year}-{fiscal_period} income_statement: "
                f"interest_expense=${interest_expense/1e6:.2f}M, income_tax=${income_tax_expense/1e9:.2f}B"
            )

            # DEBUG: Log successful retrieval with detailed YTD info
            revenue = data["income_statement"]["total_revenue"]
            ocf = data["cash_flow"]["operating_cash_flow"]
            capex = data["cash_flow"]["capital_expenditures"]
            fcf = data["cash_flow"]["free_cash_flow"]

            # Show YTD status for each statement type
            inc_ytd_str = "YTD" if is_ytd_income else "PIT"
            cf_ytd_str = "YTD" if is_ytd_cashflow else "PIT"

            self.logger.info(
                f"✅ [PROCESSED_TABLE] {symbol} {fiscal_year}-{fiscal_period}: "
                f"Income({inc_ytd_str}, qtrs={income_qtrs_val or 'NULL'}), "
                f"CashFlow({cf_ytd_str}, qtrs={cashflow_qtrs_val or 'NULL'})"
            )
            self.logger.info(
                f"   📊 Raw DB Values: Revenue=${revenue/1e9:.2f}B, "
                f"OCF=${ocf/1e9:.2f}B, CapEx=${capex/1e9:.2f}B, FCF=${fcf/1e9:.2f}B"
            )

            return data

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
        """
        Assess data quality for a single quarter.

        Args:
            financial_data: Financial metrics for the quarter

        Returns:
            Dictionary with quality metrics (completeness, consistency)
        """
        required_fields = [
            "revenues",
            "net_income",
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "operating_cash_flow",
            "capital_expenditures",
        ]

        # Calculate completeness (% of required fields present and non-zero)
        present_fields = sum(1 for field in required_fields if financial_data.get(field, 0) != 0)
        completeness = (present_fields / len(required_fields)) * 100

        # Calculate consistency (basic sanity checks)
        consistency_score = 100.0
        issues = []

        # Check: Assets = Liabilities + Equity (within 5% tolerance)
        assets = financial_data.get("total_assets", 0)
        liabilities = financial_data.get("total_liabilities", 0)
        equity = financial_data.get("stockholders_equity", 0)

        if assets > 0:
            balance = liabilities + equity
            balance_error = abs(assets - balance) / assets
            if balance_error > 0.05:  # 5% tolerance
                consistency_score -= 20
                issues.append(f"Balance sheet mismatch: {balance_error:.1%}")

        # Check: Revenue > 0 and reasonable relative to assets
        revenue = financial_data.get("revenues", 0)
        if revenue <= 0:
            consistency_score -= 30
            issues.append("Zero or negative revenue")

        # Check: OCF should be reasonable relative to net income
        ocf = financial_data.get("operating_cash_flow", 0)
        net_income = financial_data.get("net_income", 0)
        if net_income != 0 and ocf != 0:
            cash_conversion = ocf / net_income
            if cash_conversion < 0.3 or cash_conversion > 5.0:
                consistency_score -= 15
                issues.append(f"Unusual cash conversion: {cash_conversion:.1f}x")

        return {
            "completeness": completeness,
            "consistency": max(0, consistency_score),  # Ensure non-negative
            "issues": issues,
        }

    def _analyze_revenue_trend(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """
        Analyze revenue trend: accelerating, stable, or decelerating.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with:
                - trend: 'accelerating', 'stable', or 'decelerating'
                - q_over_q_growth: List of quarter-over-quarter growth rates (%)
                - y_over_y_growth: List of year-over-year growth rates (%)
                - average_growth: Average Q/Q growth rate (%)
                - volatility: Standard deviation of Q/Q growth (%)
                - consistency_score: 0-100 (higher = more consistent)
        """
        if len(quarterly_data) < 2:
            return {
                "trend": "insufficient_data",
                "q_over_q_growth": [],
                "y_over_y_growth": [],
                "average_growth": 0.0,
                "volatility": 0.0,
                "consistency_score": 0.0,
            }

        # Extract revenues
        revenues = [q.financial_data.get("revenues", 0) for q in quarterly_data]

        # Calculate Q/Q growth rates
        qoq_growth = []
        for i in range(1, len(revenues)):
            if revenues[i - 1] > 0:
                growth = ((float(revenues[i]) - float(revenues[i - 1])) / float(revenues[i - 1])) * 100
                qoq_growth.append(growth)
            else:
                qoq_growth.append(0.0)

        # Calculate Y/Y growth rates (4 quarters lag)
        yoy_growth = []
        for i in range(4, len(revenues)):
            if revenues[i - 4] > 0:
                growth = ((float(revenues[i]) - float(revenues[i - 4])) / float(revenues[i - 4])) * 100
                yoy_growth.append(growth)
            else:
                yoy_growth.append(0.0)

        # Calculate average growth
        avg_growth = sum(qoq_growth) / len(qoq_growth) if qoq_growth else 0.0

        # Calculate volatility (standard deviation)
        if len(qoq_growth) > 1:
            variance = sum((g - avg_growth) ** 2 for g in qoq_growth) / len(qoq_growth)
            volatility = variance**0.5
        else:
            volatility = 0.0

        # Calculate consistency score (0-100, inverse of volatility)
        # Low volatility → high consistency
        # Normalize: volatility of 10% → score of 0, volatility of 0% → score of 100
        if volatility > 0:
            consistency_score = max(0, min(100, 100 - (volatility * 5)))
        else:
            consistency_score = 100.0

        # Determine trend (accelerating, stable, or decelerating)
        # Compare early vs late quarters
        if len(qoq_growth) >= 6:
            early_avg = sum(qoq_growth[:3]) / 3
            late_avg = sum(qoq_growth[-3:]) / 3

            # Threshold for acceleration/deceleration: 2 percentage points
            if late_avg > early_avg + 2.0:
                trend = "accelerating"
            elif late_avg < early_avg - 2.0:
                trend = "decelerating"
            else:
                trend = "stable"
        else:
            # Not enough data for trend determination
            trend = "stable"

        return {
            "trend": trend,
            "q_over_q_growth": [round(g, 2) for g in qoq_growth],
            "y_over_y_growth": [round(g, 2) for g in yoy_growth],
            "average_growth": round(avg_growth, 2),
            "volatility": round(volatility, 2),
            "consistency_score": round(consistency_score, 1),
        }

    def _analyze_margin_trend(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """
        Analyze margin trends: expanding, stable, or contracting.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with margin trends and historical values
        """
        if len(quarterly_data) < 2:
            return {
                "gross_margin_trend": "insufficient_data",
                "operating_margin_trend": "insufficient_data",
                "net_margin_trend": "insufficient_data",
                "gross_margins": [],
                "operating_margins": [],
                "net_margins": [],
            }

        gross_margins = []
        operating_margins = []
        net_margins = []

        for q in quarterly_data:
            revenue = q.financial_data.get("revenues", 0)
            net_income = q.financial_data.get("net_income", 0)

            # Calculate margins
            if revenue > 0:
                # Net margin (we have this directly)
                net_margin = (float(net_income) / float(revenue)) * 100
                net_margins.append(net_margin)

                # For gross and operating margins, use ratios if available
                if q.ratios:
                    gross_margin = q.ratios.get("profit_margin", net_margin)
                    operating_margin = q.ratios.get("profit_margin", net_margin)
                else:
                    # Fallback to net margin as proxy
                    gross_margin = net_margin
                    operating_margin = net_margin

                gross_margins.append(gross_margin)
                operating_margins.append(operating_margin)
            else:
                gross_margins.append(0.0)
                operating_margins.append(0.0)
                net_margins.append(0.0)

        # Determine trends (compare early vs late quarters)
        def determine_margin_trend(margins):
            if len(margins) < 4:
                return "stable"

            early_avg = sum(float(m) for m in margins[: len(margins) // 2]) / (len(margins) // 2)
            late_avg = sum(float(m) for m in margins[len(margins) // 2 :]) / (len(margins) - len(margins) // 2)

            # Threshold: 1 percentage point
            if late_avg > early_avg + 1.0:
                return "expanding"
            elif late_avg < early_avg - 1.0:
                return "contracting"
            else:
                return "stable"

        return {
            "gross_margin_trend": determine_margin_trend(gross_margins),
            "operating_margin_trend": determine_margin_trend(operating_margins),
            "net_margin_trend": determine_margin_trend(net_margins),
            "gross_margins": [round(m, 2) for m in gross_margins],
            "operating_margins": [round(m, 2) for m in operating_margins],
            "net_margins": [round(m, 2) for m in net_margins],
        }

    def _analyze_cash_flow_trend(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """
        Analyze cash flow quality and trend.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with cash flow metrics and quality score
        """
        if len(quarterly_data) < 2:
            return {
                "trend": "insufficient_data",
                "operating_cash_flow": [],
                "free_cash_flow": [],
                "cash_conversion_ratio": [],
                "quality_of_earnings": 0.0,
            }

        ocf_values = []
        fcf_values = []
        cash_conversion = []

        for q in quarterly_data:
            ocf = q.financial_data.get("operating_cash_flow", 0)
            capex = q.financial_data.get("capital_expenditures", 0)
            net_income = q.financial_data.get("net_income", 0)

            ocf_values.append(ocf)
            fcf = ocf - capex
            fcf_values.append(fcf)

            # Cash conversion ratio (OCF / Net Income)
            if net_income > 0:
                conversion = (ocf / net_income) * 100
                cash_conversion.append(conversion)
            else:
                cash_conversion.append(0.0)

        # Determine trend
        if len(ocf_values) >= 4:
            early_avg = sum(ocf_values[: len(ocf_values) // 2]) / (len(ocf_values) // 2)
            late_avg = sum(ocf_values[len(ocf_values) // 2 :]) / (len(ocf_values) - len(ocf_values) // 2)

            if late_avg > early_avg * 1.1:  # 10% improvement
                trend = "improving"
            elif late_avg < early_avg * 0.9:  # 10% decline
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Quality of earnings score (0-100)
        # Based on cash conversion ratio
        # >100% = excellent (95-100), 80-100% = good (80-95), 50-80% = fair (50-80), <50% = poor (0-50)
        if cash_conversion:
            avg_conversion = sum(cash_conversion) / len(cash_conversion)
            if avg_conversion >= 100:
                quality_score = min(100, 95 + (avg_conversion - 100) / 20)
            elif avg_conversion >= 80:
                quality_score = 80 + (avg_conversion - 80)
            elif avg_conversion >= 50:
                quality_score = 50 + (avg_conversion - 50) * 0.6
            else:
                quality_score = avg_conversion
        else:
            quality_score = 0.0

        return {
            "trend": trend,
            "operating_cash_flow": [round(ocf, 0) for ocf in ocf_values],
            "free_cash_flow": [round(fcf, 0) for fcf in fcf_values],
            "cash_conversion_ratio": [round(cc, 1) for cc in cash_conversion],
            "quality_of_earnings": round(quality_score, 1),
        }

    def _calculate_quarterly_comparisons(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """
        Calculate quarter-over-quarter and year-over-year comparisons.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with Q/Q and Y/Y comparisons for key metrics
        """
        if len(quarterly_data) < 2:
            return {
                "q_over_q": {"revenue": [], "net_income": [], "eps": []},
                "y_over_y": {"revenue": [], "net_income": [], "eps": []},
            }

        revenues = [q.financial_data.get("revenues", 0) for q in quarterly_data]
        net_incomes = [q.financial_data.get("net_income", 0) for q in quarterly_data]

        # Q/Q comparisons
        qoq_revenue = []
        qoq_net_income = []

        for i in range(1, len(revenues)):
            if revenues[i - 1] > 0:
                qoq_revenue.append(((float(revenues[i]) - float(revenues[i - 1])) / float(revenues[i - 1])) * 100)
            else:
                qoq_revenue.append(0.0)

            if net_incomes[i - 1] > 0:
                qoq_net_income.append(
                    ((float(net_incomes[i]) - float(net_incomes[i - 1])) / float(net_incomes[i - 1])) * 100
                )
            else:
                qoq_net_income.append(0.0)

        # Y/Y comparisons (4 quarters lag)
        yoy_revenue = []
        yoy_net_income = []

        for i in range(4, len(revenues)):
            if revenues[i - 4] > 0:
                yoy_revenue.append(((float(revenues[i]) - float(revenues[i - 4])) / float(revenues[i - 4])) * 100)
            else:
                yoy_revenue.append(0.0)

            if net_incomes[i - 4] > 0:
                yoy_net_income.append(
                    ((float(net_incomes[i]) - float(net_incomes[i - 4])) / float(net_incomes[i - 4])) * 100
                )
            else:
                yoy_net_income.append(0.0)

        return {
            "q_over_q": {
                "revenue": [round(g, 2) for g in qoq_revenue],
                "net_income": [round(g, 2) for g in qoq_net_income],
                "eps": [],  # EPS not yet calculated
            },
            "y_over_y": {
                "revenue": [round(g, 2) for g in yoy_revenue],
                "net_income": [round(g, 2) for g in yoy_net_income],
                "eps": [],  # EPS not yet calculated
            },
        }

    def _detect_cyclical_patterns(self, quarterly_data: List[QuarterlyData]) -> Dict:
        """
        Detect seasonal/cyclical business patterns.

        Args:
            quarterly_data: List of QuarterlyData objects (chronologically sorted)

        Returns:
            Dictionary with cyclical pattern analysis
        """
        if len(quarterly_data) < 8:
            return {
                "is_cyclical": False,
                "seasonal_pattern": "insufficient_data",
                "quarterly_strength": {},
                "pattern_confidence": 0.0,
            }

        # Group revenues by quarter (Q1, Q2, Q3, Q4)
        quarter_revenues = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

        for q in quarterly_data:
            period = q.fiscal_period
            revenue = q.financial_data.get("revenues", 0)
            if period in quarter_revenues:
                quarter_revenues[period].append(revenue)

        # Calculate average revenue per quarter
        quarter_averages = {}
        for period, revenues in quarter_revenues.items():
            if revenues:
                quarter_averages[period] = sum(revenues) / len(revenues)
            else:
                quarter_averages[period] = 0

        # Calculate overall average
        overall_avg = sum(quarter_averages.values()) / len(quarter_averages) if quarter_averages else 0

        # Calculate strength (% above/below average)
        quarterly_strength = {}
        for period, avg in quarter_averages.items():
            if overall_avg > 0:
                strength = ((avg - overall_avg) / overall_avg) * 100
                quarterly_strength[period] = round(strength, 1)
            else:
                quarterly_strength[period] = 0.0

        # Determine if cyclical (any quarter > 15% different from average)
        max_deviation = max(abs(s) for s in quarterly_strength.values())
        is_cyclical = max_deviation > 15.0

        # Identify strongest quarter
        strongest_quarter = max(quarterly_strength, key=quarterly_strength.get)

        # Pattern confidence (based on consistency across years)
        # Higher confidence if pattern repeats
        if len(quarterly_data) >= 8:
            pattern_confidence = min(100, max_deviation * 3)
        else:
            pattern_confidence = max_deviation * 2

        # Determine seasonal pattern
        if is_cyclical:
            if quarterly_strength[strongest_quarter] > 15:
                seasonal_pattern = f"{strongest_quarter}_strong"
            else:
                seasonal_pattern = "moderate_cyclical"
        else:
            seasonal_pattern = "non_cyclical"

        return {
            "is_cyclical": is_cyclical,
            "seasonal_pattern": seasonal_pattern,
            "quarterly_strength": quarterly_strength,
            "pattern_confidence": round(pattern_confidence, 1),
        }

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
        """
        Assess the quality and completeness of financial data.
        Returns a data quality score (0-100) and detailed assessment.

        FEATURE #1: Data Quality Scoring (migrated from old solution)
        Similar to old solution's extraction-level quality scoring.

        Also tracks raw extraction quality vs enhanced quality to quantify
        value of data enrichment (market data integration, ratio calculations).

        ENHANCEMENT: Now uses DataNormalizer for schema harmonization and
        includes debt metrics in completeness scoring.
        """
        financials = company_data.get("financials") or {}
        market_data = company_data.get("market_data") or {}

        # Step 1: Normalize field names to snake_case for internal consistency
        # Note: DataNormalizer.assess_completeness() internally converts to camelCase
        # for checking against CORE_METRICS, so we normalize to snake_case here first
        normalized_financials = DataNormalizer.normalize_field_names(financials, to_camel_case=False)

        # Step 2: Use DataNormalizer's enhanced completeness assessment (includes debt metrics)
        completeness_assessment = DataNormalizer.assess_completeness(normalized_financials, include_debt_metrics=True)

        core_populated = completeness_assessment["core_metrics_count"]
        debt_populated = completeness_assessment["debt_metrics_count"]

        # Log warnings for missing debt metrics (explicit upstream gap tracking)
        if completeness_assessment["missing_debt"]:
            symbol = company_data.get("symbol", "UNKNOWN")
            missing_debt_str = ", ".join(completeness_assessment["missing_debt"])
            self.logger.warning(
                f"⚠️  UPSTREAM DATA GAP for {symbol}: Missing debt metrics: {missing_debt_str}. "
                f"Debt-related ratios may be unreliable."
            )

        # Market data metrics (check both camelCase and snake_case for compatibility).
        # NOTE: this dual check keeps legacy prompts (marketCap) and new pipeline fields (market_cap)
        # in sync, which prevents false "1/2 populated" warnings in data quality logs.
        has_price = market_data.get("current_price", 0) != 0 or company_data.get("current_price", 0) != 0
        has_market_cap = (
            market_data.get("market_cap", 0) != 0
            or market_data.get("market_cap", 0) != 0
            or company_data.get("market_cap", 0) != 0
            or company_data.get("market_cap", 0) != 0
        )
        market_populated = (1 if has_price else 0) + (1 if has_market_cap else 0)

        # Calculated ratio metrics (from _calculate_financial_ratios)
        ratio_metrics = [
            "pe_ratio",
            "price_to_book",
            "current_ratio",
            "debt_to_equity",
            "roe",
            "roa",
            "gross_margin",
            "operating_margin",
        ]
        ratio_populated = sum(1 for m in ratio_metrics if ratios.get(m, 0) != 0)

        # Calculate completeness scores
        # Use DataNormalizer's score for core metrics, then add market/ratio components
        core_completeness = completeness_assessment["score"]  # Already includes debt metrics
        market_completeness = (market_populated / 2) * 100  # 2 metrics: price + market_cap
        ratio_completeness = (ratio_populated / len(ratio_metrics)) * 100

        # Overall completeness (weighted average)
        # Core+Debt: 50%, Market data: 25%, Ratios: 25%
        completeness_score = core_completeness * 0.50 + market_completeness * 0.25 + ratio_completeness * 0.25

        # Check for data consistency (red flags)
        consistency_issues = []

        # Check for impossible values (with None-safe comparisons)
        net_income = financials.get("net_income") or 0
        total_revenue = financials.get("revenues") or 0
        if net_income < 0 and total_revenue > 0:
            if abs(net_income) > total_revenue:
                consistency_issues.append("Net loss exceeds revenue (possible data error)")

        current_liabilities = financials.get("current_liabilities") or 0
        total_assets = financials.get("total_assets") or 0
        if current_liabilities > 0 and total_assets > 0 and current_liabilities > total_assets:
            consistency_issues.append("Current liabilities exceed total assets (data warning)")

        current_ratio = ratios.get("current_ratio") or 0
        if current_ratio > 100:  # Impossibly high current ratio
            consistency_issues.append("Unrealistic current ratio (possible unit error)")

        # Calculate consistency score (100 if no issues, -10 for each issue)
        consistency_score = max(0, 100 - (len(consistency_issues) * 10))

        # ENHANCEMENT: Explicit warnings for zeroed critical ratios due to upstream gaps
        symbol = company_data.get("symbol", "UNKNOWN")
        DataNormalizer.validate_and_warn(ratios, symbol, self.logger)

        # Calculate overall data quality score
        # Completeness: 70%, Consistency: 30%
        data_quality_score = (completeness_score * 0.70) + (consistency_score * 0.30)

        # Determine quality grade
        if data_quality_score >= 90:
            quality_grade = "Excellent"
        elif data_quality_score >= 75:
            quality_grade = "Good"
        elif data_quality_score >= 60:
            quality_grade = "Fair"
        elif data_quality_score >= 40:
            quality_grade = "Poor"
        else:
            quality_grade = "Very Poor"

        # FEATURE #3: Enhanced vs Extraction Quality Comparison
        # Calculate "extraction quality" (raw financial data only, before enrichment)
        extraction_completeness = core_completeness  # Only SEC financial data
        extraction_quality = (extraction_completeness * 0.70) + (consistency_score * 0.30)

        # Calculate enhancement delta
        quality_improvement = data_quality_score - extraction_quality
        improvement_sources = []

        if market_populated > 0:
            improvement_sources.append(f"market data (+{market_populated} metrics)")
        if ratio_populated > 0:
            improvement_sources.append(f"calculated ratios (+{ratio_populated} metrics)")

        # Generate enhancement summary
        if quality_improvement > 0:
            enhancement_summary = (
                f"Data enrichment improved quality by {quality_improvement:.1f} points "
                f"({extraction_quality:.1f}% → {data_quality_score:.1f}%) through: "
                f"{', '.join(improvement_sources)}"
            )
        else:
            enhancement_summary = "No data enrichment applied (extraction-only data)"

        return {
            "data_quality_score": round(data_quality_score, 1),
            "quality_grade": quality_grade,
            "completeness_score": round(completeness_score, 1),
            "consistency_score": round(consistency_score, 1),
            "core_metrics_populated": f"{core_populated}/{completeness_assessment['core_metrics_total']}",
            "market_metrics_populated": f"{market_populated}/2",
            "ratio_metrics_populated": f"{ratio_populated}/{len(ratio_metrics)}",
            "consistency_issues": consistency_issues,
            "assessment": f"Data quality is {quality_grade.lower()} with {completeness_score:.0f}% completeness",
            # FEATURE #3: Enhanced vs extraction quality tracking
            "extraction_quality": round(extraction_quality, 1),
            "quality_improvement": round(quality_improvement, 1),
            "improvement_sources": improvement_sources,
            "enhancement_summary": enhancement_summary,
        }

    def _calculate_confidence_level(self, data_quality: Dict) -> Dict:
        """
        Calculate confidence level based on data quality score.

        FEATURE #2: Confidence Level Adjustment (migrated from old solution concept)
        Maps data quality score to confidence level for investment decisions.

        Args:
            data_quality: Data quality assessment from _assess_data_quality()

        Returns:
            Dict with confidence_level, confidence_score, and rationale
        """
        data_quality_score = data_quality.get("data_quality_score", 0)
        quality_grade = data_quality.get("quality_grade", "Unknown")
        consistency_issues = data_quality.get("consistency_issues", [])

        # Map data quality score to confidence level
        if data_quality_score >= 90:
            confidence_level = "VERY HIGH"
            confidence_score = 95
            rationale = "Excellent data quality with complete, consistent financial metrics"
        elif data_quality_score >= 75:
            confidence_level = "HIGH"
            confidence_score = 85
            rationale = "Good data quality with minor gaps, analysis is reliable"
        elif data_quality_score >= 60:
            confidence_level = "MODERATE"
            confidence_score = 70
            rationale = "Fair data quality with some gaps, exercise caution in decision-making"
        elif data_quality_score >= 40:
            confidence_level = "LOW"
            confidence_score = 50
            rationale = "Poor data quality with significant gaps, recommendations should be treated with skepticism"
        else:
            confidence_level = "VERY LOW"
            confidence_score = 30
            rationale = "Very poor data quality, analysis may be unreliable, seek additional data sources"

        # Adjust confidence down if there are consistency issues
        if consistency_issues:
            confidence_score -= 10
            rationale += f" (adjusted down due to {len(consistency_issues)} data consistency issue(s))"

        return {
            "confidence_level": confidence_level,
            "confidence_score": confidence_score,
            "rationale": rationale,
            "based_on_data_quality": data_quality_score,
            "quality_grade": quality_grade,
        }

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
        return {
            "response": payload,
            "prompt": "",
            "model_info": {
                "model": f"deterministic-{label}",
                "temperature": 0.0,
                "top_p": 0.0,
                "format": "json",
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "analysis_type": label,
                "cache_type": "deterministic_analysis",
            },
        }

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

        cache_key: Dict[str, Any] = {"symbol": symbol, "llm_type": label}
        if period:
            cache_key["period"] = period

        wrapped = {
            "response": payload,
            "metadata": {
                "cached_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "analysis_type": label,
                "period": period,
            },
        }

        try:
            self.cache.set(CacheType.LLM_RESPONSE, cache_key, wrapped)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to store deterministic %s for %s: %s", label, symbol, exc)

    async def _analyze_financial_health(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """Evaluate liquidity, solvency, and working-capital resilience without LLM calls."""
        financials = self._require_financials(company_data)

        def assess_liquidity() -> tuple[str, str, float]:
            current_ratio = ratios.get("current_ratio")
            quick_ratio = ratios.get("quick_ratio")
            if current_ratio is None:
                return (
                    "Unknown",
                    "Current ratio unavailable; liquidity requires manual review.",
                    55.0,
                )
            if current_ratio >= 2.0:
                label, score = "Strong", 95.0
            elif current_ratio >= 1.2:
                label, score = "Adequate", 75.0
            else:
                label, score = "Weak", 45.0
            if quick_ratio:
                commentary = f"Current ratio {current_ratio:.2f}; quick ratio {quick_ratio:.2f}."
            else:
                commentary = f"Current ratio {current_ratio:.2f}."
            return label, commentary, score

        def assess_solvency() -> tuple[str, str, float]:
            debt_to_equity = ratios.get("debt_to_equity")
            debt_to_assets = ratios.get("debt_to_assets")
            total_debt = financials.get("total_debt") or 0
            if total_debt == 0:
                return "Debt Free", "Company has no financial leverage.", 95.0
            if debt_to_equity is None:
                return "Unknown", "Debt-to-equity unavailable; solvency indeterminate.", 55.0
            if debt_to_equity <= 1.0:
                label, score = "Comfortable", 80.0
            elif debt_to_equity <= 2.0:
                label, score = "Managed", 65.0
            else:
                label, score = "Leveraged", 45.0

            def _fmt_ratio(value: Optional[float]) -> str:
                if value is None:
                    return "n/a"
                try:
                    return f"{value:.2f}"
                except (TypeError, ValueError):
                    return "n/a"

            commentary = f"Debt/Equity {debt_to_equity:.2f}; Debt/Assets {_fmt_ratio(debt_to_assets)}."
            return label, commentary, score

        def assess_capital_structure(sol_label: str, sol_score: float) -> tuple[str, str, float]:
            net_debt = (financials.get("total_debt") or 0) - (financials.get("cash") or 0)
            if net_debt <= 0:
                return "Net Cash", "Cash reserves exceed total debt.", max(sol_score, 85.0)
            return sol_label, f"Net debt approximately ${net_debt:,.0f}.", sol_score

        def assess_working_capital() -> tuple[str, str, float]:
            ocf = financials.get("operating_cash_flow")
            revenue = financials.get("revenues")
            if ocf is None or revenue in (None, 0):
                return "Mixed", "Operating cash-flow data unavailable.", 60.0
            ocf_margin = ocf / revenue if revenue else 0
            if ocf_margin >= 0.2:
                label, score = "Efficient", 85.0
            elif ocf_margin >= 0.1:
                label, score = "Stable", 70.0
            else:
                label, score = "Tight", 55.0
            commentary = f"OCF margin {ocf_margin:.1%}."
            return label, commentary, score

        def assess_debt_serviceability() -> tuple[str, str, float]:
            interest_coverage = ratios.get("interest_coverage") or company_data.get("interest_coverage")
            total_debt = financials.get("total_debt") or 0
            if total_debt == 0:
                return "Not Applicable", "No debt outstanding.", 90.0
            if not interest_coverage:
                return "Unknown", "Interest coverage unavailable.", 55.0
            if interest_coverage >= 6:
                return "Comfortable", f"Coverage {interest_coverage:.1f}×.", 85.0
            if interest_coverage >= 2:
                return "Adequate", f"Coverage {interest_coverage:.1f}×.", 70.0
            return "Stressed", f"Coverage {interest_coverage:.1f}×.", 45.0

        def assess_flexibility() -> tuple[str, str, float]:
            cash = financials.get("cash") or 0
            total_debt = financials.get("total_debt") or 0
            if total_debt == 0:
                return "High", "Balance sheet unlevered with cash cushion.", 90.0
            liquidity_ratio = (cash + (financials.get("short_term_investments") or 0)) / total_debt
            if liquidity_ratio >= 0.75:
                return "High", f"Liquid assets cover {liquidity_ratio:.0%} of debt.", 80.0
            if liquidity_ratio >= 0.4:
                return "Moderate", f"Liquid assets cover {liquidity_ratio:.0%} of debt.", 65.0
            return "Limited", f"Liquid assets cover {liquidity_ratio:.0%} of debt.", 50.0

        liquidity = assess_liquidity()
        solvency = assess_solvency()
        capital_structure = assess_capital_structure(solvency[0], solvency[2])
        working_capital = assess_working_capital()
        debt_serviceability = assess_debt_serviceability()
        flexibility = assess_flexibility()

        score_components = [
            liquidity[2],
            solvency[2],
            capital_structure[2],
            working_capital[2],
            debt_serviceability[2],
            flexibility[2],
        ]
        overall_health_score = round(sum(score_components) / len(score_components), 1)

        risk_factors: List[Dict[str, str]] = []
        if liquidity[0] == "Weak":
            risk_factors.append({"risk": "Tight liquidity", "commentary": liquidity[1]})
        if solvency[0] == "Leveraged":
            risk_factors.append({"risk": "High leverage", "commentary": solvency[1]})
        if debt_serviceability[0] == "Stressed":
            risk_factors.append({"risk": "Debt service pressure", "commentary": debt_serviceability[1]})

        payload = {
            "liquidity_position": {"assessment": liquidity[0], "commentary": liquidity[1]},
            "solvency": {"assessment": solvency[0], "commentary": solvency[1]},
            "capital_structure_quality": {"assessment": capital_structure[0], "commentary": capital_structure[1]},
            "working_capital_management": {"assessment": working_capital[0], "commentary": working_capital[1]},
            "debt_serviceability": {"assessment": debt_serviceability[0], "commentary": debt_serviceability[1]},
            "financial_flexibility": {"assessment": flexibility[0], "commentary": flexibility[1]},
            "risk_factors": risk_factors,
            "overall_health_score": overall_health_score,
        }

        return self._build_deterministic_response("financial_health", payload)

    async def _analyze_growth(self, company_data: Dict, symbol: str) -> Dict:
        """Deterministic growth analysis leveraging computed trend data."""
        trend = company_data.get("trend_analysis") or {}
        revenue_trend = trend.get("revenue") or {}
        margin_trend = trend.get("margins") or {}

        def _summarize_series(series: List[float]) -> Dict[str, Optional[float]]:
            if not series:
                return {"avg": None, "latest": None, "quantiles": {}}
            window = series[-min(len(series), 6) :]
            summary = {"avg": statistics.mean(window) if window else None, "latest": window[-1] if window else None}
            quantiles = {}
            if len(window) >= 4:
                try:
                    q1, q2, q3 = statistics.quantiles(window, n=4, method="inclusive")
                    quantiles = {"p25": q1, "p50": q2, "p75": q3}
                except statistics.StatisticsError:
                    quantiles = {}
            summary["quantiles"] = quantiles
            return summary

        yoy_summary = _summarize_series(revenue_trend.get("y_over_y_growth") or [])
        qoq_summary = _summarize_series(revenue_trend.get("q_over_q_growth") or [])
        comparisons = {
            "avg_yoy_growth": yoy_summary["avg"],
            "latest_qoq_growth": qoq_summary["latest"],
            "yoy_quantiles": yoy_summary["quantiles"],
            "qoq_quantiles": qoq_summary["quantiles"],
        }

        def classify_growth(value: Optional[float]) -> tuple[str, float]:
            if value is None:
                return "Unknown", 60.0
            if value >= 8.0:
                return "High", 90.0
            if value >= 3.0:
                return "Moderate", 75.0
            if value >= 0.0:
                return "Stable", 65.0
            return "Contracting", 45.0

        yoy_growth = comparisons.get("avg_yoy_growth")
        qoq_growth = comparisons.get("latest_qoq_growth")
        yoy_label, yoy_score = classify_growth(yoy_growth)
        qoq_label, qoq_score = classify_growth(qoq_growth)

        consistency_score = revenue_trend.get("consistency_score")
        if consistency_score is None:
            consistency_label, consistency_pts = "Unknown", 60.0
        elif consistency_score >= 80:
            consistency_label, consistency_pts = "Low Volatility", 85.0
        elif consistency_score >= 60:
            consistency_label, consistency_pts = "Manageable", 70.0
        else:
            consistency_label, consistency_pts = "Choppy", 55.0

        margin_direction = margin_trend.get("net_margin_trend") or "stable"
        margin_map = {
            "expanding": ("Improving", 85.0),
            "stable": ("Stable", 70.0),
            "contracting": ("Weak", 55.0),
        }
        earnings_label, earnings_score = margin_map.get(margin_direction, ("Stable", 70.0))

        market_share_trend = revenue_trend.get("trend", "stable")
        market_map = {
            "accelerating": ("Gaining", 85.0),
            "stable": ("Holding", 70.0),
            "decelerating": ("Losing", 55.0),
        }
        market_label, market_score = market_map.get(market_share_trend, ("Holding", 70.0))

        growth_drivers: List[str] = []
        growth_risks: List[str] = []
        if yoy_growth and yoy_growth >= 5:
            growth_drivers.append("Product demand momentum")
        if margin_direction == "expanding":
            growth_drivers.append("Operational leverage")
        if trend.get("cyclical", {}).get("is_cyclical"):
            growth_risks.append("Seasonality swings")
        if yoy_growth is not None and yoy_growth < 0:
            growth_risks.append("Negative revenue comp")
        if not growth_drivers:
            growth_drivers.append("Core franchise stability")
        if not growth_risks:
            growth_risks.append("Execution risk")

        score_components = [yoy_score, qoq_score, consistency_pts, earnings_score, market_score]
        growth_score = round(sum(score_components) / len(score_components), 1)

        payload = {
            "revenue_growth_sustainability": {
                "assessment": yoy_label,
                "commentary": (
                    f"Average Y/Y growth {yoy_growth:.1f}%" if yoy_growth is not None else "Insufficient data"
                ),
            },
            "earnings_growth_quality": {
                "assessment": earnings_label,
                "commentary": f"Margin trend {margin_direction.upper()}.",
            },
            "growth_consistency_and_volatility": {
                "assessment": consistency_label,
                "commentary": (
                    f"Consistency score {consistency_score:.0f}/100"
                    if consistency_score is not None
                    else "Consistency data unavailable."
                ),
            },
            "market_share_trends": {
                "assessment": market_label,
                "commentary": f"Revenue trend classified as {market_share_trend}.",
            },
            "growth_drivers_and_catalysts": growth_drivers,
            "future_growth_potential": {
                "assessment": "High" if yoy_label in {"High", "Moderate"} else "Balanced",
                "commentary": (
                    "Pipeline supported by demand indicators."
                    if yoy_label in {"High", "Moderate"}
                    else "Requires catalysts to re-accelerate."
                ),
            },
            "growth_risks_and_headwinds": growth_risks,
            "distribution_snapshot": {
                "yoy_percentiles": comparisons.get("yoy_quantiles"),
                "qoq_percentiles": comparisons.get("qoq_quantiles"),
            },
            "growth_score": growth_score,
        }

        return self._build_deterministic_response("growth_analysis", payload)

    async def _analyze_profitability(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """Deterministic profitability assessment using core ratios."""
        trend = company_data.get("trend_analysis") or {}
        margin_trend = trend.get("margins", {})

        def pct(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value*100:.1f}%" if value <= 1 else f"{value:.1f}%"

        gross_margin = ratios.get("gross_margin")
        operating_margin = ratios.get("operating_margin")
        net_margin = ratios.get("net_margin")
        roe = ratios.get("roe")
        roa = ratios.get("roa")
        asset_turnover = ratios.get("asset_turnover")

        def classify_margin(value: Optional[float]) -> tuple[str, float]:
            if value is None:
                return "Unknown", 60.0
            if value >= 0.25:
                return "Wide", 90.0
            if value >= 0.15:
                return "Healthy", 75.0
            if value >= 0.05:
                return "Thin", 60.0
            return "Negative", 45.0

        gross_label, gross_score = classify_margin(gross_margin)
        op_label, op_score = classify_margin(operating_margin)
        net_label, net_score = classify_margin(net_margin)

        gross_history = margin_trend.get("gross_margins") or []
        op_history = margin_trend.get("operating_margins") or []
        net_history = margin_trend.get("net_margins") or []

        def classify_returns(value: Optional[float]) -> tuple[str, float]:
            if value is None:
                return "Unknown", 60.0
            if value >= 0.18:
                return "High", 90.0
            if value >= 0.10:
                return "Solid", 75.0
            if value >= 0.05:
                return "Moderate", 60.0
            return "Low", 45.0

        roe_label, roe_score = classify_returns(roe)
        roa_label, roa_score = classify_returns(roa)

        margin_direction = margin_trend.get("net_margin_trend", "stable")
        direction_comment = {
            "expanding": "Margins trending higher.",
            "contracting": "Margins compressing.",
            "stable": "Margins steady year over year.",
        }.get(margin_direction, "Margin trend unavailable.")

        pricing_power_label = gross_label if gross_label != "Unknown" else op_label
        pricing_comment = f"Gross margin {pct(gross_margin)}; operating margin {pct(operating_margin)}."

        operating_leverage = None
        if gross_margin is not None and operating_margin is not None:
            operating_leverage = gross_margin - operating_margin
        if operating_leverage is None:
            leverage_label, leverage_score = "Unknown", 60.0
            leverage_comment = "Insufficient data to assess operating leverage."
        elif operating_leverage <= 0.05:
            leverage_label, leverage_score = "Low", 60.0
            leverage_comment = "Limited drop from gross to operating margin."
        elif operating_leverage <= 0.15:
            leverage_label, leverage_score = "Balanced", 75.0
            leverage_comment = "Moderate fixed-cost absorption."
        else:
            leverage_label, leverage_score = "High", 85.0
            leverage_comment = "High fixed-cost base amplifies swings."

        cost_structure_spread = None
        if gross_margin is not None and operating_margin is not None:
            cost_structure_spread = gross_margin - operating_margin
        if cost_structure_spread is None:
            cost_label, cost_score = "Unknown", 60.0
            cost_comment = "Unable to derive cost structure spread."
        elif cost_structure_spread <= 0.10:
            cost_label, cost_score = "Lean", 85.0
            cost_comment = "Operating expenses tightly managed."
        elif cost_structure_spread <= 0.20:
            cost_label, cost_score = "Balanced", 70.0
            cost_comment = "Cost structure consistent with peers."
        else:
            cost_label, cost_score = "Heavy", 55.0
            cost_comment = "Operating expenses absorb large share of gross profit."

        profitability_drivers: List[str] = []
        if gross_history:
            try:
                profitability_drivers.append(f"Median gross margin {statistics.median(gross_history):.1f}%")
            except statistics.StatisticsError:
                pass
        if net_history:
            try:
                median_net = statistics.median(net_history)
                if median_net > 15:
                    profitability_drivers.append("Consistent double-digit net margins")
            except statistics.StatisticsError:
                pass
        if gross_label in {"Wide", "Healthy"}:
            profitability_drivers.append("Premium margin profile")
        if roe_label in {"High", "Solid"}:
            profitability_drivers.append("Efficient capital deployment")
        if not profitability_drivers:
            profitability_drivers.append("Scaling opportunities")

        profitability_score = round(
            sum(
                [
                    gross_score,
                    op_score,
                    net_score,
                    roe_score,
                    roa_score,
                    leverage_score,
                    cost_score,
                ]
            )
            / 7,
            1,
        )

        payload = {
            "margin_trends_and_sustainability": {
                "assessment": margin_direction.capitalize(),
                "commentary": direction_comment,
            },
            "return_on_capital_efficiency": {
                "assessment": roe_label,
                "commentary": f"ROE {pct(roe)}; ROA {pct(roa)}.",
            },
            "competitive_advantages_moat": {
                "assessment": gross_label,
                "commentary": f"Gross margin {pct(gross_margin)} suggests {gross_label.lower()} moat.",
            },
            "pricing_power_indicators": {
                "assessment": pricing_power_label,
                "commentary": pricing_comment,
            },
            "cost_structure_analysis": {
                "assessment": cost_label,
                "commentary": cost_comment,
            },
            "operating_leverage": {
                "assessment": leverage_label,
                "commentary": leverage_comment,
            },
            "profitability_drivers": profitability_drivers,
            "profitability_score": profitability_score,
        }

        return self._build_deterministic_response("profitability_analysis", payload)

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

        # === P/E MULTIPLE MODEL ===
        ttm_eps = ratios.get("eps") or ratios.get("eps_basic") or ratios.get("eps_diluted")
        sector_median_pe = (
            company_data.get("sector_metrics", {}).get("median_pe")
            or company_data.get("sector_data", {}).get("median_pe")
            or self._lookup_sector_multiple(company_profile.sector, "pe")
        )
        growth_adjusted_pe = None
        peg_ratio = ratios.get("peg_ratio") or ratios.get("peg")
        if peg_ratio and peg_ratio > 0:
            growth_adjusted_pe = sector_median_pe * (1 + min(peg_ratio, 3)) if sector_median_pe else None

        pe_model = PEMultipleModel(
            company_profile=company_profile,
            ttm_eps=ttm_eps,
            current_price=market_data.get("price")
            or market_data.get("close")
            or market_data.get("current_price")
            or ratios.get("current_price"),
            sector_median_pe=sector_median_pe,
            growth_adjusted_pe=growth_adjusted_pe,
            earnings_quality_score=company_profile.earnings_quality_score,
        )

        pe_result = pe_model.calculate()
        normalized_pe = normalize_model_output(pe_result)
        valuation_results["pe"] = normalized_pe

        # Log P/E result immediately
        log_individual_model_result(self.logger, symbol, "P/E", normalized_pe)

        # === EV/EBITDA MODEL ===
        ttm_ebitda = financials.get("ebitda") or ratios.get("ebitda") or financials.get("operating_income")
        enterprise_value = self._calculate_enterprise_value(market_data, financials)
        sector_ev_ebitda = (
            company_data.get("sector_metrics", {}).get("median_ev_ebitda")
            or company_data.get("sector_data", {}).get("median_ev_ebitda")
            or self._lookup_sector_multiple(company_profile.sector, "ev_ebitda")
        )

        leverage_adjusted_multiple = None
        if sector_ev_ebitda and company_profile.net_debt_to_ebitda is not None:
            leverage_delta = max(company_profile.net_debt_to_ebitda - 2.0, 0.0)
            leverage_adjusted_multiple = sector_ev_ebitda * clamp(1.0 - 0.06 * leverage_delta, 0.6, 1.1)

        ev_ebitda_model = EVEBITDAModel(
            company_profile=company_profile,
            ttm_ebitda=ttm_ebitda,
            enterprise_value=enterprise_value,
            sector_median_ev_ebitda=sector_ev_ebitda,
            leverage_adjusted_multiple=leverage_adjusted_multiple,
            interest_coverage=ratios.get("interest_coverage") or ratios.get("interest_coverage_ratio"),
        )

        ev_ebitda_result = ev_ebitda_model.calculate()
        normalized_ev_ebitda = normalize_model_output(ev_ebitda_result)
        valuation_results["ev_ebitda"] = normalized_ev_ebitda

        # Log EV/EBITDA result immediately
        log_individual_model_result(self.logger, symbol, "EV/EBITDA", normalized_ev_ebitda)

        # === P/S MULTIPLE MODEL ===
        revenue_per_share = None

        # Calculate revenue per share
        if ratios.get("revenue_per_share"):
            revenue_per_share = ratios.get("revenue_per_share")
        elif financials.get("revenues") and company_profile.shares_outstanding:
            try:
                revenue_per_share = float(financials.get("revenues")) / float(company_profile.shares_outstanding)
            except (TypeError, ValueError, ZeroDivisionError) as e:
                revenue_per_share = None
                self.logger.debug(f"{symbol} - Failed to calculate revenue_per_share: {e}")
        else:
            revenue_per_share = None

        # Get sector P/S multiple (priority: sector_metrics > sector_data > lookup)
        sector_ps_from_metrics = company_data.get("sector_metrics", {}).get("median_ps")
        sector_ps_from_data = company_data.get("sector_data", {}).get("median_ps")
        sector_ps_from_lookup = self._lookup_sector_multiple(company_profile.sector, "ps")

        sector_ps = sector_ps_from_metrics or sector_ps_from_data or sector_ps_from_lookup

        valuation_settings = getattr(self.config, "valuation", None)
        liquidity_floor = 5_000_000
        if isinstance(valuation_settings, dict):
            liquidity_floor = valuation_settings.get("liquidity_floor_usd", liquidity_floor)
        elif valuation_settings is not None:
            liquidity_floor = getattr(valuation_settings, "liquidity_floor_usd", liquidity_floor)

        current_price = (
            market_data.get("price")
            or market_data.get("close")
            or market_data.get("current_price")
            or ratios.get("current_price")
        )

        ps_model = PSMultipleModel(
            company_profile=company_profile,
            revenue_per_share=revenue_per_share,
            current_price=current_price,
            sector_median_ps=sector_ps,
            liquidity_floor_usd=liquidity_floor,
        )

        ps_result = ps_model.calculate()
        normalized_ps = normalize_model_output(ps_result)
        valuation_results["ps"] = normalized_ps

        # Log P/S result immediately
        log_individual_model_result(self.logger, symbol, "P/S", normalized_ps)

        # === P/B MULTIPLE MODEL ===
        sector_pb = (
            company_data.get("sector_metrics", {}).get("median_pb")
            or company_data.get("sector_data", {}).get("median_pb")
            or self._lookup_sector_multiple(company_profile.sector, "pb")
        )

        pb_model = PBMultipleModel(
            company_profile=company_profile,
            book_value_per_share=company_profile.book_value_per_share,
            tangible_book_value_per_share=ratios.get("tangible_book_value_per_share"),
            current_price=market_data.get("price")
            or market_data.get("close")
            or market_data.get("current_price")
            or ratios.get("current_price"),
            sector_median_pb=sector_pb,
        )

        pb_result = pb_model.calculate()
        normalized_pb = normalize_model_output(pb_result)
        valuation_results["pb"] = normalized_pb

        # === INSURANCE VALUATION OVERRIDE ===
        # For insurance companies, override generic P/B with sector-specific P/BV valuation
        # This ensures the insurance-specific fair value (based on ROE-adjusted target P/BV)
        # is used instead of the generic book value calculation
        sector_specific = valuation_results.get("sector_specific")
        if sector_specific and "P/BV" in sector_specific.get("method", ""):
            # Convert insurance ValuationResult to orchestrator-compatible format
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            insurance_confidence = confidence_map.get(sector_specific.get("confidence", "medium"), 0.7)

            # Override normalized_pb with insurance-specific values
            normalized_pb = {
                "model": "pb",
                "fair_value_per_share": sector_specific.get("fair_value"),
                "applicable": True,
                "confidence_score": insurance_confidence,
                "method": sector_specific.get("method"),
                "details": sector_specific.get("details", {}),
                "warnings": sector_specific.get("warnings", []),
                # Preserve upside calculation
                "upside_percent": sector_specific.get("upside_percent"),
                "current_price": sector_specific.get("current_price"),
            }
            valuation_results["pb"] = normalized_pb

            self.logger.info(
                f"🏦 {symbol} - INSURANCE OVERRIDE: Using P/BV insurance valuation for P/B model "
                f"(FV=${sector_specific.get('fair_value', 0):.2f}, confidence={sector_specific.get('confidence')})"
            )

        # Log P/B result immediately
        log_individual_model_result(self.logger, symbol, "P/B", normalized_pb)

        # === GORDON GROWTH MODEL (dividend stocks only) ===
        # Determine if stock pays SIGNIFICANT dividends (payout ratio ≥ 20%)
        # Growth stocks may pay token dividends (0.5%), but GGM requires meaningful dividend policy
        common_divs = abs(financials.get("dividends_paid", 0) or 0)
        preferred_divs = abs(financials.get("preferred_stock_dividends", 0) or 0)
        dividends_paid = common_divs + preferred_divs

        # Data quality note: preferred_stock_dividends has only 27% coverage across Russell 1000
        if preferred_divs > 0:
            self.logger.debug(
                f"{symbol} - Preferred stock dividends found: ${preferred_divs:,.0f} " f"(27%% coverage - rare field)"
            )

        net_income = financials.get("net_income", 0) or 0

        # Calculate payout ratio to determine if GGM is appropriate
        payout_ratio = (dividends_paid / net_income * 100) if net_income > 0 else 0
        is_significant_dividend_stock = dividends_paid > 0 and payout_ratio >= 20.0

        if is_significant_dividend_stock:
            # Calculate cost of equity using CAPM for GGM
            cost_of_equity = self._calculate_cost_of_equity(symbol)
            self.logger.info(f"{symbol} - GGM cost_of_equity passed: {cost_of_equity*100:.2f}%")

            # Calculate GGM valuation
            ggm_result = await self._calculate_ggm(symbol, cost_of_equity, quarterly_data, company_profile)
            valuation_results["ggm"] = ggm_result  # Gordon Growth Model

            self.logger.info(
                f"{symbol} - GGM applicable: payout ratio {payout_ratio:.1f}% "
                f"(≥20% threshold for meaningful dividend policy)"
            )

            # Log GGM result immediately
            log_individual_model_result(self.logger, symbol, "GGM", ggm_result)
        else:
            # Not a significant dividend payer
            if dividends_paid > 0 and payout_ratio < 20.0:
                reason = f"Low payout ratio ({payout_ratio:.1f}%) - token dividend, not meaningful dividend policy (need ≥20%)"
            elif dividends_paid == 0:
                reason = "No dividends paid - GGM requires dividend-paying stock"
            else:
                reason = "Negative net income - cannot calculate meaningful payout ratio"

            valuation_results["ggm"] = {
                "applicable": False,
                "reason": reason,
                "fair_value_per_share": 0,
                "payout_ratio": payout_ratio,
            }

            ggm_result = valuation_results["ggm"]

            self.logger.info(f"{symbol} - GGM not applicable: {reason}")

            # Log GGM not applicable
            log_individual_model_result(self.logger, symbol, "GGM", ggm_result)

        # === DAMODARAN 3-STAGE DCF (Milestone 7.1) ===
        # Uses 3-stage growth projection with Monte Carlo sensitivity analysis
        try:
            damodaran_model = DamodaranDCFModel(company_profile)
            damodaran_result = damodaran_model.calculate(
                current_fcf=financials.get("free_cash_flow") or financials.get("fcf"),
                revenue_growth=company_profile.revenue_growth_yoy,
                fcf_margin=ratios.get("fcf_margin") or ratios.get("free_cash_flow_margin"),
                current_revenue=financials.get("revenues")
                or financials.get("revenue")
                or financials.get("total_revenue"),
                shares_outstanding=company_profile.shares_outstanding,
            )
            normalized_damodaran = normalize_model_output(damodaran_result)
            valuation_results["damodaran_dcf"] = normalized_damodaran
            log_individual_model_result(self.logger, symbol, "Damodaran DCF", normalized_damodaran)
        except Exception as e:
            self.logger.warning(f"{symbol} - Damodaran DCF failed: {e}")
            valuation_results["damodaran_dcf"] = {"applicable": False, "reason": str(e), "model": "damodaran_dcf"}

        # === RULE OF 40 VALUATION (Milestone 7.2) ===
        # Applies to growth companies, especially SaaS
        is_saas_company = company_profile.industry and any(
            kw in company_profile.industry.lower() for kw in ["software", "saas", "cloud", "internet"]
        )
        is_growth_company = company_profile.revenue_growth_yoy and company_profile.revenue_growth_yoy > 0.10

        if is_saas_company or is_growth_company:
            try:
                rule_of_40_model = RuleOf40Valuation(company_profile)
                rule_of_40_result = rule_of_40_model.calculate(
                    revenue_growth=company_profile.revenue_growth_yoy,
                    fcf_margin=ratios.get("fcf_margin") or ratios.get("free_cash_flow_margin"),
                    current_revenue=financials.get("revenues")
                    or financials.get("revenue")
                    or financials.get("total_revenue"),
                    current_price=market_data.get("price")
                    or market_data.get("close")
                    or market_data.get("current_price"),
                    shares_outstanding=company_profile.shares_outstanding,
                )
                normalized_rule_of_40 = normalize_model_output(rule_of_40_result)
                valuation_results["rule_of_40"] = normalized_rule_of_40
                log_individual_model_result(self.logger, symbol, "Rule of 40", normalized_rule_of_40)
            except Exception as e:
                self.logger.warning(f"{symbol} - Rule of 40 failed: {e}")
                valuation_results["rule_of_40"] = {"applicable": False, "reason": str(e), "model": "rule_of_40"}
        else:
            valuation_results["rule_of_40"] = {
                "applicable": False,
                "reason": "Not a growth/SaaS company (requires >10% revenue growth or SaaS industry)",
                "model": "rule_of_40",
            }

        # === SAAS VALUATION MODEL (Milestone 7.3) ===
        # Applies specifically to SaaS companies with recurring revenue metrics
        if is_saas_company:
            try:
                saas_model = SaaSValuationModel(company_profile)
                saas_result = saas_model.calculate(
                    revenue_growth=company_profile.revenue_growth_yoy,
                    current_revenue=financials.get("revenues")
                    or financials.get("revenue")
                    or financials.get("total_revenue"),
                    current_price=market_data.get("price")
                    or market_data.get("close")
                    or market_data.get("current_price"),
                    shares_outstanding=company_profile.shares_outstanding,
                    gross_margin=ratios.get("gross_margin") or ratios.get("gross_profit_margin"),
                    # SaaS-specific metrics (may not be available for all companies)
                    nrr=ratios.get("net_revenue_retention") or ratios.get("nrr"),
                    ltv_cac=ratios.get("ltv_cac") or ratios.get("ltv_cac_ratio"),
                    fcf_margin=ratios.get("fcf_margin") or ratios.get("free_cash_flow_margin"),
                )
                normalized_saas = normalize_model_output(saas_result)
                valuation_results["saas"] = normalized_saas
                log_individual_model_result(self.logger, symbol, "SaaS", normalized_saas)
            except Exception as e:
                self.logger.warning(f"{symbol} - SaaS valuation failed: {e}")
                valuation_results["saas"] = {"applicable": False, "reason": str(e), "model": "saas"}
        else:
            valuation_results["saas"] = {
                "applicable": False,
                "reason": "Not a SaaS company (requires software/cloud/internet industry)",
                "model": "saas",
            }

        # === MULTI-MODEL BLENDING ===
        try:
            models_for_blending: List[Dict[str, Any]] = []
            if isinstance(dcf_professional, dict):
                models_for_blending.append(dcf_professional)
            if isinstance(valuation_results.get("ggm"), dict):
                models_for_blending.append(valuation_results["ggm"])
            if isinstance(normalized_pe, dict):
                models_for_blending.append(normalized_pe)
            if isinstance(normalized_ev_ebitda, dict):
                models_for_blending.append(normalized_ev_ebitda)
            if isinstance(normalized_ps, dict):
                models_for_blending.append(normalized_ps)
            if isinstance(normalized_pb, dict):
                models_for_blending.append(normalized_pb)
            # Note: Insurance P/BV valuation is already integrated into normalized_pb above
            # Only add other sector-specific valuations (bank ROE, REIT FFO) as separate models
            sector_spec = valuation_results.get("sector_specific")
            if isinstance(sector_spec, dict) and "P/BV" not in sector_spec.get("method", ""):
                models_for_blending.append(sector_spec)
                self.logger.info(
                    f"✅ [SECTOR_VALUATION] {symbol} - Added sector-specific valuation to blending: {sector_spec.get('method')}"
                )

            # Add new valuation models (Milestone 7)
            if isinstance(valuation_results.get("damodaran_dcf"), dict) and valuation_results["damodaran_dcf"].get(
                "applicable"
            ):
                models_for_blending.append(valuation_results["damodaran_dcf"])
                self.logger.info(f"✅ {symbol} - Added Damodaran DCF to blending")
            if isinstance(valuation_results.get("rule_of_40"), dict) and valuation_results["rule_of_40"].get(
                "applicable"
            ):
                models_for_blending.append(valuation_results["rule_of_40"])
                self.logger.info(f"✅ {symbol} - Added Rule of 40 to blending")
            if isinstance(valuation_results.get("saas"), dict) and valuation_results["saas"].get("applicable"):
                models_for_blending.append(valuation_results["saas"])
                self.logger.info(f"✅ {symbol} - Added SaaS valuation to blending")

            self.logger.debug(f"{symbol} - Models for blending: {[m.get('model') for m in models_for_blending]}")

            allowed_models = self._select_models_for_company(company_profile)
            if allowed_models is not None:
                # CRITICAL: For insurance companies, always include 'pb' regardless of archetype
                # Insurance companies are valued primarily on book value (P/BV)
                is_insurance = company_profile.industry and "insur" in company_profile.industry.lower()
                if is_insurance and "pb" not in allowed_models:
                    allowed_models = list(allowed_models) + ["pb"]
                    self.logger.info(f"🏦 {symbol} - Added 'pb' to allowed_models for insurance company")

                models_for_blending = [model for model in models_for_blending if model.get("model") in allowed_models]
                self.logger.debug(
                    f"{symbol} - Filtered models (allowed={allowed_models}): {[m.get('model') for m in models_for_blending]}"
                )

            # CRITICAL FIX: Ensure market_cap and related fields are in financials dict for dynamic weighting
            # Dynamic weighting service expects these in financials, but they're calculated in ratios
            # Copy them to financials to ensure dynamic weighting has all required data
            if ratios:
                if "market_cap" in ratios and "market_cap" not in financials:
                    financials["market_cap"] = ratios["market_cap"]
                    self.logger.debug(
                        f"{symbol} - Copied market_cap from ratios to financials: ${ratios['market_cap']:,.0f}"
                    )
                if "shares_outstanding" in ratios and "shares_outstanding" not in financials:
                    financials["shares_outstanding"] = ratios["shares_outstanding"]
                if "current_price" in ratios and "current_price" not in financials:
                    financials["current_price"] = ratios["current_price"]

            # CRITICAL FIX: Ensure revenue key exists in financials for P/S model applicability
            # Model applicability checker requires exactly "revenue" key (not "revenues" or "total_revenue")
            # This prevents P/S from being incorrectly filtered out with "Negative/zero revenue: $0"
            if "revenue" not in financials or financials.get("revenue", 0) == 0:
                # Try common revenue key variants
                ttm_metrics = company_data.get("ttm_metrics", {})
                revenue_value = (
                    financials.get("revenues")
                    or financials.get("total_revenue")
                    or ttm_metrics.get("revenues")
                    or ttm_metrics.get("total_revenue")
                    or ttm_metrics.get("revenue")
                    or 0
                )
                if revenue_value and revenue_value > 0:
                    financials["revenue"] = revenue_value
                    self.logger.info(f"{symbol} - Added missing 'revenue' key to financials: ${revenue_value:,.0f}")

            # CRITICAL FIX: Add FCF quarters count for DCF applicability check
            # ModelApplicabilityRules requires this field to determine if DCF is applicable
            fcf_quarters_count = 0
            if hasattr(company_profile, "quarterly_metrics") and company_profile.quarterly_metrics:
                for quarter in company_profile.quarterly_metrics:
                    if isinstance(quarter, dict):
                        cash_flow = quarter.get("cash_flow", {})
                        if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                            fcf_quarters_count += 1
                    else:
                        if hasattr(quarter, "cash_flow"):
                            cash_flow = getattr(quarter, "cash_flow", {})
                            if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                                fcf_quarters_count += 1

            financials["fcf_quarters_count"] = fcf_quarters_count

            # Add TTM free cash flow if available
            if hasattr(company_profile, "free_cash_flow") and company_profile.free_cash_flow:
                financials["free_cash_flow"] = company_profile.free_cash_flow
            elif hasattr(company_profile, "ttm_metrics") and isinstance(company_profile.ttm_metrics, dict):
                financials["free_cash_flow"] = company_profile.ttm_metrics.get("free_cash_flow", 0)

            # CRITICAL FIX: Add dividends_paid for GGM applicability check
            # ModelApplicabilityRules._check_ggm_applicability requires this field
            if hasattr(company_profile, "dividends_paid") and company_profile.dividends_paid:
                financials["dividends_paid"] = company_profile.dividends_paid
            elif "dividends_paid" not in financials or financials.get("dividends_paid", 0) == 0:
                # Fallback to extracting from financials dict itself
                financials["dividends_paid"] = abs(financials.get("dividends_paid", 0) or 0)

            # CRITICAL FIX: Add ebitda for EV/EBITDA applicability check
            if hasattr(company_profile, "ebitda") and company_profile.ebitda:
                financials["ebitda"] = company_profile.ebitda
            elif "ebitda" not in financials or financials.get("ebitda", 0) == 0:
                # Fallback: Try to get from ttm_metrics or calculate from operating_income
                ttm_metrics = company_data.get("ttm_metrics", {})
                ebitda_value = (
                    ttm_metrics.get("ebitda")
                    or ttm_metrics.get("operating_income")
                    or financials.get("operating_income")
                    or 0
                )
                if ebitda_value:
                    financials["ebitda"] = ebitda_value

            # CRITICAL FIX: Add payout_ratio for GGM applicability check
            if hasattr(company_profile, "dividend_payout_ratio") and company_profile.dividend_payout_ratio:
                financials["payout_ratio"] = company_profile.dividend_payout_ratio
            elif ratios and ratios.get("payout_ratio"):
                financials["payout_ratio"] = ratios["payout_ratio"]
            elif ratios and ratios.get("dividend_payout_ratio"):
                financials["payout_ratio"] = ratios["dividend_payout_ratio"]

            # CRITICAL FIX: Add net_income for P/E and GGM applicability
            if hasattr(company_profile, "net_income") and company_profile.net_income:
                financials["net_income"] = company_profile.net_income
            elif "net_income" not in financials or financials.get("net_income", 0) == 0:
                ttm_metrics = company_data.get("ttm_metrics", {})
                net_income_value = ttm_metrics.get("net_income") or 0
                if net_income_value:
                    financials["net_income"] = net_income_value

            # CRITICAL FIX: Add stockholders_equity for P/B applicability
            if "stockholders_equity" not in financials and "book_value" not in financials:
                book_value = (
                    financials.get("stockholders_equity")
                    or financials.get("total_stockholders_equity")
                    or company_data.get("ttm_metrics", {}).get("stockholders_equity")
                    or 0
                )
                if book_value:
                    financials["stockholders_equity"] = book_value

            # CRITICAL FIX: If fcf_quarters_count is 0 but we have FCF value, infer quarters
            # If TTM FCF exists, it implies we have at least 4 quarters of data
            if fcf_quarters_count == 0 and financials.get("free_cash_flow", 0) > 0:
                fcf_quarters_count = 4  # TTM implies 4 quarters
                financials["fcf_quarters_count"] = fcf_quarters_count
                self.logger.info(f"{symbol} - Inferred fcf_quarters_count=4 from TTM FCF value")

            self.logger.info(
                f"{symbol} - Applicability fields added: fcf_quarters={fcf_quarters_count}, "
                f"fcf=${financials.get('free_cash_flow', 0)/1e9:.2f}B, "
                f"ebitda=${financials.get('ebitda', 0)/1e9:.2f}B, "
                f"dividends_paid=${financials.get('dividends_paid', 0)/1e9:.2f}B, "
                f"payout_ratio={financials.get('payout_ratio', 0):.1f}%, "
                f"net_income=${financials.get('net_income', 0)/1e9:.2f}B, "
                f"book_value=${financials.get('stockholders_equity', financials.get('book_value', 0))/1e9:.2f}B"
            )

            # Get fallback weights and tier classification (pass financials and ratios directly)
            fallback_weights_result = self._resolve_fallback_weights(
                company_profile, models_for_blending, financials, ratios
            )
            if isinstance(fallback_weights_result, tuple):
                fallback_weights, tier_classification = fallback_weights_result
            else:
                # Fallback case: old return type (just weights dict)
                fallback_weights = fallback_weights_result
                tier_classification = None

            multi_model_summary = self.multi_model_orchestrator.combine(
                company_profile,
                models_for_blending,
                fallback_weights=fallback_weights,
                tier_classification=tier_classification,
            )
            valuation_results["multi_model"] = multi_model_summary

            # Update individual model weights from orchestrator result
            try:
                weight_lookup = {
                    model.get("model"): model.get("weight")
                    for model in multi_model_summary.get("models", [])
                    if isinstance(model, dict)
                }
                if isinstance(dcf_professional, dict) and "dcf" in weight_lookup:
                    dcf_professional["weight"] = weight_lookup["dcf"]
                ggm_entry = valuation_results.get("ggm")
                if isinstance(ggm_entry, dict) and "ggm" in weight_lookup:
                    ggm_entry["weight"] = weight_lookup["ggm"]
                if isinstance(normalized_pe, dict) and "pe" in weight_lookup:
                    normalized_pe["weight"] = weight_lookup["pe"]
                if isinstance(normalized_ev_ebitda, dict) and "ev_ebitda" in weight_lookup:
                    normalized_ev_ebitda["weight"] = weight_lookup["ev_ebitda"]
                if isinstance(normalized_ps, dict) and "ps" in weight_lookup:
                    normalized_ps["weight"] = weight_lookup["ps"]
                if isinstance(normalized_pb, dict) and "pb" in weight_lookup:
                    normalized_pb["weight"] = weight_lookup["pb"]
            except Exception as exc2:  # pragma: no cover - defensive
                self.logger.warning(f"{symbol} - Weight lookup failed: {exc2}")
                pass
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            self.logger.error(f"{symbol} - Multi-model blending failed: {exc}")
            self.logger.debug(f"{symbol} - Traceback: {traceback.format_exc()}")

        # Synthesize fair value estimate
        multi_model_summary = valuation_results.get("multi_model", {})
        blended_fair_value = multi_model_summary.get("blended_fair_value")
        overall_confidence = multi_model_summary.get("overall_confidence")
        model_agreement_score = multi_model_summary.get("model_agreement_score")
        divergence_flag = multi_model_summary.get("divergence_flag")
        applicable_models = multi_model_summary.get("applicable_models")
        notes = multi_model_summary.get("notes", [])

        log_valuation_snapshot(self.logger, symbol, valuation_results)

        # Format and log comprehensive valuation summary table
        try:
            # Get current price from company_data
            current_price = company_data.get("current_price", 0)

            # Collect all model data for table
            all_models_data = []

            # DCF
            if isinstance(dcf_professional, dict):
                all_models_data.append(
                    {
                        "name": "DCF",
                        "fair_value": dcf_professional.get("fair_value_per_share", 0),
                        "confidence": dcf_professional.get("confidence_score", 0) * 100,  # Convert to percentage
                        "weight": dcf_professional.get("weight", 0),
                        "applicable": dcf_professional.get("applicable", True),
                    }
                )

            # GGM
            ggm_entry = valuation_results.get("ggm", {})
            if isinstance(ggm_entry, dict):
                all_models_data.append(
                    {
                        "name": "GGM",
                        "fair_value": ggm_entry.get("fair_value_per_share", 0),
                        "confidence": ggm_entry.get("confidence_score", 0) * 100,
                        "weight": ggm_entry.get("weight", 0),
                        "applicable": ggm_entry.get("applicable", False),
                    }
                )

            # P/E
            if isinstance(normalized_pe, dict):
                all_models_data.append(
                    {
                        "name": "P/E",
                        "fair_value": normalized_pe.get("fair_value_per_share", 0),
                        "confidence": normalized_pe.get("confidence_score", 0) * 100,
                        "weight": normalized_pe.get("weight", 0),
                        "applicable": normalized_pe.get("applicable", True),
                    }
                )

            # EV/EBITDA
            if isinstance(normalized_ev_ebitda, dict):
                all_models_data.append(
                    {
                        "name": "EV/EBITDA",
                        "fair_value": normalized_ev_ebitda.get("fair_value_per_share", 0),
                        "confidence": normalized_ev_ebitda.get("confidence_score", 0) * 100,
                        "weight": normalized_ev_ebitda.get("weight", 0),
                        "applicable": normalized_ev_ebitda.get("applicable", True),
                    }
                )

            # P/S
            if isinstance(normalized_ps, dict):
                all_models_data.append(
                    {
                        "name": "P/S",
                        "fair_value": normalized_ps.get("fair_value_per_share", 0),
                        "confidence": normalized_ps.get("confidence_score", 0) * 100,
                        "weight": normalized_ps.get("weight", 0),
                        "applicable": normalized_ps.get("applicable", True),
                    }
                )

            # P/B
            if isinstance(normalized_pb, dict):
                all_models_data.append(
                    {
                        "name": "P/B",
                        "fair_value": normalized_pb.get("fair_value_per_share", 0),
                        "confidence": normalized_pb.get("confidence_score", 0) * 100,
                        "weight": normalized_pb.get("weight", 0),
                        "applicable": normalized_pb.get("applicable", True),
                    }
                )

            # Get tier classification
            tier_display = tier_classification if tier_classification else "N/A"

            # Format and log the comprehensive valuation summary table
            valuation_table = ValuationTableFormatter.format_valuation_summary_table(
                symbol=symbol,
                all_models=all_models_data,
                dynamic_weights={m["name"].lower(): m["weight"] for m in all_models_data},
                blended_fair_value=blended_fair_value if blended_fair_value else 0,
                current_price=current_price,
                tier=tier_display,
                notes=multi_model_summary.get("notes"),
            )
            self.logger.info(valuation_table)

        except Exception as e:
            self.logger.warning(f"{symbol} - Failed to format valuation summary table: {e}")

        # Log blended fair value explicitly for visibility
        if blended_fair_value and blended_fair_value > 0:
            # Handle None values for agreement score and confidence
            agreement_str = f"{model_agreement_score:.2f}" if model_agreement_score is not None else "N/A"
            confidence_str = f"{overall_confidence:.1%}" if overall_confidence is not None else "N/A"
            log_msg = (
                f"✅ {symbol} - Multi-Model Blended Fair Value: ${blended_fair_value:.2f} | "
                f"Confidence: {confidence_str} | "
                f"Agreement: {agreement_str} | "
                f"Applicable Models: {applicable_models}"
            )
            self.logger.info(log_msg)

            if divergence_flag and model_agreement_score is not None:
                self.logger.warning(
                    f"⚠️  {symbol} - Model divergence detected! "
                    f"Agreement score {model_agreement_score:.2f} indicates significant spread between model outputs."
                )
        else:
            self.logger.warning(
                f"⚠️  {symbol} - No blended fair value calculated (applicable models: {applicable_models})"
            )

        # Helper functions now imported from safe_formatters module
        def _fmt_float(value: Any, decimals: int = 1) -> str:
            """Safe float formatting with specified decimal places."""
            return format_number(value, decimals=decimals, thousands_separator=False)

        models_detail_lines: List[str] = []
        for model in multi_model_summary.get("models", []):
            if not isinstance(model, dict):
                continue
            model_name = model.get("methodology") or model.get("model")
            if model.get("applicable"):
                line = (
                    f"- {model_name}: Fair Value {_fmt_currency(model.get('fair_value_per_share'))}, "
                    f"Weight {round(model.get('weight', 0.0), 3):.3f}, "
                    f"Confidence {round(model.get('confidence_score', 0.0), 3):.3f}"
                )
                assumptions = model.get("assumptions") or {}
                metadata = model.get("metadata") or {}
                extra_bits: List[str] = []
                if model.get("model") == "dcf":
                    if "wacc" in assumptions:
                        extra_bits.append(f"WACC {_fmt_pct(assumptions.get('wacc'))}")
                    if metadata.get("rule_of_40"):
                        r40 = metadata["rule_of_40"]
                        score = round_for_prompt(r40.get("score", 0), 1)
                        if score is not None:
                            extra_bits.append(f"Rule of 40 {score:.1f} ({r40.get('classification', '').upper()})")
                if model.get("model") == "ggm":
                    if "growth_rate" in assumptions:
                        extra_bits.append(f"Growth {_fmt_pct(assumptions.get('growth_rate'))}")
                    if "discount_rate" in assumptions:
                        extra_bits.append(f"Cost of equity {_fmt_pct(assumptions.get('discount_rate'))}")
                if model.get("model") == "pe":
                    if assumptions.get("target_pe"):
                        extra_bits.append(f"Target P/E {_fmt_float(assumptions.get('target_pe'), 2)}")
                    if assumptions.get("sector_median_pe"):
                        extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_pe'), 2)}")
                if model.get("model") == "ev_ebitda":
                    if assumptions.get("target_ev_ebitda"):
                        extra_bits.append(f"Target EV/EBITDA {_fmt_float(assumptions.get('target_ev_ebitda'), 2)}")
                    if assumptions.get("sector_median_ev_ebitda"):
                        extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_ev_ebitda'), 2)}")
                if model.get("model") == "ps":
                    if assumptions.get("target_ps"):
                        extra_bits.append(f"Target P/S {_fmt_float(assumptions.get('target_ps'), 2)}")
                    if assumptions.get("sector_median_ps"):
                        extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_ps'), 2)}")
                if model.get("model") == "pb":
                    if assumptions.get("target_pb"):
                        extra_bits.append(f"Target P/B {_fmt_float(assumptions.get('target_pb'), 2)}")
                    if assumptions.get("sector_median_pb"):
                        extra_bits.append(f"Sector Median {_fmt_float(assumptions.get('sector_median_pb'), 2)}")
                if extra_bits:
                    line += " | " + ", ".join(extra_bits)
            else:
                line = f"- {model_name}: Not applicable ({model.get('reason', 'no reason provided')})"
            models_detail_lines.append(line)

        if not models_detail_lines:
            models_detail_lines.append("- No valuation models produced outputs.")

        notes_section = "\n".join(f"- {note}" for note in notes) if notes else "- None"

        archetype_labels = ", ".join(company_profile.archetype_labels()) or "Unclassified"

        models_detail_section = "\n".join(models_detail_lines)

        prompt = f"""
        Synthesize a fair value estimate using the multi-model valuation summary below. Anchor your assessment on the
        blended output and explain any material disagreements across models.

        DATA QUALITY ASSESSMENT:
        - Overall Quality: {data_quality.get('quality_grade', 'Unknown')} ({_safe_fmt_pct(data_quality.get('data_quality_score', 0))})
        - {data_quality.get('assessment', 'Data quality information not available')}
        - Core Metrics: {data_quality.get('core_metrics_populated', 'N/A')} populated
        - Market Data: {data_quality.get('market_metrics_populated', 'N/A')} populated
        - Consistency Issues: {', '.join(data_quality.get('consistency_issues', [])) or 'None detected'}
        {trend_context}

        COMPANY PROFILE SNAPSHOT:
        - Sector: {company_profile.sector}
        - Industry: {company_profile.industry or 'N/A'}
        - Archetypes: {archetype_labels}
        - Data Flags: {', '.join(company_profile_payload.get('data_quality_flags', [])) or 'None'}

        MARKET CONTEXT:
        - Current Price: {_fmt_currency(market_data.get('price'))}
        - Market Cap: ${_fmt_int_comma(market_data.get('market_cap', 0))}
        - Dividend Payout Ratio: {_safe_fmt_pct(payout_ratio)}

        === MULTI-MODEL VALUATION SUMMARY ===
        - Blended Fair Value: {_fmt_currency(blended_fair_value)}
        - Overall Confidence: {_fmt_float(overall_confidence or 0, 3)}
        - Model Agreement Score: {_fmt_float(model_agreement_score or 0, 3)} (lower implies divergence)
        - Divergence Flag: {divergence_flag}
        - Applicable Models: {applicable_models}
        - Notes:\n{notes_section}

        === INDIVIDUAL MODEL DETAIL ===
        {models_detail_section}

        VALUATION SYNTHESIS INSTRUCTIONS:
        1. Produce a blended fair value estimate (state the number) and the implied upside/downside vs. current price.
        2. Explain how each model influences the final number, especially when weights differ.
        3. Comment on confidence and whether divergence or data-quality flags warrant caution.
        4. Highlight key drivers/assumptions (growth, margins, discount rates) that matter most.
        5. Outline valuation risks or scenarios that could shift the blend materially.
        6. Recommend a valuation stance (Undervalued / Fairly Valued / Overvalued) and suggest a margin of safety target.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object that captures these points and follows the schema below (values are illustrative):
        {{
          "fair_value_estimate": 150.00,
          "implied_upside_downside": 0.15,
          "model_influence_explanation": "The DCF model has the highest weight (50%) and is the primary driver of the fair value estimate. The P/E and P/S models are used as secondary inputs.",
          "confidence_and_caution": "Confidence is high due to good data quality and model agreement. No divergence flags were raised.",
          "key_drivers_and_assumptions": "The key drivers are the assumed 5% terminal growth rate and the 10% WACC.",
          "valuation_risks": "A significant increase in interest rates or a slowdown in revenue growth could negatively impact the valuation.",
          "valuation_stance": "Undervalued",
          "margin_of_safety_target": 0.20
        }}
        """

        # Check if deterministic valuation synthesis is enabled (saves tokens, faster)
        if self.use_deterministic and self.deterministic_valuation_synthesis:
            self.logger.debug(f"{symbol} - Using deterministic valuation synthesis (LLM bypass)")

            # Use deterministic synthesis instead of LLM
            response_data = synthesize_valuation(
                symbol=symbol,
                current_price=market_data.get("current_price", market_data.get("price", 0)),
                valuation_results=valuation_results,
                multi_model_summary=multi_model_summary,
                data_quality=data_quality,
                company_profile=company_profile_payload,
                notes=notes,
            )

            # Add valuation methods and current price
            response_data["valuation_methods"] = valuation_results
            response_data["current_price"] = market_data.get("current_price", market_data.get("price", 0))
            response_data["company_profile"] = company_profile_payload

            return self._build_deterministic_response("valuation_synthesis", response_data)

        # === LLM Path (fallback when deterministic is disabled) ===
        # Save prompt to cache for auditing
        prompt_name = "_perform_valuation_synthesis_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        response = await self.ollama.generate(
            model=self.models["valuation"],
            prompt=prompt,
            system="Synthesize valuation analysis and provide fair value estimate.",
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
            llm_type="fundamental_valuation",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
        )

        # Extract response data and add valuation details
        if isinstance(response, dict) and "response" in response:
            response_data = response["response"]
        else:
            response_data = response

        # Parse response if it's a string
        if isinstance(response_data, str):

            try:
                response_data = json.loads(response_data.strip())
            except:
                response_data = {}

        # Add valuation methods and current price
        if isinstance(response_data, dict):
            response_data["valuation_methods"] = valuation_results
            response_data["current_price"] = market_data.get("current_price", market_data.get("price", 0))
            response_data["company_profile"] = company_profile_payload

        return self._wrap_llm_response(
            response=response_data,
            model=self.models["valuation"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
            period=company_data.get("fiscal_period"),  # Period-based caching
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
    ) -> Optional[Dict[str, float]]:
        """
        Determine dynamic model weights using tier-based classification.

        UPDATED: Uses DynamicModelWeightingService for config-driven dynamic weighting.

        Flow:
        1. Use provided financials and ratios dicts directly (already computed)
        2. Call dynamic_weighting_service.determine_weights()
        3. Returns weights as percentages (e.g., {"dcf": 50, "pe": 30, ...})
        4. MultiModelValuationOrchestrator.combine() expects percentages (not decimals)

        Args:
            company_profile: CompanyProfile with metrics
            models_for_blending: List of model results (for context)
            financials: Financial metrics dict (net_income, revenue, etc.)
            ratios: Financial ratios dict (payout_ratio, rule_of_40_score, etc.)

        Returns:
            Tuple of (weights_dict, tier_classification) or just weights_dict for backward compat
        """
        try:
            # Extract symbol
            symbol = company_profile.symbol if hasattr(company_profile, "symbol") else "UNKNOWN"

            # Use provided financials and ratios if available, otherwise reconstruct from company_profile
            if financials is None:
                # Backward compatibility: reconstruct from company_profile
                self.logger.debug(f"{symbol} - No financials provided, reconstructing from company_profile")
                financials = {
                    "net_income": getattr(company_profile, "net_income", 0),
                    "revenue": getattr(company_profile, "revenue", 0),
                    "dividends_paid": getattr(company_profile, "dividends_paid", 0),
                    "ebitda": getattr(company_profile, "ebitda", 0),
                    "book_value": getattr(company_profile, "book_value", 0),
                    "market_cap": 0,  # Will be extracted from model results below
                    "current_price": 0,  # Will be extracted from model results below
                }

            # CRITICAL FIX: Add fields required for model applicability checks
            # These fields were missing, causing DCF, EV/EBITDA, and P/B to be incorrectly filtered out

            # Count quarters with FCF data for DCF applicability
            fcf_quarters_count = 0
            if hasattr(company_profile, "quarterly_metrics") and company_profile.quarterly_metrics:
                for quarter in company_profile.quarterly_metrics:
                    # Check if this quarter has FCF data
                    if isinstance(quarter, dict):
                        cash_flow = quarter.get("cash_flow", {})
                        if isinstance(cash_flow, dict):
                            fcf = cash_flow.get("free_cash_flow")
                            if fcf is not None:
                                fcf_quarters_count += 1
                    else:
                        # Handle object access pattern
                        if hasattr(quarter, "cash_flow"):
                            cash_flow = getattr(quarter, "cash_flow", {})
                            if isinstance(cash_flow, dict):
                                fcf = cash_flow.get("free_cash_flow")
                                if fcf is not None:
                                    fcf_quarters_count += 1

            financials["fcf_quarters_count"] = fcf_quarters_count

            # Add TTM free cash flow for DCF calculation
            if hasattr(company_profile, "free_cash_flow"):
                financials["free_cash_flow"] = getattr(company_profile, "free_cash_flow", 0)
            elif hasattr(company_profile, "ttm_metrics") and isinstance(company_profile.ttm_metrics, dict):
                financials["free_cash_flow"] = company_profile.ttm_metrics.get("free_cash_flow", 0)
            else:
                financials["free_cash_flow"] = 0

            # CRITICAL FIX: Add dividends_paid for GGM applicability
            if hasattr(company_profile, "dividends_paid") and company_profile.dividends_paid:
                financials["dividends_paid"] = company_profile.dividends_paid

            # CRITICAL FIX: Add ebitda for EV/EBITDA applicability
            if hasattr(company_profile, "ebitda") and company_profile.ebitda:
                financials["ebitda"] = company_profile.ebitda

            # CRITICAL FIX: Infer fcf_quarters_count from TTM FCF if not detected from quarterly data
            if fcf_quarters_count == 0 and financials.get("free_cash_flow", 0) > 0:
                fcf_quarters_count = 4  # TTM implies 4 quarters
                financials["fcf_quarters_count"] = fcf_quarters_count
                self.logger.info(f"{symbol} - Inferred fcf_quarters_count=4 from TTM FCF value")

            self.logger.info(
                f"{symbol} - Applicability fields: fcf_quarters={fcf_quarters_count}, "
                f"fcf=${financials.get('free_cash_flow', 0)/1e9:.2f}B, "
                f"ebitda=${financials.get('ebitda', 0)/1e9:.2f}B, "
                f"dividends_paid=${financials.get('dividends_paid', 0)/1e9:.2f}B, "
                f"book_value=${financials.get('book_value', 0)/1e9:.2f}B"
            )

            if ratios is None:
                # Backward compatibility: reconstruct from company_profile
                self.logger.debug(f"{symbol} - No ratios provided, reconstructing from company_profile")
                ratios = {
                    "payout_ratio": getattr(company_profile, "dividend_payout_ratio", 0),
                    "rule_of_40_score": getattr(company_profile, "rule_of_40_score", 0),
                    "revenue_growth_pct": getattr(company_profile, "revenue_growth_yoy", 0),
                    "fcf_margin_pct": (
                        getattr(company_profile, "fcf_margin", 0) * 100
                        if getattr(company_profile, "fcf_margin", None)
                        else 0
                    ),
                    "ttm_eps": 0,  # Will be extracted from model results below
                }

            # Extract enhanced weighting data from model results
            # This data is in the model assumptions/metadata, not company_profile
            self.logger.debug(
                f"{symbol} - DEBUG: models_for_blending type: {type(models_for_blending)}, length: {len(models_for_blending) if models_for_blending else 0}"
            )

            for idx, model_result in enumerate(models_for_blending):
                if model_result is None:
                    continue

                # Debug logging to understand structure
                self.logger.debug(f"{symbol} - DEBUG: model_result[{idx}] type: {type(model_result)}")
                if isinstance(model_result, dict):
                    self.logger.debug(f"{symbol} - DEBUG: model_result[{idx}] keys: {list(model_result.keys())}")
                    if "model_name" in model_result:
                        self.logger.debug(
                            f"{symbol} - DEBUG: model_result[{idx}] model_name: {model_result.get('model_name')}"
                        )
                else:
                    self.logger.debug(
                        f"{symbol} - DEBUG: model_result[{idx}] attributes: {[a for a in dir(model_result) if not a.startswith('_')][:10]}"
                    )
                    if hasattr(model_result, "model_name"):
                        self.logger.debug(
                            f"{symbol} - DEBUG: model_result[{idx}] model_name: {getattr(model_result, 'model_name', None)}"
                        )

                # Handle both dict and object access patterns
                # NOTE: model results use 'model' key, not 'model_name'
                model_name = (
                    model_result.get("model")
                    if isinstance(model_result, dict)
                    else getattr(model_result, "model", None)
                )

                # Extract from P/E model (has current_price and ttm_eps)
                if model_name == "pe":
                    if isinstance(model_result, dict):
                        assumptions = model_result.get("assumptions", {})
                        metadata = model_result.get("metadata", {})
                    else:
                        assumptions = getattr(model_result, "assumptions", {})
                        metadata = getattr(model_result, "metadata", {})

                    current_price = metadata.get("current_price")
                    if current_price is not None:
                        financials["current_price"] = current_price

                    ttm_eps = assumptions.get("ttm_eps")
                    if ttm_eps is not None:
                        ratios["ttm_eps"] = ttm_eps

                # Extract from any model that has market_cap
                if isinstance(model_result, dict):
                    assumptions = model_result.get("assumptions", {})
                    metadata = model_result.get("metadata", {})
                else:
                    assumptions = (
                        getattr(model_result, "assumptions", {}) if hasattr(model_result, "assumptions") else {}
                    )
                    metadata = getattr(model_result, "metadata", {}) if hasattr(model_result, "metadata") else {}

                market_cap_from_assumptions = assumptions.get("market_cap", 0) if isinstance(assumptions, dict) else 0
                if market_cap_from_assumptions > 0:
                    financials["market_cap"] = market_cap_from_assumptions

                market_cap_from_metadata = metadata.get("market_cap", 0) if isinstance(metadata, dict) else 0
                if market_cap_from_metadata > 0:
                    financials["market_cap"] = market_cap_from_metadata

            # Calculate market_cap from current_price and shares if not found
            if financials["market_cap"] == 0 and financials["current_price"] > 0:
                # Try to get shares_outstanding from model results
                for model_result in models_for_blending:
                    if model_result is None:
                        continue

                    if isinstance(model_result, dict):
                        assumptions = model_result.get("assumptions", {})
                    else:
                        assumptions = getattr(model_result, "assumptions", {})

                    shares = assumptions.get("shares_outstanding", 0) if isinstance(assumptions, dict) else 0
                    if shares > 0:
                        financials["market_cap"] = financials["current_price"] * shares
                        break

            # Extract data quality (if available)
            data_quality = getattr(company_profile, "data_quality", None)

            # CRITICAL FIX: Add FCF quarters count for DCF applicability check
            # Count quarters with FCF data - required for ModelApplicabilityRules
            fcf_quarters_count = 0
            if hasattr(company_profile, "quarterly_metrics") and company_profile.quarterly_metrics:
                for quarter in company_profile.quarterly_metrics:
                    # Check if this quarter has FCF data
                    if isinstance(quarter, dict):
                        cash_flow = quarter.get("cash_flow", {})
                        if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                            fcf_quarters_count += 1
                    else:
                        # Handle object access pattern
                        if hasattr(quarter, "cash_flow"):
                            cash_flow = getattr(quarter, "cash_flow", {})
                            if isinstance(cash_flow, dict) and cash_flow.get("free_cash_flow") is not None:
                                fcf_quarters_count += 1

            # Add to financials dict for applicability check
            financials["fcf_quarters_count"] = fcf_quarters_count

            # Add TTM free cash flow if available
            if hasattr(company_profile, "free_cash_flow") and company_profile.free_cash_flow:
                financials["free_cash_flow"] = company_profile.free_cash_flow
            elif hasattr(company_profile, "ttm_metrics") and isinstance(company_profile.ttm_metrics, dict):
                financials["free_cash_flow"] = company_profile.ttm_metrics.get("free_cash_flow", 0)

            # CRITICAL FIX: Add dividends_paid for GGM applicability
            if hasattr(company_profile, "dividends_paid") and company_profile.dividends_paid:
                financials["dividends_paid"] = company_profile.dividends_paid

            # CRITICAL FIX: Add ebitda for EV/EBITDA applicability
            if hasattr(company_profile, "ebitda") and company_profile.ebitda:
                financials["ebitda"] = company_profile.ebitda

            # CRITICAL FIX: Infer fcf_quarters_count from TTM FCF if not detected from quarterly data
            if fcf_quarters_count == 0 and financials.get("free_cash_flow", 0) > 0:
                fcf_quarters_count = 4  # TTM implies 4 quarters
                financials["fcf_quarters_count"] = fcf_quarters_count
                self.logger.info(f"{symbol} - Inferred fcf_quarters_count=4 from TTM FCF value")

            self.logger.info(
                f"{symbol} - Applicability fields added: fcf_quarters={fcf_quarters_count}, "
                f"fcf=${financials.get('free_cash_flow', 0)/1e9:.2f}B, "
                f"ebitda=${financials.get('ebitda', 0)/1e9:.2f}B, "
                f"dividends_paid=${financials.get('dividends_paid', 0)/1e9:.2f}B, "
                f"book_value=${financials.get('book_value', 0)/1e9:.2f}B"
            )

            # Call dynamic weighting service (returns 3-tuple: weights, tier_classification, audit_trail)
            # TODO: Build MarketContext from technical/market context agent results
            # For now, pass market_context=None (weights use tier-based defaults only)
            weights, tier_classification, audit_trail = self.dynamic_weighting_service.determine_weights(
                symbol=symbol,
                financials=financials,
                ratios=ratios,
                data_quality=data_quality,
                market_context=None,  # Future: extract from agent results
            )

            # Log audit trail summary if available
            if audit_trail:
                audit_trail.log_summary()

            self.logger.info(
                f"{symbol} - Dynamic weights determined (tier={tier_classification}): "
                f"{', '.join([f'{model.upper()}={weight}%' for model, weight in weights.items() if weight > 0])}"
            )

            return weights, tier_classification

        except Exception as e:
            self.logger.warning(f"Dynamic weighting failed: {e}. Falling back to static weights.")

            # Fallback to old static logic if dynamic weighting fails
            valuation_settings = getattr(self.config, "valuation", None)
            if isinstance(valuation_settings, dict):
                fallback_cfg = valuation_settings.get("model_fallback", {})
            else:
                fallback_cfg = getattr(valuation_settings, "model_fallback", {}) if valuation_settings else {}

            if not isinstance(fallback_cfg, dict) or not fallback_cfg:
                return None, "fallback_error"  # Return tuple with error tier

            def _normalize_key(key: Optional[str]) -> Optional[str]:
                return key.lower() if key else None

            primary_key = _normalize_key(
                company_profile.primary_archetype.name if company_profile.primary_archetype else None
            )
            fallback_node = None
            if primary_key and primary_key in fallback_cfg:
                fallback_node = fallback_cfg[primary_key]
            elif primary_key and primary_key.capitalize() in fallback_cfg:
                fallback_node = fallback_cfg[primary_key.capitalize()]
            else:
                fallback_node = fallback_cfg.get("default")

            fallback_tier = "static_fallback"
            if not fallback_node:
                return None, "no_fallback_node"

            weights = (
                fallback_node.get("weights")
                if isinstance(fallback_node, dict)
                else getattr(fallback_node, "weights", None)
            )
            if not isinstance(weights, dict):
                return None, "invalid_fallback_weights"

            available_models = {model.get("model") for model in models_for_blending if model.get("model")}
            resolved = {
                model_key: float(weight)
                for model_key, weight in weights.items()
                if model_key in available_models and weight is not None
            }

            # Convert decimals to percentages if needed
            if resolved and all(v <= 1.0 for v in resolved.values()):
                resolved = {k: v * 100 for k, v in resolved.items()}

            return (resolved, fallback_tier) if resolved else (None, "no_resolved_weights")

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
        # This would extract multi-year historical data
        # Placeholder for demonstration
        return {
            "revenue_trend": [100, 110, 121, 133],  # Millions
            "earnings_trend": [10, 12, 15, 18],
            "years": [2021, 2022, 2023, 2024],
        }

    def _summarize_company_data(self, company_data: Dict) -> Dict:
        """Create summary of company data for report"""
        financials = company_data.get("financials") or {}
        market_data = company_data.get("market_data", {})

        # Try multiple locations for market_cap (stored at top level after calculation)
        market_cap = (
            company_data.get("market_cap", 0)
            or market_data.get("market_cap", 0)
            or market_data.get("market_cap", 0)
            or 0
        )

        return {
            "symbol": company_data["symbol"],
            "market_cap": market_cap,
            "price": market_data.get("price", 0) or company_data.get("current_price", 0),
            "revenue": financials.get("revenues") or 0,
            "net_income": financials.get("net_income") or 0,
            "total_assets": financials.get("total_assets") or 0,
            "total_equity": financials.get("stockholders_equity") or 0,
        }

    def _extract_latest_financials(self, quarterly_data: List) -> Dict:
        """Extract latest financial statement from quarterly data (supports both Dict and QuarterlyData objects)"""
        from data.models import QuarterlyData

        if not quarterly_data or len(quarterly_data) == 0:
            return {}

        # Get the most recent quarter
        latest_quarter = quarterly_data[0] if isinstance(quarterly_data, list) else quarterly_data

        # Handle QuarterlyData objects (dataclass with financial_data attribute)
        if isinstance(latest_quarter, QuarterlyData):
            financial_data = latest_quarter.financial_data

            # Helper to safely get values from nested dictionaries
            def safe_get(statement_dict, key, default=0):
                if statement_dict and isinstance(statement_dict, dict):
                    return statement_dict.get(key, default)
                return default

            # Extract from income_statement, balance_sheet, cash_flow_statement
            income_stmt = financial_data.income_statement if hasattr(financial_data, "income_statement") else {}
            balance_sheet = financial_data.balance_sheet if hasattr(financial_data, "balance_sheet") else {}
            cash_flow = financial_data.cash_flow_statement if hasattr(financial_data, "cash_flow_statement") else {}
            quarterly = financial_data.quarterly_data if hasattr(financial_data, "quarterly_data") else {}

            # Extract depreciation_amortization from cash_flow for EBITDA calculation
            operating_income = safe_get(income_stmt, "operating_income") or safe_get(quarterly, "operating_income")
            depreciation_amortization = safe_get(cash_flow, "depreciation_amortization") or safe_get(
                quarterly, "depreciation_amortization"
            )

            # Calculate EBITDA on-the-fly: Operating Income + D&A
            ebitda = 0
            if operating_income and depreciation_amortization:
                ebitda = operating_income + depreciation_amortization
            elif operating_income:
                # If D&A not available, EBITDA = operating_income (conservative)
                ebitda = operating_income

            return {
                "revenues": safe_get(income_stmt, "revenue")
                or safe_get(income_stmt, "revenues")
                or safe_get(quarterly, "revenue"),
                "net_income": safe_get(income_stmt, "net_income")
                or safe_get(income_stmt, "earnings")
                or safe_get(quarterly, "net_income"),
                "total_assets": safe_get(balance_sheet, "total_assets") or safe_get(quarterly, "total_assets"),
                "total_liabilities": safe_get(balance_sheet, "total_liabilities")
                or safe_get(quarterly, "total_liabilities"),
                "stockholders_equity": safe_get(balance_sheet, "stockholders_equity")
                or safe_get(balance_sheet, "shareholderEquity")
                or safe_get(quarterly, "stockholders_equity"),
                "total_debt": safe_get(balance_sheet, "total_debt")
                or safe_get(balance_sheet, "long_term_debt")
                or safe_get(quarterly, "total_debt"),
                "cash": safe_get(balance_sheet, "cash")
                or safe_get(balance_sheet, "cash_and_equivalents")
                or safe_get(quarterly, "cash"),
                "current_assets": safe_get(balance_sheet, "current_assets") or safe_get(quarterly, "current_assets"),
                "current_liabilities": safe_get(balance_sheet, "current_liabilities")
                or safe_get(quarterly, "current_liabilities"),
                "gross_profit": safe_get(income_stmt, "gross_profit") or safe_get(quarterly, "gross_profit"),
                "operating_income": operating_income,
                "depreciation_amortization": depreciation_amortization,  # NEW: Extract D&A
                "ebitda": ebitda,  # NEW: Calculate EBITDA on-the-fly
                "operating_cash_flow": safe_get(cash_flow, "operating_cash_flow")
                or safe_get(quarterly, "operating_cash_flow"),
                "capital_expenditures": safe_get(cash_flow, "capital_expenditures")
                or safe_get(quarterly, "capital_expenditures"),
                "free_cash_flow": safe_get(cash_flow, "free_cash_flow") or safe_get(quarterly, "free_cash_flow"),
                "inventory": safe_get(balance_sheet, "inventory") or safe_get(quarterly, "inventory"),
                "cost_of_revenue": safe_get(income_stmt, "cost_of_revenue") or safe_get(quarterly, "cost_of_revenue"),
                "dividends": safe_get(cash_flow, "dividends") or safe_get(quarterly, "dividends"),
                "shares_outstanding": safe_get(balance_sheet, "shares_outstanding")
                or safe_get(quarterly, "shares_outstanding"),
            }

        # Handle legacy Dict format (for backwards compatibility)
        elif isinstance(latest_quarter, dict):
            # Extract from nested structure if present (dict format from sec_companyfacts_processed)
            income_stmt = latest_quarter.get("income_statement", {})
            cash_flow = latest_quarter.get("cash_flow", {})
            balance_sheet = latest_quarter.get("balance_sheet", {})

            # Calculate EBITDA on-the-fly from dict format
            operating_income = latest_quarter.get("operating_income", 0) or income_stmt.get("operating_income", 0)
            depreciation_amortization = latest_quarter.get("depreciation_amortization", 0) or cash_flow.get(
                "depreciation_amortization", 0
            )

            # Calculate EBITDA
            ebitda = 0
            if operating_income and depreciation_amortization:
                ebitda = operating_income + depreciation_amortization
            elif operating_income:
                ebitda = operating_income

            return {
                "revenues": latest_quarter.get("revenue", 0)
                or latest_quarter.get("revenues", 0)
                or income_stmt.get("total_revenue", 0),
                "net_income": latest_quarter.get("net_income", 0)
                or latest_quarter.get("earnings", 0)
                or income_stmt.get("net_income", 0),
                "total_assets": latest_quarter.get("total_assets", 0) or balance_sheet.get("total_assets", 0),
                "total_liabilities": latest_quarter.get("total_liabilities", 0)
                or balance_sheet.get("total_liabilities", 0),
                "stockholders_equity": latest_quarter.get("stockholders_equity", 0)
                or latest_quarter.get("shareholderEquity", 0)
                or balance_sheet.get("stockholders_equity", 0),
                "total_debt": latest_quarter.get("total_debt", 0)
                or latest_quarter.get("long_term_debt", 0)
                or balance_sheet.get("total_debt", 0),
                "cash": latest_quarter.get("cash", 0)
                or latest_quarter.get("cash_and_equivalents", 0)
                or balance_sheet.get("cash_and_equivalents", 0),
                "current_assets": latest_quarter.get("current_assets", 0) or balance_sheet.get("current_assets", 0),
                "current_liabilities": latest_quarter.get("current_liabilities", 0)
                or balance_sheet.get("current_liabilities", 0),
                "gross_profit": latest_quarter.get("gross_profit", 0) or income_stmt.get("gross_profit", 0),
                "operating_income": operating_income,
                "depreciation_amortization": depreciation_amortization,  # NEW: Extract D&A
                "ebitda": ebitda,  # NEW: Calculate EBITDA on-the-fly
                "operating_cash_flow": latest_quarter.get("operating_cash_flow", 0)
                or cash_flow.get("operating_cash_flow", 0),
                "capital_expenditures": latest_quarter.get("capital_expenditures", 0)
                or cash_flow.get("capital_expenditures", 0),
                "free_cash_flow": latest_quarter.get("free_cash_flow", 0) or cash_flow.get("free_cash_flow", 0),
                "inventory": latest_quarter.get("inventory", 0) or balance_sheet.get("inventory", 0),
                "cost_of_revenue": latest_quarter.get("cost_of_revenue", 0) or income_stmt.get("cost_of_revenue", 0),
                "dividends": latest_quarter.get("dividends", 0) or cash_flow.get("dividends_paid", 0),
                "shares_outstanding": latest_quarter.get("shares_outstanding", 0)
                or balance_sheet.get("shares_outstanding", 0),
            }

        return {}
