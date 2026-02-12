#!/usr/bin/env python3
"""
RL Backtesting Script - Full Valuation Framework Integration

Generates historical valuation predictions for RL training by simulating
analyses at past dates using only data available at those times.

This script uses the FULL valuation framework:
- SectorValuationRouter for sector-specific logic (banks, REITs, biotech, etc.)
- Growth-adjusted valuation with sustainability discounts
- DCF, GGM, PE, PS, PB, EV/EBITDA models
- Dynamic tier classification
- Proper context features for RL state representation

Usage:
    python3 scripts/rl_backtest.py --symbols AAPL MSFT GOOGL --lookback 12 9 6 3
    python3 scripts/rl_backtest.py --all-symbols --lookback 12 9 6 3
    python3 scripts/rl_backtest.py --top-n 100 --lookback 12 9 6 3

Author: Victor-Invest Team
Date: 2025-12-30
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.insert(0, "src")

# Shared lookback configuration
from investigator.config.lookback_periods import RL_BACKTEST_PERIODS

# Unified data source facade
from investigator.domain.services.data_sources import get_data_source_facade

# Database and config
from investigator.infrastructure.database.db import get_db_manager, safe_json_dumps

# Dynamic model weighting (tier classification)
from investigator.domain.services.dynamic_model_weighting import DynamicModelWeightingService

# RL model weighting (drop-in replacement that uses trained RL policy)
from investigator.domain.services.rl.rl_model_weighting import RLModelWeightingService

# Full valuation models (DCF and GGM)
from investigator.domain.services.valuation.dcf import DCFValuation
from investigator.domain.services.valuation.ggm import GordonGrowthModel

# Sector-specific valuation modules
from investigator.domain.services.valuation.bank_valuation import value_bank, extract_bank_metrics_from_xbrl, BankType
from investigator.domain.services.valuation.semiconductor_valuation import (
    value_semiconductor,
    is_semiconductor_industry,
    classify_semiconductor_company,
)
from investigator.domain.services.valuation.insurance_valuation import value_insurance_company, InsuranceType
from investigator.domain.services.valuation.reit_valuation import value_reit, detect_reit_property_type
from investigator.domain.services.valuation.defense_valuation import (
    value_defense_contractor,
    is_defense_industry,
    classify_defense_contractor,
)
from investigator.domain.services.valuation.biotech_valuation import (
    value_biotech,
    is_biotech_company,
    is_pre_revenue_biotech,
)

# Growth-adjusted valuation
from investigator.domain.services.valuation.growth_adjusted_valuation import (
    calculate_growth_adjusted_valuation,
    classify_growth_profile,
    GrowthProfile,
)

# Shared reward calculator (ensures consistency with outcome_tracker and rl_update_outcomes)
from investigator.domain.services.rl.reward_calculator import (
    RewardCalculator,
    get_reward_calculator,
)

# RL models for context
from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize

# Dual RL Policy (technical + fundamental)
from investigator.domain.services.rl.policy import load_dual_policy, DualRLPolicy

# Shared symbol repository for consistent ticker fetching
from investigator.infrastructure.database.symbol_repository import SymbolRepository

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
    FairValueService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/rl_backtest_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)

# Holding periods from shared configuration
# Short-term: 1m, 3m (momentum/earnings)
# Medium-term: 6m, 12m (business cycle)
# Long-term: 18m, 24m, 36m (fundamental value thesis)
HOLDING_PERIODS = RL_BACKTEST_PERIODS.holding_periods


class RLBacktester:
    """
    Generates historical predictions for RL training using FULL valuation framework.

    Simulates running analyses at past dates using only data
    available at those times, then calculates rewards based
    on actual outcomes.
    """

    def __init__(
        self,
        use_rl_policy: bool = False,
        rl_policy_path: str = "data/rl_models/active_policy.pkl",
        use_dual_policy: bool = False,
    ):
        """
        Initialize RLBacktester.

        Args:
            use_rl_policy: If True, use trained RL policy for weight determination.
                          Otherwise, use rule-based DynamicModelWeightingService.
            rl_policy_path: Path to the trained RL policy file.
            use_dual_policy: If True, use dual RL policy (technical + fundamental).
        """
        self.db = get_db_manager()  # SEC database
        self.use_rl_policy = use_rl_policy
        self.use_dual_policy = use_dual_policy
        self.dual_policy = None

        # Load dual policy if requested
        if use_dual_policy:
            logger.info("Loading Dual RL Policy (technical + fundamental)...")
            self.dual_policy = load_dual_policy()
            logger.info(f"Dual policy loaded: tech={self.dual_policy.technical._update_count}, fund={self.dual_policy.fundamental._update_count}")

        # Shared symbol repository for consistent ticker fetching
        self.symbol_repo = SymbolRepository()
        self.stock_engine = self.symbol_repo.stock_engine

        # Session maker for stock database
        from sqlalchemy.orm import sessionmaker
        self.StockSession = sessionmaker(bind=self.stock_engine)

        # Load config
        import yaml

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.valuation_config = config.get("valuation", {})

        # Create weighting service (RL or rule-based)
        base_weighting_service = DynamicModelWeightingService(self.valuation_config)

        if use_rl_policy:
            logger.info(f"Using RL policy from: {rl_policy_path}")
            self.weighting_service = RLModelWeightingService(
                rl_enabled=True,
                fallback_service=base_weighting_service,
                policy_path=rl_policy_path,
                normalizer_path=rl_policy_path.replace("policy.pkl", "normalizer.pkl"),
            )
            # Check if policy loaded successfully
            if hasattr(self.weighting_service, "policy") and self.weighting_service.policy:
                if self.weighting_service.policy.is_ready():
                    logger.info("RL policy loaded and ready")
                else:
                    logger.warning("RL policy loaded but not ready - falling back to rule-based")
            else:
                logger.warning("RL policy not loaded - falling back to rule-based")
        else:
            logger.info("Using rule-based DynamicModelWeightingService")
            self.weighting_service = base_weighting_service

        # Initialize shared market data services
        # These provide consistent implementations across rl_backtest, batch_analysis_runner, and victor_invest
        self.shares_service = SharesService()
        self.price_service = PriceService()
        self.metadata_service = SymbolMetadataService()
        self.validation_service = DataValidationService()
        self.technical_analysis_service = get_technical_analysis_service()
        logger.info("Shared market data services initialized (including TechnicalAnalysisService)")

        # Initialize unified data source facade for insider, institutional, macro data
        self.data_source_facade = get_data_source_facade()
        logger.info("DataSourceFacade initialized for unified data access")

        # Initialize DataSourceManager for consolidated data access (optional, Phase 1)
        try:
            from investigator.domain.services.data_sources.manager import get_data_source_manager
            self.data_source_manager = get_data_source_manager()
            logger.info("DataSourceManager initialized for consolidated data access")
        except Exception as e:
            logger.debug(f"DataSourceManager not available, using legacy fetchers: {e}")
            self.data_source_manager = None

        # Initialize shared valuation config services
        # Single source of truth for sector multiples, CAPM, GGM defaults
        self.valuation_config_service = ValuationConfigService()
        self.sector_multiples_service = SectorMultiplesService(self.valuation_config_service)
        logger.info("Shared valuation config services initialized")

    def get_quarterly_metrics_structured(
        self,
        symbol: str,
        as_of_date: date,
        num_quarters: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Get quarterly metrics in the structured format expected by DCF/GGM.

        Transforms flat sec_companyfacts_processed data into nested structure:
        {
            'fiscal_year': 2024,
            'fiscal_period': 'Q3',
            'income_statement': {...},
            'cash_flow': {...},
            'balance_sheet': {...},
        }
        """
        with self.db.get_session() as session:
            query = """
                SELECT
                    symbol, fiscal_year, fiscal_period, filed_date,
                    total_revenue, net_income, gross_profit, operating_income,
                    operating_cash_flow, free_cash_flow, capital_expenditures,
                    dividends_paid,
                    total_assets, total_liabilities, stockholders_equity,
                    cash_and_equivalents, long_term_debt, short_term_debt,
                    current_assets, current_liabilities,
                    shares_outstanding, interest_expense, income_tax_expense,
                    depreciation_amortization, roe, period_end_date
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND filed_date <= :as_of_date
                ORDER BY fiscal_year DESC,
                         CASE fiscal_period
                             WHEN 'FY' THEN 5
                             WHEN 'Q4' THEN 4
                             WHEN 'Q3' THEN 3
                             WHEN 'Q2' THEN 2
                             WHEN 'Q1' THEN 1
                         END DESC
                LIMIT :num_quarters
            """
            results = session.execute(
                text(query), {"symbol": symbol, "as_of_date": as_of_date, "num_quarters": num_quarters}
            ).fetchall()

            quarterly_metrics = []
            for row in results:
                operating_income = self._to_float(row[7])
                depreciation = self._to_float(row[23])
                ebitda = operating_income + depreciation if operating_income else None

                # Convert period_end_date to string if it's a date object
                period_end = row[25]
                if period_end and hasattr(period_end, "isoformat"):
                    period_end = period_end.isoformat()

                quarterly_metrics.append(
                    {
                        "fiscal_year": row[1],
                        "fiscal_period": row[2],
                        "filed_date": row[3],
                        "period_end_date": period_end,
                        "income_statement": {
                            "total_revenue": self._to_float(row[4]),
                            "net_income": self._to_float(row[5]),
                            "gross_profit": self._to_float(row[6]),
                            "operating_income": operating_income,
                            "interest_expense": self._to_float(row[21]),
                            "income_tax_expense": self._to_float(row[22]),
                            "ebitda": ebitda,
                        },
                        "cash_flow": {
                            "operating_cash_flow": self._to_float(row[8]),
                            "free_cash_flow": self._to_float(row[9]),
                            "capital_expenditures": self._to_float(row[10]),
                            "dividends_paid": self._to_float(row[11]),
                        },
                        "balance_sheet": {
                            "total_assets": self._to_float(row[12]),
                            "total_liabilities": self._to_float(row[13]),
                            "stockholders_equity": self._to_float(row[14]),
                            "cash_and_equivalents": self._to_float(row[15]),
                            "long_term_debt": self._to_float(row[16]),
                            "short_term_debt": self._to_float(row[17]),
                            "current_assets": self._to_float(row[18]),
                            "current_liabilities": self._to_float(row[19]),
                        },
                        "shares_outstanding": self._to_float(row[20]),
                        "roe": self._to_float(row[24]),
                    }
                )

            return quarterly_metrics

    def get_multi_year_data(
        self,
        symbol: str,
        as_of_date: date,
        num_years: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get multi-year annual data for growth calculations.

        Returns FY (full year) data for the specified number of years.
        """
        with self.db.get_session() as session:
            query = """
                SELECT
                    symbol, fiscal_year, fiscal_period,
                    total_revenue, net_income, gross_profit, operating_income,
                    operating_cash_flow, free_cash_flow, capital_expenditures,
                    dividends_paid,
                    total_assets, total_liabilities, stockholders_equity,
                    shares_outstanding
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND filed_date <= :as_of_date
                  AND fiscal_period = 'FY'
                ORDER BY fiscal_year DESC
                LIMIT :num_years
            """
            results = session.execute(
                text(query), {"symbol": symbol, "as_of_date": as_of_date, "num_years": num_years}
            ).fetchall()

            multi_year_data = []
            for row in results:
                multi_year_data.append(
                    {
                        "fiscal_year": row[1],
                        "fiscal_period": row[2],
                        "total_revenue": self._to_float(row[3]),
                        "net_income": self._to_float(row[4]),
                        "gross_profit": self._to_float(row[5]),
                        "operating_income": self._to_float(row[6]),
                        "operating_cash_flow": self._to_float(row[7]),
                        "free_cash_flow": self._to_float(row[8]),
                        "capital_expenditures": self._to_float(row[9]),
                        "dividends_paid": self._to_float(row[10]),
                        "total_assets": self._to_float(row[11]),
                        "total_liabilities": self._to_float(row[12]),
                        "stockholders_equity": self._to_float(row[13]),
                        "shares_outstanding": self._to_float(row[14]),
                    }
                )

            # Reverse to chronological order (oldest first)
            return list(reversed(multi_year_data))

    def get_symbols_to_backtest(
        self,
        symbols: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Get list of symbols to backtest (legacy method - uses get_symbols_with_sec_data)."""
        if symbols:
            return [s.upper() for s in symbols]

        # Use shared repository - gets symbols with both stock and SEC data
        valid_symbols = self.symbol_repo.get_symbols_with_sec_data(min_market_cap=1_000_000_000)
        return valid_symbols[: top_n or 500]

    def get_russell1000_symbols(self) -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_russell1000_symbols()

    def get_sp500_symbols(self) -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_sp500_symbols()

    def get_all_symbols(self, us_only: bool = True, order_by: str = "stockid") -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_all_symbols(us_only=us_only, order_by=order_by)

    def get_top_n_symbols(self, n: int, us_only: bool = True) -> List[str]:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_top_n_symbols(n, us_only=us_only)

    def get_domestic_filers(self) -> set:
        """Delegate to shared SymbolRepository."""
        return self.symbol_repo.get_domestic_filers()

    def get_historical_financials(
        self,
        symbol: str,
        as_of_date: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Get financial data as it would have been available on as_of_date.

        Returns the most recent annual (FY) or TTM quarterly data filed BEFORE as_of_date.
        """
        with self.db.get_session() as session:
            # Get FY (full year) data
            fy_query = """
                SELECT
                    symbol, fiscal_year, fiscal_period,
                    total_revenue, net_income,
                    operating_cash_flow, free_cash_flow, capital_expenditures,
                    total_assets, total_liabilities, stockholders_equity,
                    cash_and_equivalents, long_term_debt, short_term_debt,
                    gross_profit, operating_income,
                    current_assets, current_liabilities, dividends_paid,
                    shares_outstanding
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND filed_date <= :as_of_date
                  AND fiscal_period = 'FY'
                ORDER BY fiscal_year DESC
                LIMIT 1
            """
            fy_result = session.execute(text(fy_query), {"symbol": symbol, "as_of_date": as_of_date}).fetchone()

            # Get most recent quarterly data for balance sheet
            q_query = """
                SELECT
                    symbol, fiscal_year, fiscal_period,
                    total_revenue, net_income,
                    operating_cash_flow, free_cash_flow, capital_expenditures,
                    total_assets, total_liabilities, stockholders_equity,
                    cash_and_equivalents, long_term_debt, short_term_debt,
                    gross_profit, operating_income,
                    current_assets, current_liabilities, dividends_paid,
                    shares_outstanding
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND filed_date <= :as_of_date
                ORDER BY fiscal_year DESC,
                         CASE fiscal_period
                             WHEN 'FY' THEN 5
                             WHEN 'Q4' THEN 4
                             WHEN 'Q3' THEN 3
                             WHEN 'Q2' THEN 2
                             WHEN 'Q1' THEN 1
                         END DESC
                LIMIT 8
            """
            q_results = session.execute(text(q_query), {"symbol": symbol, "as_of_date": as_of_date}).fetchall()

            # Decide which data to use
            if fy_result:
                latest = fy_result
                ttm_revenue = self._to_float(fy_result[3])
                ttm_net_income = self._to_float(fy_result[4])
                ttm_fcf = self._to_float(fy_result[6])
                ttm_operating_income = self._to_float(fy_result[15])
                ttm_dividends = abs(self._to_float(fy_result[18]) or 0)
                quarters_available = 4

                balance_sheet = q_results[0] if q_results else fy_result
            elif q_results and len(q_results) >= 4:
                # Calculate TTM from 4 quarters
                latest = q_results[0]
                balance_sheet = latest
                ttm_revenue = sum(self._to_float(r[3]) for r in q_results[:4] if r[3])
                ttm_net_income = sum(self._to_float(r[4]) for r in q_results[:4] if r[4])
                ttm_fcf = sum(self._to_float(r[6]) for r in q_results[:4] if r[6])
                ttm_operating_income = sum(self._to_float(r[15]) for r in q_results[:4] if r[15])
                ttm_dividends = sum(abs(self._to_float(r[18]) or 0) for r in q_results[:4] if r[18])
                quarters_available = len(q_results)
            elif q_results:
                latest = q_results[0]
                balance_sheet = latest
                ttm_revenue = self._to_float(latest[3])
                ttm_net_income = self._to_float(latest[4])
                ttm_fcf = self._to_float(latest[6])
                ttm_operating_income = self._to_float(latest[15])
                ttm_dividends = abs(self._to_float(latest[18]) or 0)
                quarters_available = len(q_results)
            else:
                return None

            # EBITDA approximation
            ttm_ebitda = ttm_operating_income + (self._to_float(balance_sheet[8]) * 0.05)

            # Get shares_outstanding from the filing (index 19 after adding to query)
            filing_shares = self._to_float(balance_sheet[19]) if len(balance_sheet) > 19 else None

            return {
                "symbol": symbol,
                "fiscal_year": latest[1],
                "fiscal_period": latest[2],
                "total_revenue": ttm_revenue,
                "net_income": ttm_net_income,
                "ebitda": ttm_ebitda,
                "operating_cash_flow": self._to_float(balance_sheet[5]),
                "free_cash_flow": ttm_fcf,
                "capital_expenditures": self._to_float(balance_sheet[7]),
                "total_assets": self._to_float(balance_sheet[8]),
                "total_liabilities": self._to_float(balance_sheet[9]),
                "stockholders_equity": self._to_float(balance_sheet[10]),
                "cash_and_equivalents": self._to_float(balance_sheet[11]),
                "long_term_debt": self._to_float(balance_sheet[12]),
                "short_term_debt": self._to_float(balance_sheet[13]),
                "dividends_paid": ttm_dividends,
                "gross_profit": self._to_float(balance_sheet[14]),
                "operating_income": self._to_float(balance_sheet[15]),
                "current_assets": self._to_float(balance_sheet[16]),
                "current_liabilities": self._to_float(balance_sheet[17]),
                "quarters_available": quarters_available,
                "shares_outstanding": filing_shares,  # From SEC filing at that time
            }

    def get_historical_price(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[float]:
        """Get stock price on or near target_date from stock database.

        Delegates to shared PriceService for consistent implementation
        across rl_backtest, batch_analysis_runner, and victor_invest.
        """
        return self.price_service.get_price(symbol, target_date)

    def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get symbol metadata (sector, industry, beta, shares) from stock database.

        Delegates to shared SymbolMetadataService for consistent implementation
        across rl_backtest, batch_analysis_runner, and victor_invest.
        """
        metadata = self.metadata_service.get_metadata(symbol)
        if metadata:
            return {
                "sector": metadata.sector,
                "industry": metadata.industry,
                "beta": metadata.beta or 1.0,
                "market_cap": metadata.market_cap or 0,
                "shares_outstanding": metadata.shares_outstanding,
            }
        return {
            "sector": "Unknown",
            "industry": "Unknown",
            "beta": 1.0,
            "market_cap": 0,
            "shares_outstanding": None,
        }

    def get_shares_history_for_normalization(self, symbol: str, lookback_months: List[int]) -> pd.DataFrame:
        """
        Get shares outstanding history and detect/normalize for splits.

        Since prices in our DB are split-adjusted, we need to normalize shares so that:
        - Pre-split periods get multiplied by split factor to match split-adjusted prices
        - Post-split periods remain unchanged

        Delegates to shared SharesService for consistent implementation
        across rl_backtest, batch_analysis_runner, and victor_invest.

        Returns DataFrame with columns: [months_back, as_of_date, raw_shares, split_factor, adjusted_shares]
        """
        return self.shares_service.get_shares_history(symbol, lookback_months)

    def calculate_ratios(
        self,
        financials: Dict[str, Any],
        current_price: float,
        metadata: Optional[Dict[str, Any]] = None,
        symbol: str = "",
        adjusted_shares: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate financial ratios for valuation."""
        # Priority for shares:
        # 1. Pre-computed adjusted_shares (split-normalized from get_shares_history_for_normalization)
        # 2. SEC filing shares
        # 3. Symbol table shares
        # 4. Derived from market cap
        shares_source = "unknown"

        if adjusted_shares and adjusted_shares > 0:
            shares = adjusted_shares
            shares_source = "split_normalized"
        elif financials.get("shares_outstanding") and financials["shares_outstanding"] > 0:
            shares = financials["shares_outstanding"]
            shares_source = "SEC_filing"
        elif metadata and metadata.get("shares_outstanding") and metadata["shares_outstanding"] > 0:
            shares = metadata["shares_outstanding"]
            shares_source = "symbol_table"
        elif metadata and metadata.get("market_cap", 0) > 0 and current_price > 0:
            shares = metadata["market_cap"] / current_price
            shares_source = "derived_from_mktcap"
        else:
            raise ValueError(f"{symbol}: No shares data available - cannot calculate ratios")

        market_cap = shares * current_price

        logger.debug(f"{symbol} shares: {shares/1e9:.3f}B ({shares_source}), mktcap: ${market_cap/1e9:.1f}B")
        financials["shares_outstanding"] = shares

        revenue = financials.get("total_revenue", 0) or 0
        net_income = financials.get("net_income", 0) or 0
        ebitda = financials.get("ebitda", 0) or 0
        fcf = financials.get("free_cash_flow", 0) or 0
        equity = financials.get("stockholders_equity", 0) or 1
        dividends = abs(financials.get("dividends_paid", 0) or 0)
        gross_profit = financials.get("gross_profit", 0) or 0
        debt = (financials.get("long_term_debt", 0) or 0) + (financials.get("short_term_debt", 0) or 0)

        eps = net_income / shares if shares > 0 else 0
        bvps = equity / shares if shares > 0 else 0

        return {
            "market_cap": market_cap,
            "shares_outstanding": shares,
            "pe_ratio": current_price / eps if eps > 0 else 0,
            "ps_ratio": market_cap / revenue if revenue > 0 else 0,
            "pb_ratio": current_price / bvps if bvps > 0 else 0,
            "ev_ebitda": (market_cap + debt) / ebitda if ebitda > 0 else 0,
            "payout_ratio": dividends / net_income if net_income > 0 else 0,
            "revenue_growth_pct": 0,  # Would need prior year data
            "fcf_margin": fcf / revenue if revenue > 0 else 0,
            "gross_margin": gross_profit / revenue if revenue > 0 else 0,
            "operating_margin": (financials.get("operating_income", 0) or 0) / revenue if revenue > 0 else 0,
            "net_margin": net_income / revenue if revenue > 0 else 0,
            "roe": net_income / equity if equity > 0 else 0,
            "debt_to_equity": debt / equity if equity > 0 else 0,
            "rule_of_40_score": 0 + (fcf / revenue * 100 if revenue > 0 else 0),
            "ttm_eps": eps,
            "book_value_per_share": bvps,
            "fcf_quarters": financials.get("quarters_available", 4),
        }

    def calculate_fair_values_full_framework(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, float],
        metadata: Dict[str, Any],
        current_price: float,
    ) -> Tuple[Dict[str, Optional[float]], str, Dict[str, Any]]:
        """
        Calculate fair values using FULL valuation framework.

        Uses:
        - Sector-specific valuation (banks, REITs, biotech, etc.)
        - Growth-adjusted valuation
        - All valuation models (DCF, GGM, PE, PS, PB, EV/EBITDA)

        Returns:
            Tuple of (fair_values dict, tier classification, audit trail)
        """
        sector = metadata.get("sector", "Unknown")
        industry = metadata.get("industry", "Unknown")
        shares = ratios.get("shares_outstanding", 1) or 1
        eps = ratios.get("ttm_eps", 0)
        bvps = ratios.get("book_value_per_share", 0)
        revenue = financials.get("total_revenue", 0) or 0
        ebitda = financials.get("ebitda", 0) or 0
        fcf = financials.get("free_cash_flow", 0) or 0
        dividends = abs(financials.get("dividends_paid", 0) or 0)
        net_income = financials.get("net_income", 0) or 0
        market_cap = ratios.get("market_cap", 0)

        fair_values = {}
        audit = {"sector_specific": False, "models_used": []}

        # 1. Check for sector-specific valuation
        try:
            # Banks
            if sector == "Financials" and "bank" in industry.lower():
                bank_result = value_bank(
                    symbol=symbol,
                    financials=financials,
                    current_price=current_price,
                    shares_outstanding=shares,
                )
                if bank_result and bank_result.fair_value:
                    fair_values["pb"] = bank_result.fair_value
                    audit["sector_specific"] = True
                    audit["models_used"].append("bank_pb")

            # Semiconductors
            elif is_semiconductor_industry(industry):
                semi_result = value_semiconductor(
                    symbol=symbol,
                    financials=financials,
                    current_price=current_price,
                    shares_outstanding=shares,
                )
                if semi_result and semi_result.fair_value:
                    fair_values["semi_adjusted"] = semi_result.fair_value
                    audit["sector_specific"] = True
                    audit["models_used"].append("semiconductor_cycle")

            # REITs
            elif sector == "Real Estate":
                reit_result = value_reit(
                    symbol=symbol,
                    financials=financials,
                    current_price=current_price,
                    shares_outstanding=shares,
                )
                if reit_result and reit_result.fair_value:
                    fair_values["ffo"] = reit_result.fair_value
                    audit["sector_specific"] = True
                    audit["models_used"].append("reit_ffo")

            # Insurance, Managed Care & Health Care Distribution
            # (health insurers and drug distributors are in Health Care sector, not Finance)
            # All have same economics: high revenue, thin margins (1-3%), PS model inappropriate
            elif any(term in industry.lower() for term in [
                "insurance", "insurers", "insurer",
                "managed health care", "managed care", "hmo", "health maintenance",
                "health care distribution", "healthcare distribution"
            ]):
                ins_result = value_insurance_company(
                    symbol=symbol,
                    financials=financials,
                    current_price=current_price,
                    shares_outstanding=shares,
                )
                if ins_result:
                    fair_values["insurance_pb"] = ins_result.get("fair_value")
                    audit["sector_specific"] = True
                    audit["models_used"].append("insurance_combined_ratio")

            # Defense
            elif is_defense_industry(industry):
                defense_result = value_defense_contractor(
                    symbol=symbol,
                    financials=financials,
                    current_price=current_price,
                    shares_outstanding=shares,
                )
                if defense_result and defense_result.fair_value:
                    fair_values["defense_backlog"] = defense_result.fair_value
                    audit["sector_specific"] = True
                    audit["models_used"].append("defense_backlog")

        except Exception as e:
            logger.debug(f"Sector-specific valuation failed for {symbol}: {e}")

        # 2. Standard valuation models

        # P/E valuation with growth adjustment
        if eps > 0:
            try:
                revenue_growth = ratios.get("revenue_growth_pct", 0) / 100
                growth_result = calculate_growth_adjusted_valuation(
                    ttm_eps=eps,
                    forward_eps=eps * 1.1,  # Assume 10% growth
                    revenue_growth=revenue_growth,
                    current_price=current_price,
                    sector=sector,
                )
                if growth_result and growth_result.fair_value:
                    fair_values["pe"] = growth_result.fair_value
                    audit["models_used"].append("growth_adjusted_pe")
            except Exception as e:
                # Fallback to simple P/E
                pe_multiple = self._get_sector_pe_multiple(sector)
                fair_values["pe"] = eps * pe_multiple
                audit["models_used"].append("simple_pe")
        else:
            fair_values["pe"] = None

        # P/S valuation
        if revenue > 0:
            ps_multiple = self._get_sector_ps_multiple(sector)
            fair_values["ps"] = (revenue / shares) * ps_multiple
            audit["models_used"].append("ps")
        else:
            fair_values["ps"] = None

        # P/B valuation (if not already set by sector-specific)
        if "pb" not in fair_values and bvps > 0:
            pb_multiple = self._get_sector_pb_multiple(sector)
            fair_values["pb"] = bvps * pb_multiple
            audit["models_used"].append("pb")

        # EV/EBITDA valuation
        if ebitda > 0:
            ev_multiple = self._get_sector_ev_multiple(sector)
            ev = ebitda * ev_multiple
            debt = (financials.get("long_term_debt", 0) or 0) + (financials.get("short_term_debt", 0) or 0)
            cash = financials.get("cash_and_equivalents", 0) or 0
            equity_value = ev - debt + cash
            fair_values["ev_ebitda"] = equity_value / shares if shares > 0 else None
            audit["models_used"].append("ev_ebitda")
        else:
            fair_values["ev_ebitda"] = None

        # DCF valuation (using FULL DCFValuation model)
        if fcf > 0 and hasattr(self, "_current_quarterly_metrics") and self._current_quarterly_metrics:
            try:
                dcf_model = DCFValuation(
                    symbol=symbol,
                    quarterly_metrics=self._current_quarterly_metrics,
                    multi_year_data=self._current_multi_year_data or [],
                    db_manager=self.db,
                )
                dcf_result = dcf_model.calculate_dcf_valuation()
                if dcf_result.get("applicable", False) and dcf_result.get("fair_value_per_share", 0) > 0:
                    fair_values["dcf"] = dcf_result["fair_value_per_share"]
                    audit["models_used"].append("dcf_full")
                    audit["dcf_assumptions"] = dcf_result.get("assumptions", {})
                else:
                    # Fallback to simple multiple
                    fair_values["dcf"] = (fcf * 12) / shares
                    audit["models_used"].append("dcf_simple")
            except Exception as e:
                logger.debug(f"Full DCF failed for {symbol}, using simple: {e}")
                fair_values["dcf"] = (fcf * 12) / shares
                audit["models_used"].append("dcf_simple")
        elif fcf > 0:
            fair_values["dcf"] = (fcf * 12) / shares
            audit["models_used"].append("dcf_simple")
        else:
            fair_values["dcf"] = None

        # GGM (using FULL GordonGrowthModel)
        payout_ratio = ratios.get("payout_ratio", 0)
        if dividends > 0 and payout_ratio >= 0.20:
            if hasattr(self, "_current_quarterly_metrics") and self._current_quarterly_metrics:
                try:
                    ggm_model = GordonGrowthModel(
                        symbol=symbol,
                        quarterly_metrics=self._current_quarterly_metrics,
                        multi_year_data=self._current_multi_year_data or [],
                        db_manager=self.db,
                    )
                    # Estimate cost of equity from beta using config-based CAPM
                    beta = metadata.get("beta", 1.0)
                    capm_params = self.valuation_config_service.get_capm_params()
                    risk_free_rate = capm_params.get("risk_free_rate", 0.04)
                    market_premium = capm_params.get("market_equity_premium", 0.05)
                    cost_of_equity = risk_free_rate + beta * market_premium

                    ggm_result = ggm_model.calculate_ggm_valuation(cost_of_equity=cost_of_equity)
                    if ggm_result.get("applicable", False) and ggm_result.get("fair_value_per_share", 0) > 0:
                        fair_values["ggm"] = ggm_result["fair_value_per_share"]
                        audit["models_used"].append("ggm_full")
                        audit["ggm_assumptions"] = ggm_result.get("assumptions", {})
                    else:
                        # Fallback to simple GGM with config defaults
                        dps = dividends / shares
                        ggm_defaults = self.valuation_config_service.get_ggm_defaults()
                        growth_rate = ggm_defaults.get("growth_rate", 0.03)
                        if cost_of_equity > growth_rate:
                            d1 = dps * (1 + growth_rate)
                            fair_values["ggm"] = d1 / (cost_of_equity - growth_rate)
                            audit["models_used"].append("ggm_simple")
                        else:
                            fair_values["ggm"] = None
                except Exception as e:
                    logger.debug(f"Full GGM failed for {symbol}, using simple: {e}")
                    dps = dividends / shares
                    ggm_defaults = self.valuation_config_service.get_ggm_defaults()
                    growth_rate = ggm_defaults.get("growth_rate", 0.03)
                    cost_of_equity = ggm_defaults.get("cost_of_equity", 0.08)
                    if cost_of_equity > growth_rate:
                        d1 = dps * (1 + growth_rate)
                        fair_values["ggm"] = d1 / (cost_of_equity - growth_rate)
                        audit["models_used"].append("ggm_simple")
                    else:
                        fair_values["ggm"] = None
            else:
                # No quarterly data, use simple GGM with config defaults
                dps = dividends / shares
                ggm_defaults = self.valuation_config_service.get_ggm_defaults()
                growth_rate = ggm_defaults.get("growth_rate", 0.03)
                cost_of_equity = ggm_defaults.get("cost_of_equity", 0.08)
                if cost_of_equity > growth_rate:
                    d1 = dps * (1 + growth_rate)
                    fair_values["ggm"] = d1 / (cost_of_equity - growth_rate)
                    audit["models_used"].append("ggm_simple")
                else:
                    fair_values["ggm"] = None
        else:
            fair_values["ggm"] = None

        # 3. Get dynamic tier classification and weights
        financials["sector"] = sector
        financials["industry"] = industry
        financials["market_cap"] = market_cap
        # Add key aliases expected by determine_weights (which uses 'revenue' not 'total_revenue')
        financials["revenue"] = financials.get("total_revenue", 0) or 0
        financials["current_price"] = current_price

        weights, tier, weight_audit = self.weighting_service.determine_weights(
            symbol=symbol,
            financials=financials,
            ratios=ratios,
        )
        audit["tier"] = tier
        audit["weights"] = weights

        # Sanitize fair values - filter out complex numbers or non-numeric values
        sanitized_fvs = {}
        for model, fv in fair_values.items():
            if fv is None:
                sanitized_fvs[model] = None
            elif isinstance(fv, (int, float)) and not isinstance(fv, complex):
                sanitized_fvs[model] = fv
            else:
                logger.warning(f"{symbol} - Discarding {model} fair value (non-numeric: {type(fv).__name__})")
                sanitized_fvs[model] = None

        return sanitized_fvs, tier, audit

    def _get_sector_pe_multiple(self, sector: str) -> float:
        """Get sector-appropriate P/E multiple from config."""
        return self.sector_multiples_service.get_pe(sector)

    def _get_sector_ps_multiple(self, sector: str) -> float:
        """Get sector-appropriate P/S multiple from config."""
        return self.sector_multiples_service.get_ps(sector)

    def _get_sector_pb_multiple(self, sector: str) -> float:
        """Get sector-appropriate P/B multiple from config."""
        return self.sector_multiples_service.get_pb(sector)

    def _get_sector_ev_multiple(self, sector: str) -> float:
        """Get sector-appropriate EV/EBITDA multiple from config."""
        return self.sector_multiples_service.get_ev_ebitda(sector)

    def calculate_blended_fair_value(
        self,
        fair_values: Dict[str, Optional[float]],
        weights: Dict[str, float],
        current_price: Optional[float] = None,
        max_upside_multiple: float = 3.0,
        min_downside_multiple: float = 0.2,
        tier: Optional[str] = None,
    ) -> float:
        """
        Calculate weighted blended fair value with tier-aware sanity caps.

        Delegates to shared FairValueService for consistency across all code paths:
        - rl_backtest.py (this file)
        - batch_analysis_runner.py
        - CLI analysis
        - workflow handlers

        Args:
            fair_values: Dict mapping model name to fair value
            weights: Dict mapping model name to weight
            current_price: Current stock price for sanity capping (optional)
            max_upside_multiple: Base max fair value as multiple of price (default 3x = 200% upside)
            min_downside_multiple: Min fair value as multiple of price (default 0.2x = -80%)
            tier: Optional tier classification for tier-aware capping (growth=5x, mature=3x)

        Returns:
            Blended fair value
        """
        # Filter out complex numbers before passing to shared service
        clean_fair_values = {}
        for model, fv in fair_values.items():
            if fv is not None and isinstance(fv, (int, float)) and fv > 0:
                clean_fair_values[model] = float(fv)
            elif fv is not None and not isinstance(fv, (int, float)):
                logger.warning(f"Skipping {model} with non-numeric fair value: {type(fv).__name__}")

        # Use shared FairValueService for consistent calculation
        fv_service = FairValueService()
        blended, _audit = fv_service.calculate_blended_fair_value(
            fair_values=clean_fair_values,
            weights=weights,
            current_price=current_price,
            max_upside_multiple=max_upside_multiple,
            min_downside_multiple=min_downside_multiple,
            tier=tier,
        )
        return blended

    def calculate_reward(
        self,
        predicted_fv: float,
        price_at_prediction: float,
        actual_price: float,
        days: int = 90,
        beta: float = 1.0,
    ) -> float:
        """
        Calculate risk-adjusted, annualized ROI-weighted reward signal.

        Uses shared RewardCalculator for consistency across all code paths:
        - rl_backtest.py (this file)
        - outcome_tracker.py (production tracking)
        - rl_update_outcomes.py (cron job)

        See RewardCalculator for full documentation of the reward formula.
        """
        calculator = get_reward_calculator()
        return calculator.calculate_simple(
            predicted_fv=predicted_fv,
            price_at_prediction=price_at_prediction,
            actual_price=actual_price,
            days=days,
            beta=beta,
        )

    def calculate_per_model_rewards(
        self,
        fair_values: Dict[str, Optional[float]],
        price_at_prediction: float,
        actual_price: float,
        days: int = 90,
        beta: float = 1.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate risk-adjusted, annualized ROI-weighted rewards for each model.

        Uses shared RewardCalculator for consistency.
        """
        calculator = get_reward_calculator()
        per_model = {}

        for model, fv in fair_values.items():
            # Protect against complex numbers (from sqrt of negative in growth calculations)
            if fv is not None and isinstance(fv, (int, float)) and fv > 0 and actual_price > 0 and price_at_prediction > 0:
                result = calculator.calculate(
                    predicted_fv=fv,
                    price_at_prediction=price_at_prediction,
                    actual_price=actual_price,
                    days=days,
                    beta=beta,
                )

                # Error metrics for reporting
                error = abs(fv - actual_price) / actual_price
                raw_return = (actual_price - price_at_prediction) / price_at_prediction

                per_model[model] = {
                    "fair_value": round(fv, 2),
                    "error_pct": round(error * 100, 2),
                    "raw_return_pct": round(raw_return * 100, 2),
                    "position_return_pct": round(result.position_return * 100, 2),
                    "annualized_return_pct": round(result.annualized_return * 100, 2),
                    "risk_adjusted_pct": round(result.risk_adjusted_return * 100, 2),
                    "direction_correct": result.direction_correct,
                    "reward_90d": round(result.reward, 4),
                }

        return per_model

    def build_context_features(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, float],
        metadata: Dict[str, Any],
        current_price: float,
        analysis_date: date,
        fair_value: Optional[float] = None,
        valuation_gap: float = 0.0,
        valuation_confidence: float = 0.5,
        position_signal: int = 0,
        insider_data: Optional[Dict[str, Any]] = None,
        economic_indicators: Optional[Dict[str, Any]] = None,
    ) -> ValuationContext:
        """
        Build ValuationContext for RL state representation.

        Includes fundamental, technical, entry/exit signal, insider sentiment,
        and economic indicator features (Regional Fed + CBOE).
        """
        insider_data = insider_data or {}
        economic_indicators = economic_indicators or {}

        # Extract economic indicator values
        regional_fed = economic_indicators.get("regional_fed", {})
        cboe = economic_indicators.get("cboe", {})
        fed_summary = regional_fed.get("summary", {}) if isinstance(regional_fed, dict) else {}
        market_cap = ratios.get("market_cap", 0)

        # Company size classification
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

        # Growth stage classification
        net_income = financials.get("net_income", 0) or 0
        ebitda = financials.get("ebitda", 0) or 0
        payout_ratio = ratios.get("payout_ratio", 0)
        revenue_growth = ratios.get("revenue_growth_pct", 0) / 100

        if net_income < 0 and ebitda <= 0:
            growth_stage = GrowthStage.PRE_PROFIT
        elif payout_ratio > 0.30:
            growth_stage = GrowthStage.DIVIDEND_PAYING
        elif revenue_growth > 0.25:
            growth_stage = GrowthStage.HIGH_GROWTH
        elif net_income > 0:
            growth_stage = GrowthStage.MATURE
        else:
            growth_stage = GrowthStage.EARLY_GROWTH

        # Get technical features from TechnicalAnalysisService
        try:
            tech_features = self.technical_analysis_service.get_technical_features(
                symbol=symbol,
                analysis_date=analysis_date,
                lookback_days=365,
                fair_value=fair_value,
            )
        except Exception as e:
            logger.warning(f"Could not get technical features for {symbol}: {e}")
            from investigator.domain.services.market_data import TechnicalFeatures
            tech_features = TechnicalFeatures()

        return ValuationContext(
            symbol=symbol,
            analysis_date=analysis_date,
            sector=metadata.get("sector", "Unknown"),
            industry=metadata.get("industry", "Unknown"),
            growth_stage=growth_stage,
            company_size=company_size,
            profitability_score=min(1.0, max(0, ratios.get("net_margin", 0) + 0.1) / 0.3),
            pe_level=min(1.0, ratios.get("pe_ratio", 20) / 50),
            revenue_growth=revenue_growth,
            fcf_margin=ratios.get("fcf_margin", 0),
            rule_of_40_score=ratios.get("rule_of_40_score", 0),
            payout_ratio=payout_ratio,
            debt_to_equity=min(3.0, ratios.get("debt_to_equity", 0)),
            gross_margin=ratios.get("gross_margin", 0),
            operating_margin=ratios.get("operating_margin", 0),
            net_margin=self._calculate_net_margin(financials, ratios),
            margin_bin=self._calculate_margin_bin(financials, ratios),
            is_low_margin_industry=self._is_low_margin_industry(metadata.get("sector", ""), metadata.get("industry", "")),
            data_quality_score=75.0,  # Default for backtest
            quarters_available=financials.get("quarters_available", 4),
            current_price=current_price,
            # Technical indicators (from TechnicalAnalysisService)
            rsi_14=tech_features.rsi_14,
            macd_histogram=tech_features.macd_histogram,
            obv_trend=tech_features.obv_trend,
            adx_14=tech_features.adx_14,
            stoch_k=tech_features.stoch_k,
            mfi_14=tech_features.mfi_14,
            # Entry/Exit signal features (from TechnicalAnalysisService)
            entry_signal_strength=tech_features.entry_signal_strength,
            exit_signal_strength=tech_features.exit_signal_strength,
            signal_confluence=tech_features.signal_confluence,
            days_from_support=tech_features.days_from_support,
            risk_reward_ratio=tech_features.risk_reward_ratio,
            # Market context
            volatility=tech_features.volatility,
            technical_trend=(tech_features.price_vs_sma_20 + tech_features.price_vs_sma_50 + tech_features.price_vs_sma_200) / 3,
            # Insider sentiment features (from DataSourceFacade)
            insider_sentiment=insider_data.get("sentiment_score", 0.0),
            insider_buy_ratio=self._calculate_insider_buy_ratio(insider_data),
            insider_transaction_value=self._normalize_insider_value(insider_data),
            insider_cluster_signal=self._calculate_cluster_signal(insider_data),
            insider_key_exec_activity=insider_data.get("key_exec_activity", 0.0),
            # Position signal features
            valuation_gap=valuation_gap,
            valuation_confidence=valuation_confidence,
            position_signal=position_signal,
            # Economic indicators (Regional Fed)
            gdpnow=fed_summary.get("gdpnow") or 2.0,
            cfnai=fed_summary.get("cfnai") or 0.0,
            nfci=fed_summary.get("nfci") or 0.0,
            kcfsi=fed_summary.get("kcfsi") or 0.0,
            inflation_expectations=fed_summary.get("inflation_expectations") or 2.5,
            recession_probability=fed_summary.get("recession_probability") or 0.15,
            empire_state_mfg=fed_summary.get("empire_state_mfg") or 0.0,
            # CBOE volatility data
            vix=cboe.get("vix") or 18.0,
            vix_term_structure=(cboe.get("vix3m") or 18.0) / max(cboe.get("vix") or 18.0, 1.0),
            skew=cboe.get("skew") or 120.0,
            volatility_regime=self._classify_volatility_regime_int(cboe.get("volatility_regime")),
            is_backwardation=cboe.get("is_backwardation", False),
        )

    def _calculate_insider_buy_ratio(self, insider_data: Dict[str, Any]) -> float:
        """Calculate buy ratio from insider transaction counts."""
        buy_count = insider_data.get("buy_count", 0) or 0
        sell_count = insider_data.get("sell_count", 0) or 0
        total = buy_count + sell_count
        if total > 0:
            return buy_count / total
        return 0.5  # Neutral when no activity

    def _normalize_insider_value(self, insider_data: Dict[str, Any]) -> float:
        """Normalize net insider transaction value to -1 to +1 range."""
        buy_value = insider_data.get("buy_value", 0) or 0
        sell_value = insider_data.get("sell_value", 0) or 0
        net_value = buy_value - sell_value
        # Normalize: $10M+ buying = +1, $10M+ selling = -1
        normalized = net_value / 10_000_000
        return max(-1.0, min(1.0, normalized))

    def _calculate_cluster_signal(self, insider_data: Dict[str, Any]) -> float:
        """Calculate cluster signal based on cluster detection and net direction."""
        if not insider_data.get("cluster_detected", False):
            return 0.0
        # Positive cluster (buying) vs negative cluster (selling)
        buy_value = insider_data.get("buy_value", 0) or 0
        sell_value = insider_data.get("sell_value", 0) or 0
        if buy_value >= sell_value:
            return 1.0
        return -1.0

    def _calculate_net_margin(self, financials: Dict[str, Any], ratios: Dict[str, float]) -> float:
        """Calculate net profit margin from financials or ratios."""
        # Try ratios first
        net_margin = ratios.get("net_margin", 0.0)
        if net_margin != 0.0:
            return net_margin
        # Calculate from financials
        revenue = financials.get("total_revenue") or financials.get("revenue") or 0
        net_income = financials.get("net_income") or 0
        if revenue > 0:
            return net_income / revenue
        return 0.0

    def _calculate_margin_bin(self, financials: Dict[str, Any], ratios: Dict[str, float]) -> int:
        """Categorize net margin into bins for RL learning.

        Bins:
            0: very_low (<2%) - PS weight should be minimal
            1: low (2-5%) - PS weight should be reduced
            2: medium (5-10%) - normal PS weight
            3: high (>10%) - PS weight appropriate
        """
        net_margin = self._calculate_net_margin(financials, ratios)
        if net_margin < 0.02:
            return 0  # very_low
        elif net_margin < 0.05:
            return 1  # low
        elif net_margin < 0.10:
            return 2  # medium
        else:
            return 3  # high

    def _is_low_margin_industry(self, sector: str, industry: str) -> bool:
        """Check if industry is known to have structurally low margins (<5%)."""
        if not industry:
            return False

        industry_lower = industry.lower()

        low_margin_industries = [
            "department", "specialty retail", "discount stores", "warehouse clubs",
            "food chains", "grocery", "supermarket",
            "consumer electronics/video chains",
            "computer manufacturing", "computer hardware",
            "electrical products", "electronic components",
            "air freight", "airlines", "airline",
            "meat/poultry/fish", "packaged foods", "food processing",
            "farm products", "farming/seeds",
            "hospital", "nursing", "medical/nursing services",
            "insurance", "insurers", "managed health care", "managed care",
            "health care distribution",
            "cable", "pay television",
            "oil/gas transmission", "gas distribution",
            "beverages",
        ]

        for term in low_margin_industries:
            if term in industry_lower:
                return True
        return False

    def _classify_volatility_regime_int(self, regime: Optional[str]) -> int:
        """Convert volatility regime string to integer for RL features.

        Returns:
            0=very_low, 1=low, 2=normal, 3=elevated, 4=high, 5=extreme
        """
        regime_map = {
            "very_low": 0,
            "low": 1,
            "normal": 2,
            "elevated": 3,
            "high": 4,
            "extreme": 5,
        }
        return regime_map.get(regime or "normal", 2)

    def fetch_insider_data_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch insider sentiment data using DataSourceFacade (synchronous)."""
        try:
            analysis_data = self.data_source_facade.get_historical_data_sync(
                symbol=symbol,
                as_of_date=as_of_date,
            )
            return analysis_data.insider_data
        except Exception as e:
            logger.debug(f"Could not fetch insider data for {symbol}: {e}")
            return {}

    def fetch_economic_indicators_sync(self, as_of_date: date) -> Dict[str, Any]:
        """Fetch economic indicators (Regional Fed + CBOE) using DataSourceFacade."""
        try:
            # Use a dummy symbol - macro data is symbol-independent
            analysis_data = self.data_source_facade.get_historical_data_sync(
                symbol="_MACRO",
                as_of_date=as_of_date,
            )
            return {
                "regional_fed": analysis_data.regional_fed_indicators,
                "cboe": analysis_data.cboe_data,
            }
        except Exception as e:
            logger.debug(f"Could not fetch economic indicators for {as_of_date}: {e}")
            return {"regional_fed": {}, "cboe": {}}

    def get_rl_context_features(
        self, symbol: str, as_of_date: date
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get consolidated RL context features using DataSourceManager.

        This method provides a unified interface for fetching all data needed
        for RL feature extraction, falling back to legacy methods if
        DataSourceManager is unavailable.

        Returns:
            Tuple of (insider_data, economic_indicators)
        """
        # Try DataSourceManager first (consolidated, efficient)
        if self.data_source_manager is not None:
            try:
                consolidated = self.data_source_manager.get_data(symbol, as_of_date)

                # Extract insider data - handle nested sentiment_score structure
                insider_data = {}
                if consolidated.insider:
                    summary = consolidated.insider.get("summary", {})
                    sentiment_score_obj = consolidated.insider.get("sentiment_score", {})
                    insider_data = {
                        # sentiment_score is nested: {"score": float, "buy_value": float, ...}
                        "sentiment_score": sentiment_score_obj.get("score", 0.0) if isinstance(sentiment_score_obj, dict) else 0.0,
                        "buy_count": summary.get("buys", 0),
                        "sell_count": summary.get("sells", 0),
                        # buy_value/sell_value from sentiment_score object
                        "buy_value": sentiment_score_obj.get("buy_value", 0) if isinstance(sentiment_score_obj, dict) else 0,
                        "sell_value": sentiment_score_obj.get("sell_value", 0) if isinstance(sentiment_score_obj, dict) else 0,
                        "cluster_detected": summary.get("cluster_detected", False),
                        "key_exec_activity": summary.get("key_exec_activity", 0.0),
                    }

                # Extract economic indicators
                economic_indicators = {}
                if consolidated.fed_districts:
                    economic_indicators["regional_fed"] = consolidated.fed_districts
                if consolidated.volatility:
                    economic_indicators["cboe"] = consolidated.volatility
                if consolidated.macro:
                    economic_indicators["macro"] = consolidated.macro
                if consolidated.treasury:
                    economic_indicators["treasury"] = consolidated.treasury

                logger.debug(
                    f"Used DataSourceManager for {symbol} RL context (sources: {consolidated.sources_succeeded})"
                )
                return insider_data, economic_indicators

            except Exception as e:
                logger.debug(f"DataSourceManager fetch failed for {symbol}, falling back to legacy: {e}")

        # Fallback to legacy fetchers
        insider_data = self.fetch_insider_data_sync(symbol, as_of_date)
        economic_indicators = self.fetch_economic_indicators_sync(as_of_date)
        return insider_data, economic_indicators

    def record_prediction(
        self,
        symbol: str,
        analysis_date: date,
        fiscal_period: str,
        blended_fair_value: float,
        current_price: float,
        fair_values: Dict[str, Optional[float]],
        weights: Dict[str, float],
        tier_classification: str,
        context_features: ValuationContext,
        actual_price_30d: Optional[float],
        actual_price_90d: Optional[float],
        reward_30d: Optional[float],
        reward_90d: Optional[float],
        per_model_rewards: Dict[str, Dict[str, Any]],
        position_type: str = "inferred",
        entry_date: Optional[date] = None,
        exit_date_30d: Optional[date] = None,
        exit_date_90d: Optional[date] = None,
    ) -> Optional[int]:
        """Record prediction to database.

        Args:
            position_type: 'LONG', 'SHORT', or 'inferred' (legacy behavior based on FV vs Price)
            entry_date: Date when position was entered (defaults to analysis_date)
            exit_date_30d: Date 30 days after entry (when 30d price was measured)
            exit_date_90d: Date 90 days after entry (when 90d price was measured)

        Returns:
            Primary key of inserted record, or None if validation failed
        """
        # Validation: Don't save records with $0 fair value (indicates all models failed)
        if blended_fair_value <= 0:
            logger.warning(
                f"Skipping save for {symbol} {analysis_date}: blended_fair_value is ${blended_fair_value:.2f}"
            )
            return None

        try:
            predicted_upside_pct = (
                ((blended_fair_value - current_price) / current_price * 100) if current_price > 0 else 0
            )

            # Default entry_date to analysis_date if not provided
            effective_entry_date = entry_date or analysis_date

            with self.db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        INSERT INTO valuation_outcomes (
                            symbol, analysis_date, fiscal_period,
                            blended_fair_value, current_price, predicted_upside_pct,
                            dcf_fair_value, pe_fair_value, ps_fair_value,
                            evebitda_fair_value, pb_fair_value, ggm_fair_value,
                            model_weights, tier_classification, context_features,
                            actual_price_30d, actual_price_90d,
                            reward_30d, reward_90d, per_model_rewards,
                            outcome_updated_at, ab_test_group, policy_version, position_type,
                            entry_date, exit_date_30d, exit_date_90d
                        ) VALUES (
                            :symbol, :analysis_date, :fiscal_period,
                            :blended_fair_value, :current_price, :predicted_upside_pct,
                            :dcf_fair_value, :pe_fair_value, :ps_fair_value,
                            :evebitda_fair_value, :pb_fair_value, :ggm_fair_value,
                            :model_weights, :tier_classification, :context_features,
                            :actual_price_30d, :actual_price_90d,
                            :reward_30d, :reward_90d, :per_model_rewards,
                            CURRENT_TIMESTAMP, 'backtest', 'backtest_v3_dual_position', :position_type,
                            :entry_date, :exit_date_30d, :exit_date_90d
                        )
                        ON CONFLICT (symbol, analysis_date, position_type) DO UPDATE SET
                            blended_fair_value = EXCLUDED.blended_fair_value,
                            current_price = EXCLUDED.current_price,
                            predicted_upside_pct = EXCLUDED.predicted_upside_pct,
                            dcf_fair_value = EXCLUDED.dcf_fair_value,
                            pe_fair_value = EXCLUDED.pe_fair_value,
                            ps_fair_value = EXCLUDED.ps_fair_value,
                            evebitda_fair_value = EXCLUDED.evebitda_fair_value,
                            pb_fair_value = EXCLUDED.pb_fair_value,
                            ggm_fair_value = EXCLUDED.ggm_fair_value,
                            model_weights = EXCLUDED.model_weights,
                            tier_classification = EXCLUDED.tier_classification,
                            context_features = EXCLUDED.context_features,
                            actual_price_30d = EXCLUDED.actual_price_30d,
                            actual_price_90d = EXCLUDED.actual_price_90d,
                            reward_30d = EXCLUDED.reward_30d,
                            reward_90d = EXCLUDED.reward_90d,
                            per_model_rewards = EXCLUDED.per_model_rewards,
                            entry_date = EXCLUDED.entry_date,
                            exit_date_30d = EXCLUDED.exit_date_30d,
                            exit_date_90d = EXCLUDED.exit_date_90d,
                            outcome_updated_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """
                    ),
                    {
                        "symbol": symbol,
                        "analysis_date": analysis_date,
                        "fiscal_period": fiscal_period,
                        "blended_fair_value": blended_fair_value,
                        "current_price": current_price,
                        "predicted_upside_pct": predicted_upside_pct,
                        "dcf_fair_value": fair_values.get("dcf"),
                        "pe_fair_value": fair_values.get("pe"),
                        "ps_fair_value": fair_values.get("ps"),
                        "evebitda_fair_value": fair_values.get("ev_ebitda"),
                        "pb_fair_value": fair_values.get("pb"),
                        "ggm_fair_value": fair_values.get("ggm"),
                        "model_weights": safe_json_dumps(weights),
                        "tier_classification": tier_classification,
                        "context_features": safe_json_dumps(context_features.to_dict()),
                        "actual_price_30d": actual_price_30d,
                        "actual_price_90d": actual_price_90d,
                        "reward_30d": reward_30d,
                        "reward_90d": reward_90d,
                        "per_model_rewards": safe_json_dumps(per_model_rewards),
                        "position_type": position_type,
                        "entry_date": effective_entry_date,
                        "exit_date_30d": exit_date_30d,
                        "exit_date_90d": exit_date_90d,
                    },
                )
                row = result.fetchone()
                session.commit()
                return row[0] if row else None

        except Exception as e:
            logger.error(f"Error recording prediction for {symbol}: {e}")
            return None

    def calculate_position_rewards(
        self,
        current_price: float,
        actual_price_30d: Optional[float],
        actual_price_90d: Optional[float],
        beta: float = 1.0,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Calculate rewards for both LONG and SHORT positions.

        Returns rewards that are direction-aware:
        - LONG reward: positive if price went up
        - SHORT reward: positive if price went down
        """
        rewards = {
            "LONG": {"reward_30d": None, "reward_90d": None},
            "SHORT": {"reward_30d": None, "reward_90d": None},
        }

        if current_price <= 0:
            return rewards

        # Ensure beta is valid (positive) to avoid complex numbers from sqrt
        safe_beta = max(0.01, beta) if isinstance(beta, (int, float)) else 1.0

        # 30-day rewards
        if actual_price_30d and actual_price_30d > 0:
            price_return = (actual_price_30d - current_price) / current_price
            # Annualize and risk-adjust
            annualized = price_return * (365 / 30)

            # LONG: positive when price goes up
            long_raw = annualized
            if long_raw >= 0:
                rewards["LONG"]["reward_30d"] = long_raw / max(0.5, safe_beta ** 0.5)
            else:
                rewards["LONG"]["reward_30d"] = long_raw * max(1.0, safe_beta ** 0.75)

            # SHORT: positive when price goes down (inverse of LONG)
            short_raw = -annualized
            if short_raw >= 0:
                rewards["SHORT"]["reward_30d"] = short_raw / max(0.5, safe_beta ** 0.5)
            else:
                rewards["SHORT"]["reward_30d"] = short_raw * max(1.0, safe_beta ** 0.75)

        # 90-day rewards
        if actual_price_90d and actual_price_90d > 0:
            price_return = (actual_price_90d - current_price) / current_price
            annualized = price_return * (365 / 90)

            # LONG
            long_raw = annualized
            if long_raw >= 0:
                rewards["LONG"]["reward_90d"] = long_raw / max(0.5, safe_beta ** 0.5)
            else:
                rewards["LONG"]["reward_90d"] = long_raw * max(1.0, safe_beta ** 0.75)

            # SHORT
            short_raw = -annualized
            if short_raw >= 0:
                rewards["SHORT"]["reward_90d"] = short_raw / max(0.5, safe_beta ** 0.5)
            else:
                rewards["SHORT"]["reward_90d"] = short_raw * max(1.0, safe_beta ** 0.75)

        # Clamp rewards to [-1, 1]
        for pos in rewards:
            for key in rewards[pos]:
                if rewards[pos][key] is not None:
                    rewards[pos][key] = max(-1.0, min(1.0, rewards[pos][key]))

        return rewards

    def calculate_multi_period_rewards(
        self,
        current_price: float,
        future_prices: Dict[str, Optional[float]],
        beta: float = 1.0,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Calculate rewards for both LONG and SHORT positions across multiple holding periods.

        Args:
            current_price: Price at prediction time
            future_prices: Dict of period -> price, e.g. {"1m": 150.0, "3m": 160.0, ...}
            beta: Stock beta for risk adjustment

        Returns:
            Dict with structure: {
                "LONG": {"1m": 0.15, "3m": 0.22, "6m": 0.35, ...},
                "SHORT": {"1m": -0.15, "3m": -0.22, "6m": -0.35, ...}
            }
        """
        rewards = {
            "LONG": {period: None for period in HOLDING_PERIODS},
            "SHORT": {period: None for period in HOLDING_PERIODS},
        }

        if current_price <= 0:
            return rewards

        # Ensure beta is valid (positive) to avoid complex numbers from sqrt
        safe_beta = max(0.01, beta) if isinstance(beta, (int, float)) else 1.0

        for period, days in HOLDING_PERIODS.items():
            actual_price = future_prices.get(period)
            if not actual_price or actual_price <= 0:
                continue

            price_return = (actual_price - current_price) / current_price
            annualized = price_return * (365 / days)

            # LONG: positive when price goes up
            long_raw = annualized
            if long_raw >= 0:
                rewards["LONG"][period] = long_raw / max(0.5, safe_beta ** 0.5)
            else:
                rewards["LONG"][period] = long_raw * max(1.0, safe_beta ** 0.75)

            # SHORT: positive when price goes down
            short_raw = -annualized
            if short_raw >= 0:
                rewards["SHORT"][period] = short_raw / max(0.5, safe_beta ** 0.5)
            else:
                rewards["SHORT"][period] = short_raw * max(1.0, safe_beta ** 0.75)

        # Clamp rewards to [-1, 1]
        for pos in rewards:
            for period in rewards[pos]:
                if rewards[pos][period] is not None:
                    rewards[pos][period] = round(max(-1.0, min(1.0, rewards[pos][period])), 4)

        return rewards

    def get_multi_period_prices(
        self,
        symbol: str,
        analysis_date: date,
    ) -> Dict[str, Optional[float]]:
        """Fetch future prices for all holding periods.

        Args:
            symbol: Stock symbol
            analysis_date: Date of the prediction

        Returns:
            Dict of period -> price, e.g. {"1m": 150.0, "3m": 160.0, ...}
        """
        prices = {}
        for period, days in HOLDING_PERIODS.items():
            target_date = analysis_date + timedelta(days=days)
            prices[period] = self.get_historical_price(symbol, target_date)
        return prices

    def run_backtest_for_symbol(
        self,
        symbol: str,
        lookback_months: List[int],
    ) -> Dict[str, Any]:
        """Run backtest for a single symbol at multiple lookback periods."""
        results = {
            "symbol": symbol,
            "predictions": [],
            "errors": [],
        }

        today = date.today()
        metadata = self.get_symbol_metadata(symbol)

        # Pre-compute split-adjusted shares for all lookback periods
        shares_df = self.get_shares_history_for_normalization(symbol, lookback_months)
        shares_lookup = {}
        if not shares_df.empty:
            shares_lookup = dict(zip(shares_df["months_back"], shares_df["adjusted_shares"]))
            # Log if splits were detected
            if (shares_df["split_factor"] != 1.0).any():
                logger.info(
                    f"{symbol}: Split-adjusted shares: {shares_df[['months_back', 'raw_shares', 'split_factor', 'adjusted_shares']].to_dict('records')}"
                )

        for months_back in lookback_months:
            analysis_date = today - relativedelta(months=months_back)

            try:
                # Get historical data as of analysis_date
                financials = self.get_historical_financials(symbol, analysis_date)
                if not financials:
                    results["errors"].append(f"{months_back}m: No financial data")
                    continue

                # Get price on analysis_date
                price_at_prediction = self.get_historical_price(symbol, analysis_date)
                if not price_at_prediction:
                    results["errors"].append(f"{months_back}m: No price data")
                    continue

                # Calculate ratios (use split-adjusted shares if available)
                adjusted_shares = shares_lookup.get(months_back)
                ratios = self.calculate_ratios(
                    financials, price_at_prediction, metadata, symbol=symbol, adjusted_shares=adjusted_shares
                )

                # Fetch structured quarterly metrics for DCF/GGM (FULL framework)
                self._current_quarterly_metrics = self.get_quarterly_metrics_structured(
                    symbol, analysis_date, num_quarters=12
                )
                self._current_multi_year_data = self.get_multi_year_data(symbol, analysis_date, num_years=5)

                # Calculate fair values using FULL framework
                fair_values, tier, audit = self.calculate_fair_values_full_framework(
                    symbol=symbol,
                    financials=financials,
                    ratios=ratios,
                    metadata=metadata,
                    current_price=price_at_prediction,
                )

                # Get weights from audit
                weights = audit.get("weights", {})
                if not weights:
                    # Fallback to weighting service
                    financials["sector"] = metadata.get("sector")
                    financials["industry"] = metadata.get("industry")
                    financials["market_cap"] = ratios.get("market_cap", 0)
                    weights, tier, _ = self.weighting_service.determine_weights(
                        symbol=symbol,
                        financials=financials,
                        ratios=ratios,
                    )

                # Calculate blended fair value (with tier-aware sanity caps to prevent extreme outliers)
                blended_fv = self.calculate_blended_fair_value(
                    fair_values, weights, current_price=price_at_prediction, tier=tier
                )

                if blended_fv <= 0:
                    results["errors"].append(f"{months_back}m: Could not calculate fair value")
                    continue

                # Calculate position signal: Long=1, Short=-1, Skip=0
                valuation_gap = (blended_fv - price_at_prediction) / price_at_prediction if price_at_prediction > 0 else 0

                # Calculate valuation confidence based on model agreement
                # Filter out complex numbers (from sqrt of negative in growth calculations)
                valid_fvs = [fv for fv in fair_values.values() if fv and isinstance(fv, (int, float)) and fv > 0]
                if len(valid_fvs) >= 2:
                    # How many models agree on direction (above or below current price)?
                    above_price = sum(1 for fv in valid_fvs if fv > price_at_prediction)
                    below_price = len(valid_fvs) - above_price
                    valuation_confidence = max(above_price, below_price) / len(valid_fvs)
                else:
                    valuation_confidence = 0.5  # Single model, moderate confidence

                # Determine position signal with skip logic
                MIN_GAP_FOR_SIGNAL = 0.05  # 5% minimum gap
                MIN_CONFIDENCE = 0.6  # 60% model agreement

                if abs(valuation_gap) < MIN_GAP_FOR_SIGNAL:
                    position_signal = 0  # Skip: gap too small
                elif valuation_confidence < MIN_CONFIDENCE:
                    position_signal = 0  # Skip: models disagree
                elif valuation_gap > 0:
                    position_signal = 1  # Long: FV > Price
                else:
                    position_signal = -1  # Short: FV < Price

                # Get actual prices for all holding periods
                multi_period_prices = self.get_multi_period_prices(symbol, analysis_date)
                actual_price_30d = multi_period_prices.get("1m")
                actual_price_90d = multi_period_prices.get("3m")

                # Get beta for risk-adjusted reward calculation
                stock_beta = metadata.get("beta", 1.0)

                # Calculate multi-period rewards for LONG and SHORT positions
                multi_period_rewards = self.calculate_multi_period_rewards(
                    current_price=price_at_prediction,
                    future_prices=multi_period_prices,
                    beta=stock_beta,
                )

                # Calculate per-model rewards (direction-agnostic, based on fair value prediction)
                per_model_rewards = {}
                if actual_price_90d:
                    per_model_rewards = self.calculate_per_model_rewards(
                        fair_values, price_at_prediction, actual_price_90d, days=90, beta=stock_beta
                    )

                # Build exit dates for all holding periods
                exit_dates = {}
                for period, days in HOLDING_PERIODS.items():
                    exit_date = analysis_date + timedelta(days=days)
                    # Only include exit date if we have price data for that period
                    if multi_period_prices.get(period) is not None:
                        exit_dates[period] = exit_date.isoformat()

                # Add multi-period data to per_model_rewards for storage (JSONB is source of truth)
                per_model_rewards["multi_period"] = {
                    "entry_date": analysis_date.isoformat(),
                    "prices": {k: round(v, 2) if v else None for k, v in multi_period_prices.items()},
                    "exit_dates": exit_dates,
                    "long_rewards": multi_period_rewards["LONG"],
                    "short_rewards": multi_period_rewards["SHORT"],
                }

                # Fetch insider sentiment data for this symbol and date
                insider_data = self.fetch_insider_data_sync(symbol, analysis_date)

                # Fetch economic indicators (Regional Fed + CBOE) for this date
                economic_indicators = self.fetch_economic_indicators_sync(analysis_date)

                # Build context features (includes technical indicators, entry/exit signals, and economic data)
                context_features = self.build_context_features(
                    symbol=symbol,
                    financials=financials,
                    ratios=ratios,
                    metadata=metadata,
                    current_price=price_at_prediction,
                    analysis_date=analysis_date,
                    fair_value=blended_fv,  # Pass fair value for entry/exit signal calculation
                    valuation_gap=valuation_gap,
                    valuation_confidence=valuation_confidence,
                    position_signal=position_signal,
                    insider_data=insider_data,
                    economic_indicators=economic_indicators,
                )

                # Calculate DUAL-POSITION rewards for balanced training
                position_rewards = self.calculate_position_rewards(
                    current_price=price_at_prediction,
                    actual_price_30d=actual_price_30d,
                    actual_price_90d=actual_price_90d,
                    beta=stock_beta,
                )

                # Calculate exit dates for position tracking
                exit_date_30d = analysis_date + timedelta(days=30) if actual_price_30d else None
                exit_date_90d = analysis_date + timedelta(days=90) if actual_price_90d else None

                # Record BOTH LONG and SHORT positions for balanced RL training
                record_ids = []
                for position_type in ["LONG", "SHORT"]:
                    record_id = self.record_prediction(
                        symbol=symbol,
                        analysis_date=analysis_date,
                        fiscal_period=f"{financials.get('fiscal_year')}-{financials.get('fiscal_period')}",
                        blended_fair_value=blended_fv,
                        current_price=price_at_prediction,
                        fair_values=fair_values,
                        weights=weights,
                        tier_classification=tier,
                        context_features=context_features,
                        actual_price_30d=actual_price_30d,
                        actual_price_90d=actual_price_90d,
                        reward_30d=position_rewards[position_type]["reward_30d"],
                        reward_90d=position_rewards[position_type]["reward_90d"],
                        per_model_rewards=per_model_rewards,
                        position_type=position_type,
                        entry_date=analysis_date,
                        exit_date_30d=exit_date_30d,
                        exit_date_90d=exit_date_90d,
                    )
                    record_ids.append(record_id)

                # Get rewards for logging
                long_reward_90d = position_rewards["LONG"]["reward_90d"]
                short_reward_90d = position_rewards["SHORT"]["reward_90d"]

                # Position signal label (must be defined before results append)
                pos_label = {1: "LONG", -1: "SHORT", 0: "SKIP"}.get(position_signal, "?")

                # Find optimal holding period (highest reward across all periods)
                best_long_period, best_long_reward = None, -999
                best_short_period, best_short_reward = None, -999
                for period in HOLDING_PERIODS.keys():
                    l_rew = multi_period_rewards["LONG"].get(period)
                    s_rew = multi_period_rewards["SHORT"].get(period)
                    if l_rew is not None and l_rew > best_long_reward:
                        best_long_reward = l_rew
                        best_long_period = period
                    if s_rew is not None and s_rew > best_short_reward:
                        best_short_reward = s_rew
                        best_short_period = period

                # Determine optimal position and holding period
                if best_long_reward > best_short_reward and best_long_reward > 0:
                    optimal_position = "LONG"
                    optimal_period = best_long_period
                    optimal_reward = best_long_reward
                elif best_short_reward > 0:
                    optimal_position = "SHORT"
                    optimal_period = best_short_period
                    optimal_reward = best_short_reward
                else:
                    optimal_position = "SKIP"
                    optimal_period = "N/A"
                    optimal_reward = 0.0

                results["predictions"].append(
                    {
                        "lookback_months": months_back,
                        "analysis_date": str(analysis_date),
                        "record_ids": record_ids,  # Now a list of [LONG_id, SHORT_id]
                        "tier": tier,
                        "models_used": audit.get("models_used", []),
                        "sector_specific": audit.get("sector_specific", False),
                        "blended_fv": round(blended_fv, 2),
                        "price_at_pred": round(price_at_prediction, 2),
                        "actual_90d": round(actual_price_90d, 2) if actual_price_90d else None,
                        "long_reward_90d": round(long_reward_90d, 3) if long_reward_90d else None,
                        "short_reward_90d": round(short_reward_90d, 3) if short_reward_90d else None,
                        "upside_pct": round((blended_fv / price_at_prediction - 1) * 100, 1),
                        # Position signal
                        "position_signal": position_signal,
                        "position_signal_label": pos_label,
                        "valuation_gap": round(valuation_gap * 100, 1),
                        "valuation_confidence": round(valuation_confidence * 100, 0),
                        # Multi-period rewards (1m, 3m, 6m, 12m, 18m, 24m, 36m)
                        "multi_period_long": multi_period_rewards["LONG"],
                        "multi_period_short": multi_period_rewards["SHORT"],
                        # Optimal position and holding period (hindsight)
                        "optimal_position": optimal_position,
                        "optimal_holding_period": optimal_period,
                        "optimal_reward": round(optimal_reward, 3) if optimal_reward != 0 else 0,
                    }
                )

                long_str = f"{long_reward_90d:.3f}" if long_reward_90d is not None else "N/A"
                short_str = f"{short_reward_90d:.3f}" if short_reward_90d is not None else "N/A"

                logger.info(
                    f"{symbol} [{months_back}m back]: FV=${blended_fv:.2f}, "
                    f"Price=${price_at_prediction:.2f}, Gap={valuation_gap:+.1%}, "
                    f"Signal={pos_label}, Conf={valuation_confidence:.0%}, "
                    f"Reward(L/S)={long_str}/{short_str}, "
                    f"Optimal={optimal_position}@{optimal_period}({optimal_reward:+.3f})"
                )

            except Exception as e:
                logger.error(f"Error processing {symbol} at {months_back}m: {e}")
                results["errors"].append(f"{months_back}m: {str(e)}")

        return results

    def run_full_backtest(
        self,
        symbols: List[str],
        lookback_months: List[int],
        parallel: int = 1,
    ) -> Dict[str, Any]:
        """Run backtest for all symbols with optional parallelization."""
        logger.info(f"Starting FULL FRAMEWORK backtest for {len(symbols)} symbols at {lookback_months} month lookbacks")
        if parallel > 1:
            logger.info(f"Parallel mode: {parallel} symbols concurrently")

        all_results = {
            "symbols_processed": 0,
            "predictions_recorded": 0,
            "errors": 0,
            "summary_by_lookback": {},
            "symbol_results": [],
        }

        def process_symbol(args):
            """Process a single symbol (for parallel execution)."""
            idx, symbol, total = args
            try:
                logger.info(f"Processing [{idx+1}/{total}] {symbol}")
                result = self.run_backtest_for_symbol(symbol, lookback_months)
                return {"success": True, "symbol": symbol, "result": result}
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                return {"success": False, "symbol": symbol, "error": str(e)}

        if parallel <= 1:
            # Sequential processing (original behavior)
            for i, symbol in enumerate(symbols):
                outcome = process_symbol((i, symbol, len(symbols)))
                if outcome["success"]:
                    result = outcome["result"]
                    all_results["symbols_processed"] += 1
                    all_results["predictions_recorded"] += len(result["predictions"])
                    all_results["errors"] += len(result["errors"])
                    all_results["symbol_results"].append(result)
                    self._aggregate_predictions(all_results, result)
                else:
                    all_results["errors"] += 1
        else:
            # Parallel processing using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            work_items = [(i, symbol, len(symbols)) for i, symbol in enumerate(symbols)]

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(process_symbol, item): item for item in work_items}

                for future in as_completed(futures):
                    outcome = future.result()
                    if outcome["success"]:
                        result = outcome["result"]
                        all_results["symbols_processed"] += 1
                        all_results["predictions_recorded"] += len(result["predictions"])
                        all_results["errors"] += len(result["errors"])
                        all_results["symbol_results"].append(result)
                        self._aggregate_predictions(all_results, result)
                    else:
                        all_results["errors"] += 1

        # Calculate average rewards
        for months, data in all_results["summary_by_lookback"].items():
            if data["rewards"]:
                data["avg_reward"] = round(np.mean(data["rewards"]), 3)
            del data["rewards"]  # Remove list

        return all_results

    def _aggregate_predictions(self, all_results: Dict, result: Dict):
        """Aggregate predictions by lookback period."""
        for pred in result["predictions"]:
            months = pred["lookback_months"]
            if months not in all_results["summary_by_lookback"]:
                all_results["summary_by_lookback"][months] = {
                    "count": 0,
                    "rewards": [],
                    "avg_reward": 0,
                    "sector_specific_count": 0,
                }
            all_results["summary_by_lookback"][months]["count"] += 1
            reward = pred.get("reward_90d") or pred.get("reward")
            if reward is not None:
                all_results["summary_by_lookback"][months]["rewards"].append(reward)
            if pred.get("sector_specific"):
                all_results["summary_by_lookback"][months]["sector_specific_count"] += 1

    def _to_float(self, value) -> float:
        """Convert value to float."""
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def main():
    parser = argparse.ArgumentParser(description="RL Backtesting Script - Full Valuation Framework")

    # Symbol source options
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backtest")
    parser.add_argument("--russell1000", action="store_true", help="Backtest Russell 1000 symbols")
    parser.add_argument("--sp500", action="store_true", help="Backtest S&P 500 symbols")
    parser.add_argument("--all", action="store_true", help="Backtest ALL stocks from symbol table")
    parser.add_argument("--all-symbols", action="store_true", help="(Deprecated) Same as --all")
    parser.add_argument("--top-n", type=int, default=100, help="Top N symbols by market cap")
    parser.add_argument(
        "--order-by",
        choices=["stockid", "mktcap", "ticker"],
        default="stockid",
        help="Sort order: stockid (ascending), mktcap (descending), ticker (alphabetical). Default: stockid",
    )

    # Filtering options
    parser.add_argument(
        "--skip-domestic-filter",
        action="store_true",
        help="Skip the domestic filer filter (process all stocks even without SEC quarterly data)",
    )
    parser.add_argument(
        "--include-foreign",
        action="store_true",
        help="Include foreign stocks without SEC CIK (default: US only)",
    )
    parser.add_argument(
        "--lookback", nargs="+", type=int,
        default=list(RL_BACKTEST_PERIODS.standard_lookback_months),
        help=f"Lookback periods in months (default: {list(RL_BACKTEST_PERIODS.standard_lookback_months)})"
    )
    parser.add_argument(
        "--lookback-range",
        type=int,
        help="Generate lookback periods from 3 months to this value (e.g., 120 for 10 years)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["monthly", "quarterly"],
        default="quarterly",
        help="Interval for lookback-range: monthly (1mo) or quarterly (3mo)",
    )
    parser.add_argument(
        "--use-rl-policy",
        action="store_true",
        help="Use trained RL policy for weight determination (default: rule-based)",
    )
    parser.add_argument(
        "--rl-policy-path", type=str, default="data/rl_models/active_policy.pkl", help="Path to trained RL policy file"
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of symbols to process in parallel (default: 1 = sequential)"
    )
    parser.add_argument(
        "--skip-processed", type=str, help="Path to log file - skip symbols already processed in that log"
    )

    args = parser.parse_args()

    # --use-rl-policy now uses dual policy (technical + fundamental)
    args.use_dual_policy = args.use_rl_policy

    # Generate lookback periods from range if specified
    if args.lookback_range:
        interval_months = 1 if args.interval == "monthly" else 3
        # Generate lookback periods from 3 months to lookback_range
        # E.g., for 120 months quarterly: [3, 6, 9, 12, ..., 117, 120]
        args.lookback = list(range(interval_months, args.lookback_range + 1, interval_months))
        logger.info(
            f"Generated {len(args.lookback)} lookback periods ({args.interval}) "
            f"from {args.lookback[0]} to {args.lookback[-1]} months"
        )

    # Create logs directory
    import os

    os.makedirs("logs", exist_ok=True)

    print("Starting RL Backtest...", flush=True)
    print(f"  Lookback periods: {args.lookback}", flush=True)

    backtester = RLBacktester(
        use_rl_policy=args.use_rl_policy,
        rl_policy_path=args.rl_policy_path,
        use_dual_policy=args.use_dual_policy,
    )
    print("  Backtester initialized successfully", flush=True)

    # Get symbols based on source
    us_only = not getattr(args, 'include_foreign', False)
    if args.symbols:
        print(f"  Processing {len(args.symbols)} specified symbols", flush=True)
        symbols = [s.upper() for s in args.symbols]
    elif getattr(args, 'russell1000', False):
        print("  Fetching Russell 1000 symbols...", flush=True)
        symbols = backtester.get_russell1000_symbols()
        print(f"  Found {len(symbols)} Russell 1000 symbols", flush=True)
    elif getattr(args, 'sp500', False):
        print("  Fetching S&P 500 symbols...", flush=True)
        symbols = backtester.get_sp500_symbols()
        print(f"  Found {len(symbols)} S&P 500 symbols", flush=True)
    elif getattr(args, 'all', False) or args.all_symbols:
        print(f"  Fetching ALL stocks from symbol table (order: {args.order_by})...", flush=True)
        symbols = backtester.get_all_symbols(us_only=us_only, order_by=args.order_by)
        print(f"  Found {len(symbols)} symbols", flush=True)
    else:
        print(f"  Fetching top {args.top_n} symbols by market cap...", flush=True)
        symbols = backtester.get_symbols_to_backtest(top_n=args.top_n)
        print(f"  Found {len(symbols)} symbols", flush=True)

    # Filter domestic filers (unless skipped)
    skip_domestic = getattr(args, 'skip_domestic_filter', False)
    if not skip_domestic:
        print("  Checking for domestic filers...", flush=True)
        domestic_filers = backtester.get_domestic_filers()
        original_count = len(symbols)
        symbols = [s for s in symbols if s in domestic_filers]
        filtered_count = original_count - len(symbols)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} foreign filers -> {len(symbols)} remaining", flush=True)
    else:
        print("  Skipping domestic filer filter (--skip-domestic-filter)", flush=True)

    # Skip already-processed symbols if resuming
    if args.skip_processed and os.path.exists(args.skip_processed):
        import re
        with open(args.skip_processed, "r") as f:
            log_content = f.read()
        # Extract symbols from log: "INFO - AAPL ["
        processed = set(re.findall(r" - INFO - ([A-Z]+) \[", log_content))
        original_count = len(symbols)
        symbols = [s for s in symbols if s not in processed]
        print(f"  Resuming: skipping {len(processed)} already-processed symbols, {len(symbols)}/{original_count} remaining", flush=True)
        logger.info(f"Resuming: skipping {len(processed)} already-processed symbols, {len(symbols)}/{original_count} remaining")

    if args.use_dual_policy:
        weighting_mode = "Dual RL Policy (Technical + Fundamental)"
    elif args.use_rl_policy:
        weighting_mode = "RL Policy"
    else:
        weighting_mode = "Rule-Based"

    print("=" * 60, flush=True)
    print("RL BACKTEST RUNNER", flush=True)
    print("=" * 60, flush=True)
    print(f"  Total symbols: {len(symbols)}", flush=True)
    print(f"  Lookback periods: {args.lookback}", flush=True)
    print(f"  Weighting mode: {weighting_mode}", flush=True)
    print(f"  Parallel workers: {args.parallel}", flush=True)
    print("=" * 60, flush=True)

    logger.info(f"Backtesting {len(symbols)} symbols with lookbacks: {args.lookback}")
    logger.info(f"Weighting mode: {weighting_mode}")
    logger.info("Using FULL valuation framework (sector-specific, growth-adjusted, all models)")

    # Run backtest
    results = backtester.run_full_backtest(symbols, args.lookback, parallel=args.parallel)

    # Print summary
    print("\n" + "=" * 70)
    print(f"BACKTEST SUMMARY ({weighting_mode} Weights)")
    print("=" * 70)
    print(f"Symbols processed: {results['symbols_processed']}")
    print(f"Predictions recorded: {results['predictions_recorded']}")
    print(f"Errors: {results['errors']}")
    print("\nBy Lookback Period:")
    for months, data in sorted(results["summary_by_lookback"].items()):
        print(
            f"  {months} months back: {data['count']} predictions, "
            f"avg_reward={data['avg_reward']:.3f}, "
            f"sector_specific={data['sector_specific_count']}"
        )
    print("=" * 70)

    # Save results
    output_file = f"logs/backtest_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
