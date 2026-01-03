# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Fair Value Service - Unified fair value calculation orchestration.

This service provides a consistent interface for calculating fair values across
all valuation consumers. It uses config-driven sector multiples and delegates
sector-specific logic to specialized modules.

Key features:
- Multi-model valuation (PE, PS, PB, EV/EBITDA, DCF, GGM)
- Config-driven sector multiples
- Blended fair value calculation with configurable weights
- Model applicability detection
- Audit trail for transparency

Example:
    from investigator.domain.services.valuation_shared import FairValueService

    service = FairValueService()

    # Calculate fair values
    fair_values = service.calculate_fair_values(
        financials=ttm_data,
        ratios=ratios,
        sector="Technology",
        current_price=150.0,
        shares=1_000_000_000,
    )

    # Get blended fair value
    blended = service.calculate_blended_fair_value(
        fair_values=fair_values,
        weights={"dcf": 40, "pe": 30, "ps": 20, "ev_ebitda": 10},
    )
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .valuation_config_service import ValuationConfigService
from .sector_multiples_service import SectorMultiplesService

logger = logging.getLogger(__name__)


class FairValueService:
    """
    Unified service for fair value calculation.

    Orchestrates valuation model calculations and provides blended fair values.
    """

    def __init__(
        self,
        config_service: Optional[ValuationConfigService] = None,
        multiples_service: Optional[SectorMultiplesService] = None,
    ):
        """
        Initialize FairValueService.

        Args:
            config_service: ValuationConfigService instance
            multiples_service: SectorMultiplesService instance
        """
        self._config = config_service or ValuationConfigService()
        self._multiples = multiples_service or SectorMultiplesService(self._config)

    def calculate_pe_fair_value(
        self,
        eps: float,
        sector: str,
        industry: Optional[str] = None,
        pe_multiple: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate fair value using P/E model.

        Fair Value = EPS × Sector P/E Multiple

        Args:
            eps: Earnings per share (TTM)
            sector: Company sector
            industry: Optional industry for more specific multiple
            pe_multiple: Override P/E multiple (uses config if not provided)

        Returns:
            Fair value per share or None if not calculable
        """
        if eps is None or eps <= 0:
            return None

        if pe_multiple is None:
            pe_multiple = self._multiples.get_pe(sector, industry)

        return eps * pe_multiple

    def calculate_ps_fair_value(
        self,
        revenue_per_share: float,
        sector: str,
        industry: Optional[str] = None,
        ps_multiple: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate fair value using P/S model.

        Fair Value = Revenue Per Share × Sector P/S Multiple

        Args:
            revenue_per_share: Revenue per share (TTM)
            sector: Company sector
            industry: Optional industry
            ps_multiple: Override P/S multiple

        Returns:
            Fair value per share or None if not calculable
        """
        if revenue_per_share is None or revenue_per_share <= 0:
            return None

        if ps_multiple is None:
            ps_multiple = self._multiples.get_ps(sector, industry)

        return revenue_per_share * ps_multiple

    def calculate_pb_fair_value(
        self,
        book_value_per_share: float,
        sector: str,
        industry: Optional[str] = None,
        pb_multiple: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate fair value using P/B model.

        Fair Value = Book Value Per Share × Sector P/B Multiple

        Args:
            book_value_per_share: Book value per share
            sector: Company sector
            industry: Optional industry
            pb_multiple: Override P/B multiple

        Returns:
            Fair value per share or None if not calculable
        """
        if book_value_per_share is None or book_value_per_share <= 0:
            return None

        if pb_multiple is None:
            pb_multiple = self._multiples.get_pb(sector, industry)

        return book_value_per_share * pb_multiple

    def calculate_ev_ebitda_fair_value(
        self,
        ebitda: float,
        shares: float,
        total_debt: float = 0,
        cash: float = 0,
        sector: str = "Unknown",
        industry: Optional[str] = None,
        ev_ebitda_multiple: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate fair value using EV/EBITDA model.

        Fair EV = EBITDA × Sector EV/EBITDA Multiple
        Fair Market Cap = Fair EV - Debt + Cash
        Fair Value Per Share = Fair Market Cap / Shares

        Args:
            ebitda: TTM EBITDA
            shares: Shares outstanding
            total_debt: Total debt
            cash: Cash and equivalents
            sector: Company sector
            industry: Optional industry
            ev_ebitda_multiple: Override EV/EBITDA multiple

        Returns:
            Fair value per share or None if not calculable
        """
        if ebitda is None or ebitda <= 0 or shares <= 0:
            return None

        if ev_ebitda_multiple is None:
            ev_ebitda_multiple = self._multiples.get_ev_ebitda(sector, industry)

        fair_ev = ebitda * ev_ebitda_multiple
        fair_market_cap = fair_ev - total_debt + cash

        if fair_market_cap <= 0:
            return None

        return fair_market_cap / shares

    def calculate_ggm_fair_value(
        self,
        dividend_per_share: float,
        growth_rate: Optional[float] = None,
        cost_of_equity: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate fair value using Gordon Growth Model.

        Fair Value = D1 / (r - g)
        where D1 = DPS × (1 + g)

        Args:
            dividend_per_share: Current annual dividend per share
            growth_rate: Expected perpetual growth rate (default from config)
            cost_of_equity: Required rate of return (default from config)

        Returns:
            Fair value per share or None if not calculable
        """
        if dividend_per_share is None or dividend_per_share <= 0:
            return None

        ggm_defaults = self._config.get_ggm_defaults()

        if growth_rate is None:
            growth_rate = ggm_defaults.get("growth_rate", 0.03)

        if cost_of_equity is None:
            cost_of_equity = ggm_defaults.get("cost_of_equity", 0.08)

        # Ensure cost of equity > growth rate
        if cost_of_equity <= growth_rate:
            return None

        d1 = dividend_per_share * (1 + growth_rate)
        return d1 / (cost_of_equity - growth_rate)

    def calculate_fair_values(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        sector: str,
        current_price: float,
        shares: float,
        industry: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Calculate fair values using all applicable models.

        Args:
            financials: TTM financial data (nested or flat)
            ratios: Calculated ratios dict
            sector: Company sector
            current_price: Current stock price
            shares: Shares outstanding
            industry: Optional industry

        Returns:
            Dict mapping model name to fair value per share
        """
        fair_values = {}

        # Extract values from nested or flat structure
        if "income_statement" in financials:
            income = financials.get("income_statement", {})
            cash_flow = financials.get("cash_flow", {})
            balance = financials.get("balance_sheet", {})

            revenue = income.get("total_revenue", 0) or 0
            net_income = income.get("net_income", 0) or 0
            ebitda = income.get("ebitda", 0) or 0

            dividends = abs(cash_flow.get("dividends_paid", 0) or 0)

            equity = balance.get("stockholders_equity", 0) or 0
            total_debt = (balance.get("long_term_debt", 0) or 0) + (balance.get("short_term_debt", 0) or 0)
            cash = balance.get("cash_and_equivalents", 0) or 0
        else:
            revenue = financials.get("total_revenue", 0) or 0
            net_income = financials.get("net_income", 0) or 0
            ebitda = financials.get("ebitda", 0) or 0
            dividends = abs(financials.get("dividends_paid", 0) or 0)
            equity = financials.get("stockholders_equity", 0) or 0
            total_debt = (financials.get("long_term_debt", 0) or 0) + (financials.get("short_term_debt", 0) or 0)
            cash = financials.get("cash_and_equivalents", 0) or 0

        # Per-share metrics
        eps = ratios.get("ttm_eps") or (net_income / shares if shares > 0 else 0)
        bvps = ratios.get("book_value_per_share") or (equity / shares if shares > 0 else 0)
        revenue_per_share = revenue / shares if shares > 0 else 0
        dps = dividends / shares if shares > 0 else 0

        # P/E Fair Value
        fair_values["pe"] = self.calculate_pe_fair_value(eps, sector, industry)

        # P/S Fair Value
        fair_values["ps"] = self.calculate_ps_fair_value(revenue_per_share, sector, industry)

        # P/B Fair Value
        fair_values["pb"] = self.calculate_pb_fair_value(bvps, sector, industry)

        # EV/EBITDA Fair Value
        fair_values["ev_ebitda"] = self.calculate_ev_ebitda_fair_value(
            ebitda=ebitda,
            shares=shares,
            total_debt=total_debt,
            cash=cash,
            sector=sector,
            industry=industry,
        )

        # GGM Fair Value (if applicable)
        if dps > 0:
            fair_values["ggm"] = self.calculate_ggm_fair_value(dps)
        else:
            fair_values["ggm"] = None

        return fair_values

    def calculate_blended_fair_value(
        self,
        fair_values: Dict[str, Optional[float]],
        weights: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate weighted blended fair value.

        Args:
            fair_values: Dict mapping model name to fair value
            weights: Dict mapping model name to weight (0-100 or 0.0-1.0)

        Returns:
            Tuple of (blended fair value, audit dict)
        """
        total_weight = 0
        weighted_sum = 0
        models_used = []

        # Normalize weights if they sum to 100
        weight_sum = sum(weights.values())
        normalize = weight_sum > 1 and weight_sum <= 100

        for model, weight in weights.items():
            fair_value = fair_values.get(model)

            if fair_value is not None and fair_value > 0 and weight > 0:
                # Normalize weight if needed
                w = weight / 100 if normalize else weight

                weighted_sum += fair_value * w
                total_weight += w
                models_used.append({
                    "model": model,
                    "fair_value": fair_value,
                    "weight": w,
                    "contribution": fair_value * w,
                })

        if total_weight > 0:
            blended = weighted_sum / total_weight
        else:
            blended = 0

        audit = {
            "blended_fair_value": blended,
            "total_weight": total_weight,
            "models_used": models_used,
            "models_skipped": [m for m in weights if fair_values.get(m) is None or fair_values.get(m) <= 0],
        }

        return blended, audit

    def get_model_applicability(
        self,
        financials: Dict[str, Any],
        sector: str,
    ) -> Dict[str, bool]:
        """
        Determine which valuation models are applicable.

        Args:
            financials: TTM financial data
            sector: Company sector

        Returns:
            Dict mapping model name to applicability boolean
        """
        # Extract values
        if "income_statement" in financials:
            income = financials.get("income_statement", {})
            cash_flow = financials.get("cash_flow", {})
            balance = financials.get("balance_sheet", {})

            net_income = income.get("net_income", 0) or 0
            revenue = income.get("total_revenue", 0) or 0
            ebitda = income.get("ebitda", 0) or 0
            fcf = cash_flow.get("free_cash_flow", 0) or 0
            dividends = abs(cash_flow.get("dividends_paid", 0) or 0)
            equity = balance.get("stockholders_equity", 0) or 0
        else:
            net_income = financials.get("net_income", 0) or 0
            revenue = financials.get("total_revenue", 0) or 0
            ebitda = financials.get("ebitda", 0) or 0
            fcf = financials.get("free_cash_flow", 0) or 0
            dividends = abs(financials.get("dividends_paid", 0) or 0)
            equity = financials.get("stockholders_equity", 0) or 0

        # Calculate payout ratio for GGM
        payout_ratio = dividends / net_income if net_income > 0 else 0
        min_payout = self._config.get_ggm_min_payout_ratio()

        return {
            "pe": net_income > 0,
            "ps": revenue > 0,
            "pb": equity > 0,
            "ev_ebitda": ebitda > 0,
            "dcf": fcf != 0,  # Can handle negative FCF with revenue bridge
            "ggm": dividends > 0 and net_income > 0 and payout_ratio >= min_payout,
        }

    def get_valuation_summary(
        self,
        fair_values: Dict[str, Optional[float]],
        current_price: float,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Generate a valuation summary with upside/downside analysis.

        Args:
            fair_values: Dict mapping model name to fair value
            current_price: Current stock price
            weights: Dict mapping model name to weight

        Returns:
            Summary dict with blended value, upside, and model breakdown
        """
        blended, audit = self.calculate_blended_fair_value(fair_values, weights)

        upside_pct = ((blended / current_price) - 1) * 100 if current_price > 0 else 0

        # Per-model upside
        model_upsides = {}
        for model, fv in fair_values.items():
            if fv is not None and fv > 0:
                model_upsides[model] = ((fv / current_price) - 1) * 100

        return {
            "current_price": current_price,
            "blended_fair_value": blended,
            "upside_pct": upside_pct,
            "recommendation": self._get_recommendation(upside_pct),
            "fair_values": fair_values,
            "model_upsides": model_upsides,
            "weights_applied": weights,
            "audit": audit,
        }

    def _get_recommendation(self, upside_pct: float) -> str:
        """Get buy/hold/sell recommendation based on upside."""
        if upside_pct >= 20:
            return "STRONG_BUY"
        elif upside_pct >= 10:
            return "BUY"
        elif upside_pct >= -10:
            return "HOLD"
        elif upside_pct >= -20:
            return "SELL"
        else:
            return "STRONG_SELL"
