"""
Company profiling utilities used by the multi-model valuation framework.

The profile aggregates observable characteristics about a company that
determine which valuation models are applicable and how they should be
weighted. Detection logic for archetypes will consume these data points
in subsequent phases; Phase 1 focuses on structuring the data contract.

Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.company_profile
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class CompanyArchetype(Enum):
    """High-level archetypes that influence valuation model selection."""

    HIGH_GROWTH = auto()
    MATURE_DIVIDEND = auto()
    CYCLICAL = auto()
    CAPITAL_INTENSIVE = auto()
    FINANCIAL = auto()
    TURNAROUND = auto()
    ASSET_HEAVY = auto()


class DataQualityFlag(Enum):
    """Flags surfaced when required inputs are missing or look suspicious."""

    MISSING_QUARTERS = auto()
    RESTATED_FINANCIALS = auto()
    NEGATIVE_DENOMINATOR = auto()
    OUTLIER_DETECTED = auto()
    STALE_REFERENCE_DATA = auto()
    LOW_LIQUIDITY = auto()
    INCOMPLETE_DIVIDEND_HISTORY = auto()


@dataclass
class CompanyProfile:
    """
    Classification snapshot for a company.

    Subsequent phases will populate this object via dedicated classifiers. For
    now we expose the contract so valuation models can depend on it without
    circular imports.
    """

    symbol: str
    sector: str
    industry: Optional[str] = None

    # Profitability indicators
    has_positive_fcf: Optional[bool] = None
    has_positive_earnings: Optional[bool] = None
    has_positive_ebitda: Optional[bool] = None

    # Cash flow metrics
    ttm_fcf: Optional[float] = None
    fcf_margin: Optional[float] = None  # Free cash flow / revenue
    fcf_volatility_cv: Optional[float] = None  # Coefficient of variation

    # Growth and quality signals
    revenue_cagr_3y: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    revenue_volatility: Optional[float] = None
    gross_margin_trend: Optional[float] = None
    gross_margin: Optional[float] = None  # Actual gross profit / revenue (for P/S quality premium)
    net_revenue_retention: Optional[float] = None  # NRR for SaaS companies (for P/S quality premium)
    ebitda_margin_trend: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    return_on_equity: Optional[float] = None
    earnings_quality_score: Optional[float] = None  # 0.0 - 1.0, accruals based

    # Capital structure & payout
    net_debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_payout_ratio: Optional[float] = None
    dividend_growth_rate: Optional[float] = None
    pays_dividends: Optional[bool] = None

    # Market behaviour
    beta: Optional[float] = None
    daily_liquidity_usd: Optional[float] = None

    # Balance sheet & valuation context
    book_value_per_share: Optional[float] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    cash: Optional[float] = None
    total_debt: Optional[float] = None

    # Absolute financial values for model applicability checks
    # (ModelApplicabilityRules needs actual values, not just booleans)
    free_cash_flow: Optional[float] = None  # TTM free cash flow in dollars
    ebitda: Optional[float] = None  # TTM EBITDA in dollars
    net_income: Optional[float] = None  # TTM net income in dollars
    revenue: Optional[float] = None  # TTM revenue in dollars
    dividends_paid: Optional[float] = None  # TTM dividends paid in dollars (absolute)

    # Rule of 40 diagnostics
    rule_of_40_score: Optional[float] = None
    rule_of_40_classification: Optional[str] = None

    # Derived attributes
    primary_archetype: Optional[CompanyArchetype] = None
    secondary_archetype: Optional[CompanyArchetype] = None
    data_quality_flags: List[DataQualityFlag] = field(default_factory=list)

    # Data completeness
    quarters_available: Optional[int] = None
    data_completeness_score: Optional[float] = None  # 0.0 - 1.0

    def add_flag(self, flag: DataQualityFlag) -> None:
        """Attach a new data-quality flag if it's not already present."""
        if flag not in self.data_quality_flags:
            self.data_quality_flags.append(flag)

    def has_flag(self, flag: DataQualityFlag) -> bool:
        """Convenience helper for checking whether a flag is active."""
        return flag in self.data_quality_flags

    def archetype_labels(self) -> List[str]:
        """Return the human-readable archetype labels currently assigned."""
        labels: List[str] = []
        if self.primary_archetype:
            labels.append(self.primary_archetype.name)
        if self.secondary_archetype:
            labels.append(self.secondary_archetype.name)
        return labels
