"""
Bank Industry Dataset

Implements industry-specific metrics extraction for banking companies.
Handles ROE-based P/B valuation methodology for financial institutions.

Covers:
- Diversified Banks (JPM, BAC, WFC, C)
- Regional Banks (USB, PNC, TFC)
- Investment Banks (GS, MS)

Author: Claude Code
Date: 2025-12-30
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from investigator.domain.services.industry_datasets.base import (
    BaseIndustryDataset,
    IndustryMetrics,
    MetricDefinition,
    MetricQuality,
    ValuationAdjustment,
)
from investigator.domain.services.industry_datasets.registry import register_dataset

logger = logging.getLogger(__name__)


# Known bank symbols
KNOWN_BANK_SYMBOLS = {
    # Diversified Banks
    "JPM",
    "BAC",
    "WFC",
    "C",
    # Regional Banks
    "USB",
    "PNC",
    "TFC",
    "FITB",
    "KEY",
    "RF",
    "CFG",
    "MTB",
    "HBAN",
    "ZION",
    # Investment Banks
    "GS",
    "MS",
    # Trust Banks
    "BK",
    "STT",
    "NTRS",
}


class BankDataset(BaseIndustryDataset):
    """
    Dataset for banking industry metrics extraction.

    Key metrics:
    - Net Interest Margin (NIM) - primary profitability driver
    - Tier 1 Capital Ratio - regulatory health
    - Efficiency Ratio - cost management
    - NPL Ratio - credit quality
    - ROE - drives P/B valuation

    Valuation approach:
    - P/B valuation driven by ROE vs cost of equity
    - Premium P/B for high-ROE banks
    - Discount for elevated NPL or weak capital
    """

    @property
    def name(self) -> str:
        return "bank"

    @property
    def display_name(self) -> str:
        return "Banking Industry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_industry_names(self) -> List[str]:
        return [
            "Banks",
            "Diversified Banks",
            "Regional Banks",
            "Money Center Banks",
            "Investment Banking & Brokerage",
            "Banks - Regional",
            "Banks - Diversified",
            "Banks - Money Center",
        ]

    def get_known_symbols(self) -> Set[str]:
        return KNOWN_BANK_SYMBOLS.copy()

    def get_metric_definitions(self) -> List[MetricDefinition]:
        return [
            MetricDefinition(
                name="net_interest_margin",
                display_name="Net Interest Margin",
                description="Net interest income as % of average earning assets",
                xbrl_tags=[
                    "NetInterestMargin",
                    "InterestMarginNet",
                    "NetInterestIncomeToEarningAssets",
                ],
                unit="percent",
                is_required=True,
                min_value=0.01,
                max_value=0.10,
            ),
            MetricDefinition(
                name="tier_1_capital_ratio",
                display_name="Tier 1 Capital Ratio",
                description="Core capital as % of risk-weighted assets",
                xbrl_tags=[
                    "Tier1CapitalRatio",
                    "CoreCapitalRatio",
                    "Tier1RiskBasedCapitalRatio",
                ],
                unit="percent",
                is_required=True,
                min_value=0.08,
                max_value=0.25,
            ),
            MetricDefinition(
                name="efficiency_ratio",
                display_name="Efficiency Ratio",
                description="Non-interest expense / revenue (lower is better)",
                xbrl_tags=[
                    "EfficiencyRatio",
                    "NonInterestExpenseToRevenue",
                ],
                unit="percent",
                is_required=False,
                min_value=0.30,
                max_value=0.90,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="npl_ratio",
                display_name="NPL Ratio",
                description="Non-performing loans as % of total loans",
                xbrl_tags=[
                    "NonPerformingLoansRatio",
                    "NonAccruingLoansRatio",
                    "NonperformingLoansToTotalLoans",
                ],
                unit="percent",
                is_required=True,
                min_value=0.0,
                max_value=0.10,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="roe",
                display_name="Return on Equity",
                description="Net income / average shareholders equity",
                xbrl_tags=["ReturnOnEquity", "ROE"],
                unit="percent",
                is_required=True,
                min_value=0.0,
                max_value=0.30,
            ),
            MetricDefinition(
                name="loan_to_deposit",
                display_name="Loan to Deposit Ratio",
                description="Total loans / total deposits",
                xbrl_tags=["LoanToDepositRatio", "LoansToDeposits"],
                unit="ratio",
                is_required=False,
                min_value=0.50,
                max_value=1.20,
            ),
        ]

    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """Extract bank-specific metrics from XBRL data and financials."""
        metrics = IndustryMetrics(
            industry="bank",
            symbol=symbol,
            metrics={},
            metadata={"source": "xbrl+financials"},
        )
        warnings = []

        # Extract Net Interest Margin
        nim = self._extract_from_xbrl(xbrl_data, "net_interest_margin", ["NetInterestMargin", "InterestMarginNet"])
        if nim:
            metrics.metrics["net_interest_margin"] = nim
        else:
            # Try to calculate from interest income
            nim = self._calculate_nim(financials)
            if nim:
                metrics.metrics["net_interest_margin"] = nim
            else:
                warnings.append("Could not extract/calculate NIM")

        # Extract Tier 1 Capital Ratio
        tier1 = self._extract_from_xbrl(
            xbrl_data, "tier_1_capital_ratio", ["Tier1CapitalRatio", "CoreCapitalRatio", "Tier1RiskBasedCapitalRatio"]
        )
        if tier1:
            metrics.metrics["tier_1_capital_ratio"] = tier1
        else:
            warnings.append("Tier 1 capital ratio not available")

        # Extract Efficiency Ratio
        efficiency = self._extract_from_xbrl(
            xbrl_data, "efficiency_ratio", ["EfficiencyRatio", "NonInterestExpenseToRevenue"]
        )
        if efficiency:
            metrics.metrics["efficiency_ratio"] = efficiency
        else:
            efficiency = self._calculate_efficiency_ratio(financials)
            if efficiency:
                metrics.metrics["efficiency_ratio"] = efficiency

        # Extract NPL Ratio
        npl = self._extract_from_xbrl(
            xbrl_data, "npl_ratio", ["NonPerformingLoansRatio", "NonperformingLoansToTotalLoans"]
        )
        if npl:
            metrics.metrics["npl_ratio"] = npl
        else:
            # Use industry average as fallback
            metrics.metrics["npl_ratio"] = 0.01
            warnings.append("NPL ratio not available, using industry average")

        # Extract ROE
        roe = financials.get("roe") or financials.get("returnOnEquity")
        if roe:
            metrics.metrics["roe"] = roe
        else:
            roe = self._calculate_roe(financials)
            if roe:
                metrics.metrics["roe"] = roe
            else:
                warnings.append("ROE not available")

        # Extract Loan to Deposit ratio
        ltd = self._extract_from_xbrl(xbrl_data, "loan_to_deposit", ["LoanToDepositRatio", "LoansToDeposits"])
        if ltd:
            metrics.metrics["loan_to_deposit"] = ltd

        # Determine bank type
        bank_type = self._determine_bank_type(symbol, financials)
        metrics.metrics["bank_type"] = bank_type
        metrics.metadata["bank_type"] = bank_type

        # Set warnings
        metrics.warnings = warnings

        # Assess quality
        quality, description = self.assess_quality(metrics)
        metrics.quality = quality
        metrics.metadata["quality_description"] = description

        # Calculate coverage
        definitions = self.get_metric_definitions()
        available = sum(1 for d in definitions if metrics.has(d.name))
        metrics.coverage = available / len(definitions) if definitions else 0.0

        return metrics

    def _calculate_nim(self, financials: Dict) -> Optional[float]:
        """Calculate NIM from available financial data."""
        net_interest = financials.get("netInterestIncome")
        earning_assets = financials.get("totalAssets")

        if net_interest and earning_assets and earning_assets > 0:
            return net_interest / earning_assets

        return None

    def _calculate_efficiency_ratio(self, financials: Dict) -> Optional[float]:
        """Calculate efficiency ratio."""
        non_interest_expense = financials.get("nonInterestExpense")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if non_interest_expense and revenue and revenue > 0:
            return non_interest_expense / revenue

        return None

    def _calculate_roe(self, financials: Dict) -> Optional[float]:
        """Calculate ROE from available data."""
        net_income = financials.get("netIncome")
        equity = financials.get("totalEquity") or financials.get("shareholdersEquity")

        if net_income and equity and equity > 0:
            return net_income / equity

        return None

    def _determine_bank_type(self, symbol: str, financials: Dict) -> str:
        """Determine bank type based on symbol and characteristics."""
        symbol_upper = symbol.upper()

        if symbol_upper in {"GS", "MS"}:
            return "investment"
        elif symbol_upper in {"JPM", "BAC", "WFC", "C"}:
            return "diversified"
        else:
            return "regional"

    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """Assess quality of bank metrics."""
        required_metrics = ["net_interest_margin", "tier_1_capital_ratio", "npl_ratio", "roe"]
        important_metrics = ["efficiency_ratio", "loan_to_deposit"]

        required_available = sum(1 for m in required_metrics if metrics.has(m))
        important_available = sum(1 for m in important_metrics if metrics.has(m))

        if required_available == len(required_metrics) and important_available >= 1:
            return (MetricQuality.EXCELLENT, "All key bank metrics available")
        elif required_available >= 3:
            return (MetricQuality.GOOD, "Most required bank metrics available")
        elif required_available >= 2:
            return (MetricQuality.FAIR, "Partial bank metrics available")
        else:
            return (MetricQuality.POOR, "Missing key metrics for bank valuation")

    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """Calculate bank-specific valuation adjustments."""
        adjustments = []

        # ROE-based P/B adjustment
        roe = metrics.get("roe")
        cost_of_equity = 0.10  # Assume 10% CoE

        if roe:
            if roe > cost_of_equity + 0.05:
                # High ROE deserves premium P/B
                premium = min((roe - cost_of_equity) * 2, 0.30)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.0 + premium,
                        reason=f"Superior ROE ({roe:.1%}) vs cost of equity ({cost_of_equity:.1%})",
                        confidence=0.8,
                        affects_models=["pb"],
                    )
                )
            elif roe < cost_of_equity - 0.02:
                # Low ROE deserves discount
                discount = min((cost_of_equity - roe) * 2, 0.20)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 - discount,
                        reason=f"Below cost-of-equity ROE ({roe:.1%})",
                        confidence=0.7,
                        affects_models=["pb"],
                    )
                )

        # Capital strength adjustment
        tier1 = metrics.get("tier_1_capital_ratio")
        if tier1:
            if tier1 > 0.14:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Strong capital position (Tier 1: {tier1:.1%})",
                        confidence=0.7,
                        affects_models=["pb", "pe"],
                    )
                )
            elif tier1 < 0.10:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.90,
                        reason=f"Weak capital position (Tier 1: {tier1:.1%})",
                        confidence=0.8,
                        affects_models=["pb", "pe"],
                    )
                )

        # Credit quality adjustment
        npl = metrics.get("npl_ratio")
        if npl:
            if npl > 0.03:
                # High NPLs - apply discount
                discount = min(npl * 3, 0.20)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 - discount,
                        reason=f"Elevated NPL ratio ({npl:.2%})",
                        confidence=0.8,
                        affects_models=["pb", "pe"],
                    )
                )
            elif npl < 0.005:
                # Very clean book
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.03,
                        reason=f"Excellent credit quality (NPL: {npl:.2%})",
                        confidence=0.7,
                        affects_models=["pb"],
                    )
                )

        # Efficiency adjustment
        efficiency = metrics.get("efficiency_ratio")
        if efficiency:
            if efficiency < 0.55:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Superior efficiency ratio ({efficiency:.1%})",
                        confidence=0.7,
                        affects_models=["pe"],
                    )
                )
            elif efficiency > 0.70:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"High efficiency ratio ({efficiency:.1%})",
                        confidence=0.6,
                        affects_models=["pe"],
                    )
                )

        return adjustments

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """Return recommended tier weights for banks."""
        return {
            "pb": 50,
            "pe": 30,
            "dcf": 10,
            "ev_ebitda": 0,  # Not applicable to banks
            "ps": 0,
            "ggm": 10,  # Dividend-based
        }


# Auto-register when module is imported
_dataset = BankDataset()
register_dataset(_dataset)
