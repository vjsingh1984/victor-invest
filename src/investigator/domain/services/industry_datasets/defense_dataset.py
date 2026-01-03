"""
Defense Contractor Industry Dataset

Implements industry-specific metrics extraction for aerospace & defense companies.
Handles backlog-adjusted valuations and contract visibility analysis.

Covers:
- Prime Contractors (LMT, RTX, NOC, GD, BA)
- Defense Electronics (LHX, LDOS)
- Shipbuilding (HII)
- Services (CACI, SAIC)

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


# Known defense contractor symbols
KNOWN_DEFENSE_SYMBOLS = {
    # Prime Contractors
    "LMT",
    "RTX",
    "NOC",
    "GD",
    "BA",
    # Defense Electronics
    "LHX",
    "LDOS",
    # Shipbuilding
    "HII",
    # Defense Services
    "CACI",
    "SAIC",
    "BAH",
    # Missiles/Space
    "AXON",
    "KTOS",
}


class DefenseDataset(BaseIndustryDataset):
    """
    Dataset for defense industry metrics extraction.

    Key metrics:
    - Order Backlog (contract visibility)
    - Backlog-to-Revenue Ratio (years of visibility)
    - Book-to-Bill Ratio (new orders vs revenue)
    - Operating Margin (program execution)
    - Free Cash Flow Conversion

    Valuation approach:
    - Backlog-adjusted DCF (visibility premium)
    - PE with margin adjustment
    - Premium for high backlog coverage
    """

    @property
    def name(self) -> str:
        return "defense"

    @property
    def display_name(self) -> str:
        return "Defense Contractor Industry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_industry_names(self) -> List[str]:
        return [
            "Aerospace & Defense",
            "Defense",
            "Aerospace/Defense",
            "Aerospace & Defense - Major",
            "Defense Contractors",
            "Defense Electronics",
            "Aerospace - Defense",
            "Defense Primes",
        ]

    def get_known_symbols(self) -> Set[str]:
        return KNOWN_DEFENSE_SYMBOLS.copy()

    def get_metric_definitions(self) -> List[MetricDefinition]:
        return [
            MetricDefinition(
                name="order_backlog",
                display_name="Order Backlog",
                description="Total unfulfilled contract value",
                xbrl_tags=[
                    "OrderBacklog",
                    "ContractBacklog",
                    "UnfulfilledContractOrders",
                    "RemainingPerformanceObligation",
                ],
                unit="USD",
                is_required=True,
            ),
            MetricDefinition(
                name="backlog_to_revenue",
                display_name="Backlog to Revenue Ratio",
                description="Years of backlog coverage",
                xbrl_tags=["BacklogToRevenueRatio"],
                unit="ratio",
                is_required=True,
                min_value=1.0,
                max_value=8.0,
            ),
            MetricDefinition(
                name="book_to_bill",
                display_name="Book-to-Bill Ratio",
                description="New orders divided by revenue",
                xbrl_tags=[
                    "BookToOrderRatio",
                    "OrdersToRevenue",
                ],
                unit="ratio",
                is_required=False,
                default_value=1.0,
                min_value=0.6,
                max_value=2.0,
            ),
            MetricDefinition(
                name="operating_margin",
                display_name="Operating Margin",
                description="Operating income as % of revenue",
                xbrl_tags=["OperatingIncomeMargin", "OperatingMargin"],
                unit="percent",
                is_required=True,
                min_value=0.05,
                max_value=0.20,
            ),
            MetricDefinition(
                name="fcf_conversion",
                display_name="FCF Conversion",
                description="Free cash flow as % of net income",
                xbrl_tags=["FreeCashFlowConversion"],
                unit="percent",
                is_required=False,
                min_value=0.50,
                max_value=1.50,
            ),
            MetricDefinition(
                name="international_revenue_pct",
                display_name="International Revenue %",
                description="Percentage of revenue from international sales",
                xbrl_tags=[
                    "InternationalRevenuePct",
                    "ForeignRevenuePct",
                ],
                unit="percent",
                is_required=False,
                min_value=0.0,
                max_value=0.60,
            ),
            MetricDefinition(
                name="rd_to_revenue",
                display_name="R&D to Revenue",
                description="R&D spending as percentage of revenue",
                xbrl_tags=["ResearchAndDevelopmentExpenseToRevenue"],
                unit="percent",
                is_required=False,
                min_value=0.01,
                max_value=0.10,
            ),
        ]

    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """Extract defense-specific metrics from XBRL data and financials."""
        metrics = IndustryMetrics(
            industry="defense",
            symbol=symbol,
            metrics={},
            metadata={"source": "xbrl+financials"},
        )
        warnings = []

        # Extract Order Backlog
        backlog = self._extract_from_xbrl(
            xbrl_data, "order_backlog", ["OrderBacklog", "ContractBacklog", "RemainingPerformanceObligation"]
        )
        if backlog:
            metrics.metrics["order_backlog"] = backlog
        else:
            warnings.append("Order backlog not available in filings")

        # Calculate Backlog-to-Revenue Ratio
        revenue = financials.get("revenue") or financials.get("totalRevenue")
        if backlog and revenue and revenue > 0:
            backlog_ratio = backlog / revenue
            metrics.metrics["backlog_to_revenue"] = backlog_ratio
        else:
            # Use industry typical
            metrics.metrics["backlog_to_revenue"] = 2.5
            warnings.append("Backlog-to-revenue estimated at 2.5x")

        # Extract Book-to-Bill
        btb = self._extract_from_xbrl(xbrl_data, "book_to_bill", ["BookToOrderRatio", "OrdersToRevenue"])
        if btb:
            metrics.metrics["book_to_bill"] = btb
        else:
            metrics.metrics["book_to_bill"] = 1.0
            warnings.append("Book-to-bill estimated at 1.0x")

        # Extract Operating Margin
        op_margin = financials.get("operating_margin") or financials.get("operatingMargin")
        if op_margin:
            metrics.metrics["operating_margin"] = op_margin
        else:
            op_margin = self._calculate_operating_margin(financials)
            if op_margin:
                metrics.metrics["operating_margin"] = op_margin

        # Calculate FCF Conversion
        fcf_conv = self._calculate_fcf_conversion(financials)
        if fcf_conv:
            metrics.metrics["fcf_conversion"] = fcf_conv

        # Extract International Revenue %
        intl = self._extract_from_xbrl(
            xbrl_data, "international_revenue_pct", ["InternationalRevenuePct", "ForeignRevenuePct"]
        )
        if intl:
            metrics.metrics["international_revenue_pct"] = intl

        # Extract R&D ratio
        rd_ratio = self._calculate_rd_ratio(financials)
        if rd_ratio:
            metrics.metrics["rd_to_revenue"] = rd_ratio

        # Determine contractor type
        contractor_type = self._determine_contractor_type(symbol.upper())
        metrics.metrics["contractor_type"] = contractor_type
        metrics.metadata["contractor_type"] = contractor_type

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

    def _calculate_operating_margin(self, financials: Dict) -> Optional[float]:
        """Calculate operating margin from available data."""
        operating_income = financials.get("operatingIncome")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if operating_income and revenue and revenue > 0:
            return operating_income / revenue

        return None

    def _calculate_fcf_conversion(self, financials: Dict) -> Optional[float]:
        """Calculate FCF conversion rate."""
        fcf = financials.get("freeCashFlow") or financials.get("fcf")
        net_income = financials.get("netIncome")

        if fcf and net_income and net_income > 0:
            return fcf / net_income

        return None

    def _calculate_rd_ratio(self, financials: Dict) -> Optional[float]:
        """Calculate R&D to revenue ratio."""
        rd = financials.get("rd_expense") or financials.get("researchAndDevelopment")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if rd and revenue and revenue > 0:
            return rd / revenue

        return None

    def _determine_contractor_type(self, symbol: str) -> str:
        """Determine contractor type based on symbol."""
        if symbol in {"LMT", "RTX", "NOC", "GD", "BA"}:
            return "prime"
        elif symbol in {"LHX", "LDOS"}:
            return "electronics"
        elif symbol in {"HII"}:
            return "shipbuilding"
        elif symbol in {"CACI", "SAIC", "BAH"}:
            return "services"
        else:
            return "diversified"

    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """Assess quality of defense metrics."""
        required_metrics = ["order_backlog", "backlog_to_revenue", "operating_margin"]
        important_metrics = ["book_to_bill", "fcf_conversion"]

        required_available = sum(1 for m in required_metrics if metrics.has(m))
        important_available = sum(1 for m in important_metrics if metrics.has(m))

        if required_available == len(required_metrics) and important_available >= 1:
            return (MetricQuality.EXCELLENT, "All key defense metrics available")
        elif required_available >= 2:
            return (MetricQuality.GOOD, "Most defense metrics available")
        elif required_available >= 1:
            return (MetricQuality.FAIR, "Partial defense metrics available")
        else:
            return (MetricQuality.POOR, "Missing key metrics for defense valuation")

    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """Calculate defense-specific valuation adjustments."""
        adjustments = []

        # Backlog-to-revenue premium/discount
        backlog_ratio = metrics.get("backlog_to_revenue")
        if backlog_ratio:
            if backlog_ratio > 3.5:
                # Strong visibility deserves premium
                premium = min((backlog_ratio - 2.5) * 0.03, 0.15)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.0 + premium,
                        reason=f"Strong contract visibility ({backlog_ratio:.1f}x backlog-to-revenue)",
                        confidence=0.8,
                        affects_models=["dcf", "pe"],
                    )
                )
            elif backlog_ratio < 2.0:
                # Weak visibility
                discount = min((2.5 - backlog_ratio) * 0.05, 0.10)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 - discount,
                        reason=f"Limited contract visibility ({backlog_ratio:.1f}x backlog-to-revenue)",
                        confidence=0.7,
                        affects_models=["dcf", "pe"],
                    )
                )

        # Book-to-bill adjustment (growth indicator)
        btb = metrics.get("book_to_bill")
        if btb:
            if btb > 1.2:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Strong new orders (book-to-bill: {btb:.2f}x)",
                        confidence=0.7,
                        affects_models=["dcf"],
                    )
                )
            elif btb < 0.85:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"Weak new orders (book-to-bill: {btb:.2f}x)",
                        confidence=0.7,
                        affects_models=["dcf"],
                    )
                )

        # Operating margin adjustment (program execution)
        op_margin = metrics.get("operating_margin")
        if op_margin:
            if op_margin > 0.12:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Superior operating margin ({op_margin:.1%})",
                        confidence=0.7,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )
            elif op_margin < 0.08:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"Below-average operating margin ({op_margin:.1%})",
                        confidence=0.6,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )

        # FCF conversion adjustment
        fcf_conv = metrics.get("fcf_conversion")
        if fcf_conv:
            if fcf_conv > 1.1:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.03,
                        reason=f"Strong FCF conversion ({fcf_conv:.0%})",
                        confidence=0.6,
                        affects_models=["dcf"],
                    )
                )
            elif fcf_conv < 0.7:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"Weak FCF conversion ({fcf_conv:.0%})",
                        confidence=0.6,
                        affects_models=["dcf"],
                    )
                )

        # Prime contractor premium (scale advantages)
        contractor_type = metrics.get("contractor_type")
        if contractor_type == "prime":
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.05,
                    reason="Prime contractor scale and relationship advantages",
                    confidence=0.6,
                    affects_models=["pe", "ev_ebitda"],
                )
            )

        return adjustments

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """Return recommended tier weights for defense contractors."""
        return {
            "dcf": 35,
            "pe": 30,
            "ev_ebitda": 25,
            "pb": 5,
            "ps": 5,
            "ggm": 0,
        }


# Auto-register when module is imported
_dataset = DefenseDataset()
register_dataset(_dataset)
