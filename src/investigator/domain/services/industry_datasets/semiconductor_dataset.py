"""
Semiconductor Industry Dataset

Implements industry-specific metrics extraction for semiconductor companies.
Handles cycle-adjusted valuations based on inventory levels and book-to-bill ratios.

Covers:
- Logic chip makers (NVDA, AMD, INTC, QCOM)
- Memory manufacturers (MU, WDC)
- Semiconductor equipment (AMAT, LRCX, KLAC, ASML)
- Analog/mixed-signal (TXN, ADI, MCHP)

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


# Known semiconductor companies by symbol
KNOWN_SEMICONDUCTOR_SYMBOLS = {
    # Logic/GPU
    "NVDA",
    "AMD",
    "INTC",
    "QCOM",
    "AVGO",
    "MRVL",
    "NXPI",
    # Memory
    "MU",
    "WDC",
    # Equipment
    "AMAT",
    "LRCX",
    "KLAC",
    "ASML",
    # Analog/Mixed-Signal
    "TXN",
    "ADI",
    "MCHP",
    "ON",
    "SWKS",
    "QRVO",
    # Foundry/TSMC-like
    "TSM",
}


class SemiconductorDataset(BaseIndustryDataset):
    """
    Dataset for semiconductor industry metrics extraction.

    Key metrics:
    - Inventory Days Outstanding (cycle indicator)
    - Book-to-Bill Ratio (demand indicator)
    - Inventory-to-Sales Ratio
    - R&D as % of Revenue (innovation investment)
    - Gross Margin (pricing power)

    Cycle adjustments:
    - Peak (high inventory): Apply 10-20% discount
    - Trough (low inventory): Apply 10-15% premium
    - Normal: No adjustment
    """

    @property
    def name(self) -> str:
        return "semiconductor"

    @property
    def display_name(self) -> str:
        return "Semiconductor Industry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_industry_names(self) -> List[str]:
        return [
            "Semiconductors",
            "Semiconductor Equipment",
            "Semiconductors & Semiconductor Equipment",
            "Semiconductor Equipment & Materials",
            "Semiconductor - Integrated Circuits",
            "Semiconductor - Memory",
            "Semiconductor - Analog",
        ]

    def get_known_symbols(self) -> Set[str]:
        return KNOWN_SEMICONDUCTOR_SYMBOLS.copy()

    def get_metric_definitions(self) -> List[MetricDefinition]:
        return [
            MetricDefinition(
                name="inventory_days",
                display_name="Inventory Days Outstanding",
                description="Days of inventory on hand - key cycle indicator",
                xbrl_tags=[
                    "DaysInventoryOutstanding",
                    "InventoryDaysOfSalesOutstanding",
                    "AverageInventoryDays",
                ],
                unit="days",
                is_required=True,
                min_value=30,
                max_value=300,
                invert_for_quality=True,  # Lower is better
            ),
            MetricDefinition(
                name="book_to_bill",
                display_name="Book-to-Bill Ratio",
                description="Ratio of orders received to orders shipped",
                xbrl_tags=[
                    "BookToOrderRatio",
                    "OrderBacklogToCurrentRevenue",
                ],
                unit="ratio",
                is_required=False,
                default_value=1.0,
                min_value=0.5,
                max_value=2.0,
            ),
            MetricDefinition(
                name="inventory_to_sales",
                display_name="Inventory to Sales Ratio",
                description="Inventory as percentage of annual sales",
                xbrl_tags=["InventoryToRevenue"],
                unit="ratio",
                is_required=False,
                min_value=0.05,
                max_value=0.50,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="rd_to_revenue",
                display_name="R&D to Revenue",
                description="R&D spending as percentage of revenue",
                xbrl_tags=[
                    "ResearchAndDevelopmentExpenseToRevenue",
                    "RDExpenseRatio",
                ],
                unit="percent",
                is_required=False,
                min_value=0.05,
                max_value=0.40,
            ),
            MetricDefinition(
                name="gross_margin",
                display_name="Gross Margin",
                description="Gross profit as percentage of revenue",
                xbrl_tags=["GrossProfit"],
                unit="percent",
                is_required=True,
                min_value=0.20,
                max_value=0.90,
            ),
        ]

    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """Extract semiconductor-specific metrics from XBRL data and financials."""
        metrics = IndustryMetrics(
            industry="semiconductor",
            symbol=symbol,
            metrics={},
            metadata={"source": "xbrl+financials"},
        )
        warnings = []

        # Extract inventory days
        inventory_days = self._extract_inventory_days(xbrl_data, financials)
        if inventory_days:
            metrics.metrics["inventory_days"] = inventory_days
        else:
            warnings.append("Could not extract inventory days")

        # Extract book-to-bill (often not in standard XBRL)
        book_to_bill = self._extract_from_xbrl(xbrl_data, "book_to_bill", ["BookToOrderRatio"], default=None)
        if book_to_bill:
            metrics.metrics["book_to_bill"] = book_to_bill
        else:
            # Use default or estimate
            metrics.metrics["book_to_bill"] = 1.0
            warnings.append("Book-to-bill estimated at 1.0 (not in filings)")

        # Extract inventory-to-sales ratio
        inv_to_sales = self._calculate_inventory_to_sales(financials)
        if inv_to_sales:
            metrics.metrics["inventory_to_sales"] = inv_to_sales

        # Extract R&D to revenue
        rd_ratio = self._calculate_rd_ratio(financials)
        if rd_ratio:
            metrics.metrics["rd_to_revenue"] = rd_ratio

        # Extract gross margin
        gross_margin = financials.get("gross_margin") or financials.get("grossMargin")
        if gross_margin:
            metrics.metrics["gross_margin"] = gross_margin

        # Determine cycle position
        cycle_position = self._determine_cycle_position(metrics.metrics)
        metrics.metrics["cycle_position"] = cycle_position
        metrics.metadata["cycle_position"] = cycle_position

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

    def _extract_inventory_days(self, xbrl_data: Optional[Dict], financials: Dict) -> Optional[float]:
        """Extract or calculate inventory days."""
        # Try direct XBRL extraction first
        inv_days = self._extract_from_xbrl(
            xbrl_data,
            "inventory_days",
            ["DaysInventoryOutstanding", "InventoryDaysOfSalesOutstanding"],
        )
        if inv_days:
            return inv_days

        # Calculate from financials
        inventory = financials.get("inventory") or financials.get("totalInventory")
        cogs = financials.get("cogs") or financials.get("costOfRevenue")

        if inventory and cogs and cogs > 0:
            return (inventory / cogs) * 365

        return None

    def _calculate_inventory_to_sales(self, financials: Dict) -> Optional[float]:
        """Calculate inventory to sales ratio."""
        inventory = financials.get("inventory") or financials.get("totalInventory")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if inventory and revenue and revenue > 0:
            return inventory / revenue

        return None

    def _calculate_rd_ratio(self, financials: Dict) -> Optional[float]:
        """Calculate R&D to revenue ratio."""
        rd = financials.get("rd_expense") or financials.get("researchAndDevelopment")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if rd and revenue and revenue > 0:
            return rd / revenue

        return None

    def _determine_cycle_position(self, metrics_dict: Dict) -> str:
        """
        Determine semiconductor cycle position.

        Based on:
        - Inventory days: > 120 = peak, < 70 = trough
        - Book-to-bill: > 1.1 = expansion, < 0.9 = contraction
        """
        inv_days = metrics_dict.get("inventory_days")
        btb = metrics_dict.get("book_to_bill", 1.0)

        if inv_days:
            if inv_days > 120:
                return "peak"  # High inventory = weak demand
            elif inv_days < 70:
                return "trough"  # Low inventory = strong demand

        if btb:
            if btb > 1.1:
                return "expansion"
            elif btb < 0.9:
                return "contraction"

        return "normal"

    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """Assess quality of semiconductor metrics."""
        required_metrics = ["inventory_days", "gross_margin"]
        important_metrics = ["book_to_bill", "rd_to_revenue"]

        required_available = sum(1 for m in required_metrics if metrics.has(m))
        important_available = sum(1 for m in important_metrics if metrics.has(m))

        if required_available == len(required_metrics) and important_available >= 1:
            return (MetricQuality.EXCELLENT, "All key semiconductor metrics available")
        elif required_available == len(required_metrics):
            return (MetricQuality.GOOD, "Required metrics available, some optional missing")
        elif required_available >= 1:
            return (MetricQuality.FAIR, "Partial metrics available, cycle assessment may be limited")
        else:
            return (MetricQuality.POOR, "Missing key metrics for semiconductor valuation")

    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """Calculate cycle-based valuation adjustments."""
        adjustments = []

        cycle_position = metrics.get("cycle_position", "normal")
        inv_days = metrics.get("inventory_days")

        # Cycle adjustment
        if cycle_position == "peak":
            # High inventory = apply discount
            discount = 0.15 if inv_days and inv_days > 150 else 0.10
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="discount",
                    factor=1.0 - discount,
                    reason=(
                        f"Semiconductor cycle peak - elevated inventory ({inv_days:.0f} days)"
                        if inv_days
                        else "Semiconductor cycle peak"
                    ),
                    confidence=0.7 if inv_days else 0.5,
                    affects_models=["ev_ebitda", "pe", "dcf"],
                )
            )
        elif cycle_position == "trough":
            # Low inventory = apply premium
            premium = 0.15 if inv_days and inv_days < 60 else 0.10
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.0 + premium,
                    reason=(
                        f"Semiconductor cycle trough - lean inventory ({inv_days:.0f} days)"
                        if inv_days
                        else "Semiconductor cycle trough"
                    ),
                    confidence=0.7 if inv_days else 0.5,
                    affects_models=["ev_ebitda", "pe", "dcf"],
                )
            )

        # Gross margin adjustment (pricing power)
        gross_margin = metrics.get("gross_margin")
        if gross_margin:
            if gross_margin > 0.60:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Superior gross margin ({gross_margin:.1%}) indicates pricing power",
                        confidence=0.8,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )
            elif gross_margin < 0.35:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"Below-average gross margin ({gross_margin:.1%})",
                        confidence=0.7,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )

        # R&D investment adjustment
        rd_ratio = metrics.get("rd_to_revenue")
        if rd_ratio and rd_ratio > 0.20:
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.03,
                    reason=f"High R&D investment ({rd_ratio:.1%}) supports future growth",
                    confidence=0.6,
                    affects_models=["dcf"],
                )
            )

        return adjustments

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """Return recommended tier weights for semiconductors."""
        return {
            "ev_ebitda": 45,
            "dcf": 25,
            "pe": 20,
            "pb": 10,
            "ps": 0,
            "ggm": 0,
        }


# Auto-register when module is imported
_dataset = SemiconductorDataset()
register_dataset(_dataset)
