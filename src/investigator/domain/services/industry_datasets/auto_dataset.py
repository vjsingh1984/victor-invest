"""
Auto Manufacturing Industry Dataset

Implements industry-specific metrics extraction for automotive companies.
Handles EV transition adjustments and legacy vs EV manufacturer differentiation.

Covers:
- Traditional OEMs (F, GM, STLA)
- EV Leaders (TSLA, RIVN, LCID)
- Luxury/Premium (TM, HMC)
- EV Transitioning (BMW, VWAGY)

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


# Known auto symbols with EV classification
KNOWN_AUTO_SYMBOLS = {
    # Pure EV
    "TSLA",
    "RIVN",
    "LCID",
    "NIO",
    "XPEV",
    "LI",
    # Traditional OEMs
    "F",
    "GM",
    "STLA",
    # Japanese OEMs
    "TM",
    "HMC",
    # German (ADRs)
    "VWAGY",
    "BMWYY",
    "MBGYY",
    # Korean
    "HYMTF",
}

# EV revenue percentage estimates (can be overridden by XBRL data)
EV_REVENUE_ESTIMATES = {
    "TSLA": 0.95,
    "RIVN": 1.00,
    "LCID": 1.00,
    "NIO": 1.00,
    "XPEV": 1.00,
    "LI": 1.00,
    "F": 0.08,
    "GM": 0.05,
    "TM": 0.03,
    "HMC": 0.02,
    "STLA": 0.06,
    "VWAGY": 0.12,
    "BMWYY": 0.15,
}


class AutoDataset(BaseIndustryDataset):
    """
    Dataset for automotive industry metrics extraction.

    Key metrics:
    - EV Sales Mix (% of revenue from EVs)
    - Vehicle Unit Sales
    - Average Selling Price (ASP)
    - Warranty Reserve Ratio
    - R&D Spending (EV investment proxy)
    - Capacity Utilization

    Valuation approach:
    - EV-weighted P/S for EV leaders
    - Traditional DCF/PE for legacy OEMs
    - Premium for high EV mix
    - Discount for heavy warranty reserves
    """

    @property
    def name(self) -> str:
        return "auto_manufacturing"

    @property
    def display_name(self) -> str:
        return "Auto Manufacturing Industry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_industry_names(self) -> List[str]:
        return [
            "Auto Manufacturing",
            "Auto Manufacturers",
            "Automobiles",
            "Automobile Manufacturers",
            "Auto - Major",
            "Auto Manufacturers - Major",
            "Electric Vehicles",
            "Auto & Truck Manufacturers",
        ]

    def get_known_symbols(self) -> Set[str]:
        return KNOWN_AUTO_SYMBOLS.copy()

    def get_metric_definitions(self) -> List[MetricDefinition]:
        return [
            MetricDefinition(
                name="ev_sales_mix",
                display_name="EV Sales Mix",
                description="Percentage of revenue from electric vehicles",
                xbrl_tags=[
                    "ElectricVehicleRevenuePct",
                    "EVtoTotalRevenueRatio",
                    "ElectricVehicleSalesRatio",
                ],
                unit="percent",
                is_required=True,
                min_value=0.0,
                max_value=1.0,
            ),
            MetricDefinition(
                name="vehicle_unit_sales",
                display_name="Vehicle Unit Sales",
                description="Total vehicles sold in reporting period",
                xbrl_tags=[
                    "UnitsSold",
                    "VehicleUnitsSold",
                    "TotalVehicleDeliveries",
                ],
                unit="units",
                is_required=False,
            ),
            MetricDefinition(
                name="average_selling_price",
                display_name="Average Selling Price",
                description="Revenue per vehicle sold",
                xbrl_tags=[
                    "AverageVehicleSellingPrice",
                    "RevenuePerVehicle",
                ],
                unit="USD",
                is_required=False,
            ),
            MetricDefinition(
                name="warranty_reserve_ratio",
                display_name="Warranty Reserve Ratio",
                description="Warranty reserves as % of revenue",
                xbrl_tags=[
                    "WarrantyReserveRatio",
                    "AccrualForWarranties",
                    "ProductWarrantyAccrual",
                ],
                unit="percent",
                is_required=False,
                min_value=0.01,
                max_value=0.10,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="rd_to_revenue",
                display_name="R&D to Revenue",
                description="R&D spending as percentage of revenue",
                xbrl_tags=[
                    "ResearchAndDevelopmentExpenseToRevenue",
                ],
                unit="percent",
                is_required=False,
                min_value=0.02,
                max_value=0.15,
            ),
            MetricDefinition(
                name="gross_margin",
                display_name="Gross Margin",
                description="Gross profit as percentage of revenue",
                xbrl_tags=["GrossProfit"],
                unit="percent",
                is_required=True,
                min_value=0.05,
                max_value=0.35,
            ),
            MetricDefinition(
                name="capacity_utilization",
                display_name="Capacity Utilization",
                description="Production as % of manufacturing capacity",
                xbrl_tags=["CapacityUtilization", "PlantUtilizationRate"],
                unit="percent",
                is_required=False,
                min_value=0.50,
                max_value=1.00,
            ),
        ]

    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """Extract auto-specific metrics from XBRL data and financials."""
        metrics = IndustryMetrics(
            industry="auto_manufacturing",
            symbol=symbol,
            metrics={},
            metadata={"source": "xbrl+financials"},
        )
        warnings = []

        symbol_upper = symbol.upper()

        # Extract EV Sales Mix
        ev_mix = self._extract_from_xbrl(
            xbrl_data, "ev_sales_mix", ["ElectricVehicleRevenuePct", "EVtoTotalRevenueRatio"]
        )
        if ev_mix:
            metrics.metrics["ev_sales_mix"] = ev_mix
        else:
            # Fall back to known estimates
            ev_mix = EV_REVENUE_ESTIMATES.get(symbol_upper, 0.05)
            metrics.metrics["ev_sales_mix"] = ev_mix
            warnings.append(f"EV mix estimated at {ev_mix:.0%} (not in filings)")

        # Extract Vehicle Unit Sales
        units = self._extract_from_xbrl(
            xbrl_data, "vehicle_unit_sales", ["UnitsSold", "VehicleUnitsSold", "TotalVehicleDeliveries"]
        )
        if units:
            metrics.metrics["vehicle_unit_sales"] = units

        # Calculate Average Selling Price
        if units:
            revenue = financials.get("revenue") or financials.get("totalRevenue")
            if revenue and units > 0:
                metrics.metrics["average_selling_price"] = revenue / units

        # Extract Warranty Reserve Ratio
        warranty = self._extract_warranty_ratio(xbrl_data, financials)
        if warranty:
            metrics.metrics["warranty_reserve_ratio"] = warranty
        else:
            # Industry average
            metrics.metrics["warranty_reserve_ratio"] = 0.025
            warnings.append("Warranty reserve ratio estimated at 2.5%")

        # Extract R&D to Revenue
        rd_ratio = self._calculate_rd_ratio(financials)
        if rd_ratio:
            metrics.metrics["rd_to_revenue"] = rd_ratio

        # Extract Gross Margin
        gross_margin = financials.get("gross_margin") or financials.get("grossMargin")
        if gross_margin:
            metrics.metrics["gross_margin"] = gross_margin
        else:
            gross_margin = self._calculate_gross_margin(financials)
            if gross_margin:
                metrics.metrics["gross_margin"] = gross_margin

        # Determine manufacturer type
        mfr_type = self._determine_manufacturer_type(symbol_upper, ev_mix)
        metrics.metrics["manufacturer_type"] = mfr_type
        metrics.metadata["manufacturer_type"] = mfr_type

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

    def _extract_warranty_ratio(self, xbrl_data: Optional[Dict], financials: Dict) -> Optional[float]:
        """Extract or calculate warranty reserve ratio."""
        warranty = self._extract_from_xbrl(
            xbrl_data,
            "warranty_reserve_ratio",
            ["WarrantyReserveRatio", "AccrualForWarranties", "ProductWarrantyAccrual"],
        )
        if warranty:
            # If it's an absolute value, convert to ratio
            revenue = financials.get("revenue") or financials.get("totalRevenue")
            if warranty > 1 and revenue and revenue > 0:
                return warranty / revenue
            return warranty

        return None

    def _calculate_rd_ratio(self, financials: Dict) -> Optional[float]:
        """Calculate R&D to revenue ratio."""
        rd = financials.get("rd_expense") or financials.get("researchAndDevelopment")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if rd and revenue and revenue > 0:
            return rd / revenue

        return None

    def _calculate_gross_margin(self, financials: Dict) -> Optional[float]:
        """Calculate gross margin from available data."""
        gross_profit = financials.get("grossProfit")
        revenue = financials.get("revenue") or financials.get("totalRevenue")

        if gross_profit and revenue and revenue > 0:
            return gross_profit / revenue

        return None

    def _determine_manufacturer_type(self, symbol: str, ev_mix: float) -> str:
        """Determine manufacturer type based on EV mix."""
        if ev_mix >= 0.80:
            return "ev_pure_play"
        elif ev_mix >= 0.30:
            return "ev_leader"
        elif ev_mix >= 0.10:
            return "ev_transitioning"
        else:
            return "traditional"

    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """Assess quality of auto metrics."""
        required_metrics = ["ev_sales_mix", "gross_margin"]
        important_metrics = ["vehicle_unit_sales", "warranty_reserve_ratio", "rd_to_revenue"]

        required_available = sum(1 for m in required_metrics if metrics.has(m))
        important_available = sum(1 for m in important_metrics if metrics.has(m))

        if required_available == len(required_metrics) and important_available >= 2:
            return (MetricQuality.EXCELLENT, "All key auto metrics available")
        elif required_available == len(required_metrics):
            return (MetricQuality.GOOD, "Required auto metrics available")
        elif required_available >= 1:
            return (MetricQuality.FAIR, "Partial auto metrics available")
        else:
            return (MetricQuality.POOR, "Missing key metrics for auto valuation")

    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """Calculate auto-specific valuation adjustments."""
        adjustments = []

        mfr_type = metrics.get("manufacturer_type", "traditional")
        ev_mix = metrics.get("ev_sales_mix", 0.0)

        # EV mix premium/discount
        if mfr_type == "ev_pure_play":
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.25,
                    reason=f"Pure-play EV manufacturer ({ev_mix:.0%} EV revenue)",
                    confidence=0.7,
                    affects_models=["ps", "ev_ebitda"],
                )
            )
        elif mfr_type == "ev_leader":
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.15,
                    reason=f"EV leadership position ({ev_mix:.0%} EV revenue)",
                    confidence=0.7,
                    affects_models=["ps", "ev_ebitda"],
                )
            )
        elif mfr_type == "ev_transitioning":
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.05,
                    reason=f"Active EV transition ({ev_mix:.0%} EV revenue)",
                    confidence=0.6,
                    affects_models=["dcf"],
                )
            )

        # Gross margin adjustment
        gross_margin = metrics.get("gross_margin")
        if gross_margin:
            if gross_margin > 0.20:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.10,
                        reason=f"Superior gross margin ({gross_margin:.1%}) for auto industry",
                        confidence=0.8,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )
            elif gross_margin < 0.10:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.90,
                        reason=f"Below-average gross margin ({gross_margin:.1%})",
                        confidence=0.7,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )

        # Warranty reserve adjustment (quality indicator)
        warranty = metrics.get("warranty_reserve_ratio")
        if warranty:
            if warranty > 0.04:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"Elevated warranty reserves ({warranty:.1%}) suggest quality issues",
                        confidence=0.6,
                        affects_models=["pe", "dcf"],
                    )
                )
            elif warranty < 0.02:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.03,
                        reason=f"Low warranty reserves ({warranty:.1%}) indicate quality",
                        confidence=0.5,
                        affects_models=["pe"],
                    )
                )

        # R&D investment adjustment (future growth)
        rd_ratio = metrics.get("rd_to_revenue")
        if rd_ratio and rd_ratio > 0.06:
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.05,
                    reason=f"High R&D investment ({rd_ratio:.1%}) supports EV transition",
                    confidence=0.6,
                    affects_models=["dcf"],
                )
            )

        return adjustments

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """Return recommended tier weights for auto manufacturing."""
        return {
            "ev_ebitda": 40,
            "dcf": 25,
            "pe": 20,
            "ps": 10,
            "pb": 5,
            "ggm": 0,
        }


# Auto-register when module is imported
_dataset = AutoDataset()
register_dataset(_dataset)
