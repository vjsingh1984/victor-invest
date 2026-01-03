"""
REIT Industry Dataset

Implements industry-specific metrics extraction for Real Estate Investment Trusts.
Uses FFO-based valuation methodology with occupancy and lease adjustments.

Covers:
- Residential REITs (EQR, AVB, MAA)
- Industrial REITs (PLD, AMT, EQIX)
- Retail REITs (SPG, O, VICI)
- Office REITs (BXP, ARE)
- Healthcare REITs (WELL, VTR)
- Data Center REITs (EQIX, DLR)
- Cell Tower REITs (AMT, CCI)

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


# Known REIT symbols by type
KNOWN_REIT_SYMBOLS = {
    # Industrial/Logistics
    "PLD",
    # Data Centers
    "EQIX",
    "DLR",
    # Cell Towers
    "AMT",
    "CCI",
    "SBAC",
    # Residential
    "EQR",
    "AVB",
    "MAA",
    "UDR",
    "ESS",
    "CPT",
    "AIV",
    # Retail
    "SPG",
    "O",
    "VICI",
    "NNN",
    "REG",
    "KIM",
    "FRT",
    # Office
    "BXP",
    "ARE",
    "VNO",
    "SLG",
    # Healthcare
    "WELL",
    "VTR",
    "HTA",
    "DOC",
    # Self-Storage
    "PSA",
    "EXR",
    "CUBE",
    "LSI",
    # Diversified
    "WPC",
    "BRX",
    "EPR",
}


class REITDataset(BaseIndustryDataset):
    """
    Dataset for REIT industry metrics extraction.

    Key metrics:
    - Funds From Operations (FFO) - primary profitability metric
    - AFFO (Adjusted FFO) - more accurate recurring cash flow
    - Occupancy Rate - operational health
    - Same-Store NOI Growth - organic growth
    - Debt/EBITDA - leverage

    Valuation approach:
    - P/FFO is primary metric (analogous to P/E)
    - NAV-based valuation for asset-heavy REITs
    - Premium for high occupancy, growth markets
    """

    @property
    def name(self) -> str:
        return "reit"

    @property
    def display_name(self) -> str:
        return "Real Estate Investment Trust Industry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_industry_names(self) -> List[str]:
        return [
            "REIT",
            "REITs",
            "Real Estate Investment Trust",
            "Real Estate Investment Trusts",
            "Equity Real Estate Investment Trusts",
            "REIT - Residential",
            "REIT - Retail",
            "REIT - Industrial",
            "REIT - Office",
            "REIT - Healthcare",
            "REIT - Diversified",
            "REIT - Hotel & Motel",
            "REIT - Specialty",
            "Data Center REITs",
            "Industrial REITs",
            "Specialized REITs",
            "Telecom Tower REITs",
        ]

    def get_known_symbols(self) -> Set[str]:
        return KNOWN_REIT_SYMBOLS.copy()

    def get_metric_definitions(self) -> List[MetricDefinition]:
        return [
            MetricDefinition(
                name="ffo",
                display_name="Funds From Operations",
                description="Net income + depreciation - gains on sales",
                xbrl_tags=[
                    "FundsFromOperations",
                    "FundsFromOperationsAttributableToParent",
                    "FFO",
                ],
                unit="USD",
                is_required=True,
            ),
            MetricDefinition(
                name="ffo_per_share",
                display_name="FFO Per Share",
                description="FFO divided by diluted shares",
                xbrl_tags=[
                    "FundsFromOperationsPerShare",
                    "FundsFromOperationsPerShareDiluted",
                ],
                unit="USD",
                is_required=True,
            ),
            MetricDefinition(
                name="affo",
                display_name="Adjusted FFO",
                description="FFO adjusted for recurring capex and straight-line rent",
                xbrl_tags=[
                    "AdjustedFundsFromOperations",
                    "CoreFundsFromOperations",
                    "AFFO",
                ],
                unit="USD",
                is_required=False,
            ),
            MetricDefinition(
                name="occupancy_rate",
                display_name="Occupancy Rate",
                description="Percentage of rentable space occupied",
                xbrl_tags=[
                    "OccupancyRate",
                    "PortfolioOccupancy",
                    "PropertyOccupancyRate",
                ],
                unit="percent",
                is_required=True,
                min_value=0.50,
                max_value=1.00,
            ),
            MetricDefinition(
                name="same_store_noi_growth",
                display_name="Same-Store NOI Growth",
                description="Year-over-year growth in same-property net operating income",
                xbrl_tags=[
                    "SameStoreNetOperatingIncomeGrowth",
                    "ComparableNOIGrowth",
                ],
                unit="percent",
                is_required=False,
                min_value=-0.20,
                max_value=0.30,
            ),
            MetricDefinition(
                name="debt_to_ebitda",
                display_name="Debt to EBITDA",
                description="Total debt divided by EBITDA",
                xbrl_tags=["DebtToEBITDA", "NetDebtToEBITDA"],
                unit="ratio",
                is_required=False,
                min_value=2.0,
                max_value=12.0,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="nav_per_share",
                display_name="NAV Per Share",
                description="Net Asset Value per share",
                xbrl_tags=["NetAssetValuePerShare", "NAVPerShare"],
                unit="USD",
                is_required=False,
            ),
            MetricDefinition(
                name="cap_rate",
                display_name="Implied Cap Rate",
                description="NOI / Property Value",
                xbrl_tags=["CapitalizationRate", "ImpliedCapRate"],
                unit="percent",
                is_required=False,
                min_value=0.03,
                max_value=0.12,
            ),
        ]

    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """Extract REIT-specific metrics from XBRL data and financials."""
        metrics = IndustryMetrics(
            industry="reit",
            symbol=symbol,
            metrics={},
            metadata={"source": "xbrl+financials"},
        )
        warnings = []

        # Extract FFO
        ffo = self._extract_from_xbrl(
            xbrl_data, "ffo", ["FundsFromOperations", "FundsFromOperationsAttributableToParent"]
        )
        if ffo:
            metrics.metrics["ffo"] = ffo
        else:
            # Try to calculate FFO
            ffo = self._calculate_ffo(financials)
            if ffo:
                metrics.metrics["ffo"] = ffo
            else:
                warnings.append("FFO not available")

        # Extract FFO per share
        ffo_ps = self._extract_from_xbrl(
            xbrl_data, "ffo_per_share", ["FundsFromOperationsPerShare", "FundsFromOperationsPerShareDiluted"]
        )
        if ffo_ps:
            metrics.metrics["ffo_per_share"] = ffo_ps
        elif ffo:
            shares = financials.get("sharesOutstanding") or financials.get("dilutedShares")
            if shares and shares > 0:
                metrics.metrics["ffo_per_share"] = ffo / shares

        # Extract AFFO
        affo = self._extract_from_xbrl(xbrl_data, "affo", ["AdjustedFundsFromOperations", "CoreFundsFromOperations"])
        if affo:
            metrics.metrics["affo"] = affo

        # Extract Occupancy Rate
        occupancy = self._extract_from_xbrl(
            xbrl_data, "occupancy_rate", ["OccupancyRate", "PortfolioOccupancy", "PropertyOccupancyRate"]
        )
        if occupancy:
            # Convert to decimal if needed
            if occupancy > 1:
                occupancy = occupancy / 100
            metrics.metrics["occupancy_rate"] = occupancy
        else:
            # Use industry average
            metrics.metrics["occupancy_rate"] = 0.93
            warnings.append("Occupancy rate not available, using industry average (93%)")

        # Extract Same-Store NOI Growth
        ss_noi = self._extract_from_xbrl(
            xbrl_data, "same_store_noi_growth", ["SameStoreNetOperatingIncomeGrowth", "ComparableNOIGrowth"]
        )
        if ss_noi:
            metrics.metrics["same_store_noi_growth"] = ss_noi

        # Extract Debt/EBITDA
        debt_ebitda = self._calculate_debt_to_ebitda(financials)
        if debt_ebitda:
            metrics.metrics["debt_to_ebitda"] = debt_ebitda

        # Extract/calculate Cap Rate
        cap_rate = self._calculate_cap_rate(financials)
        if cap_rate:
            metrics.metrics["cap_rate"] = cap_rate

        # Determine REIT type
        reit_type = self._determine_reit_type(symbol, kwargs.get("industry", ""))
        metrics.metrics["reit_type"] = reit_type
        metrics.metadata["reit_type"] = reit_type

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

    def _calculate_ffo(self, financials: Dict) -> Optional[float]:
        """Calculate FFO from available data."""
        net_income = financials.get("netIncome")
        depreciation = financials.get("depreciation") or financials.get("depreciationAmortization")

        if net_income and depreciation:
            # FFO = Net Income + Depreciation (simplified)
            return net_income + depreciation

        return None

    def _calculate_debt_to_ebitda(self, financials: Dict) -> Optional[float]:
        """Calculate Debt/EBITDA ratio."""
        total_debt = financials.get("totalDebt") or financials.get("longTermDebt")
        ebitda = financials.get("ebitda")

        if total_debt and ebitda and ebitda > 0:
            return total_debt / ebitda

        return None

    def _calculate_cap_rate(self, financials: Dict) -> Optional[float]:
        """Calculate implied cap rate."""
        noi = financials.get("noi") or financials.get("netOperatingIncome")
        total_assets = financials.get("totalAssets")

        if noi and total_assets and total_assets > 0:
            return noi / total_assets

        return None

    def _determine_reit_type(self, symbol: str, industry: str) -> str:
        """Determine REIT type based on symbol and industry."""
        symbol_upper = symbol.upper()

        # Symbol-based classification
        if symbol_upper in {"AMT", "CCI", "SBAC"}:
            return "cell_tower"
        elif symbol_upper in {"EQIX", "DLR"}:
            return "data_center"
        elif symbol_upper in {"PLD"}:
            return "industrial"
        elif symbol_upper in {"EQR", "AVB", "MAA", "UDR", "ESS"}:
            return "residential"
        elif symbol_upper in {"SPG", "O", "VICI", "NNN"}:
            return "retail"
        elif symbol_upper in {"BXP", "ARE", "VNO"}:
            return "office"
        elif symbol_upper in {"WELL", "VTR"}:
            return "healthcare"
        elif symbol_upper in {"PSA", "EXR", "CUBE"}:
            return "self_storage"

        # Industry-based classification
        industry_lower = industry.lower() if industry else ""
        if "data center" in industry_lower or "technology" in industry_lower:
            return "data_center"
        elif "tower" in industry_lower or "telecom" in industry_lower:
            return "cell_tower"
        elif "industrial" in industry_lower or "logistics" in industry_lower:
            return "industrial"
        elif "residential" in industry_lower or "apartment" in industry_lower:
            return "residential"
        elif "retail" in industry_lower or "shopping" in industry_lower:
            return "retail"
        elif "office" in industry_lower:
            return "office"
        elif "healthcare" in industry_lower or "medical" in industry_lower:
            return "healthcare"

        return "diversified"

    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """Assess quality of REIT metrics."""
        required_metrics = ["ffo", "ffo_per_share", "occupancy_rate"]
        important_metrics = ["affo", "debt_to_ebitda", "same_store_noi_growth"]

        required_available = sum(1 for m in required_metrics if metrics.has(m))
        important_available = sum(1 for m in important_metrics if metrics.has(m))

        if required_available == len(required_metrics) and important_available >= 2:
            return (MetricQuality.EXCELLENT, "All key REIT metrics available")
        elif required_available == len(required_metrics):
            return (MetricQuality.GOOD, "Required REIT metrics available")
        elif required_available >= 2:
            return (MetricQuality.FAIR, "Partial REIT metrics available")
        else:
            return (MetricQuality.POOR, "Missing key metrics for REIT valuation")

    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """Calculate REIT-specific valuation adjustments."""
        adjustments = []

        reit_type = metrics.get("reit_type", "diversified")

        # Occupancy adjustment
        occupancy = metrics.get("occupancy_rate")
        if occupancy:
            if occupancy > 0.96:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"High occupancy ({occupancy:.1%}) indicates strong demand",
                        confidence=0.8,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )
            elif occupancy < 0.88:
                discount = min((0.93 - occupancy) * 2, 0.15)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 - discount,
                        reason=f"Below-average occupancy ({occupancy:.1%})",
                        confidence=0.7,
                        affects_models=["pe", "ev_ebitda"],
                    )
                )

        # Same-store NOI growth adjustment
        ss_noi_growth = metrics.get("same_store_noi_growth")
        if ss_noi_growth:
            if ss_noi_growth > 0.05:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.0 + min(ss_noi_growth, 0.10),
                        reason=f"Strong same-store NOI growth ({ss_noi_growth:.1%})",
                        confidence=0.7,
                        affects_models=["dcf", "pe"],
                    )
                )
            elif ss_noi_growth < 0:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 + ss_noi_growth,  # Negative growth = discount
                        reason=f"Negative same-store NOI growth ({ss_noi_growth:.1%})",
                        confidence=0.8,
                        affects_models=["dcf", "pe"],
                    )
                )

        # Leverage adjustment
        debt_ebitda = metrics.get("debt_to_ebitda")
        if debt_ebitda:
            if debt_ebitda > 7.0:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.90,
                        reason=f"High leverage (Debt/EBITDA: {debt_ebitda:.1f}x)",
                        confidence=0.7,
                        affects_models=["pe", "pb"],
                    )
                )
            elif debt_ebitda < 4.0:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Conservative leverage (Debt/EBITDA: {debt_ebitda:.1f}x)",
                        confidence=0.6,
                        affects_models=["pe", "pb"],
                    )
                )

        # REIT type premium/discount
        premium_types = {"data_center", "cell_tower", "industrial"}
        discount_types = {"office", "retail"}

        if reit_type in premium_types:
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.10,
                    reason=f"{reit_type.replace('_', ' ').title()} REIT - secular growth tailwinds",
                    confidence=0.7,
                    affects_models=["pe", "ev_ebitda"],
                )
            )
        elif reit_type in discount_types:
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="discount",
                    factor=0.95,
                    reason=f"{reit_type.replace('_', ' ').title()} REIT - structural headwinds",
                    confidence=0.6,
                    affects_models=["pe", "ev_ebitda"],
                )
            )

        return adjustments

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """Return recommended tier weights for REITs."""
        return {
            "ev_ebitda": 35,
            "pe": 30,  # P/FFO in reality
            "pb": 20,
            "dcf": 15,
            "ps": 0,
            "ggm": 0,
        }


# Auto-register when module is imported
_dataset = REITDataset()
register_dataset(_dataset)
