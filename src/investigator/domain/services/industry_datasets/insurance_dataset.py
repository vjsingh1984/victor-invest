"""
Insurance Industry Dataset

Implements industry-specific metrics extraction for insurance companies.
Handles combined ratio analysis and underwriting profitability assessment.

Covers:
- Property & Casualty (ALL, PGR, TRV, CB)
- Life Insurance (MET, PRU, AFL)
- Reinsurance (RNR, RE)
- Specialty (CINF, WRB)

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


# Known insurance symbols by type
KNOWN_INSURANCE_SYMBOLS = {
    # P&C Insurance
    "ALL",
    "PGR",
    "TRV",
    "CB",
    "AIG",
    "HIG",
    # Life Insurance
    "MET",
    "PRU",
    "AFL",
    "LNC",
    "VOYA",
    # Reinsurance
    "RNR",
    "RE",
    "ACGL",
    # Specialty
    "CINF",
    "WRB",
    "KNSL",
    # Diversified
    "BRK.B",
    "BRK.A",
}


class InsuranceDataset(BaseIndustryDataset):
    """
    Dataset for insurance industry metrics extraction.

    Key metrics:
    - Combined Ratio (underwriting profitability)
    - Loss Ratio (claims efficiency)
    - Expense Ratio (operational efficiency)
    - Investment Yield (float management)
    - ROE (overall profitability)
    - Reserve-to-Premium Ratio (reserve adequacy)

    Valuation approach:
    - P/B based on underwriting quality
    - Premium for combined ratio < 95%
    - Discount for reserve deficiency signals
    """

    @property
    def name(self) -> str:
        return "insurance"

    @property
    def display_name(self) -> str:
        return "Insurance Industry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_industry_names(self) -> List[str]:
        return [
            "Insurance",
            "Property & Casualty Insurance",
            "Property-Casualty Insurance",
            "Life Insurance",
            "Insurance - Life",
            "Insurance - Property & Casualty",
            "Insurance - Diversified",
            "Insurance - Reinsurance",
            "Insurance - Specialty",
            "Multi-line Insurance",
            "Reinsurance",
        ]

    def get_known_symbols(self) -> Set[str]:
        return KNOWN_INSURANCE_SYMBOLS.copy()

    def get_metric_definitions(self) -> List[MetricDefinition]:
        return [
            MetricDefinition(
                name="combined_ratio",
                display_name="Combined Ratio",
                description="Loss ratio + expense ratio (< 100% = underwriting profit)",
                xbrl_tags=[
                    "CombinedRatio",
                    "PropertyCasualtyInsuranceCombinedRatio",
                ],
                unit="percent",
                is_required=True,
                min_value=0.70,
                max_value=1.20,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="loss_ratio",
                display_name="Loss Ratio",
                description="Incurred losses / earned premiums",
                xbrl_tags=[
                    "LossRatio",
                    "InsuranceLossRatio",
                    "LossAndLossAdjustmentExpenseRatio",
                ],
                unit="percent",
                is_required=True,
                min_value=0.40,
                max_value=0.90,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="expense_ratio",
                display_name="Expense Ratio",
                description="Underwriting expenses / earned premiums",
                xbrl_tags=[
                    "ExpenseRatio",
                    "InsuranceExpenseRatio",
                    "UnderwritingExpenseRatio",
                ],
                unit="percent",
                is_required=False,
                min_value=0.20,
                max_value=0.45,
                invert_for_quality=True,
            ),
            MetricDefinition(
                name="investment_yield",
                display_name="Investment Yield",
                description="Investment income / average invested assets",
                xbrl_tags=[
                    "InvestmentYield",
                    "NetInvestmentIncomeYield",
                ],
                unit="percent",
                is_required=False,
                min_value=0.02,
                max_value=0.08,
            ),
            MetricDefinition(
                name="roe",
                display_name="Return on Equity",
                description="Net income / average shareholders equity",
                xbrl_tags=["ReturnOnEquity", "ROE"],
                unit="percent",
                is_required=True,
                min_value=0.0,
                max_value=0.25,
            ),
            MetricDefinition(
                name="reserve_to_premium",
                display_name="Reserve to Premium Ratio",
                description="Loss reserves / net premiums earned",
                xbrl_tags=[
                    "ReserveToPremiumRatio",
                    "LossReserveToEarnedPremium",
                ],
                unit="ratio",
                is_required=False,
                min_value=0.50,
                max_value=3.00,
            ),
            MetricDefinition(
                name="premium_growth",
                display_name="Premium Growth Rate",
                description="Year-over-year growth in net premiums written",
                xbrl_tags=[
                    "PremiumGrowthRate",
                    "NetPremiumsWrittenGrowth",
                ],
                unit="percent",
                is_required=False,
                min_value=-0.20,
                max_value=0.30,
            ),
        ]

    def extract_metrics(self, symbol: str, xbrl_data: Optional[Dict], financials: Dict, **kwargs) -> IndustryMetrics:
        """Extract insurance-specific metrics from XBRL data and financials."""
        metrics = IndustryMetrics(
            industry="insurance",
            symbol=symbol,
            metrics={},
            metadata={"source": "xbrl+financials"},
        )
        warnings = []

        # Extract Combined Ratio
        combined = self._extract_from_xbrl(
            xbrl_data, "combined_ratio", ["CombinedRatio", "PropertyCasualtyInsuranceCombinedRatio"]
        )
        if combined:
            # Convert to decimal if needed
            if combined > 2:
                combined = combined / 100
            metrics.metrics["combined_ratio"] = combined
        else:
            # Try to calculate from loss + expense ratios
            combined = self._calculate_combined_ratio(xbrl_data, financials)
            if combined:
                metrics.metrics["combined_ratio"] = combined
            else:
                warnings.append("Combined ratio not available")

        # Extract Loss Ratio
        loss_ratio = self._extract_from_xbrl(
            xbrl_data, "loss_ratio", ["LossRatio", "InsuranceLossRatio", "LossAndLossAdjustmentExpenseRatio"]
        )
        if loss_ratio:
            if loss_ratio > 2:
                loss_ratio = loss_ratio / 100
            metrics.metrics["loss_ratio"] = loss_ratio
        else:
            loss_ratio = self._calculate_loss_ratio(financials)
            if loss_ratio:
                metrics.metrics["loss_ratio"] = loss_ratio

        # Extract Expense Ratio
        expense_ratio = self._extract_from_xbrl(
            xbrl_data, "expense_ratio", ["ExpenseRatio", "InsuranceExpenseRatio", "UnderwritingExpenseRatio"]
        )
        if expense_ratio:
            if expense_ratio > 2:
                expense_ratio = expense_ratio / 100
            metrics.metrics["expense_ratio"] = expense_ratio

        # Extract Investment Yield
        inv_yield = self._extract_from_xbrl(
            xbrl_data, "investment_yield", ["InvestmentYield", "NetInvestmentIncomeYield"]
        )
        if inv_yield:
            metrics.metrics["investment_yield"] = inv_yield
        else:
            inv_yield = self._calculate_investment_yield(financials)
            if inv_yield:
                metrics.metrics["investment_yield"] = inv_yield

        # Extract ROE
        roe = financials.get("roe") or financials.get("returnOnEquity")
        if roe:
            metrics.metrics["roe"] = roe
        else:
            roe = self._calculate_roe(financials)
            if roe:
                metrics.metrics["roe"] = roe

        # Extract Reserve to Premium Ratio
        reserve_ratio = self._extract_from_xbrl(
            xbrl_data, "reserve_to_premium", ["ReserveToPremiumRatio", "LossReserveToEarnedPremium"]
        )
        if reserve_ratio:
            metrics.metrics["reserve_to_premium"] = reserve_ratio

        # Extract Premium Growth
        prem_growth = self._extract_from_xbrl(
            xbrl_data, "premium_growth", ["PremiumGrowthRate", "NetPremiumsWrittenGrowth"]
        )
        if prem_growth:
            metrics.metrics["premium_growth"] = prem_growth

        # Determine insurance type
        ins_type = self._determine_insurance_type(symbol.upper(), kwargs.get("industry", ""))
        metrics.metrics["insurance_type"] = ins_type
        metrics.metadata["insurance_type"] = ins_type

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

    def _calculate_combined_ratio(self, xbrl_data: Optional[Dict], financials: Dict) -> Optional[float]:
        """Calculate combined ratio from loss and expense ratios."""
        loss_ratio = self._extract_from_xbrl(xbrl_data, "loss_ratio", ["LossRatio", "InsuranceLossRatio"])
        expense_ratio = self._extract_from_xbrl(xbrl_data, "expense_ratio", ["ExpenseRatio", "InsuranceExpenseRatio"])

        if loss_ratio and expense_ratio:
            # Convert to decimal if needed
            if loss_ratio > 2:
                loss_ratio = loss_ratio / 100
            if expense_ratio > 2:
                expense_ratio = expense_ratio / 100
            return loss_ratio + expense_ratio

        return None

    def _calculate_loss_ratio(self, financials: Dict) -> Optional[float]:
        """Calculate loss ratio from available data."""
        incurred_losses = financials.get("incurredLosses") or financials.get("claimsExpense")
        earned_premiums = financials.get("earnedPremiums") or financials.get("netPremiumsEarned")

        if incurred_losses and earned_premiums and earned_premiums > 0:
            return incurred_losses / earned_premiums

        return None

    def _calculate_investment_yield(self, financials: Dict) -> Optional[float]:
        """Calculate investment yield."""
        inv_income = financials.get("investmentIncome") or financials.get("netInvestmentIncome")
        invested_assets = financials.get("investedAssets") or financials.get("totalInvestments")

        if inv_income and invested_assets and invested_assets > 0:
            return inv_income / invested_assets

        return None

    def _calculate_roe(self, financials: Dict) -> Optional[float]:
        """Calculate ROE from available data."""
        net_income = financials.get("netIncome")
        equity = financials.get("totalEquity") or financials.get("shareholdersEquity")

        if net_income and equity and equity > 0:
            return net_income / equity

        return None

    def _determine_insurance_type(self, symbol: str, industry: str) -> str:
        """Determine insurance type based on symbol and industry."""
        if symbol in {"ALL", "PGR", "TRV", "CB", "AIG", "HIG"}:
            return "property_casualty"
        elif symbol in {"MET", "PRU", "AFL", "LNC", "VOYA"}:
            return "life"
        elif symbol in {"RNR", "RE", "ACGL"}:
            return "reinsurance"
        elif symbol in {"CINF", "WRB", "KNSL"}:
            return "specialty"
        elif symbol in {"BRK.B", "BRK.A"}:
            return "diversified"

        # Industry-based classification
        industry_lower = industry.lower() if industry else ""
        if "life" in industry_lower:
            return "life"
        elif "reinsurance" in industry_lower:
            return "reinsurance"
        elif "property" in industry_lower or "casualty" in industry_lower:
            return "property_casualty"

        return "diversified"

    def assess_quality(self, metrics: IndustryMetrics) -> Tuple[MetricQuality, str]:
        """Assess quality of insurance metrics."""
        required_metrics = ["combined_ratio", "loss_ratio", "roe"]
        important_metrics = ["expense_ratio", "investment_yield", "reserve_to_premium"]

        required_available = sum(1 for m in required_metrics if metrics.has(m))
        important_available = sum(1 for m in important_metrics if metrics.has(m))

        if required_available == len(required_metrics) and important_available >= 2:
            return (MetricQuality.EXCELLENT, "All key insurance metrics available")
        elif required_available >= 2:
            return (MetricQuality.GOOD, "Most insurance metrics available")
        elif required_available >= 1:
            return (MetricQuality.FAIR, "Partial insurance metrics available")
        else:
            return (MetricQuality.POOR, "Missing key metrics for insurance valuation")

    def get_valuation_adjustments(
        self, metrics: IndustryMetrics, financials: Dict, **kwargs
    ) -> List[ValuationAdjustment]:
        """Calculate insurance-specific valuation adjustments."""
        adjustments = []

        # Combined ratio adjustment (key underwriting metric)
        combined = metrics.get("combined_ratio")
        if combined:
            if combined < 0.93:
                # Excellent underwriting
                premium = min((0.97 - combined) * 2, 0.15)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.0 + premium,
                        reason=f"Superior underwriting (combined ratio: {combined:.1%})",
                        confidence=0.8,
                        affects_models=["pb", "pe"],
                    )
                )
            elif combined > 1.02:
                # Underwriting loss
                discount = min((combined - 0.98) * 2, 0.15)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 - discount,
                        reason=f"Underwriting loss (combined ratio: {combined:.1%})",
                        confidence=0.8,
                        affects_models=["pb", "pe"],
                    )
                )

        # ROE-based P/B adjustment
        roe = metrics.get("roe")
        cost_of_equity = 0.10  # Assume 10% CoE

        if roe:
            if roe > cost_of_equity + 0.05:
                premium = min((roe - cost_of_equity) * 1.5, 0.20)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.0 + premium,
                        reason=f"Superior ROE ({roe:.1%}) vs cost of equity",
                        confidence=0.7,
                        affects_models=["pb"],
                    )
                )
            elif roe < cost_of_equity - 0.03:
                discount = min((cost_of_equity - roe) * 1.5, 0.15)
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=1.0 - discount,
                        reason=f"Below cost-of-equity ROE ({roe:.1%})",
                        confidence=0.7,
                        affects_models=["pb"],
                    )
                )

        # Investment yield adjustment
        inv_yield = metrics.get("investment_yield")
        if inv_yield:
            if inv_yield > 0.045:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.03,
                        reason=f"Strong investment yield ({inv_yield:.1%})",
                        confidence=0.6,
                        affects_models=["pe"],
                    )
                )
            elif inv_yield < 0.025:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.97,
                        reason=f"Weak investment yield ({inv_yield:.1%})",
                        confidence=0.5,
                        affects_models=["pe"],
                    )
                )

        # Premium growth adjustment
        prem_growth = metrics.get("premium_growth")
        if prem_growth:
            if prem_growth > 0.10:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="premium",
                        factor=1.05,
                        reason=f"Strong premium growth ({prem_growth:.1%})",
                        confidence=0.6,
                        affects_models=["dcf", "pe"],
                    )
                )
            elif prem_growth < -0.05:
                adjustments.append(
                    ValuationAdjustment(
                        adjustment_type="discount",
                        factor=0.95,
                        reason=f"Declining premiums ({prem_growth:.1%})",
                        confidence=0.6,
                        affects_models=["dcf", "pe"],
                    )
                )

        # Insurance type adjustment
        ins_type = metrics.get("insurance_type")
        if ins_type == "reinsurance":
            adjustments.append(
                ValuationAdjustment(
                    adjustment_type="premium",
                    factor=1.05,
                    reason="Reinsurance business model - higher ROE potential",
                    confidence=0.5,
                    affects_models=["pb"],
                )
            )

        return adjustments

    def get_tier_weights(self) -> Optional[Dict[str, int]]:
        """Return recommended tier weights for insurance companies."""
        return {
            "pb": 45,
            "pe": 30,
            "dcf": 15,
            "ev_ebitda": 0,  # Not applicable
            "ps": 5,
            "ggm": 5,
        }


# Auto-register when module is imported
_dataset = InsuranceDataset()
register_dataset(_dataset)
