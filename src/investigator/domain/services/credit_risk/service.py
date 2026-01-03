# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Credit Risk Service.

This module provides a unified service interface for credit risk assessment,
coordinating the individual calculators and providing data transformation
from SEC XBRL data to the standardized FinancialData format.

The service acts as a Facade (SOLID: Single entry point) that:
1. Transforms raw SEC data to FinancialData format
2. Orchestrates individual credit score calculations
3. Provides composite risk assessment
4. Caches results for efficiency

Example:
    service = get_credit_risk_service()

    # Calculate all scores from SEC data
    result = await service.calculate_from_symbol("AAPL")

    # Or calculate from pre-extracted financial data
    fin_data = FinancialData(symbol="AAPL", total_assets=...)
    result = service.calculate_composite(fin_data)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from investigator.domain.services.credit_risk.protocols import FinancialData
from investigator.domain.services.credit_risk.altman_zscore import (
    AltmanZScoreCalculator,
    AltmanZScoreResult,
)
from investigator.domain.services.credit_risk.beneish_mscore import (
    BeneishMScoreCalculator,
    BeneishMScoreResult,
)
from investigator.domain.services.credit_risk.piotroski_fscore import (
    PiotroskiFScoreCalculator,
    PiotroskiFScoreResult,
)
from investigator.domain.services.credit_risk.composite_distress import (
    CompositeDistressCalculator,
    CompositeCreditRiskResult,
)

logger = logging.getLogger(__name__)

# Singleton instance
_credit_risk_service: Optional["CreditRiskService"] = None


@dataclass
class CreditRiskAssessment:
    """Complete credit risk assessment for a symbol.

    Contains all individual scores plus composite assessment.
    """
    symbol: str
    altman: Optional[AltmanZScoreResult] = None
    beneish: Optional[BeneishMScoreResult] = None
    piotroski: Optional[PiotroskiFScoreResult] = None
    composite: Optional[CompositeCreditRiskResult] = None
    data_quality: str = "unknown"
    warnings: List[str] = None
    calculation_date: Optional[date] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.calculation_date is None:
            self.calculation_date = date.today()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "calculation_date": str(self.calculation_date),
            "data_quality": self.data_quality,
            "warnings": self.warnings,
            "scores": {
                "altman_zscore": self.altman.to_dict() if self.altman else None,
                "beneish_mscore": self.beneish.to_dict() if self.beneish else None,
                "piotroski_fscore": self.piotroski.to_dict() if self.piotroski else None,
            },
            "composite": self.composite.to_dict() if self.composite else None,
            "summary": self._get_summary(),
        }

    def _get_summary(self) -> Dict[str, Any]:
        """Get summary of credit risk assessment."""
        summary = {
            "symbol": self.symbol,
        }

        if self.composite:
            summary.update({
                "distress_tier": self.composite.distress_tier.name if self.composite.distress_tier else None,
                "distress_probability": self.composite.distress_probability,
                "valuation_discount": self.composite.valuation_discount,
                "risk_factors_count": len(self.composite.risk_factors),
                "positive_factors_count": len(self.composite.positive_factors),
            })

        if self.altman:
            summary["altman_zone"] = self.altman.zone.value if self.altman.zone else None
            summary["altman_score"] = self.altman.score

        if self.beneish:
            summary["beneish_risk"] = self.beneish.risk_level.value if self.beneish.risk_level else None
            summary["beneish_score"] = self.beneish.score

        if self.piotroski:
            summary["piotroski_strength"] = self.piotroski.strength.value if self.piotroski.strength else None
            summary["piotroski_score"] = self.piotroski.score

        return summary


class CreditRiskService:
    """Unified service for credit risk assessment.

    Provides a high-level interface for calculating credit risk scores,
    with support for:
    - Individual score calculations (Altman, Beneish, Piotroski)
    - Composite risk assessment
    - Automatic data transformation from SEC XBRL format
    - Async support for integration with existing infrastructure

    SOLID: Facade pattern providing single entry point
    """

    def __init__(self):
        """Initialize the credit risk service."""
        self._altman = AltmanZScoreCalculator()
        self._beneish = BeneishMScoreCalculator()
        self._piotroski = PiotroskiFScoreCalculator()
        self._composite = CompositeDistressCalculator(
            altman_calculator=self._altman,
            beneish_calculator=self._beneish,
            piotroski_calculator=self._piotroski,
        )

    def calculate_altman(self, data: FinancialData) -> AltmanZScoreResult:
        """Calculate Altman Z-Score.

        Args:
            data: Standardized financial data

        Returns:
            AltmanZScoreResult with score and zone
        """
        return self._altman.calculate(data)

    def calculate_beneish(self, data: FinancialData) -> BeneishMScoreResult:
        """Calculate Beneish M-Score.

        Requires prior period data for year-over-year comparisons.

        Args:
            data: Standardized financial data with prior_period

        Returns:
            BeneishMScoreResult with score and risk level
        """
        return self._beneish.calculate(data)

    def calculate_piotroski(self, data: FinancialData) -> PiotroskiFScoreResult:
        """Calculate Piotroski F-Score.

        Requires prior period data for trend analysis.

        Args:
            data: Standardized financial data with prior_period

        Returns:
            PiotroskiFScoreResult with score and strength
        """
        return self._piotroski.calculate(data)

    def calculate_composite(self, data: FinancialData) -> CompositeCreditRiskResult:
        """Calculate composite credit risk assessment.

        Combines all three scores into unified distress tier.

        Args:
            data: Standardized financial data with prior_period

        Returns:
            CompositeCreditRiskResult with tier, discount, and factors
        """
        return self._composite.calculate(data)

    def calculate_all(self, data: FinancialData) -> CreditRiskAssessment:
        """Calculate all credit risk scores.

        Args:
            data: Standardized financial data

        Returns:
            CreditRiskAssessment with all scores and composite
        """
        assessment = CreditRiskAssessment(symbol=data.symbol)

        # Calculate individual scores
        assessment.altman = self._altman.calculate(data)
        assessment.beneish = self._beneish.calculate(data)
        assessment.piotroski = self._piotroski.calculate(data)

        # Calculate composite
        assessment.composite = self._composite.calculate(data)

        # Assess data quality
        all_warnings = []
        all_warnings.extend(assessment.altman.warnings if assessment.altman else [])
        all_warnings.extend(assessment.beneish.warnings if assessment.beneish else [])
        all_warnings.extend(assessment.piotroski.warnings if assessment.piotroski else [])
        assessment.warnings = list(set(all_warnings))

        # Determine data quality rating
        if len(assessment.warnings) == 0:
            assessment.data_quality = "excellent"
        elif len(assessment.warnings) < 5:
            assessment.data_quality = "good"
        elif len(assessment.warnings) < 10:
            assessment.data_quality = "fair"
        else:
            assessment.data_quality = "poor"

        return assessment

    async def calculate_from_symbol(
        self,
        symbol: str,
        sec_data: Optional[Dict[str, Any]] = None,
    ) -> CreditRiskAssessment:
        """Calculate credit risk from symbol, fetching SEC data if needed.

        Args:
            symbol: Stock ticker symbol
            sec_data: Pre-fetched SEC data (optional)

        Returns:
            CreditRiskAssessment with all scores
        """
        # Transform SEC data to FinancialData format
        if sec_data is None:
            # Fetch from SEC infrastructure
            fin_data = await self._fetch_financial_data(symbol)
        else:
            fin_data = self._transform_sec_data(symbol, sec_data)

        if fin_data is None:
            assessment = CreditRiskAssessment(symbol=symbol)
            assessment.warnings.append("Unable to retrieve financial data")
            assessment.data_quality = "unavailable"
            return assessment

        return self.calculate_all(fin_data)

    async def _fetch_financial_data(self, symbol: str) -> Optional[FinancialData]:
        """Fetch financial data from SEC infrastructure.

        Args:
            symbol: Stock ticker symbol

        Returns:
            FinancialData or None if unavailable
        """
        try:
            # Lazy import to avoid circular dependencies
            from investigator.infrastructure.sec.companyfacts_extractor import (
                get_sec_companyfacts_extractor
            )

            loop = asyncio.get_event_loop()
            extractor = get_sec_companyfacts_extractor()

            # Get current period metrics
            metrics = await loop.run_in_executor(
                None,
                extractor.extract_financial_metrics,
                symbol
            )

            if not metrics:
                logger.warning(f"No SEC metrics found for {symbol}")
                return None

            # Transform to FinancialData
            fin_data = self._transform_sec_metrics(symbol, metrics)

            # Try to get prior period data
            # Note: This requires quarterly data access which may need enhancement
            # For now, we'll work with single period

            return fin_data

        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {e}")
            return None

    def _transform_sec_metrics(
        self,
        symbol: str,
        metrics: Dict[str, Any]
    ) -> FinancialData:
        """Transform SEC extracted metrics to FinancialData format.

        Args:
            symbol: Stock ticker
            metrics: Metrics from SEC companyfacts extractor

        Returns:
            FinancialData instance
        """
        return FinancialData(
            symbol=symbol,
            fiscal_year=metrics.get("fiscal_year"),
            fiscal_period=metrics.get("fiscal_period"),
            # Balance Sheet - Assets
            total_assets=metrics.get("assets"),
            current_assets=metrics.get("assets_current"),
            cash_and_equivalents=metrics.get("cash_and_equivalents"),
            accounts_receivable=metrics.get("accounts_receivable"),
            inventory=metrics.get("inventory"),
            property_plant_equipment=metrics.get("property_plant_equipment"),
            # Balance Sheet - Liabilities & Equity
            total_liabilities=metrics.get("liabilities"),
            current_liabilities=metrics.get("liabilities_current"),
            total_debt=metrics.get("total_debt"),
            long_term_debt=metrics.get("long_term_debt"),
            short_term_debt=metrics.get("debt_current"),
            stockholders_equity=metrics.get("equity"),
            retained_earnings=metrics.get("retained_earnings"),
            # Income Statement
            revenue=metrics.get("revenues"),
            gross_profit=metrics.get("gross_profit"),
            operating_income=metrics.get("operating_income"),
            net_income=metrics.get("net_income"),
            cost_of_revenue=metrics.get("cost_of_revenue"),
            sga_expense=metrics.get("selling_general_admin"),
            depreciation_amortization=metrics.get("depreciation_amortization"),
            interest_expense=metrics.get("interest_expense"),
            # Cash Flow
            operating_cash_flow=metrics.get("operating_cash_flow"),
            capital_expenditures=metrics.get("capital_expenditures"),
            # Market Data
            market_cap=metrics.get("market_cap"),
            shares_outstanding=(
                metrics.get("shares_outstanding") or
                metrics.get("common_stock_shares_outstanding") or
                metrics.get("weighted_average_shares_diluted")
            ),
        )

    def _transform_sec_data(
        self,
        symbol: str,
        sec_data: Dict[str, Any]
    ) -> FinancialData:
        """Transform raw SEC tool data to FinancialData format.

        Args:
            symbol: Stock ticker
            sec_data: Data from SECFilingTool.execute()

        Returns:
            FinancialData instance
        """
        # Handle nested structure from SECFilingTool
        bs = sec_data.get("balance_sheet", {})
        is_ = sec_data.get("income_statement", {})
        cf = sec_data.get("cash_flow", {})
        ratios = sec_data.get("ratios", {})

        return FinancialData(
            symbol=symbol,
            fiscal_year=sec_data.get("fiscal_year"),
            fiscal_period=sec_data.get("fiscal_period"),
            # Balance Sheet - Assets
            total_assets=bs.get("total_assets"),
            current_assets=bs.get("current_assets"),
            cash_and_equivalents=bs.get("cash_and_equivalents"),
            accounts_receivable=bs.get("accounts_receivable"),
            inventory=bs.get("inventory"),
            property_plant_equipment=bs.get("property_plant_equipment"),
            # Balance Sheet - Liabilities & Equity
            total_liabilities=bs.get("total_liabilities"),
            current_liabilities=bs.get("current_liabilities"),
            total_debt=bs.get("total_debt"),
            long_term_debt=bs.get("long_term_debt"),
            short_term_debt=bs.get("short_term_debt"),
            stockholders_equity=bs.get("stockholders_equity"),
            retained_earnings=bs.get("retained_earnings"),
            # Income Statement
            revenue=is_.get("revenue") or is_.get("revenues") or is_.get("total_revenue"),
            gross_profit=is_.get("gross_profit"),
            operating_income=is_.get("operating_income"),
            net_income=is_.get("net_income") or is_.get("net_income_loss"),
            cost_of_revenue=is_.get("cost_of_revenue"),
            sga_expense=is_.get("sga_expense"),
            depreciation_amortization=is_.get("depreciation_amortization"),
            interest_expense=is_.get("interest_expense"),
            # Cash Flow
            operating_cash_flow=cf.get("operating_cash_flow"),
            capital_expenditures=cf.get("capital_expenditures") or cf.get("capex"),
            # Market Data
            shares_outstanding=bs.get("shares_outstanding") or ratios.get("shares_outstanding"),
        )


def get_credit_risk_service() -> CreditRiskService:
    """Get or create the singleton credit risk service instance.

    Returns:
        CreditRiskService instance
    """
    global _credit_risk_service
    if _credit_risk_service is None:
        _credit_risk_service = CreditRiskService()
    return _credit_risk_service
