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

"""Beneish M-Score Calculator.

The Beneish M-Score is a mathematical model used to detect earnings manipulation
in financial statements. Developed by Professor Messod Beneish in 1999, it uses
eight financial ratios to identify the likelihood that a company has manipulated
its reported earnings.

Formula:
    M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI
        + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI

Where:
    DSRI = Days Sales in Receivables Index
    GMI = Gross Margin Index
    AQI = Asset Quality Index
    SGI = Sales Growth Index
    DEPI = Depreciation Index
    SGAI = SG&A Index
    TATA = Total Accruals to Total Assets
    LVGI = Leverage Index

Interpretation:
    M > -1.78: Likely manipulator (higher probability of earnings manipulation)
    M ≤ -1.78: Likely non-manipulator

References:
    Beneish, M. D. (1999). "The Detection of Earnings Manipulation".
    Financial Analysts Journal, 55(5), 24-36.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from investigator.domain.services.credit_risk.protocols import (
    CreditScoreResult,
    FinancialData,
)

logger = logging.getLogger(__name__)


class ManipulationRisk(Enum):
    """Beneish M-Score manipulation risk classification."""
    LOW = "low"             # M < -2.22 (very unlikely)
    MODERATE = "moderate"   # -2.22 ≤ M ≤ -1.78
    HIGH = "high"           # M > -1.78 (likely manipulator)


@dataclass
class BeneishMScoreResult(CreditScoreResult):
    """Result of Beneish M-Score calculation.

    Attributes:
        risk_level: Classification of manipulation risk
        manipulation_probability: Estimated probability of manipulation
    """
    risk_level: Optional[ManipulationRisk] = None
    manipulation_probability: Optional[float] = None
    score_name: str = "Beneish M-Score"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = super().to_dict()
        result.update({
            "risk_level": self.risk_level.value if self.risk_level else None,
            "manipulation_probability": self.manipulation_probability,
        })
        return result


class BeneishMScoreCalculator:
    """Calculator for Beneish M-Score earnings manipulation detection.

    The M-Score requires two periods of data (current and prior) to calculate
    year-over-year changes in financial ratios.

    SOLID: Single Responsibility - only calculates Beneish M-Score
    """

    # M-Score coefficients
    INTERCEPT = -4.84
    COEF_DSRI = 0.920   # Days Sales in Receivables Index
    COEF_GMI = 0.528    # Gross Margin Index
    COEF_AQI = 0.404    # Asset Quality Index
    COEF_SGI = 0.892    # Sales Growth Index
    COEF_DEPI = 0.115   # Depreciation Index
    COEF_SGAI = -0.172  # SG&A Index
    COEF_TATA = 4.679   # Total Accruals to Total Assets
    COEF_LVGI = -0.327  # Leverage Index

    # Thresholds
    MANIPULATION_THRESHOLD = -1.78
    LOW_RISK_THRESHOLD = -2.22

    def __init__(self):
        """Initialize the M-Score calculator."""
        self._name = "Beneish M-Score Calculator"
        self._description = (
            "Earnings manipulation detection model. M > -1.78 indicates "
            "likely manipulation; M < -2.22 indicates low manipulation risk."
        )

    @property
    def name(self) -> str:
        """Return calculator name."""
        return self._name

    @property
    def description(self) -> str:
        """Return calculator description."""
        return self._description

    def validate_data(self, data: FinancialData) -> List[str]:
        """Validate required financial data fields.

        M-Score requires both current and prior period data.

        Args:
            data: Financial data to validate

        Returns:
            List of missing field names
        """
        missing = []

        # Current period required fields
        current_fields = [
            ("accounts_receivable", data.accounts_receivable),
            ("revenue", data.revenue),
            ("gross_profit", data.gross_profit),
            ("total_assets", data.total_assets),
            ("property_plant_equipment", data.property_plant_equipment),
            ("depreciation_amortization", data.depreciation_amortization),
            ("sga_expense", data.sga_expense),
            ("net_income", data.net_income),
            ("operating_cash_flow", data.operating_cash_flow),
            ("current_liabilities", data.current_liabilities),
            ("long_term_debt", data.long_term_debt),
        ]

        for field_name, value in current_fields:
            if value is None:
                missing.append(f"current.{field_name}")

        # Prior period required fields
        if data.prior_period is None:
            missing.append("prior_period (required for YoY comparison)")
        else:
            prior = data.prior_period
            prior_fields = [
                ("accounts_receivable", prior.accounts_receivable),
                ("revenue", prior.revenue),
                ("gross_profit", prior.gross_profit),
                ("total_assets", prior.total_assets),
                ("property_plant_equipment", prior.property_plant_equipment),
                ("depreciation_amortization", prior.depreciation_amortization),
                ("sga_expense", prior.sga_expense),
                ("current_liabilities", prior.current_liabilities),
                ("long_term_debt", prior.long_term_debt),
            ]

            for field_name, value in prior_fields:
                if value is None:
                    missing.append(f"prior.{field_name}")

        return missing

    def calculate(self, data: FinancialData) -> BeneishMScoreResult:
        """Calculate Beneish M-Score from financial data.

        Args:
            data: Standardized financial data with prior period

        Returns:
            BeneishMScoreResult with score, risk level, and components
        """
        result = BeneishMScoreResult(
            symbol=data.symbol,
            calculation_date=date.today(),
            data_date=data.data_date,
        )

        # Validate data
        missing = self.validate_data(data)
        if missing:
            result.warnings.extend([f"Missing: {f}" for f in missing])

        # Must have prior period data
        if data.prior_period is None:
            result.interpretation = "Cannot calculate - requires prior period data for comparison"
            return result

        try:
            # Calculate individual indices
            indices = self._calculate_indices(data)
            result.components = indices

            # Count how many indices we successfully calculated
            valid_indices = [v for k, v in indices.items() if k.endswith("_index") and v is not None]

            if len(valid_indices) < 4:  # Need at least 4 indices for meaningful calculation
                result.warnings.append(f"Only {len(valid_indices)} indices calculated (need 4+)")
                result.interpretation = "Insufficient data for reliable M-Score"
                return result

            # Calculate M-Score
            m_score = self._calculate_mscore(indices)
            result.score = round(m_score, 4) if m_score is not None else None

            # Classify risk level
            result.risk_level = self._classify_risk(m_score)

            # Set interpretation
            result.interpretation = self._get_interpretation(result.risk_level, result.score)

            # Estimate manipulation probability
            result.manipulation_probability = self._estimate_manipulation_prob(m_score)

            logger.info(
                f"{data.symbol}: Beneish M-Score = {result.score} "
                f"({result.risk_level.value if result.risk_level else 'N/A'})"
            )

        except Exception as e:
            logger.error(f"Error calculating Beneish M-Score for {data.symbol}: {e}")
            result.warnings.append(f"Calculation error: {str(e)}")
            result.interpretation = "Calculation failed"

        return result

    def _calculate_indices(self, data: FinancialData) -> Dict[str, Any]:
        """Calculate the eight M-Score indices.

        Args:
            data: Current period financial data with prior_period

        Returns:
            Dictionary of index values
        """
        indices = {}
        curr = data
        prior = data.prior_period

        # DSRI: Days Sales in Receivables Index
        # (AR_t / Sales_t) / (AR_t-1 / Sales_t-1)
        dsri = self._safe_ratio_index(
            curr.accounts_receivable, curr.revenue,
            prior.accounts_receivable, prior.revenue
        )
        indices["dsri_index"] = dsri
        indices["dsri_current_ratio"] = self._safe_divide(curr.accounts_receivable, curr.revenue)
        indices["dsri_prior_ratio"] = self._safe_divide(prior.accounts_receivable, prior.revenue)

        # GMI: Gross Margin Index
        # ((Sales_t-1 - COGS_t-1) / Sales_t-1) / ((Sales_t - COGS_t) / Sales_t)
        # Note: Inverted - prior margin / current margin
        curr_gm_ratio = self._safe_divide(curr.gross_profit, curr.revenue)
        prior_gm_ratio = self._safe_divide(prior.gross_profit, prior.revenue)
        gmi = self._safe_divide(prior_gm_ratio, curr_gm_ratio) if curr_gm_ratio and prior_gm_ratio else None
        indices["gmi_index"] = gmi
        indices["gmi_current_margin"] = curr_gm_ratio
        indices["gmi_prior_margin"] = prior_gm_ratio

        # AQI: Asset Quality Index
        # (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_t-1 + PPE_t-1) / TA_t-1)
        curr_tangible = self._safe_sum(curr.current_assets, curr.property_plant_equipment)
        prior_tangible = self._safe_sum(prior.current_assets, prior.property_plant_equipment)
        curr_aqi_ratio = self._safe_divide(curr_tangible, curr.total_assets)
        prior_aqi_ratio = self._safe_divide(prior_tangible, prior.total_assets)

        if curr_aqi_ratio is not None and prior_aqi_ratio is not None:
            curr_quality = 1 - curr_aqi_ratio
            prior_quality = 1 - prior_aqi_ratio
            aqi = self._safe_divide(curr_quality, prior_quality)
        else:
            aqi = None
        indices["aqi_index"] = aqi

        # SGI: Sales Growth Index
        # Sales_t / Sales_t-1
        sgi = self._safe_divide(curr.revenue, prior.revenue)
        indices["sgi_index"] = sgi

        # DEPI: Depreciation Index
        # (Dep_t-1 / (Dep_t-1 + PPE_t-1)) / (Dep_t / (Dep_t + PPE_t))
        curr_dep_ratio = self._safe_divide(
            curr.depreciation_amortization,
            self._safe_sum(curr.depreciation_amortization, curr.property_plant_equipment)
        )
        prior_dep_ratio = self._safe_divide(
            prior.depreciation_amortization,
            self._safe_sum(prior.depreciation_amortization, prior.property_plant_equipment)
        )
        depi = self._safe_divide(prior_dep_ratio, curr_dep_ratio) if curr_dep_ratio and prior_dep_ratio else None
        indices["depi_index"] = depi

        # SGAI: SG&A Index
        # (SGA_t / Sales_t) / (SGA_t-1 / Sales_t-1)
        sgai = self._safe_ratio_index(
            curr.sga_expense, curr.revenue,
            prior.sga_expense, prior.revenue
        )
        indices["sgai_index"] = sgai

        # TATA: Total Accruals to Total Assets
        # (Net Income - CFO) / Total Assets
        if curr.net_income is not None and curr.operating_cash_flow is not None and curr.total_assets:
            tata = (curr.net_income - curr.operating_cash_flow) / curr.total_assets
        else:
            tata = None
        indices["tata_index"] = tata
        indices["tata_accruals"] = (
            curr.net_income - curr.operating_cash_flow
            if curr.net_income is not None and curr.operating_cash_flow is not None
            else None
        )

        # LVGI: Leverage Index
        # ((CL_t + LTD_t) / TA_t) / ((CL_t-1 + LTD_t-1) / TA_t-1)
        lvgi = self._safe_ratio_index(
            self._safe_sum(curr.current_liabilities, curr.long_term_debt), curr.total_assets,
            self._safe_sum(prior.current_liabilities, prior.long_term_debt), prior.total_assets
        )
        indices["lvgi_index"] = lvgi

        return indices

    def _calculate_mscore(self, indices: Dict[str, Any]) -> Optional[float]:
        """Calculate M-Score from indices.

        M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI
            + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI
        """
        dsri = indices.get("dsri_index") or 1.0
        gmi = indices.get("gmi_index") or 1.0
        aqi = indices.get("aqi_index") or 1.0
        sgi = indices.get("sgi_index") or 1.0
        depi = indices.get("depi_index") or 1.0
        sgai = indices.get("sgai_index") or 1.0
        tata = indices.get("tata_index") or 0.0
        lvgi = indices.get("lvgi_index") or 1.0

        m_score = (
            self.INTERCEPT +
            self.COEF_DSRI * dsri +
            self.COEF_GMI * gmi +
            self.COEF_AQI * aqi +
            self.COEF_SGI * sgi +
            self.COEF_DEPI * depi +
            self.COEF_SGAI * sgai +
            self.COEF_TATA * tata +
            self.COEF_LVGI * lvgi
        )

        return m_score

    def _classify_risk(self, m_score: Optional[float]) -> Optional[ManipulationRisk]:
        """Classify M-Score into risk levels."""
        if m_score is None:
            return None

        if m_score > self.MANIPULATION_THRESHOLD:
            return ManipulationRisk.HIGH
        elif m_score < self.LOW_RISK_THRESHOLD:
            return ManipulationRisk.LOW
        else:
            return ManipulationRisk.MODERATE

    def _get_interpretation(self, risk: Optional[ManipulationRisk], score: Optional[float]) -> str:
        """Generate human-readable interpretation."""
        if risk is None or score is None:
            return "Unable to calculate M-Score due to missing data"

        score_str = f"{score:.2f}"

        if risk == ManipulationRisk.HIGH:
            return (
                f"High Risk (M = {score_str}): M-Score above -1.78 threshold. "
                "Company shows patterns consistent with earnings manipulation. "
                "Requires careful due diligence of accounting practices."
            )
        elif risk == ManipulationRisk.LOW:
            return (
                f"Low Risk (M = {score_str}): M-Score below -2.22. "
                "Company shows no significant signs of earnings manipulation. "
                "Financial statements appear reliable."
            )
        else:
            return (
                f"Moderate Risk (M = {score_str}): M-Score in grey zone. "
                "Some indicators warrant attention but not conclusive evidence. "
                "Recommend detailed review of accounting policies."
            )

    def _estimate_manipulation_prob(self, m_score: Optional[float]) -> Optional[float]:
        """Estimate manipulation probability from M-Score.

        Based on Beneish's research:
        M > -1.78: ~76% accuracy in detecting manipulators
        M < -1.78: ~82% accuracy in identifying non-manipulators
        """
        if m_score is None:
            return None

        import math
        try:
            # Logistic approximation centered at threshold
            prob = 1.0 / (1.0 + math.exp(-1.5 * (m_score + 1.78)))
            return round(prob, 4)
        except (OverflowError, ValueError):
            return 1.0 if m_score > -1.78 else 0.0

    @staticmethod
    def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safely divide two numbers, handling None and zero."""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def _safe_sum(*values: Optional[float]) -> Optional[float]:
        """Safely sum values, returning None if any are None."""
        if any(v is None for v in values):
            return None
        return sum(values)

    def _safe_ratio_index(
        self,
        curr_num: Optional[float], curr_den: Optional[float],
        prior_num: Optional[float], prior_den: Optional[float]
    ) -> Optional[float]:
        """Calculate ratio index: (curr_num/curr_den) / (prior_num/prior_den)."""
        curr_ratio = self._safe_divide(curr_num, curr_den)
        prior_ratio = self._safe_divide(prior_num, prior_den)
        return self._safe_divide(curr_ratio, prior_ratio)
