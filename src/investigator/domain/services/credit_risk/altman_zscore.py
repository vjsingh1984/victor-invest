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

"""Altman Z-Score Calculator.

The Altman Z-Score is a bankruptcy prediction model developed by Edward Altman
in 1968. It uses five financial ratios to predict the probability of a company
going bankrupt within two years.

Original Formula (Manufacturing):
    Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets

Interpretation:
    Z > 2.99: Safe Zone - Low bankruptcy risk
    1.81 ≤ Z ≤ 2.99: Grey Zone - Uncertain
    Z < 1.81: Distress Zone - High bankruptcy risk

References:
    Altman, E. I. (1968). "Financial Ratios, Discriminant Analysis and the
    Prediction of Corporate Bankruptcy". Journal of Finance, 23(4), 589-609.
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


class AltmanZone(Enum):
    """Altman Z-Score zones indicating bankruptcy risk."""
    SAFE = "safe"           # Z > 2.99
    GREY = "grey"           # 1.81 ≤ Z ≤ 2.99
    DISTRESS = "distress"   # Z < 1.81


class AltmanModel(Enum):
    """Different Altman Z-Score model variants."""
    ORIGINAL = "original"       # Manufacturing companies (1968)
    REVISED = "revised"         # Non-manufacturing (Z')
    EMERGING = "emerging"       # Emerging markets (Z'')
    SERVICE = "service"         # Service companies


@dataclass
class AltmanZScoreResult(CreditScoreResult):
    """Result of Altman Z-Score calculation.

    Attributes:
        zone: Risk zone classification (Safe, Grey, Distress)
        model_used: Which Altman model variant was applied
        bankruptcy_probability: Estimated probability of bankruptcy
    """
    zone: Optional[AltmanZone] = None
    model_used: AltmanModel = AltmanModel.ORIGINAL
    bankruptcy_probability: Optional[float] = None
    score_name: str = "Altman Z-Score"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = super().to_dict()
        result.update({
            "zone": self.zone.value if self.zone else None,
            "model_used": self.model_used.value,
            "bankruptcy_probability": self.bankruptcy_probability,
        })
        return result


class AltmanZScoreCalculator:
    """Calculator for Altman Z-Score bankruptcy prediction.

    The calculator automatically selects the appropriate model variant based
    on the company's characteristics (manufacturing vs non-manufacturing).

    SOLID: Single Responsibility - only calculates Altman Z-Score
    """

    # Altman Z-Score coefficients (Original model)
    COEF_X1 = 1.2   # Working Capital / Total Assets
    COEF_X2 = 1.4   # Retained Earnings / Total Assets
    COEF_X3 = 3.3   # EBIT / Total Assets
    COEF_X4 = 0.6   # Market Value Equity / Total Liabilities
    COEF_X5 = 1.0   # Sales / Total Assets

    # Zone thresholds
    SAFE_THRESHOLD = 2.99
    DISTRESS_THRESHOLD = 1.81

    # Revised model coefficients (Non-manufacturing, Z')
    REVISED_COEF_X1 = 6.56
    REVISED_COEF_X2 = 3.26
    REVISED_COEF_X3 = 6.72
    REVISED_COEF_X4 = 1.05
    REVISED_SAFE_THRESHOLD = 2.60
    REVISED_DISTRESS_THRESHOLD = 1.10

    def __init__(self, model: AltmanModel = AltmanModel.ORIGINAL):
        """Initialize calculator with specified model variant.

        Args:
            model: Which Altman model variant to use
        """
        self.model = model
        self._name = "Altman Z-Score Calculator"
        self._description = (
            "Bankruptcy prediction model using five financial ratios. "
            "Z > 2.99 indicates safety, Z < 1.81 indicates distress."
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

        Args:
            data: Financial data to validate

        Returns:
            List of missing field names
        """
        required_fields = [
            ("total_assets", data.total_assets),
            ("current_assets", data.current_assets),
            ("current_liabilities", data.current_liabilities),
            ("retained_earnings", data.retained_earnings),
            ("operating_income", data.operating_income),  # EBIT
            ("market_cap", data.market_cap),
            ("total_liabilities", data.total_liabilities),
            ("revenue", data.revenue),
        ]

        missing = []
        for field_name, value in required_fields:
            if value is None:
                missing.append(field_name)

        return missing

    def calculate(self, data: FinancialData) -> AltmanZScoreResult:
        """Calculate Altman Z-Score from financial data.

        Args:
            data: Standardized financial data

        Returns:
            AltmanZScoreResult with score, zone, and components
        """
        result = AltmanZScoreResult(
            symbol=data.symbol,
            calculation_date=date.today(),
            data_date=data.data_date,
            model_used=self.model,
        )

        # Validate data
        missing = self.validate_data(data)
        if missing:
            result.warnings.extend([f"Missing field: {f}" for f in missing])

        # Check for zero total assets (would cause division by zero)
        if not data.total_assets or data.total_assets <= 0:
            result.warnings.append("Total assets must be positive")
            result.interpretation = "Cannot calculate - missing or invalid total assets"
            return result

        # Check for zero total liabilities (X4 calculation)
        if not data.total_liabilities or data.total_liabilities <= 0:
            result.warnings.append("Total liabilities must be positive for X4")

        try:
            # Calculate components
            components = self._calculate_components(data)
            result.components = components

            # Calculate Z-Score based on model
            if self.model == AltmanModel.ORIGINAL:
                z_score = self._calculate_original(components)
                result.score = round(z_score, 4) if z_score is not None else None
                result.zone = self._classify_original(z_score)
            elif self.model == AltmanModel.REVISED:
                z_score = self._calculate_revised(components)
                result.score = round(z_score, 4) if z_score is not None else None
                result.zone = self._classify_revised(z_score)
            else:
                # Default to original
                z_score = self._calculate_original(components)
                result.score = round(z_score, 4) if z_score is not None else None
                result.zone = self._classify_original(z_score)

            # Set interpretation
            result.interpretation = self._get_interpretation(result.zone, result.score)

            # Estimate bankruptcy probability
            result.bankruptcy_probability = self._estimate_bankruptcy_prob(z_score)

            logger.info(
                f"{data.symbol}: Altman Z-Score = {result.score} ({result.zone.value if result.zone else 'N/A'})"
            )

        except Exception as e:
            logger.error(f"Error calculating Altman Z-Score for {data.symbol}: {e}")
            result.warnings.append(f"Calculation error: {str(e)}")
            result.interpretation = "Calculation failed"

        return result

    def _calculate_components(self, data: FinancialData) -> Dict[str, Any]:
        """Calculate individual Z-Score components.

        Args:
            data: Financial data

        Returns:
            Dictionary of component values
        """
        components = {}
        ta = data.total_assets or 0

        # X1: Working Capital / Total Assets
        # Working Capital = Current Assets - Current Liabilities
        if data.working_capital is not None and ta > 0:
            components["X1_working_capital_ratio"] = data.working_capital / ta
        else:
            components["X1_working_capital_ratio"] = None

        # X2: Retained Earnings / Total Assets
        if data.retained_earnings is not None and ta > 0:
            components["X2_retained_earnings_ratio"] = data.retained_earnings / ta
        else:
            components["X2_retained_earnings_ratio"] = None

        # X3: EBIT / Total Assets (Return on Assets proxy)
        if data.ebit is not None and ta > 0:
            components["X3_ebit_ratio"] = data.ebit / ta
        else:
            components["X3_ebit_ratio"] = None

        # X4: Market Value of Equity / Total Liabilities
        if data.market_cap is not None and data.total_liabilities and data.total_liabilities > 0:
            components["X4_equity_liability_ratio"] = data.market_cap / data.total_liabilities
        else:
            components["X4_equity_liability_ratio"] = None

        # X5: Sales / Total Assets (Asset Turnover)
        if data.revenue is not None and ta > 0:
            components["X5_asset_turnover"] = data.revenue / ta
        else:
            components["X5_asset_turnover"] = None

        # Add raw values for transparency
        components["working_capital"] = data.working_capital
        components["total_assets"] = data.total_assets
        components["retained_earnings"] = data.retained_earnings
        components["ebit"] = data.ebit
        components["market_cap"] = data.market_cap
        components["total_liabilities"] = data.total_liabilities
        components["revenue"] = data.revenue

        return components

    def _calculate_original(self, components: Dict[str, Any]) -> Optional[float]:
        """Calculate original Altman Z-Score (manufacturing).

        Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
        """
        x1 = components.get("X1_working_capital_ratio")
        x2 = components.get("X2_retained_earnings_ratio")
        x3 = components.get("X3_ebit_ratio")
        x4 = components.get("X4_equity_liability_ratio")
        x5 = components.get("X5_asset_turnover")

        # Need at least X1, X3, and X4 for meaningful calculation
        if any(v is None for v in [x1, x3, x4]):
            return None

        # Use 0 for missing components (conservative)
        x1 = x1 or 0
        x2 = x2 or 0
        x3 = x3 or 0
        x4 = x4 or 0
        x5 = x5 or 0

        z_score = (
            self.COEF_X1 * x1 +
            self.COEF_X2 * x2 +
            self.COEF_X3 * x3 +
            self.COEF_X4 * x4 +
            self.COEF_X5 * x5
        )

        return z_score

    def _calculate_revised(self, components: Dict[str, Any]) -> Optional[float]:
        """Calculate revised Altman Z'-Score (non-manufacturing).

        Z' = 6.56×X1 + 3.26×X2 + 6.72×X3 + 1.05×X4
        Note: X5 (Sales/Assets) is excluded for non-manufacturing
        """
        x1 = components.get("X1_working_capital_ratio")
        x2 = components.get("X2_retained_earnings_ratio")
        x3 = components.get("X3_ebit_ratio")
        x4 = components.get("X4_equity_liability_ratio")

        if any(v is None for v in [x1, x3, x4]):
            return None

        x1 = x1 or 0
        x2 = x2 or 0
        x3 = x3 or 0
        x4 = x4 or 0

        z_score = (
            self.REVISED_COEF_X1 * x1 +
            self.REVISED_COEF_X2 * x2 +
            self.REVISED_COEF_X3 * x3 +
            self.REVISED_COEF_X4 * x4
        )

        return z_score

    def _classify_original(self, z_score: Optional[float]) -> Optional[AltmanZone]:
        """Classify Z-Score into risk zones (original model)."""
        if z_score is None:
            return None

        if z_score > self.SAFE_THRESHOLD:
            return AltmanZone.SAFE
        elif z_score < self.DISTRESS_THRESHOLD:
            return AltmanZone.DISTRESS
        else:
            return AltmanZone.GREY

    def _classify_revised(self, z_score: Optional[float]) -> Optional[AltmanZone]:
        """Classify Z-Score into risk zones (revised model)."""
        if z_score is None:
            return None

        if z_score > self.REVISED_SAFE_THRESHOLD:
            return AltmanZone.SAFE
        elif z_score < self.REVISED_DISTRESS_THRESHOLD:
            return AltmanZone.DISTRESS
        else:
            return AltmanZone.GREY

    def _get_interpretation(self, zone: Optional[AltmanZone], score: Optional[float]) -> str:
        """Generate human-readable interpretation of the score."""
        if zone is None or score is None:
            return "Unable to calculate Z-Score due to missing data"

        score_str = f"{score:.2f}"

        if zone == AltmanZone.SAFE:
            return (
                f"Safe Zone (Z = {score_str}): Low bankruptcy risk. "
                "Company appears financially healthy with strong fundamentals."
            )
        elif zone == AltmanZone.GREY:
            return (
                f"Grey Zone (Z = {score_str}): Uncertain financial position. "
                "Company requires careful monitoring - neither clearly safe nor distressed."
            )
        else:  # DISTRESS
            return (
                f"Distress Zone (Z = {score_str}): High bankruptcy risk. "
                "Company shows signs of financial stress and potential default."
            )

    def _estimate_bankruptcy_prob(self, z_score: Optional[float]) -> Optional[float]:
        """Estimate bankruptcy probability from Z-Score.

        Based on Altman's research, approximate probabilities:
        Z > 3.0: ~0.5%
        Z = 2.7: ~5%
        Z = 1.8: ~50%
        Z < 1.0: ~95%
        """
        if z_score is None:
            return None

        # Logistic approximation
        # P(bankruptcy) ≈ 1 / (1 + exp(1.5 × (Z - 1.8)))
        import math
        try:
            prob = 1.0 / (1.0 + math.exp(1.5 * (z_score - 1.8)))
            return round(prob, 4)
        except (OverflowError, ValueError):
            return 0.0 if z_score > 1.8 else 1.0
