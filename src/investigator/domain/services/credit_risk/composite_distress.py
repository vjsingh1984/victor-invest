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

"""Composite Distress Calculator.

This module combines multiple credit risk models (Altman Z-Score, Beneish M-Score,
Piotroski F-Score) into a unified distress assessment. The composite approach
provides a more robust evaluation by leveraging the strengths of each model:

- Altman Z-Score: Bankruptcy prediction (quantitative risk)
- Beneish M-Score: Earnings manipulation detection (accounting quality)
- Piotroski F-Score: Financial strength assessment (fundamental health)

Distress Tiers:
    1. Healthy: Strong on all metrics
    2. Watch: Minor concerns in 1-2 areas
    3. Caution: Moderate concerns across metrics
    4. Warning: Significant concerns - close monitoring
    5. Critical: Multiple severe indicators - high risk

Integration with Valuation:
    Distress tier can be used to apply risk-adjusted discounts:
    - Healthy: No discount
    - Watch: 5% valuation discount
    - Caution: 15% valuation discount
    - Warning: 30% valuation discount
    - Critical: 50% valuation discount
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from investigator.domain.services.credit_risk.protocols import (
    CreditScoreResult,
    FinancialData,
)
from investigator.domain.services.credit_risk.altman_zscore import (
    AltmanZScoreCalculator,
    AltmanZScoreResult,
    AltmanZone,
)
from investigator.domain.services.credit_risk.beneish_mscore import (
    BeneishMScoreCalculator,
    BeneishMScoreResult,
    ManipulationRisk,
)
from investigator.domain.services.credit_risk.piotroski_fscore import (
    PiotroskiFScoreCalculator,
    PiotroskiFScoreResult,
    FinancialStrength,
)

logger = logging.getLogger(__name__)


class DistressTier(Enum):
    """Composite distress tier classification."""
    HEALTHY = 1     # Strong fundamentals, low risk
    WATCH = 2       # Minor concerns, requires monitoring
    CAUTION = 3     # Moderate concerns, increased risk
    WARNING = 4     # Significant concerns, high risk
    CRITICAL = 5    # Severe distress, very high risk

    @property
    def valuation_discount(self) -> float:
        """Get recommended valuation discount percentage."""
        discounts = {
            DistressTier.HEALTHY: 0.0,
            DistressTier.WATCH: 0.05,
            DistressTier.CAUTION: 0.15,
            DistressTier.WARNING: 0.30,
            DistressTier.CRITICAL: 0.50,
        }
        return discounts.get(self, 0.0)

    @property
    def signal(self) -> str:
        """Get investment signal for this tier."""
        signals = {
            DistressTier.HEALTHY: "green",
            DistressTier.WATCH: "yellow",
            DistressTier.CAUTION: "orange",
            DistressTier.WARNING: "red",
            DistressTier.CRITICAL: "dark_red",
        }
        return signals.get(self, "unknown")


@dataclass
class CompositeCreditRiskResult(CreditScoreResult):
    """Composite credit risk assessment result.

    Combines results from all three credit risk models into a unified
    distress assessment with actionable signals for valuation.

    Attributes:
        distress_tier: Overall distress classification
        distress_probability: Composite probability of financial distress
        valuation_discount: Recommended discount to apply to valuations
        altman_result: Full Altman Z-Score result
        beneish_result: Full Beneish M-Score result
        piotroski_result: Full Piotroski F-Score result
        risk_factors: List of identified risk factors
        positive_factors: List of positive indicators
    """
    distress_tier: Optional[DistressTier] = None
    distress_probability: Optional[float] = None
    valuation_discount: float = 0.0
    altman_result: Optional[AltmanZScoreResult] = None
    beneish_result: Optional[BeneishMScoreResult] = None
    piotroski_result: Optional[PiotroskiFScoreResult] = None
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)
    score_name: str = "Composite Credit Risk"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = super().to_dict()
        result.update({
            "distress_tier": self.distress_tier.name if self.distress_tier else None,
            "distress_tier_value": self.distress_tier.value if self.distress_tier else None,
            "distress_probability": self.distress_probability,
            "valuation_discount": self.valuation_discount,
            "valuation_discount_pct": f"{self.valuation_discount * 100:.0f}%",
            "signal": self.distress_tier.signal if self.distress_tier else "unknown",
            "risk_factors": self.risk_factors,
            "positive_factors": self.positive_factors,
            "individual_scores": {
                "altman_zscore": self.altman_result.to_dict() if self.altman_result else None,
                "beneish_mscore": self.beneish_result.to_dict() if self.beneish_result else None,
                "piotroski_fscore": self.piotroski_result.to_dict() if self.piotroski_result else None,
            },
        })
        return result


class CompositeDistressCalculator:
    """Calculator for composite credit risk assessment.

    Combines Altman Z-Score, Beneish M-Score, and Piotroski F-Score into
    a unified distress tier with actionable valuation implications.

    SOLID Principles:
    - Single Responsibility: Combines individual scores into composite assessment
    - Open/Closed: New individual calculators can be added without modification
    - Dependency Inversion: Depends on calculator protocols, not implementations
    """

    def __init__(
        self,
        altman_calculator: Optional[AltmanZScoreCalculator] = None,
        beneish_calculator: Optional[BeneishMScoreCalculator] = None,
        piotroski_calculator: Optional[PiotroskiFScoreCalculator] = None,
    ):
        """Initialize composite calculator with optional custom calculators.

        Args:
            altman_calculator: Custom Altman calculator (or default)
            beneish_calculator: Custom Beneish calculator (or default)
            piotroski_calculator: Custom Piotroski calculator (or default)
        """
        self._altman = altman_calculator or AltmanZScoreCalculator()
        self._beneish = beneish_calculator or BeneishMScoreCalculator()
        self._piotroski = piotroski_calculator or PiotroskiFScoreCalculator()

        self._name = "Composite Distress Calculator"
        self._description = (
            "Unified credit risk assessment combining Altman Z-Score, "
            "Beneish M-Score, and Piotroski F-Score into a single distress tier."
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
        """Validate data for all underlying calculators."""
        missing = set()
        missing.update(self._altman.validate_data(data))
        missing.update(self._beneish.validate_data(data))
        missing.update(self._piotroski.validate_data(data))
        return list(missing)

    def calculate(self, data: FinancialData) -> CompositeCreditRiskResult:
        """Calculate composite credit risk assessment.

        Args:
            data: Standardized financial data

        Returns:
            CompositeCreditRiskResult with tier, probability, and recommendations
        """
        result = CompositeCreditRiskResult(
            symbol=data.symbol,
            calculation_date=date.today(),
            data_date=data.data_date,
        )

        try:
            # Calculate individual scores
            altman_result = self._altman.calculate(data)
            beneish_result = self._beneish.calculate(data)
            piotroski_result = self._piotroski.calculate(data)

            result.altman_result = altman_result
            result.beneish_result = beneish_result
            result.piotroski_result = piotroski_result

            # Collect warnings from all calculators
            result.warnings.extend(altman_result.warnings)
            result.warnings.extend(beneish_result.warnings)
            result.warnings.extend(piotroski_result.warnings)

            # Identify risk and positive factors
            risk_factors, positive_factors = self._analyze_factors(
                altman_result, beneish_result, piotroski_result
            )
            result.risk_factors = risk_factors
            result.positive_factors = positive_factors

            # Calculate composite distress tier
            tier = self._calculate_tier(altman_result, beneish_result, piotroski_result)
            result.distress_tier = tier
            result.valuation_discount = tier.valuation_discount if tier else 0.0

            # Calculate composite distress probability
            result.distress_probability = self._calculate_composite_probability(
                altman_result, beneish_result, piotroski_result
            )

            # Set score as tier value (1-5)
            result.score = tier.value if tier else None

            # Store components
            result.components = {
                "altman_zscore": altman_result.score,
                "altman_zone": altman_result.zone.value if altman_result.zone else None,
                "beneish_mscore": beneish_result.score,
                "beneish_risk": beneish_result.risk_level.value if beneish_result.risk_level else None,
                "piotroski_fscore": piotroski_result.score,
                "piotroski_strength": piotroski_result.strength.value if piotroski_result.strength else None,
            }

            # Generate interpretation
            result.interpretation = self._get_interpretation(result)

            logger.info(
                f"{data.symbol}: Composite Distress Tier = {tier.name if tier else 'N/A'} "
                f"(Discount: {result.valuation_discount * 100:.0f}%)"
            )

        except Exception as e:
            logger.error(f"Error calculating composite distress for {data.symbol}: {e}")
            result.warnings.append(f"Calculation error: {str(e)}")
            result.interpretation = "Calculation failed"

        return result

    def _analyze_factors(
        self,
        altman: AltmanZScoreResult,
        beneish: BeneishMScoreResult,
        piotroski: PiotroskiFScoreResult,
    ) -> tuple[List[str], List[str]]:
        """Analyze individual results for risk and positive factors."""
        risk_factors = []
        positive_factors = []

        # Altman Z-Score analysis
        if altman.zone == AltmanZone.DISTRESS:
            risk_factors.append(f"Bankruptcy risk elevated (Z={altman.score:.2f})")
        elif altman.zone == AltmanZone.GREY:
            risk_factors.append(f"Financial position uncertain (Z={altman.score:.2f})")
        elif altman.zone == AltmanZone.SAFE:
            positive_factors.append(f"Low bankruptcy risk (Z={altman.score:.2f})")

        # Beneish M-Score analysis
        if beneish.risk_level == ManipulationRisk.HIGH:
            risk_factors.append(f"Earnings manipulation risk (M={beneish.score:.2f})")
        elif beneish.risk_level == ManipulationRisk.LOW:
            positive_factors.append(f"Clean accounting signals (M={beneish.score:.2f})")

        # Piotroski F-Score analysis
        if piotroski.strength == FinancialStrength.WEAK:
            risk_factors.append(f"Weak fundamentals (F={piotroski.score}/9)")
        elif piotroski.strength == FinancialStrength.STRONG:
            positive_factors.append(f"Strong fundamentals (F={piotroski.score}/9)")

        # Component-level analysis from Piotroski
        if piotroski.criteria_details:
            # Check specific concerns
            if not piotroski.criteria_details.get("F1_positive_roa", True):
                risk_factors.append("Negative return on assets")
            if not piotroski.criteria_details.get("F2_positive_cfo", True):
                risk_factors.append("Negative operating cash flow")
            if not piotroski.criteria_details.get("F4_quality_earnings", True):
                risk_factors.append("Poor earnings quality (accruals > cash)")
            if not piotroski.criteria_details.get("F5_leverage_decreasing", True):
                risk_factors.append("Leverage increasing")

            # Check specific positives
            if piotroski.criteria_details.get("F3_roa_improving", False):
                positive_factors.append("Improving profitability")
            if piotroski.criteria_details.get("F6_current_ratio_improving", False):
                positive_factors.append("Improving liquidity")
            if piotroski.criteria_details.get("F7_no_dilution", False):
                positive_factors.append("No share dilution")

        return risk_factors, positive_factors

    def _calculate_tier(
        self,
        altman: AltmanZScoreResult,
        beneish: BeneishMScoreResult,
        piotroski: PiotroskiFScoreResult,
    ) -> DistressTier:
        """Calculate composite distress tier from individual scores.

        Scoring logic:
        - Each severe indicator adds +2 to risk score
        - Each moderate indicator adds +1 to risk score
        - Each positive indicator subtracts -0.5 from risk score

        Final tier based on risk score:
        - < 0.5: HEALTHY
        - 0.5 - 1.5: WATCH
        - 1.5 - 3.0: CAUTION
        - 3.0 - 4.5: WARNING
        - >= 4.5: CRITICAL
        """
        risk_score = 0.0

        # Altman Z-Score contribution
        if altman.zone == AltmanZone.DISTRESS:
            risk_score += 2.0
        elif altman.zone == AltmanZone.GREY:
            risk_score += 1.0
        elif altman.zone == AltmanZone.SAFE:
            risk_score -= 0.5

        # Beneish M-Score contribution
        if beneish.risk_level == ManipulationRisk.HIGH:
            risk_score += 2.0
        elif beneish.risk_level == ManipulationRisk.MODERATE:
            risk_score += 0.5
        elif beneish.risk_level == ManipulationRisk.LOW:
            risk_score -= 0.5

        # Piotroski F-Score contribution
        if piotroski.strength == FinancialStrength.WEAK:
            risk_score += 1.5
        elif piotroski.strength == FinancialStrength.MODERATE:
            risk_score += 0.5
        elif piotroski.strength == FinancialStrength.STRONG:
            risk_score -= 0.5

        # Map risk score to tier
        if risk_score < 0.5:
            return DistressTier.HEALTHY
        elif risk_score < 1.5:
            return DistressTier.WATCH
        elif risk_score < 3.0:
            return DistressTier.CAUTION
        elif risk_score < 4.5:
            return DistressTier.WARNING
        else:
            return DistressTier.CRITICAL

    def _calculate_composite_probability(
        self,
        altman: AltmanZScoreResult,
        beneish: BeneishMScoreResult,
        piotroski: PiotroskiFScoreResult,
    ) -> Optional[float]:
        """Calculate composite distress probability.

        Weighted average of individual probabilities:
        - Altman bankruptcy probability: 40% weight
        - Beneish manipulation probability: 30% weight
        - Piotroski inverse strength: 30% weight
        """
        weights = []
        probs = []

        # Altman bankruptcy probability
        if altman.bankruptcy_probability is not None:
            probs.append(altman.bankruptcy_probability)
            weights.append(0.4)

        # Beneish manipulation probability
        if beneish.manipulation_probability is not None:
            probs.append(beneish.manipulation_probability)
            weights.append(0.3)

        # Piotroski inverse (higher score = lower risk)
        if piotroski.score is not None:
            # Convert 0-9 score to probability (9 = 0% risk, 0 = 100% risk)
            pio_prob = (9 - piotroski.score) / 9.0
            probs.append(pio_prob)
            weights.append(0.3)

        if not probs:
            return None

        # Weighted average
        total_weight = sum(weights)
        composite = sum(p * w for p, w in zip(probs, weights)) / total_weight

        return round(composite, 4)

    def _get_interpretation(self, result: CompositeCreditRiskResult) -> str:
        """Generate comprehensive interpretation."""
        if result.distress_tier is None:
            return "Unable to assess credit risk due to insufficient data"

        tier = result.distress_tier
        discount = result.valuation_discount * 100

        tier_descriptions = {
            DistressTier.HEALTHY: (
                f"HEALTHY: Company demonstrates strong credit profile. "
                f"No significant financial distress indicators. "
                f"No valuation discount recommended."
            ),
            DistressTier.WATCH: (
                f"WATCH: Minor credit concerns detected. "
                f"Fundamentals generally sound but warrant monitoring. "
                f"Recommend {discount:.0f}% valuation discount for safety margin."
            ),
            DistressTier.CAUTION: (
                f"CAUTION: Moderate credit concerns across multiple metrics. "
                f"Financial position requires careful analysis. "
                f"Recommend {discount:.0f}% valuation discount."
            ),
            DistressTier.WARNING: (
                f"WARNING: Significant credit deterioration detected. "
                f"Multiple distress indicators present. "
                f"Recommend {discount:.0f}% valuation discount and close monitoring."
            ),
            DistressTier.CRITICAL: (
                f"CRITICAL: Severe financial distress indicators. "
                f"High probability of adverse credit events. "
                f"Recommend {discount:.0f}% valuation discount. Consider avoiding."
            ),
        }

        base_interp = tier_descriptions.get(tier, "Unknown tier")

        # Add factor summary
        if result.risk_factors:
            base_interp += f" Key risks: {'; '.join(result.risk_factors[:3])}"
        if result.positive_factors:
            base_interp += f" Positives: {'; '.join(result.positive_factors[:2])}"

        return base_interp
