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

"""Credit Risk Models Package.

This package provides financial distress and quality scoring models for
investment analysis:

- Altman Z-Score: Bankruptcy prediction model
- Beneish M-Score: Earnings manipulation detection
- Piotroski F-Score: Financial strength indicator
- Composite Distress: Aggregated risk assessment

All models follow SOLID principles:
- Single Responsibility: Each model calculates one type of score
- Open/Closed: New models can be added without modifying existing code
- Liskov Substitution: All models implement the CreditScoreCalculator protocol
- Interface Segregation: Clean protocol with minimal required methods
- Dependency Inversion: High-level modules depend on abstractions

Example:
    from investigator.domain.services.credit_risk import (
        get_credit_risk_service,
        calculate_all_scores,
    )

    # Get comprehensive credit risk assessment
    service = get_credit_risk_service()
    result = await service.calculate_all(symbol="AAPL", financial_data=data)

    print(f"Altman Z: {result.altman_zscore}")
    print(f"Beneish M: {result.beneish_mscore}")
    print(f"Piotroski F: {result.piotroski_fscore}")
    print(f"Distress Probability: {result.composite_distress_prob}")
"""

from investigator.domain.services.credit_risk.altman_zscore import (
    AltmanZone,
    AltmanZScoreCalculator,
    AltmanZScoreResult,
)
from investigator.domain.services.credit_risk.beneish_mscore import (
    BeneishMScoreCalculator,
    BeneishMScoreResult,
    ManipulationRisk,
)
from investigator.domain.services.credit_risk.composite_distress import (
    CompositeCreditRiskResult,
    CompositeDistressCalculator,
    DistressTier,
)
from investigator.domain.services.credit_risk.piotroski_fscore import (
    FinancialStrength,
    PiotroskiFScoreCalculator,
    PiotroskiFScoreResult,
)
from investigator.domain.services.credit_risk.protocols import (
    CreditScoreCalculator,
    CreditScoreResult,
    FinancialData,
)
from investigator.domain.services.credit_risk.service import (
    CreditRiskService,
    get_credit_risk_service,
)

__all__ = [
    # Protocols and base types
    "CreditScoreCalculator",
    "CreditScoreResult",
    "FinancialData",
    # Altman Z-Score
    "AltmanZScoreCalculator",
    "AltmanZScoreResult",
    "AltmanZone",
    # Beneish M-Score
    "BeneishMScoreCalculator",
    "BeneishMScoreResult",
    "ManipulationRisk",
    # Piotroski F-Score
    "PiotroskiFScoreCalculator",
    "PiotroskiFScoreResult",
    "FinancialStrength",
    # Composite Distress
    "CompositeDistressCalculator",
    "CompositeCreditRiskResult",
    "DistressTier",
    # Service
    "CreditRiskService",
    "get_credit_risk_service",
]
