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

"""Market Regime Detection Services.

This package provides services for analyzing market regime indicators
including yield curve shape, credit cycle, recession probability, and
economic conditions.

Available Services:
- YieldCurveAnalyzer: Analyze treasury yield curve shape and implications
- RecessionIndicator: Assess recession probability and economic risk
- CreditCycleAnalyzer: Analyze credit cycle phase and conditions

Example:
    from investigator.domain.services.market_regime import (
        YieldCurveAnalyzer,
        RecessionIndicator,
        CreditCycleAnalyzer,
        get_yield_curve_analyzer,
        get_recession_indicator,
        get_credit_cycle_analyzer,
    )

    # Analyze yield curve
    analyzer = get_yield_curve_analyzer()
    analysis = await analyzer.analyze()

    # Get recession assessment
    indicator = get_recession_indicator()
    assessment = await indicator.assess()

    # Get credit cycle analysis
    cc_analyzer = get_credit_cycle_analyzer()
    cc_analysis = await cc_analyzer.analyze()
"""

from investigator.domain.services.market_regime.credit_cycle_analyzer import (
    CreditCycleAnalysis,
    CreditCycleAnalyzer,
    get_credit_cycle_analyzer,
)
from investigator.domain.services.market_regime.recession_indicator import (
    EconomicPhase,
    RecessionAssessment,
    RecessionIndicator,
    get_recession_indicator,
)
from investigator.domain.services.market_regime.yield_curve_analyzer import (
    InvestmentSignal,
    YieldCurveAnalysis,
    YieldCurveAnalyzer,
    YieldCurveShape,
    get_yield_curve_analyzer,
)

__all__ = [
    # Yield Curve
    "YieldCurveAnalyzer",
    "YieldCurveShape",
    "YieldCurveAnalysis",
    "InvestmentSignal",
    "get_yield_curve_analyzer",
    # Recession
    "RecessionIndicator",
    "RecessionAssessment",
    "EconomicPhase",
    "get_recession_indicator",
    # Credit Cycle
    "CreditCycleAnalyzer",
    "CreditCycleAnalysis",
    "get_credit_cycle_analyzer",
]
