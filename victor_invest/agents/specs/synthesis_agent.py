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

"""Synthesis Agent Specification.

Defines the agent responsible for combining multiple analysis
streams into actionable investment recommendations.
"""

from victor.agent.specs.models import (
    AgentCapabilities,
    AgentConstraints,
    AgentSpec,
    ModelPreference,
    OutputFormat,
)

from victor_invest.prompts.investment_prompts import SYNTHESIS_PROMPT

SYNTHESIS_AGENT_SPEC = AgentSpec(
    name="synthesis_analyst",
    description="Investment synthesis specialist combining analysis streams into actionable recommendations",
    capabilities=AgentCapabilities(
        tools={"sec_filing", "valuation", "technical_indicators", "market_data", "cache"},
        can_browse_web=False,
        can_execute_code=True,  # May need to aggregate scores
        can_modify_files=False,
        can_delegate=True,  # Can delegate to specialized analysts
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=50,
        max_tool_calls=100,
        timeout_seconds=900.0,  # Synthesis is comprehensive
    ),
    model_preference=ModelPreference.REASONING,  # Needs strong reasoning for synthesis
    output_format=OutputFormat.STRUCTURED,
    system_prompt=SYNTHESIS_PROMPT,
    version="1.0",
    tags={"investment", "synthesis", "recommendation", "orchestration"},
    metadata={
        "domain": "investment",
        "specialty": "synthesis",
        "weight_distribution": {
            "fundamental": 0.35,
            "technical": 0.20,
            "market_context": 0.15,
            "sentiment": 0.15,
            "sec_quality": 0.15,
        },
        "decision_thresholds": {
            "strong_buy": 80,
            "buy": 65,
            "hold_upper": 65,
            "hold_lower": 35,
            "sell": 35,
            "strong_sell": 20,
        },
    },
)
