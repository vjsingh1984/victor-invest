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

"""Technical Analyst Agent Specification.

Defines the agent responsible for interpreting
price action and market structure.
"""

from victor.agent.specs.models import (
    AgentCapabilities,
    AgentConstraints,
    AgentSpec,
    ModelPreference,
    OutputFormat,
)

from victor_invest.prompts.investment_prompts import TECHNICAL_ANALYST_PROMPT

TECHNICAL_AGENT_SPEC = AgentSpec(
    name="technical_analyst",
    description="Technical analysis specialist interpreting price action and market structure",
    capabilities=AgentCapabilities(
        tools={"technical_indicators", "market_data", "cache"},
        can_browse_web=False,
        can_execute_code=True,  # May need indicator calculations
        can_modify_files=False,
        can_delegate=False,
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=25,
        max_tool_calls=60,
        timeout_seconds=180.0,  # Technical analysis is typically faster
    ),
    model_preference=ModelPreference.FAST,  # Technical analysis needs quick responses
    output_format=OutputFormat.STRUCTURED,
    system_prompt=TECHNICAL_ANALYST_PROMPT,
    version="1.0",
    tags={"investment", "technical", "charts", "indicators"},
    metadata={
        "domain": "investment",
        "specialty": "technical_analysis",
        "indicator_categories": ["trend", "momentum", "volatility", "volume"],
    },
)
