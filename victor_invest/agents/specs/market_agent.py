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

"""Market Analyst Agent Specification.

Defines the agent responsible for analyzing sector dynamics
and macro factors.
"""

from victor.agent.specs.models import (
    AgentCapabilities,
    AgentConstraints,
    AgentSpec,
    ModelPreference,
    OutputFormat,
)

from victor_invest.prompts.investment_prompts import MARKET_ANALYST_PROMPT

MARKET_AGENT_SPEC = AgentSpec(
    name="market_analyst",
    description="Market context analyst focusing on sector dynamics and macro factors",
    capabilities=AgentCapabilities(
        tools={"market_data", "cache"},
        can_browse_web=True,  # May need to fetch market news
        can_execute_code=False,
        can_modify_files=False,
        can_delegate=False,
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=20,
        max_tool_calls=40,
        timeout_seconds=180.0,
    ),
    model_preference=ModelPreference.FAST,  # Market data queries are time-sensitive
    output_format=OutputFormat.STRUCTURED,
    system_prompt=MARKET_ANALYST_PROMPT,
    version="1.0",
    tags={"investment", "market", "sector", "macro"},
    metadata={
        "domain": "investment",
        "specialty": "market_context",
        "analysis_areas": ["sector", "macro", "cross_asset", "regime"],
    },
)
