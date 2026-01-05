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

"""Fundamental Analyst Agent Specification.

Defines the agent responsible for company valuation
and financial health analysis.
"""

from victor.agents.spec import (
    AgentCapabilities,
    AgentConstraints,
    AgentSpec,
    ModelPreference,
    OutputFormat,
)

from victor_invest.prompts.investment_prompts import FUNDAMENTAL_ANALYST_PROMPT

FUNDAMENTAL_AGENT_SPEC = AgentSpec(
    name="fundamental_analyst",
    description="Fundamental analysis specialist focusing on company valuation and financial health",
    capabilities=AgentCapabilities(
        tools={"valuation", "sec_filing", "cache"},
        can_browse_web=False,
        can_execute_code=True,  # May need calculations
        can_modify_files=False,
        can_delegate=False,
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=40,
        max_tool_calls=80,
        timeout_seconds=600.0,  # Valuation models can be complex
    ),
    model_preference=ModelPreference.REASONING,
    output_format=OutputFormat.STRUCTURED,
    system_prompt=FUNDAMENTAL_ANALYST_PROMPT,
    version="1.0",
    tags={"investment", "fundamental", "valuation", "financial_analysis"},
    metadata={
        "domain": "investment",
        "specialty": "fundamental_analysis",
        "valuation_models": ["DCF", "P/E", "P/S", "P/B", "EV/EBITDA", "Gordon Growth"],
    },
)
