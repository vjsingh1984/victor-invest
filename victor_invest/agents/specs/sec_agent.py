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

"""SEC Analyst Agent Specification.

Defines the agent responsible for extracting and analyzing
financial data from SEC regulatory filings.
"""

from victor.agents.spec import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
    OutputFormat,
)

from victor_invest.prompts.investment_prompts import SEC_ANALYST_PROMPT


SEC_AGENT_SPEC = AgentSpec(
    name="sec_analyst",
    description="SEC filings analyst specializing in extracting financial data from regulatory filings",
    capabilities=AgentCapabilities(
        tools={"sec_filing", "cache"},
        can_browse_web=True,  # May need to access SEC EDGAR
        can_execute_code=False,
        can_modify_files=False,
        can_delegate=False,
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=30,
        max_tool_calls=50,
        timeout_seconds=300.0,
    ),
    model_preference=ModelPreference.REASONING,
    output_format=OutputFormat.STRUCTURED,
    system_prompt=SEC_ANALYST_PROMPT,
    version="1.0",
    tags={"investment", "sec", "filings", "data_extraction"},
    metadata={
        "domain": "investment",
        "specialty": "sec_filings",
        "data_sources": ["SEC EDGAR", "XBRL"],
    },
)
