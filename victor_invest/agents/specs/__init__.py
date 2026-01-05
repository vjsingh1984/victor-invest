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

"""Investment Agent Specifications.

Exports all agent specs for the Victor-Invest framework:
- SEC_AGENT_SPEC: SEC filings analyst
- FUNDAMENTAL_AGENT_SPEC: Fundamental analysis specialist
- TECHNICAL_AGENT_SPEC: Technical analysis specialist
- MARKET_AGENT_SPEC: Market context analyst
- SYNTHESIS_AGENT_SPEC: Investment synthesis and recommendation
"""

from victor_invest.agents.specs.fundamental_agent import FUNDAMENTAL_AGENT_SPEC
from victor_invest.agents.specs.market_agent import MARKET_AGENT_SPEC
from victor_invest.agents.specs.sec_agent import SEC_AGENT_SPEC
from victor_invest.agents.specs.synthesis_agent import SYNTHESIS_AGENT_SPEC
from victor_invest.agents.specs.technical_agent import TECHNICAL_AGENT_SPEC

__all__ = [
    "SEC_AGENT_SPEC",
    "FUNDAMENTAL_AGENT_SPEC",
    "TECHNICAL_AGENT_SPEC",
    "MARKET_AGENT_SPEC",
    "SYNTHESIS_AGENT_SPEC",
]
