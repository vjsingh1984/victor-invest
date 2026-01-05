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

"""Investment-specific RoleToolProvider for subagent tool configuration.

This module provides investment-domain tools for subagent roles,
following Victor's RoleToolProvider protocol for OCP compliance.

Usage:
    from victor_invest.role_provider import InvestmentRoleProvider
    from victor.agent.subagents.protocols import set_role_tool_provider

    # Register globally
    set_role_tool_provider(InvestmentRoleProvider())

    # Or use directly
    provider = InvestmentRoleProvider()
    tools = provider.get_tools_for_role("researcher", vertical="investment")
"""

from typing import Dict, List, Optional


class InvestmentRoleProvider:
    """Role tool provider for investment analysis vertical.

    Provides investment-specific tools for subagent roles instead of
    the default coding-focused tools. This enables proper synthesis
    and analysis using SEC filing, valuation, and market data tools.

    Implements RoleToolProvider protocol.
    """

    # Core tools available to all investment roles
    CORE_TOOLS = ["read"]

    # Investment-specific tools by role
    ROLE_TOOLS: Dict[str, List[str]] = {
        # Researcher: Read-only analysis, no data modification
        "researcher": [
            "read",  # Generic file reading
            # Investment-specific tools are accessed via handlers, not LLM tools
            # The agent node uses goal/context for synthesis, not tool calling
        ],
        # Planner: Task decomposition
        "planner": [
            "read",
        ],
        # Executor: Can modify data (not typically used in investment)
        "executor": [
            "read",
            "write",
        ],
        # Reviewer: Quality checks
        "reviewer": [
            "read",
        ],
        # Tester: Backtesting (special for investment)
        "tester": [
            "read",
        ],
    }

    # Tool budgets (lower for investment since synthesis is context-based)
    ROLE_BUDGETS: Dict[str, int] = {
        "researcher": 5,  # Minimal - synthesis is via context, not tools
        "planner": 5,
        "executor": 10,
        "reviewer": 5,
        "tester": 10,
    }

    # Context limits (higher for investment to include analysis data)
    ROLE_CONTEXT_LIMITS: Dict[str, int] = {
        "researcher": 100000,  # Large - needs full analysis context
        "planner": 50000,
        "executor": 80000,
        "reviewer": 60000,
        "tester": 80000,
    }

    def get_tools_for_role(
        self,
        role: str,
        vertical: Optional[str] = None,
    ) -> List[str]:
        """Get investment-appropriate tools for a role.

        For investment synthesis, we rely on context stuffing rather than
        tool calling. The agent receives all analysis data via the goal
        template and generates a structured recommendation.

        Args:
            role: Role name (researcher, planner, etc.)
            vertical: Vertical name (ignored - always returns investment tools)

        Returns:
            List of tool names appropriate for investment analysis
        """
        role_lower = role.lower()
        return self.ROLE_TOOLS.get(role_lower, self.CORE_TOOLS)

    def get_budget_for_role(self, role: str) -> int:
        """Get tool budget for role.

        Investment roles have lower budgets since synthesis is primarily
        context-based (all data provided in the prompt).

        Args:
            role: Role name

        Returns:
            Tool budget for the role
        """
        return self.ROLE_BUDGETS.get(role.lower(), 5)

    def get_context_limit_for_role(self, role: str) -> int:
        """Get context limit for role.

        Investment roles have higher context limits to accommodate
        comprehensive analysis data (SEC filings, valuations, etc.).

        Args:
            role: Role name

        Returns:
            Context character limit for the role
        """
        return self.ROLE_CONTEXT_LIMITS.get(role.lower(), 100000)


def register_investment_role_provider() -> None:
    """Register the InvestmentRoleProvider globally.

    Call this when initializing the investment vertical to ensure
    subagents use investment-appropriate tool configurations.

    Example:
        from victor_invest.role_provider import register_investment_role_provider
        register_investment_role_provider()
    """
    from victor.agent.subagents.protocols import set_role_tool_provider

    set_role_tool_provider(InvestmentRoleProvider())


__all__ = [
    "InvestmentRoleProvider",
    "register_investment_role_provider",
]
