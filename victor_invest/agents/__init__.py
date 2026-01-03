# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Agent specifications for Victor Investment analysis.

This module exports all investment agent specifications:
- SEC_AGENT_SPEC: SEC filings analyst
- FUNDAMENTAL_AGENT_SPEC: Fundamental/valuation analyst
- TECHNICAL_AGENT_SPEC: Technical analysis specialist
- MARKET_AGENT_SPEC: Market context analyst
- SYNTHESIS_AGENT_SPEC: Investment synthesis specialist
"""

from victor_invest.agents.specs import (
    SEC_AGENT_SPEC,
    FUNDAMENTAL_AGENT_SPEC,
    TECHNICAL_AGENT_SPEC,
    MARKET_AGENT_SPEC,
    SYNTHESIS_AGENT_SPEC,
)

__all__ = [
    "SEC_AGENT_SPEC",
    "FUNDAMENTAL_AGENT_SPEC",
    "TECHNICAL_AGENT_SPEC",
    "MARKET_AGENT_SPEC",
    "SYNTHESIS_AGENT_SPEC",
]
