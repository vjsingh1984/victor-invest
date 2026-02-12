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

"""Shared Victor framework bootstrap utilities for investment workflows."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from victor.core.verticals import VerticalRegistry
from victor.framework import Agent

from victor_invest.role_provider import register_investment_role_provider
from victor_invest.tools import register_investment_tools
from victor_invest.vertical import InvestmentVertical

logger = logging.getLogger(__name__)

DEFAULT_SYNTHESIS_MODEL = "gpt-oss:20b"


def resolve_investment_model(provider: str, model: Optional[str]) -> Optional[str]:
    """Resolve model for investment workflows with provider-aware defaults."""
    if model is not None:
        return model
    if provider != "ollama":
        return None
    try:
        from investigator.config import get_config

        config = get_config()
        return config.ollama.models.get("synthesis", DEFAULT_SYNTHESIS_MODEL)
    except Exception:
        return DEFAULT_SYNTHESIS_MODEL


def prepare_orchestrator_for_investment(
    orchestrator,
    warning_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Register and enable investment tools on a Victor orchestrator."""
    warn = warning_callback or logger.warning

    try:
        stats = register_investment_tools(orchestrator.tools)
        if stats.get("errors"):
            warn(f"Tool registration warnings: {stats['errors']}")
    except Exception as exc:
        warn(f"Investment tool registration failed: {exc}")
        return

    try:
        orchestrator.set_enabled_tools(set(InvestmentVertical.get_tools()))
        if hasattr(orchestrator, "tool_selector"):
            orchestrator.tool_selector.invalidate_tool_cache()
    except Exception as exc:
        warn(f"Tool enablement refresh skipped: {exc}")


async def create_investment_orchestrator(
    provider: str,
    model: Optional[str] = None,
    *,
    ensure_handlers: Optional[Callable[[], None]] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    warning_callback: Optional[Callable[[str], None]] = None,
):
    """Create Victor orchestrator preconfigured for investment workflows."""
    if ensure_handlers:
        ensure_handlers()

    if not VerticalRegistry.get("investment"):
        VerticalRegistry.register(InvestmentVertical)

    register_investment_role_provider()

    resolved_model = resolve_investment_model(provider, model)

    agent = await Agent.create(
        provider=provider,
        model=resolved_model,
        vertical=InvestmentVertical,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    orchestrator = agent.get_orchestrator()
    prepare_orchestrator_for_investment(orchestrator, warning_callback=warning_callback)
    return orchestrator


__all__ = [
    "DEFAULT_SYNTHESIS_MODEL",
    "resolve_investment_model",
    "prepare_orchestrator_for_investment",
    "create_investment_orchestrator",
]
