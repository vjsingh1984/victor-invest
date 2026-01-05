#!/usr/bin/env python3
"""
InvestiGator - Application Layer Prompts
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Prompt management for LLM interactions
"""

from .prompt_manager import (
    PromptManager,
    get_enhanced_prompt_manager,
    get_prompt_manager,
)

__all__ = [
    "PromptManager",
    "get_prompt_manager",
    "get_enhanced_prompt_manager",
]
