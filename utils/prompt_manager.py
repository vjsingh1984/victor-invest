#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.application.prompts.prompt_manager
Migration Date: 2025-11-13
Phase: Phase 3-B-1 (High-Priority Application Layer Migration)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The prompt management logic (673 lines) is now properly located in the
application layer as orchestration logic, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.application.prompts import (
    PromptManager,
    get_prompt_manager,
    get_enhanced_prompt_manager,
)

__all__ = [
    'PromptManager',
    'get_prompt_manager',
    'get_enhanced_prompt_manager',
]
