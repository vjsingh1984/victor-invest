#!/usr/bin/env python3
"""
IMPORT SHIM: This module has been migrated to clean architecture

Canonical Location: investigator.application.processors.llm_response_processor
Migration Date: 2025-11-13
Phase: Phase 3-B-1 (High-Priority Application Layer Migration)

This shim provides backward compatibility during migration.
All imports have been updated to use the canonical location.

The LLM response processing logic (331 lines) is now properly located in the
application layer as data processing orchestration, following clean architecture principles.
"""

# Re-export from canonical location
from investigator.application.processors import (
    LLMResponseProcessor,
    get_llm_response_processor,
)

__all__ = [
    'LLMResponseProcessor',
    'get_llm_response_processor',
]
