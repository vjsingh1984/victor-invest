#!/usr/bin/env python3
"""
InvestiGator - Application Layer Processors
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Data processors for SEC submissions, LLM responses, and other application-level processing
"""

from .submission_processor import (
    Filing,
    SubmissionProcessor,
    get_submission_processor,
)
from .llm_response_processor import (
    LLMResponseProcessor,
    get_llm_response_processor,
)

__all__ = [
    "Filing",
    "SubmissionProcessor",
    "get_submission_processor",
    "LLMResponseProcessor",
    "get_llm_response_processor",
]
