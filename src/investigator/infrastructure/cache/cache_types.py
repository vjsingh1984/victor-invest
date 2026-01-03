#!/usr/bin/env python3
"""
InvestiGator - Cache Type Definitions
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Cache type definitions for InvestiGator caching system
"""

from enum import Enum


class CacheType(Enum):
    """Types of cache storage"""

    SEC_RESPONSE = "sec_response"
    LLM_RESPONSE = "llm_response"  # Stores both prompt and response together
    TECHNICAL_DATA = "technical_data"
    SUBMISSION_DATA = "submission_data"  # For all_submission_store - RDBMS only
    COMPANY_FACTS = "company_facts"
    QUARTERLY_METRICS = "quarterly_metrics"
    MARKET_CONTEXT = "market_context"  # Market-wide and sector data (macro indicators, ETFs)
    INDUSTRY_METRICS = "industry_metrics"  # Industry-specific metrics from IndustryDatasetRegistry
