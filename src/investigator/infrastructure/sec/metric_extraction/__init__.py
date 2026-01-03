"""
Metric Extraction Framework - SOLID-based Architecture

This module provides a robust, extensible framework for extracting financial metrics
from SEC XBRL data with multi-level fallback chains and flexible period matching.

Design Principles (SOLID):
- Single Responsibility: Each class has one job
- Open/Closed: New strategies can be added without modifying core
- Liskov Substitution: All strategies implement common interfaces
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions, inject via configuration

Architecture:
    MetricExtractionOrchestrator
        ├── PeriodMatcherChain (tries multiple matching strategies)
        │   ├── ByPeriodEndMatcher (most reliable - matches by exact date)
        │   ├── ByDateRangeMatcher (fuzzy date matching)
        │   ├── ByFrameFieldMatcher (uses CY2024Q3 frame)
        │   └── ByAdshFyFpMatcher (legacy - uses unreliable fy field)
        ├── TagFallbackChain (sector-aware XBRL tag resolution)
        └── ExtractionResult (value + metadata + audit trail)

Usage:
    from investigator.infrastructure.sec.metric_extraction import (
        MetricExtractionOrchestrator,
        ExtractionResult,
    )

    orchestrator = MetricExtractionOrchestrator(
        sector='Technology',
        industry='Electronic Components'
    )

    result = orchestrator.extract(
        canonical_key='total_revenue',
        us_gaap=company_facts['facts']['us-gaap'],
        target_period_end='2025-06-27',
        target_fiscal_period='FY'
    )

    if result.success:
        print(f"Revenue: {result.value:,.0f} (via {result.match_method})")
"""

from .strategies import (
    PeriodMatchStrategy,
    ByPeriodEndMatcher,
    ByDateRangeMatcher,
    ByFrameFieldMatcher,
    ByAdshFyFpMatcher,
)
from .result import ExtractionResult, ExtractionAudit
from .orchestrator import MetricExtractionOrchestrator

__all__ = [
    # Strategies
    'PeriodMatchStrategy',
    'ByPeriodEndMatcher',
    'ByDateRangeMatcher',
    'ByFrameFieldMatcher',
    'ByAdshFyFpMatcher',
    # Result types
    'ExtractionResult',
    'ExtractionAudit',
    # Orchestrator
    'MetricExtractionOrchestrator',
]
