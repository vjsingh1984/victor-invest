"""
Metric Extraction Orchestrator

Coordinates multi-level fallback chains for robust metric extraction:
1. Period Matcher Chain - Tries multiple matching strategies
2. Tag Fallback Chain - Tries sector-aware XBRL tags
3. Derived Value Chain - Calculates from other metrics if direct extraction fails

SOLID Principles:
- Single Responsibility: Coordinates extraction, delegates to strategies
- Dependency Inversion: Depends on abstract PeriodMatchStrategy
- Open/Closed: New strategies can be injected via configuration
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .strategies import (
    PeriodMatchStrategy,
    ByPeriodEndMatcher,
    ByDateRangeMatcher,
    ByFrameFieldMatcher,
    ByAdshFyFpMatcher,
    ByAdshOnlyMatcher,
    MatchContext,
    MatchResult,
)
from .result import (
    ExtractionResult,
    ExtractionAudit,
    ExtractionAttempt,
    MatchMethod,
    ExtractionConfidence,
)

logger = logging.getLogger(__name__)


class MetricExtractionOrchestrator:
    """
    Orchestrates metric extraction with multi-level fallback chains.

    Strategy Priority (executed in order until success):
    1. ByPeriodEndMatcher - Most reliable, matches by exact end date
    2. ByDateRangeMatcher - Handles off-by-one date differences
    3. ByFrameFieldMatcher - Uses calendar-based frame field
    4. ByAdshOnlyMatcher - Uses ADSH with duration filtering
    5. ByAdshFyFpMatcher - Legacy, least reliable (fy field can be wrong)

    For each matching strategy, tries all XBRL tags in the fallback chain
    (sector-specific first, then global fallback).

    Usage:
        orchestrator = MetricExtractionOrchestrator(
            sector='Technology',
            industry='Electronic Components',
            canonical_mapper=get_canonical_mapper()
        )

        result = orchestrator.extract(
            canonical_key='total_revenue',
            us_gaap=company_facts['facts']['us-gaap'],
            target_period_end='2025-06-27',
            target_fiscal_period='FY'
        )

        if result.success:
            print(f"Revenue: {result.value:,.0f}")
        else:
            print(f"Extraction failed: {result.error}")
    """

    # Default matcher chain (ordered by reliability)
    DEFAULT_MATCHERS = [
        ByPeriodEndMatcher(),
        ByDateRangeMatcher(),
        ByFrameFieldMatcher(),
        ByAdshOnlyMatcher(),
        ByAdshFyFpMatcher(),
    ]

    def __init__(
        self,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        canonical_mapper=None,
        matchers: Optional[List[PeriodMatchStrategy]] = None,
        enable_audit: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            sector: Company sector for sector-specific tag resolution
            industry: Company industry for industry-specific tag resolution
            canonical_mapper: CanonicalKeyMapper instance (if None, will import)
            matchers: Custom matcher chain (if None, uses DEFAULT_MATCHERS)
            enable_audit: Whether to build detailed audit trails
        """
        self.sector = sector
        self.industry = industry
        self.matchers = matchers or self.DEFAULT_MATCHERS
        self.enable_audit = enable_audit

        # Initialize canonical mapper
        if canonical_mapper is None:
            from investigator.infrastructure.sec.canonical_mapper import get_canonical_mapper
            self.canonical_mapper = get_canonical_mapper()
        else:
            self.canonical_mapper = canonical_mapper

        # Statistics
        self.stats = {
            'extractions': 0,
            'successes': 0,
            'by_strategy': {},
            'by_tag_position': {},
        }

    def extract(
        self,
        canonical_key: str,
        us_gaap: Dict,
        target_period_end: Optional[str] = None,
        target_fiscal_year: Optional[int] = None,
        target_fiscal_period: Optional[str] = None,
        target_adsh: Optional[str] = None,
        fiscal_year_end: Optional[str] = None,
        tolerance_days: int = 7,
    ) -> ExtractionResult:
        """
        Extract a metric using multi-level fallback chains.

        Args:
            canonical_key: Canonical metric key (e.g., 'total_revenue')
            us_gaap: SEC us-gaap JSON structure
            target_period_end: Target period end date (YYYY-MM-DD) - MOST IMPORTANT
            target_fiscal_year: Target fiscal year (unreliable in SEC data)
            target_fiscal_period: Target fiscal period (FY, Q1, Q2, Q3, Q4)
            target_adsh: Target accession number
            fiscal_year_end: Company's fiscal year end (e.g., '-06-27')
            tolerance_days: Days tolerance for date range matching

        Returns:
            ExtractionResult with value, metadata, and audit trail
        """
        start_time = time.time()
        self.stats['extractions'] += 1

        # Create audit trail
        audit = ExtractionAudit(
            canonical_key=canonical_key,
            target_period_end=target_period_end,
            target_fiscal_year=target_fiscal_year,
            target_fiscal_period=target_fiscal_period,
            target_adsh=target_adsh,
            started_at=datetime.now().isoformat(),
        ) if self.enable_audit else None

        # Create match context
        context = MatchContext(
            target_period_end=target_period_end,
            target_fiscal_year=target_fiscal_year,
            target_fiscal_period=target_fiscal_period,
            target_adsh=target_adsh,
            fiscal_year_end=fiscal_year_end,
            tolerance_days=tolerance_days,
        )

        # Get tag fallback chain for this canonical key
        fallback_tags = self.canonical_mapper.get_tags(
            canonical_key,
            sector=self.sector,
            industry=self.industry
        )

        if not fallback_tags:
            logger.warning(f"No XBRL tags found for canonical key '{canonical_key}'")
            if audit:
                audit.completed_at = datetime.now().isoformat()
            return ExtractionResult.not_found(
                canonical_key,
                audit=audit,
                reason=f"No XBRL tags configured for '{canonical_key}'"
            )

        # Try each matcher strategy
        for matcher in self.matchers:
            # For each matcher, try each tag in the fallback chain
            for tag_position, tag_name in enumerate(fallback_tags):
                attempt_start = time.time()

                # Get entries for this tag
                if tag_name not in us_gaap:
                    if audit:
                        audit.add_attempt(ExtractionAttempt(
                            strategy_name=matcher.name,
                            tag_name=tag_name,
                            matched=False,
                            entries_found=0,
                            reason=f"Tag '{tag_name}' not in us_gaap",
                            duration_ms=(time.time() - attempt_start) * 1000
                        ))
                    continue

                tag_data = us_gaap[tag_name]
                units = tag_data.get('units', {})

                # Get expected unit (usually USD)
                mapping = self.canonical_mapper.mappings.get(canonical_key, {})
                expected_unit = mapping.get('unit', 'USD')
                entries = units.get(expected_unit, [])

                if not entries:
                    if audit:
                        audit.add_attempt(ExtractionAttempt(
                            strategy_name=matcher.name,
                            tag_name=tag_name,
                            matched=False,
                            entries_found=0,
                            reason=f"No {expected_unit} entries for tag",
                            duration_ms=(time.time() - attempt_start) * 1000
                        ))
                    continue

                # Try this matcher with this tag
                match_result = matcher.match(entries, context)

                if audit:
                    audit.add_attempt(ExtractionAttempt(
                        strategy_name=matcher.name,
                        tag_name=tag_name,
                        matched=match_result.matched,
                        entries_found=len(match_result.entries),
                        selected_entry=match_result.entries[0] if match_result.entries else None,
                        reason=match_result.reason,
                        duration_ms=(time.time() - attempt_start) * 1000
                    ))

                if match_result.matched and match_result.entries:
                    # Select best entry (prefer individual quarter over YTD)
                    best_entry = self._select_best_entry(
                        match_result.entries,
                        target_fiscal_period
                    )

                    if best_entry and best_entry.get('val') is not None:
                        value = best_entry['val']

                        # Determine confidence based on strategy and tag position
                        confidence = self._determine_confidence(
                            matcher,
                            tag_position,
                            len(fallback_tags)
                        )

                        # Update statistics
                        self.stats['successes'] += 1
                        self.stats['by_strategy'][matcher.name] = \
                            self.stats['by_strategy'].get(matcher.name, 0) + 1
                        self.stats['by_tag_position'][tag_position] = \
                            self.stats['by_tag_position'].get(tag_position, 0) + 1

                        if audit:
                            audit.completed_at = datetime.now().isoformat()

                        logger.debug(
                            f"✓ Extracted {canonical_key} = {value:,.0f} "
                            f"via {matcher.name} using tag '{tag_name}' "
                            f"(position {tag_position + 1}/{len(fallback_tags)})"
                        )

                        return ExtractionResult.from_entry(
                            value=value,
                            source_tag=tag_name,
                            entry=best_entry,
                            match_method=match_result.method,
                            confidence=confidence,
                            audit=audit
                        )

        # All strategies exhausted - try derived value calculation
        derived_result = self._try_derived_value(canonical_key, us_gaap, context, audit)
        if derived_result and derived_result.success:
            if audit:
                audit.completed_at = datetime.now().isoformat()
            return derived_result

        # Complete failure
        if audit:
            audit.completed_at = datetime.now().isoformat()

        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(
            f"✗ Failed to extract {canonical_key} for period_end={target_period_end} "
            f"after trying {len(self.matchers)} strategies × {len(fallback_tags)} tags "
            f"({elapsed_ms:.1f}ms)"
        )

        return ExtractionResult.not_found(
            canonical_key,
            audit=audit,
            reason=f"Exhausted {len(self.matchers)} matchers × {len(fallback_tags)} tags"
        )

    def _select_best_entry(
        self,
        entries: List[Dict],
        target_fiscal_period: Optional[str]
    ) -> Optional[Dict]:
        """
        Select best entry from matched entries.

        Preference:
        1. Individual quarter (< 120 days) over YTD
        2. Most recent filed date
        3. Entry with value (not None)
        """
        if not entries:
            return None

        if len(entries) == 1:
            return entries[0]

        # Categorize by duration
        individual = []
        ytd = []
        annual = []
        unknown = []

        for entry in entries:
            if entry.get('val') is None:
                continue

            start = entry.get('start')
            end = entry.get('end')

            if not start or not end:
                unknown.append(entry)
                continue

            try:
                start_date = datetime.strptime(start, '%Y-%m-%d')
                end_date = datetime.strptime(end, '%Y-%m-%d')
                days = (end_date - start_date).days

                if days < 120:
                    individual.append((entry, days))
                elif days < 270:
                    ytd.append((entry, days))
                else:
                    annual.append((entry, days))
            except ValueError:
                unknown.append(entry)

        # Select based on target fiscal period
        if target_fiscal_period == 'FY':
            # For FY, prefer annual entries
            if annual:
                annual.sort(key=lambda x: x[0].get('filed', ''), reverse=True)
                return annual[0][0]
            if ytd:
                ytd.sort(key=lambda x: x[1], reverse=True)  # Prefer longer duration
                return ytd[-1][0]  # Actually we want longest, so reverse sort
        else:
            # For quarters, prefer individual quarter entries
            if individual:
                individual.sort(key=lambda x: x[0].get('filed', ''), reverse=True)
                return individual[0][0]
            if ytd:
                # For YTD, we'd need to normalize - just return the entry
                ytd.sort(key=lambda x: x[0].get('filed', ''), reverse=True)
                return ytd[0][0]

        # Fallback to any entry with value
        if unknown:
            unknown.sort(key=lambda x: x.get('filed', ''), reverse=True)
            return unknown[0]

        return entries[0] if entries else None

    def _determine_confidence(
        self,
        matcher: PeriodMatchStrategy,
        tag_position: int,
        total_tags: int
    ) -> ExtractionConfidence:
        """
        Determine extraction confidence based on how value was obtained.

        High confidence:
        - ByPeriodEndMatcher with first tag (position 0)

        Medium confidence:
        - ByPeriodEndMatcher with fallback tag
        - ByDateRangeMatcher with any tag

        Low confidence:
        - ByAdshFyFpMatcher (unreliable fy field)
        - Deep fallback tags (position > 2)
        """
        # Strategy-based confidence
        high_confidence_matchers = ['ByPeriodEndMatcher']
        medium_confidence_matchers = ['ByDateRangeMatcher', 'ByFrameFieldMatcher', 'ByAdshOnlyMatcher']
        # ByAdshFyFpMatcher is low confidence

        if matcher.name in high_confidence_matchers:
            if tag_position == 0:
                return ExtractionConfidence.HIGH
            elif tag_position <= 2:
                return ExtractionConfidence.MEDIUM
            else:
                return ExtractionConfidence.LOW
        elif matcher.name in medium_confidence_matchers:
            if tag_position <= 1:
                return ExtractionConfidence.MEDIUM
            else:
                return ExtractionConfidence.LOW
        else:
            return ExtractionConfidence.LOW

    def _try_derived_value(
        self,
        canonical_key: str,
        us_gaap: Dict,
        context: MatchContext,
        audit: Optional[ExtractionAudit]
    ) -> Optional[ExtractionResult]:
        """
        Try to calculate derived value from other metrics.

        Example: free_cash_flow = operating_cash_flow - capital_expenditures
        """
        mapping = self.canonical_mapper.mappings.get(canonical_key, {})
        derived_config = mapping.get('derived', {})

        if not derived_config.get('enabled', False):
            return None

        formula = derived_config.get('formula')
        if not formula:
            return None

        # Get required fields
        required_fields = derived_config.get('required_fields', [])
        if not required_fields:
            return None

        # Extract required field values
        components = {}
        for field_key in required_fields:
            # Recursively extract (but don't allow derived to prevent loops)
            field_result = self.extract(
                canonical_key=field_key,
                us_gaap=us_gaap,
                target_period_end=context.target_period_end,
                target_fiscal_year=context.target_fiscal_year,
                target_fiscal_period=context.target_fiscal_period,
                target_adsh=context.target_adsh,
            )

            if field_result.success and field_result.value is not None:
                components[field_key] = field_result.value
            else:
                # Missing required component
                return None

        # Evaluate formula
        try:
            # Simple formula evaluation (supports +, -, *, /)
            value = self._evaluate_formula(formula, components)
            if value is not None:
                logger.debug(
                    f"✓ Derived {canonical_key} = {value:,.0f} "
                    f"from formula '{formula}'"
                )
                return ExtractionResult.derived(
                    value=value,
                    formula=formula,
                    components=components,
                    audit=audit
                )
        except Exception as e:
            logger.warning(f"Failed to evaluate formula '{formula}': {e}")

        return None

    def _evaluate_formula(self, formula: str, components: Dict[str, float]) -> Optional[float]:
        """
        Safely evaluate a simple arithmetic formula.

        Supports: +, -, *, /, variable names
        """
        # Replace variable names with values
        expr = formula
        for name, value in components.items():
            expr = expr.replace(name, str(value))

        # Only allow safe characters
        allowed = set('0123456789.+-*/()')
        if not all(c in allowed or c.isspace() for c in expr):
            logger.warning(f"Formula contains unsafe characters: {formula}")
            return None

        try:
            result = eval(expr)  # Safe because we validated characters
            return float(result)
        except Exception:
            return None

    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        success_rate = (
            self.stats['successes'] / self.stats['extractions'] * 100
            if self.stats['extractions'] > 0 else 0
        )
        return {
            **self.stats,
            'success_rate': f"{success_rate:.1f}%"
        }
