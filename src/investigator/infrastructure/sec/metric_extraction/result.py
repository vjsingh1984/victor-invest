"""
Extraction Result Types

Provides structured result objects for metric extraction operations,
including value, metadata, and audit trail information.

SOLID Principle: Single Responsibility
- ExtractionResult: Holds extraction outcome
- ExtractionAudit: Holds extraction attempt history
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MatchMethod(Enum):
    """How a period match was achieved."""

    BY_PERIOD_END = "by_period_end"  # Exact end date match (most reliable)
    BY_DATE_RANGE = "by_date_range"  # Start/end date range match
    BY_FRAME_FIELD = "by_frame_field"  # CY2024Q3 frame field match
    BY_ADSH_FY_FP = "by_adsh_fy_fp"  # Legacy ADSH + fy + fp match
    BY_ADSH_ONLY = "by_adsh_only"  # ADSH match without fy/fp filter
    DERIVED = "derived"  # Calculated from other metrics
    NOT_FOUND = "not_found"  # No match found


class ExtractionConfidence(Enum):
    """Confidence level in extracted value."""

    HIGH = "high"  # Exact match on all criteria
    MEDIUM = "medium"  # Partial match or fallback tag used
    LOW = "low"  # Multiple fallbacks or fuzzy match
    DERIVED = "derived"  # Value was calculated, not extracted
    NONE = "none"  # No value found


@dataclass
class ExtractionAttempt:
    """Record of a single extraction attempt."""

    strategy_name: str
    tag_name: str
    matched: bool
    entries_found: int
    selected_entry: Optional[Dict] = None
    reason: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class ExtractionAudit:
    """
    Complete audit trail of extraction attempts.

    Provides full traceability of which strategies and tags were tried,
    enabling debugging and quality assessment.
    """

    canonical_key: str
    target_period_end: Optional[str] = None
    target_fiscal_year: Optional[int] = None
    target_fiscal_period: Optional[str] = None
    target_adsh: Optional[str] = None

    attempts: List[ExtractionAttempt] = field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def add_attempt(self, attempt: ExtractionAttempt) -> None:
        """Add an extraction attempt to the audit trail."""
        self.attempts.append(attempt)
        self.total_duration_ms += attempt.duration_ms

    def summary(self) -> str:
        """Generate human-readable summary."""
        successful = [a for a in self.attempts if a.matched]
        failed = [a for a in self.attempts if not a.matched]

        lines = [
            f"Extraction Audit: {self.canonical_key}",
            f"  Target: period_end={self.target_period_end}, "
            f"fiscal={self.target_fiscal_year}-{self.target_fiscal_period}",
            f"  Attempts: {len(self.attempts)} ({len(successful)} succeeded, {len(failed)} failed)",
        ]

        if successful:
            first_success = successful[0]
            lines.append(f"  Matched via: {first_success.strategy_name} using tag '{first_success.tag_name}'")

        if failed and not successful:
            lines.append(f"  Failed strategies: {[a.strategy_name for a in failed]}")

        return "\n".join(lines)


@dataclass
class ExtractionResult:
    """
    Result of a metric extraction operation.

    Contains the extracted value along with comprehensive metadata
    about how it was obtained and confidence level.

    Attributes:
        success: Whether extraction succeeded
        value: The extracted value (None if not found)
        source_tag: XBRL tag used for extraction
        match_method: How the period was matched
        confidence: Confidence level in the result
        period_end: Period end date of extracted entry
        entry: Full SEC entry dict (for debugging)
        audit: Complete extraction audit trail
        error: Error message if extraction failed
    """

    success: bool
    value: Optional[float] = None
    source_tag: Optional[str] = None
    match_method: MatchMethod = MatchMethod.NOT_FOUND
    confidence: ExtractionConfidence = ExtractionConfidence.NONE

    # Metadata from matched entry
    period_end: Optional[str] = None
    period_start: Optional[str] = None
    duration_days: Optional[int] = None
    form: Optional[str] = None
    filed_date: Optional[str] = None
    accn: Optional[str] = None

    # SEC fields (may be unreliable)
    sec_fy: Optional[int] = None
    sec_fp: Optional[str] = None

    # Full entry and audit
    entry: Optional[Dict] = None
    audit: Optional[ExtractionAudit] = None
    error: Optional[str] = None

    @classmethod
    def not_found(
        cls, canonical_key: str, audit: Optional[ExtractionAudit] = None, reason: str = "No matching entry found"
    ) -> "ExtractionResult":
        """Factory for failed extraction."""
        return cls(
            success=False,
            match_method=MatchMethod.NOT_FOUND,
            confidence=ExtractionConfidence.NONE,
            audit=audit,
            error=reason,
        )

    @classmethod
    def from_entry(
        cls,
        value: float,
        source_tag: str,
        entry: Dict,
        match_method: MatchMethod,
        confidence: ExtractionConfidence = ExtractionConfidence.HIGH,
        audit: Optional[ExtractionAudit] = None,
    ) -> "ExtractionResult":
        """Factory from SEC entry dict."""
        # Calculate duration
        duration_days = None
        start = entry.get("start")
        end = entry.get("end")
        if start and end:
            try:
                from datetime import datetime as dt

                start_date = dt.strptime(start, "%Y-%m-%d")
                end_date = dt.strptime(end, "%Y-%m-%d")
                duration_days = (end_date - start_date).days
            except ValueError:
                pass

        return cls(
            success=True,
            value=value,
            source_tag=source_tag,
            match_method=match_method,
            confidence=confidence,
            period_end=entry.get("end"),
            period_start=entry.get("start"),
            duration_days=duration_days,
            form=entry.get("form"),
            filed_date=entry.get("filed"),
            accn=entry.get("accn"),
            sec_fy=entry.get("fy"),
            sec_fp=entry.get("fp"),
            entry=entry,
            audit=audit,
        )

    @classmethod
    def derived(
        cls, value: float, formula: str, components: Dict[str, float], audit: Optional[ExtractionAudit] = None
    ) -> "ExtractionResult":
        """Factory for derived/calculated values."""
        return cls(
            success=True,
            value=value,
            source_tag=f"derived:{formula}",
            match_method=MatchMethod.DERIVED,
            confidence=ExtractionConfidence.DERIVED,
            audit=audit,
        )

    def __repr__(self) -> str:
        if self.success:
            return (
                f"ExtractionResult(value={self.value:,.0f}, "
                f"tag='{self.source_tag}', "
                f"method={self.match_method.value}, "
                f"confidence={self.confidence.value})"
            )
        return f"ExtractionResult(success=False, error='{self.error}')"
