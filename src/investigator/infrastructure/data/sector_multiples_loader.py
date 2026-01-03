"""Reference data loader for sector valuation multiples."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MultiplesRecord:
    sector: str
    pe: Optional[float]
    ev_ebitda: Optional[float]
    ps: Optional[float]
    pb: Optional[float]
    sample_size: Optional[int]
    last_updated: Optional[datetime]


class SectorMultiplesLoader:
    """Loads and validates sector median multiples from a JSON or CSV reference."""

    def __init__(
        self,
        *,
        reference_path: Path,
        freshness_days: int = 7,
        delta_threshold: float = 0.15,
    ) -> None:
        self.reference_path = reference_path
        self.freshness_days = freshness_days
        self.delta_threshold = delta_threshold
        self._cache: Dict[str, MultiplesRecord] = {}

    def load(self) -> Dict[str, MultiplesRecord]:
        if not self.reference_path.exists():
            logger.warning("Sector multiples reference not found at %s", self.reference_path)
            return {}

        try:
            data = json.loads(self.reference_path.read_text())
        except Exception as exc:
            logger.error("Failed to read sector multiples from %s: %s", self.reference_path, exc)
            return {}

        metadata = data.get("_metadata", {})
        records: Dict[str, MultiplesRecord] = {}

        for sector, values in data.items():
            if sector == "_metadata":
                continue
            record = MultiplesRecord(
                sector=sector,
                pe=self._to_float(values.get("pe")),
                ev_ebitda=self._to_float(values.get("ev_ebitda")),
                ps=self._to_float(values.get("ps")),
                pb=self._to_float(values.get("pb")),
                sample_size=self._to_int(values.get("sample_size")),
                last_updated=self._parse_datetime(values.get("last_updated")),
            )
            self._validate_record(record, metadata)
            records[sector.lower()] = record

        self._cache = records
        return records

    def get(self, sector: str) -> Optional[MultiplesRecord]:
        if not self._cache:
            self.load()
        return self._cache.get(sector.lower())

    def _validate_record(self, record: MultiplesRecord, metadata: Dict[str, any]) -> None:
        if record.last_updated:
            # Ensure both datetimes are timezone-aware for comparison
            now = datetime.now(timezone.utc)
            last_updated = record.last_updated if record.last_updated.tzinfo else record.last_updated.replace(tzinfo=timezone.utc)
            age = now - last_updated
            if age > timedelta(days=self.freshness_days):
                logger.warning(
                    "Sector multiples for %s are stale (age %s days)",
                    record.sector,
                    age.days,
                )

        if record.sample_size is not None and record.sample_size < 5:
            logger.warning("Sector multiples for %s have low sample size (%s)", record.sector, record.sample_size)

        previous_container = metadata.get("previous") if isinstance(metadata, dict) else None
        previous = {} if not isinstance(previous_container, dict) else previous_container.get(record.sector, {})
        for key in ("pe", "ev_ebitda", "ps", "pb"):
            old_value = self._to_float(previous.get(key))
            new_value = getattr(record, key)
            if old_value and new_value:
                delta = abs(new_value - old_value) / old_value
                if delta > self.delta_threshold:
                    logger.warning(
                        "Sector multiple %s for %s shifted by %.1f%% (old %.2f → new %.2f)",
                        key,
                        record.sector,
                        delta * 100,
                        old_value,
                        new_value,
                    )

    @staticmethod
    def _to_float(value: Optional[float]) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Optional[int]) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
