"""
Credential Rotation Management.

Provides automated credential rotation support:
1. RotationScheduler - Tracks credential rotation schedules
2. RotationPolicy - Defines rotation rules per credential type
3. RotationExecutor - Executes credential rotation

Usage:
    from investigator.infrastructure.credential_rotation import (
        RotationScheduler,
        RotationPolicy,
        schedule_rotation,
    )

    # Define rotation policy
    policy = RotationPolicy(
        credential_name="api_key:anthropic",
        rotation_interval_days=90,
        notify_before_days=14,
    )

    # Schedule rotation
    scheduler = RotationScheduler()
    scheduler.add_policy(policy)

    # Check for pending rotations
    pending = scheduler.get_pending_rotations()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RotationStatus(Enum):
    """Status of a credential rotation."""
    SCHEDULED = "scheduled"
    PENDING = "pending"  # Due soon
    OVERDUE = "overdue"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RotationPolicy:
    """Policy defining how and when to rotate a credential.

    Attributes:
        credential_name: Full credential identifier (type:name)
        rotation_interval_days: Days between rotations
        notify_before_days: Days before rotation to send notification
        auto_rotate: Whether to rotate automatically
        rotation_callback: Optional callback for custom rotation logic
        require_approval: Whether rotation requires human approval
    """
    credential_name: str
    rotation_interval_days: int = 90
    notify_before_days: int = 14
    auto_rotate: bool = False
    require_approval: bool = True
    rotation_callback: Optional[Callable[[], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "credential_name": self.credential_name,
            "rotation_interval_days": self.rotation_interval_days,
            "notify_before_days": self.notify_before_days,
            "auto_rotate": self.auto_rotate,
            "require_approval": self.require_approval,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RotationPolicy":
        """Create from dictionary."""
        return cls(
            credential_name=data["credential_name"],
            rotation_interval_days=data.get("rotation_interval_days", 90),
            notify_before_days=data.get("notify_before_days", 14),
            auto_rotate=data.get("auto_rotate", False),
            require_approval=data.get("require_approval", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RotationRecord:
    """Record of a credential rotation event."""
    credential_name: str
    rotation_date: datetime
    status: RotationStatus
    rotated_by: str = "system"
    notes: str = ""
    previous_version_hash: str = ""  # Hash of old credential (not the credential itself)
    new_version_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "credential_name": self.credential_name,
            "rotation_date": self.rotation_date.isoformat(),
            "status": self.status.value,
            "rotated_by": self.rotated_by,
            "notes": self.notes,
        }


@dataclass
class RotationScheduleEntry:
    """Scheduled rotation for a credential."""
    credential_name: str
    policy: RotationPolicy
    last_rotation: Optional[datetime] = None
    next_rotation: Optional[datetime] = None
    status: RotationStatus = RotationStatus.SCHEDULED

    @property
    def days_until_rotation(self) -> Optional[int]:
        """Days until next rotation."""
        if not self.next_rotation:
            return None
        delta = self.next_rotation - datetime.now()
        return delta.days

    @property
    def is_pending(self) -> bool:
        """Check if rotation is pending (due within notify period)."""
        if not self.next_rotation:
            return False
        days = self.days_until_rotation
        return days is not None and days <= self.policy.notify_before_days

    @property
    def is_overdue(self) -> bool:
        """Check if rotation is overdue."""
        if not self.next_rotation:
            return False
        return datetime.now() > self.next_rotation


class RotationScheduler:
    """Manages credential rotation schedules.

    Features:
    - Track rotation schedules for all credentials
    - Send notifications for upcoming rotations
    - Execute automatic rotations if configured
    - Maintain rotation history
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize rotation scheduler.

        Args:
            storage_path: Path to persist rotation state
        """
        self._schedules: Dict[str, RotationScheduleEntry] = {}
        self._history: List[RotationRecord] = []
        self._storage_path = storage_path
        self._notification_callbacks: List[Callable[[RotationScheduleEntry], None]] = []

        if storage_path and storage_path.exists():
            self._load_state()

    def add_policy(self, policy: RotationPolicy) -> None:
        """Add a rotation policy for a credential.

        Args:
            policy: Rotation policy to add
        """
        # Calculate next rotation date
        now = datetime.now()
        next_rotation = now + timedelta(days=policy.rotation_interval_days)

        entry = RotationScheduleEntry(
            credential_name=policy.credential_name,
            policy=policy,
            last_rotation=None,
            next_rotation=next_rotation,
            status=RotationStatus.SCHEDULED,
        )

        self._schedules[policy.credential_name] = entry
        logger.info(f"Added rotation policy for {policy.credential_name}: "
                   f"rotate every {policy.rotation_interval_days} days")

    def remove_policy(self, credential_name: str) -> None:
        """Remove rotation policy for a credential."""
        if credential_name in self._schedules:
            del self._schedules[credential_name]
            logger.info(f"Removed rotation policy for {credential_name}")

    def record_rotation(
        self,
        credential_name: str,
        rotated_by: str = "system",
        notes: str = "",
    ) -> None:
        """Record that a credential was rotated.

        Args:
            credential_name: Credential that was rotated
            rotated_by: Who performed the rotation
            notes: Optional notes about the rotation
        """
        now = datetime.now()

        # Create rotation record
        record = RotationRecord(
            credential_name=credential_name,
            rotation_date=now,
            status=RotationStatus.COMPLETED,
            rotated_by=rotated_by,
            notes=notes,
        )
        self._history.append(record)

        # Update schedule
        if credential_name in self._schedules:
            entry = self._schedules[credential_name]
            entry.last_rotation = now
            entry.next_rotation = now + timedelta(
                days=entry.policy.rotation_interval_days
            )
            entry.status = RotationStatus.SCHEDULED

        logger.info(f"Recorded rotation for {credential_name} by {rotated_by}")
        self._save_state()

    def get_pending_rotations(self) -> List[RotationScheduleEntry]:
        """Get credentials that need rotation soon.

        Returns:
            List of schedule entries for credentials needing rotation
        """
        pending = []
        for entry in self._schedules.values():
            if entry.is_overdue:
                entry.status = RotationStatus.OVERDUE
                pending.append(entry)
            elif entry.is_pending:
                entry.status = RotationStatus.PENDING
                pending.append(entry)

        return sorted(pending, key=lambda e: e.days_until_rotation or 0)

    def get_overdue_rotations(self) -> List[RotationScheduleEntry]:
        """Get credentials with overdue rotations.

        Returns:
            List of overdue rotation entries
        """
        return [e for e in self._schedules.values() if e.is_overdue]

    def get_schedule(self, credential_name: str) -> Optional[RotationScheduleEntry]:
        """Get rotation schedule for a credential.

        Args:
            credential_name: Credential to look up

        Returns:
            RotationScheduleEntry or None
        """
        return self._schedules.get(credential_name)

    def get_all_schedules(self) -> List[RotationScheduleEntry]:
        """Get all rotation schedules.

        Returns:
            List of all schedule entries
        """
        return list(self._schedules.values())

    def get_rotation_history(
        self,
        credential_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[RotationRecord]:
        """Get rotation history.

        Args:
            credential_name: Optional filter by credential
            limit: Maximum records to return

        Returns:
            List of rotation records
        """
        history = self._history
        if credential_name:
            history = [r for r in history if r.credential_name == credential_name]
        return history[-limit:]

    def add_notification_callback(
        self,
        callback: Callable[[RotationScheduleEntry], None],
    ) -> None:
        """Add callback for rotation notifications.

        Args:
            callback: Function to call when rotation is pending
        """
        self._notification_callbacks.append(callback)

    def check_and_notify(self) -> List[RotationScheduleEntry]:
        """Check for pending rotations and send notifications.

        Returns:
            List of entries for which notifications were sent
        """
        notified = []
        pending = self.get_pending_rotations()

        for entry in pending:
            for callback in self._notification_callbacks:
                try:
                    callback(entry)
                    notified.append(entry)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")

        return notified

    def _save_state(self) -> None:
        """Persist rotation state to storage."""
        if not self._storage_path:
            return

        state = {
            "schedules": {
                name: {
                    "policy": entry.policy.to_dict(),
                    "last_rotation": entry.last_rotation.isoformat() if entry.last_rotation else None,
                    "next_rotation": entry.next_rotation.isoformat() if entry.next_rotation else None,
                    "status": entry.status.value,
                }
                for name, entry in self._schedules.items()
            },
            "history": [r.to_dict() for r in self._history[-1000:]],
        }

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load rotation state from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path) as f:
                state = json.load(f)

            # Restore schedules
            for name, data in state.get("schedules", {}).items():
                policy = RotationPolicy.from_dict(data["policy"])
                entry = RotationScheduleEntry(
                    credential_name=name,
                    policy=policy,
                    last_rotation=datetime.fromisoformat(data["last_rotation"]) if data.get("last_rotation") else None,
                    next_rotation=datetime.fromisoformat(data["next_rotation"]) if data.get("next_rotation") else None,
                    status=RotationStatus(data.get("status", "scheduled")),
                )
                self._schedules[name] = entry

            # Restore history
            for record_data in state.get("history", []):
                record = RotationRecord(
                    credential_name=record_data["credential_name"],
                    rotation_date=datetime.fromisoformat(record_data["rotation_date"]),
                    status=RotationStatus(record_data["status"]),
                    rotated_by=record_data.get("rotated_by", "system"),
                    notes=record_data.get("notes", ""),
                )
                self._history.append(record)

            logger.info(f"Loaded {len(self._schedules)} rotation schedules")

        except Exception as e:
            logger.error(f"Failed to load rotation state: {e}")


# Default rotation policies for common credentials
DEFAULT_ROTATION_POLICIES = {
    "database:sec": RotationPolicy(
        credential_name="database:sec",
        rotation_interval_days=90,
        notify_before_days=14,
        require_approval=True,
    ),
    "database:stock": RotationPolicy(
        credential_name="database:stock",
        rotation_interval_days=90,
        notify_before_days=14,
        require_approval=True,
    ),
    "api_key:anthropic": RotationPolicy(
        credential_name="api_key:anthropic",
        rotation_interval_days=180,
        notify_before_days=30,
        require_approval=True,
    ),
    "api_key:openai": RotationPolicy(
        credential_name="api_key:openai",
        rotation_interval_days=180,
        notify_before_days=30,
        require_approval=True,
    ),
}


def get_default_scheduler(storage_path: Optional[Path] = None) -> RotationScheduler:
    """Get a scheduler pre-configured with default policies.

    Args:
        storage_path: Optional path for persistence

    Returns:
        Configured RotationScheduler
    """
    scheduler = RotationScheduler(storage_path)

    for policy in DEFAULT_ROTATION_POLICIES.values():
        scheduler.add_policy(policy)

    return scheduler


__all__ = [
    "RotationStatus",
    "RotationPolicy",
    "RotationRecord",
    "RotationScheduleEntry",
    "RotationScheduler",
    "DEFAULT_ROTATION_POLICIES",
    "get_default_scheduler",
]
