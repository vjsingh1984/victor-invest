"""
Event Bus Infrastructure

Asynchronous event-driven messaging system for agent coordination.

Author: InvestiGator Team
Date: 2025-11-14
"""

from investigator.infrastructure.events.event_bus import (
    Event,
    EventBus,
    EventChannel,
    EventPriority,
)

__all__ = [
    "Event",
    "EventBus",
    "EventChannel",
    "EventPriority",
]
