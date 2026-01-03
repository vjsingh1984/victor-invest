"""
InvestiGator Domain Layer

Contains core business logic, entities, and domain services.
"""

# Export base agent
from investigator.domain.agents.base import InvestmentAgent, get_cache_type_for_analysis

# Export models
from investigator.domain.models.analysis import (
    AgentCapability,
    AgentMetrics,
    AgentResult,
    AgentTask,
    AnalysisType,
    Priority,
    TaskStatus,
)

__all__ = [
    # Models
    "AnalysisType",
    "TaskStatus",
    "Priority",
    "AgentCapability",
    "AgentTask",
    "AgentResult",
    "AgentMetrics",
    # Agents
    "InvestmentAgent",
    "get_cache_type_for_analysis",
]
