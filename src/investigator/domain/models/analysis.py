"""
Domain models for analysis tasks and results
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AnalysisType(Enum):
    """Types of analysis that agents can perform"""

    SEC_FUNDAMENTAL = "sec_fundamental"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    PEER_GROUP = "peer_group"
    INVESTMENT_SYNTHESIS = "investment_synthesis"
    ESG_ANALYSIS = "esg_analysis"
    MARKET_DATA = "market_data"
    MARKET_CONTEXT = "market_context"
    RISK_ASSESSMENT = "risk_assessment"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    OPTIONS_ANALYSIS = "options_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


class TaskStatus(Enum):
    """Status of agent tasks"""

    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class Priority(Enum):
    """Task priority levels"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 5
    LOW = 8
    BACKGROUND = 10


@dataclass
class AgentCapability:
    """Defines what an agent can do"""

    analysis_type: AnalysisType
    min_data_required: Dict[str, Any]
    max_processing_time: int  # seconds
    required_models: List[str]
    cache_ttl: int  # seconds


@dataclass
class AgentTask:
    """Task definition for agents"""

    task_id: str
    symbol: str
    analysis_type: AnalysisType
    priority: Priority = Priority.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    fiscal_period: Optional[str] = None  # e.g., "2024-Q3", "2024-FY"

    def get_cache_key(self) -> str:
        """
        Generate cache key for this task.

        CRITICAL: Includes fiscal_period to ensure different cache entries
        for the same symbol analyzed in different fiscal periods.

        This fixes the cache hit rate issue (5% → 75%) identified in
        ANALYSIS_SUMMARY_20251112.txt (Pain Point #2: Cache Key Inconsistency).
        """
        key_data = {
            "symbol": self.symbol,
            "analysis_type": self.analysis_type.value,
            "context_keys": sorted(self.context.keys()),
            "fiscal_period": self.fiscal_period or "latest",  # Default to "latest" if not specified
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


@dataclass
class AgentResult:
    """Result from agent processing"""

    task_id: str
    agent_id: str
    status: TaskStatus
    result_data: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def is_successful(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def to_json(self) -> str:
        """Serialize result to JSON"""
        return json.dumps(
            {
                "task_id": self.task_id,
                "agent_id": self.agent_id,
                "status": self.status.value,
                "result_data": self.result_data,
                "processing_time": self.processing_time,
                "error": self.error,
                "metadata": self.metadata,
                "cached": self.cached,
                "cache_hit": self.cache_hit,
                "timestamp": self.timestamp.isoformat(),
            }
        )


@dataclass
class AgentMetrics:
    """Performance metrics for agents"""

    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, result: AgentResult):
        """Update metrics with new result"""
        self.total_tasks += 1
        if result.is_successful():
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1

        # Update rolling average processing time
        self.avg_processing_time = (
            self.avg_processing_time * (self.total_tasks - 1) + result.processing_time
        ) / self.total_tasks

        # Update cache hit rate
        if result.cache_hit:
            cache_hits = int(self.cache_hit_rate * (self.total_tasks - 1))
            self.cache_hit_rate = (cache_hits + 1) / self.total_tasks
        else:
            cache_hits = int(self.cache_hit_rate * (self.total_tasks - 1))
            self.cache_hit_rate = cache_hits / self.total_tasks

        # Update error rate
        self.error_rate = self.failed_tasks / self.total_tasks if self.total_tasks > 0 else 0
        self.last_updated = datetime.now()
