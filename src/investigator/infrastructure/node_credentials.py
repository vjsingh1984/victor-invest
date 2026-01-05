"""
Node Credential Context for secure credential injection into workflow nodes.

This module provides:
1. NodeCredentialContext - Scoped credential container for node execution
2. CredentialValidator - Pre-flight validation before node execution
3. CredentialAuditLogger - Audit logging for credential access

Integrates with Victor framework's CredentialManager when available,
falls back to environment variables.

Usage in workflow nodes:
    async def __call__(self, node, context, tool_registry):
        # Get credentials scoped to this node
        cred_ctx = NodeCredentialContext.from_node(node, context)

        # Access specific credentials
        db_creds = cred_ctx.get_database("sec")
        api_key = cred_ctx.get_api_key("anthropic")

        # Credentials are validated before node execution
        # and audit logged after access
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from investigator.infrastructure.credentials import (
    DatabaseCredentials,
    get_database_credentials,
)

logger = logging.getLogger(__name__)


class CredentialType(Enum):
    """Types of credentials that can be required by nodes."""

    DATABASE = "database"
    API_KEY = "api_key"
    AWS = "aws"
    OAUTH = "oauth"
    CERTIFICATE = "certificate"


@dataclass
class CredentialRequirement:
    """Specification for a required credential."""

    type: CredentialType
    name: str  # e.g., "sec", "stock", "anthropic"
    required: bool = True  # If False, node can proceed without it
    scopes: List[str] = field(default_factory=list)  # Optional permission scopes


@dataclass
class CredentialAuditEntry:
    """Audit log entry for credential access."""

    timestamp: datetime
    node_id: str
    credential_type: str
    credential_name: str
    access_granted: bool
    source: str  # "victor_framework", "environment", "keyring"
    duration_ms: float = 0.0
    error: Optional[str] = None


class CredentialAuditLogger:
    """Audit logger for credential access."""

    _entries: List[CredentialAuditEntry] = []
    _max_entries: int = 10000

    @classmethod
    def log_access(
        cls,
        node_id: str,
        credential_type: str,
        credential_name: str,
        access_granted: bool,
        source: str,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Log a credential access attempt."""
        entry = CredentialAuditEntry(
            timestamp=datetime.now(),
            node_id=node_id,
            credential_type=credential_type,
            credential_name=credential_name,
            access_granted=access_granted,
            source=source,
            duration_ms=duration_ms,
            error=error,
        )

        cls._entries.append(entry)

        # Rotate if too many entries
        if len(cls._entries) > cls._max_entries:
            cls._entries = cls._entries[-cls._max_entries :]

        # Log to standard logger
        if access_granted:
            logger.info(
                f"[CRED_AUDIT] node={node_id} type={credential_type} "
                f"name={credential_name} granted=True source={source} "
                f"duration_ms={duration_ms:.2f}"
            )
        else:
            logger.warning(
                f"[CRED_AUDIT] node={node_id} type={credential_type} "
                f"name={credential_name} granted=False error={error}"
            )

    @classmethod
    def get_entries(
        cls,
        node_id: Optional[str] = None,
        credential_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[CredentialAuditEntry]:
        """Query audit entries with optional filters."""
        entries = cls._entries

        if node_id:
            entries = [e for e in entries if e.node_id == node_id]
        if credential_type:
            entries = [e for e in entries if e.credential_type == credential_type]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries

    @classmethod
    def clear(cls) -> None:
        """Clear audit log (for testing)."""
        cls._entries = []

    @classmethod
    def detect_misuse_patterns(cls) -> List[Dict[str, Any]]:
        """Detect potential credential misuse patterns.

        Patterns detected:
        - Excessive access frequency (>100 accesses in 1 minute)
        - High failure rate (>50% failures for a credential)
        - Unexpected node access (credential accessed by unauthorized node)
        - Off-hours access (outside 6am-10pm local time)

        Returns:
            List of detected pattern violations
        """
        violations = []
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)

        # Get recent entries
        recent_entries = [e for e in cls._entries if e.timestamp >= one_minute_ago]
        hourly_entries = [e for e in cls._entries if e.timestamp >= one_hour_ago]

        # Pattern 1: Excessive access frequency
        if len(recent_entries) > 100:
            violations.append(
                {
                    "pattern": "excessive_frequency",
                    "severity": "high",
                    "message": f"Excessive credential access: {len(recent_entries)} accesses in last minute",
                    "count": len(recent_entries),
                }
            )

        # Pattern 2: High failure rate per credential
        cred_stats: Dict[str, Dict[str, int]] = {}
        for entry in hourly_entries:
            key = f"{entry.credential_type}:{entry.credential_name}"
            if key not in cred_stats:
                cred_stats[key] = {"success": 0, "failure": 0}
            if entry.access_granted:
                cred_stats[key]["success"] += 1
            else:
                cred_stats[key]["failure"] += 1

        for cred_key, stats in cred_stats.items():
            total = stats["success"] + stats["failure"]
            if total >= 5 and stats["failure"] / total > 0.5:
                violations.append(
                    {
                        "pattern": "high_failure_rate",
                        "severity": "medium",
                        "message": f"High failure rate for {cred_key}: {stats['failure']}/{total} failed",
                        "credential": cred_key,
                        "failure_rate": stats["failure"] / total,
                    }
                )

        # Pattern 3: Off-hours access (6am-10pm considered normal)
        current_hour = now.hour
        if current_hour < 6 or current_hour >= 22:
            off_hours_entries = [e for e in recent_entries if e.timestamp.hour < 6 or e.timestamp.hour >= 22]
            if off_hours_entries:
                violations.append(
                    {
                        "pattern": "off_hours_access",
                        "severity": "low",
                        "message": f"Credential access outside normal hours: {len(off_hours_entries)} accesses",
                        "count": len(off_hours_entries),
                    }
                )

        # Pattern 4: Unusual node accessing sensitive credentials
        # Track which nodes typically access which credentials
        node_cred_access: Dict[str, Set[str]] = {}
        for entry in cls._entries:
            cred_key = f"{entry.credential_type}:{entry.credential_name}"
            if entry.node_id not in node_cred_access:
                node_cred_access[entry.node_id] = set()
            node_cred_access[entry.node_id].add(cred_key)

        # Check for nodes accessing credentials they haven't accessed before
        for entry in recent_entries:
            cred_key = f"{entry.credential_type}:{entry.credential_name}"
            # Get all nodes that have accessed this credential
            nodes_for_cred = [n for n, creds in node_cred_access.items() if cred_key in creds]
            # If this is a new node accessing an established credential
            if len(nodes_for_cred) > 3:  # Established credential
                # Check if this node has only recently started accessing it
                node_history = [
                    e
                    for e in cls._entries
                    if e.node_id == entry.node_id and f"{e.credential_type}:{e.credential_name}" == cred_key
                ]
                if len(node_history) <= 2:  # New accessor
                    violations.append(
                        {
                            "pattern": "new_accessor",
                            "severity": "low",
                            "message": f"Node '{entry.node_id}' newly accessing {cred_key}",
                            "node_id": entry.node_id,
                            "credential": cred_key,
                        }
                    )

        return violations

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get credential access statistics.

        Returns:
            Dict with access statistics
        """
        if not cls._entries:
            return {"total_accesses": 0}

        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        hourly_entries = [e for e in cls._entries if e.timestamp >= one_hour_ago]

        # Count by credential
        by_credential: Dict[str, int] = {}
        by_node: Dict[str, int] = {}
        success_count = 0
        failure_count = 0

        for entry in cls._entries:
            key = f"{entry.credential_type}:{entry.credential_name}"
            by_credential[key] = by_credential.get(key, 0) + 1
            by_node[entry.node_id] = by_node.get(entry.node_id, 0) + 1
            if entry.access_granted:
                success_count += 1
            else:
                failure_count += 1

        return {
            "total_accesses": len(cls._entries),
            "hourly_accesses": len(hourly_entries),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(cls._entries) if cls._entries else 0,
            "by_credential": by_credential,
            "by_node": by_node,
            "unique_credentials": len(by_credential),
            "unique_nodes": len(by_node),
        }


class NodeCredentialContext:
    """Scoped credential container for node execution.

    Provides secure access to credentials required by a workflow node,
    with audit logging and validation.
    """

    def __init__(
        self,
        node_id: str,
        requirements: List[CredentialRequirement] = None,
        workflow_context: Any = None,
    ):
        self.node_id = node_id
        self.requirements = requirements or []
        self.workflow_context = workflow_context
        self._cache: Dict[str, Any] = {}
        self._victor_cred_mgr = None

        # Try to get Victor framework credential manager
        try:
            from victor.workflows.services.credentials import get_credential_manager

            self._victor_cred_mgr = get_credential_manager()
        except ImportError:
            pass

    @classmethod
    def from_node(cls, node: Any, context: Any) -> "NodeCredentialContext":
        """Create credential context from workflow node definition.

        Extracts credential requirements from node.credentials_required field.
        """
        requirements = []

        # Check if node has credentials_required attribute
        creds_required = getattr(node, "credentials_required", None)
        if creds_required:
            for cred_spec in creds_required:
                if isinstance(cred_spec, str):
                    # Simple format: "database:sec" or "api_key:anthropic"
                    parts = cred_spec.split(":")
                    if len(parts) == 2:
                        cred_type = CredentialType(parts[0])
                        requirements.append(
                            CredentialRequirement(
                                type=cred_type,
                                name=parts[1],
                            )
                        )
                elif isinstance(cred_spec, dict):
                    # Full format: {"type": "database", "name": "sec", "required": true}
                    requirements.append(
                        CredentialRequirement(
                            type=CredentialType(cred_spec["type"]),
                            name=cred_spec["name"],
                            required=cred_spec.get("required", True),
                            scopes=cred_spec.get("scopes", []),
                        )
                    )

        return cls(
            node_id=getattr(node, "id", "unknown"),
            requirements=requirements,
            workflow_context=context,
        )

    def get_database(self, alias: str) -> Optional[DatabaseCredentials]:
        """Get database credentials with audit logging.

        Args:
            alias: Database alias ("sec", "stock", etc.)

        Returns:
            DatabaseCredentials or None if not available
        """
        cache_key = f"database:{alias}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.time()
        source = "unknown"
        error = None
        creds = None

        try:
            # Try Victor framework first
            if self._victor_cred_mgr:
                victor_creds = self._victor_cred_mgr.get_database(alias)
                if victor_creds:
                    source = "victor_framework"
                    creds = DatabaseCredentials(
                        alias=victor_creds.alias,
                        host=victor_creds.host,
                        port=victor_creds.port,
                        database=victor_creds.database,
                        username=victor_creds.username,
                        password=victor_creds.password,
                        driver=victor_creds.driver,
                    )

            # Fall back to environment
            if not creds:
                creds = get_database_credentials(alias)
                source = "environment"

            self._cache[cache_key] = creds

        except EnvironmentError as e:
            error = str(e)
            creds = None
        except Exception as e:
            error = f"Unexpected error: {e}"
            creds = None

        duration_ms = (time.time() - start_time) * 1000

        CredentialAuditLogger.log_access(
            node_id=self.node_id,
            credential_type="database",
            credential_name=alias,
            access_granted=creds is not None,
            source=source,
            duration_ms=duration_ms,
            error=error,
        )

        return creds

    def get_api_key(self, name: str) -> Optional[str]:
        """Get API key with audit logging.

        Args:
            name: API key name ("anthropic", "openai", etc.)

        Returns:
            API key string or None if not available
        """
        import os

        cache_key = f"api_key:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.time()
        source = "unknown"
        error = None
        api_key = None

        try:
            # Try Victor framework first
            if self._victor_cred_mgr:
                victor_key = self._victor_cred_mgr.get_api_key(name)
                if victor_key:
                    source = "victor_framework"
                    api_key = victor_key.key

            # Fall back to environment variable
            if not api_key:
                env_var = f"{name.upper()}_API_KEY"
                api_key = os.environ.get(env_var)
                if api_key:
                    source = "environment"
                else:
                    error = f"API key not found: {name}"

            if api_key:
                self._cache[cache_key] = api_key

        except Exception as e:
            error = f"Unexpected error: {e}"
            api_key = None

        duration_ms = (time.time() - start_time) * 1000

        CredentialAuditLogger.log_access(
            node_id=self.node_id,
            credential_type="api_key",
            credential_name=name,
            access_granted=api_key is not None,
            source=source,
            duration_ms=duration_ms,
            error=error,
        )

        return api_key

    def validate_requirements(self) -> List[str]:
        """Validate that all required credentials are available.

        Returns:
            List of error messages for missing required credentials.
            Empty list if all requirements are satisfied.
        """
        errors = []

        for req in self.requirements:
            if req.type == CredentialType.DATABASE:
                creds = self.get_database(req.name)
                if req.required and not creds:
                    errors.append(f"Required database credential '{req.name}' not available")

            elif req.type == CredentialType.API_KEY:
                key = self.get_api_key(req.name)
                if req.required and not key:
                    errors.append(f"Required API key '{req.name}' not available")

        return errors


class CredentialValidator:
    """Pre-flight credential validation before node execution."""

    @staticmethod
    def validate_node(node: Any, context: Any) -> List[str]:
        """Validate credentials before node execution.

        Args:
            node: Workflow node to validate
            context: Workflow context

        Returns:
            List of validation errors (empty if valid)
        """
        cred_ctx = NodeCredentialContext.from_node(node, context)
        return cred_ctx.validate_requirements()

    @staticmethod
    def validate_workflow(workflow_def: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate credentials for entire workflow definition.

        Args:
            workflow_def: Workflow definition dict with nodes

        Returns:
            Dict mapping node_id to list of credential errors
        """
        errors = {}
        nodes = workflow_def.get("nodes", [])

        for node in nodes:
            node_id = node.get("id", "unknown")
            creds_required = node.get("credentials_required", [])

            if not creds_required:
                continue

            # Create mock node for validation
            @dataclass
            class MockNode:
                id: str
                credentials_required: List

            mock = MockNode(id=node_id, credentials_required=creds_required)
            cred_ctx = NodeCredentialContext.from_node(mock, None)
            node_errors = cred_ctx.validate_requirements()

            if node_errors:
                errors[node_id] = node_errors

        return errors


def inject_credentials_middleware(handler_func):
    """Decorator to inject credential context into node handlers.

    Usage:
        @inject_credentials_middleware
        async def my_handler(node, context, tool_registry, cred_ctx=None):
            db = cred_ctx.get_database("sec")
            ...
    """
    import functools
    import inspect

    @functools.wraps(handler_func)
    async def wrapper(node, context, tool_registry, **kwargs):
        # Create credential context
        cred_ctx = NodeCredentialContext.from_node(node, context)

        # Validate requirements
        errors = cred_ctx.validate_requirements()
        if errors:
            logger.warning(f"Node {node.id} credential validation warnings: {errors}")

        # Check if handler accepts cred_ctx
        sig = inspect.signature(handler_func)
        if "cred_ctx" in sig.parameters:
            kwargs["cred_ctx"] = cred_ctx

        return await handler_func(node, context, tool_registry, **kwargs)

    return wrapper


__all__ = [
    "CredentialType",
    "CredentialRequirement",
    "CredentialAuditEntry",
    "CredentialAuditLogger",
    "NodeCredentialContext",
    "CredentialValidator",
    "inject_credentials_middleware",
]
