"""
Integration tests for the credential management system.

Tests:
1. Credential resolution (database, API keys)
2. Node credential context and validation
3. Audit logging and misuse detection
4. MCP credential handling
5. Credential rotation scheduling
6. Credential sanitization and leakage detection

Run with: pytest tests/test_credentials.py -v
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDatabaseCredentials:
    """Tests for database credential resolution."""

    def test_get_database_credentials_from_env(self):
        """Test credential resolution from environment variables."""
        from investigator.infrastructure.credentials import get_database_credentials

        # Set up environment
        with patch.dict(
            os.environ,
            {
                "SEC_DB_HOST": "test-host",
                "SEC_DB_PORT": "5432",
                "SEC_DB_NAME": "test_db",
                "SEC_DB_USER": "test_user",
                "SEC_DB_PASSWORD": "test_pass",
            },
        ):
            creds = get_database_credentials("sec")

            assert creds.host == "test-host"
            assert creds.port == 5432
            assert creds.database == "test_db"
            assert creds.username == "test_user"
            assert creds.password == "test_pass"

    def test_connection_string_generation(self):
        """Test connection string generation."""
        from investigator.infrastructure.credentials import DatabaseCredentials

        creds = DatabaseCredentials(
            alias="test",
            host="localhost",
            port=5432,
            database="testdb",
            username="user",
            password="pass",
        )

        assert "postgresql://user:pass@localhost:5432/testdb" == creds.connection_string

    def test_missing_credentials_raises_error(self):
        """Test that missing credentials raise appropriate error."""
        from investigator.infrastructure.credentials import get_database_credentials

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError) as exc_info:
                get_database_credentials("nonexistent")

            assert "not found" in str(exc_info.value).lower()


class TestNodeCredentialContext:
    """Tests for node credential context."""

    def test_from_node_extracts_requirements(self):
        """Test credential requirement extraction from node."""
        from dataclasses import dataclass

        from investigator.infrastructure.node_credentials import (
            CredentialType,
            NodeCredentialContext,
        )

        @dataclass
        class MockNode:
            id: str = "test_node"
            credentials_required: list = None

        node = MockNode(
            id="fundamental_analysis",
            credentials_required=[
                {"type": "database", "name": "sec", "required": True},
                {"type": "api_key", "name": "anthropic", "required": False},
            ],
        )

        ctx = NodeCredentialContext.from_node(node, None)

        assert ctx.node_id == "fundamental_analysis"
        assert len(ctx.requirements) == 2
        assert ctx.requirements[0].type == CredentialType.DATABASE
        assert ctx.requirements[0].name == "sec"
        assert ctx.requirements[0].required is True

    def test_validate_requirements(self):
        """Test credential validation."""
        from investigator.infrastructure.node_credentials import (
            CredentialRequirement,
            CredentialType,
            NodeCredentialContext,
        )

        with patch.dict(
            os.environ,
            {
                "SEC_DB_HOST": "localhost",
                "SEC_DB_PORT": "5432",
                "SEC_DB_NAME": "sec_db",
                "SEC_DB_USER": "user",
                "SEC_DB_PASSWORD": "pass",
            },
        ):
            ctx = NodeCredentialContext(
                node_id="test",
                requirements=[
                    CredentialRequirement(
                        type=CredentialType.DATABASE,
                        name="sec",
                        required=True,
                    )
                ],
            )

            errors = ctx.validate_requirements()
            assert len(errors) == 0


class TestCredentialAuditLogger:
    """Tests for credential audit logging."""

    def test_log_access_records_entry(self):
        """Test that access is properly logged."""
        from investigator.infrastructure.node_credentials import CredentialAuditLogger

        CredentialAuditLogger.clear()

        CredentialAuditLogger.log_access(
            node_id="test_node",
            credential_type="database",
            credential_name="sec",
            access_granted=True,
            source="environment",
            duration_ms=5.0,
        )

        entries = CredentialAuditLogger.get_entries()
        assert len(entries) == 1
        assert entries[0].node_id == "test_node"
        assert entries[0].access_granted is True

    def test_get_entries_with_filters(self):
        """Test filtering audit entries."""
        from investigator.infrastructure.node_credentials import CredentialAuditLogger

        CredentialAuditLogger.clear()

        # Add multiple entries
        for i in range(3):
            CredentialAuditLogger.log_access(
                node_id=f"node_{i}",
                credential_type="database",
                credential_name="sec",
                access_granted=True,
                source="environment",
            )

        # Filter by node
        entries = CredentialAuditLogger.get_entries(node_id="node_1")
        assert len(entries) == 1
        assert entries[0].node_id == "node_1"

    def test_detect_misuse_high_failure_rate(self):
        """Test detection of high failure rate pattern."""
        from investigator.infrastructure.node_credentials import CredentialAuditLogger

        CredentialAuditLogger.clear()

        # Log multiple failures
        for i in range(6):
            CredentialAuditLogger.log_access(
                node_id="suspicious_node",
                credential_type="api_key",
                credential_name="test_key",
                access_granted=False,
                source="environment",
                error="Not found",
            )

        violations = CredentialAuditLogger.detect_misuse_patterns()
        assert len(violations) > 0
        assert any(v["pattern"] == "high_failure_rate" for v in violations)

    def test_get_statistics(self):
        """Test statistics generation."""
        from investigator.infrastructure.node_credentials import CredentialAuditLogger

        CredentialAuditLogger.clear()

        # Add some entries
        CredentialAuditLogger.log_access(
            node_id="node_a",
            credential_type="database",
            credential_name="sec",
            access_granted=True,
            source="environment",
        )
        CredentialAuditLogger.log_access(
            node_id="node_b",
            credential_type="api_key",
            credential_name="test",
            access_granted=False,
            source="environment",
        )

        stats = CredentialAuditLogger.get_statistics()
        assert stats["total_accesses"] == 2
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 1
        assert stats["unique_nodes"] == 2


class TestMCPCredentials:
    """Tests for MCP credential handling."""

    def test_mcp_credential_resolver(self):
        """Test MCP credential resolution."""
        from investigator.infrastructure.mcp_credentials import (
            MCPCredentialResolver,
            MCPServerCredentials,
        )

        with patch.dict(
            os.environ,
            {
                "SEC_DB_HOST": "localhost",
                "SEC_DB_PORT": "5432",
                "SEC_DB_NAME": "sec_db",
                "SEC_DB_USER": "user",
                "SEC_DB_PASSWORD": "pass",
            },
        ):
            resolver = MCPCredentialResolver()
            db_creds = resolver.resolve_database("sec")

            assert db_creds is not None
            assert db_creds["host"] == "localhost"

    def test_inject_mcp_credentials(self):
        """Test credential injection into MCP server config."""
        from investigator.infrastructure.mcp_credentials import (
            MCPCredentialResolver,
            MCPServerCredentials,
            inject_mcp_credentials,
        )

        with patch.dict(
            os.environ,
            {
                "SEC_DB_HOST": "localhost",
                "SEC_DB_PORT": "5432",
                "SEC_DB_NAME": "sec_db",
                "SEC_DB_USER": "user",
                "SEC_DB_PASSWORD": "pass",
            },
        ):
            server_creds = MCPServerCredentials(
                server_name="test",
                required_credentials=["sec_db"],
                env_mappings={"sec_db": "SEC_DB"},
            )

            config = {"name": "test", "command": ["python", "server.py"]}
            resolver = MCPCredentialResolver()

            injected = inject_mcp_credentials(config, server_creds, resolver)

            assert "env" in injected
            assert "SEC_DB_HOST" in injected["env"]

    def test_mcp_auth_types(self):
        """Test MCP authentication type configuration."""
        from investigator.infrastructure.mcp_credentials import (
            MCPAuthType,
            MCPClientAuth,
        )

        auth = MCPClientAuth(
            auth_type=MCPAuthType.BEARER_TOKEN,
            credential_name="test_token",
            token_prefix="Bearer",
        )

        assert auth.auth_type == MCPAuthType.BEARER_TOKEN
        assert auth.header_name == "Authorization"

        # Test serialization
        data = auth.to_dict()
        restored = MCPClientAuth.from_dict(data)
        assert restored.auth_type == auth.auth_type


class TestCredentialRotation:
    """Tests for credential rotation scheduling."""

    def test_add_rotation_policy(self):
        """Test adding rotation policy."""
        from investigator.infrastructure.credential_rotation import (
            RotationPolicy,
            RotationScheduler,
        )

        scheduler = RotationScheduler()

        policy = RotationPolicy(
            credential_name="database:sec",
            rotation_interval_days=90,
            notify_before_days=14,
        )

        scheduler.add_policy(policy)

        entry = scheduler.get_schedule("database:sec")
        assert entry is not None
        assert entry.policy.rotation_interval_days == 90

    def test_record_rotation(self):
        """Test recording a rotation event."""
        from investigator.infrastructure.credential_rotation import (
            RotationPolicy,
            RotationScheduler,
        )

        scheduler = RotationScheduler()
        scheduler.add_policy(
            RotationPolicy(
                credential_name="test:cred",
                rotation_interval_days=30,
            )
        )

        scheduler.record_rotation("test:cred", rotated_by="admin")

        entry = scheduler.get_schedule("test:cred")
        assert entry.last_rotation is not None

        history = scheduler.get_rotation_history()
        assert len(history) == 1
        assert history[0].rotated_by == "admin"

    def test_get_pending_rotations(self):
        """Test getting pending rotations."""
        from investigator.infrastructure.credential_rotation import (
            RotationPolicy,
            RotationScheduler,
        )

        scheduler = RotationScheduler()
        scheduler.add_policy(
            RotationPolicy(
                credential_name="test:cred",
                rotation_interval_days=30,
                notify_before_days=14,
            )
        )

        # Force entry to be pending
        entry = scheduler.get_schedule("test:cred")
        entry.next_rotation = datetime.now() + timedelta(days=5)

        pending = scheduler.get_pending_rotations()
        assert len(pending) == 1


class TestCredentialSanitizer:
    """Tests for credential sanitization."""

    def test_scan_for_credentials(self):
        """Test scanning text for credentials."""
        from investigator.infrastructure.credential_sanitizer import scan_for_credentials

        # API key pattern
        findings = scan_for_credentials('api_key = "sk-test123456789012345678901234567890"')
        assert len(findings) > 0

        # Clean text
        findings = scan_for_credentials("Hello world")
        assert len(findings) == 0

    def test_redact_credentials(self):
        """Test credential redaction."""
        from investigator.infrastructure.credential_sanitizer import redact_credentials

        text = "password=mysupersecretpassword"
        redacted = redact_credentials(text)

        assert "mysupersecretpassword" not in redacted
        assert "***" in redacted

    def test_sanitizer_dict_scanning(self):
        """Test scanning nested dict for credentials."""
        from investigator.infrastructure.credential_sanitizer import CredentialSanitizer

        sanitizer = CredentialSanitizer()

        data = {
            "safe_key": "safe_value",
            "config": {
                "database_password": "secretpassword123",
            },
        }

        result = sanitizer.scan(data)

        assert result.has_credentials
        assert len(result.findings) > 0
        assert "secretpassword123" not in str(result.redacted_output)

    def test_strict_mode_raises_exception(self):
        """Test that strict mode raises exception on credential detection."""
        from investigator.infrastructure.credential_sanitizer import (
            CredentialLeakageError,
            CredentialSanitizer,
        )

        sanitizer = CredentialSanitizer(strict_mode=True)

        with pytest.raises(CredentialLeakageError):
            sanitizer.scan({"password": "secretvalue123"})


class TestCredentialWorkflowIntegration:
    """Integration tests for credentials in workflow context."""

    def test_workflow_yaml_credentials_required(self):
        """Test that workflow YAML supports credentials_required field."""
        import yaml

        with open("victor_invest/workflows/comprehensive.yaml") as f:
            workflow_def = yaml.safe_load(f)

        comprehensive = workflow_def["workflows"]["comprehensive"]
        nodes = comprehensive.get("nodes", [])

        # Find fetch_sec_data node
        sec_node = next((n for n in nodes if n.get("id") == "fetch_sec_data"), None)

        assert sec_node is not None
        assert "credentials_required" in sec_node

        creds = sec_node["credentials_required"]
        assert any(c.get("name") == "sec" for c in creds)

    def test_credential_validator_with_workflow(self):
        """Test credential validation against workflow definition."""
        import yaml

        from investigator.infrastructure.node_credentials import CredentialValidator

        with open("victor_invest/workflows/comprehensive.yaml") as f:
            workflow_def = yaml.safe_load(f)

        comprehensive = workflow_def["workflows"]["comprehensive"]

        # This should work if environment is set up
        with patch.dict(
            os.environ,
            {
                "SEC_DB_HOST": "localhost",
                "SEC_DB_PORT": "5432",
                "SEC_DB_NAME": "sec_db",
                "SEC_DB_USER": "user",
                "SEC_DB_PASSWORD": "pass",
                "STOCK_DB_HOST": "localhost",
                "STOCK_DB_PORT": "5432",
                "STOCK_DB_NAME": "stock",
                "STOCK_DB_USER": "user",
                "STOCK_DB_PASSWORD": "pass",
            },
        ):
            errors = CredentialValidator.validate_workflow(comprehensive)

            # Should have no errors with credentials set
            assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
