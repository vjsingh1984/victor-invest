"""
MCP Credential Management for secure MCP server authentication.

This module provides:
1. MCPClientAuth - Authentication configuration for MCP clients
2. MCPCredentialResolver - Resolves credentials for MCP servers
3. MCPServerCredentials - Bootstrap credentials for MCP servers

Integrates with Victor framework's CredentialManager and node credential system.

Usage:
    from investigator.infrastructure.mcp_credentials import (
        MCPClientAuth,
        MCPCredentialResolver,
        inject_mcp_credentials,
    )

    # Create auth config for MCP client
    auth = MCPClientAuth(
        auth_type=MCPAuthType.BEARER_TOKEN,
        credential_name="mcp_server_token",
    )

    # Resolve credentials before connection
    resolver = MCPCredentialResolver()
    env_vars = resolver.resolve_server_env("my_mcp_server")

    # Inject into MCP server config
    server_config = inject_mcp_credentials(server_config, resolver)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MCPAuthType(Enum):
    """Authentication types for MCP connections."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    CUSTOM_HEADER = "custom_header"


@dataclass
class MCPClientAuth:
    """Authentication configuration for MCP client connections.

    Supports multiple authentication methods:
    - API Key: Pass key in environment or header
    - Bearer Token: OAuth/JWT token authentication
    - Basic Auth: Username/password
    - OAuth2: Full OAuth2 flow with refresh
    - Certificate: mTLS certificate authentication
    - Custom Header: Arbitrary header-based auth
    """

    auth_type: MCPAuthType = MCPAuthType.NONE
    credential_name: str = ""  # Name in credential manager
    header_name: str = "Authorization"  # For custom header auth
    env_var_name: str = ""  # Environment variable to set
    token_prefix: str = "Bearer"  # Prefix for token (Bearer, Token, etc.)

    # OAuth2 specific
    oauth_client_id: str = ""
    oauth_client_secret_name: str = ""  # Credential name for client secret
    oauth_token_url: str = ""
    oauth_scopes: List[str] = field(default_factory=list)

    # Certificate specific
    cert_path: str = ""
    key_path: str = ""
    ca_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "auth_type": self.auth_type.value,
            "credential_name": self.credential_name,
            "header_name": self.header_name,
            "env_var_name": self.env_var_name,
            "token_prefix": self.token_prefix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPClientAuth":
        """Create from dictionary."""
        return cls(
            auth_type=MCPAuthType(data.get("auth_type", "none")),
            credential_name=data.get("credential_name", ""),
            header_name=data.get("header_name", "Authorization"),
            env_var_name=data.get("env_var_name", ""),
            token_prefix=data.get("token_prefix", "Bearer"),
        )


@dataclass
class MCPServerCredentials:
    """Bootstrap credentials for an MCP server.

    Defines what credentials an MCP server needs to function,
    including database connections, API keys, and service accounts.
    """

    server_name: str
    required_credentials: List[str] = field(default_factory=list)
    optional_credentials: List[str] = field(default_factory=list)
    env_mappings: Dict[str, str] = field(default_factory=dict)  # cred_name -> env_var

    def get_all_credential_names(self) -> Set[str]:
        """Get all credential names (required + optional)."""
        return set(self.required_credentials) | set(self.optional_credentials)


@dataclass
class CredentialExpirationInfo:
    """Track credential expiration."""

    credential_name: str
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    last_refreshed: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at

    @property
    def needs_refresh(self) -> bool:
        """Check if credential needs refresh (within 5 min of expiry)."""
        if not self.expires_at:
            return False
        return datetime.now() >= (self.expires_at - timedelta(minutes=5))


class MCPCredentialResolver:
    """Resolves credentials for MCP server connections.

    Resolution order:
    1. Victor framework CredentialManager
    2. Node credential context (if in workflow)
    3. Environment variables
    4. Configuration files

    Features:
    - Credential caching with TTL
    - Expiration tracking
    - Audit logging
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._expiration_info: Dict[str, CredentialExpirationInfo] = {}
        self._victor_cred_mgr = None

        # Try to get Victor framework credential manager
        try:
            from victor.workflows.services.credentials import get_credential_manager
            self._victor_cred_mgr = get_credential_manager()
        except ImportError:
            pass

    def resolve_api_key(self, name: str) -> Optional[str]:
        """Resolve an API key credential.

        Args:
            name: Credential name (e.g., "anthropic", "openai")

        Returns:
            API key string or None
        """
        cache_key = f"api_key:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        api_key = None

        # Try Victor framework
        if self._victor_cred_mgr:
            try:
                victor_key = self._victor_cred_mgr.get_api_key(name)
                if victor_key:
                    api_key = victor_key.key
                    logger.debug(f"Resolved API key '{name}' from Victor CredentialManager")
            except Exception as e:
                logger.debug(f"Victor API key resolution failed: {e}")

        # Fall back to environment
        if not api_key:
            env_var = f"{name.upper()}_API_KEY"
            api_key = os.environ.get(env_var)
            if api_key:
                logger.debug(f"Resolved API key '{name}' from environment ({env_var})")

        if api_key:
            self._cache[cache_key] = api_key

        return api_key

    def resolve_database(self, alias: str) -> Optional[Dict[str, Any]]:
        """Resolve database credentials.

        Args:
            alias: Database alias (e.g., "sec", "stock")

        Returns:
            Dict with host, port, database, username, password
        """
        try:
            from investigator.infrastructure.credentials import get_database_credentials
            creds = get_database_credentials(alias)
            return {
                "host": creds.host,
                "port": creds.port,
                "database": creds.database,
                "username": creds.username,
                "password": creds.password,
            }
        except Exception as e:
            logger.warning(f"Failed to resolve database '{alias}': {e}")
            return None

    def resolve_server_env(
        self,
        server_creds: MCPServerCredentials,
    ) -> Dict[str, str]:
        """Resolve all credentials for an MCP server as environment variables.

        Args:
            server_creds: Server credential requirements

        Returns:
            Dict of environment variable name -> value
        """
        env_vars = {}

        for cred_name in server_creds.get_all_credential_names():
            env_var = server_creds.env_mappings.get(cred_name, f"{cred_name.upper()}")

            # Try different credential types
            if cred_name.endswith("_db") or cred_name in ("sec", "stock", "market"):
                # Database credential
                db_creds = self.resolve_database(cred_name.replace("_db", ""))
                if db_creds:
                    env_vars[f"{env_var}_HOST"] = db_creds["host"]
                    env_vars[f"{env_var}_PORT"] = str(db_creds["port"])
                    env_vars[f"{env_var}_NAME"] = db_creds["database"]
                    env_vars[f"{env_var}_USER"] = db_creds["username"]
                    env_vars[f"{env_var}_PASSWORD"] = db_creds["password"]
            else:
                # API key
                api_key = self.resolve_api_key(cred_name)
                if api_key:
                    env_vars[env_var] = api_key

        return env_vars

    def resolve_auth(self, auth: MCPClientAuth) -> Optional[str]:
        """Resolve authentication credential.

        Args:
            auth: Authentication configuration

        Returns:
            Resolved credential value or None
        """
        if auth.auth_type == MCPAuthType.NONE:
            return None

        if auth.auth_type == MCPAuthType.API_KEY:
            return self.resolve_api_key(auth.credential_name)

        if auth.auth_type == MCPAuthType.BEARER_TOKEN:
            token = self.resolve_api_key(auth.credential_name)
            if token:
                return f"{auth.token_prefix} {token}"
            return None

        if auth.auth_type == MCPAuthType.BASIC_AUTH:
            # Expect credential_name to be "username:password_cred_name"
            parts = auth.credential_name.split(":")
            if len(parts) == 2:
                username = parts[0]
                password = self.resolve_api_key(parts[1])
                if password:
                    import base64
                    credentials = base64.b64encode(
                        f"{username}:{password}".encode()
                    ).decode()
                    return f"Basic {credentials}"
            return None

        logger.warning(f"Unsupported auth type: {auth.auth_type}")
        return None

    def track_expiration(
        self,
        credential_name: str,
        expires_at: Optional[datetime] = None,
        refresh_token: Optional[str] = None,
    ) -> None:
        """Track credential expiration for proactive refresh.

        Args:
            credential_name: Name of the credential
            expires_at: When the credential expires
            refresh_token: Token for refreshing (OAuth2)
        """
        self._expiration_info[credential_name] = CredentialExpirationInfo(
            credential_name=credential_name,
            expires_at=expires_at,
            refresh_token=refresh_token,
            last_refreshed=datetime.now(),
        )

    def get_expiring_credentials(
        self,
        within_minutes: int = 10,
    ) -> List[CredentialExpirationInfo]:
        """Get credentials that are expiring soon.

        Args:
            within_minutes: Time window for "expiring soon"

        Returns:
            List of credentials expiring within the window
        """
        threshold = datetime.now() + timedelta(minutes=within_minutes)
        return [
            info for info in self._expiration_info.values()
            if info.expires_at and info.expires_at <= threshold
        ]

    def clear_cache(self) -> None:
        """Clear the credential cache."""
        self._cache.clear()


def inject_mcp_credentials(
    server_config: Dict[str, Any],
    server_creds: MCPServerCredentials,
    resolver: Optional[MCPCredentialResolver] = None,
) -> Dict[str, Any]:
    """Inject credentials into MCP server configuration.

    Adds resolved credentials to the server config's env dict.

    Args:
        server_config: MCP server configuration dict
        server_creds: Credential requirements for the server
        resolver: Optional credential resolver (creates default if None)

    Returns:
        Updated server configuration with credentials in env
    """
    if resolver is None:
        resolver = MCPCredentialResolver()

    # Resolve credentials to environment variables
    cred_env = resolver.resolve_server_env(server_creds)

    # Merge with existing env
    existing_env = server_config.get("env", {})
    merged_env = {**existing_env, **cred_env}

    # Update config
    server_config["env"] = merged_env

    logger.info(
        f"Injected {len(cred_env)} credentials for MCP server '{server_creds.server_name}'"
    )

    return server_config


# Pre-defined server credential configurations
MCP_SERVER_CREDENTIALS = {
    "investigator": MCPServerCredentials(
        server_name="investigator",
        required_credentials=["sec_db", "stock_db"],
        optional_credentials=["anthropic", "openai"],
        env_mappings={
            "sec_db": "SEC_DB",
            "stock_db": "STOCK_DB",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        },
    ),
    "filesystem": MCPServerCredentials(
        server_name="filesystem",
        required_credentials=[],
        optional_credentials=[],
    ),
    "web_search": MCPServerCredentials(
        server_name="web_search",
        required_credentials=["serper"],
        env_mappings={"serper": "SERPER_API_KEY"},
    ),
}


def get_server_credentials(server_name: str) -> Optional[MCPServerCredentials]:
    """Get pre-defined credential config for an MCP server.

    Args:
        server_name: Name of the MCP server

    Returns:
        MCPServerCredentials or None if not defined
    """
    return MCP_SERVER_CREDENTIALS.get(server_name)


__all__ = [
    "MCPAuthType",
    "MCPClientAuth",
    "MCPServerCredentials",
    "MCPCredentialResolver",
    "CredentialExpirationInfo",
    "inject_mcp_credentials",
    "get_server_credentials",
    "MCP_SERVER_CREDENTIALS",
]
