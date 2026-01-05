"""
Credential management for InvestiGator.

Provides a unified interface for database credentials that:
1. Integrates with Victor framework's CredentialManager (if available)
2. Falls back to environment variables
3. Supports ~/.investigator/env sourcing pattern

Usage:
    from investigator.infrastructure.credentials import get_database_credentials

    # Get SEC database credentials
    creds = get_database_credentials("sec")

    # Get Stock database credentials
    creds = get_database_credentials("stock")

    # Access credentials
    print(creds.host, creds.port, creds.database, creds.username, creds.password)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DatabaseCredentials:
    """Database connection credentials."""

    alias: str
    host: str
    port: int
    database: str
    username: str
    password: str
    driver: str = "postgresql"

    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"{self.driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def psycopg2_dsn(self) -> str:
        """Generate psycopg2 DSN string."""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.username} password={self.password}"


def _get_from_victor_framework(alias: str) -> Optional[DatabaseCredentials]:
    """Try to get credentials from Victor framework's CredentialManager."""
    try:
        from victor.workflows.services.credentials import get_credential_manager

        cred_mgr = get_credential_manager()
        victor_creds = cred_mgr.get_database(alias)

        if victor_creds:
            logger.debug(f"Got {alias} database credentials from Victor CredentialManager")
            return DatabaseCredentials(
                alias=victor_creds.alias,
                host=victor_creds.host,
                port=victor_creds.port,
                database=victor_creds.database,
                username=victor_creds.username,
                password=victor_creds.password,
                driver=victor_creds.driver,
            )
    except ImportError:
        logger.debug("Victor framework not available, falling back to environment variables")
    except Exception as e:
        logger.debug(f"Victor CredentialManager error: {e}, falling back to environment variables")

    return None


def _get_from_environment(alias: str) -> Optional[DatabaseCredentials]:
    """Get credentials from environment variables.

    Supports two patterns:
    - SEC database: SEC_DB_HOST, SEC_DB_PORT, SEC_DB_NAME, SEC_DB_USER, SEC_DB_PASSWORD
    - Stock database: STOCK_DB_HOST, STOCK_DB_PORT, STOCK_DB_NAME, STOCK_DB_USER, STOCK_DB_PASSWORD
    - Legacy: DB_HOST, DB_PASSWORD, DB_USERNAME, DB_DATABASE (maps to SEC)
    """
    prefix_map = {
        "sec": "SEC_DB",
        "sec_database": "SEC_DB",
        "stock": "STOCK_DB",
        "market": "STOCK_DB",
    }

    prefix = prefix_map.get(alias.lower())

    if not prefix:
        logger.warning(f"Unknown database alias: {alias}")
        return None

    # Try prefixed environment variables
    host = os.environ.get(f"{prefix}_HOST")
    port = os.environ.get(f"{prefix}_PORT", "5432")
    database = os.environ.get(f"{prefix}_NAME")
    username = os.environ.get(f"{prefix}_USER")
    password = os.environ.get(f"{prefix}_PASSWORD")

    # Fallback to legacy variables for SEC database
    if alias.lower() in ("sec", "sec_database") and not host:
        host = os.environ.get("DB_HOST")
        password = password or os.environ.get("DB_PASSWORD")
        username = username or os.environ.get("DB_USERNAME")
        database = database or os.environ.get("DB_DATABASE")

    if not all([host, database, username, password]):
        missing = []
        if not host:
            missing.append(f"{prefix}_HOST")
        if not database:
            missing.append(f"{prefix}_NAME")
        if not username:
            missing.append(f"{prefix}_USER")
        if not password:
            missing.append(f"{prefix}_PASSWORD")
        logger.debug(f"Missing environment variables for {alias}: {missing}")
        return None

    return DatabaseCredentials(
        alias=alias,
        host=host,
        port=int(port),
        database=database,
        username=username,
        password=password,
    )


def get_database_credentials(alias: str) -> DatabaseCredentials:
    """Get database credentials with layered resolution.

    Resolution order:
    1. Victor framework CredentialManager (if available)
    2. Environment variables (SEC_DB_*, STOCK_DB_*, or legacy DB_*)

    Args:
        alias: Database alias ("sec", "stock", "market")

    Returns:
        DatabaseCredentials object

    Raises:
        EnvironmentError: If credentials cannot be found
    """
    # Try Victor framework first
    creds = _get_from_victor_framework(alias)
    if creds:
        return creds

    # Fall back to environment variables
    creds = _get_from_environment(alias)
    if creds:
        logger.debug(f"Got {alias} database credentials from environment variables")
        return creds

    # No credentials found
    raise EnvironmentError(
        f"Database credentials for '{alias}' not found. "
        f"Please either:\n"
        f"  1. Source your environment file: source ~/.investigator/env\n"
        f"  2. Set environment variables: {alias.upper()}_DB_HOST, {alias.upper()}_DB_PASSWORD, etc.\n"
        f"  3. Configure Victor framework credentials"
    )


def validate_database_connection(alias: str) -> bool:
    """Validate that database credentials work.

    Args:
        alias: Database alias to validate

    Returns:
        True if connection successful, False otherwise
    """
    try:
        import psycopg2
        creds = get_database_credentials(alias)

        conn = psycopg2.connect(
            host=creds.host,
            port=creds.port,
            dbname=creds.database,
            user=creds.username,
            password=creds.password,
            connect_timeout=5,
        )
        conn.close()
        logger.info(f"✓ Database connection validated: {alias} @ {creds.host}")
        return True

    except EnvironmentError as e:
        logger.error(f"✗ Credentials not found for {alias}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Database connection failed for {alias}: {e}")
        return False


__all__ = [
    "DatabaseCredentials",
    "get_database_credentials",
    "validate_database_connection",
]
