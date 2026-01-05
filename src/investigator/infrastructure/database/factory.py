# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Database Factory - Multi-RDBMS Support

Provides a unified database interface that works with both SQLite and PostgreSQL.
This allows the system to run tests with SQLite while using PostgreSQL in production.

Usage:
    from investigator.infrastructure.database.factory import get_database, DatabaseType

    # Get default database (from config)
    db = get_database()

    # Get specific database type
    db = get_database(DatabaseType.SQLITE, path="test.db")
    db = get_database(DatabaseType.POSTGRES, url="postgresql://...")

    # Execute queries
    with db.connect() as conn:
        result = conn.execute("SELECT * FROM symbols WHERE ticker = ?", ["AAPL"])
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"


class DatabaseConnection(ABC):
    """Abstract database connection interface."""

    @abstractmethod
    def execute(self, sql: str, params: Optional[List] = None) -> Any:
        """Execute SQL query and return cursor/result."""
        pass

    @abstractmethod
    def fetchone(self) -> Optional[Tuple]:
        """Fetch one row from last query."""
        pass

    @abstractmethod
    def fetchall(self) -> List[Tuple]:
        """Fetch all rows from last query."""
        pass

    @abstractmethod
    def fetchmany(self, size: int) -> List[Tuple]:
        """Fetch N rows from last query."""
        pass

    @abstractmethod
    def commit(self):
        """Commit transaction."""
        pass

    @abstractmethod
    def rollback(self):
        """Rollback transaction."""
        pass

    @abstractmethod
    def close(self):
        """Close connection."""
        pass

    @property
    @abstractmethod
    def rowcount(self) -> int:
        """Number of rows affected by last query."""
        pass


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = None
        self._rowcount = 0

    def execute(self, sql: str, params: Optional[List] = None) -> "SQLiteConnection":
        """Execute SQL with SQLite parameter style (?)."""
        # Convert PostgreSQL style (:name) to SQLite style (?)
        converted_sql, converted_params = self._convert_params(sql, params)
        self.cursor = self.conn.execute(converted_sql, converted_params or [])
        self._rowcount = self.cursor.rowcount
        return self

    def _convert_params(self, sql: str, params: Optional[Union[List, Dict]]) -> Tuple[str, Optional[List]]:
        """Convert PostgreSQL named params to SQLite positional params."""
        if params is None:
            return sql, None

        if isinstance(params, dict):
            # Convert :name style to ? style
            import re

            param_list = []
            pattern = re.compile(r":(\w+)")

            def replacer(match):
                name = match.group(1)
                if name in params:
                    param_list.append(params[name])
                    return "?"
                return match.group(0)

            converted_sql = pattern.sub(replacer, sql)
            return converted_sql, param_list
        else:
            return sql, params

    def fetchone(self) -> Optional[Tuple]:
        return self.cursor.fetchone() if self.cursor else None

    def fetchall(self) -> List[Tuple]:
        return self.cursor.fetchall() if self.cursor else []

    def fetchmany(self, size: int) -> List[Tuple]:
        return self.cursor.fetchmany(size) if self.cursor else []

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

    @property
    def rowcount(self) -> int:
        return self._rowcount


class PostgresConnection(DatabaseConnection):
    """PostgreSQL database connection using SQLAlchemy."""

    def __init__(self, engine):
        from sqlalchemy import text

        self.engine = engine
        self.conn = engine.connect()
        self.result = None
        self._rowcount = 0
        self._text = text

    def execute(self, sql: str, params: Optional[Union[List, Dict]] = None) -> "PostgresConnection":
        """Execute SQL with PostgreSQL."""
        # Handle both list and dict params
        if isinstance(params, list):
            # Convert list to dict with numbered keys
            params = {str(i): v for i, v in enumerate(params)}

        self.result = self.conn.execute(self._text(sql), params or {})
        self._rowcount = self.result.rowcount
        return self

    def fetchone(self) -> Optional[Tuple]:
        return self.result.fetchone() if self.result else None

    def fetchall(self) -> List[Tuple]:
        return self.result.fetchall() if self.result else []

    def fetchmany(self, size: int) -> List[Tuple]:
        return self.result.fetchmany(size) if self.result else []

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

    @property
    def rowcount(self) -> int:
        return self._rowcount


class Database(ABC):
    """Abstract database interface."""

    @abstractmethod
    def connect(self) -> DatabaseConnection:
        """Get a new connection."""
        pass

    @contextmanager
    def session(self) -> Iterator[DatabaseConnection]:
        """Context manager for database sessions."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @abstractmethod
    def execute_script(self, sql: str):
        """Execute multi-statement SQL script."""
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        pass

    @abstractmethod
    def get_schema_version(self) -> Optional[str]:
        """Get current schema version."""
        pass

    @property
    @abstractmethod
    def db_type(self) -> DatabaseType:
        """Get database type."""
        pass


class SQLiteDatabase(Database):
    """SQLite database implementation."""

    def __init__(self, path: str):
        self.path = path
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"SQLite database: {path}")

    def connect(self) -> SQLiteConnection:
        return SQLiteConnection(self.path)

    def execute_script(self, sql: str):
        """Execute multi-statement SQL script."""
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(sql)
            conn.commit()
        finally:
            conn.close()

    def table_exists(self, table_name: str) -> bool:
        conn = self.connect()
        try:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table_name]
            ).fetchone()
            return result is not None
        finally:
            conn.close()

    def get_schema_version(self) -> Optional[str]:
        if not self.table_exists("schema_version"):
            return None
        conn = self.connect()
        try:
            result = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    @property
    def db_type(self) -> DatabaseType:
        return DatabaseType.SQLITE


class PostgresDatabase(Database):
    """PostgreSQL database implementation using SQLAlchemy."""

    def __init__(self, url: str):
        from sqlalchemy import create_engine

        self.url = url
        self.engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        logger.info(f"PostgreSQL database connected")

    def connect(self) -> PostgresConnection:
        return PostgresConnection(self.engine)

    def execute_script(self, sql: str):
        """Execute multi-statement SQL script."""
        from sqlalchemy import text

        with self.engine.begin() as conn:
            # Split by semicolon and execute
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"SQL error: {e}")

    def table_exists(self, table_name: str) -> bool:
        conn = self.connect()
        try:
            result = conn.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = :name
                """,
                {"name": table_name},
            ).fetchone()
            return result is not None
        finally:
            conn.close()

    def get_schema_version(self) -> Optional[str]:
        if not self.table_exists("schema_version"):
            return None
        conn = self.connect()
        try:
            result = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    @property
    def db_type(self) -> DatabaseType:
        return DatabaseType.POSTGRES


# Global database instance
_database: Optional[Database] = None


def get_database(
    db_type: Optional[DatabaseType] = None,
    path: Optional[str] = None,
    url: Optional[str] = None,
) -> Database:
    """
    Get database instance.

    Args:
        db_type: Database type (SQLITE or POSTGRES). If None, uses config.
        path: SQLite database path (for SQLITE type)
        url: PostgreSQL connection URL (for POSTGRES type)

    Returns:
        Database instance
    """
    global _database

    # Return cached instance if no specific params
    if _database is not None and db_type is None and path is None and url is None:
        return _database

    # Determine type from params or config
    if db_type is None:
        if path:
            db_type = DatabaseType.SQLITE
        elif url:
            db_type = DatabaseType.POSTGRES
        else:
            # Get from config
            try:
                from investigator.config import get_config

                config = get_config()
                url = config.database.url
                db_type = DatabaseType.POSTGRES
            except Exception:
                # Default to SQLite for testing
                db_type = DatabaseType.SQLITE
                path = "investigator.db"

    # Create database
    if db_type == DatabaseType.SQLITE:
        db = SQLiteDatabase(path or "investigator.db")
    elif db_type == DatabaseType.POSTGRES:
        if not url:
            from investigator.config import get_config

            config = get_config()
            url = config.database.url
        db = PostgresDatabase(url)
    else:
        raise ValueError(f"Unknown database type: {db_type}")

    # Cache if no specific params
    if path is None and url is None:
        _database = db

    return db


def reset_database():
    """Reset global database instance (for testing)."""
    global _database
    _database = None


def install_schema(db: Database, schema_dir: Optional[Path] = None) -> bool:
    """
    Install database schema from SQL files.

    Args:
        db: Database instance
        schema_dir: Path to schema/install directory

    Returns:
        True if successful
    """
    if schema_dir is None:
        schema_dir = Path(__file__).parent.parent.parent.parent.parent / "schema" / "install"

    schema_files = [
        "00_core_tables.sql",
        "01_market_data_tables.sql",
        "02_sentiment_tables.sql",
        "03_macro_indicators_tables.sql",
        "04_rl_tables.sql",
    ]

    try:
        for filename in schema_files:
            filepath = schema_dir / filename
            if not filepath.exists():
                logger.warning(f"Schema file not found: {filepath}")
                continue

            with open(filepath, "r") as f:
                sql = f.read()

            logger.info(f"Installing {filename}...")
            db.execute_script(sql)

        version = db.get_schema_version()
        logger.info(f"Schema installed, version: {version}")
        return True

    except Exception as e:
        logger.error(f"Schema installation failed: {e}")
        return False
