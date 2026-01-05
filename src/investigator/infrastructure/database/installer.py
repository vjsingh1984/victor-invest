# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Database Schema Installer

Installs the InvestiGator database schema to SQLite or PostgreSQL.
Supports both fresh installs and migrations.

Usage:
    # Install to SQLite (creates new database)
    python -m investigator.infrastructure.database.installer --sqlite investigator.db

    # Install to PostgreSQL
    python -m investigator.infrastructure.database.installer --postgres postgresql://user:pass@host/db

    # Check schema version
    python -m investigator.infrastructure.database.installer --check --sqlite investigator.db
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Schema directory relative to this file
SCHEMA_DIR = Path(__file__).parent.parent.parent.parent.parent / "schema" / "install"

# Schema files in order
SCHEMA_FILES = [
    "00_core_tables.sql",
    "01_market_data_tables.sql",
    "02_sentiment_tables.sql",
    "03_macro_indicators_tables.sql",
    "04_rl_tables.sql",
]


def get_schema_dir() -> Path:
    """Get the schema install directory."""
    # Try relative path first
    if SCHEMA_DIR.exists():
        return SCHEMA_DIR

    # Try from current working directory
    cwd_path = Path.cwd() / "schema" / "install"
    if cwd_path.exists():
        return cwd_path

    raise FileNotFoundError(f"Schema directory not found. Tried: {SCHEMA_DIR}, {cwd_path}")


def load_schema_sql(schema_dir: Path) -> str:
    """Load all schema SQL files and concatenate."""
    sql_parts = []

    for filename in SCHEMA_FILES:
        filepath = schema_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Schema file not found: {filepath}")

        with open(filepath, "r") as f:
            content = f.read()
            sql_parts.append(f"-- Source: {filename}")
            sql_parts.append(content)
            sql_parts.append("")

    return "\n".join(sql_parts)


def install_sqlite(db_path: str, schema_sql: str) -> Tuple[bool, str]:
    """Install schema to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute schema (SQLite can handle multi-statement)
        cursor.executescript(schema_sql)
        conn.commit()

        # Verify
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        cursor.execute("SELECT version, description FROM schema_version ORDER BY version DESC LIMIT 1")
        version_row = cursor.fetchone()
        version = version_row[0] if version_row else "unknown"

        conn.close()

        return True, f"Installed {table_count} tables, schema version: {version}"

    except Exception as e:
        return False, f"SQLite install failed: {e}"


def install_postgres(db_url: str, schema_sql: str) -> Tuple[bool, str]:
    """Install schema to PostgreSQL database."""
    try:
        from sqlalchemy import create_engine, text

        # Convert SQLite-specific syntax to PostgreSQL
        pg_sql = convert_sqlite_to_postgres(schema_sql)

        engine = create_engine(db_url)

        with engine.connect() as conn:
            # Execute each statement separately
            statements = split_sql_statements(pg_sql)
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        # Skip errors for IF NOT EXISTS statements that might conflict
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Statement failed: {stmt[:100]}... Error: {e}")
            conn.commit()

        # Verify
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
            table_count = result.scalar()

            result = conn.execute(text("SELECT version, description FROM schema_version ORDER BY version DESC LIMIT 1"))
            version_row = result.fetchone()
            version = version_row[0] if version_row else "unknown"

        return True, f"Installed {table_count} tables, schema version: {version}"

    except ImportError:
        return False, "SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary"
    except Exception as e:
        return False, f"PostgreSQL install failed: {e}"


def convert_sqlite_to_postgres(sql: str) -> str:
    """Convert SQLite-specific syntax to PostgreSQL."""
    import re

    pg_sql = sql

    # AUTOINCREMENT -> SERIAL (handled differently)
    # In PostgreSQL, we use SERIAL which auto-creates sequence
    pg_sql = re.sub(r"INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY", pg_sql, flags=re.IGNORECASE)

    # datetime('now') -> NOW()
    pg_sql = re.sub(r"datetime\('now'\)", "NOW()", pg_sql, flags=re.IGNORECASE)

    # INSERT OR IGNORE -> INSERT ... ON CONFLICT DO NOTHING
    pg_sql = re.sub(r"INSERT OR IGNORE INTO", "INSERT INTO", pg_sql, flags=re.IGNORECASE)

    # Add ON CONFLICT DO NOTHING where appropriate
    pg_sql = re.sub(r"(VALUES \([^)]+\))(\s*;)", r"\1 ON CONFLICT DO NOTHING\2", pg_sql)

    return pg_sql


def split_sql_statements(sql: str) -> list:
    """Split SQL into individual statements, respecting strings and comments."""
    statements = []
    current = []
    in_string = False
    string_char = None

    for line in sql.split("\n"):
        stripped = line.strip()

        # Skip pure comment lines
        if stripped.startswith("--"):
            continue

        # Track string state
        for i, char in enumerate(line):
            if char in ("'", '"') and (i == 0 or line[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

        current.append(line)

        # Statement ends with ; outside of string
        if stripped.endswith(";") and not in_string:
            statement = "\n".join(current).strip()
            if statement and not statement.startswith("--"):
                statements.append(statement)
            current = []

    # Handle any remaining content
    if current:
        statement = "\n".join(current).strip()
        if statement and not statement.startswith("--"):
            statements.append(statement)

    return statements


def check_schema_version(db_url: str, is_sqlite: bool = False) -> Tuple[bool, str]:
    """Check the current schema version."""
    try:
        if is_sqlite:
            conn = sqlite3.connect(db_url)
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT version, description, applied_at FROM schema_version ORDER BY version DESC")
                rows = cursor.fetchall()
                if rows:
                    result = "\n".join([f"  {r[0]}: {r[1]} ({r[2]})" for r in rows])
                    return True, f"Schema versions:\n{result}"
                else:
                    return True, "No schema versions found (empty database)"
            except sqlite3.OperationalError:
                return False, "schema_version table does not exist"
            finally:
                conn.close()
        else:
            from sqlalchemy import create_engine, text

            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT version, description, applied_at FROM schema_version ORDER BY version DESC")
                )
                rows = result.fetchall()
                if rows:
                    result_str = "\n".join([f"  {r[0]}: {r[1]} ({r[2]})" for r in rows])
                    return True, f"Schema versions:\n{result_str}"
                else:
                    return True, "No schema versions found"

    except Exception as e:
        return False, f"Check failed: {e}"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Install InvestiGator database schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh SQLite install
  python -m investigator.infrastructure.database.installer --sqlite investigator.db

  # PostgreSQL install
  python -m investigator.infrastructure.database.installer --postgres postgresql://user:pass@host/db

  # Check current version
  python -m investigator.infrastructure.database.installer --check --sqlite investigator.db
        """,
    )

    parser.add_argument("--sqlite", metavar="PATH", help="SQLite database path")
    parser.add_argument("--postgres", metavar="URL", help="PostgreSQL connection URL")
    parser.add_argument("--check", action="store_true", help="Check schema version instead of installing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if not args.sqlite and not args.postgres:
        parser.error("Either --sqlite or --postgres is required")

    if args.check:
        if args.sqlite:
            success, message = check_schema_version(args.sqlite, is_sqlite=True)
        else:
            success, message = check_schema_version(args.postgres, is_sqlite=False)
        print(message)
        sys.exit(0 if success else 1)

    # Load schema
    try:
        schema_dir = get_schema_dir()
        logger.info(f"Loading schema from: {schema_dir}")
        schema_sql = load_schema_sql(schema_dir)
        logger.info(f"Loaded {len(schema_sql)} characters of SQL")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Install
    if args.sqlite:
        logger.info(f"Installing to SQLite: {args.sqlite}")
        success, message = install_sqlite(args.sqlite, schema_sql)
    else:
        logger.info(f"Installing to PostgreSQL: {args.postgres}")
        success, message = install_postgres(args.postgres, schema_sql)

    if success:
        logger.info(f"✓ {message}")
        sys.exit(0)
    else:
        logger.error(f"✗ {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
