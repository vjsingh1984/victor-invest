#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base utilities for scheduled data collection scripts.

Provides common functionality for:
- Logging setup with rotation
- Database connection management
- Error handling and retry logic
- Metrics collection
- Lock file management (prevent concurrent runs)
- Incremental and idempotent data collection
- Hash-based change detection
"""

import hashlib
import json
import logging
import os
import sys
import time
import fcntl
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class CollectionMetrics:
    """Metrics for a data collection run."""
    job_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    records_skipped: int = 0  # Records skipped due to no changes (incremental)
    high_watermark_date: Optional[date] = None
    high_watermark_value: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_processed": self.records_processed,
            "records_inserted": self.records_inserted,
            "records_updated": self.records_updated,
            "records_failed": self.records_failed,
            "success": self.success,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }

    def log_summary(self, logger: logging.Logger) -> None:
        """Log a summary of the collection run."""
        if self.success:
            logger.info(
                f"[{self.job_name}] Completed successfully in {self.duration_seconds:.1f}s - "
                f"Processed: {self.records_processed}, "
                f"Inserted: {self.records_inserted}, "
                f"Updated: {self.records_updated}, "
                f"Skipped: {self.records_skipped}"
            )
        else:
            logger.error(
                f"[{self.job_name}] Failed after {self.duration_seconds:.1f}s - "
                f"Errors: {len(self.errors)}, "
                f"Processed: {self.records_processed}"
            )
            for error in self.errors[:5]:
                logger.error(f"  - {error}")


class LockFile:
    """Context manager for preventing concurrent script execution."""

    def __init__(self, job_name: str, lock_dir: Optional[Path] = None):
        self.job_name = job_name
        self.lock_dir = lock_dir or PROJECT_ROOT / "logs" / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.lock_dir / f"{job_name}.lock"
        self.lock_file = None

    def __enter__(self):
        self.lock_file = open(self.lock_path, "w")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_file.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.lock_file.flush()
            return self
        except BlockingIOError:
            self.lock_file.close()
            raise RuntimeError(
                f"Job '{self.job_name}' is already running. "
                f"Lock file: {self.lock_path}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass


def setup_logging(
    job_name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Setup logging for a scheduled job.

    Args:
        job_name: Name of the job (used for log file name)
        log_dir: Directory for log files (default: PROJECT_ROOT/logs)
        level: Logging level
        console: Whether to also log to console

    Returns:
        Configured logger
    """
    log_dir = log_dir or PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(job_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # File handler with rotation
    log_file = log_dir / f"{job_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_database_connection():
    """Get a database connection using the application config.

    Returns:
        Database connection object
    """
    try:
        from investigator.config.config import get_config
        import psycopg2

        config = get_config()
        db_config = config.database

        connection = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            dbname=db_config.database,
            user=db_config.username,
            password=db_config.password,
        )
        return connection
    except Exception as e:
        raise RuntimeError(f"Failed to connect to database: {e}")


def get_sp500_symbols() -> List[str]:
    """Get list of S&P 500 symbols from database.

    Returns:
        List of S&P 500 ticker symbols
    """
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol FROM companies
            WHERE is_sp500 = true AND is_active = true
            ORDER BY symbol
        """)
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return symbols
    except Exception:
        # Fallback to a static list if database query fails
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
            "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "ABBV",
            "MRK", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
            "TMO", "ABT", "DHR", "ACN", "VZ", "ADBE", "CRM", "CMCSA", "PFE",
            "INTC", "NKE", "DIS", "TXN", "WFC", "PM", "NEE", "UPS", "RTX",
            "BMY", "QCOM", "HON", "LOW",
        ]


def get_russell1000_symbols() -> List[str]:
    """Get list of Russell 1000 symbols from database.

    Returns:
        List of Russell 1000 ticker symbols
    """
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol FROM companies
            WHERE is_russell1000 = true AND is_active = true
            ORDER BY symbol
        """)
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return symbols if symbols else get_sp500_symbols()
    except Exception:
        return get_sp500_symbols()


def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
):
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        logger: Logger for retry messages

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                if logger:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                if logger:
                    logger.error(f"All {max_retries + 1} attempts failed: {e}")

    raise last_exception


# ============================================================================
# Rate Limiting with Fibonacci Backoff
# ============================================================================


class FibonacciRateLimiter:
    """Rate limiter with Fibonacci backoff for API calls.

    Features:
    - Enforces calls per minute limit using sliding window
    - Fibonacci backoff on rate limit errors (429)
    - Thread-safe for concurrent usage
    - Automatic recovery after backoff period

    Usage:
        limiter = FibonacciRateLimiter(calls_per_minute=60)

        # Option 1: Context manager
        with limiter:
            response = requests.get(url)

        # Option 2: Decorator
        @limiter.limit
        def make_request():
            return requests.get(url)

        # Option 3: With retry on 429
        response = limiter.execute_with_retry(lambda: requests.get(url))
    """

    # Fibonacci sequence for backoff: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...
    FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    def __init__(
        self,
        calls_per_minute: int = 60,
        max_retries: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize rate limiter.

        Args:
            calls_per_minute: Maximum API calls allowed per minute
            max_retries: Maximum retries on rate limit (429) errors
            logger: Optional logger for status messages
        """
        self.calls_per_minute = calls_per_minute
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)

        # Sliding window of request timestamps
        self._request_times: deque = deque()
        self._lock = threading.Lock()

        # Backoff state
        self._consecutive_429s = 0
        self._backoff_until: Optional[float] = None

    def _get_fibonacci_delay(self, attempt: int) -> float:
        """Get Fibonacci backoff delay for given attempt number."""
        idx = min(attempt, len(self.FIBONACCI) - 1)
        return float(self.FIBONACCI[idx])

    def _wait_for_slot(self) -> None:
        """Wait until a request slot is available."""
        with self._lock:
            now = time.time()

            # Check if we're in backoff period
            if self._backoff_until and now < self._backoff_until:
                wait_time = self._backoff_until - now
                self.logger.debug(f"In backoff period, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                now = time.time()

            # Remove timestamps older than 60 seconds
            cutoff = now - 60.0
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()

            # Wait if at capacity
            if len(self._request_times) >= self.calls_per_minute:
                oldest = self._request_times[0]
                wait_time = oldest + 60.0 - now + 0.1  # Add small buffer
                if wait_time > 0:
                    self.logger.debug(
                        f"Rate limit reached ({len(self._request_times)}/{self.calls_per_minute}), "
                        f"waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    # Clean up after waiting
                    now = time.time()
                    cutoff = now - 60.0
                    while self._request_times and self._request_times[0] < cutoff:
                        self._request_times.popleft()

            # Record this request
            self._request_times.append(time.time())

    def record_success(self) -> None:
        """Record a successful request (resets backoff state)."""
        with self._lock:
            self._consecutive_429s = 0
            self._backoff_until = None

    def record_rate_limit(self) -> float:
        """Record a 429 rate limit error and return backoff delay.

        Returns:
            Backoff delay in seconds
        """
        with self._lock:
            self._consecutive_429s += 1
            delay = self._get_fibonacci_delay(self._consecutive_429s)
            self._backoff_until = time.time() + delay
            self.logger.warning(
                f"Rate limited (429). Fibonacci backoff: {delay}s "
                f"(attempt {self._consecutive_429s})"
            )
            return delay

    def __enter__(self) -> "FibonacciRateLimiter":
        """Context manager entry - wait for available slot."""
        self._wait_for_slot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass

    def limit(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate limit a function."""
        def wrapper(*args, **kwargs) -> T:
            self._wait_for_slot()
            return func(*args, **kwargs)
        return wrapper

    def execute_with_retry(
        self,
        func: Callable[[], T],
        is_rate_limit: Optional[Callable[[Exception], bool]] = None,
    ) -> T:
        """Execute function with automatic retry on rate limit errors.

        Args:
            func: Function to execute (should return response or raise exception)
            is_rate_limit: Optional function to check if exception is rate limit.
                          Default checks for 429 status code.

        Returns:
            Result of func()

        Raises:
            Last exception if all retries exhausted
        """
        if is_rate_limit is None:
            def is_rate_limit(e: Exception) -> bool:
                # Check common patterns for 429 errors
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    return e.response.status_code == 429
                return "429" in str(e) or "rate limit" in str(e).lower()

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_slot()
                result = func()
                self.record_success()
                return result
            except Exception as e:
                last_exception = e
                if is_rate_limit(e) and attempt < self.max_retries:
                    delay = self.record_rate_limit()
                    time.sleep(delay)
                else:
                    raise

        raise last_exception

    @property
    def current_usage(self) -> int:
        """Get current number of requests in the sliding window."""
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            return len(self._request_times)


# Shared rate limiter instances for common APIs
_finnhub_limiter: Optional[FibonacciRateLimiter] = None
_fred_limiter: Optional[FibonacciRateLimiter] = None


def get_finnhub_rate_limiter() -> FibonacciRateLimiter:
    """Get shared Finnhub rate limiter (60 calls/minute)."""
    global _finnhub_limiter
    if _finnhub_limiter is None:
        _finnhub_limiter = FibonacciRateLimiter(
            calls_per_minute=55,  # Leave buffer below 60
            max_retries=5,
            logger=logging.getLogger("finnhub_rate_limiter"),
        )
    return _finnhub_limiter


def get_fred_rate_limiter() -> FibonacciRateLimiter:
    """Get shared FRED rate limiter (120 calls/minute)."""
    global _fred_limiter
    if _fred_limiter is None:
        _fred_limiter = FibonacciRateLimiter(
            calls_per_minute=100,  # Leave buffer below 120
            max_retries=5,
            logger=logging.getLogger("fred_rate_limiter"),
        )
    return _fred_limiter


# ============================================================================
# Incremental Collection Utilities
# ============================================================================


def compute_record_hash(data: Union[Dict, str, Any]) -> str:
    """Compute a SHA256 hash of record content for change detection.

    Args:
        data: Dictionary, string, or any JSON-serializable data

    Returns:
        64-character hex string (SHA256 hash)
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        content = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, str):
        content = data
    else:
        content = json.dumps(data, default=str)

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_watermark(
    cursor,
    table_name: str,
    key_column: str = "symbol",
    key_value: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get the high watermark for incremental fetching.

    Args:
        cursor: Database cursor
        table_name: Watermark table name (e.g., 'form4_fetch_watermarks')
        key_column: Primary key column name
        key_value: Key value to look up

    Returns:
        Dictionary with watermark data or None
    """
    try:
        if key_value:
            cursor.execute(
                f"SELECT * FROM {table_name} WHERE {key_column} = %s",
                (key_value,),
            )
        else:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")

        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except Exception:
        return None


def update_watermark(
    cursor,
    table_name: str,
    key_column: str,
    key_value: str,
    watermark_data: Dict[str, Any],
) -> None:
    """Update the high watermark after successful fetch.

    Args:
        cursor: Database cursor
        table_name: Watermark table name
        key_column: Primary key column name
        key_value: Key value
        watermark_data: Dictionary of column -> value to update
    """
    # Build column list and values
    columns = list(watermark_data.keys())
    placeholders = ["%s"] * len(columns)
    updates = [f"{col} = EXCLUDED.{col}" for col in columns if col != key_column]

    sql = f"""
        INSERT INTO {table_name} ({key_column}, {', '.join(columns)})
        VALUES (%s, {', '.join(placeholders)})
        ON CONFLICT ({key_column}) DO UPDATE SET
            {', '.join(updates)},
            last_fetch_timestamp = NOW()
    """

    cursor.execute(sql, (key_value, *watermark_data.values()))


def check_record_changed(
    cursor,
    table_name: str,
    key_columns: Dict[str, Any],
    new_hash: str,
) -> Tuple[bool, Optional[int]]:
    """Check if a record has changed based on content hash.

    Args:
        cursor: Database cursor
        table_name: Table to check
        key_columns: Dictionary of column -> value for lookup
        new_hash: New content hash to compare

    Returns:
        Tuple of (has_changed: bool, existing_id: Optional[int])
    """
    where_clauses = [f"{col} = %s" for col in key_columns.keys()]
    where_sql = " AND ".join(where_clauses)

    cursor.execute(
        f"""
        SELECT id, source_hash
        FROM {table_name}
        WHERE {where_sql}
        """,
        tuple(key_columns.values()),
    )

    row = cursor.fetchone()
    if row is None:
        # New record
        return True, None

    existing_id, existing_hash = row
    if existing_hash == new_hash:
        # No change
        return False, existing_id

    # Changed
    return True, existing_id


def get_last_sequence_id(
    cursor,
    table_name: str,
    sequence_column: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> Optional[str]:
    """Get the last sequence ID for incremental fetching.

    This is useful for sources that provide sequence IDs (like SEC accession numbers).

    Args:
        cursor: Database cursor
        table_name: Table to query
        sequence_column: Column containing sequence ID (e.g., 'accession_number')
        filter_column: Optional column to filter by (e.g., 'symbol')
        filter_value: Value for filter column

    Returns:
        Last sequence ID or None
    """
    if filter_column and filter_value:
        cursor.execute(
            f"""
            SELECT {sequence_column}
            FROM {table_name}
            WHERE {filter_column} = %s
            ORDER BY {sequence_column} DESC
            LIMIT 1
            """,
            (filter_value,),
        )
    else:
        cursor.execute(
            f"""
            SELECT {sequence_column}
            FROM {table_name}
            ORDER BY {sequence_column} DESC
            LIMIT 1
            """
        )

    row = cursor.fetchone()
    return row[0] if row else None


def get_last_date(
    cursor,
    table_name: str,
    date_column: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> Optional[date]:
    """Get the last date for incremental fetching.

    Args:
        cursor: Database cursor
        table_name: Table to query
        date_column: Column containing date
        filter_column: Optional column to filter by
        filter_value: Value for filter column

    Returns:
        Last date or None
    """
    if filter_column and filter_value:
        cursor.execute(
            f"""
            SELECT MAX({date_column})
            FROM {table_name}
            WHERE {filter_column} = %s
            """,
            (filter_value,),
        )
    else:
        cursor.execute(
            f"""
            SELECT MAX({date_column})
            FROM {table_name}
            """
        )

    row = cursor.fetchone()
    return row[0] if row else None


def upsert_with_hash(
    cursor,
    table_name: str,
    key_columns: List[str],
    data: Dict[str, Any],
    content_for_hash: Union[Dict, str, None] = None,
) -> Tuple[str, bool]:
    """Insert or update a record with hash-based change detection.

    Args:
        cursor: Database cursor
        table_name: Table to insert into
        key_columns: List of columns that form the unique key
        data: Dictionary of column -> value to insert/update
        content_for_hash: Optional content to hash (defaults to data)

    Returns:
        Tuple of (action: 'inserted'|'updated'|'skipped', is_new: bool)
    """
    # Compute hash
    hash_content = content_for_hash if content_for_hash else data
    new_hash = compute_record_hash(hash_content)

    # Check if exists and changed
    key_lookup = {col: data[col] for col in key_columns}

    # Try to find existing record with hash
    where_clauses = [f"{col} = %s" for col in key_columns]
    where_sql = " AND ".join(where_clauses)

    cursor.execute(
        f"SELECT source_hash FROM {table_name} WHERE {where_sql}",
        tuple(key_lookup.values()),
    )
    existing = cursor.fetchone()

    if existing:
        if existing[0] == new_hash:
            # No change, skip
            return "skipped", False

        # Update existing
        update_cols = [
            col for col in data.keys()
            if col not in key_columns
        ]
        set_clauses = [f"{col} = %s" for col in update_cols]
        set_clauses.append("source_hash = %s")
        set_clauses.append("source_fetch_timestamp = NOW()")
        set_clauses.append("updated_at = NOW()")

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET {', '.join(set_clauses)}
            WHERE {where_sql}
            """,
            tuple([data[col] for col in update_cols] + [new_hash] + list(key_lookup.values())),
        )
        return "updated", False

    # Insert new
    all_columns = list(data.keys()) + ["source_hash", "source_fetch_timestamp"]
    placeholders = ["%s"] * (len(data) + 1) + ["NOW()"]

    cursor.execute(
        f"""
        INSERT INTO {table_name} ({', '.join(all_columns)})
        VALUES ({', '.join(placeholders)})
        """,
        tuple(list(data.values()) + [new_hash]),
    )
    return "inserted", True


class BaseCollector(ABC):
    """Abstract base class for data collectors.

    Provides common functionality for all scheduled collection jobs.

    Example:
        class TreasuryCollector(BaseCollector):
            def __init__(self):
                super().__init__("collect_treasury_data")

            def collect(self) -> CollectionMetrics:
                # Implement collection logic
                pass

        if __name__ == "__main__":
            collector = TreasuryCollector()
            collector.run()
    """

    def __init__(self, job_name: str):
        self.job_name = job_name
        self.logger = setup_logging(job_name)
        self.metrics = CollectionMetrics(
            job_name=job_name,
            start_time=datetime.now(),
        )

    @abstractmethod
    def collect(self) -> CollectionMetrics:
        """Implement the collection logic.

        Returns:
            CollectionMetrics with results
        """
        pass

    def run(self) -> int:
        """Run the collection job with locking and error handling.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            with LockFile(self.job_name):
                self.logger.info(f"Starting {self.job_name}...")

                self.metrics = self.collect()
                self.metrics.end_time = datetime.now()

                self.metrics.log_summary(self.logger)
                self._save_metrics()

                return 0 if self.metrics.success else 1

        except RuntimeError as e:
            # Lock file error - job already running
            self.logger.warning(str(e))
            return 0  # Not an error, just skip

        except Exception as e:
            self.logger.exception(f"Unhandled error in {self.job_name}: {e}")
            self.metrics.errors.append(str(e))
            self.metrics.end_time = datetime.now()
            self._save_metrics()
            return 1

    def _save_metrics(self) -> None:
        """Save metrics to database for monitoring."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO scheduler_job_runs
                    (job_name, start_time, end_time, duration_seconds,
                     records_processed, records_inserted, records_updated,
                     records_failed, records_skipped, success, error_count, errors,
                     high_watermark_date, high_watermark_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                self.metrics.job_name,
                self.metrics.start_time,
                self.metrics.end_time,
                self.metrics.duration_seconds,
                self.metrics.records_processed,
                self.metrics.records_inserted,
                self.metrics.records_updated,
                self.metrics.records_failed,
                self.metrics.records_skipped,
                self.metrics.success,
                len(self.metrics.errors),
                "\n".join(self.metrics.errors[:10]) if self.metrics.errors else None,
                self.metrics.high_watermark_date,
                self.metrics.high_watermark_value,
            ))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            self.logger.debug(f"Could not save metrics to database: {e}")
