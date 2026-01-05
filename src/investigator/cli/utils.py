"""
Shared CLI utilities and decorators for InvestiGator
"""

import asyncio
import functools
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import click
import yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure application logging with production-friendly defaults.

    Use INVESTIGATOR_LOG_PROFILE=debug for verbose tracing.
    """
    profile = os.getenv("INVESTIGATOR_LOG_PROFILE", "prod").strip().lower()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Promote high-volume modules to WARNING in prod
    if profile != "debug" and numeric_level >= logging.INFO:
        noisy_loggers = [
            "investigator.infrastructure.llm.pool",
            "investigator.infrastructure.llm.semaphore",
            "investigator.infrastructure.cache.cache_manager",
            "investigator.infrastructure.cache.cache_cleaner",
            "agent.synth_agent_1",
            "agent.fund_agent_1",
            "utils.market_data_fetcher",
            "utils.sec_data_strategy",
        ]
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_file)

    if not config_path.exists():
        default_config = {
            "ollama": {"base_url": "http://localhost:11434", "timeout": 300, "max_retries": 3},
            "cache": {"redis_url": "redis://localhost:6379", "file_cache_path": "data/cache", "ttl_default": 3600},
            "orchestrator": {"max_concurrent_analyses": 5, "max_concurrent_agents": 10},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "monitoring": {"export_interval": 60, "metrics_port": 9090},
        }

        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def async_command(f: Callable) -> Callable:
    """Decorator to run async functions in Click commands"""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def validate_symbols(ctx, param, value):
    """Validate and normalize stock symbols"""
    if not value:
        return value
    if isinstance(value, str):
        return [s.strip().upper() for s in value.split(",") if s.strip()]
    return [s.upper() for s in value]


def validate_date(ctx, param, value):
    """Validate date format YYYY-MM-DD"""
    if not value:
        return value
    from datetime import datetime

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Invalid date format: {value}. Use YYYY-MM-DD")


class MutuallyExclusiveOption(click.Option):
    """Click option that enforces mutual exclusivity with other options"""

    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        help_text = kwargs.get("help", "")
        if self.mutually_exclusive:
            ex_str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = help_text + (f" (mutually exclusive with {ex_str})")
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise click.UsageError(
                f"Option --{self.name.replace('_', '-')} is mutually exclusive with "
                f"{', '.join(['--' + m.replace('_', '-') for m in self.mutually_exclusive if m in opts])}"
            )
        return super().handle_parse_result(ctx, opts, args)


def print_table(headers: list, rows: list, widths: Optional[list] = None):
    """Print a formatted table to stdout"""
    if not widths:
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows) if rows else 0) + 2 for i, h in enumerate(headers)]

    # Header
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, widths))
    click.echo(header_line)
    click.echo("-" * len(header_line))

    # Rows
    for row in rows:
        row_line = "".join(str(c).ljust(w) for c, w in zip(row, widths))
        click.echo(row_line)


def format_currency(value: Optional[float], symbol: str = "$") -> str:
    """Format a value as currency"""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{symbol}{value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value/1_000:.2f}K"
    return f"{symbol}{value:.2f}"


def format_percent(value: Optional[float]) -> str:
    """Format a value as percentage"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def status_icon(value: bool) -> str:
    """Return colored status icon"""
    return click.style("OK", fg="green") if value else click.style("FAIL", fg="red")


def warn_icon() -> str:
    """Return warning icon"""
    return click.style("!", fg="yellow")


def error_exit(message: str, code: int = 1):
    """Print error message and exit"""
    click.echo(click.style(f"Error: {message}", fg="red"), err=True)
    sys.exit(code)


def require_database():
    """Decorator that ensures database is available"""

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                from investigator.infrastructure.database.db import get_engine

                engine = get_engine()
                # Quick test connection
                with engine.connect():
                    pass
            except Exception as e:
                error_exit(f"Database connection failed: {e}")
            return f(*args, **kwargs)

        return wrapper

    return decorator


def require_ollama():
    """Decorator that ensures Ollama is available"""

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            import requests

            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                if resp.status_code != 200:
                    raise Exception("Ollama not responding")
            except Exception as e:
                error_exit(f"Ollama connection failed: {e}. Is Ollama running?")
            return f(*args, **kwargs)

        return wrapper

    return decorator
