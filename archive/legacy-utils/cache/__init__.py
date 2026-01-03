"""
Compatibility layer for legacy cache imports.

New implementations live in ``src/investigator/infrastructure/cache`` but a large
portion of the codebase – including historic tests – still imports from
``utils.cache``. This module re-exports the modern classes to avoid breaking
those callers during the migration.
"""

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    """Add the project ``src`` directory to ``sys.path`` when running in-place."""
    src_path = Path(__file__).resolve().parents[2] / "src"
    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()

from investigator.infrastructure.cache.cache_manager import CacheManager  # noqa: E402,F401
from investigator.infrastructure.cache.cache_types import CacheType  # noqa: E402,F401

__all__ = ["CacheManager", "CacheType"]
