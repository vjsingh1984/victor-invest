"""
Legacy import shim for CacheManager.

The canonical implementation lives in
``src.investigator.infrastructure.cache.cache_manager``; this shim keeps older
imports working while code migrates to the new module path.
"""

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_path = Path(__file__).resolve().parents[2] / "src"
    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()

from investigator.infrastructure.cache.cache_manager import CacheManager  # noqa: E402

__all__ = ["CacheManager"]
