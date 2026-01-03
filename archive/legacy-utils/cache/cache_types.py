"""
Legacy import shim for CacheType enumeration.
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

from investigator.infrastructure.cache.cache_types import CacheType  # noqa: E402

__all__ = ["CacheType"]
