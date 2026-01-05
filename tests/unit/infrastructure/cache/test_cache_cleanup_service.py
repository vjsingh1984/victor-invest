import asyncio
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from investigator.infrastructure.cache.cache_cleaner import CacheCleanupService
from investigator.infrastructure.cache.cache_types import CacheType


def _write_cache_file(path: Path, *, age: timedelta, metadata: Optional[Dict[str, str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "response": {"data": "payload"},
        "metadata": metadata or {},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    past = datetime.now() - age
    timestamp = past.timestamp()
    import os

    os.utime(path, (timestamp, timestamp))
    return path


def test_cleanup_service_start_stop(monkeypatch):
    async def runner():
        service = CacheCleanupService(cache_manager=Mock(), cleanup_interval_seconds=0.01)
        monkeypatch.setattr(service, "_run_cleanup", AsyncMock())

        await service.start()
        assert service.running is True

        await asyncio.sleep(0)
        await service.stop()
        assert service.running is False
        service._run_cleanup.assert_called()  # type: ignore[attr-defined]

    asyncio.run(runner())


def test_cleanup_removes_expired_files(tmp_path: Path):
    async def runner():
        service = CacheCleanupService(cache_manager=Mock())
        cache_dir = tmp_path / "llm_cache" / "AAPL"

        expired_file = _write_cache_file(
            cache_dir / "expired.json.gz",
            age=timedelta(days=8),
            metadata={"cached_at": (datetime.now() - timedelta(days=8)).isoformat()},
        )
        fresh_file = _write_cache_file(
            cache_dir / "fresh.json.gz",
            age=timedelta(days=1),
            metadata={"cached_at": (datetime.now() - timedelta(days=1)).isoformat()},
        )

        assert expired_file.exists() and fresh_file.exists()

        await service._cleanup_directory(str(tmp_path / "llm_cache"), timedelta(days=7), CacheType.LLM_RESPONSE)

        assert not expired_file.exists()
        assert fresh_file.exists()

    asyncio.run(runner())


def test_cleanup_uses_metadata_expiry(tmp_path: Path):
    service = CacheCleanupService(cache_manager=Mock())
    cache_file = _write_cache_file(
        tmp_path / "sec_cache" / "responses" / "file.json.gz",
        age=timedelta(hours=1),
        metadata={"expires_at": (datetime.now() - timedelta(minutes=5)).isoformat()},
    )

    expires_at = service._get_expires_at_from_file(cache_file)
    assert expires_at is not None
    assert expires_at < datetime.now()


def test_cleanup_stats_and_force_cleanup(monkeypatch):
    async def runner():
        service = CacheCleanupService(cache_manager=Mock())
        monkeypatch.setattr(service, "_run_cleanup", AsyncMock())

        initial = service.get_stats()
        assert initial["total_runs"] == 0

        await service.start()
        await asyncio.sleep(0)
        await service.stop()

        stats = service.get_stats()
        assert stats["is_running"] is False
        assert stats["total_runs"] >= 0

        await service.force_cleanup()
        assert service._run_cleanup.await_count >= 2  # type: ignore[attr-defined]

    asyncio.run(runner())
