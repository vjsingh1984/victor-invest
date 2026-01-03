#!/usr/bin/env python3
"""
Cache Cleanup Utility - Background TTL Enforcement

Automatically removes expired cache entries based on TTL settings.
Phase 4.1 of Implementation Plan

Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0
"""

import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .cache_types import CacheType

logger = logging.getLogger(__name__)


class CacheCleanupService:
    """Background service for cleaning up expired cache entries"""

    def __init__(self, cache_manager, cleanup_interval_seconds: int = 3600):
        """
        Initialize cache cleanup service

        Args:
            cache_manager: CacheManager instance
            cleanup_interval_seconds: How often to run cleanup (default: 1 hour)
        """
        self.cache_manager = cache_manager
        self.cleanup_interval = cleanup_interval_seconds
        self.running = False
        self._cleanup_task = None
        self.stats = {
            "total_runs": 0,
            "total_files_scanned": 0,
            "total_files_removed": 0,
            "total_bytes_freed": 0,
            "last_run": None,
            "errors": 0,
        }

    async def start(self):
        """Start background cleanup task"""
        if self.running:
            logger.warning("Cache cleanup service already running")
            return

        self.running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Cache cleanup service started " f"(interval: {self.cleanup_interval}s)")

    async def stop(self):
        """Stop background cleanup task"""
        if not self.running:
            return

        self.running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Cache cleanup service stopped")

    async def _cleanup_loop(self):
        """Main cleanup loop running in background"""
        while self.running:
            try:
                await self._run_cleanup()
                self.stats["total_runs"] += 1
                self.stats["last_run"] = datetime.now()

                logger.info(
                    f"Cache cleanup complete. "
                    f"Scanned: {self.stats['total_files_scanned']}, "
                    f"Removed: {self.stats['total_files_removed']}, "
                    f"Freed: {self.stats['total_bytes_freed'] / (1024*1024):.2f} MB"
                )

            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                self.stats["errors"] += 1

            # Wait before next cleanup
            await asyncio.sleep(self.cleanup_interval)

    async def _run_cleanup(self):
        """Run cleanup for all cache types"""
        # Get TTL settings for each cache type
        from .cache_manager import CacheManager

        cache_dirs = {
            CacheType.LLM_RESPONSE: ("data/llm_cache", timedelta(hours=6)),
            CacheType.TECHNICAL_DATA: ("data/technical_cache", timedelta(days=7)),
            CacheType.COMPANY_FACTS: ("data/sec_cache/facts/processed", timedelta(days=90)),
            CacheType.SEC_RESPONSE: ("data/sec_cache/responses", timedelta(hours=6)),
            CacheType.SUBMISSION_DATA: ("data/sec_cache/submissions", timedelta(days=90)),
            CacheType.QUARTERLY_METRICS: ("data/sec_cache/quarterlymetrics", timedelta(hours=24)),
        }

        for cache_type, (cache_dir, ttl) in cache_dirs.items():
            await self._cleanup_directory(cache_dir, ttl, cache_type)

    async def _cleanup_directory(self, cache_dir: str, ttl: timedelta, cache_type: CacheType):
        """
        Cleanup expired files in a cache directory

        Args:
            cache_dir: Directory path to clean
            ttl: Time-to-live for cache files
            cache_type: Type of cache being cleaned
        """
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return

        cutoff_time = datetime.now() - ttl
        files_removed = 0
        bytes_freed = 0

        # Recursively scan all .json.gz files
        for cache_file in cache_path.rglob("*.json.gz"):
            self.stats["total_files_scanned"] += 1

            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)

                # Try to get expires_at from metadata if available
                expires_at = self._get_expires_at_from_file(cache_file)

                # Use metadata expiry if available, otherwise use mtime + TTL
                if expires_at:
                    is_expired = datetime.now() > expires_at
                else:
                    is_expired = mtime < cutoff_time

                if is_expired:
                    # Get file size before deletion
                    file_size = cache_file.stat().st_size

                    # Delete expired file
                    cache_file.unlink()

                    files_removed += 1
                    bytes_freed += file_size
                    self.stats["total_files_removed"] += 1
                    self.stats["total_bytes_freed"] += file_size

                    logger.debug(
                        f"Removed expired cache file: {cache_file.name} " f"(mtime: {mtime.strftime('%Y-%m-%d %H:%M')})"
                    )

            except Exception as e:
                logger.warning(f"Failed to process {cache_file}: {e}")

        if files_removed > 0:
            logger.info(
                f"Cleaned {cache_type.value}: "
                f"removed {files_removed} files, "
                f"freed {bytes_freed / (1024*1024):.2f} MB"
            )

    def _get_expires_at_from_file(self, cache_file: Path) -> Optional[datetime]:
        """
        Extract expires_at timestamp from cache file metadata

        Args:
            cache_file: Path to cache file

        Returns:
            Expiry datetime if found, None otherwise
        """
        try:
            with gzip.open(cache_file, "rt") as f:
                data = json.load(f)

                # Check for metadata.expires_at
                if isinstance(data, dict):
                    metadata = data.get("metadata", {})
                    if metadata and isinstance(metadata, dict):
                        expires_at_str = metadata.get("expires_at")
                        if expires_at_str:
                            return datetime.fromisoformat(expires_at_str)

        except Exception:
            # Silently fail - will use mtime fallback
            pass

        return None

    def get_stats(self) -> Dict:
        """Get cleanup statistics"""
        return {
            **self.stats,
            "is_running": self.running,
            "next_run_in": self.cleanup_interval if self.running else None,
        }

    async def force_cleanup(self):
        """Force immediate cleanup (useful for testing)"""
        logger.info("Running forced cache cleanup...")
        await self._run_cleanup()
        logger.info("Forced cleanup complete")


# Convenience function for standalone usage
async def cleanup_expired_caches(cache_manager, run_once: bool = False):
    """
    Run cache cleanup (standalone or continuous)

    Args:
        cache_manager: CacheManager instance
        run_once: If True, run once and exit; if False, run continuously

    Example:
        # Run once
        await cleanup_expired_caches(cache_manager, run_once=True)

        # Run continuously
        asyncio.create_task(cleanup_expired_caches(cache_manager))
    """
    cleanup_service = CacheCleanupService(cache_manager)

    if run_once:
        await cleanup_service._run_cleanup()
    else:
        await cleanup_service.start()
        # Will run until cancelled
