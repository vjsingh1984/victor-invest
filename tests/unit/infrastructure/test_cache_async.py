"""
Tests for async cache operations (Issue #3 fix)

Verifies that cache I/O doesn't block the event loop and allows
concurrent agent execution even during slow disk/DB operations.
"""

import asyncio
import pytest
import time
from pathlib import Path
from typing import Dict

from investigator.infrastructure.cache.cache_manager import CacheManager
from investigator.infrastructure.cache.cache_types import CacheType


@pytest.fixture
def cache_manager(tmp_path):
    """Create cache manager with temporary directory"""
    # Mock config for testing
    class MockCacheControl:
        use_cache = True
        read_from_cache = True
        write_to_cache = True
        force_refresh = False
        force_refresh_symbols = None

        def is_cache_type_enabled(self, cache_type: str) -> bool:
            return True

        def get_cache_config(self, cache_type: str):
            class MockCacheConfig:
                enabled = True

                class disk:
                    enabled = True
                    priority = 20
                    settings = {"base_path": str(tmp_path / f"{cache_type}_cache")}

                class rdbms:
                    enabled = False
                    priority = 10

            return MockCacheConfig()

    class MockConfig:
        cache_control = MockCacheControl()

    manager = CacheManager(config=MockConfig())
    return manager


@pytest.mark.asyncio
async def test_async_cache_concurrent_writes(cache_manager):
    """
    Test that multiple concurrent cache writes don't block each other.

    Before fix: 10 writes × 200ms each = 2000ms (sequential blocking)
    After fix: 10 writes in parallel via thread pool = ~200-400ms
    """
    start = time.time()

    # Simulate 10 concurrent cache writes
    tasks = [
        cache_manager.set_async(
            CacheType.LLM_RESPONSE,
            {"symbol": f"TEST{i}", "llm_type": "test"},
            {"response": {"data": f"value_{i}" * 1000}}  # Proper structure with response key
        )
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Verify all writes succeeded
    assert all(results), "All cache writes should succeed"

    # Verify execution was concurrent (not sequential)
    # Should complete in <1s, not 2s+ (10 × 200ms)
    assert elapsed < 1.0, f"Cache writes blocked: {elapsed:.2f}s (expected <1s)"
    print(f"✅ Concurrent writes completed in {elapsed:.2f}s (target: <1s)")


@pytest.mark.asyncio
async def test_async_cache_concurrent_reads(cache_manager):
    """
    Test that multiple concurrent cache reads don't block each other.
    """
    # Pre-populate cache with proper LLM response structure
    for i in range(10):
        await cache_manager.set_async(
            CacheType.LLM_RESPONSE,
            {"symbol": f"TEST{i}", "llm_type": "test"},
            {"response": {"data": f"cached_value_{i}", "analysis": "test"}}
        )

    start = time.time()

    # Concurrent cache reads
    tasks = [
        cache_manager.get_async(
            CacheType.LLM_RESPONSE,
            {"symbol": f"TEST{i}", "llm_type": "test"}
        )
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Verify all reads succeeded
    assert all(results), "All cache reads should return data"
    assert len(results) == 10

    # Verify execution was concurrent
    assert elapsed < 0.5, f"Cache reads blocked: {elapsed:.2f}s (expected <0.5s)"
    print(f"✅ Concurrent reads completed in {elapsed:.2f}s (target: <0.5s)")


@pytest.mark.asyncio
async def test_async_cache_mixed_operations(cache_manager):
    """
    Test that reads and writes can happen concurrently without blocking.
    Simulates real-world agent behavior: some agents reading cache while
    others are writing.
    """
    # Pre-populate some data with proper structure
    for i in range(5):
        await cache_manager.set_async(
            CacheType.LLM_RESPONSE,
            {"symbol": f"EXISTING{i}", "llm_type": "test"},
            {"response": {"data": f"existing_{i}", "analysis": "test"}}
        )

    start = time.time()

    # Mix of reads and writes happening concurrently
    tasks = []

    # 5 reads of existing data
    for i in range(5):
        tasks.append(
            cache_manager.get_async(
                CacheType.LLM_RESPONSE,
                {"symbol": f"EXISTING{i}", "llm_type": "test"}
            )
        )

    # 5 writes of new data with proper structure
    for i in range(5):
        tasks.append(
            cache_manager.set_async(
                CacheType.LLM_RESPONSE,
                {"symbol": f"NEW{i}", "llm_type": "test"},
                {"response": {"data": f"new_value_{i}", "analysis": "test"}}
            )
        )

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Verify operations succeeded
    reads = results[:5]
    writes = results[5:]

    assert all(r is not None for r in reads), "All reads should return data"
    assert all(writes), "All writes should succeed"

    # Verify concurrent execution
    assert elapsed < 1.0, f"Mixed operations blocked: {elapsed:.2f}s (expected <1s)"
    print(f"✅ Mixed operations completed in {elapsed:.2f}s (target: <1s)")


@pytest.mark.asyncio
async def test_async_cache_doesnt_block_event_loop(cache_manager):
    """
    Test that cache I/O doesn't block other async operations.

    This simulates the real problem: cache writes blocking agent execution.
    """
    event_loop_blocked = False
    heartbeat_count = 0

    async def heartbeat_monitor():
        """Monitor that runs every 50ms - should not be blocked by cache I/O"""
        nonlocal heartbeat_count, event_loop_blocked
        try:
            for _ in range(20):  # 20 × 50ms = 1 second total
                await asyncio.sleep(0.05)
                heartbeat_count += 1
        except Exception:
            event_loop_blocked = True

    async def cache_operations():
        """Perform heavy cache operations"""
        for i in range(5):
            await cache_manager.set_async(
                CacheType.LLM_RESPONSE,
                {"symbol": f"HEAVY{i}", "llm_type": "test"},
                {"response": {"data": "x" * 10000, "analysis": "heavy"}}  # Large payload with proper structure
            )

    # Run both concurrently - heartbeat should not be starved
    await asyncio.gather(
        heartbeat_monitor(),
        cache_operations()
    )

    # Verify event loop was not blocked
    assert not event_loop_blocked, "Event loop was blocked during cache I/O"
    assert heartbeat_count >= 15, f"Heartbeat starved (count={heartbeat_count}, expected ≥15)"
    print(f"✅ Event loop remained responsive (heartbeat_count={heartbeat_count}/20)")


@pytest.mark.asyncio
async def test_cache_shutdown_cleanup(cache_manager):
    """Test that cache manager properly cleans up thread pool"""
    # Perform some operations
    await cache_manager.set_async(
        CacheType.LLM_RESPONSE,
        {"symbol": "TEST", "llm_type": "test"},
        {"response": {"data": "test_value", "analysis": "test"}}
    )

    # Shutdown should complete without errors
    cache_manager.shutdown()

    # Verify executor is shutdown
    assert cache_manager._executor._shutdown, "Executor should be shutdown"

    print("✅ Cache manager shutdown cleanly")


@pytest.mark.asyncio
async def test_backward_compatibility_sync_methods(cache_manager):
    """
    Test that sync methods still work for backward compatibility.
    Old code using sync methods should continue to work.
    """
    # Sync write with proper structure
    success = cache_manager.set(
        CacheType.LLM_RESPONSE,
        {"symbol": "SYNC_TEST", "llm_type": "test"},
        {"response": {"data": "sync_value", "analysis": "test"}}
    )
    assert success, "Sync write should succeed"

    # Sync read
    result = cache_manager.get(
        CacheType.LLM_RESPONSE,
        {"symbol": "SYNC_TEST", "llm_type": "test"}
    )
    assert result is not None, "Sync read should return data"
    assert result["response"]["data"] == "sync_value"

    print("✅ Backward compatibility maintained - sync methods work")


def test_cache_manager_initialization():
    """Test that cache manager initializes thread pool correctly"""
    manager = CacheManager()

    # Verify executor is created
    assert hasattr(manager, '_executor'), "Cache manager should have executor"
    assert manager._executor is not None, "Executor should be initialized"
    assert manager._executor._max_workers == 4, "Should have 4 worker threads"

    manager.shutdown()
    print("✅ Cache manager initialized with thread pool")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
