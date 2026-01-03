#!/usr/bin/env python3
"""
Test script to verify semaphore model reuse tracking
Simulates two concurrent tasks using the same model
"""
import asyncio
import sys
from core.llm_semaphore import DynamicLLMSemaphore, TaskType


async def simulate_task(task_name: str, model: str, delay: float = 1.0):
    """Simulate a task that uses the semaphore"""
    semaphore = DynamicLLMSemaphore()

    print(f"\n[{task_name}] Requesting semaphore for {model}...")
    task_id = await semaphore.acquire(
        model=model, task_type=TaskType.TECHNICAL.value, prompt_tokens=5000, response_tokens=2000
    )

    try:
        print(f"[{task_name}] Acquired semaphore with task_id={task_id}")
        print(f"[{task_name}] Loaded models: {semaphore.loaded_models}")
        print(f"[{task_name}] Active tasks per model: {semaphore.active_tasks_per_model}")
        print(f"[{task_name}] Used VRAM: {semaphore.used_vram_gb:.2f}GB / {semaphore.available_vram_gb:.2f}GB")
        print(f"[{task_name}] Concurrent tasks: {len(semaphore.active_tasks)}")

        # Simulate work
        await asyncio.sleep(delay)

        print(f"[{task_name}] Releasing semaphore...")
    finally:
        semaphore.release(task_id)

    print(f"[{task_name}] Done!")


async def test_sequential():
    """Test sequential execution (baseline)"""
    print("\n" + "=" * 60)
    print("TEST 1: Sequential execution (one task at a time)")
    print("=" * 60)

    await simulate_task("TASK1", "qwen3:30b", delay=0.5)
    await simulate_task("TASK2", "qwen3:30b", delay=0.5)


async def test_concurrent():
    """Test concurrent execution (should show model reuse)"""
    print("\n" + "=" * 60)
    print("TEST 2: Concurrent execution (all 3 tasks run together)")
    print("=" * 60)
    print("Expected: TASK1 allocates full model (18.2GB)")
    print("Expected: TASK2 allocates KV cache only (2.0GB)")
    print("Expected: TASK3 allocates KV cache only (2.0GB)")
    print("Expected: Total: ~22.2GB, Concurrent tasks: 3")
    print()

    # Start all 3 tasks concurrently
    tasks = [
        simulate_task("TASK1", "qwen3:30b", delay=2.0),
        simulate_task("TASK2", "qwen3:30b", delay=2.0),
        simulate_task("TASK3", "qwen3:30b", delay=2.0),
    ]

    await asyncio.gather(*tasks)


async def main():
    """Run all tests"""
    print("Testing Dynamic LLM Semaphore Model Reuse Tracking")
    print("=" * 60)

    # Test 1: Sequential (baseline)
    await test_sequential()

    # Test 2: Concurrent (model reuse)
    await test_concurrent()

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
