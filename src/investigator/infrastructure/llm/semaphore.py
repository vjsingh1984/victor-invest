"""
Dynamic LLM Semaphore for VRAM-aware concurrent LLM operations
Dynamically adjusts concurrency based on model VRAM requirements and available memory.

High-level flow:

    +-----------+      acquire()       +----------------------+
    |  Agent    | -------------------->| DynamicLLMSemaphore  |
    |  Request  |<---------------------|  (tracks VRAM usage) |
    +-----------+     release()        +----+-----------+-----+
                                               |           ^
                                               | updates   |
                                    +----------v-----------+------+
                                    |  active_tasks / queue       |
                                    |  • per-model reference      |
                                    |  • pessimistic VRAM totals  |
                                    +-----------------------------+

Concurrency is therefore limited by the actual GPU headroom, not an arbitrary task count.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from investigator.config import get_config
from investigator.infrastructure.llm.vram_calculator import estimate_model_vram_requirement

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size categories with VRAM requirements"""

    TINY = "tiny"  # <2GB (phi3:mini, tinyllama)
    SMALL = "small"  # 2-8GB (llama3.1:8b)
    MEDIUM = "medium"  # 8-16GB (phi3:14b, qwen2.5:14b)
    LARGE = "large"  # 16-32GB (qwen2.5:32b)
    XLARGE = "xlarge"  # 32GB+ (llama3.3:70b, phi4-reasoning)


class TaskType(Enum):
    """Task types with different resource requirements"""

    TECHNICAL = "technical"  # Medium complexity
    FUNDAMENTAL = "fundamental"  # High complexity
    SEC = "sec"  # High complexity
    SYNTHESIS = "synthesis"  # Medium complexity
    SUMMARY = "summary"  # Low complexity
    QUICK = "quick"  # Low complexity


class DynamicLLMSemaphore:
    """
    Dynamic semaphore that adjusts concurrency based on:
    - Available VRAM (48GB total)
    - Model VRAM requirements
    - Task complexity
    - Cache status (cached tasks use less resources)
    """

    _instance: Optional["DynamicLLMSemaphore"] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            # Auto-detect VRAM configuration
            from utils.system_info import SystemInfo

            system_info = SystemInfo.get_system_summary()

            self.total_vram_gb = system_info["gpu_memory_gb"]
            self.reserved_vram_gb = system_info["reserved_memory_gb"]
            self.available_vram_gb = system_info["available_memory_gb"]

            logger.info(
                f"Auto-detected GPU: {self.total_vram_gb:.1f}GB total, "
                f"{self.available_vram_gb:.1f}GB available for LLM"
            )

            # Model VRAM requirements (in GB)
            self.model_vram_requirements = {
                # Exact model names
                "phi3:mini": 2,
                "tinyllama:latest": 1,
                "llama3.1:8b": 6,
                "llama3.1:8b-instruct-q8_0": 8,
                "phi3:14b-medium-4k-instruct-q4_1": 12,
                "qwen2.5:14b": 14,
                "qwen2.5:32b-instruct-q4_K_M": 32,
                "llama3.3:70b-instruct-q4_k_m": 42,
                "phi4:latest": 35,
                "phi4-reasoning": 38,
                # Size categories for fallback
                ModelSize.TINY.value: 2,
                ModelSize.SMALL.value: 8,
                ModelSize.MEDIUM.value: 16,
                ModelSize.LARGE.value: 32,
                ModelSize.XLARGE.value: 40,
            }

            self.model_specs = {}
            try:
                cfg = get_config()
                raw_specs = getattr(cfg.ollama, "model_specs", {}) or {}
                # Convert to proper dict if needed
                if hasattr(raw_specs, "items"):
                    self.model_specs = dict(raw_specs.items()) if not isinstance(raw_specs, dict) else raw_specs
                    logger.info(f"Loaded {len(self.model_specs)} model specs: {list(self.model_specs.keys())}")
                    for name, spec in self.model_specs.items():
                        vram = getattr(spec, "weights_vram_gb", None) or getattr(spec, "memory_gb", None)
                        if vram:
                            self.model_vram_requirements[name] = float(vram)
                else:
                    logger.warning(f"model_specs is not dict-like: {type(raw_specs)}")
            except Exception as e:
                logger.warning(f"Unable to load model specs from config: {e}")
                self.model_specs = {}

            # Task complexity multipliers (minimal - VRAM is fixed by model, not task)
            # These only account for minor overhead from processing, not model size
            self.task_complexity = {
                TaskType.QUICK.value: 1.0,  # Simple queries
                TaskType.SUMMARY.value: 1.0,  # Summarization
                TaskType.SYNTHESIS.value: 1.0,  # Standard synthesis
                TaskType.TECHNICAL.value: 1.0,  # Technical analysis
                TaskType.FUNDAMENTAL.value: 1.0,  # Complex analysis
                TaskType.SEC.value: 1.0,  # SEC filing analysis
            }

            # Cached task resource reduction
            self.cache_reduction_factor = 0.6  # Cached tasks use 60% less resources

            # Current resource tracking
            self.active_tasks: Dict[str, Dict] = {}
            self.used_vram_gb = 0
            self.queue: List[Dict] = []

            # Model reuse tracking (for concurrent requests on same model)
            self.loaded_models: set[str] = set()  # Models currently loaded in VRAM
            self.active_tasks_per_model: Dict[str, int] = {}  # Task count per model

            # Statistics
            self._stats = {"total_requests": 0, "concurrent_peak": 0, "vram_peak": 0, "cache_hits": 0, "queue_waits": 0}

            self._initialized = True
            logger.info(f"Dynamic LLM Semaphore initialized - Available VRAM: {self.available_vram_gb}GB")

    def _estimate_kv_cache_gb(
        self, model: str, prompt_tokens: Optional[int], response_tokens: Optional[int], context_tokens: Optional[int]
    ) -> float:
        """
        Estimate KV cache memory in GB for a model

        Uses centralized VRAM calculator for consistency across all components.
        """
        spec = getattr(self, "model_specs", {}).get(model)
        if spec:
            try:
                # Convert spec object to dict for vram_calculator
                spec_dict = {
                    "kv_cache_mb_per_1k_tokens": getattr(spec, "kv_cache_mb_per_1k_tokens", 120.0),
                    "kv_cache_overhead_pct": getattr(spec, "kv_cache_overhead_pct", 0.15),
                    "context_window": getattr(spec, "context_window", 32768),
                    "weights_vram_gb": getattr(spec, "weights_vram_gb", 16.0),
                }
                result = estimate_model_vram_requirement(spec_dict, include_kv_cache=True)
                return result["kv_cache_gb"]
            except Exception as exc:
                logger.debug("KV cache estimate failed for %s: %s", model, exc)
        # Fallback: assume 32K context window with default params
        fallback_spec = {
            "kv_cache_mb_per_1k_tokens": 120.0,
            "kv_cache_overhead_pct": 0.15,
            "context_window": 32768,
            "weights_vram_gb": 16.0,
        }
        result = estimate_model_vram_requirement(fallback_spec, include_kv_cache=True)
        kv_cache_gb = result["kv_cache_gb"]
        return kv_cache_gb * 1.2

    def _get_model_vram_requirement(
        self,
        model: str,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
    ) -> float:
        """Get VRAM requirement for a model"""
        # Try exact match first
        if model in self.model_vram_requirements:
            weights = self.model_vram_requirements[model]
            return weights + self._estimate_kv_cache_gb(model, prompt_tokens, response_tokens, context_tokens)

        # Try pattern matching for common models
        model_lower = model.lower()
        if "phi3" in model_lower and "mini" in model_lower:
            return 2
        elif "phi3" in model_lower and "14b" in model_lower:
            return 12
        elif "phi4" in model_lower:
            return 35
        elif "llama3.1" in model_lower and "8b" in model_lower:
            return 8
        elif "qwen2.5" in model_lower and "14b" in model_lower:
            return 14
        elif "qwen2.5" in model_lower and "32b" in model_lower:
            return 32
        elif "70b" in model_lower:
            return 42

        # Default to medium model
        logger.warning(f"Unknown model {model}, assuming 16GB VRAM requirement")
        base = 16
        return base + self._estimate_kv_cache_gb(model, prompt_tokens, response_tokens, context_tokens)

    def _calculate_task_vram(
        self,
        model: str,
        task_type: str,
        is_cached: bool = False,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
    ) -> float:
        """
        Calculate actual VRAM requirement for a task

        Uses centralized VRAM calculator and accounts for model reuse:
        - First task: Full model (weights + KV cache)
        - Concurrent tasks: Only KV cache (weights already loaded)
        """
        spec = self.model_specs.get(model)
        if not spec:
            # Fallback if model spec not found
            logger.warning(
                f"Model spec not found for '{model}', using fallback. Available: {list(self.model_specs.keys())}"
            )
            base_vram = self._get_model_vram_requirement(model, prompt_tokens, response_tokens, context_tokens)
        else:
            # Use centralized calculator with model reuse logic
            model_already_loaded = model in self.loaded_models

            if model_already_loaded:
                # Model weights already in VRAM - only need KV cache for concurrent task
                vram_result = estimate_model_vram_requirement(
                    spec, include_kv_cache=True, prompt_tokens=prompt_tokens, response_tokens=response_tokens
                )
                base_vram = vram_result["kv_cache_gb"]  # Just KV cache
                logger.info(f"Model {model} already loaded, allocating KV cache only: {base_vram:.2f}GB")
            else:
                # First task for this model - need full model (weights + KV cache)
                vram_result = estimate_model_vram_requirement(
                    spec, include_kv_cache=True, prompt_tokens=prompt_tokens, response_tokens=response_tokens
                )
                base_vram = vram_result["total_gb"]  # Full model
                logger.info(f"First load of {model}, allocating full model: {base_vram:.2f}GB")

        # Apply complexity multiplier (currently 1.0 for all tasks)
        complexity_multiplier = self.task_complexity.get(task_type, 1.0)
        vram_with_complexity = base_vram * complexity_multiplier

        # Apply cache reduction if applicable
        if is_cached:
            vram_with_complexity *= self.cache_reduction_factor

        return vram_with_complexity

    def _can_accommodate_task(self, required_vram: float) -> bool:
        """Check if we can accommodate a new task"""
        return (self.used_vram_gb + required_vram) <= self.available_vram_gb

    async def acquire(
        self,
        model: str,
        task_type: str = "summary",
        is_cached: bool = False,
        task_id: str = None,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
    ) -> str:
        """
        Acquire LLM resources dynamically based on requirements

        Args:
            model: Model name to use
            task_type: Type of task (technical, fundamental, sec, synthesis, etc.)
            is_cached: Whether this task will use cached data
            task_id: Optional task identifier

        Returns:
            Resource allocation ID
        """
        required_vram = self._calculate_task_vram(
            model,
            task_type,
            is_cached,
            prompt_tokens,
            response_tokens,
            context_tokens,
        )
        allocation_id = f"{task_id or 'task'}_{datetime.now().timestamp()}"

        # Create task info
        task_info = {
            "allocation_id": allocation_id,
            "model": model,
            "task_type": task_type,
            "is_cached": is_cached,
            "required_vram": required_vram,
            "start_time": datetime.now(),
        }

        # Check if we can run immediately
        async with self._lock:
            if self._can_accommodate_task(required_vram):
                # Can run immediately
                self.active_tasks[allocation_id] = task_info
                self.used_vram_gb += required_vram
                self._stats["total_requests"] += 1

                # Track model loading
                self.loaded_models.add(model)
                self.active_tasks_per_model[model] = self.active_tasks_per_model.get(model, 0) + 1

                # Update peak stats
                current_concurrent = len(self.active_tasks)
                self._stats["concurrent_peak"] = max(self._stats["concurrent_peak"], current_concurrent)
                self._stats["vram_peak"] = max(self._stats["vram_peak"], self.used_vram_gb)

                if is_cached:
                    self._stats["cache_hits"] += 1

                logger.info(
                    f"✅ LLM allocated: {allocation_id} | Model: {model} | "
                    f"VRAM: {required_vram:.1f}GB | Used: {self.used_vram_gb:.1f}/{self.available_vram_gb}GB | "
                    f"Concurrent: {current_concurrent} | Cached: {is_cached}"
                )

                return allocation_id
            else:
                # Need to queue
                self.queue.append(task_info)
                self._stats["queue_waits"] += 1

                queue_position = len(self.queue)
                logger.info(
                    f"🔄 LLM queued: {allocation_id} | Position: {queue_position} | "
                    f"Required: {required_vram:.1f}GB | Available: {self.available_vram_gb - self.used_vram_gb:.1f}GB"
                )

        # Wait for resources to become available
        while True:
            await asyncio.sleep(0.1)  # Check every 100ms

            async with self._lock:
                if self._can_accommodate_task(required_vram):
                    # Remove from queue
                    if task_info in self.queue:
                        self.queue.remove(task_info)

                    # Allocate resources
                    self.active_tasks[allocation_id] = task_info
                    self.used_vram_gb += required_vram

                    # Track model loading
                    self.loaded_models.add(model)
                    self.active_tasks_per_model[model] = self.active_tasks_per_model.get(model, 0) + 1

                    wait_time = (datetime.now() - task_info["start_time"]).total_seconds()
                    current_concurrent = len(self.active_tasks)

                    # Update peak stats
                    self._stats["concurrent_peak"] = max(self._stats["concurrent_peak"], current_concurrent)
                    self._stats["vram_peak"] = max(self._stats["vram_peak"], self.used_vram_gb)

                    logger.info(
                        f"✅ LLM allocated after wait: {allocation_id} | Wait: {wait_time:.2f}s | "
                        f"VRAM: {required_vram:.1f}GB | Used: {self.used_vram_gb:.1f}/{self.available_vram_gb}GB | "
                        f"Concurrent: {current_concurrent}"
                    )

                    return allocation_id

    def release(self, allocation_id: str) -> None:
        """
        Release LLM resources

        Args:
            allocation_id: Resource allocation ID returned by acquire()
        """
        if allocation_id not in self.active_tasks:
            logger.warning(f"Attempting to release unknown allocation: {allocation_id}")
            return

        task_info = self.active_tasks[allocation_id]
        required_vram = task_info["required_vram"]
        model = task_info["model"]

        # Release resources
        del self.active_tasks[allocation_id]
        self.used_vram_gb -= required_vram

        # Track model task count
        if model in self.active_tasks_per_model:
            self.active_tasks_per_model[model] -= 1
            if self.active_tasks_per_model[model] <= 0:
                self.active_tasks_per_model.pop(model, None)
            # Note: Keep model in loaded_models set to reflect Ollama's actual behavior
            # Ollama keeps models loaded for ~5 minutes after last use for performance
            # Removing immediately would cause false "first load" logs and incorrect VRAM calculations

        # Ensure we don't go negative due to rounding
        self.used_vram_gb = max(0, self.used_vram_gb)

        execution_time = (datetime.now() - task_info["start_time"]).total_seconds()
        current_concurrent = len(self.active_tasks)

        logger.info(
            f"🔓 LLM released: {allocation_id} | Model: {model} | "
            f"Duration: {execution_time:.2f}s | VRAM freed: {required_vram:.1f}GB | "
            f"Remaining: {self.used_vram_gb:.1f}/{self.available_vram_gb}GB | "
            f"Queue: {len(self.queue)} | Concurrent: {current_concurrent}"
        )

    def get_stats(self) -> dict:
        """Get comprehensive resource statistics"""
        avg_vram = self._stats["vram_peak"] / self._stats["total_requests"] if self._stats["total_requests"] > 0 else 0

        return {
            "total_requests": self._stats["total_requests"],
            "current_concurrent": len(self.active_tasks),
            "peak_concurrent": self._stats["concurrent_peak"],
            "queue_size": len(self.queue),
            "queue_waits": self._stats["queue_waits"],
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["total_requests"] * 100
                if self._stats["total_requests"] > 0
                else 0
            ),
            "vram_used": self.used_vram_gb,
            "vram_available": self.available_vram_gb,
            "vram_utilization": (self.used_vram_gb / self.available_vram_gb * 100),
            "vram_peak": self._stats["vram_peak"],
            "avg_vram_per_task": avg_vram,
            "active_tasks": list(self.active_tasks.keys()),
        }

    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for optimizing resource usage"""
        suggestions = []
        stats = self.get_stats()

        if stats["queue_waits"] > stats["total_requests"] * 0.3:
            suggestions.append("High queue wait rate - consider using smaller models for simple tasks")

        if stats["cache_hit_rate"] < 30:
            suggestions.append("Low cache hit rate - enable more aggressive caching")

        if stats["vram_utilization"] > 90:
            suggestions.append("High VRAM utilization - consider running fewer concurrent tasks")
        elif stats["vram_utilization"] < 50:
            suggestions.append("Low VRAM utilization - can increase concurrency for better throughput")

        if stats["peak_concurrent"] == 1:
            suggestions.append("Only running 1 task at a time - check if smaller models can run concurrently")

        return suggestions


# Global semaphore instance
_dynamic_llm_semaphore: Optional[DynamicLLMSemaphore] = None


def get_dynamic_llm_semaphore() -> DynamicLLMSemaphore:
    """Get the global dynamic LLM semaphore instance"""
    global _dynamic_llm_semaphore
    if _dynamic_llm_semaphore is None:
        _dynamic_llm_semaphore = DynamicLLMSemaphore()
    return _dynamic_llm_semaphore


# Backward compatibility
def get_llm_semaphore() -> DynamicLLMSemaphore:
    """Get the LLM semaphore (backward compatibility)"""
    return get_dynamic_llm_semaphore()


# Dynamic decorator for LLM operations
def llm_resource_managed(model: str, task_type: str = "summary", is_cached: bool = False):
    """
    Decorator for dynamic VRAM-aware LLM resource management

    Args:
        model: Model name to use
        task_type: Type of task (technical, fundamental, sec, synthesis, etc.)
        is_cached: Whether this task will use cached data
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            semaphore = get_dynamic_llm_semaphore()
            task_id = f"{func.__name__}_{model}_{task_type}"

            # Acquire resources
            allocation_id = await semaphore.acquire(model, task_type, is_cached, task_id)

            try:
                logger.debug(f"Executing LLM task: {task_id} (allocation: {allocation_id})")
                result = await func(*args, **kwargs)
                logger.debug(f"Completed LLM task: {task_id}")
                return result
            finally:
                # Always release resources
                semaphore.release(allocation_id)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Context manager for dynamic resource management
class DynamicLLMContext:
    """Context manager for dynamic LLM resource allocation"""

    def __init__(
        self,
        model: str,
        task_type: str = "summary",
        is_cached: bool = False,
        task_id: str = None,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
    ):
        self.model = model
        self.task_type = task_type
        self.is_cached = is_cached
        self.task_id = task_id
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.context_tokens = context_tokens
        self.allocation_id = None
        self.semaphore = get_dynamic_llm_semaphore()

    async def __aenter__(self):
        self.allocation_id = await self.semaphore.acquire(
            self.model,
            self.task_type,
            self.is_cached,
            self.task_id,
            self.prompt_tokens,
            self.response_tokens,
            self.context_tokens,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.allocation_id:
            self.semaphore.release(self.allocation_id)
