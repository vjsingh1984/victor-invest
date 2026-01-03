"""
Centralized VRAM calculation for LLM models

This module provides a single source of truth for VRAM requirements,
used by both resource_aware_pool and llm_semaphore to ensure consistency.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Maximum context window to use for VRAM calculations (tokens)
# Even models with 256K+ context get capped here to allow concurrent execution
MAX_CONTEXT_FOR_VRAM_CALC = 32768


def estimate_model_vram_requirement(
    model_spec: Any,  # Dict or object
    include_kv_cache: bool = True,
    prompt_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate total VRAM requirement for a model

    Args:
        model_spec: Model specification (dict or object) with attributes:
            - weights_vram_gb: Base model weights size
            - context_window: Maximum context window size
            - kv_cache_mb_per_1k_tokens: KV cache per 1K tokens
            - kv_cache_overhead_pct: Overhead percentage (default 0.15)
        include_kv_cache: Whether to include KV cache in calculation
        prompt_tokens: Actual prompt tokens (if None, uses capped context_window)
        response_tokens: Expected response tokens (if None, uses 0)

    Returns:
        Dictionary with:
            - weights_gb: Model weights VRAM
            - kv_cache_gb: KV cache VRAM (with overhead)
            - total_gb: Total VRAM requirement
            - context_tokens_used: Actual tokens used for calculation
    """

    # Extract model weights (supports both dict and object)
    weights_gb = float(get_spec_value(model_spec, "weights_vram_gb", get_spec_value(model_spec, "memory_gb", 16.0)))

    if not include_kv_cache:
        return {"weights_gb": weights_gb, "kv_cache_gb": 0.0, "total_gb": weights_gb, "context_tokens_used": 0}

    # Extract KV cache parameters (supports both dict and object)
    # EMPIRICAL VALUE: Based on qwen3:30b actual measurements from ollama ps:
    # 4K context = 20GB, 16K context = 21GB → 12K token difference
    # Conservative estimate: 2GB for 12K tokens = 170 MB/1K (was 120 MB - too high, 85 MB - too low)
    kv_cache_mb_per_1k = float(get_spec_value(model_spec, "kv_cache_mb_per_1k_tokens", 170.0))
    kv_cache_overhead_pct = float(get_spec_value(model_spec, "kv_cache_overhead_pct", 0.15))
    context_window = int(get_spec_value(model_spec, "context_window", 32768))

    # Determine total tokens for KV cache calculation
    if prompt_tokens is not None and response_tokens is not None:
        # Use actual tokens if provided
        total_tokens = max(0, prompt_tokens + response_tokens)
    else:
        # Reserve for reasonable max context, capped to prevent excessive VRAM
        # Models like qwen3:30b have 262K context but we cap at 32K for practical VRAM usage
        total_tokens = min(context_window, MAX_CONTEXT_FOR_VRAM_CALC)

    # Calculate KV cache size
    kv_cache_base_gb = (total_tokens / 1000.0) * kv_cache_mb_per_1k / 1024.0
    kv_cache_with_overhead_gb = kv_cache_base_gb * (1.0 + kv_cache_overhead_pct)

    total_gb = weights_gb + kv_cache_with_overhead_gb

    logger.debug(
        f"VRAM estimate: weights={weights_gb:.2f}GB, "
        f"kv_cache={kv_cache_with_overhead_gb:.2f}GB "
        f"(tokens={total_tokens}, overhead={kv_cache_overhead_pct:.0%}), "
        f"total={total_gb:.2f}GB"
    )

    return {
        "weights_gb": weights_gb,
        "kv_cache_gb": kv_cache_with_overhead_gb,
        "total_gb": total_gb,
        "context_tokens_used": total_tokens,
    }


def get_spec_value(spec: Any, key: str, default: Any) -> Any:
    """
    Safely extract value from spec (supports both dict and object)

    Args:
        spec: Model spec (dict or object)
        key: Key/attribute name
        default: Default value if not found

    Returns:
        Value from spec or default
    """
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def estimate_kv_cache_only(model_spec: Dict[str, Any], prompt_tokens: int = 0, response_tokens: int = 0) -> float:
    """
    Estimate only KV cache VRAM (without model weights)

    Args:
        model_spec: Model specification dictionary
        prompt_tokens: Number of prompt tokens
        response_tokens: Number of response tokens

    Returns:
        KV cache VRAM in GB (with overhead)
    """
    result = estimate_model_vram_requirement(
        model_spec, include_kv_cache=True, prompt_tokens=prompt_tokens, response_tokens=response_tokens
    )
    return result["kv_cache_gb"]
