"""Shared payload builders for deterministic fundamental analyses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple


def build_deterministic_response(agent_id: str, label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic analysis response compatible with LLM-shaped contracts."""
    return {
        "response": payload,
        "prompt": "",
        "model_info": {
            "model": f"deterministic-{label}",
            "temperature": 0.0,
            "top_p": 0.0,
            "format": "json",
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "agent_id": agent_id,
            "analysis_type": label,
            "cache_type": "deterministic_analysis",
        },
    }


def build_deterministic_cache_record(
    *,
    symbol: str,
    agent_id: str,
    label: str,
    payload: Dict[str, Any],
    period: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build cache key/value pair for persisted deterministic analyses."""
    cache_key: Dict[str, Any] = {"symbol": symbol, "llm_type": label}
    if period:
        cache_key["period"] = period

    wrapped = {
        "response": payload,
        "metadata": {
            "cached_at": datetime.now().isoformat(),
            "agent_id": agent_id,
            "analysis_type": label,
            "period": period,
        },
    }

    return cache_key, wrapped
