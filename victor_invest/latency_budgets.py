"""Latency budgets for Victor investment workflow modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


LATENCY_BUDGET_PROFILES: Dict[str, Dict[str, float]] = {
    # End-to-end budgets for full CLI analysis runs.
    "production": {
        "quick": 15.0,
        "standard": 45.0,
        "comprehensive": 90.0,
    },
    # Deterministic CI gate budgets for stubbed workflow execution.
    "ci_stub": {
        "quick": 5.0,
        "standard": 8.0,
        "comprehensive": 12.0,
    },
}


@dataclass(frozen=True)
class LatencyEvaluation:
    mode: str
    profile: str
    elapsed_seconds: float
    budget_seconds: float

    @property
    def passed(self) -> bool:
        return self.elapsed_seconds <= self.budget_seconds

    @property
    def delta_seconds(self) -> float:
        return self.elapsed_seconds - self.budget_seconds


def get_latency_budget(mode: str, profile: str = "production") -> float:
    """Return the configured latency budget for a workflow mode/profile."""
    normalized_mode = mode.lower().strip()
    normalized_profile = profile.lower().strip()

    profile_budgets = LATENCY_BUDGET_PROFILES.get(normalized_profile)
    if profile_budgets is None:
        supported_profiles = ", ".join(sorted(LATENCY_BUDGET_PROFILES.keys()))
        raise ValueError(f"Unsupported profile '{profile}'. Supported profiles: {supported_profiles}")

    if normalized_mode not in profile_budgets:
        supported = ", ".join(sorted(profile_budgets.keys()))
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}")
    return profile_budgets[normalized_mode]


def evaluate_latency(mode: str, elapsed_seconds: float, profile: str = "production") -> LatencyEvaluation:
    """Evaluate an observed latency against the configured budget."""
    budget = get_latency_budget(mode, profile=profile)
    return LatencyEvaluation(
        mode=mode.lower().strip(),
        profile=profile.lower().strip(),
        elapsed_seconds=elapsed_seconds,
        budget_seconds=budget,
    )
