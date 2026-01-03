"""
Common contracts for valuation models.

Each concrete valuation model should inherit from ``BaseValuationModel`` and
return ``ValuationModelResult`` instances so the orchestrator can blend
outputs consistently.

Migration Date: 2025-11-14
Phase: Phase 4 (Valuation Framework Migration)
Canonical Location: investigator.domain.services.valuation.models.base
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from investigator.domain.services.valuation.models.company_profile import CompanyProfile


@dataclass(slots=True)
class ModelDiagnostics:
    """Diagnostics that describe how reliable a model output is."""

    data_quality_score: float = 0.0
    fit_score: float = 0.0
    calibration_score: float = 0.0
    flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ValuationModelResult:
    """Normalized payload emitted by valuation models."""

    model_name: str
    fair_value: Optional[float]
    confidence_score: float
    weight: float = 0.0
    methodology: str = ""
    assumptions: Dict[str, Any] = field(default_factory=dict)
    diagnostics: ModelDiagnostics = field(default_factory=ModelDiagnostics)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelNotApplicable:
    """Return type for models that refuse to produce a valuation."""

    model_name: str
    reason: str
    diagnostics: ModelDiagnostics = field(default_factory=ModelDiagnostics)


ValuationOutput = Union[ValuationModelResult, ModelNotApplicable]


class BaseValuationModel(ABC):
    """
    Base class for all valuation models.

    Child classes must implement ``calculate`` and ``estimate_confidence``.
    ``explain`` provides a standardized structure that can be enriched as
    needed.
    """

    model_name: str
    methodology: str

    def __init__(self, company_profile: CompanyProfile):
        self.company_profile = company_profile

    @abstractmethod
    def calculate(self, **kwargs: Any) -> ValuationOutput:
        """Execute the valuation model and return a normalized result."""

    @abstractmethod
    def estimate_confidence(self, raw_output: Dict[str, Any]) -> float:
        """
        Best-effort confidence estimate for the raw model output.

        Later phases will replace these heuristics with the dedicated confidence
        scorer, but model classes should still surface a baseline score so the
        orchestrator can reason about missing data.
        """

    def explain(self, result: ValuationOutput) -> Dict[str, Any]:
        """
        Provide a generic explanation payload for downstream consumers.

        Concrete models may override this to include additional details, but the
        structure should remain stable so audit logging stays consistent.
        """
        if isinstance(result, ModelNotApplicable):
            return {
                "model": self.model_name,
                "methodology": getattr(self, "methodology", self.model_name),
                "applicable": False,
                "reason": result.reason,
                "diagnostics": asdict(result.diagnostics),
            }

        return {
            "model": result.model_name,
            "methodology": result.methodology or getattr(self, "methodology", result.model_name),
            "applicable": True,
            "fair_value": result.fair_value,
            "confidence_score": result.confidence_score,
            "assumptions": result.assumptions,
            "diagnostics": asdict(result.diagnostics),
            "metadata": result.metadata,
        }
