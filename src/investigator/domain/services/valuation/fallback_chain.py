"""
Fallback Chain - Graceful degradation for valuation models.

Provides:
1. Fallback model chains when primary models fail
2. Confidence adjustments for fallback usage
3. Audit trail of which fallbacks were used
4. Configurable fallback strategies

Problem being solved:
- Valuation models can fail due to missing data
- Some companies don't have data for certain models (e.g., no dividends for GGM)
- No systematic fallback when a model fails

Solution:
- Define fallback chains for each model type
- Apply confidence penalty when using fallbacks
- Track which fallbacks were used for transparency
- Allow graceful degradation to simpler models

Usage:
    from investigator.domain.services.valuation.fallback_chain import FallbackChain

    chain = FallbackChain()

    # Get fallback for a failed model
    fallback_model, penalty = chain.get_fallback('ggm')  # Returns ('dcf', 0.90)

    # Get full fallback chain for a model
    chain_list = chain.get_chain('dcf')  # ['dcf', 'pe', 'ps']
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reason for using fallback."""

    DATA_MISSING = "data_missing"
    CALCULATION_ERROR = "calculation_error"
    INVALID_OUTPUT = "invalid_output"
    MODEL_NOT_APPLICABLE = "model_not_applicable"
    TIMEOUT = "timeout"


@dataclass
class FallbackResult:
    """Result of attempting a fallback."""

    original_model: str
    fallback_model: Optional[str]
    reason: FallbackReason
    confidence_penalty: float  # Multiplier (e.g., 0.90 = 10% penalty)
    fallback_level: int  # 0 = primary, 1 = first fallback, etc.
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Get summary of fallback."""
        if self.fallback_model:
            return (
                f"{self.original_model} → {self.fallback_model} "
                f"(level {self.fallback_level}, {self.confidence_penalty:.0%} confidence)"
            )
        else:
            return f"{self.original_model} failed, no fallback available"


@dataclass
class FallbackChainConfig:
    """Configuration for a model's fallback chain."""

    model: str
    fallbacks: List[str]
    penalties: List[float]  # Confidence penalty for each fallback level

    def get_fallback(self, level: int) -> Optional[Tuple[str, float]]:
        """Get fallback at a specific level."""
        if level < len(self.fallbacks):
            return (self.fallbacks[level], self.penalties[level])
        return None


class FallbackChain:
    """
    Manages fallback chains for valuation models.

    When a valuation model fails, this class provides the next best
    alternative along with an appropriate confidence penalty.

    Fallback Chain Examples:
    - DCF fails → try P/E → try P/S
    - GGM fails → try DCF → try P/E
    - P/E fails → try P/S → try EV/EBITDA

    Confidence Penalties:
    - First fallback: 90% (10% penalty)
    - Second fallback: 80% (20% penalty)
    - Third fallback: 70% (30% penalty)

    Example:
        chain = FallbackChain()

        # Model failed, get fallback
        fallback, penalty = chain.get_fallback('ggm')
        if fallback:
            print(f"Use {fallback} instead with {penalty:.0%} confidence")

        # Execute with fallback chain
        result = chain.execute_with_fallbacks(
            model_type='dcf',
            executor_func=run_valuation,
            financials=financials
        )
    """

    # Default fallback chains
    # Format: model -> [fallback1, fallback2, ...]
    DEFAULT_CHAINS = {
        "dcf": ["pe", "ps", "ev_ebitda"],
        "damodaran_dcf": ["dcf", "pe", "ps"],
        "pe": ["ps", "ev_ebitda", "pb"],
        "ps": ["ev_revenue", "ev_ebitda"],
        "ggm": ["dcf", "pe", "ps"],
        "ev_ebitda": ["pe", "ps"],
        "pb": ["ps", "ev_revenue"],
        "ev_revenue": ["ps"],
        "rule_of_40": ["ps", "ev_revenue"],
        "saas": ["ps", "rule_of_40", "ev_revenue"],
    }

    # Default penalties by fallback level
    DEFAULT_PENALTIES = [0.90, 0.80, 0.70, 0.60]

    def __init__(self, chains: Optional[Dict[str, List[str]]] = None, penalties: Optional[List[float]] = None):
        """
        Initialize fallback chain.

        Args:
            chains: Custom fallback chains (overrides defaults)
            penalties: Custom penalties by fallback level
        """
        self.chains = {**self.DEFAULT_CHAINS}
        self.penalties = penalties or self.DEFAULT_PENALTIES

        if chains:
            self.chains.update(chains)

        # Build chain configs
        self._configs: Dict[str, FallbackChainConfig] = {}
        for model, fallbacks in self.chains.items():
            self._configs[model] = FallbackChainConfig(
                model=model, fallbacks=fallbacks, penalties=self.penalties[: len(fallbacks)]
            )

    def get_fallback(self, model: str, failed_models: Optional[List[str]] = None) -> Optional[Tuple[str, float]]:
        """
        Get the next fallback model for a failed model.

        Args:
            model: Model that failed
            failed_models: List of already-failed models to skip

        Returns:
            Tuple of (fallback_model, confidence_penalty) or None
        """
        failed_models = failed_models or []
        config = self._configs.get(model)

        if not config:
            logger.debug(f"No fallback chain defined for {model}")
            return None

        # Find first non-failed fallback
        for level, fallback in enumerate(config.fallbacks):
            if fallback not in failed_models:
                penalty = config.penalties[level] if level < len(config.penalties) else 0.50
                logger.info(f"Fallback: {model} → {fallback} (level {level + 1}, {penalty:.0%})")
                return (fallback, penalty)

        logger.warning(f"All fallbacks exhausted for {model}")
        return None

    def get_chain(self, model: str) -> List[str]:
        """
        Get the full fallback chain for a model.

        Args:
            model: Model to get chain for

        Returns:
            List starting with model, followed by fallbacks
        """
        chain = [model]
        config = self._configs.get(model)

        if config:
            chain.extend(config.fallbacks)

        return chain

    def get_penalty(self, model: str, fallback: str) -> float:
        """
        Get the confidence penalty for using a specific fallback.

        Args:
            model: Original model
            fallback: Fallback model being used

        Returns:
            Confidence penalty multiplier (0.0 to 1.0)
        """
        config = self._configs.get(model)

        if not config:
            return 0.80  # Default penalty for undefined chain

        try:
            level = config.fallbacks.index(fallback)
            return config.penalties[level] if level < len(config.penalties) else 0.50
        except ValueError:
            # Fallback not in chain
            return 0.70  # Conservative penalty

    def execute_with_fallbacks(
        self, model_type: str, executor_func: Callable[..., Any], max_fallbacks: int = 3, **kwargs
    ) -> Tuple[Optional[Any], FallbackResult]:
        """
        Execute a valuation with automatic fallbacks.

        Args:
            model_type: Primary model to try
            executor_func: Function to execute (called with model_type and kwargs)
            max_fallbacks: Maximum number of fallbacks to try
            **kwargs: Arguments to pass to executor

        Returns:
            Tuple of (result, FallbackResult) where result may be None
        """
        failed_models: List[str] = []
        fallback_level = 0
        current_model = model_type

        while fallback_level <= max_fallbacks:
            try:
                # Attempt execution
                result = executor_func(model_type=current_model, **kwargs)

                # Success
                if fallback_level == 0:
                    return (
                        result,
                        FallbackResult(
                            original_model=model_type,
                            fallback_model=None,
                            reason=FallbackReason.DATA_MISSING,  # Placeholder
                            confidence_penalty=1.0,
                            fallback_level=0,
                            notes=["Primary model succeeded"],
                        ),
                    )
                else:
                    penalty = self.get_penalty(model_type, current_model)
                    return (
                        result,
                        FallbackResult(
                            original_model=model_type,
                            fallback_model=current_model,
                            reason=FallbackReason.CALCULATION_ERROR,
                            confidence_penalty=penalty,
                            fallback_level=fallback_level,
                            notes=[f"Used fallback after {', '.join(failed_models)} failed"],
                        ),
                    )

            except Exception as e:
                logger.warning(f"Model {current_model} failed: {e}")
                failed_models.append(current_model)

                # Get next fallback
                fallback = self.get_fallback(model_type, failed_models)

                if fallback:
                    current_model, _ = fallback
                    fallback_level += 1
                else:
                    # No more fallbacks
                    break

        # All attempts failed
        return (
            None,
            FallbackResult(
                original_model=model_type,
                fallback_model=None,
                reason=FallbackReason.CALCULATION_ERROR,
                confidence_penalty=0.0,
                fallback_level=fallback_level,
                notes=[f"All models failed: {', '.join(failed_models)}"],
            ),
        )

    def get_applicable_models(
        self, available_data: Dict[str, bool], preferred_order: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get list of applicable models based on available data.

        Args:
            available_data: Dict of data field -> is_available
            preferred_order: Preferred model order

        Returns:
            List of (model, confidence) tuples in order
        """
        # Data requirements by model
        requirements = {
            "dcf": ["fcf", "discount_rate"],
            "pe": ["eps", "pe_ratio"],
            "ps": ["revenue", "ps_ratio"],
            "ggm": ["dividend", "payout_ratio"],
            "ev_ebitda": ["ebitda", "enterprise_value"],
            "pb": ["book_value"],
            "rule_of_40": ["revenue_growth", "fcf_margin"],
        }

        applicable: List[Tuple[str, float]] = []

        models_to_check = preferred_order or list(requirements.keys())

        for model in models_to_check:
            if model not in requirements:
                continue

            required = requirements[model]
            all_available = all(available_data.get(field, False) for field in required)

            if all_available:
                applicable.append((model, 1.0))
            else:
                # Check if we can fallback
                fallback_result = self.get_fallback(model)
                if fallback_result:
                    fallback_model, penalty = fallback_result
                    # Check if fallback has data
                    if fallback_model in requirements:
                        fb_required = requirements[fallback_model]
                        fb_available = all(available_data.get(f, False) for f in fb_required)
                        if fb_available:
                            applicable.append((fallback_model, penalty))

        return applicable


# Singleton instance
_fallback_chain: Optional[FallbackChain] = None


def get_fallback_chain() -> FallbackChain:
    """Get the singleton FallbackChain instance."""
    global _fallback_chain
    if _fallback_chain is None:
        _fallback_chain = FallbackChain()
    return _fallback_chain
