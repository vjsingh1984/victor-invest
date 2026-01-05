"""
Model Applicability Rules

Centralized service for determining which valuation models are applicable
given a company's financial characteristics.

Author: InvestiGator Team
Date: 2025-11-07
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelApplicabilityRules:
    """
    Service for checking valuation model applicability.

    Centralizes business logic for determining which models can be used
    based on financial data availability and fundamental assumptions.

    Examples:
    - DCF requires FCF data (at least 4 quarters)
    - GGM requires dividends + positive earnings + 40%+ payout
    - P/E requires positive earnings
    - P/S requires positive revenue
    - P/B requires positive book value
    - EV/EBITDA requires positive EBITDA
    """

    def __init__(self, applicability_config: Optional[Dict[str, Any]] = None):
        """
        Initialize with applicability configuration.

        Args:
            applicability_config: Dict from config.json with model-specific rules
                Example:
                {
                    "dcf": {"min_quarters_data": 4, ...},
                    "ggm": {"min_payout_ratio": 40, ...},
                    ...
                }
        """
        self.config = applicability_config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default applicability configuration.

        Returns:
            Default applicability rules for each model
        """
        return {
            "dcf": {
                "min_quarters_data": 4,
                "require_positive_fcf": False,  # Can have negative FCF
                "reason": "DCF requires at least 4 quarters of cash flow data",
            },
            "ggm": {
                "min_payout_ratio": 0.40,  # Ratio format (0.40 = 40%)
                "require_positive_earnings": True,
                "require_dividends": True,
                "reason": "GGM requires consistent dividends and 40%+ payout ratio",
            },
            "pe": {"require_positive_earnings": True, "reason": "P/E multiple requires positive earnings"},
            "ps": {"require_positive_revenue": True, "reason": "P/S multiple requires positive revenue"},
            "pb": {"require_positive_book_value": True, "reason": "P/B multiple requires positive book value"},
            "ev_ebitda": {"require_positive_ebitda": True, "reason": "EV/EBITDA requires positive EBITDA"},
        }

    def is_applicable(self, model_name: str, financials: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a model is applicable given financial data.

        Args:
            model_name: Model identifier (dcf, ggm, pe, ps, pb, ev_ebitda)
            financials: Financial metrics dict with keys like:
                - net_income
                - revenue
                - dividends_paid
                - payout_ratio
                - book_value
                - ebitda
                - fcf_quarters_count

        Returns:
            (is_applicable: bool, reason: str) tuple
            Example: (True, "All requirements met")
                    (False, "Negative earnings - P/E not applicable")
        """
        model_config = self.config.get(model_name)

        if not model_config:
            logger.warning(f"No applicability rules for model '{model_name}', assuming applicable")
            return True, "No specific rules configured"

        # Check model-specific requirements
        if model_name == "dcf":
            return self._check_dcf_applicability(financials, model_config)
        elif model_name == "ggm":
            return self._check_ggm_applicability(financials, model_config)
        elif model_name == "pe":
            return self._check_pe_applicability(financials, model_config)
        elif model_name == "ps":
            return self._check_ps_applicability(financials, model_config)
        elif model_name == "pb":
            return self._check_pb_applicability(financials, model_config)
        elif model_name == "ev_ebitda":
            return self._check_ev_ebitda_applicability(financials, model_config)
        else:
            logger.warning(f"Unknown model '{model_name}', assuming applicable")
            return True, "Unknown model"

    def _check_dcf_applicability(self, financials: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check DCF model applicability.

        Requirements:
        - At least N quarters of FCF data (default: 4)
        - Optional: Positive FCF (usually not required)
        """
        fcf_quarters = financials.get("fcf_quarters_count", 0)
        min_quarters = config.get("min_quarters_data", 4)

        if fcf_quarters < min_quarters:
            return False, f"Insufficient FCF data: {fcf_quarters} quarters < {min_quarters} required"

        require_positive = config.get("require_positive_fcf", False)
        if require_positive:
            fcf = financials.get("free_cash_flow", 0) or 0
            if fcf <= 0:
                return False, f"Negative FCF: ${fcf:,.0f}"

        return True, "All DCF requirements met"

    def _check_ggm_applicability(self, financials: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check Gordon Growth Model applicability.

        Requirements:
        - Positive earnings (if require_positive_earnings = True)
        - Dividends paid - common or preferred (if require_dividends = True)
        - Payout ratio >= min_payout_ratio (default: 0.40 = 40%)
        """
        # min_payout is in ratio format (0.0-1.0), default 0.40 = 40%
        min_payout = config.get("min_payout_ratio", 0.40)
        require_earnings = config.get("require_positive_earnings", True)
        require_divs = config.get("require_dividends", True)

        # Check earnings
        if require_earnings:
            net_income = financials.get("net_income", 0) or 0
            if net_income <= 0:
                return False, f"Negative/zero earnings: ${net_income:,.0f}"

        # Check dividends (common or preferred)
        if require_divs:
            common_divs = abs(financials.get("dividends_paid", 0) or 0)
            preferred_divs = abs(financials.get("preferred_stock_dividends", 0) or 0)
            total_divs = common_divs + preferred_divs
            if total_divs <= 0:
                return False, "No dividends paid (neither common nor preferred)"

        # Check payout ratio (in ratio format 0.0-1.0)
        payout_ratio = financials.get("payout_ratio", 0) or 0
        if payout_ratio < min_payout:
            return False, f"Low payout ratio: {payout_ratio*100:.1f}% < {min_payout*100:.0f}%"

        return True, "All GGM requirements met"

    def _check_pe_applicability(self, financials: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check P/E model applicability.

        Requirements:
        - Positive earnings
        """
        require_positive = config.get("require_positive_earnings", True)

        if require_positive:
            net_income = financials.get("net_income", 0) or 0
            if net_income <= 0:
                return False, f"Negative/zero earnings: ${net_income:,.0f}"

        return True, "P/E requirements met (positive earnings)"

    def _check_ps_applicability(self, financials: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check P/S model applicability.

        Requirements:
        - Positive revenue
        """
        require_positive = config.get("require_positive_revenue", True)

        if require_positive:
            revenue = financials.get("revenue", 0) or 0
            if revenue <= 0:
                return False, f"Negative/zero revenue: ${revenue:,.0f}"

        return True, "P/S requirements met (positive revenue)"

    def _check_pb_applicability(self, financials: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check P/B model applicability.

        Requirements:
        - Positive book value (or stockholders_equity as fallback)
        """
        require_positive = config.get("require_positive_book_value", True)

        if require_positive:
            # Check book_value, book_value_per_share, or stockholders_equity
            book_value = (
                financials.get("book_value", 0)
                or financials.get("book_value_per_share", 0)
                or financials.get("stockholders_equity", 0)
                or 0
            )
            if book_value <= 0:
                return False, f"Negative/zero book value: ${book_value:,.0f}"

        return True, "P/B requirements met (positive book value)"

    def _check_ev_ebitda_applicability(self, financials: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check EV/EBITDA model applicability.

        Requirements:
        - Positive EBITDA
        """
        require_positive = config.get("require_positive_ebitda", True)

        if require_positive:
            ebitda = financials.get("ebitda", 0) or 0
            if ebitda <= 0:
                return False, f"Negative/zero EBITDA: ${ebitda:,.0f}"

        return True, "EV/EBITDA requirements met (positive EBITDA)"

    def filter_applicable_models(self, models: list[str], financials: Dict[str, Any]) -> Dict[str, Tuple[bool, str]]:
        """
        Check applicability for multiple models at once.

        Args:
            models: List of model names to check
            financials: Financial metrics dict

        Returns:
            Dict mapping model_name → (is_applicable, reason)
            Example:
            {
                "dcf": (True, "All DCF requirements met"),
                "ggm": (False, "Low payout ratio: 15% < 40%"),
                "pe": (True, "P/E requirements met"),
                ...
            }
        """
        results = {}
        for model in models:
            is_applicable, reason = self.is_applicable(model, financials)
            results[model] = (is_applicable, reason)
        return results

    def get_applicable_models_only(self, models: list[str], financials: Dict[str, Any]) -> list[str]:
        """
        Get list of only applicable models (convenience method).

        Args:
            models: List of model names to check
            financials: Financial metrics dict

        Returns:
            List of applicable model names only
            Example: ["dcf", "pe", "ps"]
        """
        applicable = []
        for model in models:
            is_applicable, _ = self.is_applicable(model, financials)
            if is_applicable:
                applicable.append(model)
        return applicable

    def get_inapplicable_models_with_reasons(self, models: list[str], financials: Dict[str, Any]) -> Dict[str, str]:
        """
        Get dict of inapplicable models with reasons (convenience method).

        Args:
            models: List of model names to check
            financials: Financial metrics dict

        Returns:
            Dict mapping inapplicable model_name → reason
            Example: {"ggm": "Low payout ratio: 15% < 40%"}
        """
        inapplicable = {}
        for model in models:
            is_applicable, reason = self.is_applicable(model, financials)
            if not is_applicable:
                inapplicable[model] = reason
        return inapplicable
