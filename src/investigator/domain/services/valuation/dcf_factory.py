"""
DCF Valuation Factory - Unified model selection for DCF variants.

Implements Factory pattern (SOLID: Open/Closed Principle) to:
1. Provide a single entry point for DCF model instantiation
2. Select appropriate model based on company characteristics
3. Allow easy addition of new DCF variants

Usage:
    from investigator.domain.services.valuation.dcf_factory import DCFFactory

    # For companies with prepared CompanyProfile
    model = DCFFactory.create("damodaran", company_profile=profile)
    result = model.calculate(...)

    # For legacy integration with raw data
    model = DCFFactory.create_legacy(symbol, quarterly_metrics, multi_year_data, db_manager)
    result = model.calculate_dcf(...)

    # Auto-select based on company characteristics
    model = DCFFactory.auto_select(company_profile=profile, has_positive_fcf=True)

Author: InvestiGator Team
Date: 2025-01-05
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class DCFModelType(Enum):
    """Available DCF model types."""

    STANDARD = "standard"  # DCFValuation - legacy, works with raw quarterly data
    DAMODARAN = "damodaran"  # DamodaranDCFModel - 3-stage with Monte Carlo


class DCFFactory:
    """
    Factory for creating DCF valuation models.

    Supports two model types:
    1. STANDARD (DCFValuation): Legacy model for raw quarterly data integration
    2. DAMODARAN (DamodaranDCFModel): Modern 3-stage DCF with Monte Carlo

    Design Principles:
    - Open/Closed: Add new models without modifying existing code
    - Single Responsibility: Factory only handles model selection/creation
    - Dependency Inversion: Models are created via abstract factory interface
    """

    # Registry of model types to their classes (lazy loaded)
    _model_registry: Dict[DCFModelType, Type] = {}
    _initialized = False

    @classmethod
    def _ensure_initialized(cls):
        """Lazy initialization of model registry."""
        if cls._initialized:
            return

        # Import models here to avoid circular imports
        from investigator.domain.services.valuation.damodaran_dcf import DamodaranDCFModel
        from investigator.domain.services.valuation.dcf import DCFValuation

        cls._model_registry = {
            DCFModelType.STANDARD: DCFValuation,
            DCFModelType.DAMODARAN: DamodaranDCFModel,
        }
        cls._initialized = True

    @classmethod
    def create(
        cls,
        model_type: str,
        company_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a DCF model instance.

        Args:
            model_type: Model type ("standard", "damodaran")
            company_profile: CompanyProfile for modern models (required for damodaran)
            **kwargs: Additional arguments passed to model constructor

        Returns:
            DCF model instance

        Raises:
            ValueError: If model_type is unknown or required args missing

        Example:
            >>> from investigator.domain.services.valuation.dcf_factory import DCFFactory
            >>> model = DCFFactory.create("damodaran", company_profile=profile)
            >>> result = model.calculate(current_fcf=5e9, ...)
        """
        cls._ensure_initialized()

        # Normalize model type
        try:
            if isinstance(model_type, str):
                model_enum = DCFModelType(model_type.lower())
            else:
                model_enum = model_type
        except ValueError:
            valid_types = [t.value for t in DCFModelType]
            raise ValueError(f"Unknown model type: {model_type}. Valid types: {valid_types}")

        model_class = cls._model_registry.get(model_enum)
        if model_class is None:
            raise ValueError(f"Model type {model_enum} not registered")

        # Create model based on type
        if model_enum == DCFModelType.DAMODARAN:
            if company_profile is None:
                raise ValueError("company_profile required for Damodaran DCF model")
            return model_class(company_profile, **kwargs)

        elif model_enum == DCFModelType.STANDARD:
            # Standard DCF requires different initialization
            raise ValueError("Standard DCF requires legacy initialization. " "Use DCFFactory.create_legacy() instead.")

        return model_class(**kwargs)

    @classmethod
    def create_legacy(
        cls,
        symbol: str,
        quarterly_metrics: List[Dict],
        multi_year_data: List[Dict],
        db_manager: Any,
    ) -> Any:
        """
        Create legacy DCFValuation model for raw data integration.

        This method maintains backward compatibility with existing code that
        uses DCFValuation directly with quarterly metrics and database manager.

        Args:
            symbol: Stock ticker symbol
            quarterly_metrics: List of quarterly financial metrics
            multi_year_data: Multi-year historical data
            db_manager: Database manager instance

        Returns:
            DCFValuation instance

        Example:
            >>> model = DCFFactory.create_legacy("AAPL", quarterly_data, multi_year, db)
            >>> result = model.calculate_dcf()
        """
        cls._ensure_initialized()

        model_class = cls._model_registry[DCFModelType.STANDARD]
        return model_class(
            symbol=symbol,
            quarterly_metrics=quarterly_metrics,
            multi_year_data=multi_year_data,
            db_manager=db_manager,
        )

    @classmethod
    def auto_select(
        cls,
        company_profile: Optional[Any] = None,
        has_positive_fcf: bool = True,
        prefer_monte_carlo: bool = False,
        symbol: Optional[str] = None,
        quarterly_metrics: Optional[List[Dict]] = None,
        multi_year_data: Optional[List[Dict]] = None,
        db_manager: Optional[Any] = None,
    ) -> Any:
        """
        Automatically select and create the most appropriate DCF model.

        Selection criteria:
        1. If company_profile provided → Use Damodaran (modern, cleaner)
        2. If negative FCF → Use Damodaran (has revenue bridge)
        3. If Monte Carlo requested → Use Damodaran
        4. If only raw data available → Use Standard (legacy)

        Args:
            company_profile: CompanyProfile for modern models
            has_positive_fcf: Whether company has positive free cash flow
            prefer_monte_carlo: Whether Monte Carlo analysis is desired
            symbol: Stock symbol (for legacy model)
            quarterly_metrics: Raw quarterly data (for legacy model)
            multi_year_data: Historical data (for legacy model)
            db_manager: Database manager (for legacy model)

        Returns:
            Selected DCF model instance

        Example:
            >>> # Auto-select based on available data
            >>> model = DCFFactory.auto_select(
            ...     company_profile=profile,
            ...     has_positive_fcf=False,  # Company is pre-profitable
            ...     prefer_monte_carlo=True,
            ... )
        """
        cls._ensure_initialized()

        # Decision tree for model selection
        if company_profile is not None:
            # Modern model preferred when profile available
            if not has_positive_fcf:
                # Damodaran has revenue bridge for negative FCF
                logger.info(f"Auto-selected Damodaran DCF (negative FCF, revenue bridge)")
                return cls.create(DCFModelType.DAMODARAN, company_profile=company_profile)

            if prefer_monte_carlo:
                # Damodaran has built-in Monte Carlo
                logger.info(f"Auto-selected Damodaran DCF (Monte Carlo requested)")
                return cls.create(DCFModelType.DAMODARAN, company_profile=company_profile)

            # Default to Damodaran for modern architecture
            logger.info(f"Auto-selected Damodaran DCF (modern architecture)")
            return cls.create(DCFModelType.DAMODARAN, company_profile=company_profile)

        # Fall back to legacy model if only raw data available
        if all([symbol, quarterly_metrics is not None, multi_year_data is not None, db_manager]):
            logger.info(f"Auto-selected Standard DCF (legacy data integration)")
            return cls.create_legacy(
                symbol=symbol,
                quarterly_metrics=quarterly_metrics,
                multi_year_data=multi_year_data,
                db_manager=db_manager,
            )

        raise ValueError(
            "Cannot auto-select DCF model. Provide either: "
            "1) company_profile for modern models, or "
            "2) symbol, quarterly_metrics, multi_year_data, and db_manager for legacy model"
        )

    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available DCF model types.

        Returns:
            List of model type names

        Example:
            >>> DCFFactory.get_available_models()
            ['standard', 'damodaran']
        """
        return [t.value for t in DCFModelType]

    @classmethod
    def get_model_description(cls, model_type: str) -> Dict[str, str]:
        """
        Get description of a DCF model type.

        Args:
            model_type: Model type name

        Returns:
            Dict with model metadata (name, description, use_cases)
        """
        descriptions = {
            "standard": {
                "name": "Standard DCF (DCFValuation)",
                "description": "Legacy DCF model integrated with quarterly metrics and database",
                "use_cases": [
                    "Direct integration with SEC quarterly data",
                    "When db_manager is required for lookups",
                    "Backward compatibility with existing workflows",
                ],
                "features": [
                    "TTM calculations from quarterly data",
                    "SEC format detection",
                    "Sector-based terminal growth",
                ],
            },
            "damodaran": {
                "name": "Damodaran 3-Stage DCF",
                "description": "Modern 3-stage DCF based on Aswath Damodaran's methodology",
                "use_cases": [
                    "High-growth companies with growth decay",
                    "Pre-profitable companies (revenue bridge)",
                    "When Monte Carlo sensitivity analysis is needed",
                ],
                "features": [
                    "3-stage growth model (high → transition → terminal)",
                    "Monte Carlo sensitivity analysis",
                    "Revenue bridge for negative FCF",
                    "Industry-specific cost of capital",
                ],
            },
        }
        return descriptions.get(model_type.lower(), {"error": f"Unknown model: {model_type}"})


# Convenience functions for common use cases
def create_dcf_model(
    model_type: str = "damodaran",
    company_profile: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Convenience function to create a DCF model.

    Args:
        model_type: "standard" or "damodaran" (default: "damodaran")
        company_profile: CompanyProfile for modern models
        **kwargs: Additional model arguments

    Returns:
        DCF model instance

    Example:
        >>> from investigator.domain.services.valuation.dcf_factory import create_dcf_model
        >>> model = create_dcf_model("damodaran", company_profile=profile)
    """
    return DCFFactory.create(model_type, company_profile=company_profile, **kwargs)


def auto_select_dcf(
    company_profile: Optional[Any] = None,
    has_positive_fcf: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Convenience function to auto-select and create DCF model.

    Args:
        company_profile: CompanyProfile for modern models
        has_positive_fcf: Whether company has positive FCF
        **kwargs: Additional selection criteria

    Returns:
        Auto-selected DCF model instance

    Example:
        >>> from investigator.domain.services.valuation.dcf_factory import auto_select_dcf
        >>> model = auto_select_dcf(company_profile=profile, has_positive_fcf=False)
    """
    return DCFFactory.auto_select(
        company_profile=company_profile,
        has_positive_fcf=has_positive_fcf,
        **kwargs,
    )
