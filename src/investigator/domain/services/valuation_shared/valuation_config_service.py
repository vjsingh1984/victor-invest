# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""
Valuation Config Service - Centralized config access for all valuation consumers.

This service provides a single source of truth for all valuation parameters,
eliminating hardcoded values scattered across:
- scripts/rl_backtest.py
- victor_invest/tools/valuation.py
- batch_analysis_runner.py

All values are read from config.yaml with sensible defaults.

Example:
    service = ValuationConfigService()

    # Get sector multiples
    pe = service.get_sector_pe_multiple("Technology")  # Returns 28
    ps = service.get_sector_ps_multiple("Technology")  # Returns 6

    # Get CAPM parameters
    capm = service.get_capm_params()
    # {'risk_free_rate': 0.04, 'market_equity_premium': 0.05}

    # Get GGM defaults
    ggm = service.get_ggm_defaults()
    # {'growth_rate': 0.03, 'cost_of_equity': 0.08, 'min_payout_ratio': 0.20}
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ValuationConfigService:
    """
    Centralized service for accessing valuation configuration.

    Provides type-safe access to all valuation parameters with defaults.
    Caches config for performance.
    """

    # Default values matching previously hardcoded values in rl_backtest.py
    DEFAULT_SECTOR_MULTIPLES = {
        "pe": {
            "Technology": 28,
            "Healthcare": 22,
            "Financials": 12,
            "Consumer Cyclical": 20,
            "Consumer Discretionary": 20,
            "Consumer Defensive": 22,
            "Consumer Staples": 22,
            "Industrials": 18,
            "Energy": 12,
            "Materials": 15,
            "Real Estate": 35,
            "Utilities": 18,
            "Communication Services": 20,
            "Default": 18,
        },
        "ps": {
            "Technology": 6,
            "Healthcare": 4,
            "Financials": 3,
            "Consumer Cyclical": 1.5,
            "Consumer Discretionary": 1.5,
            "Consumer Defensive": 2,
            "Consumer Staples": 2,
            "Industrials": 1.5,
            "Energy": 1,
            "Materials": 1.2,
            "Real Estate": 8,
            "Utilities": 2,
            "Communication Services": 3,
            "Default": 2,
        },
        "pb": {
            "Technology": 5,
            "Healthcare": 4,
            "Financials": 1.2,
            "Consumer Cyclical": 3,
            "Consumer Discretionary": 3,
            "Consumer Defensive": 4,
            "Consumer Staples": 4,
            "Industrials": 3,
            "Energy": 1.5,
            "Materials": 2,
            "Real Estate": 1.5,
            "Utilities": 1.8,
            "Communication Services": 3,
            "Default": 2.5,
        },
        "ev_ebitda": {
            "Technology": 18,
            "Healthcare": 14,
            "Financials": 8,
            "Consumer Cyclical": 12,
            "Consumer Discretionary": 12,
            "Consumer Defensive": 14,
            "Consumer Staples": 14,
            "Industrials": 10,
            "Energy": 6,
            "Materials": 8,
            "Real Estate": 18,
            "Utilities": 10,
            "Communication Services": 10,
            "Default": 10,
        },
    }

    DEFAULT_CAPM = {
        "risk_free_rate": 0.04,
        "market_equity_premium": 0.05,
    }

    DEFAULT_GGM = {
        "growth_rate": 0.03,
        "cost_of_equity": 0.08,
        "min_payout_ratio": 0.20,
    }

    DEFAULT_GROWTH_ASSUMPTIONS = {
        "forward_eps_growth_pct": 0.10,
    }

    DEFAULT_FALLBACKS = {
        "dcf_simple_multiplier": 12,
        "ebitda_da_approximation_pct": 0.05,
    }

    _instance = None
    _config = None

    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern for config service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ValuationConfigService.

        Args:
            config_path: Path to config.yaml. If None, searches standard locations.
        """
        if self._initialized:
            return

        self._config_path = config_path
        self._config = self._load_config()
        self._initialized = True

    def _load_config(self) -> Dict[str, Any]:
        """Load config from YAML file."""
        if self._config_path:
            config_file = Path(self._config_path)
        else:
            # Search standard locations
            search_paths = [
                Path.cwd() / "config.yaml",
                Path(__file__).parents[5] / "config.yaml",  # Project root
                Path.home() / ".investigator" / "config.yaml",
            ]
            config_file = None
            for path in search_paths:
                if path.exists():
                    config_file = path
                    break

        if config_file and config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded valuation config from {config_file}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")

        logger.warning("Using default valuation config (no config.yaml found)")
        return {}

    def reload_config(self) -> None:
        """Force reload of config from file."""
        self._config = self._load_config()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._config = None

    # =========================================================================
    # Sector Multiple Accessors
    # =========================================================================

    def get_sector_pe_multiple(self, sector: str) -> float:
        """
        Get P/E multiple for a sector.

        Args:
            sector: Sector name (e.g., "Technology", "Healthcare")

        Returns:
            P/E multiple for the sector, or default if not found
        """
        multiples = self._get_nested(
            "valuation.sector_multiples.pe",
            self.DEFAULT_SECTOR_MULTIPLES["pe"]
        )
        return multiples.get(sector, multiples.get("Default", 18))

    def get_sector_ps_multiple(self, sector: str) -> float:
        """
        Get P/S multiple for a sector.

        Args:
            sector: Sector name

        Returns:
            P/S multiple for the sector
        """
        multiples = self._get_nested(
            "valuation.sector_multiples.ps",
            self.DEFAULT_SECTOR_MULTIPLES["ps"]
        )
        return multiples.get(sector, multiples.get("Default", 2))

    def get_sector_pb_multiple(self, sector: str) -> float:
        """
        Get P/B multiple for a sector.

        Args:
            sector: Sector name

        Returns:
            P/B multiple for the sector
        """
        multiples = self._get_nested(
            "valuation.sector_multiples.pb",
            self.DEFAULT_SECTOR_MULTIPLES["pb"]
        )
        return multiples.get(sector, multiples.get("Default", 2.5))

    def get_sector_ev_ebitda_multiple(self, sector: str) -> float:
        """
        Get EV/EBITDA multiple for a sector.

        Args:
            sector: Sector name

        Returns:
            EV/EBITDA multiple for the sector
        """
        multiples = self._get_nested(
            "valuation.sector_multiples.ev_ebitda",
            self.DEFAULT_SECTOR_MULTIPLES["ev_ebitda"]
        )
        return multiples.get(sector, multiples.get("Default", 10))

    def get_all_sector_multiples(self, sector: str) -> Dict[str, float]:
        """
        Get all multiples for a sector.

        Args:
            sector: Sector name

        Returns:
            Dict with pe, ps, pb, ev_ebitda multiples
        """
        return {
            "pe": self.get_sector_pe_multiple(sector),
            "ps": self.get_sector_ps_multiple(sector),
            "pb": self.get_sector_pb_multiple(sector),
            "ev_ebitda": self.get_sector_ev_ebitda_multiple(sector),
        }

    # =========================================================================
    # CAPM Parameters
    # =========================================================================

    def get_capm_params(self) -> Dict[str, float]:
        """
        Get CAPM parameters.

        Returns:
            Dict with risk_free_rate and market_equity_premium
        """
        return self._get_nested("valuation.capm", self.DEFAULT_CAPM)

    def get_risk_free_rate(self) -> float:
        """Get risk-free rate (default: 4%)."""
        return self.get_capm_params().get("risk_free_rate", 0.04)

    def get_market_equity_premium(self) -> float:
        """Get market equity premium (default: 5%)."""
        return self.get_capm_params().get("market_equity_premium", 0.05)

    def get_cost_of_equity(self, beta: float = 1.0) -> float:
        """
        Calculate cost of equity using CAPM.

        Args:
            beta: Stock beta (default: 1.0)

        Returns:
            Cost of equity = risk_free + beta * market_premium
        """
        params = self.get_capm_params()
        rf = params.get("risk_free_rate", 0.04)
        mrp = params.get("market_equity_premium", 0.05)
        return rf + beta * mrp

    # =========================================================================
    # GGM (Gordon Growth Model) Parameters
    # =========================================================================

    def get_ggm_defaults(self) -> Dict[str, float]:
        """
        Get Gordon Growth Model default parameters.

        Returns:
            Dict with growth_rate, cost_of_equity, min_payout_ratio
        """
        return self._get_nested("valuation.ggm_defaults", self.DEFAULT_GGM)

    def get_ggm_growth_rate(self) -> float:
        """Get default GGM growth rate (default: 3%)."""
        return self.get_ggm_defaults().get("growth_rate", 0.03)

    def get_ggm_cost_of_equity(self) -> float:
        """Get default GGM cost of equity (default: 8%)."""
        return self.get_ggm_defaults().get("cost_of_equity", 0.08)

    def get_ggm_min_payout_ratio(self) -> float:
        """Get minimum payout ratio for GGM applicability (default: 20%)."""
        return self.get_ggm_defaults().get("min_payout_ratio", 0.20)

    # =========================================================================
    # Growth Assumptions
    # =========================================================================

    def get_growth_assumptions(self) -> Dict[str, float]:
        """
        Get growth assumptions.

        Returns:
            Dict with forward_eps_growth_pct
        """
        return self._get_nested(
            "valuation.growth_assumptions",
            self.DEFAULT_GROWTH_ASSUMPTIONS
        )

    def get_forward_eps_growth(self) -> float:
        """Get default forward EPS growth rate (default: 10%)."""
        return self.get_growth_assumptions().get("forward_eps_growth_pct", 0.10)

    # =========================================================================
    # Fallback Calculations
    # =========================================================================

    def get_fallbacks(self) -> Dict[str, float]:
        """
        Get fallback calculation parameters.

        Returns:
            Dict with dcf_simple_multiplier, ebitda_da_approximation_pct
        """
        return self._get_nested("valuation.fallbacks", self.DEFAULT_FALLBACKS)

    def get_dcf_simple_multiplier(self) -> float:
        """Get simple DCF multiplier fallback (default: 12x)."""
        return self.get_fallbacks().get("dcf_simple_multiplier", 12)

    def get_ebitda_da_approximation_pct(self) -> float:
        """Get D&A approximation as % of assets (default: 5%)."""
        return self.get_fallbacks().get("ebitda_da_approximation_pct", 0.05)

    # =========================================================================
    # DCF Configuration
    # =========================================================================

    def get_dcf_sector_params(self, sector: str) -> Dict[str, Any]:
        """
        Get DCF parameters for a sector.

        Args:
            sector: Sector name

        Returns:
            Dict with terminal_growth_rate, projection_years, rationale
        """
        sector_params = self._get_nested(
            f"dcf_valuation.sector_based_parameters.{sector}",
            None
        )
        if sector_params:
            return sector_params

        # Return default
        return self._get_nested(
            "dcf_valuation.sector_based_parameters.Default",
            {
                "terminal_growth_rate": 0.03,
                "projection_years": 5,
                "rationale": "Default parameters",
            }
        )

    def get_dcf_default_params(self) -> Dict[str, Any]:
        """Get DCF default parameters."""
        return self._get_nested(
            "dcf_valuation.default_parameters",
            {
                "terminal_growth_rate": 0.03,
                "projection_years": 5,
                "max_terminal_growth_rate": 0.03,
                "min_terminal_growth_rate": 0.025,
            }
        )

    # =========================================================================
    # Tier Configuration
    # =========================================================================

    def get_tier_thresholds(self, tier_name: str) -> Optional[Dict[str, Any]]:
        """
        Get threshold configuration for a tier.

        Args:
            tier_name: Tier name (e.g., "high_growth", "dividend_aristocrat")

        Returns:
            Dict with tier thresholds or None
        """
        return self._get_nested(f"valuation.tier_thresholds.{tier_name}", None)

    def get_tier_base_weights(self, tier_name: str) -> Optional[Dict[str, int]]:
        """
        Get base weights for a tier.

        Args:
            tier_name: Tier name

        Returns:
            Dict mapping model names to weights (0-100)
        """
        return self._get_nested(f"valuation.tier_base_weights.{tier_name}", None)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get nested config value using dot notation.

        Args:
            path: Dot-separated path (e.g., "valuation.capm.risk_free_rate")
            default: Default value if not found

        Returns:
            Config value or default
        """
        keys = path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value if value is not None else default

    def get(self, path: str, default: Any = None) -> Any:
        """
        Generic config getter using dot notation.

        Args:
            path: Dot-separated path
            default: Default value

        Returns:
            Config value or default
        """
        return self._get_nested(path, default)
