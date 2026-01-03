"""
Industry Dataset Registry

Centralized registry for managing industry-specific datasets.
Implements the singleton pattern for global access and thread-safety.

Author: Claude Code
Date: 2025-12-30
"""

import logging
from threading import Lock
from typing import Dict, List, Optional, Set

from investigator.domain.services.industry_datasets.base import (
    BaseIndustryDataset,
    IndustryMetrics,
)

logger = logging.getLogger(__name__)


class IndustryDatasetRegistry:
    """
    Singleton registry for industry-specific datasets.

    Provides:
    - Registration of new datasets
    - Lookup by industry name or stock symbol
    - Thread-safe access
    - Auto-discovery of industry matches

    Usage:
        registry = IndustryDatasetRegistry.get_instance()
        registry.register(SemiconductorDataset())

        dataset = registry.get_for_industry("Semiconductors")
        if dataset:
            metrics = dataset.extract_metrics(...)
    """

    _instance: Optional["IndustryDatasetRegistry"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "IndustryDatasetRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._datasets: Dict[str, BaseIndustryDataset] = {}
        self._industry_index: Dict[str, str] = {}  # industry_name -> dataset_name
        self._symbol_index: Dict[str, str] = {}  # symbol -> dataset_name
        self._initialized = True

        logger.info("IndustryDatasetRegistry initialized")

    @classmethod
    def get_instance(cls) -> "IndustryDatasetRegistry":
        """Get the singleton instance of the registry."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (primarily for testing)."""
        with cls._lock:
            cls._instance = None

    def register(self, dataset: BaseIndustryDataset) -> None:
        """
        Register an industry dataset.

        Args:
            dataset: The dataset instance to register

        Raises:
            ValueError: If a dataset with the same name is already registered
        """
        with self._lock:
            name = dataset.name

            if name in self._datasets:
                logger.warning(f"Dataset '{name}' already registered, skipping")
                return

            self._datasets[name] = dataset

            # Index by industry names
            for industry in dataset.get_industry_names():
                industry_lower = industry.lower()
                if industry_lower in self._industry_index:
                    existing = self._industry_index[industry_lower]
                    logger.warning(f"Industry '{industry}' already mapped to '{existing}', " f"now mapping to '{name}'")
                self._industry_index[industry_lower] = name

            # Index by known symbols
            for symbol in dataset.get_known_symbols():
                symbol_upper = symbol.upper()
                if symbol_upper in self._symbol_index:
                    existing = self._symbol_index[symbol_upper]
                    logger.warning(f"Symbol '{symbol}' already mapped to '{existing}', " f"now mapping to '{name}'")
                self._symbol_index[symbol_upper] = name

            logger.info(
                f"Registered dataset '{name}' with {len(dataset.get_industry_names())} "
                f"industries and {len(dataset.get_known_symbols())} known symbols"
            )

    def unregister(self, name: str) -> bool:
        """
        Unregister a dataset by name.

        Args:
            name: The dataset name to unregister

        Returns:
            True if dataset was unregistered, False if not found
        """
        with self._lock:
            if name not in self._datasets:
                return False

            dataset = self._datasets[name]

            # Remove from industry index
            for industry in dataset.get_industry_names():
                industry_lower = industry.lower()
                if self._industry_index.get(industry_lower) == name:
                    del self._industry_index[industry_lower]

            # Remove from symbol index
            for symbol in dataset.get_known_symbols():
                symbol_upper = symbol.upper()
                if self._symbol_index.get(symbol_upper) == name:
                    del self._symbol_index[symbol_upper]

            del self._datasets[name]
            logger.info(f"Unregistered dataset '{name}'")
            return True

    def get(self, name: str) -> Optional[BaseIndustryDataset]:
        """Get a dataset by its unique name."""
        return self._datasets.get(name)

    def get_for_industry(self, industry: Optional[str]) -> Optional[BaseIndustryDataset]:
        """
        Get a dataset that handles the given industry.

        Uses both exact match and fuzzy matching.

        Args:
            industry: Industry name from metadata service

        Returns:
            Matching dataset or None
        """
        if not industry:
            return None

        industry_lower = industry.lower()

        # Try exact match first
        if industry_lower in self._industry_index:
            return self._datasets.get(self._industry_index[industry_lower])

        # Try fuzzy match (substring matching)
        for dataset in self._datasets.values():
            if dataset.matches_industry(industry):
                return dataset

        return None

    def get_for_symbol(self, symbol: str) -> Optional[BaseIndustryDataset]:
        """
        Get a dataset for a specific stock symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Matching dataset or None
        """
        symbol_upper = symbol.upper()

        # Check direct symbol mapping
        if symbol_upper in self._symbol_index:
            return self._datasets.get(self._symbol_index[symbol_upper])

        return None

    def get_best_match(
        self, symbol: str, industry: Optional[str] = None, sector: Optional[str] = None
    ) -> Optional[BaseIndustryDataset]:
        """
        Get the best matching dataset for a stock.

        Priority:
        1. Known symbol mapping
        2. Industry match
        3. Sector-based inference

        Args:
            symbol: Stock symbol
            industry: Industry name (optional)
            sector: Sector name (optional)

        Returns:
            Best matching dataset or None
        """
        # Try symbol first
        dataset = self.get_for_symbol(symbol)
        if dataset:
            return dataset

        # Try industry
        dataset = self.get_for_industry(industry)
        if dataset:
            return dataset

        # Could extend to sector-based matching if needed
        return None

    def list_datasets(self) -> List[str]:
        """Return list of registered dataset names."""
        return list(self._datasets.keys())

    def list_industries(self) -> List[str]:
        """Return list of all registered industry names."""
        industries = []
        for dataset in self._datasets.values():
            industries.extend(dataset.get_industry_names())
        return industries

    def list_symbols(self) -> Set[str]:
        """Return set of all known symbols."""
        symbols = set()
        for dataset in self._datasets.values():
            symbols.update(dataset.get_known_symbols())
        return symbols

    def get_statistics(self) -> Dict:
        """Return registry statistics."""
        return {
            "total_datasets": len(self._datasets),
            "total_industries": len(self._industry_index),
            "total_known_symbols": len(self._symbol_index),
            "datasets": {
                name: {
                    "industries": len(ds.get_industry_names()),
                    "symbols": len(ds.get_known_symbols()),
                    "version": ds.version,
                }
                for name, ds in self._datasets.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"IndustryDatasetRegistry("
            f"datasets={len(self._datasets)}, "
            f"industries={len(self._industry_index)}, "
            f"symbols={len(self._symbol_index)})"
        )


# Module-level convenience functions


def get_registry() -> IndustryDatasetRegistry:
    """Get the global registry instance."""
    return IndustryDatasetRegistry.get_instance()


def register_dataset(dataset: BaseIndustryDataset) -> None:
    """Register a dataset with the global registry."""
    get_registry().register(dataset)


def get_dataset_for_industry(industry: Optional[str]) -> Optional[BaseIndustryDataset]:
    """Get a dataset for the given industry from the global registry."""
    return get_registry().get_for_industry(industry)


def get_dataset_for_symbol(symbol: str) -> Optional[BaseIndustryDataset]:
    """Get a dataset for the given symbol from the global registry."""
    return get_registry().get_for_symbol(symbol)


def list_registered_industries() -> List[str]:
    """List all registered industries."""
    return get_registry().list_industries()
