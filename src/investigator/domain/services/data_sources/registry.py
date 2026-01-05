"""
Data Source Registry - Factory and Registry Patterns

Provides centralized management of all data sources with:
- Dynamic registration of new sources
- Factory methods for source instantiation
- Dependency injection support
- Configuration-driven source selection
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .base import (
    CompositeDataSource,
    DataCategory,
    DataFrequency,
    DataResult,
    DataSource,
    MacroDataSource,
    SourceMetadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Source Registration Decorator
# =============================================================================

# Global registry storage
_SOURCE_REGISTRY: Dict[str, Type[DataSource]] = {}
_SOURCE_INSTANCES: Dict[str, DataSource] = {}
_SOURCE_FACTORIES: Dict[str, Callable[[], DataSource]] = {}


def register_source(name: str, category: Optional[DataCategory] = None):
    """
    Decorator to register a data source class.

    Usage:
        @register_source("treasury_yields", DataCategory.FIXED_INCOME)
        class TreasuryYieldSource(DataSource):
            ...
    """

    def decorator(cls: Type[DataSource]) -> Type[DataSource]:
        _SOURCE_REGISTRY[name] = cls
        logger.debug(f"Registered data source: {name} -> {cls.__name__}")
        return cls

    return decorator


def register_factory(name: str):
    """
    Decorator to register a factory function for creating sources.

    Usage:
        @register_factory("composite_macro")
        def create_macro_source():
            return CompositeDataSource(...)
    """

    def decorator(func: Callable[[], DataSource]) -> Callable[[], DataSource]:
        _SOURCE_FACTORIES[name] = func
        logger.debug(f"Registered factory: {name}")
        return func

    return decorator


# =============================================================================
# Data Source Registry
# =============================================================================


@dataclass
class SourceConfig:
    """Configuration for a data source"""

    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    cache_ttl_hours: int = 24
    rate_limit_per_minute: Optional[int] = None
    fallback_sources: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


class DataSourceRegistry:
    """
    Central registry for all data sources.

    Implements:
    - Factory pattern for source creation
    - Singleton pattern for source instances
    - Strategy pattern for source selection
    """

    _instance: Optional["DataSourceRegistry"] = None

    def __new__(cls) -> "DataSourceRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._sources: Dict[str, DataSource] = {}
        self._configs: Dict[str, SourceConfig] = {}
        self._category_index: Dict[DataCategory, Set[str]] = {cat: set() for cat in DataCategory}
        self._logger = logging.getLogger("DataSourceRegistry")

    def register(self, name: str, source: DataSource, config: Optional[SourceConfig] = None) -> None:
        """Register a data source instance"""
        self._sources[name] = source
        self._configs[name] = config or SourceConfig()
        self._category_index[source.category].add(name)
        self._logger.info(f"Registered source: {name} ({source.category.name})")

    def register_class(
        self, name: str, source_class: Type[DataSource], config: Optional[SourceConfig] = None, **kwargs
    ) -> None:
        """Register a data source class (lazy instantiation)"""
        _SOURCE_REGISTRY[name] = source_class
        self._configs[name] = config or SourceConfig()

    def get(self, name: str) -> Optional[DataSource]:
        """Get a data source by name (creates if needed)"""
        if name in self._sources:
            return self._sources[name]

        # Try to create from registry
        if name in _SOURCE_REGISTRY:
            source = _SOURCE_REGISTRY[name]()
            self._sources[name] = source
            self._category_index[source.category].add(name)
            return source

        # Try factory
        if name in _SOURCE_FACTORIES:
            source = _SOURCE_FACTORIES[name]()
            self._sources[name] = source
            self._category_index[source.category].add(name)
            return source

        return None

    def get_by_category(self, category: DataCategory) -> List[DataSource]:
        """Get all sources in a category"""
        names = self._category_index.get(category, set())
        return [self.get(name) for name in names if self.get(name)]

    def get_enabled(self) -> List[DataSource]:
        """Get all enabled sources"""
        return [self.get(name) for name, config in self._configs.items() if config.enabled and self.get(name)]

    def get_by_priority(self, category: Optional[DataCategory] = None) -> List[DataSource]:
        """Get sources sorted by priority"""
        sources = []
        for name, config in self._configs.items():
            if not config.enabled:
                continue
            source = self.get(name)
            if source and (category is None or source.category == category):
                sources.append((config.priority, source))

        sources.sort(key=lambda x: x[0])
        return [s for _, s in sources]

    def create_composite(
        self, name: str, source_names: List[str], strategy: str = "first_success"
    ) -> CompositeDataSource:
        """Create a composite source from multiple sources"""
        sources = [self.get(n) for n in source_names if self.get(n)]
        if not sources:
            raise ValueError(f"No valid sources found: {source_names}")

        composite = CompositeDataSource(name=name, category=sources[0].category, sources=sources, strategy=strategy)
        self.register(name, composite)
        return composite

    def list_sources(self) -> List[Dict[str, Any]]:
        """List all registered sources with metadata"""
        result = []
        all_names = set(self._sources.keys()) | set(_SOURCE_REGISTRY.keys()) | set(_SOURCE_FACTORIES.keys())

        for name in sorted(all_names):
            source = self.get(name)
            if source:
                config = self._configs.get(name, SourceConfig())
                result.append(
                    {
                        "name": name,
                        "category": source.category.name,
                        "frequency": source.frequency.name,
                        "enabled": config.enabled,
                        "priority": config.priority,
                        "metadata": source.metadata.__dict__ if hasattr(source, "metadata") else {},
                    }
                )
        return result

    def invalidate_all_caches(self) -> None:
        """Invalidate caches for all sources"""
        for source in self._sources.values():
            source.invalidate_cache()


# =============================================================================
# Global Registry Access
# =============================================================================


@lru_cache(maxsize=1)
def get_registry() -> DataSourceRegistry:
    """Get the global data source registry"""
    return DataSourceRegistry()


def get_source(name: str) -> Optional[DataSource]:
    """Convenience function to get a source by name"""
    return get_registry().get(name)


def fetch_data(source_name: str, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
    """Convenience function to fetch data from a named source"""
    source = get_source(source_name)
    if not source:
        return DataResult(success=False, error=f"Source not found: {source_name}", source=source_name)
    return source.fetch(symbol, as_of_date)


# =============================================================================
# Source Group Definitions (for common combinations)
# =============================================================================

SOURCE_GROUPS = {
    "macro_all": [
        "fred_macro",
        "atlanta_fed",
        "chicago_fed",
        "cleveland_fed",
        "dallas_fed",
        "kansas_city_fed",
        "new_york_fed",
        "philadelphia_fed",
        "richmond_fed",
    ],
    "volatility": [
        "cboe_vix",
        "cboe_skew",
        "cboe_term_structure",
    ],
    "fixed_income": [
        "treasury_yields",
        "treasury_spreads",
        "credit_spreads",
    ],
    "sentiment": [
        "insider_transactions",
        "institutional_holdings",
        "short_interest",
    ],
    "fundamental": [
        "sec_quarterly",
        "sec_annual",
        "company_facts",
    ],
    "market_data": [
        "price_history",
        "technical_indicators",
        "market_regime",
    ],
}


def get_source_group(group_name: str) -> List[DataSource]:
    """Get all sources in a predefined group"""
    registry = get_registry()
    source_names = SOURCE_GROUPS.get(group_name, [])
    return [registry.get(name) for name in source_names if registry.get(name)]


# =============================================================================
# Configuration Loader
# =============================================================================


def load_sources_from_config(config_path: str) -> None:
    """Load and register sources from YAML configuration"""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found: {config_path}")
        return

    with open(path) as f:
        config = yaml.safe_load(f)

    registry = get_registry()

    for source_name, source_config in config.get("sources", {}).items():
        cfg = SourceConfig(
            enabled=source_config.get("enabled", True),
            priority=source_config.get("priority", 100),
            cache_ttl_hours=source_config.get("cache_ttl_hours", 24),
            rate_limit_per_minute=source_config.get("rate_limit"),
            fallback_sources=source_config.get("fallbacks", []),
            options=source_config.get("options", {}),
        )
        registry._configs[source_name] = cfg


# =============================================================================
# Health Check
# =============================================================================


def check_source_health(source_name: str) -> Dict[str, Any]:
    """Check health of a data source"""
    source = get_source(source_name)
    if not source:
        return {"name": source_name, "status": "not_found"}

    try:
        # Try a simple fetch (use _MACRO for macro sources)
        test_symbol = "_MACRO" if source.category == DataCategory.MACRO else "AAPL"
        result = source.fetch(test_symbol)

        return {
            "name": source_name,
            "status": "healthy" if result.success else "degraded",
            "category": source.category.name,
            "last_error": result.error if not result.success else None,
            "quality": result.quality.name if result.success else None,
        }
    except Exception as e:
        return {
            "name": source_name,
            "status": "error",
            "error": str(e),
        }


def check_all_sources_health() -> List[Dict[str, Any]]:
    """Check health of all registered sources"""
    registry = get_registry()
    return [check_source_health(name) for name in registry._sources.keys()]
