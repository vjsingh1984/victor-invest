"""
Abstract Base Classes for Data Sources - SOLID Principles

This module defines the core abstractions for all data sources following:
- Single Responsibility: Each source handles one data type
- Open/Closed: New sources extend base, don't modify existing
- Liskov Substitution: All sources are interchangeable via interface
- Interface Segregation: Specific interfaces for different capabilities
- Dependency Inversion: High-level modules depend on abstractions
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Categories
# =============================================================================


class DataCategory(Enum):
    """Categories of data sources for organization and filtering"""

    MARKET_DATA = auto()  # Prices, volumes, technical indicators
    FUNDAMENTAL = auto()  # SEC filings, financials, ratios
    MACRO = auto()  # Economic indicators, Fed data
    SENTIMENT = auto()  # News, social, insider/institutional
    VOLATILITY = auto()  # VIX, SKEW, options data
    FIXED_INCOME = auto()  # Treasury, credit spreads
    ALTERNATIVE = auto()  # Satellite, web traffic, etc.


class DataFrequency(Enum):
    """Data update frequency"""

    REAL_TIME = auto()
    INTRADAY = auto()
    DAILY = auto()
    WEEKLY = auto()
    BIWEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    ANNUAL = auto()


class DataQuality(Enum):
    """Data quality levels"""

    HIGH = auto()  # Primary source, validated
    MEDIUM = auto()  # Secondary source or computed
    LOW = auto()  # Fallback or estimated
    STALE = auto()  # Data is outdated


# =============================================================================
# Data Result Types
# =============================================================================


@dataclass
class DataResult:
    """Standard result wrapper for all data source responses"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    quality: DataQuality = DataQuality.MEDIUM
    cache_hit: bool = False
    staleness_days: int = 0

    @property
    def is_stale(self) -> bool:
        return self.staleness_days > 7 or self.quality == DataQuality.STALE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality.name,
            "cache_hit": self.cache_hit,
            "staleness_days": self.staleness_days,
        }


@dataclass
class SourceMetadata:
    """Metadata about a data source"""

    name: str
    category: DataCategory
    frequency: DataFrequency
    description: str
    provider: str
    is_free: bool = True
    requires_api_key: bool = False
    rate_limit_per_minute: Optional[int] = None
    lookback_days: int = 365
    symbols_supported: bool = True  # False for macro-only sources


# =============================================================================
# Protocol Interfaces (Interface Segregation Principle)
# =============================================================================


class Fetchable(Protocol):
    """Protocol for sources that can fetch data"""

    def fetch(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult: ...


class BatchFetchable(Protocol):
    """Protocol for sources that support batch fetching"""

    def fetch_batch(self, symbols: List[str], as_of_date: Optional[date] = None) -> Dict[str, DataResult]: ...


class Refreshable(Protocol):
    """Protocol for sources that support data refresh"""

    def refresh(self, symbol: Optional[str] = None) -> bool: ...


class Cacheable(Protocol):
    """Protocol for sources that support caching"""

    def get_cached(self, symbol: str) -> Optional[DataResult]: ...
    def invalidate_cache(self, symbol: Optional[str] = None) -> None: ...


class HistoricalFetchable(Protocol):
    """Protocol for sources that support historical data"""

    def fetch_historical(self, symbol: str, start_date: date, end_date: date) -> DataResult: ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    Implements Template Method pattern for common operations.
    Subclasses must implement _fetch_impl for specific data retrieval.
    """

    def __init__(self, name: str, category: DataCategory, frequency: DataFrequency):
        self.name = name
        self.category = category
        self.frequency = frequency
        self._logger = logging.getLogger(f"datasource.{name}")
        self._cache: Dict[str, DataResult] = {}
        self._last_refresh: Optional[datetime] = None

    @property
    @abstractmethod
    def metadata(self) -> SourceMetadata:
        """Return metadata about this data source"""
        pass

    @abstractmethod
    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Implementation-specific fetch logic"""
        pass

    def fetch(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """
        Template method for fetching data with standard pre/post processing.

        1. Check cache
        2. Validate inputs
        3. Fetch from source
        4. Post-process and cache
        """
        cache_key = self._get_cache_key(symbol, as_of_date)

        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_stale:
                cached.cache_hit = True
                return cached

        # Validate
        if not self._validate_symbol(symbol):
            return DataResult(
                success=False,
                error=f"Invalid symbol: {symbol}",
                source=self.name,
            )

        try:
            # Fetch from implementation
            result = self._fetch_impl(symbol, as_of_date)
            result.source = self.name

            # Cache successful results
            if result.success:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            self._logger.error(f"Error fetching {symbol}: {e}")
            return DataResult(
                success=False,
                error=str(e),
                source=self.name,
            )

    def _get_cache_key(self, symbol: str, as_of_date: Optional[date]) -> str:
        """Generate cache key"""
        date_str = as_of_date.isoformat() if as_of_date else "latest"
        return f"{symbol}:{date_str}"

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format"""
        if not self.metadata.symbols_supported:
            return True  # Macro sources don't need symbol validation
        return bool(symbol) and len(symbol) <= 10

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Invalidate cache for symbol or all"""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{symbol}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


class MacroDataSource(DataSource):
    """
    Base class for macro-economic data sources (no symbol required).
    """

    def __init__(self, name: str, frequency: DataFrequency):
        super().__init__(name, DataCategory.MACRO, frequency)

    def fetch(self, symbol: str = "_MACRO", as_of_date: Optional[date] = None) -> DataResult:
        """Macro sources use _MACRO as default symbol"""
        return super().fetch(symbol or "_MACRO", as_of_date)


class MarketDataSource(DataSource):
    """
    Base class for market data sources (requires symbol).
    """

    def __init__(self, name: str, frequency: DataFrequency):
        super().__init__(name, DataCategory.MARKET_DATA, frequency)

    @abstractmethod
    def fetch_historical(self, symbol: str, start_date: date, end_date: date) -> DataResult:
        """Fetch historical data range"""
        pass


class SentimentDataSource(DataSource):
    """
    Base class for sentiment data sources (news, social, insider).
    """

    def __init__(self, name: str, frequency: DataFrequency):
        super().__init__(name, DataCategory.SENTIMENT, frequency)

    @abstractmethod
    def get_sentiment_score(self, symbol: str) -> float:
        """Return normalized sentiment score (-1 to 1)"""
        pass


# =============================================================================
# Composite Pattern for Multiple Sources
# =============================================================================


class CompositeDataSource(DataSource):
    """
    Combines multiple data sources with fallback logic.

    Implements Composite pattern for treating groups of sources uniformly.
    """

    def __init__(
        self,
        name: str,
        category: DataCategory,
        sources: List[DataSource],
        strategy: str = "first_success",  # or "merge", "best_quality"
    ):
        super().__init__(name, category, DataFrequency.DAILY)
        self.sources = sources
        self.strategy = strategy

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name=self.name,
            category=self.category,
            frequency=self.frequency,
            description=f"Composite of {len(self.sources)} sources",
            provider="composite",
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Try sources in order based on strategy"""
        if self.strategy == "first_success":
            return self._first_success_strategy(symbol, as_of_date)
        elif self.strategy == "merge":
            return self._merge_strategy(symbol, as_of_date)
        elif self.strategy == "best_quality":
            return self._best_quality_strategy(symbol, as_of_date)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _first_success_strategy(self, symbol: str, as_of_date: Optional[date]) -> DataResult:
        """Return first successful result"""
        errors = []
        for source in self.sources:
            result = source.fetch(symbol, as_of_date)
            if result.success:
                return result
            errors.append(f"{source.name}: {result.error}")

        return DataResult(
            success=False,
            error=f"All sources failed: {'; '.join(errors)}",
            source=self.name,
        )

    def _merge_strategy(self, symbol: str, as_of_date: Optional[date]) -> DataResult:
        """Merge data from all sources"""
        merged_data = {}
        best_quality = DataQuality.LOW

        for source in self.sources:
            result = source.fetch(symbol, as_of_date)
            if result.success and result.data:
                merged_data.update(result.data)
                if result.quality.value < best_quality.value:
                    best_quality = result.quality

        if merged_data:
            return DataResult(
                success=True,
                data=merged_data,
                source=self.name,
                quality=best_quality,
            )

        return DataResult(success=False, error="No data from any source", source=self.name)

    def _best_quality_strategy(self, symbol: str, as_of_date: Optional[date]) -> DataResult:
        """Return highest quality result"""
        results = []
        for source in self.sources:
            result = source.fetch(symbol, as_of_date)
            if result.success:
                results.append(result)

        if results:
            # Sort by quality (lower enum value = better quality)
            best = min(results, key=lambda r: r.quality.value)
            return best

        return DataResult(success=False, error="No successful results", source=self.name)


# =============================================================================
# Observer Pattern for Data Updates
# =============================================================================


class DataUpdateObserver(Protocol):
    """Protocol for observing data updates"""

    def on_data_updated(self, source: str, symbol: str, data: DataResult) -> None: ...


class ObservableDataSource(DataSource):
    """Data source that notifies observers on updates"""

    def __init__(self, name: str, category: DataCategory, frequency: DataFrequency):
        super().__init__(name, category, frequency)
        self._observers: List[DataUpdateObserver] = []

    def add_observer(self, observer: DataUpdateObserver) -> None:
        self._observers.append(observer)

    def remove_observer(self, observer: DataUpdateObserver) -> None:
        self._observers.remove(observer)

    def _notify_observers(self, symbol: str, data: DataResult) -> None:
        for observer in self._observers:
            try:
                observer.on_data_updated(self.name, symbol, data)
            except Exception as e:
                self._logger.error(f"Observer notification failed: {e}")

    def fetch(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        result = super().fetch(symbol, as_of_date)
        if result.success and not result.cache_hit:
            self._notify_observers(symbol, result)
        return result
