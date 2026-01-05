"""
Cache Infrastructure

Multi-tier caching system with file, parquet, and RDBMS backends.
"""

from investigator.infrastructure.cache.cache_manager import CacheManager, get_cache_manager
from investigator.infrastructure.cache.cache_types import CacheType
from investigator.infrastructure.cache.file_cache_handler import FileCacheStorageHandler
from investigator.infrastructure.cache.industry_metrics_cache import (
    IndustryBenchmarksCacheEntry,
    IndustryMetricsCache,
    IndustryMetricsCacheEntry,
)
from investigator.infrastructure.cache.parquet_cache_handler import ParquetCacheStorageHandler
from investigator.infrastructure.cache.rdbms_cache_handler import RdbmsCacheStorageHandler

__all__ = [
    "CacheManager",
    "CacheType",
    "get_cache_manager",
    "FileCacheStorageHandler",
    "ParquetCacheStorageHandler",
    "RdbmsCacheStorageHandler",
    "IndustryMetricsCache",
    "IndustryMetricsCacheEntry",
    "IndustryBenchmarksCacheEntry",
]
