# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cache Tool for Victor Invest.

This tool wraps the existing cache infrastructure to provide cache
management operations including get, set, delete, and stats.

Infrastructure wrapped:
- investigator.infrastructure.cache.cache_manager.CacheManager

Example:
    tool = CacheTool()

    # Get cached data
    result = await tool.execute(
        action="get",
        cache_type="llm_response",
        key={"symbol": "AAPL", "llm_type": "fundamental"}
    )

    # Set cache entry
    result = await tool.execute(
        action="set",
        cache_type="llm_response",
        key={"symbol": "AAPL", "llm_type": "fundamental"},
        value={"analysis": "..."}
    )

    # Get cache stats
    result = await tool.execute(action="get_stats")

    # Clear cache for a symbol
    result = await tool.execute(
        action="delete_by_symbol",
        symbol="AAPL"
    )
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class CacheTool(BaseTool):
    """Tool for managing the cache system.

    Provides operations for:
    - Getting cached data
    - Setting cache entries
    - Deleting cache entries
    - Getting cache statistics
    - Managing cache by symbol
    - Cache validation and cleanup

    Attributes:
        name: "cache"
        description: Tool description for agent discovery
    """

    name = "cache"
    description = """Manage the caching system for analysis data.

Actions:
- get: Retrieve cached data by type and key
- set: Store data in cache
- exists: Check if cache entry exists
- delete: Delete specific cache entry
- delete_by_symbol: Delete all cache entries for a symbol
- clear_type: Clear all entries for a cache type
- get_stats: Get cache performance statistics
- get_recent_ops: Get recent cache operations for debugging
- validate: Validate cache entry integrity
- invalidate_sec: Invalidate stale SEC-related cache entries

Cache Types:
- llm_response: LLM analysis responses
- sec_response: SEC API responses
- company_facts: Company financial facts
- technical_data: Technical indicator data
- quarterly_metrics: Quarterly financial metrics
- submission_data: SEC submission data

Parameters:
- action: Operation to perform (required)
- cache_type: Type of cache (for get/set/delete)
- key: Cache key (dict or tuple)
- value: Data to cache (for set action)
- symbol: Symbol for symbol-based operations

Returns cache data, operation status, or statistics.
"""

    # Map string cache type names to enum values
    CACHE_TYPE_MAP = {
        "llm_response": "LLM_RESPONSE",
        "sec_response": "SEC_RESPONSE",
        "company_facts": "COMPANY_FACTS",
        "technical_data": "TECHNICAL_DATA",
        "quarterly_metrics": "QUARTERLY_METRICS",
        "submission_data": "SUBMISSION_DATA",
        "market_context": "MARKET_CONTEXT",
    }

    def __init__(self, config: Optional[Any] = None):
        """Initialize Cache Tool.

        Args:
            config: Optional investigator config object
        """
        super().__init__(config)
        self._cache_manager = None
        self._cache_type_enum = None

    async def initialize(self) -> None:
        """Initialize cache infrastructure."""
        try:
            from investigator.infrastructure.cache.cache_manager import get_cache_manager
            from investigator.infrastructure.cache.cache_types import CacheType

            if self.config is None:
                from investigator.config import get_config

                self.config = get_config()

            self._cache_manager = get_cache_manager()
            self._cache_type_enum = CacheType

            self._initialized = True
            logger.info("CacheTool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CacheTool: {e}")
            raise

    def _get_cache_type(self, cache_type_str: str):
        """Convert string cache type to enum.

        Args:
            cache_type_str: Cache type string (e.g., "llm_response")

        Returns:
            CacheType enum value

        Raises:
            ValueError: If invalid cache type
        """
        cache_type_str = cache_type_str.lower().strip()

        if cache_type_str in self.CACHE_TYPE_MAP:
            enum_name = self.CACHE_TYPE_MAP[cache_type_str]
            return getattr(self._cache_type_enum, enum_name)

        # Try direct enum access
        try:
            return getattr(self._cache_type_enum, cache_type_str.upper())
        except AttributeError:
            valid_types = list(self.CACHE_TYPE_MAP.keys())
            raise ValueError(f"Invalid cache type: {cache_type_str}. " f"Valid types: {valid_types}")

    async def execute(
        self,
        _exec_ctx: Optional[Dict[str, Any]] = None,
        action: str = "",
        cache_type: Optional[str] = None,
        key: Optional[Union[Dict, Tuple]] = None,
        value: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute cache operation.

        Args:
            action: Operation to perform:
                - "get": Get cached data
                - "set": Set cache entry
                - "exists": Check if entry exists
                - "delete": Delete entry
                - "delete_by_symbol": Delete all for a symbol
                - "clear_type": Clear all entries of a type
                - "get_stats": Get cache statistics
                - "get_recent_ops": Get recent operations
                - "validate": Validate cache entry
                - "invalidate_sec": Invalidate SEC-related cache
            cache_type: Cache type string
            key: Cache key (dict or tuple)
            value: Data to cache (for set action)
            symbol: Stock symbol for symbol-based operations
            **kwargs: Additional parameters

        Returns:
            ToolResult with operation result or error
        """
        try:
            await self.ensure_initialized()

            action = action.lower().strip()

            if action == "get":
                return await self._cache_get(cache_type, key)
            elif action == "set":
                return await self._cache_set(cache_type, key, value)
            elif action == "exists":
                return await self._cache_exists(cache_type, key)
            elif action == "delete":
                return await self._cache_delete(cache_type, key)
            elif action == "delete_by_symbol":
                return await self._delete_by_symbol(symbol)
            elif action == "clear_type":
                return await self._clear_type(cache_type)
            elif action == "get_stats":
                return await self._get_stats()
            elif action == "get_recent_ops":
                cache_type_obj = self._get_cache_type(cache_type) if cache_type else None
                return await self._get_recent_ops(cache_type_obj)
            elif action == "validate":
                return await self._validate_entry(cache_type, key)
            elif action == "invalidate_sec":
                filing_date = kwargs.get("filing_date")
                dry_run = kwargs.get("dry_run", True)
                return await self._invalidate_sec(symbol, filing_date, dry_run)
            elif action == "ping":
                return await self._ping()
            else:
                return ToolResult.create_failure(
                    f"Unknown action: {action}. Valid actions: "
                    "get, set, exists, delete, delete_by_symbol, clear_type, "
                    "get_stats, get_recent_ops, validate, invalidate_sec, ping"
                )

        except Exception as e:
            logger.error(f"CacheTool execute error: {e}")
            return ToolResult.create_failure(f"Cache operation failed: {str(e)}", metadata={"action": action})

    async def _cache_get(self, cache_type: str, key: Union[Dict, Tuple]) -> ToolResult:
        """Get data from cache.

        Args:
            cache_type: Cache type string
            key: Cache key

        Returns:
            ToolResult with cached data or miss indicator
        """
        try:
            if not cache_type:
                return ToolResult.create_failure("cache_type is required")
            if not key:
                return ToolResult.create_failure("key is required")

            cache_type_obj = self._get_cache_type(cache_type)

            # Use async method for non-blocking I/O
            data = await self._cache_manager.get_async(cache_type_obj, key)

            if data is not None:
                return ToolResult.create_success(output={"hit": True, "cache_type": cache_type, "key": key, "data": data}, metadata={"cache_hit": True}
                )
            else:
                return ToolResult.create_success(output={"hit": False, "cache_type": cache_type, "key": key, "data": None},
                    metadata={"cache_hit": False},
                )

        except ValueError as e:
            return ToolResult.create_failure(str(e))
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return ToolResult.create_failure(f"Cache get failed: {str(e)}")

    async def _cache_set(self, cache_type: str, key: Union[Dict, Tuple], value: Dict[str, Any]) -> ToolResult:
        """Set data in cache.

        Args:
            cache_type: Cache type string
            key: Cache key
            value: Data to cache

        Returns:
            ToolResult with operation status
        """
        try:
            if not cache_type:
                return ToolResult.create_failure("cache_type is required")
            if not key:
                return ToolResult.create_failure("key is required")
            if value is None:
                return ToolResult.create_failure("value is required")

            cache_type_obj = self._get_cache_type(cache_type)

            # Create standardized metadata
            metadata = self._cache_manager.create_cache_metadata(cache_type_obj, key)

            # Wrap value with metadata if not already present
            if "metadata" not in value:
                value = {"data": value, "metadata": metadata}
            else:
                # Merge metadata
                value["metadata"].update(metadata)

            # Use async method for non-blocking I/O
            success = await self._cache_manager.set_async(cache_type_obj, key, value)

            return ToolResult.create_success(output={
                    "success": success,
                    "cache_type": cache_type,
                    "key": key,
                },
                metadata={"write_success": success},
            )

        except ValueError as e:
            return ToolResult.create_failure(str(e))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return ToolResult.create_failure(f"Cache set failed: {str(e)}")

    async def _cache_exists(self, cache_type: str, key: Union[Dict, Tuple]) -> ToolResult:
        """Check if cache entry exists.

        Args:
            cache_type: Cache type string
            key: Cache key

        Returns:
            ToolResult with existence status
        """
        try:
            if not cache_type:
                return ToolResult.create_failure("cache_type is required")
            if not key:
                return ToolResult.create_failure("key is required")

            cache_type_obj = self._get_cache_type(cache_type)

            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(None, self._cache_manager.exists, cache_type_obj, key)

            return ToolResult.create_success(output={
                    "exists": exists,
                    "cache_type": cache_type,
                    "key": key,
                }
            )

        except ValueError as e:
            return ToolResult.create_failure(str(e))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return ToolResult.create_failure(f"Cache exists check failed: {str(e)}")

    async def _cache_delete(self, cache_type: str, key: Union[Dict, Tuple]) -> ToolResult:
        """Delete cache entry.

        Args:
            cache_type: Cache type string
            key: Cache key

        Returns:
            ToolResult with deletion status
        """
        try:
            if not cache_type:
                return ToolResult.create_failure("cache_type is required")
            if not key:
                return ToolResult.create_failure("key is required")

            cache_type_obj = self._get_cache_type(cache_type)

            loop = asyncio.get_event_loop()
            deleted = await loop.run_in_executor(None, self._cache_manager.delete, cache_type_obj, key)

            return ToolResult.create_success(output={
                    "deleted": deleted,
                    "cache_type": cache_type,
                    "key": key,
                }
            )

        except ValueError as e:
            return ToolResult.create_failure(str(e))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return ToolResult.create_failure(f"Cache delete failed: {str(e)}")

    async def _delete_by_symbol(self, symbol: str) -> ToolResult:
        """Delete all cache entries for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            ToolResult with deletion counts by cache type
        """
        try:
            if not symbol:
                return ToolResult.create_failure("symbol is required")

            symbol = symbol.upper().strip()

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._cache_manager.delete_by_symbol, symbol)

            total_deleted = sum(results.values())

            return ToolResult.create_success(output={"symbol": symbol, "total_deleted": total_deleted, "deleted_by_type": results}
            )

        except Exception as e:
            logger.error(f"Delete by symbol error: {e}")
            return ToolResult.create_failure(f"Delete by symbol failed: {str(e)}")

    async def _clear_type(self, cache_type: str) -> ToolResult:
        """Clear all entries for a cache type.

        Args:
            cache_type: Cache type string

        Returns:
            ToolResult with clear status
        """
        try:
            if not cache_type:
                return ToolResult.create_failure("cache_type is required")

            cache_type_obj = self._get_cache_type(cache_type)

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._cache_manager.clear_cache_type, cache_type_obj)

            return ToolResult.create_success(output={
                    "success": success,
                    "cache_type": cache_type,
                }
            )

        except ValueError as e:
            return ToolResult.create_failure(str(e))
        except Exception as e:
            logger.error(f"Clear cache type error: {e}")
            return ToolResult.create_failure(f"Clear cache type failed: {str(e)}")

    async def _get_stats(self) -> ToolResult:
        """Get cache performance statistics.

        Returns:
            ToolResult with cache statistics
        """
        try:
            loop = asyncio.get_event_loop()

            # Get both stats methods
            performance_stats = await loop.run_in_executor(None, self._cache_manager.get_performance_stats)

            general_stats = await loop.run_in_executor(None, self._cache_manager.get_stats)

            return ToolResult.create_success(output={"performance": performance_stats, "configuration": general_stats})

        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return ToolResult.create_failure(f"Get stats failed: {str(e)}")

    async def _get_recent_ops(self, cache_type=None) -> ToolResult:
        """Get recent cache operations for debugging.

        Args:
            cache_type: Optional cache type to filter by

        Returns:
            ToolResult with recent operations
        """
        try:
            loop = asyncio.get_event_loop()
            recent_ops = await loop.run_in_executor(None, self._cache_manager.get_recent_operations, cache_type, 20)

            return ToolResult.create_success(output={"recent_operations": recent_ops})

        except Exception as e:
            logger.error(f"Get recent ops error: {e}")
            return ToolResult.create_failure(f"Get recent ops failed: {str(e)}")

    async def _validate_entry(self, cache_type: str, key: Union[Dict, Tuple]) -> ToolResult:
        """Validate a cache entry for integrity.

        Args:
            cache_type: Cache type string
            key: Cache key

        Returns:
            ToolResult with validation status
        """
        try:
            if not cache_type:
                return ToolResult.create_failure("cache_type is required")
            if not key:
                return ToolResult.create_failure("key is required")

            cache_type_obj = self._get_cache_type(cache_type)

            # Get the entry first
            data = await self._cache_manager.get_async(cache_type_obj, key)

            if data is None:
                return ToolResult.create_success(output={"valid": False, "exists": False, "issues": ["Entry not found"]})

            # Validate the entry
            loop = asyncio.get_event_loop()
            is_valid, issues = await loop.run_in_executor(
                None, self._cache_manager.validate_cache_entry, data, cache_type_obj, True  # strict mode
            )

            return ToolResult.create_success(output={
                    "valid": is_valid,
                    "exists": True,
                    "issues": issues,
                    "cache_type": cache_type,
                    "key": key,
                }
            )

        except ValueError as e:
            return ToolResult.create_failure(str(e))
        except Exception as e:
            logger.error(f"Validate entry error: {e}")
            return ToolResult.create_failure(f"Validate entry failed: {str(e)}")

    async def _invalidate_sec(self, symbol: str, filing_date: str, dry_run: bool = True) -> ToolResult:
        """Invalidate SEC-related cache entries after new filing.

        Args:
            symbol: Stock ticker
            filing_date: New filing date (ISO format)
            dry_run: If True, only report what would be invalidated

        Returns:
            ToolResult with invalidation summary
        """
        try:
            if not symbol:
                return ToolResult.create_failure("symbol is required")
            if not filing_date:
                return ToolResult.create_failure("filing_date is required")

            symbol = symbol.upper().strip()

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._cache_manager.invalidate_on_sec_update, symbol, filing_date, dry_run
            )

            return ToolResult.create_success(output=result, metadata={"dry_run": dry_run})

        except Exception as e:
            logger.error(f"Invalidate SEC cache error: {e}")
            return ToolResult.create_failure(f"Invalidate SEC cache failed: {str(e)}")

    async def _ping(self) -> ToolResult:
        """Health check for cache system.

        Returns:
            ToolResult with health status
        """
        try:
            is_healthy = await self._cache_manager.ping()

            return ToolResult.create_success(output={"healthy": is_healthy, "status": "operational" if is_healthy else "degraded"}
            )

        except Exception as e:
            logger.error(f"Cache ping error: {e}")
            return ToolResult.create_success(output={"healthy": False, "status": "error", "error": str(e)})

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Cache Tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get",
                        "set",
                        "exists",
                        "delete",
                        "delete_by_symbol",
                        "clear_type",
                        "get_stats",
                        "get_recent_ops",
                        "validate",
                        "invalidate_sec",
                        "ping",
                    ],
                    "description": "Cache operation to perform",
                },
                "cache_type": {
                    "type": "string",
                    "enum": list(self.CACHE_TYPE_MAP.keys()),
                    "description": "Type of cache",
                },
                "key": {"type": "object", "description": "Cache key (dict with symbol, llm_type, etc.)"},
                "value": {"type": "object", "description": "Data to cache (for set action)"},
                "symbol": {"type": "string", "description": "Symbol for symbol-based operations"},
                "filing_date": {"type": "string", "description": "Filing date for SEC invalidation (ISO format)"},
                "dry_run": {"type": "boolean", "description": "Preview mode for invalidation", "default": True},
            },
            "required": ["action"],
        }
