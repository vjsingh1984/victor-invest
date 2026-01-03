#!/usr/bin/env python3
"""
InvestiGator - Parquet Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Parquet cache handler for efficient storage of tabular data with compression
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from .cache_base import CacheStorageHandler
from .cache_types import CacheType

logger = logging.getLogger(__name__)


class ParquetCacheStorageHandler(CacheStorageHandler):
    """Cache handler for storing data in Parquet format with gzip compression"""

    def __init__(self, cache_type: CacheType, base_path: Path, priority: int = 10, config=None):
        """
        Initialize Parquet cache handler

        Args:
            cache_type: Type of cache (TECHNICAL_DATA, SUBMISSION_DATA, etc.)
            base_path: Base directory for cache storage
            priority: Handler priority (higher = checked first)
            config: Optional Config object with parquet settings
        """
        super().__init__(cache_type, priority)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Determine parquet engine availability
        self._parquet_engine: Optional[str] = None
        for engine in ("pyarrow", "fastparquet"):
            try:
                __import__(engine)
                self._parquet_engine = engine
                break
            except ImportError:
                continue
        self._storage_format = "parquet" if self._parquet_engine else "pickle"
        if not self._parquet_engine:
            logger.info("Parquet engine not available; using pickle-based caching for %s", cache_type.value)

        # Get parquet config from Config object or use defaults
        if config and hasattr(config, "parquet"):
            self.parquet_config = config.parquet
        else:
            # Default configuration - uniform gzip compression
            from config.config import ParquetConfig

            self.parquet_config = ParquetConfig()

    def _normalize_key(self, key: Union[Tuple, Dict]) -> Dict[str, Any]:
        """Normalize cache key to dictionary format"""
        if isinstance(key, tuple):
            # Handle different tuple formats
            if len(key) >= 3:
                # Format: (symbol, data_type, timeframe) - e.g., ('AAPL', 'technical_data', '365d')
                return {"symbol": key[0], "data_type": key[1], "timeframe": key[2]}
            elif len(key) >= 2:
                # Format: (symbol, timeframe) - e.g., ('AAPL', 'recent_365')
                # or (symbol, data_type) - e.g., ('AAPL', 'technical_data')
                if key[1].startswith("recent_") or key[1].endswith("d"):
                    # It's a timeframe
                    return {"symbol": key[0], "data_type": "technical_data", "timeframe": key[1]}
                else:
                    # It's a data type
                    return {"symbol": key[0], "data_type": key[1], "timeframe": "default"}
            else:
                return {"symbol": key[0], "data_type": "technical_data", "timeframe": "default"}
        return key

    def _get_file_path(self, key_dict: Dict[str, Any]) -> Path:
        """Generate file path based on cache key (only for TECHNICAL_DATA)"""
        # Stock-specific cache types (TECHNICAL_DATA only)
        symbol = key_dict.get("symbol", "unknown")
        data_type = key_dict.get("data_type", "data")
        timeframe = key_dict.get("timeframe", "default")

        # Create subdirectory for symbol
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if self._storage_format == "parquet":
            filename = f"{data_type}_{timeframe}.parquet.gz"
        else:
            filename = f"{data_type}_{timeframe}.pkl.gz"
        return symbol_dir / filename

    def _get_metadata_path(self, parquet_path: Path) -> Path:
        """Get metadata file path for a parquet file"""
        return parquet_path.with_suffix(".meta.json")

    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Retrieve data from parquet cache"""
        if self.priority < 0:
            return None  # Skip lookup for negative priority

        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)

            if file_path.exists() and metadata_path.exists():
                # Read metadata
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Read parquet data
                if self._storage_format == "parquet":
                    df = pd.read_parquet(file_path, engine=self._parquet_engine or "auto")
                else:
                    df = pd.read_pickle(file_path)

                # Convert DataFrame to dict for compatibility
                data = {
                    "dataframe": df,
                    "data": df.to_dict("records"),
                    "metadata": metadata,
                    "cache_info": {
                        "cached_at": metadata.get("cached_at"),
                        "cache_type": self.cache_type.value,
                        "compression": metadata.get("compression", "gzip"),
                        "records": len(df),
                    },
                }

                logger.debug(
                    f"✅ Parquet cache HIT: {file_path} ({len(df)} records, {metadata.get('compression', 'unknown')} compression)"
                )
                return data

            logger.debug(
                f"❌ Parquet cache MISS: {file_path} (exists: {file_path.exists()}, meta exists: {metadata_path.exists()})"
            )
            return None

        except Exception as e:
            logger.error(f"Error reading from parquet cache: {e}")
            return None

    def set(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """Store data in parquet cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)

            # Extract DataFrame or create from data
            if "dataframe" in value and isinstance(value["dataframe"], pd.DataFrame):
                df = value["dataframe"]
            elif "data" in value:
                # Convert data to DataFrame
                if isinstance(value["data"], list) and len(value["data"]) > 0:
                    df = pd.DataFrame(value["data"])
                elif isinstance(value["data"], dict):
                    df = pd.DataFrame([value["data"]])
                else:
                    logger.debug(f"Skipping parquet cache for non-tabular data: {type(value.get('data'))}")
                    return False
            elif isinstance(value, pd.DataFrame):
                df = value
            else:
                # Skip parquet cache for non-DataFrame data
                logger.debug("Skipping parquet cache - no suitable DataFrame data found")
                return False

            # Ensure datetime columns are properly formatted
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        # Try to convert to datetime
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass  # Keep as is if conversion fails

            # Save DataFrame to parquet using configuration
            write_kwargs = self.parquet_config.get_write_kwargs()

            # Remove 'engine' from write_kwargs to avoid duplicate parameter error
            # We'll pass it directly to to_parquet instead
            write_kwargs_clean = {k: v for k, v in write_kwargs.items() if k != "engine"}

            # Check if the engine is available
            try:
                if self._storage_format == "parquet":
                    df.to_parquet(
                        file_path, engine=self._parquet_engine or self.parquet_config.engine, **write_kwargs_clean
                    )
                else:
                    df.to_pickle(file_path, compression="gzip")
            except Exception as e:
                logger.error(f"Error writing to {self._storage_format} cache: {e}")
                return False

            # Save metadata
            metadata = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "cache_key": key_dict,
                "cache_type": self.cache_type.value,
                "engine": self._parquet_engine or self.parquet_config.engine,
                "compression": (
                    self.parquet_config.compression
                    if self.parquet_config.engine == "fastparquet"
                    else self.parquet_config.pyarrow_compression if self._parquet_engine else "gzip"
                ),
                "storage_format": self._storage_format,
                "records": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "file_size_bytes": file_path.stat().st_size,
                "original_metadata": value.get("metadata", {}),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ Cached to parquet: {file_path} ({len(df)} records, {file_path.stat().st_size:,} bytes, {self.parquet_config.compression} compression)"
            )
            return True

        except Exception as e:
            logger.error(f"Error writing to parquet cache: {e}")
            return False

    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in parquet cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)

            exists = file_path.exists() and metadata_path.exists()

            if exists:
                # Get file size for logging
                file_size = file_path.stat().st_size
                logger.debug(f"📁 Parquet cache EXISTS: {file_path} ({file_size:,} bytes)")
            else:
                logger.debug(
                    f"📂 Parquet cache NOT EXISTS: {file_path} (file: {file_path.exists()}, meta: {metadata_path.exists()})"
                )

            return exists
        except Exception as e:
            logger.error(f"💥 Error checking parquet cache existence: {e}")
            return False

    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete data from parquet cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)

            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True

            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True

            if deleted:
                logger.debug(f"Deleted from parquet cache: {file_path}")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting from parquet cache: {e}")
            return False

    def delete_by_symbol(self, symbol: str) -> int:
        """
        Optimized symbol-based deletion for Parquet cache using symbol directories.
        Deletes entire symbol directory for clean isolation.

        Structure: data/technical_cache/{symbol}/

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            symbol = symbol.upper()  # Normalize to uppercase

            # Delete symbol-specific directory (already using symbol directories)
            symbol_dir = self.base_path / symbol
            if symbol_dir.exists() and symbol_dir.is_dir():
                import shutil

                try:
                    # Count files before deletion for accurate reporting
                    file_count = sum(1 for _ in symbol_dir.rglob("*") if _.is_file())
                    shutil.rmtree(symbol_dir)
                    deleted_count = file_count
                    logger.info(f"Deleted symbol directory with {file_count} files: {symbol_dir}")
                except Exception as e:
                    logger.error(f"Error deleting symbol directory {symbol_dir}: {e}")
            else:
                logger.debug(f"Symbol directory does not exist: {symbol_dir}")

            logger.info(
                f"Symbol cleanup [{self.__class__.__name__}]: Deleted {deleted_count} parquet files for symbol {symbol}"
            )
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting by symbol '{symbol}' from parquet cache: {e}")
            return 0

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all cache entries matching a pattern (legacy method)"""
        try:
            import fnmatch

            deleted_count = 0

            # Walk through cache directory and find matching files
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                    try:
                        # Delete both parquet file and metadata
                        if file_path.suffix in [".parquet", ".parquet.gz"]:
                            metadata_path = self._get_metadata_path(file_path)
                            if metadata_path.exists():
                                metadata_path.unlink()

                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted parquet file matching pattern '{pattern}': {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting parquet file {file_path}: {e}")

            logger.info(f"Deleted {deleted_count} parquet files matching pattern '{pattern}'")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting by pattern from parquet cache: {e}")
            return 0

    def clear_all(self) -> bool:
        """Clear all data from parquet cache"""
        try:
            import shutil

            deleted_count = 0

            # Remove all files in cache directory but keep the directory structure
            for item in self.base_path.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_count += 1

            logger.info(f"Cleared all parquet cache data ({deleted_count} items)")
            return True

        except Exception as e:
            logger.error(f"Error clearing parquet cache: {e}")
            return False

    def get_cache_info(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Get cache metadata without loading the data"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    return json.load(f)

            return None

        except Exception as e:
            logger.error(f"Error reading parquet cache metadata: {e}")
            return None
