#!/usr/bin/env python3
"""
InvestiGator - Industry Metrics Cache (Hybrid Architecture)
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Hybrid cache for industry-specific metrics with two-tier storage:
1. Industry-level cache: Shared benchmarks, peer statistics, cycle indicators
2. Symbol-level cache: Company-specific extracted metrics

Directory Structure:
    data/industry_metrics_cache/
    ├── _index.parquet           # Symbol index
    ├── _industry_index.parquet  # Industry index
    ├── industries/              # Shared industry data
    │   ├── semiconductors.parquet.gz
    │   ├── banks.parquet.gz
    │   └── insurance.parquet.gz
    └── symbols/                 # Company-specific metrics
        ├── NVDA.parquet.gz
        ├── AMD.parquet.gz
        └── INTC.parquet.gz

Usage:
    from investigator.infrastructure.cache.industry_metrics_cache import IndustryMetricsCache

    cache = IndustryMetricsCache()

    # Symbol-level (company-specific)
    cache.set(symbol="NVDA", industry="Semiconductors", metrics=metrics_dict)
    entry = cache.get("NVDA")

    # Industry-level (shared benchmarks)
    benchmarks = cache.get_industry_benchmarks("Semiconductors")
    cache.compute_and_cache_industry_benchmarks("Semiconductors")
"""

import json
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .cache_types import CacheType

logger = logging.getLogger(__name__)


@dataclass
class IndustryMetricsCacheEntry:
    """Cache entry for symbol-specific metrics"""

    symbol: str
    industry: str
    sector: Optional[str]
    dataset_name: str
    dataset_version: str
    quality: str  # excellent, good, fair, poor
    coverage: float
    metrics: Dict[str, Any]
    adjustments: List[Dict[str, Any]]
    tier_weights: Optional[Dict[str, int]]
    warnings: List[str]
    metadata: Dict[str, Any]
    cached_at: str
    expires_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndustryMetricsCacheEntry":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class IndustryBenchmarksCacheEntry:
    """Cache entry for industry-level shared benchmarks"""

    industry: str
    sector: Optional[str]
    symbol_count: int  # Number of symbols used to compute benchmarks
    symbols_included: List[str]  # Symbols used in computation

    # Peer statistics for numeric metrics
    # Format: {"metric_name": {"median": x, "mean": y, "std": z, "min": a, "max": b, "p25": c, "p75": d}}
    peer_statistics: Dict[str, Dict[str, float]]

    # Industry benchmarks (thresholds, typical ranges)
    benchmarks: Dict[str, Any]

    # Recommended tier weights for this industry
    tier_weights: Dict[str, int]

    # Cycle/market position indicators
    cycle_indicators: Dict[str, Any]

    # Metadata
    dataset_name: str
    dataset_version: str
    cached_at: str
    expires_at: Optional[str] = None
    computation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndustryBenchmarksCacheEntry":
        """Create from dictionary"""
        # Handle missing fields with defaults
        data.setdefault("computation_notes", [])
        return cls(**data)


class IndustryMetricsStorageBackend(ABC):
    """Abstract base for industry metrics storage backends"""

    # Symbol-level methods
    @abstractmethod
    def get(self, symbol: str) -> Optional[IndustryMetricsCacheEntry]:
        """Retrieve cached metrics for a symbol"""
        pass

    @abstractmethod
    def set(self, entry: IndustryMetricsCacheEntry) -> bool:
        """Store metrics for a symbol"""
        pass

    @abstractmethod
    def exists(self, symbol: str) -> bool:
        """Check if metrics exist for a symbol"""
        pass

    @abstractmethod
    def delete(self, symbol: str) -> bool:
        """Delete metrics for a symbol"""
        pass

    @abstractmethod
    def get_by_industry(self, industry: str) -> List[IndustryMetricsCacheEntry]:
        """Get all cached metrics for an industry"""
        pass

    @abstractmethod
    def list_symbols(self) -> List[str]:
        """List all cached symbols"""
        pass

    # Industry-level methods
    @abstractmethod
    def get_industry_benchmarks(self, industry: str) -> Optional[IndustryBenchmarksCacheEntry]:
        """Get cached industry-level benchmarks"""
        pass

    @abstractmethod
    def set_industry_benchmarks(self, entry: IndustryBenchmarksCacheEntry) -> bool:
        """Store industry-level benchmarks"""
        pass

    @abstractmethod
    def list_industries(self) -> List[str]:
        """List all industries with cached benchmarks"""
        pass

    # Common methods
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all cached metrics"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class ParquetIndustryMetricsBackend(IndustryMetricsStorageBackend):
    """Parquet-based storage backend with hybrid industry/symbol structure"""

    def __init__(self, base_path: Path, compression: str = "gzip"):
        """
        Initialize Parquet backend with hybrid directory structure

        Args:
            base_path: Base directory for cache
            compression: Compression algorithm (gzip, snappy, etc.)
        """
        self.base_path = Path(base_path)
        self.symbols_path = self.base_path / "symbols"
        self.industries_path = self.base_path / "industries"

        # Create directory structure
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.symbols_path.mkdir(parents=True, exist_ok=True)
        self.industries_path.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self._symbol_index_path = self.base_path / "_index.parquet"
        self._industry_index_path = self.base_path / "_industry_index.parquet"

        self._ensure_indexes()

        # Determine parquet engine
        self._engine: Optional[str] = None
        for engine in ("pyarrow", "fastparquet"):
            try:
                __import__(engine)
                self._engine = engine
                break
            except ImportError:
                continue

        if not self._engine:
            logger.warning("No parquet engine available, falling back to pickle")

    def _ensure_indexes(self):
        """Ensure index files exist"""
        # Symbol index
        if not self._symbol_index_path.exists():
            df = pd.DataFrame(
                columns=[
                    "symbol",
                    "industry",
                    "sector",
                    "dataset_name",
                    "quality",
                    "coverage",
                    "cached_at",
                    "file_path",
                ]
            )
            df.to_parquet(self._symbol_index_path, engine="pyarrow", compression="gzip")

        # Industry index
        if not self._industry_index_path.exists():
            df = pd.DataFrame(columns=["industry", "sector", "symbol_count", "dataset_name", "cached_at", "file_path"])
            df.to_parquet(self._industry_index_path, engine="pyarrow", compression="gzip")

    # ========== Symbol-Level Methods ==========

    def _get_symbol_file_path(self, symbol: str) -> Path:
        """Get file path for a symbol"""
        return self.symbols_path / f"{symbol.upper()}.parquet.gz"

    def _update_symbol_index(self, entry: IndustryMetricsCacheEntry, file_path: Path):
        """Update the symbol index with a new entry"""
        try:
            index_df = pd.read_parquet(self._symbol_index_path)
            index_df = index_df[index_df["symbol"] != entry.symbol.upper()]

            new_row = pd.DataFrame(
                [
                    {
                        "symbol": entry.symbol.upper(),
                        "industry": entry.industry,
                        "sector": entry.sector,
                        "dataset_name": entry.dataset_name,
                        "quality": entry.quality,
                        "coverage": entry.coverage,
                        "cached_at": entry.cached_at,
                        "file_path": str(file_path),
                    }
                ]
            )

            if len(index_df) == 0:
                index_df = new_row
            else:
                index_df = pd.concat([index_df, new_row], ignore_index=True)
            index_df.to_parquet(self._symbol_index_path, engine="pyarrow", compression="gzip")

        except Exception as e:
            logger.warning(f"Failed to update symbol index: {e}")

    def _remove_from_symbol_index(self, symbol: str):
        """Remove a symbol from the index"""
        try:
            index_df = pd.read_parquet(self._symbol_index_path)
            index_df = index_df[index_df["symbol"] != symbol.upper()]
            index_df.to_parquet(self._symbol_index_path, engine="pyarrow", compression="gzip")
        except Exception as e:
            logger.warning(f"Failed to remove from symbol index: {e}")

    def get(self, symbol: str) -> Optional[IndustryMetricsCacheEntry]:
        """Retrieve cached metrics for a symbol"""
        file_path = self._get_symbol_file_path(symbol)

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            if len(df) == 0:
                return None

            row = df.iloc[0].to_dict()

            # Parse JSON fields
            for fld in ["metrics", "adjustments", "tier_weights", "warnings", "metadata"]:
                if fld in row and isinstance(row[fld], str):
                    row[fld] = json.loads(row[fld])

            return IndustryMetricsCacheEntry.from_dict(row)

        except Exception as e:
            logger.error(f"Error reading industry metrics for {symbol}: {e}")
            return None

    def set(self, entry: IndustryMetricsCacheEntry) -> bool:
        """Store metrics for a symbol"""
        file_path = self._get_symbol_file_path(entry.symbol)

        try:
            data = entry.to_dict()
            for fld in ["metrics", "adjustments", "tier_weights", "warnings", "metadata"]:
                if fld in data and not isinstance(data[fld], str):
                    data[fld] = json.dumps(data[fld])

            df = pd.DataFrame([data])
            df.to_parquet(file_path, engine=self._engine or "pyarrow", compression=self.compression)

            self._update_symbol_index(entry, file_path)

            logger.debug(f"Cached symbol metrics for {entry.symbol} ({entry.industry})")
            return True

        except Exception as e:
            logger.error(f"Error caching symbol metrics for {entry.symbol}: {e}")
            return False

    def exists(self, symbol: str) -> bool:
        """Check if metrics exist for a symbol"""
        return self._get_symbol_file_path(symbol).exists()

    def delete(self, symbol: str) -> bool:
        """Delete metrics for a symbol"""
        file_path = self._get_symbol_file_path(symbol)

        try:
            if file_path.exists():
                file_path.unlink()
                self._remove_from_symbol_index(symbol)
                logger.debug(f"Deleted symbol cache for {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting symbol cache for {symbol}: {e}")
            return False

    def get_by_industry(self, industry: str) -> List[IndustryMetricsCacheEntry]:
        """Get all cached metrics for an industry"""
        try:
            index_df = pd.read_parquet(self._symbol_index_path)
            industry_rows = index_df[index_df["industry"].str.lower() == industry.lower()]

            entries = []
            for _, row in industry_rows.iterrows():
                entry = self.get(row["symbol"])
                if entry:
                    entries.append(entry)

            return entries

        except Exception as e:
            logger.error(f"Error getting metrics for industry {industry}: {e}")
            return []

    def list_symbols(self) -> List[str]:
        """List all cached symbols"""
        try:
            index_df = pd.read_parquet(self._symbol_index_path)
            return index_df["symbol"].tolist()
        except Exception:
            return []

    # ========== Industry-Level Methods ==========

    def _get_industry_file_path(self, industry: str) -> Path:
        """Get file path for industry benchmarks"""
        # Normalize industry name for filename
        safe_name = industry.lower().replace(" ", "_").replace("&", "and").replace("-", "_")
        return self.industries_path / f"{safe_name}.parquet.gz"

    def _update_industry_index(self, entry: IndustryBenchmarksCacheEntry, file_path: Path):
        """Update the industry index"""
        try:
            index_df = pd.read_parquet(self._industry_index_path)
            index_df = index_df[index_df["industry"].str.lower() != entry.industry.lower()]

            new_row = pd.DataFrame(
                [
                    {
                        "industry": entry.industry,
                        "sector": entry.sector,
                        "symbol_count": entry.symbol_count,
                        "dataset_name": entry.dataset_name,
                        "cached_at": entry.cached_at,
                        "file_path": str(file_path),
                    }
                ]
            )

            if len(index_df) == 0:
                index_df = new_row
            else:
                index_df = pd.concat([index_df, new_row], ignore_index=True)
            index_df.to_parquet(self._industry_index_path, engine="pyarrow", compression="gzip")

        except Exception as e:
            logger.warning(f"Failed to update industry index: {e}")

    def get_industry_benchmarks(self, industry: str) -> Optional[IndustryBenchmarksCacheEntry]:
        """Get cached industry-level benchmarks"""
        file_path = self._get_industry_file_path(industry)

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            if len(df) == 0:
                return None

            row = df.iloc[0].to_dict()

            # Parse JSON fields
            for fld in [
                "symbols_included",
                "peer_statistics",
                "benchmarks",
                "tier_weights",
                "cycle_indicators",
                "computation_notes",
            ]:
                if fld in row and isinstance(row[fld], str):
                    row[fld] = json.loads(row[fld])

            return IndustryBenchmarksCacheEntry.from_dict(row)

        except Exception as e:
            logger.error(f"Error reading industry benchmarks for {industry}: {e}")
            return None

    def set_industry_benchmarks(self, entry: IndustryBenchmarksCacheEntry) -> bool:
        """Store industry-level benchmarks"""
        file_path = self._get_industry_file_path(entry.industry)

        try:
            data = entry.to_dict()
            for fld in [
                "symbols_included",
                "peer_statistics",
                "benchmarks",
                "tier_weights",
                "cycle_indicators",
                "computation_notes",
            ]:
                if fld in data and not isinstance(data[fld], str):
                    data[fld] = json.dumps(data[fld])

            df = pd.DataFrame([data])
            df.to_parquet(file_path, engine=self._engine or "pyarrow", compression=self.compression)

            self._update_industry_index(entry, file_path)

            logger.info(f"Cached industry benchmarks for {entry.industry} " f"({entry.symbol_count} symbols)")
            return True

        except Exception as e:
            logger.error(f"Error caching industry benchmarks for {entry.industry}: {e}")
            return False

    def list_industries(self) -> List[str]:
        """List all industries with cached benchmarks"""
        try:
            index_df = pd.read_parquet(self._industry_index_path)
            return index_df["industry"].tolist()
        except Exception:
            return []

    # ========== Common Methods ==========

    def clear_all(self) -> bool:
        """Clear all cached metrics (both symbols and industries)"""
        try:
            import shutil

            # Clear symbols
            if self.symbols_path.exists():
                shutil.rmtree(self.symbols_path)
                self.symbols_path.mkdir(parents=True, exist_ok=True)

            # Clear industries
            if self.industries_path.exists():
                shutil.rmtree(self.industries_path)
                self.industries_path.mkdir(parents=True, exist_ok=True)

            # Recreate indexes
            self._symbol_index_path.unlink(missing_ok=True)
            self._industry_index_path.unlink(missing_ok=True)
            self._ensure_indexes()

            logger.info("Cleared all industry metrics cache (symbols and industries)")
            return True

        except Exception as e:
            logger.error(f"Error clearing industry metrics cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            symbol_df = pd.read_parquet(self._symbol_index_path)
            industry_df = pd.read_parquet(self._industry_index_path)

            # Calculate sizes
            symbol_size = sum(f.stat().st_size for f in self.symbols_path.glob("*.parquet.gz"))
            industry_size = sum(f.stat().st_size for f in self.industries_path.glob("*.parquet.gz"))

            return {
                "backend": "parquet_hybrid",
                "symbols": {
                    "count": len(symbol_df),
                    "by_industry": symbol_df["industry"].value_counts().to_dict() if len(symbol_df) > 0 else {},
                    "by_quality": symbol_df["quality"].value_counts().to_dict() if len(symbol_df) > 0 else {},
                    "size_bytes": symbol_size,
                    "size_mb": round(symbol_size / (1024 * 1024), 2),
                },
                "industries": {
                    "count": len(industry_df),
                    "names": industry_df["industry"].tolist() if len(industry_df) > 0 else [],
                    "size_bytes": industry_size,
                    "size_mb": round(industry_size / (1024 * 1024), 2),
                },
                "total_size_mb": round((symbol_size + industry_size) / (1024 * 1024), 2),
                "cache_path": str(self.base_path),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"backend": "parquet_hybrid", "error": str(e)}


class PostgreSQLIndustryMetricsBackend(IndustryMetricsStorageBackend):
    """PostgreSQL-based storage backend for industry metrics (optional)"""

    SYMBOLS_TABLE = "industry_metrics_cache"
    INDUSTRIES_TABLE = "industry_benchmarks_cache"

    def __init__(self):
        """Initialize PostgreSQL backend"""
        from investigator.infrastructure.database.db import DatabaseManager

        self.db_manager = DatabaseManager()
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure tables exist"""
        from sqlalchemy import text

        # Symbols table
        create_symbols_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.SYMBOLS_TABLE} (
            symbol VARCHAR(20) PRIMARY KEY,
            industry VARCHAR(100),
            sector VARCHAR(100),
            dataset_name VARCHAR(100),
            dataset_version VARCHAR(20),
            quality VARCHAR(20),
            coverage FLOAT,
            metrics JSONB,
            adjustments JSONB,
            tier_weights JSONB,
            warnings JSONB,
            metadata JSONB,
            cached_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_imc_industry ON {self.SYMBOLS_TABLE}(industry);
        """

        # Industries table
        create_industries_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.INDUSTRIES_TABLE} (
            industry VARCHAR(100) PRIMARY KEY,
            sector VARCHAR(100),
            symbol_count INT,
            symbols_included JSONB,
            peer_statistics JSONB,
            benchmarks JSONB,
            tier_weights JSONB,
            cycle_indicators JSONB,
            dataset_name VARCHAR(100),
            dataset_version VARCHAR(20),
            cached_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            computation_notes JSONB,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        try:
            with self.db_manager.get_session() as session:
                session.execute(text(create_symbols_sql))
                session.execute(text(create_industries_sql))
                session.commit()
        except Exception as e:
            logger.warning(f"Could not ensure cache tables: {e}")

    # Symbol-level methods (abbreviated - same pattern as before)
    def get(self, symbol: str) -> Optional[IndustryMetricsCacheEntry]:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    text(f"SELECT * FROM {self.SYMBOLS_TABLE} WHERE symbol = :symbol"), {"symbol": symbol.upper()}
                ).fetchone()
                if not result:
                    return None
                row = dict(result._mapping)
                if row.get("cached_at"):
                    row["cached_at"] = row["cached_at"].isoformat()
                if row.get("expires_at"):
                    row["expires_at"] = row["expires_at"].isoformat()
                row.pop("updated_at", None)
                return IndustryMetricsCacheEntry.from_dict(row)
        except Exception as e:
            logger.error(f"Error retrieving metrics for {symbol}: {e}")
            return None

    def set(self, entry: IndustryMetricsCacheEntry) -> bool:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                upsert_sql = text(
                    f"""
                INSERT INTO {self.SYMBOLS_TABLE}
                    (symbol, industry, sector, dataset_name, dataset_version,
                     quality, coverage, metrics, adjustments, tier_weights,
                     warnings, metadata, cached_at, expires_at)
                VALUES
                    (:symbol, :industry, :sector, :dataset_name, :dataset_version,
                     :quality, :coverage, :metrics, :adjustments, :tier_weights,
                     :warnings, :metadata, :cached_at, :expires_at)
                ON CONFLICT (symbol) DO UPDATE SET
                    industry = EXCLUDED.industry, sector = EXCLUDED.sector,
                    dataset_name = EXCLUDED.dataset_name, dataset_version = EXCLUDED.dataset_version,
                    quality = EXCLUDED.quality, coverage = EXCLUDED.coverage,
                    metrics = EXCLUDED.metrics, adjustments = EXCLUDED.adjustments,
                    tier_weights = EXCLUDED.tier_weights, warnings = EXCLUDED.warnings,
                    metadata = EXCLUDED.metadata, cached_at = EXCLUDED.cached_at,
                    expires_at = EXCLUDED.expires_at, updated_at = NOW()
                """
                )
                session.execute(
                    upsert_sql,
                    {
                        "symbol": entry.symbol.upper(),
                        "industry": entry.industry,
                        "sector": entry.sector,
                        "dataset_name": entry.dataset_name,
                        "dataset_version": entry.dataset_version,
                        "quality": entry.quality,
                        "coverage": entry.coverage,
                        "metrics": json.dumps(entry.metrics),
                        "adjustments": json.dumps(entry.adjustments),
                        "tier_weights": json.dumps(entry.tier_weights) if entry.tier_weights else None,
                        "warnings": json.dumps(entry.warnings),
                        "metadata": json.dumps(entry.metadata),
                        "cached_at": entry.cached_at,
                        "expires_at": entry.expires_at,
                    },
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error caching metrics for {entry.symbol}: {e}")
            return False

    def exists(self, symbol: str) -> bool:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    text(f"SELECT 1 FROM {self.SYMBOLS_TABLE} WHERE symbol = :symbol"), {"symbol": symbol.upper()}
                ).fetchone()
                return result is not None
        except Exception:
            return False

    def delete(self, symbol: str) -> bool:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    text(f"DELETE FROM {self.SYMBOLS_TABLE} WHERE symbol = :symbol"), {"symbol": symbol.upper()}
                )
                session.commit()
                return result.rowcount > 0
        except Exception:
            return False

    def get_by_industry(self, industry: str) -> List[IndustryMetricsCacheEntry]:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                results = session.execute(
                    text(f"SELECT * FROM {self.SYMBOLS_TABLE} WHERE LOWER(industry) = LOWER(:industry)"),
                    {"industry": industry},
                ).fetchall()
                entries = []
                for result in results:
                    row = dict(result._mapping)
                    if row.get("cached_at"):
                        row["cached_at"] = row["cached_at"].isoformat()
                    if row.get("expires_at"):
                        row["expires_at"] = row["expires_at"].isoformat()
                    row.pop("updated_at", None)
                    entries.append(IndustryMetricsCacheEntry.from_dict(row))
                return entries
        except Exception:
            return []

    def list_symbols(self) -> List[str]:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                results = session.execute(text(f"SELECT symbol FROM {self.SYMBOLS_TABLE} ORDER BY symbol")).fetchall()
                return [r[0] for r in results]
        except Exception:
            return []

    # Industry-level methods
    def get_industry_benchmarks(self, industry: str) -> Optional[IndustryBenchmarksCacheEntry]:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    text(f"SELECT * FROM {self.INDUSTRIES_TABLE} WHERE LOWER(industry) = LOWER(:industry)"),
                    {"industry": industry},
                ).fetchone()
                if not result:
                    return None
                row = dict(result._mapping)
                if row.get("cached_at"):
                    row["cached_at"] = row["cached_at"].isoformat()
                if row.get("expires_at"):
                    row["expires_at"] = row["expires_at"].isoformat()
                row.pop("updated_at", None)
                return IndustryBenchmarksCacheEntry.from_dict(row)
        except Exception as e:
            logger.error(f"Error retrieving benchmarks for {industry}: {e}")
            return None

    def set_industry_benchmarks(self, entry: IndustryBenchmarksCacheEntry) -> bool:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                upsert_sql = text(
                    f"""
                INSERT INTO {self.INDUSTRIES_TABLE}
                    (industry, sector, symbol_count, symbols_included, peer_statistics,
                     benchmarks, tier_weights, cycle_indicators, dataset_name,
                     dataset_version, cached_at, expires_at, computation_notes)
                VALUES
                    (:industry, :sector, :symbol_count, :symbols_included, :peer_statistics,
                     :benchmarks, :tier_weights, :cycle_indicators, :dataset_name,
                     :dataset_version, :cached_at, :expires_at, :computation_notes)
                ON CONFLICT (industry) DO UPDATE SET
                    sector = EXCLUDED.sector, symbol_count = EXCLUDED.symbol_count,
                    symbols_included = EXCLUDED.symbols_included,
                    peer_statistics = EXCLUDED.peer_statistics,
                    benchmarks = EXCLUDED.benchmarks, tier_weights = EXCLUDED.tier_weights,
                    cycle_indicators = EXCLUDED.cycle_indicators,
                    dataset_name = EXCLUDED.dataset_name, dataset_version = EXCLUDED.dataset_version,
                    cached_at = EXCLUDED.cached_at, expires_at = EXCLUDED.expires_at,
                    computation_notes = EXCLUDED.computation_notes, updated_at = NOW()
                """
                )
                session.execute(
                    upsert_sql,
                    {
                        "industry": entry.industry,
                        "sector": entry.sector,
                        "symbol_count": entry.symbol_count,
                        "symbols_included": json.dumps(entry.symbols_included),
                        "peer_statistics": json.dumps(entry.peer_statistics),
                        "benchmarks": json.dumps(entry.benchmarks),
                        "tier_weights": json.dumps(entry.tier_weights),
                        "cycle_indicators": json.dumps(entry.cycle_indicators),
                        "dataset_name": entry.dataset_name,
                        "dataset_version": entry.dataset_version,
                        "cached_at": entry.cached_at,
                        "expires_at": entry.expires_at,
                        "computation_notes": json.dumps(entry.computation_notes),
                    },
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error caching benchmarks for {entry.industry}: {e}")
            return False

    def list_industries(self) -> List[str]:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                results = session.execute(
                    text(f"SELECT industry FROM {self.INDUSTRIES_TABLE} ORDER BY industry")
                ).fetchall()
                return [r[0] for r in results]
        except Exception:
            return []

    def clear_all(self) -> bool:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                session.execute(text(f"TRUNCATE TABLE {self.SYMBOLS_TABLE}"))
                session.execute(text(f"TRUNCATE TABLE {self.INDUSTRIES_TABLE}"))
                session.commit()
                return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        from sqlalchemy import text

        try:
            with self.db_manager.get_session() as session:
                symbol_count = session.execute(text(f"SELECT COUNT(*) FROM {self.SYMBOLS_TABLE}")).scalar()
                industry_count = session.execute(text(f"SELECT COUNT(*) FROM {self.INDUSTRIES_TABLE}")).scalar()
                return {
                    "backend": "postgresql_hybrid",
                    "symbols": {"count": symbol_count},
                    "industries": {"count": industry_count},
                }
        except Exception as e:
            return {"backend": "postgresql_hybrid", "error": str(e)}


class IndustryMetricsCache:
    """
    Main cache interface for industry-specific metrics (Hybrid Architecture).

    Two-tier caching:
    1. Symbol-level: Company-specific metrics (NVDA's inventory_days, JPM's NIM)
    2. Industry-level: Shared benchmarks, peer statistics, cycle indicators
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        parquet_path: Optional[str] = None,
    ):
        """
        Initialize industry metrics cache

        Args:
            backend: Storage backend ('parquet' or 'postgresql')
            parquet_path: Path for parquet storage
        """
        config = self._load_config()

        self._backend_type = backend or config.get("storage_backend", "parquet")
        self._parquet_path = parquet_path or config.get("parquet_path", "data/industry_metrics_cache")

        if self._backend_type == "postgresql":
            try:
                self._backend = PostgreSQLIndustryMetricsBackend()
                logger.info("Initialized PostgreSQL hybrid industry metrics cache")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable ({e}), falling back to Parquet")
                self._backend_type = "parquet"
                self._backend = ParquetIndustryMetricsBackend(Path(self._parquet_path))
        else:
            self._backend = ParquetIndustryMetricsBackend(Path(self._parquet_path))
            logger.info(f"Initialized Parquet hybrid industry metrics cache at {self._parquet_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml"""
        try:
            import yaml

            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    return config_data.get("cache_control", {}).get("industry_metrics", {})
            return {}
        except Exception:
            return {}

    # ========== Symbol-Level Methods ==========

    def get(self, symbol: str) -> Optional[IndustryMetricsCacheEntry]:
        """Retrieve cached metrics for a symbol"""
        return self._backend.get(symbol)

    def set(
        self,
        symbol: str,
        industry: str,
        sector: Optional[str],
        dataset_name: str,
        dataset_version: str,
        quality: str,
        coverage: float,
        metrics: Dict[str, Any],
        adjustments: List[Dict[str, Any]],
        tier_weights: Optional[Dict[str, int]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_days: Optional[int] = None,
    ) -> bool:
        """Store industry metrics for a symbol"""
        now = datetime.now(timezone.utc)
        expires_at = None
        if ttl_days:
            from datetime import timedelta

            expires_at = (now + timedelta(days=ttl_days)).isoformat()

        entry = IndustryMetricsCacheEntry(
            symbol=symbol.upper(),
            industry=industry,
            sector=sector,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            quality=quality,
            coverage=coverage,
            metrics=metrics,
            adjustments=adjustments,
            tier_weights=tier_weights,
            warnings=warnings or [],
            metadata=metadata or {},
            cached_at=now.isoformat(),
            expires_at=expires_at,
        )

        return self._backend.set(entry)

    def exists(self, symbol: str) -> bool:
        """Check if metrics exist for a symbol"""
        return self._backend.exists(symbol)

    def delete(self, symbol: str) -> bool:
        """Delete metrics for a symbol"""
        return self._backend.delete(symbol)

    def get_by_industry(self, industry: str) -> List[IndustryMetricsCacheEntry]:
        """Get all cached metrics for an industry"""
        return self._backend.get_by_industry(industry)

    def list_symbols(self) -> List[str]:
        """List all cached symbols"""
        return self._backend.list_symbols()

    # ========== Industry-Level Methods ==========

    def get_industry_benchmarks(self, industry: str) -> Optional[IndustryBenchmarksCacheEntry]:
        """Get cached industry-level benchmarks (shared across symbols)"""
        return self._backend.get_industry_benchmarks(industry)

    def set_industry_benchmarks(
        self,
        industry: str,
        sector: Optional[str],
        symbols_included: List[str],
        peer_statistics: Dict[str, Dict[str, float]],
        benchmarks: Dict[str, Any],
        tier_weights: Dict[str, int],
        cycle_indicators: Optional[Dict[str, Any]] = None,
        dataset_name: str = "computed",
        dataset_version: str = "1.0.0",
        ttl_days: Optional[int] = None,
        computation_notes: Optional[List[str]] = None,
    ) -> bool:
        """Store industry-level benchmarks (shared across symbols)"""
        now = datetime.now(timezone.utc)
        expires_at = None
        if ttl_days:
            from datetime import timedelta

            expires_at = (now + timedelta(days=ttl_days)).isoformat()

        entry = IndustryBenchmarksCacheEntry(
            industry=industry,
            sector=sector,
            symbol_count=len(symbols_included),
            symbols_included=symbols_included,
            peer_statistics=peer_statistics,
            benchmarks=benchmarks,
            tier_weights=tier_weights,
            cycle_indicators=cycle_indicators or {},
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            cached_at=now.isoformat(),
            expires_at=expires_at,
            computation_notes=computation_notes or [],
        )

        return self._backend.set_industry_benchmarks(entry)

    def compute_and_cache_industry_benchmarks(
        self,
        industry: str,
        ttl_days: int = 7,
    ) -> Optional[IndustryBenchmarksCacheEntry]:
        """
        Compute industry benchmarks from cached symbol data and cache the result.

        This aggregates metrics from all cached symbols in the industry to compute:
        - Peer statistics (median, mean, std, min, max, percentiles)
        - Industry benchmarks (typical ranges, thresholds)
        - Cycle indicators (if enough data)

        Args:
            industry: Industry name
            ttl_days: Cache time-to-live

        Returns:
            The computed and cached IndustryBenchmarksCacheEntry, or None if insufficient data
        """
        # Get all cached symbols for this industry
        symbol_entries = self.get_by_industry(industry)

        if len(symbol_entries) < 2:
            logger.warning(f"Not enough symbols ({len(symbol_entries)}) to compute benchmarks for {industry}")
            return None

        # Aggregate metrics
        symbols_included = [e.symbol for e in symbol_entries]
        sector = symbol_entries[0].sector
        dataset_name = symbol_entries[0].dataset_name
        dataset_version = symbol_entries[0].dataset_version
        tier_weights = symbol_entries[0].tier_weights or {}

        # Collect all numeric metrics
        all_metrics: Dict[str, List[float]] = {}
        for entry in symbol_entries:
            for metric_name, value in entry.metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(float(value))

        # Compute peer statistics
        peer_statistics: Dict[str, Dict[str, float]] = {}
        for metric_name, values in all_metrics.items():
            if len(values) >= 2:
                sorted_values = sorted(values)
                n = len(values)
                peer_statistics[metric_name] = {
                    "count": n,
                    "median": statistics.median(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if n > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "p25": sorted_values[n // 4] if n >= 4 else sorted_values[0],
                    "p75": sorted_values[3 * n // 4] if n >= 4 else sorted_values[-1],
                }

        # Compute benchmarks (thresholds based on percentiles)
        benchmarks: Dict[str, Any] = {}
        for metric_name, stats in peer_statistics.items():
            benchmarks[metric_name] = {
                "typical_low": stats["p25"],
                "typical_high": stats["p75"],
                "industry_median": stats["median"],
                "outlier_low": stats["min"],
                "outlier_high": stats["max"],
            }

        # Compute cycle indicators (simple heuristics)
        cycle_indicators: Dict[str, Any] = {
            "sample_size": len(symbol_entries),
            "data_quality": self._assess_data_quality(symbol_entries),
        }

        # Add industry-specific cycle detection
        if "inventory_days" in peer_statistics:
            inv_median = peer_statistics["inventory_days"]["median"]
            cycle_indicators["inventory_cycle"] = {
                "median_days": inv_median,
                "assessment": "elevated" if inv_median > 60 else "normal" if inv_median > 30 else "lean",
            }

        computation_notes = [
            f"Computed from {len(symbol_entries)} symbols",
            f"Metrics aggregated: {list(peer_statistics.keys())}",
        ]

        # Cache and return
        success = self.set_industry_benchmarks(
            industry=industry,
            sector=sector,
            symbols_included=symbols_included,
            peer_statistics=peer_statistics,
            benchmarks=benchmarks,
            tier_weights=tier_weights,
            cycle_indicators=cycle_indicators,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            ttl_days=ttl_days,
            computation_notes=computation_notes,
        )

        if success:
            return self.get_industry_benchmarks(industry)
        return None

    def _assess_data_quality(self, entries: List[IndustryMetricsCacheEntry]) -> str:
        """Assess overall data quality from symbol entries"""
        if not entries:
            return "none"

        quality_scores = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
        total_score = sum(quality_scores.get(e.quality, 1) for e in entries)
        avg_score = total_score / len(entries)

        if avg_score >= 3.5:
            return "excellent"
        elif avg_score >= 2.5:
            return "good"
        elif avg_score >= 1.5:
            return "fair"
        return "poor"

    def list_industries(self) -> List[str]:
        """List all industries with cached benchmarks"""
        return self._backend.list_industries()

    # ========== Common Methods ==========

    def clear_all(self) -> bool:
        """Clear all cached metrics (symbols and industries)"""
        return self._backend.clear_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._backend.get_stats()

    @property
    def backend_type(self) -> str:
        """Get current backend type"""
        return self._backend_type

    def cache_from_summary(
        self,
        summary: Dict[str, Any],
        ttl_days: Optional[int] = None,
    ) -> bool:
        """Cache metrics from an industry summary dict"""
        if not summary or summary.get("extraction_failed"):
            return False

        return self.set(
            symbol=summary.get("symbol", ""),
            industry=summary.get("industry", ""),
            sector=summary.get("sector"),
            dataset_name=summary.get("dataset_name", ""),
            dataset_version=summary.get("dataset_version", ""),
            quality=summary.get("quality", "poor"),
            coverage=summary.get("coverage", 0.0),
            metrics=summary.get("metrics", {}),
            adjustments=summary.get("adjustments", []),
            tier_weights=summary.get("tier_weights"),
            warnings=summary.get("warnings", []),
            metadata=summary.get("metadata", {}),
            ttl_days=ttl_days,
        )
