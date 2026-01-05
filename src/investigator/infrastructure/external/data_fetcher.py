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

"""Robust Data Fetcher with Registry-Driven Configuration.

This module provides a resilient, configuration-driven approach to fetching
economic data from multiple sources with automatic fallbacks.

Features:
- YAML registry for easy URL updates without code changes
- Multiple fallback sources (primary URL -> backup URLs -> FRED)
- Resilient Excel/CSV/JSON/HTML parsing
- Health monitoring and staleness alerts
- Automatic retry with exponential backoff
- Caching to reduce API calls

Usage:
    fetcher = DataFetcher()

    # Fetch with automatic fallbacks
    data = await fetcher.fetch("atlanta_fed", "gdpnow")

    # Check health
    health = await fetcher.health_check()
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import ssl
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import yaml

logger = logging.getLogger(__name__)

# Find config directory
CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "config"
REGISTRY_PATH = CONFIG_DIR / "data_sources_registry.yaml"


@dataclass
class FetchResult:
    """Result of a data fetch operation."""

    success: bool
    data: Any = None
    source: str = ""  # "primary_url", "fallback_url_1", "fred"
    url_used: str = ""
    fetch_time: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "url_used": self.url_used,
            "fetch_time": self.fetch_time.isoformat(),
            "error": self.error,
            "cached": self.cached,
        }


@dataclass
class HealthStatus:
    """Health status of a data source."""

    source_id: str
    indicator_id: str
    is_healthy: bool
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    is_stale: bool = False
    staleness_hours: float = 0.0
    primary_url_working: bool = True
    using_fallback: bool = False
    using_fred: bool = False


class DataSourceRegistry:
    """Manages the data sources registry configuration."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or REGISTRY_PATH
        self._config: Dict[str, Any] = {}
        self._load_time: Optional[datetime] = None
        self._load()

    def _load(self) -> None:
        """Load registry from YAML file."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path) as f:
                    self._config = yaml.safe_load(f) or {}
                self._load_time = datetime.now()
                logger.info(f"Loaded data sources registry: {len(self._config)} sources")
            else:
                logger.warning(f"Registry not found: {self.registry_path}")
                self._config = {}
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._config = {}

    def reload(self) -> None:
        """Reload registry from file."""
        self._load()

    def get_source(self, source_id: str, indicator_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific data source indicator."""
        source = self._config.get(source_id, {})
        return source.get(indicator_id)

    def get_urls(self, source_id: str, indicator_id: str) -> List[str]:
        """Get list of URLs for a source (primary + fallbacks)."""
        config = self.get_source(source_id, indicator_id)
        if config:
            return config.get("urls", [])
        return []

    def get_fred_series(self, source_id: str, indicator_id: str) -> Optional[str]:
        """Get FRED series ID if available."""
        config = self.get_source(source_id, indicator_id)
        if config:
            return config.get("fred_series")
        return None

    def get_parser_type(self, source_id: str, indicator_id: str) -> str:
        """Get parser type for a source."""
        config = self.get_source(source_id, indicator_id)
        if config:
            return config.get("parser", "excel")
        return "excel"

    def get_parser_hints(self, source_id: str, indicator_id: str) -> Dict[str, Any]:
        """Get parser hints for a source."""
        config = self.get_source(source_id, indicator_id)
        if config:
            return config.get("parser_hints", {})
        return {}

    def is_fred_preferred(self, source_id: str, indicator_id: str) -> bool:
        """Check if FRED is preferred over primary URLs."""
        hints = self.get_parser_hints(source_id, indicator_id)
        return hints.get("fred_preferred", False)

    def get_freshness_hours(self, source_id: str, indicator_id: str) -> int:
        """Get expected freshness in hours."""
        config = self.get_source(source_id, indicator_id)
        if config:
            return config.get("freshness_hours", 168)  # Default 1 week
        return 168

    def list_all_sources(self) -> List[Tuple[str, str]]:
        """List all (source_id, indicator_id) pairs."""
        pairs = []
        for source_id, indicators in self._config.items():
            if source_id.startswith("_"):
                continue  # Skip metadata
            if isinstance(indicators, dict):
                for indicator_id in indicators:
                    if not indicator_id.startswith("_"):
                        pairs.append((source_id, indicator_id))
        return pairs


class ResilientParser:
    """Resilient parser for various data formats."""

    @staticmethod
    def parse_excel(
        content: bytes,
        hints: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse Excel file with resilient column detection."""
        hints = hints or {}
        try:
            import pandas as pd

            # Try multiple sheet indices
            for sheet_idx in [0, "Sheet1", "Data"]:
                try:
                    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_idx)
                    if not df.empty:
                        break
                except Exception:
                    continue
            else:
                return None

            if df.empty:
                return None

            # Get latest row
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None

            # Find date column
            date_col = (
                ResilientParser._find_column(
                    df.columns, hints.get("date_col_patterns", ["date", "month", "period", "time"])
                )
                or df.columns[0]
            )

            # Find value column(s)
            value_col_patterns = hints.get("value_col_patterns", ["value", "index", "rate"])
            value_col = ResilientParser._find_column(df.columns, value_col_patterns)
            if value_col is None and len(df.columns) > 1:
                value_col = df.columns[1]

            # Parse date
            try:
                obs_date = pd.to_datetime(latest[date_col]).date()
            except Exception:
                obs_date = date.today()

            # Parse value
            try:
                value = float(latest[value_col])
            except Exception:
                value = None

            # Previous value
            prev_value = None
            if prev is not None:
                try:
                    prev_value = float(prev[value_col])
                except Exception:
                    pass

            return {
                "date": obs_date,
                "value": value,
                "previous_value": prev_value,
                "column_used": str(value_col),
                "raw_latest": latest.to_dict() if hasattr(latest, "to_dict") else {},
            }

        except Exception as e:
            logger.debug(f"Excel parsing failed: {e}")
            return None

    @staticmethod
    def parse_csv(
        content: bytes,
        hints: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse CSV file with resilient column detection."""
        hints = hints or {}
        try:
            import pandas as pd

            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    if not df.empty:
                        break
                except Exception:
                    continue
            else:
                return None

            # Same logic as Excel
            return ResilientParser.parse_excel(
                content=None,  # Won't be used
                hints=hints,
            )

        except Exception as e:
            logger.debug(f"CSV parsing failed: {e}")
            return None

    @staticmethod
    def parse_json(
        content: bytes,
        hints: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse JSON response."""
        hints = hints or {}
        try:
            data = json.loads(content)

            # Navigate to data path if specified
            json_path = hints.get("json_path", "data")
            if json_path and isinstance(data, dict):
                data = data.get(json_path, data)

            if isinstance(data, list) and data:
                latest = data[-1]
                prev = data[-2] if len(data) > 1 else None

                # Get indices from hints
                date_idx = hints.get("date_index", 0)
                value_idx = hints.get("close_index", hints.get("value_index", -1))

                if isinstance(latest, list):
                    # Array format: [date, open, high, low, close] (CBOE style)
                    try:
                        obs_date = datetime.strptime(str(latest[date_idx]), "%Y-%m-%d").date()
                    except Exception:
                        obs_date = date.today()

                    value = float(latest[value_idx]) if value_idx < len(latest) else None
                    prev_value = float(prev[value_idx]) if prev and value_idx < len(prev) else None

                    # Include OHLCV for market data
                    ohlcv = None
                    if len(latest) >= 5:
                        try:
                            ohlcv = {
                                "open": float(latest[1]),
                                "high": float(latest[2]),
                                "low": float(latest[3]),
                                "close": float(latest[4]),
                            }
                        except (IndexError, ValueError):
                            pass

                    result = {
                        "date": obs_date,
                        "value": value,
                        "previous_value": prev_value,
                    }
                    if ohlcv:
                        result["ohlcv"] = ohlcv
                    return result

                elif isinstance(latest, dict):
                    # Dictionary format - extract value if possible
                    obs_date = date.today()
                    if "date" in latest:
                        try:
                            obs_date = datetime.strptime(str(latest["date"]), "%Y-%m-%d").date()
                        except Exception:
                            pass

                    # Try to find a value field
                    value = None
                    for key in ["value", "close", "price", "level"]:
                        if key in latest:
                            try:
                                value = float(latest[key])
                                break
                            except (ValueError, TypeError):
                                pass

                    return {
                        "date": obs_date,
                        "value": value,
                        "raw": latest,
                    }

            return {"date": date.today(), "raw": data}

        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None

    @staticmethod
    def parse_html(
        content: str,
        hints: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse HTML page for data using patterns."""
        hints = hints or {}
        try:
            patterns = hints.get(
                "html_patterns",
                [
                    r"(-?\d+\.?\d*)\s*(?:percent|%)",
                ],
            )
            value_range = hints.get("value_range", [-100, 100])

            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # Sanity check
                    if value_range[0] <= value <= value_range[1]:
                        return {
                            "date": date.today(),
                            "value": value,
                            "pattern_used": pattern,
                        }

            return None

        except Exception as e:
            logger.debug(f"HTML parsing failed: {e}")
            return None

    @staticmethod
    def _find_column(
        columns: List[str],
        patterns: List[str],
    ) -> Optional[str]:
        """Find column matching any pattern."""
        for col in columns:
            col_lower = str(col).lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    return col
        return None


class DataFetcher:
    """Robust data fetcher with automatic fallbacks and FRED integration."""

    def __init__(
        self,
        registry: Optional[DataSourceRegistry] = None,
        fred_api_key: Optional[str] = None,
    ):
        self.registry = registry or DataSourceRegistry()
        self._fred_api_key = fred_api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[FetchResult, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
        self._health_status: Dict[str, HealthStatus] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with SSL handling."""
        if self._session is None or self._session.closed:
            try:
                import certifi

                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ssl_ctx = ssl.create_default_context()

            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            headers = {
                "User-Agent": "Victor-Invest/1.0 (Economic Data Collector)",
                "Accept": "*/*",
            }
            self._session = aiohttp.ClientSession(connector=connector, headers=headers)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_cache_key(self, source_id: str, indicator_id: str) -> str:
        """Generate cache key."""
        return f"{source_id}:{indicator_id}"

    def _get_cached(self, source_id: str, indicator_id: str) -> Optional[FetchResult]:
        """Get cached result if still valid."""
        key = self._get_cache_key(source_id, indicator_id)
        if key in self._cache:
            result, cached_time = self._cache[key]
            if datetime.now() - cached_time < self._cache_ttl:
                result.cached = True
                return result
        return None

    def _set_cache(self, source_id: str, indicator_id: str, result: FetchResult):
        """Cache a fetch result."""
        key = self._get_cache_key(source_id, indicator_id)
        self._cache[key] = (result, datetime.now())

    async def fetch(
        self,
        source_id: str,
        indicator_id: str,
        force_refresh: bool = False,
    ) -> FetchResult:
        """Fetch data with automatic fallbacks.

        Order of attempts:
        1. Cache (if valid and not force_refresh)
        2. FRED (if preferred)
        3. Primary URL
        4. Fallback URLs
        5. FRED (if not preferred but available)

        Args:
            source_id: Source identifier (e.g., "atlanta_fed")
            indicator_id: Indicator identifier (e.g., "gdpnow")
            force_refresh: Skip cache

        Returns:
            FetchResult with data or error
        """
        # Check cache
        if not force_refresh:
            cached = self._get_cached(source_id, indicator_id)
            if cached:
                return cached

        # Get configuration
        config = self.registry.get_source(source_id, indicator_id)
        if not config:
            return FetchResult(success=False, error=f"Unknown source: {source_id}/{indicator_id}")

        urls = self.registry.get_urls(source_id, indicator_id)
        fred_series = self.registry.get_fred_series(source_id, indicator_id)
        parser_type = self.registry.get_parser_type(source_id, indicator_id)
        hints = self.registry.get_parser_hints(source_id, indicator_id)
        fred_preferred = self.registry.is_fred_preferred(source_id, indicator_id)

        # Try FRED first if preferred
        if fred_preferred and fred_series:
            result = await self._fetch_from_fred(fred_series, source_id, indicator_id)
            if result.success:
                self._set_cache(source_id, indicator_id, result)
                self._update_health(source_id, indicator_id, True, using_fred=True)
                return result

        # Try URLs in order
        for idx, url in enumerate(urls):
            result = await self._fetch_from_url(url, parser_type, hints, source_id, indicator_id, idx)
            if result.success:
                self._set_cache(source_id, indicator_id, result)
                self._update_health(source_id, indicator_id, True, using_fallback=(idx > 0))
                return result

        # Try FRED as final fallback
        if fred_series and not fred_preferred:
            result = await self._fetch_from_fred(fred_series, source_id, indicator_id)
            if result.success:
                self._set_cache(source_id, indicator_id, result)
                self._update_health(source_id, indicator_id, True, using_fred=True)
                return result

        # All attempts failed
        self._update_health(source_id, indicator_id, False)
        return FetchResult(success=False, error=f"All fetch attempts failed for {source_id}/{indicator_id}")

    async def _fetch_from_url(
        self,
        url: str,
        parser_type: str,
        hints: Dict[str, Any],
        source_id: str,
        indicator_id: str,
        url_index: int,
    ) -> FetchResult:
        """Fetch from a specific URL."""
        try:
            session = await self._get_session()

            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return FetchResult(success=False, url_used=url, error=f"HTTP {response.status}")

                content_type = response.headers.get("Content-Type", "")

                # Determine parser based on content type and config
                if "excel" in parser_type or "spreadsheet" in content_type:
                    content = await response.read()
                    data = ResilientParser.parse_excel(content, hints)
                elif "csv" in parser_type or "csv" in content_type:
                    content = await response.read()
                    data = ResilientParser.parse_csv(content, hints)
                elif "json" in parser_type or "json" in content_type:
                    content = await response.read()
                    data = ResilientParser.parse_json(content, hints)
                elif "html" in parser_type or "html" in content_type:
                    content = await response.text()
                    data = ResilientParser.parse_html(content, hints)
                else:
                    # Try Excel first, then JSON, then HTML
                    content = await response.read()
                    data = (
                        ResilientParser.parse_excel(content, hints)
                        or ResilientParser.parse_json(content, hints)
                        or ResilientParser.parse_html(content.decode("utf-8", errors="ignore"), hints)
                    )

                if data:
                    source_label = "primary_url" if url_index == 0 else f"fallback_url_{url_index}"
                    return FetchResult(
                        success=True,
                        data=data,
                        source=source_label,
                        url_used=url,
                    )
                else:
                    return FetchResult(success=False, url_used=url, error="Parsing failed")

        except asyncio.TimeoutError:
            return FetchResult(success=False, url_used=url, error="Timeout")
        except Exception as e:
            return FetchResult(success=False, url_used=url, error=str(e))

    async def _fetch_from_fred(
        self,
        series_id: str,
        source_id: str,
        indicator_id: str,
    ) -> FetchResult:
        """Fetch from FRED API."""
        try:
            # Get FRED API key
            api_key = self._fred_api_key
            if not api_key:
                try:
                    from victor.config.api_keys import get_service_key

                    api_key = get_service_key("fred")
                except ImportError:
                    api_key = os.environ.get("FRED_API_KEY")

            if not api_key:
                return FetchResult(success=False, source="fred", error="FRED API key not available")

            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}"
                f"&api_key={api_key}"
                f"&file_type=json"
                f"&sort_order=desc"
                f"&limit=5"
            )

            session = await self._get_session()
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return FetchResult(success=False, source="fred", error=f"FRED HTTP {response.status}")

                data = await response.json()
                observations = data.get("observations", [])

                if observations:
                    latest = observations[0]
                    prev = observations[1] if len(observations) > 1 else None

                    try:
                        value = float(latest["value"])
                    except (ValueError, KeyError):
                        value = None

                    try:
                        prev_value = float(prev["value"]) if prev else None
                    except (ValueError, KeyError):
                        prev_value = None

                    return FetchResult(
                        success=True,
                        data={
                            "date": datetime.strptime(latest["date"], "%Y-%m-%d").date(),
                            "value": value,
                            "previous_value": prev_value,
                            "fred_series": series_id,
                        },
                        source="fred",
                        url_used=f"FRED:{series_id}",
                    )

                return FetchResult(success=False, source="fred", error="No observations")

        except Exception as e:
            return FetchResult(success=False, source="fred", error=f"FRED error: {e}")

    def _update_health(
        self,
        source_id: str,
        indicator_id: str,
        success: bool,
        using_fallback: bool = False,
        using_fred: bool = False,
    ):
        """Update health status for a source."""
        key = f"{source_id}:{indicator_id}"
        status = self._health_status.get(key) or HealthStatus(
            source_id=source_id,
            indicator_id=indicator_id,
            is_healthy=True,
        )

        if success:
            status.is_healthy = True
            status.last_success = datetime.now()
            status.failure_count = 0
            status.using_fallback = using_fallback
            status.using_fred = using_fred
            status.primary_url_working = not (using_fallback or using_fred)
        else:
            status.last_failure = datetime.now()
            status.failure_count += 1
            if status.failure_count >= 3:
                status.is_healthy = False

        self._health_status[key] = status

    async def health_check(self) -> Dict[str, HealthStatus]:
        """Check health of all configured sources."""
        all_sources = self.registry.list_all_sources()

        for source_id, indicator_id in all_sources:
            # Just try to fetch - will update health status
            await self.fetch(source_id, indicator_id, force_refresh=True)

        return self._health_status

    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report."""
        total = len(self._health_status)
        healthy = sum(1 for s in self._health_status.values() if s.is_healthy)
        using_fallback = sum(1 for s in self._health_status.values() if s.using_fallback)
        using_fred = sum(1 for s in self._health_status.values() if s.using_fred)

        return {
            "total_sources": total,
            "healthy": healthy,
            "unhealthy": total - healthy,
            "using_primary_url": total - using_fallback - using_fred,
            "using_fallback_url": using_fallback,
            "using_fred": using_fred,
            "sources": {
                k: {
                    "healthy": v.is_healthy,
                    "using_fallback": v.using_fallback,
                    "using_fred": v.using_fred,
                    "failure_count": v.failure_count,
                }
                for k, v in self._health_status.items()
            },
        }


# Singleton instance
_fetcher: Optional[DataFetcher] = None


def get_data_fetcher() -> DataFetcher:
    """Get shared data fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher
